"""Exact expectations, paired Monte Carlo variance, and real checkpoint continuation."""
import argparse
from copy import deepcopy
from dataclasses import replace
import itertools
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy_gradient_part1.config import Config
from policy_gradient_part1.models import make_policy
from policy_gradient_part1.reinforce import backward_episode, policy_loss, policy_weights, reward_to_go
from policy_gradient_part1.runtime import load_checkpoint, write_json
from policy_gradient_part1.trainer import train
from policy_gradient_part2.baseline import RunningConstantBaseline, validate_trajectory
from verify_pg_resume import same


def validate(output, device="cpu", integration=True):
    checks = {}
    def check(name, passed):
        checks[name] = bool(passed)
        if not passed:
            raise AssertionError(name)
    b = RunningConstantBaseline()
    check("initial_zero", b.estimate() == 0 and b.count == 0)
    b.update(-5); b.update(-3)
    check("negative_returns_and_first_observation_warmup", abs(b.estimate()+4.9) < 1e-14)
    restored = RunningConstantBaseline(); restored.load_state_dict(b.state_dict())
    check("baseline_state_round_trip", restored.state_dict() == b.state_dict())
    for invalid in (float("nan"), float("inf")):
        try: b.update(invalid)
        except ValueError: pass
        else: raise AssertionError("Accepted nonfinite baseline input")
    check("nonfinite_rejected", b.count == 2)
    for gamma in (0, .5, .95, 1):
        for length in (1, 2, 10000):
            actual = reward_to_go(np.full(length, -2.), gamma)
            expected = (-2 * np.arange(length, 0, -1) if gamma == 1 else
                        -2*(1-gamma**np.arange(length, 0, -1))/(1-gamma))
            check(f"constant_negative_rewards_gamma{gamma}_length{length}", np.allclose(actual, expected))
    for trajectory in (None, SimpleNamespace(rewards=[]),
                       SimpleNamespace(rewards=[1], observations=[], actions=[0], log_probs=[0], entropies=[0],
                                       probabilities=[[.5,.5]], terminated=True, truncated=False),
                       SimpleNamespace(rewards=[1], observations=[[0]], actions=[0], log_probs=[0], entropies=[0],
                                       probabilities=[[.5,.5]], terminated=False, truncated=False)):
        try: validate_trajectory(trajectory)
        except ValueError: pass
        else: raise AssertionError("Missing/incomplete trajectory accepted")
    check("missing_empty_mismatched_incomplete_rejected", True)

    # A two-decision policy with independent Bernoulli actions and known reward tables.
    p = np.array([.4, .6]); reward = np.array([[.2,1.], [1.,2.]])
    exact = []
    for gamma in (0., .95, 1.):
        for convention in ("relative", "absolute"):
            analytical = p[0]*p[1]*sum((gamma**t if convention == "absolute" else 1)*np.diff(reward[t])[0]
                                      for t in range(2)) * np.array([-1.,1.])
            for baseline in (-10., 0., 2., 100.):
                expectation = np.zeros(2)
                for actions in itertools.product(range(2), repeat=2):
                    probability = np.prod(p[list(actions)])
                    weights = reward_to_go([reward[t,a] for t,a in enumerate(actions)], gamma)-baseline
                    if convention == "absolute": weights *= gamma**np.arange(2)
                    scores = np.eye(2)[list(actions)]-p
                    expectation += probability*(weights[:,None]*scores).sum(axis=0)
                error = float(np.max(np.abs(expectation-analytical)))
                exact.append(dict(gamma=gamma, convention=convention, baseline=baseline, error=error))
    check("exact_gradient_and_baseline_invariance", max(r["error"] for r in exact) < 1e-12)

    # One-step trajectories; training EMA is estimated on a disjoint prior sample.
    rng = np.random.default_rng(4902); prob = np.array([.25,.75]); rewards = np.array([1.,3.])
    prior = rng.choice(2, size=2000, p=prob); ema = RunningConstantBaseline()
    for a in prior: ema.update(rewards[a])
    actions = rng.choice(2, size=200000, p=prob); scores = np.eye(2)[actions]-prob
    g0 = scores*rewards[actions,None]; gb = scores*(rewards[actions,None]-ema.estimate())
    v0 = float(g0.var(axis=0,ddof=1).sum()); vb = float(gb.var(axis=0,ddof=1).sum())
    variance = dict(episodes=len(actions), independent_prior_episodes=len(prior), baseline=ema.estimate(),
                    analytical_gradient=[-.375,.375], mean_without=g0.mean(axis=0).tolist(),
                    mean_with=gb.mean(axis=0).tolist(), trace_covariance_without=v0, trace_covariance_with=vb,
                    relative_reduction=1-vb/v0, mean_difference_norm=float(np.linalg.norm((gb-g0).mean(axis=0))),
                    paired_mean_difference_se=np.sqrt((gb-g0).var(axis=0,ddof=1)/len(actions)).tolist())
    check("paired_synthetic_variance_reduced", vb < v0)
    check("paired_mean_difference_within_5se", np.all(np.abs((gb-g0).mean(axis=0)) <=
          5*np.sqrt((gb-g0).var(axis=0,ddof=1)/len(actions))))

    torch.manual_seed(4)
    config = Config(hidden_sizes=(8,), device=device)
    model = make_policy(config).to(device)
    obs = np.random.default_rng(4).normal(size=(7,4)).astype(np.float32)
    actions = np.arange(7)%2
    with torch.no_grad():
        lp = torch.distributions.Categorical(logits=model(torch.tensor(obs,device=device))).log_prob(
            torch.tensor(actions,device=device)).cpu().numpy()
    trajectory = SimpleNamespace(observations=obs, actions=actions, rewards=np.array([0,1,0,-1,0,1,1.]), log_probs=lp)
    errors = []
    for convention in ("relative", "absolute"):
        for baseline in (0., -2., 3.):
            model.zero_grad()
            backward_episode(model,trajectory,.95,device,3,2,convention,baseline)
            chunk = torch.cat([p.grad.flatten() for p in model.parameters()]).clone()
            model.zero_grad()
            weights=reward_to_go(trajectory.rewards,.95)-baseline
            if convention == "absolute": weights *= .95**np.arange(7)
            loss=policy_loss(model(torch.tensor(obs,device=device)),torch.tensor(actions,device=device),
                             torch.tensor(weights,device=device,dtype=torch.float32))/2
            loss.backward()
            errors.append(float((chunk-torch.cat([p.grad.flatten() for p in model.parameters()])).abs().max()))
    check("chunked_baseline_gradient_matches_full_autograd", max(errors) < 2e-6)
    model.zero_grad(); backward_episode(model,trajectory,.95,device,3,1)
    old = torch.cat([p.grad.flatten() for p in model.parameters()]).clone()
    model.zero_grad(); backward_episode(model,trajectory,.95,device,3,1,baseline=0.)
    check("zero_baseline_bitwise_part1_compatibility", torch.equal(old,torch.cat([p.grad.flatten() for p in model.parameters()])))

    if integration:
        cfg=Config(episodes=6,batch_episodes=2,hidden_sizes=(8,),checkpoint_interval=2,log_interval=6,
                   max_episode_steps=16,device=device)
        train(cfg,output/"uninterrupted",baseline=RunningConstantBaseline())
        train(replace(cfg,episodes=2),output/"resumed",baseline=RunningConstantBaseline())
        train(cfg,output/"resumed",resume=output/"resumed/latest.pt",baseline=RunningConstantBaseline())
        a,b = [load_checkpoint(output/f"{name}/latest.pt") for name in ("uninterrupted","resumed")]
        for key in ("model","optimizer","rng","baseline","config","completed_episodes"):
            check(f"resumed_{key}_identical",same(a[key],b[key]))
        columns = set(a["history"][0]) - {"seconds"}
        check("resumed_numerical_history_identical", len(a["history"])==len(b["history"]) and all(
            all(x[k]==y[k] for k in columns) for x,y in zip(a["history"],b["history"])))
        check("baseline_frozen_per_batch",all(a["history"][i]["baseline_used"]==a["history"][i+1]["baseline_used"]
                                               for i in range(0,6,2)))
    result=dict(all_passed=all(checks.values()),checks=checks,exact_enumeration=exact,
                max_exact_error=max(r["error"] for r in exact),synthetic_variance=variance,
                max_chunk_gradient_error=max(errors),device=device,
                scope="Synthetic variance is not evidence of variance reduction for every game/policy/baseline")
    write_json(output/"validation.json",result)
    from policy_gradient_part2.analysis import variance_plot
    variance_plot(variance, ROOT/"figures/policy_gradient_part2/synthetic_gradient_variance.png", "Fixed-policy synthetic trajectories")
    print(f"Passed {len(checks)} checks; synthetic variance reduction={variance['relative_reduction']:.2%}")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"results/policy_gradient_part2/validation")
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--no-integration",action="store_true")
    args=parser.parse_args(); validate(args.output,args.device,not args.no_integration)
