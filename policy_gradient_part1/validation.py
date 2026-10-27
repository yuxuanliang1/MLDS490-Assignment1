"""Numerical algorithm evidence independent of success on either game."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.distributions import Categorical

from .config import Config
from .environments import make_environment, environment_metadata
from .models import make_policy
from .preprocessing import preprocess, ObservationTransform
from .reinforce import policy_loss, policy_weights, reward_to_go, backward_episode
from .rollout import sample_episode
from .runtime import configure_runtime, load_checkpoint, save_checkpoint, restore_rng, write_json


def validate_core(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = Config(seed=812, hidden_sizes=(8,), batch_episodes=2).validate()
    device = configure_runtime(config)
    checks, evidence = {}, {}
    cases = [([1, 2, 3], 0.5, [2.75, 3.5, 3]),
             ([1, 2, 3], 0, [1, 2, 3]), ([1, 2, 3], 1, [6, 5, 3]),
             ([1, 2, 3], 0.999, [5.992003, 4.997, 3]), ([7], 0.95, [7]),
             ([0, 0, -1, 0, 1], 0.5, [-0.1875, -0.375, -0.75, 0.5, 1])]
    errors = [float(np.max(np.abs(reward_to_go(r, gamma)-expected))) for r, gamma, expected in cases]
    checks["handcrafted_returns"] = max(errors) < 1e-12
    evidence["returns"] = [dict(rewards=r, gamma=g, expected=e, max_error=error)
                           for (r, g, e), error in zip(cases, errors)]

    logits = torch.tensor(np.log([0.3, 0.7]), dtype=torch.float64, requires_grad=True)
    rewards = torch.tensor([1.0, -0.4], dtype=torch.float64)
    p = logits.softmax(0).detach()
    analytic = p * ((p * rewards).sum() - rewards)  # loss gradient, negative ascent direction
    enumerated = policy_loss(logits.expand(2, -1), torch.arange(2), p * rewards)
    exact = torch.autograd.grad(enumerated, logits)[0]
    actions = Categorical(probs=p).sample((100000,))
    empirical_loss = policy_loss(logits.expand(len(actions), -1), actions, rewards[actions]) / len(actions)
    empirical = torch.autograd.grad(empirical_loss, logits)[0]
    finite_difference = []
    for i in range(2):
        plus, minus = logits.detach().clone(), logits.detach().clone()
        plus[i] += 1e-6
        minus[i] -= 1e-6
        finite_difference.append(float((-(plus.softmax(0)*rewards).sum() +
                                         (minus.softmax(0)*rewards).sum()) / 2e-6))
    evidence["analytical_gradient"] = dict(analytic=analytic.tolist(), autograd=exact.tolist(),
                                           monte_carlo=empirical.tolist(), samples=len(actions),
                                           finite_difference=finite_difference,
                                           exact_max_error=float((exact-analytic).abs().max()),
                                           empirical_max_error=float((empirical-analytic).abs().max()))
    checks["exact_gradient"] = float((exact-analytic).abs().max()) < 1e-12
    checks["empirical_gradient"] = float((empirical-analytic).abs().max()) < 0.01
    checks["finite_difference"] = np.max(np.abs(np.array(finite_difference)-analytic.numpy())) < 1e-8

    bandit_logits = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.Adam([bandit_logits], lr=0.01)
    initial_probability = float(bandit_logits.softmax(0)[1])
    initial_gradient = None
    observed_means = []
    for step in range(500):
        dist = Categorical(logits=bandit_logits)
        sampled_actions = dist.sample((64,))
        bandit_rewards = sampled_actions.float()  # action 1 reward=1, action 0 reward=0
        loss = -(dist.log_prob(sampled_actions)*bandit_rewards).mean()
        optimizer.zero_grad()
        loss.backward()
        if step == 0:
            initial_gradient = bandit_logits.grad.tolist()
        optimizer.step()
        observed_means.append(float(bandit_rewards.mean()))
    final_probability = float(bandit_logits.softmax(0)[1])
    evidence["controlled_decision"] = dict(initial_better_action_probability=initial_probability,
                                           final_better_action_probability=final_probability,
                                           initial_loss_gradient=initial_gradient,
                                           mean_reward_first50=float(np.mean(observed_means[:50])),
                                           mean_reward_last50=float(np.mean(observed_means[-50:])))
    checks["better_action_learned"] = final_probability > 0.97 and initial_gradient[1] < 0

    image = np.full((210, 160, 3), 144, dtype=np.uint8)
    image[35, 0, 0] = 255
    image[37, 2, 0] = 109
    copy = image.copy()
    transformed = preprocess(image)
    checks["preprocess_binary_and_copy"] = (transformed.shape == (80, 80) and
                                           transformed.dtype == np.float32 and transformed.sum() == 1 and
                                           transformed[0, 0] == 1 and np.array_equal(image, copy))
    transform = ObservationTransform(Config(environment="ALE/Pong-v5", gamma=0.99))
    checks["first_and_identical_frame_differences_zero"] = (np.count_nonzero(transform(image)) == 0 and
                                                          np.count_nonzero(transform(image)) == 0)
    try:
        preprocess(np.zeros((80, 80)))
        checks["invalid_frame_rejected"] = False
    except ValueError:
        checks["invalid_frame_rejected"] = True
    try:
        policy_loss(torch.tensor([[float("nan"), 0.0]]), torch.tensor([0]), torch.tensor([1.0]))
        checks["nonfinite_rejected"] = False
    except FloatingPointError:
        checks["nonfinite_rejected"] = True
    extreme = torch.tensor([[1000.0, -1000.0]], requires_grad=True)
    extreme_loss = policy_loss(extreme, torch.tensor([1]), torch.tensor([1.0]))
    extreme_loss.backward()
    checks["extreme_logits_stable"] = bool(torch.isfinite(extreme_loss) and torch.isfinite(extreme.grad).all())

    # Chunked recomputation must agree with a full retained-graph reference.
    model = make_policy(config)
    observations = np.random.randn(9, 4).astype(np.float32)
    with torch.no_grad():
        dist = Categorical(logits=model(torch.from_numpy(observations)))
        sampled = dist.sample()
        log_probs = dist.log_prob(sampled).numpy()
    trajectory = SimpleNamespace(observations=observations, actions=sampled.numpy(),
                                 rewards=np.arange(9, dtype=float) / 9, log_probs=log_probs)
    full = deepcopy(model)
    weights = torch.tensor(reward_to_go(trajectory.rewards, config.gamma), dtype=torch.float32)
    (policy_loss(full(torch.from_numpy(observations)), sampled, weights)/2).backward()
    _, lp_error = backward_episode(model, trajectory, config.gamma, device, 3, 2)
    error = max(float((a.grad-b.grad).abs().max()) for a, b in zip(model.parameters(), full.parameters()))
    checks["chunk_gradient_equivalence"] = error < 2e-6
    evidence["chunking"] = dict(max_gradient_error=error, max_log_probability_error=lp_error)

    # Actual CartPole trajectory and checkpoint continuation, with independent environment resets.
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    env = make_environment(config)
    trajectory = sample_episode(env, model, config, device, 812, trace_steps=10)
    write_json(output / "cartpole_first_steps.json", trajectory.trace)
    optimizer.zero_grad()
    backward_episode(model, trajectory, config.gamma, device, 16, 1)
    optimizer.step()
    probability_reference = model(torch.from_numpy(observations)).softmax(-1).detach().clone()
    save_checkpoint(output / "validation_checkpoint.pt", model, optimizer, config, 1, [], device, {})
    continuation = sample_episode(env, model, config, device, 813)
    optimizer.zero_grad()
    backward_episode(model, continuation, config.gamma, device, 16, 1)
    optimizer.step()
    final_parameters = [p.detach().clone() for p in model.parameters()]
    checkpoint = load_checkpoint(output / "validation_checkpoint.pt")
    loaded = make_policy(config)
    loaded.load_state_dict(checkpoint["model"])
    loaded_optimizer = torch.optim.Adam(loaded.parameters(), lr=config.learning_rate)
    loaded_optimizer.load_state_dict(checkpoint["optimizer"])
    probe = torch.from_numpy(observations)
    probability_error = float((loaded(probe).softmax(-1)-probability_reference).abs().max())
    restore_rng(checkpoint["rng"], device)
    resumed = sample_episode(env, loaded, config, device, 813)
    loaded_optimizer.zero_grad()
    backward_episode(loaded, resumed, config.gamma, device, 16, 1)
    loaded_optimizer.step()
    parameter_error = max(float((a-b).abs().max()) for a, b in zip(final_parameters, loaded.parameters()))
    checks["checkpoint_probability_match"] = probability_error == 0
    checks["checkpoint_optimizer_rng_continuation"] = parameter_error == 0 and np.array_equal(
        continuation.actions, resumed.actions)
    evidence["checkpoint"] = dict(probability_error=probability_error, continuation_parameter_error=parameter_error,
                                  matched_actions=bool(np.array_equal(continuation.actions, resumed.actions)),
                                  scope="Same CPU/software, episode boundary; not mid-episode or cross-device")
    env.close()
    short_config = Config(max_episode_steps=1)
    short_env = make_environment(short_config)
    short = sample_episode(short_env, make_policy(short_config), short_config, device, 44)
    checks["one_step_truncation"] = len(short.rewards) == 1 and short.truncated
    short_env.close()
    checks = {key: bool(value) for key, value in checks.items()}
    result = dict(checks=checks, all_passed=all(checks.values()), evidence=evidence)
    write_json(output / "summary.json", result)
    print(f"Core validation: {sum(checks.values())}/{len(checks)} passed", flush=True)
    if not result["all_passed"]:
        raise AssertionError([key for key, passed in checks.items() if not passed])
    return result


def inspect_pong(output):
    """Startup/point-progress experiment, no policy training and no GPU."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = Config(environment="ALE/Pong-v5", gamma=0.99)
    configure_runtime(config)
    env = make_environment(config)
    metadata = environment_metadata(env, config)
    comparisons = []
    try:
        for fire in (False, True):
            observation, _ = env.reset(seed=73)
            if fire:
                observation, reward, terminated, truncated, _ = env.step(1)
                if reward or terminated or truncated:
                    raise AssertionError("Unexpected FIRE initialization outcome")
            initial = preprocess(observation)
            nonzero_rewards = []
            changed_frames = 0
            transform = ObservationTransform(config)
            for step in range(3000):
                state = transform(observation)
                changed_frames += int(np.count_nonzero(state) > 0)
                observation, reward, terminated, truncated, _ = env.step((2, 3)[step % 2])
                if reward:
                    nonzero_rewards.append(dict(step=step+1, reward=float(reward)))
                if terminated or truncated:
                    break
            comparisons.append(dict(fire_on_reset=fire, steps=step+1, scoring_events=nonzero_rewards,
                                    changed_frame_inputs=changed_frames, initial_foreground_pixels=int(initial.sum()),
                                    terminated=bool(terminated), truncated=bool(truncated)))
    finally:
        env.close()
    # Multiple scoring events without FIRE establish automatic startup and continued play.
    no_fire_works = len(comparisons[0]["scoring_events"]) >= 2
    result = dict(environment=metadata, startup=comparisons, no_fire_progress_verified=no_fire_works,
                  decision="No FIRE at reset or points" if no_fire_works else "Investigate before training")
    write_json(output / "environment_inspection.json", result)
    if not no_fire_works:
        raise AssertionError("No-FIRE point progression not established")
    return result


def validate_absolute_objective(output):
    """Two-step exact enumeration distinguishes G_t from gamma^t G_t."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = Config(seed=812, hidden_sizes=(8,), discount_weighting="absolute")
    device = configure_runtime(config)
    gamma, p = 0.5, 0.3
    gradients = {}
    for convention in ("relative", "absolute"):
        theta = torch.tensor(np.log(p/(1-p)), dtype=torch.float64, requires_grad=True)
        logits = torch.stack([theta*0, theta])
        distribution = Categorical(logits=logits)
        loss = theta * 0
        for a0 in (0, 1):
            for a1 in (0, 1):
                actions = torch.tensor([a0, a1])
                probability = distribution.probs[a0].detach()*distribution.probs[a1].detach()
                weights = torch.tensor(policy_weights([a0, 2*a1], gamma, convention), dtype=torch.float64)
                loss = loss + probability * policy_loss(logits.expand(2, -1), actions, weights)
        gradients[convention] = float(torch.autograd.grad(loss, theta)[0])
    expected_absolute = -(1+2*gamma)*p*(1-p)
    expected_relative = -3*p*(1-p)
    theta = np.log(p/(1-p))
    objective = lambda value: -(1+2*gamma)/(1+np.exp(-value))
    finite_difference = (objective(theta+1e-6)-objective(theta-1e-6))/2e-6
    model = make_policy(config)
    full = deepcopy(model)
    observations = np.random.randn(11, 4).astype(np.float32)
    with torch.no_grad():
        dist = Categorical(logits=model(torch.from_numpy(observations)))
        actions = dist.sample()
        sampled_log_probs = dist.log_prob(actions).numpy()
    rewards = np.array([1, 0, -1, 0, 2, 0, 1, -1, 0, 1, 3], dtype=float)
    trajectory = SimpleNamespace(observations=observations, actions=actions.numpy(),
                                 rewards=rewards, log_probs=sampled_log_probs)
    full_weights = torch.tensor(policy_weights(rewards, 0.95, "absolute"), dtype=torch.float32)
    (policy_loss(full(torch.from_numpy(observations)), actions, full_weights)/3).backward()
    backward_episode(model, trajectory, 0.95, device, 3, 3, "absolute")
    chunk_error = max(float((a.grad-b.grad).abs().max()) for a, b in zip(model.parameters(), full.parameters()))
    checks = dict(absolute_matches_analytic=abs(gradients["absolute"]-expected_absolute) < 1e-12,
                  relative_matches_analytic=abs(gradients["relative"]-expected_relative) < 1e-12,
                  strict_objective_finite_difference=abs(gradients["absolute"]-finite_difference) < 1e-8,
                  conventions_are_distinct=abs(gradients["relative"]-gradients["absolute"]) > 0.1,
                  absolute_chunk_gradient_equivalence=chunk_error < 2e-6,
                  gamma_zero_time_zero=np.array_equal(policy_weights([1, 2, 3], 0, "absolute"), [1, 0, 0]),
                  single_step_same=np.array_equal(policy_weights([7], 0.95, "absolute"), [7]))
    checks = {key: bool(value) for key, value in checks.items()}
    result = dict(checks=checks, all_passed=all(checks.values()),
                  evidence=dict(gamma=gamma, action1_probability=p, rewards="r0=a0, r1=2*a1",
                                expected_discounted_reward=(1+2*gamma)*p,
                                loss_gradients=gradients, analytic_absolute=expected_absolute,
                                analytic_relative=expected_relative, finite_difference=float(finite_difference),
                                chunk_gradient_max_error=chunk_error,
                                exponent_origin="Episode timestep zero; never reset at chunks or points"))
    write_json(output / "absolute_objective.json", result)
    print(f"Absolute-objective validation: {sum(checks.values())}/{len(checks)} passed", flush=True)
    if not result["all_passed"]:
        raise AssertionError(checks)
    return result
