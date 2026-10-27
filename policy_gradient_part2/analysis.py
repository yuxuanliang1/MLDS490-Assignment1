"""Paired fixed-policy gradient diagnostics, separate from episode-reward variability."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from policy_gradient_part1.config import Config
from policy_gradient_part1.environments import make_environment
from policy_gradient_part1.models import make_policy
from policy_gradient_part1.reinforce import backward_episode
from policy_gradient_part1.rollout import sample_episode
from policy_gradient_part1.runtime import configure_runtime, load_checkpoint, write_json


def variance_plot(result, path, title):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(6.4,4))
    ax.bar(["No baseline", "Previous-return EMA"], [result["trace_covariance_without"],result["trace_covariance_with"]],
           color=["steelblue","darkorange"])
    ax.set(ylabel="Trace of gradient covariance",title=title)
    ax.text(.5,.95,f"Relative reduction: {result['relative_reduction']:.1%}",ha="center",va="top",transform=ax.transAxes)
    fig.tight_layout();fig.savefig(path,dpi=300);plt.close(fig)


def diagnose(checkpoint, output, episodes=64, seed=4000000, device="cpu", figure=None):
    if episodes < 32 or episodes%2:
        raise ValueError("Use an even number of at least 32 independent diagnostic episodes")
    output=Path(output)
    if (output/"summary.json").exists(): raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    saved=load_checkpoint(checkpoint)
    cfg=Config(**saved["config"]).validate();cfg.seed=seed;cfg.device=device
    resolved=configure_runtime(cfg);model=make_policy(cfg).to(resolved);model.load_state_dict(saved["model"]);model.eval()
    baseline=float(saved["baseline"]["value"])
    g0=[]; h=[];rewards=[]
    env=make_environment(cfg)
    try:
        for i in range(episodes):
            trajectory=sample_episode(env,model,cfg,resolved,seed+i)
            if cfg.environment=="ALE/Pong-v5" and trajectory.truncated:
                raise RuntimeError("Incomplete Pong diagnostic trajectory")
            vectors=[]
            for b in (0.,1.):
                model.zero_grad(set_to_none=True)
                backward_episode(model,trajectory,cfg.gamma,resolved,cfg.gradient_chunk_size,1,cfg.discount_weighting,b)
                # Positive ascent gradient; trainer computes negative loss gradients.
                vectors.append(-torch.cat([p.grad.flatten() for p in model.parameters()]).cpu().numpy().copy())
            g0.append(vectors[0]);h.append(vectors[0]-vectors[1]);rewards.append(trajectory.total_reward)
    finally: env.close()
    g0=np.asarray(g0,dtype=np.float64);h=np.asarray(h,dtype=np.float64)
    split=episodes//2
    gc=g0[:split]-g0[:split].mean(axis=0);hc=h[:split]-h[:split].mean(axis=0)
    denominator=float((hc*hc).sum())
    optimal=float((gc*hc).sum()/denominator) if denominator else 0.
    # Only the disjoint held-out half is used to assess both baseline choices.
    g=g0[split:];score=h[split:];gb=g-baseline*score;go=g-optimal*score
    v0=g.var(axis=0,ddof=1);vb=gb.var(axis=0,ddof=1);vo=go.var(axis=0,ddof=1)
    total=float(v0.sum());paired=gb-g
    result=dict(environment=cfg.environment,checkpoint=str(checkpoint),training_seed=saved["config"]["seed"],
                diagnostic_seed=seed,calibration_episodes=split,held_out_episodes=split,
                baseline=baseline,baseline_source="Frozen checkpoint EMA; no diagnostic returns used",
                trace_covariance_without=total,trace_covariance_with=float(vb.sum()),
                relative_reduction=float(1-vb.sum()/total) if total else 0.,
                mean_without_norm=float(np.linalg.norm(g.mean(axis=0))),mean_with_norm=float(np.linalg.norm(gb.mean(axis=0))),
                mean_difference_norm=float(np.linalg.norm(paired.mean(axis=0))),
                paired_mean_difference_noise_scale=float(np.sqrt(paired.var(axis=0,ddof=1).sum()/split)),
                secondary_calibrated_constant=optimal,secondary_trace_covariance=float(vo.sum()),
                secondary_relative_reduction=float(1-vo.sum()/total) if total else 0.,
                discount_weighting=cfg.discount_weighting,degenerate_zero_gradient_variance=total==0,
                interpretation="Fixed policy, paired complete trajectories. Secondary constant fitted on disjoint calibration half; not used for training. Finite-sample estimates, no universal reduction guarantee.")
    np.savez_compressed(output/"gradient_statistics.npz",mean_without=g.mean(axis=0),mean_with=gb.mean(axis=0),
                        variance_without=v0,variance_with=vb,variance_secondary=vo,
                        paired_mean_difference=paired.mean(axis=0),episode_rewards=np.asarray(rewards),
                        sample_gradient_norm_without=np.linalg.norm(g,axis=1),sample_gradient_norm_with=np.linalg.norm(gb,axis=1))
    write_json(output/"summary.json",result)
    variance_plot(result,figure or output/"gradient_variance.png",f"{cfg.environment}: fixed final policy ({split} paired games)")
    print(f"Fixed-policy variance reduction={result['relative_reduction']:.2%}; secondary={result['secondary_relative_reduction']:.2%}",flush=True)
    return result


def compare(part1, part2, output, title):
    """Compare actual saved histories/evaluations, never inferred or padded episodes."""
    from policy_gradient_part1.evaluator import reward_statistics
    def read(path,key):
        with Path(path).open(newline="",encoding="utf-8") as f: return np.array([float(r[key]) for r in csv.DictReader(f)])
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4))
    metrics={}
    for label,root in (("Part 1",Path(part1)),("Part 2",Path(part2))):
        train_dir=root/("seed_11" if "cartpole" in str(root) else "formal")
        rewards=read(train_dir/"history.csv","reward")
        axes[0].plot(np.arange(100,len(rewards)+1),np.convolve(rewards,np.ones(100)/100,"valid"),label=label)
        metrics[label]=dict(training_episodes=len(rewards),last100_mean=float(rewards[-100:].mean()),
                            last100_sample_std=float(rewards[-100:].std(ddof=1)),
                            mean_absolute_successive_MA100_change=float(np.abs(np.diff(np.convolve(rewards,np.ones(100)/100,"valid"))).mean()))
        evalpath=root/"evaluation/rewards.csv"
        if evalpath.exists() and (root/"evaluation/summary.json").exists():
            er=read(evalpath,"reward")
            if len(er)!=500: raise ValueError("Expected exactly 500 final episodes")
            metrics[label]["evaluation"]=reward_statistics(er)
            axes[1].hist(er,bins=20,alpha=.5,label=label)
    axes[0].set(xlabel="Training episode",ylabel="100-episode mean reward",title="Matched training budget")
    axes[1].set(xlabel="Evaluation reward",ylabel="Episode count",title="500 independent episodes per policy")
    for ax in axes: ax.legend();ax.grid(alpha=.2)
    fig.suptitle(title);fig.tight_layout();fig.savefig(output,dpi=300);plt.close(fig)
    write_json(output.with_suffix(".json"),metrics)
    return metrics
