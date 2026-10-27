"""Independent checkpoint evaluation. Final mode requires exactly 500 episodes."""

import csv
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .config import Config
from .environments import environment_metadata, make_environment
from .models import make_policy
from .plots import histogram
from .rollout import sample_episode
from .runtime import configure_runtime, load_checkpoint, write_json


def reward_statistics(rewards):
    rewards = np.asarray(rewards, dtype=float)
    if len(rewards) < 2 or not np.isfinite(rewards).all():
        raise ValueError("At least two finite rewards required")
    mean = float(rewards.mean())
    sd = float(rewards.std(ddof=1))
    half_width = 1.96 * sd / np.sqrt(len(rewards))
    return dict(episodes=len(rewards), mean=mean, sample_std=sd, minimum=float(rewards.min()),
                maximum=float(rewards.max()), ci95_normal=[mean-half_width, mean+half_width])


def evaluate(checkpoint_path, output, episodes=500, seed=1000000, device="cpu", mode="stochastic",
             role="final", reference=None, figure=None):
    if episodes < 2 or (role == "final" and (episodes != 500 or reference is not None)):
        raise ValueError("Final evaluation requires exactly 500 trained-policy episodes")
    if mode not in ("stochastic", "greedy") or reference not in (None, "random", "fixed"):
        raise ValueError("Invalid evaluation convention")
    checkpoint_path = Path(checkpoint_path)
    output = Path(output)
    if (output / "rewards.csv").exists():
        raise FileExistsError(f"Existing evaluation: {output}; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(checkpoint_path)
    config = Config(**checkpoint["config"]).validate()
    config.device = device
    config.seed = seed
    resolved_device = configure_runtime(config)
    model = make_policy(config).to(resolved_device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    env = make_environment(config)
    rows = []
    start = perf_counter()
    try:
        with (output / "rewards.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["episode", "reset_seed", "reward", "steps",
                                                       "terminated", "truncated", "action0_fraction"])
            writer.writeheader()
            for index in range(episodes):
                trajectory = sample_episode(env, model, config, resolved_device, seed+index,
                                            mode=mode, reference=reference, trace_steps=10 if index == 0 else 0)
                if config.environment == "ALE/Pong-v5" and role == "final" and trajectory.truncated:
                    raise RuntimeError("Pong final evaluation encountered an incomplete, time-limited game; "
                                       "this run is diagnostic, not a valid final 500-game result")
                row = dict(episode=index+1, reset_seed=seed+index, reward=trajectory.total_reward,
                           steps=len(trajectory.rewards), terminated=trajectory.terminated,
                           truncated=trajectory.truncated, action0_fraction=float((trajectory.actions == 0).mean()))
                writer.writerow(row)
                handle.flush()
                rows.append(row)
                if trajectory.trace:
                    write_json(output / "first_steps.json", trajectory.trace)
                if (index+1) % 50 == 0:
                    print(f"evaluation episodes={index+1}/{episodes} mean={np.mean([r['reward'] for r in rows]):.3f}",
                          flush=True)
    finally:
        env.close()
    rewards = [row["reward"] for row in rows]
    result = reward_statistics(rewards)
    result.update(checkpoint=str(checkpoint_path.resolve()),
                  checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                  training_episodes=checkpoint["completed_episodes"], training_seed=checkpoint["config"]["seed"],
                  evaluation_seed=seed, policy_mode=mode, reference=reference, role=role,
                  elapsed_seconds=perf_counter()-start, environment=environment_metadata(env, config),
                  terminated_count=sum(row["terminated"] for row in rows),
                  truncated_count=sum(row["truncated"] for row in rows),
                  action0_fraction=float(np.mean([row["action0_fraction"] for row in rows])))
    write_json(output / "summary.json", result)
    histogram(rewards, figure or output / "evaluation_histogram.png",
              f"{config.environment}: {episodes} {mode} evaluation episodes")
    print(f"mean={result['mean']:.3f} sample_sd={result['sample_std']:.3f} n={episodes}", flush=True)
    return result
