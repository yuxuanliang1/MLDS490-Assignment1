"""Auditable episodic training, diagnostics, and resumable batch boundaries."""

import csv
from pathlib import Path
import signal
from time import perf_counter

import numpy as np
import torch

from .config import Config
from .environments import environment_metadata, make_environment
from .models import make_policy
from .plots import training_plot
from .reinforce import backward_episode, reward_to_go
from .rollout import sample_episode
from .runtime import (configure_runtime, load_checkpoint, provenance, restore_rng,
                      save_checkpoint, write_json)


def write_history(path, history):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)


def train(config, output, resume=None, figure=None, baseline=None):
    config.validate()
    output = Path(output)
    if not resume and (output / "history.csv").exists():
        raise FileExistsError(f"Existing run: {output}; use a fresh directory or --resume")
    output.mkdir(parents=True, exist_ok=True)
    device = configure_runtime(config)
    model = make_policy(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    env = make_environment(config)
    history = []
    completed = 0
    metadata = provenance(config, model)
    if baseline is not None:
        metadata["objective"] = metadata["objective"].replace("G_t", "(G_t - b)").replace(
            "no baseline, entropy bonus or return scaling", "past-episode constant baseline; no entropy bonus or return scaling")
        metadata["baseline_protocol"] = baseline.protocol()
        import hashlib
        metadata["source_sha256"].update({str(p.relative_to(Path(__file__).resolve().parents[1])).replace("\\", "/"):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__).resolve().parents[1] / "policy_gradient_part2").glob("*.py")})
    metadata["environment"] = environment_metadata(env, config)
    if resume:
        checkpoint = load_checkpoint(resume, device)
        if (baseline is None) != ("baseline" not in checkpoint):
            raise ValueError("Cannot resume between baseline and baseline-free training")
        if baseline is not None:
            baseline.load_state_dict(checkpoint["baseline"])
        previous = Config(**checkpoint["config"]).validate()
        allowed_changes = {"episodes", "device", "checkpoint_interval", "log_interval", "threads"}
        incompatible = [k for k, v in previous.to_dict().items()
                        if k not in allowed_changes and v != config.to_dict()[k]]
        if incompatible:
            raise ValueError(f"Resume changes algorithm/environment: {incompatible}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        completed = checkpoint["completed_episodes"]
        history = checkpoint["history"]
        if completed != len(history) or config.episodes <= completed:
            raise ValueError("Resume budget must exceed a consistent completed episode count")
        restore_rng(checkpoint["rng"], device)
        metadata["resumed_from"] = str(Path(resume).resolve())
        metadata["parent_metadata"] = checkpoint["metadata"]
        # Recover exactly the checkpoint history; any later uncheckpointed CSV rows are discarded.
        write_history(output / "history.csv", history)
    write_json(output / "config.json", config.to_dict())
    write_json(output / "metadata.json", metadata)
    if baseline is not None:
        write_json(output / "baseline_protocol.json", baseline.protocol())
    stop_requested = []

    def request_stop(signum, frame):
        stop_requested.append(signum)
        print("Stop requested; finishing the current update and saving a checkpoint.", flush=True)

    old_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, request_stop)
    last_saved = completed
    start = perf_counter()
    model.train()
    try:
        while completed < config.episodes:
            batch_size = min(config.batch_episodes, config.episodes - completed)
            optimizer.zero_grad(set_to_none=True)
            batch_rows = []
            batch_baseline = baseline.estimate() if baseline is not None else 0.0
            for _ in range(batch_size):
                episode_start = perf_counter()
                episode_index = completed + len(batch_rows)
                trajectory = sample_episode(env, model, config, device, config.seed + episode_index,
                                            trace_steps=10 if episode_index == 0 else 0)
                if baseline is not None:
                    from policy_gradient_part2.baseline import validate_trajectory
                    validate_trajectory(trajectory)
                loss, logprob_error = backward_episode(model, trajectory, config.gamma, device,
                                                       config.gradient_chunk_size, batch_size,
                                                       config.discount_weighting, batch_baseline)
                if trajectory.trace:
                    write_json(output / "first_steps.json", trajectory.trace)
                batch_rows.append(dict(episode=episode_index+1, reset_seed=config.seed+episode_index,
                                       reward=trajectory.total_reward,
                                       discounted_return=float(reward_to_go(trajectory.rewards, config.gamma)[0]),
                                       steps=len(trajectory.rewards), terminated=trajectory.terminated,
                                       truncated=trajectory.truncated, loss_contribution=loss,
                                       entropy=float(trajectory.entropies.mean()),
                                       probability_action0=float(trajectory.probabilities[:, 0].mean()),
                                       action0_fraction=float((trajectory.actions == 0).mean()),
                                       logprob_recompute_max_error=logprob_error,
                                       seconds=perf_counter()-episode_start))
                if baseline is not None:
                    batch_rows[-1]["baseline_used"] = batch_baseline
                del trajectory
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            norm = float(torch.linalg.vector_norm(torch.stack([g.norm() for g in gradients])))
            if not np.isfinite(norm) or any(not torch.isfinite(g).all() for g in gradients):
                raise FloatingPointError("Nonfinite gradients; no optimizer step applied")
            if config.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite policy parameters")
            for row_index, row in enumerate(batch_rows):
                if baseline is not None:
                    # No current-batch return influences any gradient in this batch.
                    baseline.update(row["discounted_return"])
                    row["baseline_after_episode"] = baseline.estimate()
                    row["baseline_observations"] = baseline.count
                    previous_rewards = [r["reward"] for r in history[-99:]] + [r["reward"] for r in batch_rows[:row_index+1]]
                    row["moving_average_100"] = (float(np.mean(previous_rewards[-100:]))
                                                 if len(history)+row_index+1 >= 100 else "")
                row["gradient_norm_before_clip"] = norm
                row["gpu_peak_memory_bytes"] = (torch.cuda.max_memory_allocated(0)
                                                 if device.type == "cuda" else 0)
            history.extend(batch_rows)
            completed += batch_size
            write_history(output / "history.csv", history)
            if completed % config.log_interval < batch_size or completed == config.episodes:
                recent = [row["reward"] for row in history[-100:]]
                print(f"episode={completed} reward={history[-1]['reward']:.1f} "
                      f"mean_last{len(recent)}={np.mean(recent):.2f} "
                      f"entropy={history[-1]['entropy']:.4f} grad_norm={norm:.3g}", flush=True)
            if completed - last_saved >= config.checkpoint_interval or completed == config.episodes or stop_requested:
                save_checkpoint(output / "latest.pt", model, optimizer, config, completed, history, device, metadata, baseline)
                save_checkpoint(output / f"episode_{completed:06d}.pt", model, optimizer, config,
                                completed, history, device, metadata, baseline)
                last_saved = completed
            if stop_requested:
                break
    finally:
        env.close()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    summary = dict(completed_episodes=completed, requested_episodes=config.episodes,
                   complete=completed == config.episodes, elapsed_this_invocation_seconds=perf_counter()-start,
                   mean_last100=float(np.mean([row["reward"] for row in history[-100:]])),
                   training_seed=config.seed, checkpoint="latest.pt",
                   selection="Fixed budget final checkpoint; no final evaluation used for selection")
    if baseline is not None:
        summary["baseline"] = baseline.state_dict()
    write_json(output / "summary.json", summary)
    training_plot(history, figure or output / "training_rewards.png", config.environment + " REINFORCE training")
    return summary
