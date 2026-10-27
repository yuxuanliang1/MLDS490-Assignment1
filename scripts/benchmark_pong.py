"""Ten complete-game bounded benchmark; original sampler, model and estimator."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from policy_gradient_part1.config import Config
from policy_gradient_part1.environments import make_environment, environment_metadata
from policy_gradient_part1.models import make_policy
from policy_gradient_part1.reinforce import backward_episode
from policy_gradient_part1.rollout import sample_episode
from policy_gradient_part1.runtime import configure_runtime, provenance, write_json
from policy_gradient_part1.validation import validate_absolute_objective


class TimedEnvironment:
    def __init__(self, env):
        self.env = env
        self.step_seconds = self.reset_seconds = 0.0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        start = perf_counter()
        result = self.env.reset(**kwargs)
        self.reset_seconds += perf_counter()-start
        return result

    def step(self, action):
        start = perf_counter()
        result = self.env.step(action)
        self.step_seconds += perf_counter()-start
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    if args.episodes not in range(10, 21):
        parser.error("Benchmark is bounded to 10--20 complete games")
    args.output.mkdir(parents=True, exist_ok=False)
    validate_absolute_objective(args.output / "validation")
    cfg = Config.load(ROOT / "configs/pong_part1_absolute.json")
    cfg.device, cfg.episodes, cfg.seed = "cuda:0", args.episodes, 101
    device = configure_runtime(cfg)
    model = make_policy(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    env = TimedEnvironment(make_environment(cfg))
    metadata = provenance(cfg, model)
    metadata.update(environment=environment_metadata(env, cfg), load_start=list(os.getloadavg()))
    # Warm only kernels; no optimizer update, action sampling, or RNG consumption.
    model(torch.zeros((cfg.gradient_chunk_size, 1, 80, 80), device=device)).sum().backward()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    forward_wall, forward_events = [0.0], []
    measure_forward = [False]
    start_data = []

    def before_forward(module, inputs):
        if measure_forward[0]:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            start_data.append((perf_counter(), event))

    def after_forward(module, inputs, output):
        if measure_forward[0]:
            started, event = start_data.pop()
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            forward_wall[0] += perf_counter()-started
            forward_events.append((event, end))

    model.register_forward_pre_hook(before_forward)
    model.register_forward_hook(after_forward)
    samples, stop = [], threading.Event()
    uuid = os.environ["CUDA_VISIBLE_DEVICES"]

    def monitor():
        previous_cpu = None
        while not stop.is_set():
            cpu = [int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:]]
            busy = None
            if previous_cpu:
                delta = sum(cpu)-sum(previous_cpu)
                busy = 100*(1-(cpu[3]+cpu[4]-previous_cpu[3]-previous_cpu[4])/delta) if delta else 0
            previous_cpu = cpu
            query = subprocess.run(["nvidia-smi", "-i", uuid,
                                    "--query-gpu=utilization.gpu,memory.used,power.draw",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True)
            if query.returncode == 0:
                gpu, memory, power = [float(x.strip()) for x in query.stdout.split(',')]
                samples.append(dict(seconds=perf_counter()-start, gpu_percent=gpu,
                                    used_vram_MiB=memory, power_W=power, system_cpu_percent=busy))
            stop.wait(1)

    rows = []
    backward_seconds = optimizer_seconds = 0.0
    start, cpu_start = perf_counter(), process_time()
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        for batch_start in range(0, args.episodes, cfg.batch_episodes):
            batch_size = min(cfg.batch_episodes, args.episodes-batch_start)
            optimizer.zero_grad(set_to_none=True)
            for index in range(batch_start, batch_start+batch_size):
                measure_forward[0] = True
                trajectory = sample_episode(env, model, cfg, device, cfg.seed+index)
                measure_forward[0] = False
                torch.cuda.synchronize()
                if not trajectory.terminated or trajectory.truncated:
                    raise RuntimeError("Benchmark requires complete Pong games")
                backward_start = perf_counter()
                loss, error = backward_episode(model, trajectory, cfg.gamma, device,
                                                cfg.gradient_chunk_size, batch_size, cfg.discount_weighting)
                torch.cuda.synchronize()
                backward_seconds += perf_counter()-backward_start
                rows.append(dict(episode=index+1, seed=cfg.seed+index, steps=len(trajectory.rewards),
                                 reward=trajectory.total_reward, loss_contribution=loss,
                                 logprob_recompute_error=error,
                                 action_sha256=hashlib.sha256(trajectory.actions.tobytes()).hexdigest()))
                if not np.isfinite(loss) or error > 1e-4:
                    raise AssertionError("Invalid gradients or on-policy recomputation")
            update_start = perf_counter()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if any(not torch.isfinite(g).all() for g in gradients):
                raise FloatingPointError("Nonfinite benchmark gradient")
            optimizer.step()
            torch.cuda.synchronize()
            optimizer_seconds += perf_counter()-update_start
            print(f"benchmark episodes={len(rows)}/{args.episodes} mean_reward={np.mean([r['reward'] for r in rows]):.3f}", flush=True)
        elapsed, cpu_elapsed = perf_counter()-start, process_time()-cpu_start
    finally:
        stop.set()
        thread.join(timeout=5)
        env.close()
    steps = sum(row["steps"] for row in rows)
    gpu_forward = sum(a.elapsed_time(b) for a, b in forward_events)/1000
    result = dict(host=platform.node(), config=cfg.to_dict(), metadata=metadata,
                  complete_episodes=len(rows), total_steps=steps, elapsed_seconds=elapsed,
                  seconds_per_episode=elapsed/len(rows), episodes_per_second=len(rows)/elapsed,
                  environment_steps_per_second=steps/elapsed,
                  env_step_seconds=env.step_seconds, env_reset_seconds=env.reset_seconds,
                  forward_host_wall_seconds=forward_wall[0], forward_device_seconds=gpu_forward,
                  backward_seconds=backward_seconds, optimizer_seconds=optimizer_seconds,
                  other_rollout_and_transfer_seconds=elapsed-env.step_seconds-env.reset_seconds-forward_wall[0]-backward_seconds-optimizer_seconds,
                  process_cpu_percent_of_one_core=100*cpu_elapsed/elapsed,
                  gpu_utilization_mean_percent=float(np.mean([s["gpu_percent"] for s in samples])),
                  system_cpu_mean_percent=float(np.mean([s["system_cpu_percent"] for s in samples if s["system_cpu_percent"] is not None])),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), load_end=list(os.getloadavg()),
                  timing_scope="Warm kernels, complete rollouts and actual gamma^t G_t updates; excludes setup, plots/checkpoints. "
                  "CUDA events measure forward device time; host and device forward times overlap and must not be added. "
                  "Residual includes categorical sampling, diagnostics, preprocessing, Python and transfers.")
    write_json(args.output / "summary.json", result)
    write_json(args.output / "telemetry.json", samples)
    with (args.output / "episodes.csv").open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
