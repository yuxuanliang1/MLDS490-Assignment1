"""GPU pilots, training-only selection, formal training, then separate evaluation."""
import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy_gradient_part1.config import Config
from policy_gradient_part1.trainer import train
from policy_gradient_part1.runtime import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-episodes", type=int, default=30)
    parser.add_argument("--formal-episodes", type=int, default=0)
    parser.add_argument("--stage", choices=["pilots", "formal", "all"], default="pilots")
    parser.add_argument("--discount-weighting", choices=["relative", "absolute"], default="relative")
    parser.add_argument("--output", type=Path, default=ROOT / "results/policy_gradient_part1/pong")
    args = parser.parse_args()
    if args.stage != "pilots" and args.formal_episodes < 1:
        parser.error("Formal training requires an explicit positive episode budget")
    smoke = ROOT / "results/policy_gradient_part1/remote/gpu_smoke_passed.json"
    if not smoke.exists() or not json.loads(smoke.read_text())["passed"]:
        raise RuntimeError("Complete the explicit single-GPU smoke test first")
    base = Config.load(ROOT / "configs/pong_part1.json")
    base.discount_weighting = args.discount_weighting
    suffix = "_absolute" if args.discount_weighting == "absolute" else ""
    protocol = dict(pilot_episodes=args.pilot_episodes, formal_episodes=args.formal_episodes,
                    pilot_seed=101, formal_seed=11, learning_rates=[0.01, 0.003, 0.001],
                    selection="Prefer pilots with final-10 mean entropy >=0.1 and action0 fraction in [0.05,0.95]; "
                    "among those maximize last-20 training mean, breaking ties toward smaller learning rate. "
                    "If all collapse, use the smallest rate and label instability.",
                    formal_budget_reason="Compute-limited fixed budget declared before final evaluation; "
                    "at least ten 100-episode windows at a 1000-episode target. No convergence promise.",
                    primary_checkpoint="Final-budget seed 11 checkpoint, selected before evaluation",
                    evaluation_seed=1000000, evaluation_episodes=500, discount_weighting=args.discount_weighting)
    write_json(args.output / f"gpu_protocol{suffix}_{args.stage}.json", protocol)
    pilots = []
    for lr in protocol["learning_rates"] if args.stage != "formal" else []:
        directory = args.output / f"gpu_pilots{suffix}" / f"lr_{lr:g}"
        summary = train(replace(base, learning_rate=lr, episodes=args.pilot_episodes, seed=101,
                                device="cuda:0", checkpoint_interval=10, log_interval=10), directory)
        with (directory / "history.csv").open() as handle:
            history = list(csv.DictReader(handle))
        entropy = float(np.mean([float(row["entropy"]) for row in history[-10:]]))
        action0 = float(np.mean([float(row["action0_fraction"]) for row in history[-10:]]))
        score = float(np.mean([float(row["reward"]) for row in history[-20:]]))
        pilots.append(dict(learning_rate=lr, entropy_last10=entropy, action0_last10=action0,
                           mean_last20=score, eligible=entropy >= 0.1 and 0.05 <= action0 <= 0.95, **summary))
    if args.stage == "formal":
        selection = json.loads((args.output / f"gpu_pilot_selection{suffix}.json").read_text())
        selected = selection["selected_learning_rate"]
        predicted_seconds = selection["measured_mean_seconds_per_pilot_episode"]
    else:
        eligible = [row for row in pilots if row["eligible"]]
        if eligible:
            selected = max(eligible, key=lambda row: (row["mean_last20"], -row["learning_rate"]))["learning_rate"]
        else:
            selected = min(protocol["learning_rates"])
        predicted_seconds = sum(row["elapsed_this_invocation_seconds"] for row in pilots) / (3*args.pilot_episodes)
        selection = dict(pilots=pilots, selected_learning_rate=selected, all_pilots_collapsed=not bool(eligible),
                         measured_mean_seconds_per_pilot_episode=predicted_seconds,
                         estimated_formal_seconds=predicted_seconds*args.formal_episodes,
                         criterion=protocol["selection"])
        write_json(args.output / f"gpu_pilot_selection{suffix}.json", selection)
    print(f"GPU pilots complete. Selected LR={selected}; observed {predicted_seconds:.2f} seconds/game; "
          f"formal estimate {predicted_seconds*args.formal_episodes/60:.1f} min, excluding changing game length", flush=True)
    if args.stage == "pilots":
        return
    figures = ROOT / "figures/policy_gradient_part1/pong"
    formal = train(replace(base, learning_rate=selected, episodes=args.formal_episodes, seed=11,
                           device="cuda:0", checkpoint_interval=100, log_interval=25),
                   args.output / "formal", figure=figures / "training_rewards.png")
    if not formal["complete"]:
        raise RuntimeError("Training stopped early; final evaluation not launched")
    # A separate process loads the saved model; no final-set selection loop.
    command = [sys.executable, str(ROOT / "scripts/evaluate_pg.py"), "--device", "cuda:0",
               "--checkpoint", str(args.output / "formal/latest.pt"), "--output", str(args.output / "evaluation"),
               "--episodes", "500", "--seed", "1000000", "--figure", str(figures / "evaluation_histogram.png")]
    subprocess.run(command, check=True)
    evaluation = json.loads((args.output / "evaluation/summary.json").read_text())
    write_json(args.output / "gpu_experiment_summary.json", dict(protocol=protocol, selection=selection,
                                                                formal_training=formal, final_evaluation=evaluation))


if __name__ == "__main__":
    main()
