"""Predeclared pilot selection, three fresh training seeds, independent evaluation."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy_gradient_part1.config import Config
from policy_gradient_part1.trainer import train
from policy_gradient_part1.evaluator import evaluate
from policy_gradient_part1.runtime import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/policy_gradient_part1/cartpole")
    parser.add_argument("--pilot-episodes", type=int, default=300)
    parser.add_argument("--formal-episodes", type=int, default=1500)
    parser.add_argument("--resume-suite", action="store_true", help="Reuse matching completed stages without repeating evaluations")
    args = parser.parse_args()
    base = Config.load(ROOT / "configs/cartpole_part1.json")

    def train_stage(config, directory):
        summary_path = directory / "summary.json"
        if args.resume_suite and summary_path.exists():
            saved = json.loads(summary_path.read_text(encoding="utf-8"))
            saved_config = Config.load(directory / "config.json")
            if saved_config.to_dict() != config.to_dict() or not saved["complete"]:
                raise ValueError(f"Existing stage configuration/completion differs: {directory}")
            return saved
        return train(config, directory)

    def evaluation_stage(checkpoint, directory, episodes=500, seed=1000000, role="final", reference=None, figure=None):
        if args.resume_suite and (directory / "summary.json").exists():
            saved = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            expected = dict(episodes=episodes, evaluation_seed=seed, role=role, reference=reference,
                            policy_mode="stochastic", checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            if any(saved[k] != v for k, v in expected.items()):
                raise ValueError(f"Existing evaluation differs: {directory}")
            return saved
        return evaluate(checkpoint, directory, episodes=episodes, seed=seed, role=role, reference=reference, figure=figure)
    protocol = dict(pilot_seed=101, pilot_learning_rates=[0.01, 0.003, 0.001],
                    pilot_episodes=args.pilot_episodes, formal_episodes=args.formal_episodes,
                    selection="Largest trailing-100 training mean in equal-budget pilots; tie chooses larger LR",
                    primary_training_seed=11, other_training_seeds=[22, 33],
                    primary_checkpoint="Fixed-budget final checkpoint of seed 11; chosen before evaluation",
                    final_evaluation_episodes=500, final_evaluation_seed=1000000,
                    robustness_evaluation_episodes=100, robustness_evaluation_seed=2000000,
                    formal_budget_reason="Five times pilot duration to test sustained improvement and instability",
                    no_baseline=True)
    write_json(args.output / "protocol.json", protocol)
    pilots = []
    for lr in protocol["pilot_learning_rates"]:
        directory = args.output / "pilots" / f"lr_{lr:g}"
        config = replace(base, learning_rate=lr, seed=101, episodes=args.pilot_episodes)
        summary = train_stage(config, directory)
        pilots.append(dict(learning_rate=lr, **summary))
    selected = max(pilots, key=lambda row: (row["mean_last100"], row["learning_rate"]))["learning_rate"]
    write_json(args.output / "pilot_selection.json", dict(pilots=pilots, selected_learning_rate=selected,
                                                          criterion=protocol["selection"]))
    # Persist selection BEFORE any fresh formal run or final evaluation.
    formal = []
    for seed in (11, 22, 33):
        directory = args.output / f"seed_{seed}"
        config = replace(base, learning_rate=selected, seed=seed, episodes=args.formal_episodes)
        formal.append(dict(seed=seed, **train_stage(config, directory)))
    figures = ROOT / "figures/policy_gradient_part1/cartpole"
    figures.mkdir(parents=True, exist_ok=True)
    from shutil import copy2
    copy2(args.output / "seed_11/training_rewards.png", figures / "training_rewards.png")
    primary = evaluation_stage(args.output / "seed_11/latest.pt", args.output / "evaluation", seed=1000000,
                       figure=figures / "evaluation_histogram.png")
    random = evaluation_stage(args.output / "seed_11/latest.pt", args.output / "random_reference", seed=3000000,
                      role="diagnostic", reference="random")
    robustness = []
    for seed in (11, 22, 33):
        result = evaluation_stage(args.output / f"seed_{seed}/latest.pt", args.output / f"seed_{seed}/validation",
                          episodes=100, seed=2000000, role="diagnostic")
        robustness.append(result)
    difference = primary["mean"] - random["mean"]
    se = np.sqrt(primary["sample_std"]**2/500 + random["sample_std"]**2/500)
    summary = dict(protocol=protocol, selected_learning_rate=selected, pilots=pilots, formal_training=formal,
                   primary=primary, random_reference=random, seed_robustness=robustness,
                   trained_minus_random=difference, difference_ci95_normal=[difference-1.96*se, difference+1.96*se],
                   improved_beyond_random=bool(difference-1.96*se > 0))
    write_json(args.output / "experiment_summary.json", summary)
    import csv
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for row in pilots:
        path = args.output / "pilots" / f"lr_{row['learning_rate']:g}" / "history.csv"
        with path.open() as handle:
            rewards = np.array([float(r["reward"]) for r in csv.DictReader(handle)])
        axes[0].plot(np.arange(100, len(rewards)+1), np.convolve(rewards, np.ones(100)/100, "valid"),
                     label=f"LR {row['learning_rate']:g}")
    for seed in (11, 22, 33):
        with (args.output / f"seed_{seed}/history.csv").open() as handle:
            rewards = np.array([float(r["reward"]) for r in csv.DictReader(handle)])
        axes[1].plot(np.arange(100, len(rewards)+1), np.convolve(rewards, np.ones(100)/100, "valid"),
                     label=f"Seed {seed}")
    for ax, title in zip(axes, ("Equal-budget learning-rate pilots", "Fresh training seeds")):
        ax.set(xlabel="Training episode", ylabel="100-episode mean raw reward", title=title)
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "sensitivity_and_seeds.png", dpi=300)
    plt.close(fig)
    print(f"Selected LR={selected}; trained-random={difference:.2f}; CI={summary['difference_ci95_normal']}")


if __name__ == "__main__":
    main()
