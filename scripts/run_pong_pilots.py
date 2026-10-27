"""Affordable local CPU pilots; never substitutes for formal Pong training."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy_gradient_part1.config import Config
from policy_gradient_part1.trainer import train
from policy_gradient_part1.evaluator import evaluate
from policy_gradient_part1.runtime import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--reference-episodes", type=int, default=30)
    parser.add_argument("--output", type=Path, default=ROOT / "results/policy_gradient_part1/pong")
    args = parser.parse_args()
    base = Config.load(ROOT / "configs/pong_part1.json")
    protocol = dict(stage="CPU pilots only", episodes=args.episodes, seed=101,
                    learning_rates=[0.01, 0.003, 0.001], representations=["difference", "raw"],
                    reference_episodes=args.reference_episodes,
                    diagnostic_checkpoint="difference_lr_0.003/latest.pt, specified before results",
                    selection="No formal configuration selected from this very small pilot budget")
    write_json(args.output / "pilot_protocol.json", protocol)
    results = []
    for difference in (True, False):
        for lr in protocol["learning_rates"]:
            representation = "difference" if difference else "raw"
            directory = args.output / "pilots" / f"{representation}_lr_{lr:g}"
            config = replace(base, frame_difference=difference, learning_rate=lr, seed=101,
                             episodes=args.episodes, device="cpu")
            results.append(dict(representation=representation, learning_rate=lr,
                                **train(config, directory)))
    checkpoint = args.output / "pilots/difference_lr_0.003/latest.pt"
    references = {}
    for name, reference in (("pilot_policy", None), ("random", "random"), ("fixed", "fixed")):
        references[name] = evaluate(checkpoint, args.output / "diagnostics" / name,
                                    episodes=args.reference_episodes, seed=4000000,
                                    role="diagnostic", reference=reference)
    summary = dict(protocol=protocol, pilots=results, references=references,
                   formal_training_completed=False, final_500_episode_evaluation_completed=False,
                   limitation="Two updates per pilot at default budget: stability/runtime evidence only; "
                   "no claim of convergence or representation superiority")
    write_json(args.output / "pilot_summary.json", summary)


if __name__ == "__main__":
    main()
