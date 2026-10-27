"""Evaluate a saved checkpoint separately from training."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_gradient_part1.evaluator import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1000000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=["stochastic", "greedy"], default="stochastic")
    parser.add_argument("--role", choices=["final", "diagnostic"], default="final")
    parser.add_argument("--reference", choices=["random", "fixed"])
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.output, args.episodes, args.seed, args.device, args.mode,
             args.role, args.reference, args.figure)
