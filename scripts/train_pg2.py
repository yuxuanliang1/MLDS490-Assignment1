"""Matched Part 2 training; reuses Part 1 models, rollouts, optimizer and trainer."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy_gradient_part1.config import Config
from policy_gradient_part1.trainer import train
from policy_gradient_part2.baseline import RunningConstantBaseline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--baseline-alpha", type=float, default=0.05)
    for key, kind in (("episodes", int), ("seed", int), ("device", str), ("max-episode-steps", int)):
        parser.add_argument("--"+key, type=kind)
    args = parser.parse_args()
    config = Config.load(args.config)
    for key in ("episodes", "seed", "device", "max_episode_steps"):
        if getattr(args, key) is not None:
            setattr(config, key, getattr(args, key))
    train(config, args.output, args.resume, args.figure, RunningConstantBaseline(args.baseline_alpha))
