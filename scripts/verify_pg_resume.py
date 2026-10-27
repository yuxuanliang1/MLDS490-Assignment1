"""Compare uninterrupted and resumed real training checkpoints quantitatively."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy_gradient_part1.runtime import load_checkpoint, write_json


def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference, resumed = load_checkpoint(args.reference), load_checkpoint(args.resumed)
    checks = {key+"_identical": bool(same(reference[key], resumed[key]))
              for key in ("model", "optimizer", "rng", "completed_episodes", "config")}
    columns = ("episode", "reset_seed", "reward", "discounted_return", "steps", "terminated", "truncated",
               "loss_contribution", "entropy", "probability_action0", "action0_fraction", "gradient_norm_before_clip")
    checks["numerical_histories_identical"] = all(
        all(a[k] == b[k] for k in columns) for a, b in zip(reference["history"], resumed["history"]))
    max_error = max(float((reference["model"][key]-resumed["model"][key]).abs().max())
                    for key in reference["model"])
    result = dict(checks=checks, all_passed=all(checks.values()), max_parameter_error=max_error,
                  reference=str(args.reference), resumed=str(args.resumed),
                  scope="Same CPU/software; episode/update boundaries; time-limited smoke games")
    write_json(args.output, result)
    print(result)
    if not result["all_passed"]:
        raise SystemExit(1)
