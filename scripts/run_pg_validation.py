"""Run numerical REINFORCE validation and optional real Pong startup inspection."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy_gradient_part1.validation import validate_core, inspect_pong, validate_absolute_objective

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/policy_gradient_part1/validation")
    parser.add_argument("--pong", action="store_true")
    parser.add_argument("--absolute", action="store_true")
    args = parser.parse_args()
    validate_core(args.output)
    if args.absolute:
        validate_absolute_objective(args.output)
    if args.pong:
        print(inspect_pong(args.output / "pong"))
