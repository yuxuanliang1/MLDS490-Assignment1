"""Train ALE/Pong-v5 with exactly two learned actions and one device."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_gradient_part1.cli import training_main

if __name__ == "__main__":
    training_main("ALE/Pong-v5")
