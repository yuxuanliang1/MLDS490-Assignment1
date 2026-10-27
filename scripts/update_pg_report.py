"""Regenerate only the Part 1 LaTeX numerical macros from completed results."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy_gradient_part1.reporting import write_report_results

if __name__ == "__main__":
    write_report_results(ROOT)
