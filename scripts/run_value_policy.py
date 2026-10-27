"""Reproduce Component 1 from any working directory (no GPU dependencies)."""

import argparse
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from value_policy_iteration.experiments import run_experiments
from value_policy_iteration.plots import generate_plots
from value_policy_iteration.reporting import write_report_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20000, help="Monte Carlo episodes per policy/state")
    parser.add_argument("--simulation-horizon", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args()
    if args.episodes < 2 or args.simulation_horizon < 1:
        parser.error("episodes must be >=2 and simulation-horizon must be >=1")
    start = perf_counter()
    print(f"Running Component 1, including {args.episodes:,} episodes per policy/state...", flush=True)
    mdp, finite, vi, pi, summary, parameter_arrays = run_experiments(
        ROOT, args.episodes, args.simulation_horizon, args.seed)
    generate_plots(ROOT, mdp, finite, vi, pi, summary, parameter_arrays)
    write_report_results(ROOT, summary)
    comparison = summary["comparison"]
    print(f"VI: {vi.iterations} iterations, residual {vi.bellman_residual:.3e}")
    print(f"PI: {pi.iterations} policy evaluations/improvement passes, residual {pi.bellman_residual:.3e}")
    print(f"VI/PI max value difference: {comparison['max_value_difference']:.3e}; "
          f"policy disagreements: {comparison['policy_disagreements']}")
    print(f"Optimal policy: {summary['policy_iteration']['structure']}")
    print(f"Horizon 500 error: {summary['horizon_validation'][-1]['sup_error']:.3e}; "
          f"tail bound: {summary['horizon_validation'][-1]['tail_bound']:.3e}")
    print(f"Exhaustive policies verified: {summary['exhaustive']['policies_enumerated']}")
    mc = summary["monte_carlo"]
    print(f"Monte Carlo 95% CI consistency (allowing roundoff): "
          f"{mc['individual_ci_coverage_count']}/{mc['comparisons']}; "
          f"familywise consistency: {mc['all_familywise_consistent']}")
    print(f"Validation: {sum(summary['checks'].values())}/{len(summary['checks'])} checks passed")
    print(f"Results: results/value_policy_iteration/; figures: figures/value_policy_iteration/")
    print(f"Completed in {perf_counter() - start:.1f} seconds.")
    if not summary["all_checks_passed"]:
        failed = [name for name, passed in summary["checks"].items() if not passed]
        raise SystemExit("Validation failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
