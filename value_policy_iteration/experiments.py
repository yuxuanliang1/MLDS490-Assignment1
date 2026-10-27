"""Run and export experiments A--I and explicit modeling edge cases."""

import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
import numpy as np
import scipy

from .algorithms import finite_horizon, policy_iteration, value_iteration
from .mdp import MDPParameters, ShuttleMDP
from .validation import (compare_infinite, exhaustive_small_mdp, horizon_validation,
                         monte_carlo_validation, policy_structure, reference_q)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def solution_summary(solution) -> dict:
    return {"iterations": solution.iterations, "runtime_seconds": solution.runtime_seconds,
            "bellman_residual": solution.bellman_residual,
            "structure": policy_structure(solution.policy),
            "value_at_0": float(solution.values[0]),
            "value_at_200": float(solution.values[-1])}


def parameter_experiments(parameters: MDPParameters):
    results, arrays = {}, {}
    for group, settings in {
        "cost": [("baseline", {}), ("zero_dispatch", {"dispatch_cost": 0}),
                 ("zero_holding", {"holding_cost": 0}),
                 ("high_dispatch", {"dispatch_cost": 1000}),
                 ("high_holding", {"holding_cost": 10})],
        "discount": [(str(gamma), {"gamma": gamma}) for gamma in (0.5, 0.8, 0.95, 0.99)],
        "deterministic": [(str(arrival), {"arrivals": (arrival,), "probabilities": (1.0,)})
                          for arrival in (1, 3, 5)],
    }.items():
        rows = []
        arrays[group] = {}
        for name, changes in settings:
            p = replace(parameters, **changes)
            mdp = ShuttleMDP(p)
            vi, pi = value_iteration(mdp), policy_iteration(mdp)
            rows.append({"scenario": name, "dispatch_cost": p.dispatch_cost,
                         "holding_cost": p.holding_cost, "gamma": p.gamma,
                         "vi_iterations": vi.iterations, "pi_iterations": pi.iterations,
                         "threshold": policy_structure(pi.policy)["threshold"],
                         "switch_states": policy_structure(pi.policy)["switch_states"],
                         "dispatch_state_count": int(pi.policy.sum()),
                         "value_at_0": float(pi.values[0]), "value_at_200": float(pi.values[-1]),
                         "vi_pi_max_difference": float(np.max(np.abs(vi.values - pi.values))),
                         "vi_residual": vi.bellman_residual, "pi_residual": pi.bellman_residual,
                         "certified": bool(np.max(np.abs(vi.values - pi.values)) < 2e-8
                                           and np.array_equal(vi.policy, pi.policy)
                                           and vi.bellman_residual < 2e-10
                                           and pi.bellman_residual < 1e-9)})
            arrays[group][name] = {"values": pi.values, "policy": pi.policy}
        results[group] = rows
    # Ten steps in deterministic A=3; start at 15 to expose both dispatch and wait.
    trajectory = []
    scenario = arrays["deterministic"]["3"]
    state = 15
    for t in range(10):
        action = int(scenario["policy"][state])
        remaining = max(0, state - parameters.capacity * action)
        cost = parameters.dispatch_cost * action + parameters.holding_cost * remaining
        next_state = min(parameters.max_state, remaining + 3)
        trajectory.append({"time": t, "state": state, "action": action,
                           "immediate_cost": cost, "arrivals": 3, "next_state": next_state,
                           "current_state_value": float(scenario["values"][state]),
                           "bellman_equation_error": float(abs(scenario["values"][state]
                               - cost - parameters.gamma * scenario["values"][next_state]))})
        state = next_state
    results["trajectory"] = trajectory
    return results, arrays


def edge_experiments(mdp, finite, pi, parameter_results):
    p = mdp.parameters
    q = reference_q(p, pi.values)
    rows = []

    def add(case, expectation, observation, implementation, passed):
        rows.append({"case": case, "theoretical_expectation": expectation,
                     "observed": observation, "implementation": implementation,
                     "passed": bool(passed)})

    add("empty_state", "Waiting strictly dominates empty dispatch by cf=100.",
        f"action={pi.policy[0]}; dispatch-minus-wait={q[1, 0] - q[0, 0]:.12g}",
        "Both actions are allowed; they have identical successors at state zero.",
        pi.policy[0] == 0 and abs(q[1, 0] - q[0, 0] - p.dispatch_cost) < 1e-10)
    add("below_capacity", "Dispatch is permitted; optimality depends on future costs.",
        "States 1..14 actions=" + str(pi.policy[1:15].tolist()),
        "max(0,s-K) allows partial loads; full action gaps are in state_results.csv.",
        np.all(pi.policy[1:13] == 0) and np.all(pi.policy[13:15] == 1))
    for state in (15, 16, 50, 200):
        successors = mdp.next_states[1, state].tolist()
        add(f"state_{state}", "At most K=15 passengers depart; holding cost uses the remainder.",
            f"action={pi.policy[state]}; dispatch remainder={state - 15}; "
            f"dispatch cost={mdp.costs[1, state]}; successors={successors}",
            "One shuttle per epoch, including above-capacity states.",
            np.all(mdp.next_states[1, state] == np.arange(state - 14, state - 9))
            and pi.policy[state] == 1)
    add("arrival_overflow", "At s=198, waiting gives P(199)=0.2 and P(200)=0.8.",
        f"P(199)={mdp.transitions[0, 198, 199]}; P(200)={mdp.transitions[0, 198, 200]}; "
        f"P(200|200,wait)={mdp.transitions[0, 200, 200]}",
        "Accumulate probability after min(S_max,remainder+arrival); excess arrivals disappear.",
        np.isclose(mdp.transitions[0, 198, 199], 0.2)
        and np.isclose(mdp.transitions[0, 198, 200], 0.8)
        and mdp.transitions[0, 200, 200] == 1)
    cost_rows = {row["scenario"]: row for row in parameter_results["cost"]}
    for name, expectation, passed in (
        ("zero_dispatch", "Free clearing weakly improves costs; dispatch at positive states, wait at zero tie.",
         cost_rows["zero_dispatch"]["threshold"] == 1),
        ("zero_holding", "Never dispatch yields V=0; every dispatch adds positive cost.",
         cost_rows["zero_holding"]["dispatch_state_count"] == 0
         and cost_rows["zero_holding"]["value_at_200"] == 0),
        ("high_dispatch", "cf=1000 exceeds the maximum K*ch/(1-gamma)=600 holding savings per dispatch.",
         cost_rows["high_dispatch"]["dispatch_state_count"] == 0),
    ):
        row = cost_rows[name]
        add(name, expectation,
            f"threshold={row['threshold']}; dispatch states={row['dispatch_state_count']}; V(0)={row['value_at_0']:.12g}",
            "Same algorithms with changed cost parameters.", passed)
    add("last_epoch", "Immediate holding savings <=30<100, so wait at all states in the last epoch.",
        f"Last-epoch dispatch count={finite.policies[-1].sum()}; max |V_last-2s|="
        f"{np.max(np.abs(finite.values[-2] - 2 * mdp.states))}",
        "Zero terminal values; backward induction considers no post-horizon costs.",
        np.all(finite.policies[-1] == 0)
        and np.array_equal(finite.values[-2], 2 * mdp.states))
    tie_model = ShuttleMDP(replace(p, gamma=0, dispatch_cost=30))
    tied = policy_iteration(tie_model, initial_policy=np.ones(mdp.n_states, dtype=np.int8))
    perturbed = policy_iteration(ShuttleMDP(replace(p, gamma=0, dispatch_cost=30 - 1e-9)))
    zero_cost = policy_iteration(ShuttleMDP(replace(p, dispatch_cost=0, holding_cost=0)),
                                 initial_policy=np.ones(mdp.n_states, dtype=np.int8))
    add("numerical_ties", "At gamma=0,cf=30, all s>=15 tie; a 1e-9 improvement must still be taken.",
        f"Exact-tie dispatches={tied.policy[15:].sum()}; perturbed dispatches="
        f"{perturbed.policy[15:].sum()}; all-zero-cost dispatches={zero_cost.policy.sum()}",
        "8-epsilon relative tie band; stable policy must satisfy the untoleranced Bellman residual.",
        np.all(tied.policy == 0) and np.all(perturbed.policy[15:] == 1)
        and np.all(zero_cost.policy == 0))
    tolerances = []
    for tolerance in (1e-6, 1e-8, 1e-10, 1e-12):
        vi = value_iteration(mdp, tolerance=tolerance)
        tolerances.append({"tolerance": tolerance, "iterations": vi.iterations,
                           "bellman_residual": vi.bellman_residual,
                           "max_error_vs_pi": float(np.max(np.abs(vi.values - pi.values))),
                           "policy_disagreements": int(np.count_nonzero(vi.policy != pi.policy)),
                           "residual_error_bound": vi.bellman_residual / (1 - mdp.gamma)})
    add("tolerance_sensitivity", "Tighter tolerances reduce value error; residual/(1-gamma) bounds error.",
        str([{key: row[key] for key in ("tolerance", "iterations", "max_error_vs_pi")}
             for row in tolerances]),
        "Stop by successive sup difference; compute a fresh residual after stopping.",
        all(row["policy_disagreements"] == 0 for row in tolerances)
        and all(row["max_error_vs_pi"] <= row["residual_error_bound"] + 1e-9 for row in tolerances))
    add("discount_sensitivity", "Larger gamma weakens the contraction and increases VI effort.",
        str([(row["gamma"], row["vi_iterations"], row["threshold"])
             for row in parameter_results["discount"]]),
        "Discount is a model parameter; each scenario is solved independently by VI and PI.",
        all(row["certified"] for row in parameter_results["discount"]))
    return rows, tolerances


def run_experiments(root: Path, episodes: int = 20000, simulation_horizon: int = 400,
                    seed: int = 20261026):
    results_dir = root / "results" / "value_policy_iteration"
    results_dir.mkdir(parents=True, exist_ok=True)
    mdp = ShuttleMDP()
    finite = finite_horizon(mdp, 500)
    vi, pi = value_iteration(mdp), policy_iteration(mdp)
    comparison = compare_infinite(mdp, vi, pi)
    horizons = horizon_validation(mdp, pi)
    exhaustive, exhaustive_array = exhaustive_small_mdp()
    parameter_results, parameter_arrays = parameter_experiments(mdp.parameters)
    edges, tolerances = edge_experiments(mdp, finite, pi, parameter_results)
    monte_carlo = monte_carlo_validation(mdp, vi.policy, episodes, simulation_horizon, seed)
    q = reference_q(mdp.parameters, pi.values)
    checks = dict(comparison["checks"])
    checks.update({
        "vi_contraction_history": bool(np.all(vi.history[1:] <= mdp.gamma * vi.history[:-1]
                                             + 64 * np.finfo(float).eps * np.max(np.abs(vi.values)))),
        "finite_values_monotone_in_remaining_horizon": bool(
            np.all(finite.values[:-1] >= finite.values[1:] - 1e-10)),
        "finite_horizon_bounds": all(row["bound_satisfied"] for row in horizons),
        "finite_horizon_errors_decrease": all(
            b["sup_error"] <= a["sup_error"] + 1e-9 for a, b in zip(horizons, horizons[1:])),
        "exhaustive_32_policies": exhaustive["passed"],
        "parameter_scenarios_certified": all(row["certified"] for group in ("cost", "discount", "deterministic")
                                             for row in parameter_results[group]),
        "edge_cases": all(row["passed"] for row in edges),
        "deterministic_trajectory_bellman_equation": all(
            row["bellman_equation_error"] < 1e-9 for row in parameter_results["trajectory"]),
        "optimal_policy_exact_dominance": all(monte_carlo["optimal_dominates_all_states"].values()),
        "monte_carlo_familywise_consistency": monte_carlo["all_familywise_consistent"],
        "finite_terminal_values_zero": bool(np.all(finite.values[-1] == 0)),
    })
    source_files = sorted((root / "value_policy_iteration").glob("*.py")) + [root / "scripts" / "run_value_policy.py"]
    summary = {
        "metadata": {"generated_utc": datetime.now(timezone.utc).isoformat(),
                     "python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
                     "seed": seed, "episodes": episodes, "simulation_horizon": simulation_horizon,
                     "source_sha256": {str(path.relative_to(root)).replace('\\', '/'):
                                       hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}},
        "parameters": asdict(mdp.parameters), "max_one_step_cost": mdp.max_cost,
        "finite_horizon": {"horizon": 500, "runtime_seconds": finite.runtime_seconds,
                           "value_at_0": float(finite.values[0, 0])},
        "value_iteration": solution_summary(vi), "policy_iteration": solution_summary(pi),
        "comparison": comparison, "horizon_validation": horizons,
        "exhaustive": exhaustive, "parameter_experiments": parameter_results,
        "edge_cases": edges, "tolerance_sensitivity": tolerances,
        "monte_carlo": monte_carlo,
        "action_gap": {"minimum": float((q[1] - q[0]).min()),
                       "maximum": float((q[1] - q[0]).max()),
                       "near_tie_states_at_1e_8": np.flatnonzero(np.abs(q[1] - q[0]) <= 1e-8).tolist()},
        "checks": checks, "all_checks_passed": all(checks.values()),
    }
    with (results_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    write_csv(results_dir / "state_results.csv", [
        {"state": int(s), "wait_cost": float(mdp.costs[0, s]),
         "dispatch_cost": float(mdp.costs[1, s]), "finite_value": float(finite.values[0, s]),
         "finite_policy_t0": int(finite.policies[0, s]), "vi_value": float(vi.values[s]),
         "pi_value": float(pi.values[s]), "vi_policy": int(vi.policy[s]), "pi_policy": int(pi.policy[s]),
         "q_wait": float(q[0, s]), "q_dispatch": float(q[1, s]),
         "dispatch_minus_wait": float(q[1, s] - q[0, s])} for s in mdp.states])
    for filename, rows in {
        "horizon_validation.csv": horizons, "cost_sensitivity.csv": parameter_results["cost"],
        "discount_sensitivity.csv": parameter_results["discount"],
        "deterministic_arrivals.csv": parameter_results["deterministic"],
        "deterministic_trajectory.csv": parameter_results["trajectory"],
        "edge_cases.csv": edges, "tolerance_sensitivity.csv": tolerances,
        "monte_carlo.csv": monte_carlo["rows"],
        "vi_convergence.csv": [{"iteration": i + 1, "successive_sup_difference": float(x)}
                               for i, x in enumerate(vi.history)],
        "pi_convergence.csv": [{"evaluation": i + 1, "optimality_residual": float(x)}
                               for i, x in enumerate(pi.history)],
    }.items():
        write_csv(results_dir / filename, rows)
    np.savetxt(results_dir / "exhaustive_policies.csv", exhaustive_array, delimiter=",",
               header=",".join([f"action_s{s}" for s in range(5)] + [f"value_s{s}" for s in range(5)]),
               comments="", fmt=["%d"] * 5 + ["%.15g"] * 5)
    np.savez_compressed(results_dir / "solutions.npz", finite_values=finite.values,
                        finite_policies=finite.policies, vi_values=vi.values, vi_policy=vi.policy,
                        pi_values=pi.values, pi_policy=pi.policy, vi_history=vi.history,
                        pi_history=pi.history, transitions=mdp.transitions, costs=mdp.costs)
    sensitivity_arrays = {f"{group}_{name}_{kind}": data for group, scenarios in parameter_arrays.items()
                          for name, arrays in scenarios.items() for kind, data in arrays.items()}
    np.savez_compressed(results_dir / "sensitivity_solutions.npz", **sensitivity_arrays)
    return mdp, finite, vi, pi, summary, parameter_arrays
