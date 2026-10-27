"""Independent numerical certificates, exhaustive policies, and simulation.

The reference transition builder and Bellman calculation below deliberately use
scalar definitions rather than the production model's transition/cost arrays.
"""

from itertools import product

import numpy as np
from scipy.linalg import solve
from scipy.stats import norm

from .algorithms import evaluate_policy, finite_horizon, policy_iteration, value_iteration
from .mdp import MDPParameters, ShuttleMDP


def reference_model(p: MDPParameters) -> tuple[np.ndarray, np.ndarray]:
    n = p.max_state + 1
    transition = np.zeros((2, n, n))
    costs = np.zeros((2, n))
    for action in (0, 1):
        for state in range(n):
            remaining = max(0, state - action * p.capacity)
            costs[action, state] = p.dispatch_cost * action + p.holding_cost * remaining
            for arrival, probability in zip(p.arrivals, p.probabilities):
                successor = min(p.max_state, remaining + arrival)
                transition[action, state, successor] += probability
    return transition, costs


def reference_q(p: MDPParameters, values: np.ndarray) -> np.ndarray:
    q = np.empty((2, p.max_state + 1))
    for action in (0, 1):
        for state in range(p.max_state + 1):
            remaining = max(0, state - action * p.capacity)
            expected = sum(probability * values[min(p.max_state, remaining + arrival)]
                           for arrival, probability in zip(p.arrivals, p.probabilities))
            q[action, state] = (p.dispatch_cost * action + p.holding_cost * remaining
                                + p.gamma * expected)
    return q


def policy_structure(policy: np.ndarray) -> dict:
    switches = (np.flatnonzero(np.diff(policy) != 0) + 1).tolist()
    dispatch_states = np.flatnonzero(policy == 1)
    threshold = int(dispatch_states[0]) if len(dispatch_states) else None
    is_threshold = bool(threshold is not None and np.all(policy[threshold:] == 1)
                        and np.all(policy[:threshold] == 0))
    return {"is_threshold": is_threshold, "threshold": threshold if is_threshold else None,
            "switch_states": switches, "dispatch_state_count": int(policy.sum())}


def compare_infinite(mdp, vi, pi) -> dict:
    difference = np.abs(vi.values - pi.values)
    q = reference_q(mdp.parameters, pi.values)
    gap = q[1] - q[0]
    differing = np.flatnonzero(vi.policy != pi.policy)
    fixed = evaluate_policy(mdp, vi.policy)
    transition, costs = reference_model(mdp.parameters)
    states = np.arange(mdp.n_states)
    fixed_residual = float(np.max(np.abs(
        fixed - costs[vi.policy, states]
        - mdp.gamma * (transition[vi.policy, states] @ fixed))))
    residual_vi = float(np.max(np.abs(vi.values - reference_q(mdp.parameters, vi.values).min(axis=0))))
    residual_pi = float(np.max(np.abs(pi.values - q.min(axis=0))))
    checks = {
        "independent_transitions_agree": bool(np.allclose(transition, mdp.transitions, atol=1e-14, rtol=0)),
        "independent_costs_agree": bool(np.array_equal(costs, mdp.costs)),
        "vi_pi_values_agree": bool(difference.max() <= 1e-7),
        "vi_pi_policies_agree": bool(len(differing) == 0),
        "vi_residual_certified": bool(residual_vi < 2e-10),
        "pi_residual_certified": bool(residual_pi < 1e-9),
        "fixed_policy_equation_certified": bool(fixed_residual < 1e-9),
    }
    return {
        "max_value_difference": float(difference.max()),
        "mean_value_difference": float(difference.mean()),
        "policy_disagreements": len(differing), "differing_states": differing.tolist(),
        "disagreement_absolute_gaps": np.abs(gap[differing]).tolist(),
        "vi_independent_residual": residual_vi, "pi_independent_residual": residual_pi,
        "fixed_policy_residual": fixed_residual,
        "vi_residual_error_bound": residual_vi / (1 - mdp.gamma),
        "pi_residual_error_bound": residual_pi / (1 - mdp.gamma),
        "minimum_absolute_action_gap": float(np.min(np.abs(gap))),
        "nearest_tie_state": int(np.argmin(np.abs(gap))), "checks": checks,
    }


def horizon_validation(mdp: ShuttleMDP, pi) -> list[dict]:
    rows = []
    reference_error = pi.bellman_residual / (1 - mdp.gamma)
    for horizon in (5, 10, 25, 50, 100, 500):
        result = finite_horizon(mdp, horizon)
        error = float(np.max(np.abs(pi.values - result.values[0])))
        bound = mdp.gamma ** horizon * mdp.max_cost / (1 - mdp.gamma)
        # Policy-evaluation error and accumulation roundoff are reported separately.
        allowance = reference_error + 128 * np.finfo(float).eps * max(1, np.max(np.abs(pi.values))) * horizon
        rows.append({"horizon": horizon, "sup_error": error, "tail_bound": bound,
                     "numerical_allowance": float(allowance),
                     "bound_satisfied": bool(error <= bound + allowance)})
    return rows


def exhaustive_small_mdp() -> tuple[dict, np.ndarray]:
    p = MDPParameters(capacity=2, max_state=4, arrivals=(0, 1),
                      probabilities=(0.5, 0.5), dispatch_cost=3, holding_cost=1, gamma=0.8)
    transition, cost = reference_model(p)
    states = np.arange(5)
    policies = np.asarray(list(product((0, 1), repeat=5)), dtype=np.int8)
    # No VI/PI loops or production policy evaluator are used for the enumeration.
    values = np.asarray([solve(np.eye(5) - p.gamma * transition[policy, states],
                               cost[policy, states]) for policy in policies])
    optimum = values.min(axis=0)
    global_indices = np.flatnonzero(np.max(np.abs(values - optimum), axis=1) < 1e-10)
    mdp = ShuttleMDP(p)
    vi, pi = value_iteration(mdp), policy_iteration(mdp)
    vi_error = float(np.max(np.abs(vi.values - optimum)))
    pi_error = float(np.max(np.abs(pi.values - optimum)))
    # Independently evaluate the policies returned by the solvers too.
    vi_policy_index = int(np.flatnonzero(np.all(policies == vi.policy, axis=1))[0])
    pi_policy_index = int(np.flatnonzero(np.all(policies == pi.policy, axis=1))[0])
    return {
        "parameters": {"max_state": 4, "capacity": 2, "dispatch_cost": 3,
                       "holding_cost": 1, "gamma": 0.8, "arrivals": [0, 1]},
        "policies_enumerated": len(policies), "optimal_values": optimum.tolist(),
        "globally_optimal_policies": policies[global_indices].tolist(),
        "vi_max_error": vi_error, "pi_max_error": pi_error,
        "vi_policy": vi.policy.tolist(), "pi_policy": pi.policy.tolist(),
        "passed": bool(len(global_indices) and vi_error < 1e-8 and pi_error < 1e-10
                       and vi_policy_index in global_indices and pi_policy_index in global_indices),
    }, np.column_stack((policies, values))


def simulate_policy(mdp: ShuttleMDP, policy: np.ndarray, initial_state: int,
                    episodes: int, horizon: int, seed: int, exact: np.ndarray) -> dict:
    """Primitive scalar conventions, vectorized across independent episodes.

    The exact continuation at the terminal state removes truncation bias.
    Raw truncated estimates and a worst-case omitted-tail bound are also saved.
    Common random numbers across policies allow fair comparisons; trajectories
    within each policy remain independent across episodes.
    """
    p = mdp.parameters
    rng = np.random.default_rng(seed)
    states = np.full(episodes, initial_state, dtype=np.int64)
    totals = np.zeros(episodes)
    discount = 1.0
    for _ in range(horizon):
        actions = policy[states]
        remaining = np.maximum(0, states - p.capacity * actions)
        totals += discount * (p.dispatch_cost * actions + p.holding_cost * remaining)
        arrivals = rng.choice(p.arrivals, size=episodes, p=p.probabilities)
        states = np.minimum(p.max_state, remaining + arrivals)
        discount *= p.gamma
    raw_mean = float(totals.mean())
    totals += discount * exact[states]
    mean = float(totals.mean())
    deterministic = bool(np.ptp(totals) == 0)
    sd = 0.0 if deterministic else float(totals.std(ddof=1))
    se = sd / np.sqrt(episodes)
    z = float(norm.ppf(0.975))
    lower, upper = mean - z * se, mean + z * se
    policy_cost_max = float(mdp.costs[policy, mdp.states].max())
    tail_bound = discount * policy_cost_max / (1 - p.gamma)
    # A zero-width statistical interval must not turn solver/accumulation
    # roundoff into an apparent sampling failure (e.g. never dispatch at 200).
    numerical_allowance = float(64 * np.finfo(float).eps * (horizon + 1)
                                * max(1, abs(exact[initial_state])))
    return {"initial_state": initial_state, "episodes": episodes, "horizon": horizon,
            "exact_value": float(exact[initial_state]), "mean": mean,
            "standard_deviation": sd, "standard_error": float(se),
            "ci95_low": float(lower), "ci95_high": float(upper),
            "error": mean - float(exact[initial_state]), "raw_truncated_mean": raw_mean,
            "raw_tail_bound": tail_bound, "mean_terminal_correction": mean - raw_mean,
            "deterministic_return": deterministic,
            "numerical_allowance": numerical_allowance,
            "ci95_contains_exact_strict": bool(lower <= exact[initial_state] <= upper),
            "ci95_contains_exact": bool(lower - numerical_allowance <= exact[initial_state]
                                         <= upper + numerical_allowance),
            "seed": seed}


def monte_carlo_validation(mdp, optimal, episodes=20000, horizon=400, seed=20261026):
    policies = {
        "optimal": optimal,
        "never_dispatch": np.zeros(mdp.n_states, dtype=np.int8),
        "always_dispatch": np.ones(mdp.n_states, dtype=np.int8),
        "threshold_15": (mdp.states >= 15).astype(np.int8),
    }
    exact_values = {name: evaluate_policy(mdp, policy) for name, policy in policies.items()}
    optimal_values = exact_values["optimal"]
    dominance = {name: bool(np.all(optimal_values <= values + 1e-8))
                 for name, values in exact_values.items()}
    rows = []
    for name, policy in policies.items():
        for state in (0, 15, 50, 100, 200):
            row = simulate_policy(mdp, policy, state, episodes, horizon, seed, exact_values[name])
            row["policy"] = name
            rows.append(row)
    # Individual 95% intervals are diagnostics, not a demand for perfect coverage.
    # A familywise 5% Bonferroni normal check handles all 20 comparisons.
    z_family = float(norm.ppf(1 - 0.05 / (2 * len(rows))))
    for row in rows:
        row["familywise_consistent"] = bool(
            abs(row["error"]) <= z_family * row["standard_error"] + row["numerical_allowance"])
    return {
        "rows": rows, "optimal_dominates_all_states": dominance,
        "individual_ci_coverage_count": sum(row["ci95_contains_exact"] for row in rows),
        "nondegenerate_comparisons": sum(not row["deterministic_return"] for row in rows),
        "nondegenerate_ci_coverage_count": sum(row["ci95_contains_exact_strict"]
                                               for row in rows if not row["deterministic_return"]),
        "deterministic_cases": sum(row["deterministic_return"] for row in rows),
        "comparisons": len(rows), "familywise_z": z_family,
        "all_familywise_consistent": all(row["familywise_consistent"] for row in rows),
        "tail_method": "exact policy continuation at terminal state; raw tail bounds also reported",
    }
