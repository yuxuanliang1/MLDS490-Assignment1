"""Manual backward induction, value iteration, and Howard policy iteration."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.linalg import solve

from .mdp import ShuttleMDP


@dataclass
class FiniteHorizonResult:
    values: np.ndarray  # values[t, state], including the zero terminal row
    policies: np.ndarray  # policies[t, state]
    runtime_seconds: float


@dataclass
class InfiniteHorizonResult:
    values: np.ndarray
    policy: np.ndarray
    iterations: int
    history: np.ndarray
    bellman_residual: float
    runtime_seconds: float


def greedy_policy(action_values: np.ndarray) -> np.ndarray:
    """Prefer wait only within an 8-epsilon floating-point uncertainty band.

    A fixed large tie tolerance could suppress genuine improvements. The band
    scales with Q, and policy iteration separately certifies optimality against
    the *unmodified* minimum of the action values.
    """
    scale = np.maximum(1.0, np.max(np.abs(action_values), axis=0))
    band = 8 * np.finfo(float).eps * scale
    return (action_values[1] < action_values[0] - band).astype(np.int8)


def bellman_residual(mdp: ShuttleMDP, values: np.ndarray) -> float:
    return float(np.max(np.abs(values - mdp.action_values(values).min(axis=0))))


def evaluate_policy(mdp: ShuttleMDP, policy: np.ndarray) -> np.ndarray:
    transition, cost = mdp.policy_arrays(policy)
    return solve(np.eye(mdp.n_states) - mdp.gamma * transition, cost,
                 assume_a="gen", check_finite=True)


def finite_horizon(mdp: ShuttleMDP, horizon: int = 500) -> FiniteHorizonResult:
    if not isinstance(horizon, (int, np.integer)) or horizon < 0:
        raise ValueError("horizon must be a nonnegative integer")
    start = perf_counter()
    values = np.zeros((horizon + 1, mdp.n_states))
    policies = np.empty((horizon, mdp.n_states), dtype=np.int8)
    for t in range(horizon - 1, -1, -1):
        q = mdp.action_values(values[t + 1])
        values[t] = q.min(axis=0)
        policies[t] = greedy_policy(q)
    return FiniteHorizonResult(values, policies, perf_counter() - start)


def value_iteration(mdp: ShuttleMDP, tolerance: float = 1e-10,
                    max_iterations: int = 100000) -> InfiniteHorizonResult:
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    start = perf_counter()
    values = np.zeros(mdp.n_states)
    history = []
    for iteration in range(1, max_iterations + 1):
        updated = mdp.action_values(values).min(axis=0)
        difference = float(np.max(np.abs(updated - values)))
        history.append(difference)
        values = updated
        if difference <= tolerance:
            break
    else:
        raise RuntimeError("value iteration did not reach its requested tolerance")
    return InfiniteHorizonResult(
        values, greedy_policy(mdp.action_values(values)), iteration, np.asarray(history),
        bellman_residual(mdp, values), perf_counter() - start,
    )


def policy_iteration(mdp: ShuttleMDP, initial_policy: np.ndarray | None = None,
                     residual_tolerance: float = 1e-9,
                     max_iterations: int = 1000) -> InfiniteHorizonResult:
    if not np.isfinite(residual_tolerance) or residual_tolerance <= 0:
        raise ValueError("residual_tolerance must be finite and positive")
    start = perf_counter()
    policy = (np.zeros(mdp.n_states, dtype=np.int8) if initial_policy is None
              else np.asarray(initial_policy).copy())
    mdp.policy_arrays(policy)  # validate before indexing
    policy = policy.astype(np.int8)
    history = []
    for iteration in range(1, max_iterations + 1):
        values = evaluate_policy(mdp, policy)
        q = mdp.action_values(values)
        residual = float(np.max(np.abs(values - q.min(axis=0))))
        history.append(residual)
        improved = greedy_policy(q)
        if np.array_equal(improved, policy):
            if residual <= residual_tolerance:
                break
            # A tie band can never justify terminating with a failed certificate.
            improved = q.argmin(axis=0).astype(np.int8)
            if np.array_equal(improved, policy):
                raise RuntimeError("policy evaluation precision cannot certify optimality")
        policy = improved
    else:
        raise RuntimeError("policy iteration did not stabilize")
    return InfiniteHorizonResult(values, policy, iteration, np.asarray(history),
                                 residual, perf_counter() - start)
