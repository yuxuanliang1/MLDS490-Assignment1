"""The specified dispatch/holding/arrival ordering, including capped arrivals."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MDPParameters:
    capacity: int = 15
    dispatch_cost: float = 100.0
    holding_cost: float = 2.0
    max_state: int = 200
    gamma: float = 0.95
    arrivals: tuple[int, ...] = (1, 2, 3, 4, 5)
    probabilities: tuple[float, ...] = (0.2, 0.2, 0.2, 0.2, 0.2)

    def __post_init__(self):
        for name, minimum in (("capacity", 1), ("max_state", 0)):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not np.isfinite(self.gamma) or not 0 <= self.gamma < 1:
            raise ValueError("gamma must be finite and in [0, 1)")
        if any(not np.isfinite(x) or x < 0
               for x in (self.dispatch_cost, self.holding_cost)):
            raise ValueError("costs must be finite and nonnegative")
        if not self.arrivals or len(self.arrivals) != len(self.probabilities):
            raise ValueError("arrival support and probabilities must be nonempty and aligned")
        if any(not isinstance(x, (int, np.integer)) or x < 0 for x in self.arrivals):
            raise ValueError("arrivals must be nonnegative integers")
        probs = np.asarray(self.probabilities, dtype=float)
        if not np.all(np.isfinite(probs)) or np.any(probs < 0):
            raise ValueError("probabilities must be finite and nonnegative")
        if not np.isclose(probs.sum(), 1, rtol=0, atol=1e-14):
            raise ValueError("arrival probabilities must sum to one")


class ShuttleMDP:
    """Arrays have axes (action, state[, successor or arrival]).

    Both actions are permitted at every state. A single dispatch carries up to
    capacity passengers; holding cost is charged on the remainder, then arrivals
    are added and truncated at max_state. Overflow has no extra penalty.
    """

    def __init__(self, parameters: MDPParameters | None = None):
        self.parameters = parameters or MDPParameters()
        p = self.parameters
        self.states = np.arange(p.max_state + 1)
        self.n_states = len(self.states)
        self.remaining = np.maximum(0, self.states[None, :] - p.capacity * np.arange(2)[:, None])
        self.costs = p.dispatch_cost * np.arange(2)[:, None] + p.holding_cost * self.remaining
        self.probabilities = np.asarray(p.probabilities, dtype=float)
        self.next_states = np.minimum(
            p.max_state, self.remaining[:, :, None] + np.asarray(p.arrivals)[None, None, :]
        )
        self.transitions = np.zeros((2, self.n_states, self.n_states))
        # Repeated successors at the cap (or repeated arrivals) must accumulate.
        for a in range(2):
            for k, probability in enumerate(self.probabilities):
                self.transitions[a, self.states, self.next_states[a, :, k]] += probability

    @property
    def gamma(self) -> float:
        return self.parameters.gamma

    @property
    def max_cost(self) -> float:
        """Valid upper bound for every state/action, not only the chosen policy."""
        return float(self.costs.max())

    def action_values(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.shape != (self.n_states,):
            raise ValueError("one value is required per state")
        # Only a few arrivals: avoid dense matrix multiplication in every backup.
        expected = np.sum(values[self.next_states] * self.probabilities, axis=2)
        return self.costs + self.gamma * expected

    def policy_arrays(self, policy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        policy = np.asarray(policy)
        if policy.shape != (self.n_states,) or np.any((policy != 0) & (policy != 1)):
            raise ValueError("policy must contain one binary action per state")
        policy = policy.astype(np.intp, copy=False)
        return self.transitions[policy, self.states], self.costs[policy, self.states]
