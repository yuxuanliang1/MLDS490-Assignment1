"""A predictable scalar: update only after the current policy update finishes."""
import math
import numpy as np


def validate_trajectory(trajectory):
    if trajectory is None or not len(trajectory.rewards):
        raise ValueError("Missing or empty trajectory")
    n = len(trajectory.rewards)
    if any(len(getattr(trajectory, field)) != n for field in
           ("observations", "actions", "log_probs", "entropies", "probabilities")):
        raise ValueError("Mismatched trajectory fields")
    if not trajectory.terminated and not trajectory.truncated:
        raise ValueError("Incomplete trajectory: neither terminated nor time-limit truncated")
    if not np.isfinite(trajectory.rewards).all():
        raise ValueError("Nonfinite rewards")


class RunningConstantBaseline:
    def __init__(self, alpha=0.05):
        if not math.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0,1]")
        self.alpha = float(alpha)
        self.value = 0.0
        self.count = 0

    def estimate(self):
        return self.value

    def update(self, completed_return):
        value = float(completed_return)
        if not math.isfinite(value):
            raise ValueError("Completed discounted return must be finite")
        self.value = value if self.count == 0 else (1-self.alpha)*self.value + self.alpha*value
        self.count += 1

    def protocol(self):
        return dict(kind="ema_previous_discounted_G0", alpha=self.alpha,
                    statistic="G_0 = sum_t gamma^t r_t, one scalar per completed episode",
                    initialization="Zero for the first batch; first observed G_0 initializes EMA",
                    timing="Frozen across the entire update batch; sequential EMA updates only AFTER Adam.step()",
                    no_current_batch_leakage=True)

    def state_dict(self):
        return dict(kind="ema_previous_discounted_G0", alpha=self.alpha, value=self.value, count=self.count)

    def load_state_dict(self, state):
        if state.get("kind") != "ema_previous_discounted_G0" or state.get("alpha") != self.alpha:
            raise ValueError("Incompatible baseline protocol/alpha")
        value, count = state.get("value"), state.get("count")
        if (not isinstance(count, int) or isinstance(count, bool) or count < 0
                or not isinstance(value, (int, float)) or not math.isfinite(value)
                or (count == 0 and value != 0)):
            raise ValueError("Invalid baseline state")
        self.value, self.count = float(value), count
