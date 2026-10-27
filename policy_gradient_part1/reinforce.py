"""Manual Monte Carlo reward-to-go estimator; no centering or baseline."""

import numpy as np
import torch
from torch.distributions import Categorical


def reward_to_go(rewards, gamma):
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must lie in [0,1]")
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 1 or not np.isfinite(rewards).all():
        raise ValueError("rewards must be a finite vector")
    returns = np.empty_like(rewards)
    total = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        total = rewards[t] + gamma * total
        returns[t] = total
    return returns


def policy_loss(logits, actions, returns):
    if not torch.isfinite(logits).all() or not torch.isfinite(returns).all():
        raise FloatingPointError("Nonfinite logits or returns")
    return -(Categorical(logits=logits).log_prob(actions) * returns).sum()


def policy_weights(rewards, gamma, discount_weighting="relative"):
    """G_t for the brief's reference; gamma^t G_t for strict initial-state J_gamma.

    The outer exponent starts at ZERO at each episode, never at each chunk or
    Pong point. This deterministic weighting is neither normalization nor a baseline.
    """
    returns = reward_to_go(rewards, gamma)
    if discount_weighting == "absolute":
        return returns * np.power(float(gamma), np.arange(len(returns), dtype=np.float64))
    if discount_weighting != "relative":
        raise ValueError("Unknown discount weighting")
    return returns


def backward_episode(model, trajectory, gamma, device, chunk_size, episode_divisor, discount_weighting="relative",
                     baseline=0.0):
    """Recompute sampled-action log probabilities under UNCHANGED rollout weights.

    Sum time steps, average episodes (not trajectory lengths). Accumulate chunk
    gradients before ONE optimizer step. No dropout/batchnorm, no importance
    sampling and no repeated updates on a trajectory. Memory is O(chunk graph).
    """
    if not np.isfinite(baseline):
        raise ValueError("Baseline must be a finite action-independent scalar")
    if baseline == 0.0:
        # Preserve the original Part 1 numerical path exactly.
        returns = policy_weights(trajectory.rewards, gamma, discount_weighting)
    else:
        returns = reward_to_go(trajectory.rewards, gamma) - float(baseline)
        if discount_weighting == "absolute":
            returns *= np.power(float(gamma), np.arange(len(returns), dtype=np.float64))
        elif discount_weighting != "relative":
            raise ValueError("Unknown discount weighting")
    loss_sum = 0.0
    max_logprob_error = 0.0
    for start in range(0, len(returns), chunk_size):
        stop = start + chunk_size
        observations = torch.as_tensor(trajectory.observations[start:stop], device=device).float()
        actions = torch.as_tensor(trajectory.actions[start:stop], device=device, dtype=torch.long)
        weights = torch.as_tensor(returns[start:stop], device=device, dtype=torch.float32)
        logits = model(observations)
        logp = Categorical(logits=logits).log_prob(actions)
        sampled = torch.as_tensor(trajectory.log_probs[start:stop], device=device)
        max_logprob_error = max(max_logprob_error, float((logp.detach() - sampled).abs().max()))
        loss = policy_loss(logits, actions, weights) / episode_divisor
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite policy loss")
        loss.backward()
        loss_sum += float(loss.detach())
    return loss_sum, max_logprob_error
