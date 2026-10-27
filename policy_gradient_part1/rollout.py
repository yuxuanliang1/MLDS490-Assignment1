"""Serial sampling with compact CPU storage and no retained autograd graph."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions import Categorical

from .environments import PONG_ACTIONS
from .preprocessing import ObservationTransform
from .reinforce import reward_to_go


@dataclass
class Trajectory:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    log_probs: np.ndarray
    entropies: np.ndarray
    probabilities: np.ndarray
    terminated: bool
    truncated: bool
    initial_reward: float
    trace: list

    @property
    def total_reward(self):
        return float(self.initial_reward + self.rewards.sum())


@torch.no_grad()
def sample_episode(env, model, config, device, seed, mode="stochastic", reference=None, trace_steps=0):
    transform = ObservationTransform(config)
    observation, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    initial_reward = 0.0
    if config.environment == "ALE/Pong-v5" and config.fire_on_reset:
        observation, initial_reward, terminated, truncated, _ = env.step(1)
        if terminated or truncated:
            raise RuntimeError("Episode ended during FIRE initialization")
        if initial_reward != 0:
            raise RuntimeError("Nonzero FIRE reward would change the objective; investigate initialization")
    observations, actions, rewards, log_probs, entropies, probabilities, trace = [], [], [], [], [], [], []
    while True:
        state = transform(observation)
        logits = model(torch.as_tensor(state, device=device).float().unsqueeze(0)).squeeze(0)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite action logits")
        distribution = Categorical(logits=logits)
        if reference == "random":
            action = torch.randint(2, (), device=device)
        elif reference == "fixed":
            action = torch.zeros((), device=device, dtype=torch.long)
        elif mode == "greedy":
            action = logits.argmax()
        else:
            action = distribution.sample()
        index = int(action)
        env_action = PONG_ACTIONS[index] if config.environment == "ALE/Pong-v5" else index
        next_observation, reward, terminated, truncated, _ = env.step(env_action)
        observations.append(state)
        actions.append(index)
        rewards.append(float(reward))
        log_probs.append(float(distribution.log_prob(action)))
        entropies.append(float(distribution.entropy()))
        probabilities.append(distribution.probs.cpu().numpy())
        if len(trace) < trace_steps:
            summary = (dict(values=state.tolist()) if config.environment == "CartPole-v1" else
                       dict(shape=list(state.shape), minimum=int(state.min()), maximum=int(state.max()),
                            nonzero=int(np.count_nonzero(state)), mean=float(state.mean())))
            trace.append(dict(t=len(actions)-1, observation=summary, policy_action=index,
                              environment_action=env_action, log_probability=log_probs[-1], reward=float(reward),
                              terminated=bool(terminated), truncated=bool(truncated)))
        observation = next_observation
        if terminated or truncated:
            break
    returns = reward_to_go(rewards, config.gamma)
    for row in trace:
        row["reward_to_go"] = float(returns[row["t"]])
    return Trajectory(np.asarray(observations), np.asarray(actions), np.asarray(rewards),
                      np.asarray(log_probs, dtype=np.float32), np.asarray(entropies),
                      np.asarray(probabilities), bool(terminated), bool(truncated), float(initial_reward), trace)
