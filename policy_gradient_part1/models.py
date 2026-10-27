"""Two-logit policies; deliberately no critic, dropout, or batch normalization."""

import torch
from torch import nn


def make_policy(config):
    if config.environment == "CartPole-v1":
        layers = []
        width = 4
        for hidden in config.hidden_sizes:
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            width = hidden
        layers.append(nn.Linear(width, 2))
        return nn.Sequential(*layers)
    return nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=8, stride=4), nn.ReLU(),
        nn.Conv2d(8, 16, kernel_size=4, stride=2), nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3), nn.ReLU(), nn.Flatten(),
        nn.Linear(32 * 6 * 6, 64), nn.ReLU(), nn.Linear(64, 2),
    )


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())
