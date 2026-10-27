"""Required raw rewards, full-window MA100 and evaluation reward histogram."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def training_plot(history, path, title):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rewards = np.array([row["reward"] for row in history])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(np.arange(1, len(rewards)+1), rewards, alpha=0.35, lw=0.7, label="Raw episode reward")
    if len(rewards) >= 100:
        ax.plot(np.arange(100, len(rewards)+1), np.convolve(rewards, np.ones(100)/100, mode="valid"),
                color="darkorange", lw=1.8, label="100-episode moving average")
    ax.set(xlabel="Training episode", ylabel="Undiscounted episode reward", title=title)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def histogram(rewards, path, title):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.hist(rewards, bins="auto", color="steelblue", edgecolor="white")
    ax.set(xlabel="Undiscounted episode reward", ylabel="Number of episodes", title=title)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
