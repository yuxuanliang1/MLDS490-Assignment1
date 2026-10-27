"""Small, explicit experiment configuration; JSON avoids another dependency."""

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path


@dataclass
class Config:
    environment: str = "CartPole-v1"
    gamma: float = 0.95
    learning_rate: float = 0.01
    episodes: int = 1000
    batch_episodes: int = 10
    seed: int = 11
    device: str = "cpu"
    hidden_sizes: tuple = (128, 128)
    checkpoint_interval: int = 100
    log_interval: int = 25
    gradient_chunk_size: int = 256
    gradient_clip: float | None = None
    discount_weighting: str = "relative"
    frame_difference: bool = True
    frameskip: int = 4
    repeat_action_probability: float = 0.25
    fire_on_reset: bool = False
    max_episode_steps: int | None = None
    threads: int = 1

    def validate(self):
        if self.environment not in ("CartPole-v1", "ALE/Pong-v5"):
            raise ValueError("Only CartPole-v1 and ALE/Pong-v5 are supported")
        if self.discount_weighting not in ("relative", "absolute"):
            raise ValueError("discount_weighting must be relative (G_t) or absolute (gamma^t G_t)")
        if not 0 <= self.gamma <= 1 or not 0 < self.learning_rate < float("inf"):
            raise ValueError("Invalid gamma or learning rate")
        for name in ("episodes", "batch_episodes", "checkpoint_interval", "log_interval",
                     "gradient_chunk_size", "threads", "frameskip"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.hidden_sizes or any(not isinstance(n, int) or n < 1 for n in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")
        if self.gradient_clip is not None and not 0 < self.gradient_clip < float("inf"):
            raise ValueError("gradient_clip must be finite and positive")
        if self.max_episode_steps is not None and self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        if not 0 <= self.repeat_action_probability <= 1 or not 0 <= self.seed < 2**31:
            raise ValueError("Invalid stickiness or seed")
        self.hidden_sizes = tuple(self.hidden_sizes)
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(data) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {unknown}")
        return cls(**data).validate()
