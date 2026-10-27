"""Assignment-specified Pong crop and optional consecutive frame differences."""

import numpy as np


def preprocess(image):
    image = np.asarray(image)
    if image.shape != (210, 160, 3):
        raise ValueError(f"Expected RGB (210,160,3), received {image.shape}")
    # Copy: never mutate the emulator's observation storage.
    cropped = image[35:195:2, ::2, 0].copy()
    cropped[(cropped == 144) | (cropped == 109)] = 0
    return (cropped != 0).astype(np.float32)


class ObservationTransform:
    def __init__(self, config):
        self.config = config
        self.previous = None

    def reset(self):
        self.previous = None

    def __call__(self, observation):
        if self.config.environment == "CartPole-v1":
            result = np.asarray(observation, dtype=np.float32).copy()
            if result.shape != (4,) or not np.isfinite(result).all():
                raise ValueError("Invalid CartPole observation")
            return result
        current = preprocess(observation)
        if self.config.frame_difference:
            result = np.zeros_like(current) if self.previous is None else current - self.previous
        else:
            result = current
        self.previous = current
        return result.astype(np.int8)[None, :, :]
