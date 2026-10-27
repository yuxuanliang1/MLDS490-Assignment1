"""Shared CLI keeps environment-specific entry points small."""

import argparse
from pathlib import Path

from .config import Config
from .runtime import ROOT
from .trainer import train


def training_main(environment):
    name = "cartpole" if environment == "CartPole-v1" else "pong"
    parser = argparse.ArgumentParser(description=f"Train no-baseline REINFORCE on {environment}")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / f"{name}_part1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--figure", type=Path)
    for option, kind in (("episodes", int), ("learning-rate", float), ("gamma", float), ("seed", int),
                         ("device", str), ("checkpoint-interval", int), ("log-interval", int),
                         ("batch-episodes", int), ("max-episode-steps", int)):
        parser.add_argument("--"+option, type=kind)
    parser.add_argument("--raw-frames", action="store_true")
    parser.add_argument("--discount-weighting", choices=["relative", "absolute"])
    args = parser.parse_args()
    config = Config.load(args.config)
    if config.environment != environment:
        parser.error("Configuration environment does not match the training entry point")
    for key in ("episodes", "learning_rate", "gamma", "seed", "device", "checkpoint_interval", "log_interval",
                "batch_episodes", "max_episode_steps"):
        if getattr(args, key) is not None:
            setattr(config, key, getattr(args, key))
    if args.raw_frames:
        config.frame_difference = False
    if args.discount_weighting:
        config.discount_weighting = args.discount_weighting
    train(config, args.output, args.resume, args.figure)
