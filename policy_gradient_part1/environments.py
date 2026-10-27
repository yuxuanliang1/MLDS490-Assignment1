"""Exact environment identities and verified two-action Pong mapping."""

import gymnasium as gym

PONG_ACTIONS = (2, 3)
PONG_MEANINGS = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]


def make_environment(config):
    kwargs = {}
    if config.max_episode_steps is not None:
        kwargs["max_episode_steps"] = config.max_episode_steps
    if config.environment == "ALE/Pong-v5":
        try:
            import ale_py
            gym.register_envs(ale_py)
        except ImportError as exc:
            raise RuntimeError("Install requirements-pg.txt, including ale-py; no Atari fallback is used") from exc
        kwargs.update(frameskip=config.frameskip,
                      repeat_action_probability=config.repeat_action_probability,
                      full_action_space=False, obs_type="rgb")
    env = gym.make(config.environment, **kwargs)
    if env.spec.id != config.environment:
        env.close()
        raise RuntimeError("Unexpected environment identity")
    if config.environment == "ALE/Pong-v5":
        meanings = env.unwrapped.get_action_meanings()
        if env.action_space.n != 6 or meanings != PONG_MEANINGS:
            env.close()
            raise RuntimeError(f"Unexpected Pong actions: {meanings}")
    return env


def environment_metadata(env, config):
    return dict(id=env.spec.id, max_episode_steps=env.spec.max_episode_steps,
                action_meanings=(env.unwrapped.get_action_meanings()
                                 if config.environment == "ALE/Pong-v5" else ["LEFT", "RIGHT"]),
                learned_action_mapping=(list(PONG_ACTIONS) if config.environment == "ALE/Pong-v5" else [0, 1]),
                frameskip=config.frameskip if config.environment == "ALE/Pong-v5" else None,
                repeat_action_probability=(config.repeat_action_probability
                                           if config.environment == "ALE/Pong-v5" else None),
                fire_on_reset=config.fire_on_reset if config.environment == "ALE/Pong-v5" else False,
                reward_clipping=False, point_return_resets=False,
                frame_difference=config.frame_difference if config.environment == "ALE/Pong-v5" else False)
