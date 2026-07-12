"""Gymnasium environments for Sesame tasks."""

from __future__ import annotations

from gymnasium.envs.registration import register, registry

from sesame_ml.envs.core import SesameEnv, SesameEnvConfig, Task

_ENVIRONMENTS = {
    "SesameStand-v0": {"task": Task.STAND},
    "SesameRecovery-v0": {"task": Task.RECOVERY},
    "SesameLocomotion-v0": {"task": Task.LOCOMOTION},
    "SesameNavigation-v0": {"task": Task.NAVIGATION},
}


def register_environments() -> None:
    for environment_id, kwargs in _ENVIRONMENTS.items():
        if environment_id not in registry:
            register(
                id=environment_id,
                entry_point="sesame_ml.envs.core:SesameEnv",
                kwargs=kwargs,
                max_episode_steps=kwargs["task"].default_episode_steps,
            )


register_environments()

__all__ = ["SesameEnv", "SesameEnvConfig", "Task", "register_environments"]
