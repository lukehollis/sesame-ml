"""Stable-Baselines3 checkpoint adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import ActionMode, BasePolicy, Observation, residual_action


class SB3Policy(BasePolicy):
    """Expose an SB3 algorithm through the shared single-environment policy API.

    Recurrent state is retained if the wrapped algorithm supports it.  Optional
    observation normalization can be supplied as a fitted ``VecNormalize``
    object; reward normalization is never applied during inference.
    """

    action_mode = ActionMode.RESIDUAL
    evaluation_name = "sb3"

    def __init__(self, model: Any, *, normalizer: Any | None = None) -> None:
        self.model = model
        self.normalizer = normalizer
        self._state: Any = None
        self._episode_start = np.ones((1,), dtype=bool)
        self._owns_normalizer_environment = False
        shape = getattr(getattr(model, "action_space", None), "shape", None)
        if shape is not None and tuple(shape) != (8,):
            raise ValueError(f"SB3 checkpoint has action shape {shape}, expected (8,)")
        if normalizer is not None:
            normalizer.training = False
            normalizer.norm_reward = False

    @classmethod
    def load(
        cls,
        checkpoint: str | Path,
        *,
        algorithm: str = "ppo",
        device: str = "auto",
        environment: Any | None = None,
        vecnormalize_path: str | Path | None = None,
    ) -> SB3Policy:
        """Load any saved SB3 algorithm and, optionally, its normalization state."""

        try:
            from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        except ImportError as error:  # pragma: no cover - optional dependency path
            raise ImportError(
                "Stable-Baselines3 support requires `pip install sesame-ml[train]`"
            ) from error

        checkpoint_path = Path(checkpoint).expanduser()
        algorithms = {
            "a2c": A2C,
            "ddpg": DDPG,
            "dqn": DQN,
            "ppo": PPO,
            "sac": SAC,
            "td3": TD3,
        }
        try:
            algorithm_type = algorithms[algorithm.lower()]
        except KeyError as error:
            raise ValueError(
                f"unsupported SB3 algorithm {algorithm!r}; choose {sorted(algorithms)}"
            ) from error
        model = algorithm_type.load(checkpoint_path, device=device)
        normalizer = None
        if vecnormalize_path is not None:
            if environment is None:
                raise ValueError("environment is required when loading VecNormalize statistics")
            if hasattr(environment, "num_envs"):
                vec_env = environment
            else:
                vec_env = DummyVecEnv([lambda: environment])
            normalizer = VecNormalize.load(str(Path(vecnormalize_path).expanduser()), vec_env)
        policy = cls(model, normalizer=normalizer)
        policy._owns_normalizer_environment = normalizer is not None
        return policy

    def reset(self, *, seed: int | None = None) -> None:
        del seed
        self._state = None
        self._episode_start = np.ones((1,), dtype=bool)

    def _normalize(self, observation: Observation) -> Observation:
        normalized: Observation = observation
        if self.normalizer is not None:
            # VecNormalize expects a batch.  It supports ndarray and dict observations.
            if isinstance(observation, dict):
                batched = {
                    key: np.expand_dims(value, 0) for key, value in observation.items()
                }
                normalized_batch = self.normalizer.normalize_obs(batched)
                normalized = {
                    key: np.asarray(value)[0] for key, value in normalized_batch.items()
                }
            else:
                batched = np.expand_dims(observation, 0)
                normalized = np.asarray(self.normalizer.normalize_obs(batched))[0]
        # Pixel PPO checkpoints are trained behind VecTransposeImage and therefore
        # advertise CHW spaces even though SesameEnv and the camera wire format use HWC.
        expected_space = getattr(self.model, "observation_space", None)
        if isinstance(normalized, dict) and hasattr(expected_space, "spaces"):
            result = dict(normalized)
            for key, value in result.items():
                expected_shape = getattr(expected_space.spaces.get(key), "shape", None)
                if (
                    expected_shape is not None
                    and value.ndim == 3
                    and value.shape != tuple(expected_shape)
                    and (value.shape[2], value.shape[0], value.shape[1])
                    == tuple(expected_shape)
                ):
                    result[key] = np.moveaxis(value, -1, 0)
            normalized = result
        return normalized

    def predict(
        self, observation: Observation, *, deterministic: bool = True
    ) -> np.ndarray:
        normalized = self._normalize(observation)
        kwargs = {
            "state": self._state,
            "episode_start": self._episode_start,
            "deterministic": deterministic,
        }
        try:
            action, self._state = self.model.predict(normalized, **kwargs)
        except TypeError:
            # Third-party SB3-compatible algorithms may not accept recurrent kwargs.
            action, self._state = self.model.predict(normalized, deterministic=deterministic)
        self._episode_start[:] = False
        return residual_action(action)

    def save(self, checkpoint: str | Path) -> None:
        self.model.save(str(checkpoint))

    def close(self) -> None:
        """Close the auxiliary VecNormalize environment, if one was loaded."""

        if self._owns_normalizer_environment and self.normalizer is not None:
            self.normalizer.close()
            self._owns_normalizer_environment = False
