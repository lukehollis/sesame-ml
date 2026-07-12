"""Common policy contract for simulation, evaluation, and hardware deployment.

All policies in this package emit the same eight residual actions consumed by
``SesameEnv.step``: zero is the calibrated standing pose and each component is
clipped to ``[-1, 1]``.  Keeping this contract explicit prevents accidentally
sending a normalized full-servo-range action to a residual-action controller.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Protocol, TypeAlias, runtime_checkable

import numpy as np

from sesame_ml.constants import JOINT_NAMES

Observation: TypeAlias = np.ndarray | Mapping[str, np.ndarray]


class ActionMode(StrEnum):
    """Action interpretation advertised by a policy."""

    RESIDUAL = "stand_pose_residual"


@runtime_checkable
class PolicyLike(Protocol):
    """Structural interface accepted by evaluators and online runtimes."""

    action_mode: ActionMode

    def reset(self, *, seed: int | None = None) -> None:
        """Reset recurrent/oscillator state before a new episode."""

    def predict(self, observation: Observation, *, deterministic: bool = True) -> Any:
        """Return one residual action, or an SB3-style ``(action, state)`` tuple."""


class BasePolicy(ABC):
    """Convenience base class for Sesame policies."""

    action_mode = ActionMode.RESIDUAL

    def reset(self, *, seed: int | None = None) -> None:
        del seed

    @abstractmethod
    def predict(
        self, observation: Observation, *, deterministic: bool = True
    ) -> np.ndarray:
        """Produce one action in the Gym residual-action convention."""


def proprioception(observation: Observation) -> np.ndarray:
    """Extract the 43-element proprioceptive vector from either observation mode."""

    value: Any
    if isinstance(observation, Mapping):
        if "proprio" not in observation:
            raise KeyError("dictionary observation is missing 'proprio'")
        value = observation["proprio"]
    else:
        value = observation
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (43,):
        raise ValueError(f"proprioceptive observation must have shape (43,), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError("proprioceptive observation contains non-finite values")
    return result


def residual_action(value: Any, *, clip: bool = True) -> np.ndarray:
    """Validate and canonicalize a single firmware-order residual action."""

    # Stable-Baselines3 returns ``(action, recurrent_state)``.  Accept that form
    # at system boundaries while keeping native policies' return value simple.
    if isinstance(value, tuple):
        if not value:
            raise ValueError("policy returned an empty tuple")
        value = value[0]
    action = np.asarray(value, dtype=np.float32)
    if action.shape == (1, len(JOINT_NAMES)):
        action = action[0]
    if action.shape != (len(JOINT_NAMES),):
        raise ValueError(
            f"policy action must have shape ({len(JOINT_NAMES)},), got {action.shape}"
        )
    if not np.all(np.isfinite(action)):
        raise ValueError("policy action contains non-finite values")
    if not clip and (np.any(action < -1.0) or np.any(action > 1.0)):
        raise ValueError("policy residual action lies outside [-1, 1]")
    return np.clip(action, -1.0, 1.0) if clip else action


def predict_action(
    policy: PolicyLike, observation: Observation, *, deterministic: bool = True
) -> np.ndarray:
    """Invoke any compatible policy and return a validated residual action."""

    action_mode = getattr(policy, "action_mode", ActionMode.RESIDUAL)
    if ActionMode(action_mode) is not ActionMode.RESIDUAL:
        raise ValueError(f"expected a residual-action policy, got {action_mode!r}")
    return residual_action(policy.predict(observation, deterministic=deterministic))
