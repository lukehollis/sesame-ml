"""Framework-neutral remote-policy contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from sesame_ml.constants import CONTROL_DT, JOINT_LIMITS_RAD, validate_joint_vector


def _rgb_image(value: np.ndarray) -> np.ndarray:
    image = np.asarray(value)
    if image.dtype != np.uint8:
        raise ValueError(f"rgb must have dtype uint8, got {image.dtype}")
    if image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 1:
        raise ValueError(f"rgb must have shape (height, width, 3), got {image.shape}")
    return np.ascontiguousarray(image)


@dataclass(frozen=True, slots=True)
class PolicyObservation:
    """The input shared by simulation, GPU policy servers, and the Wi-Fi robot runtime."""

    state_rad: np.ndarray
    rgb: np.ndarray
    instruction: str
    timestamp_ns: int
    observation_seq: int
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        state = validate_joint_vector(self.state_rad, name="state_rad").astype(np.float32)
        image = _rgb_image(self.rgb)
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        if int(self.timestamp_ns) < 0:
            raise ValueError("timestamp_ns cannot be negative")
        if int(self.observation_seq) < 0:
            raise ValueError("observation_seq cannot be negative")
        if not isinstance(self.context, dict):
            raise TypeError("context must be a dictionary")
        object.__setattr__(self, "state_rad", state)
        object.__setattr__(self, "rgb", image)
        object.__setattr__(self, "instruction", self.instruction.strip())
        object.__setattr__(self, "timestamp_ns", int(self.timestamp_ns))
        object.__setattr__(self, "observation_seq", int(self.observation_seq))


@dataclass(frozen=True, slots=True)
class PolicyActionChunk:
    """Absolute physical servo targets returned by any policy backend."""

    actions_rad: np.ndarray
    based_on_observation_seq: int
    dt_s: float = CONTROL_DT
    policy_name: str = "unknown"
    inference_latency_s: float | None = None
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions_rad, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] != 8:
            raise ValueError(f"actions_rad must have shape (horizon, 8), got {actions.shape}")
        if not np.all(np.isfinite(actions)):
            raise ValueError("actions_rad contains non-finite values")
        low, high = JOINT_LIMITS_RAD.T
        tolerance = 1e-5
        if np.any(actions < low[None, :] - tolerance) or np.any(
            actions > high[None, :] + tolerance
        ):
            raise ValueError("policy returned a target outside the global physical servo range")
        if int(self.based_on_observation_seq) < 0:
            raise ValueError("based_on_observation_seq cannot be negative")
        if not np.isfinite(self.dt_s) or float(self.dt_s) <= 0:
            raise ValueError("dt_s must be finite and positive")
        if self.inference_latency_s is not None and (
            not np.isfinite(self.inference_latency_s) or self.inference_latency_s < 0
        ):
            raise ValueError("inference_latency_s must be finite and non-negative")
        if not isinstance(self.info, dict):
            raise TypeError("info must be a dictionary")
        object.__setattr__(
            self, "actions_rad", np.clip(actions, low[None, :], high[None, :]).astype(np.float32)
        )
        object.__setattr__(self, "based_on_observation_seq", int(self.based_on_observation_seq))
        object.__setattr__(self, "dt_s", float(self.dt_s))

    @property
    def horizon(self) -> int:
        return int(self.actions_rad.shape[0])


def canonical_action_array(value: Any, *, source: str) -> np.ndarray:
    """Require a framework response to be an unbatched, absolute 8D action chunk."""

    actions = np.asarray(value)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] != 8:
        raise ValueError(
            f"{source} must return actions with shape (horizon, 8), got {actions.shape}. "
            "Configure the served physical output projection for eight absolute joint targets."
        )
    return actions


@runtime_checkable
class RemotePolicy(Protocol):
    """Common interface consumed by online planners and the robot policy host."""

    def infer(self, observation: PolicyObservation) -> PolicyActionChunk: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...
