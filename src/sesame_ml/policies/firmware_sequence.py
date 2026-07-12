"""Regression baseline derived from the robot's shipping firmware motions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from sesame_ml.constants import CONTROL_DT, STAND_ANGLES_DEG, STAND_ANGLES_RAD

from .base import BasePolicy, Observation, proprioception


class FirmwareMotion(StrEnum):
    AUTO = "auto"
    FORWARD = "forward"
    BACKWARD = "backward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"


@dataclass(frozen=True)
class FirmwareSequenceConfig:
    """Timing and selection for the firmware-keyframe demonstrator."""

    motion: FirmwareMotion = FirmwareMotion.AUTO
    control_dt: float = CONTROL_DT
    frame_duration_s: float = 0.100
    action_scale_rad: float = 0.58
    interpolate: bool = True
    command_deadband: float = 0.08

    def __post_init__(self) -> None:
        if self.control_dt <= 0 or self.frame_duration_s <= 0:
            raise ValueError("control_dt and frame_duration_s must be positive")
        if self.action_scale_rad <= 0:
            raise ValueError("action_scale_rad must be positive")


def _pose(**updates: float) -> np.ndarray:
    """Apply named firmware-order servo updates to a fresh stand pose."""

    from sesame_ml.constants import JOINT_INDEX

    pose = STAND_ANGLES_DEG.copy()
    for name, value in updates.items():
        pose[JOINT_INDEX[name]] = value
    return pose


def _cumulative_frames(updates: tuple[dict[str, float], ...]) -> np.ndarray:
    """Reproduce firmware's stateful setServoAngle calls as complete keyframes."""

    from sesame_ml.constants import JOINT_INDEX

    current = STAND_ANGLES_DEG.copy()
    frames: list[np.ndarray] = []
    for update in updates:
        for name, value in update.items():
            current[JOINT_INDEX[name]] = value
        frames.append(current.copy())
    return np.asarray(frames, dtype=np.float64)


# These are a direct transcription of runWalkPose/runWalkBackward/runTurnLeft/
# runTurnRight in firmware/movement-sequences.h.  Each dictionary corresponds to
# the setServoAngle group before one pressingCheck(..., frameDelay).
_FORWARD_INITIAL = _cumulative_frames(
    ({"R3": 135, "L3": 45, "R2": 100, "L1": 25},)
)[0]
_FORWARD_CYCLE_UPDATES = (
    {"R3": 135, "L3": 0},
    {"L4": 135, "L2": 90, "R4": 0, "R1": 180},
    {"R2": 45, "L1": 90},
    {"R4": 45, "L4": 180},
    {"R3": 180, "L3": 45, "R2": 90, "L1": 0},
    {"L2": 135, "R1": 90},
)
_BACKWARD_CYCLE_UPDATES = (
    {"R3": 135, "L3": 0},
    {"L4": 135, "L2": 135, "R4": 0, "R1": 90},
    {"R2": 90, "L1": 0},
    {"R4": 45, "L4": 180},
    {"R3": 180, "L3": 45, "R2": 45, "L1": 90},
    {"L2": 90, "R1": 180},
)
_LEFT_CYCLE_UPDATES = (
    {"R3": 135, "L4": 135},
    {"R1": 180, "L2": 180},
    {"R3": 180, "L4": 180},
    {"R1": 135, "L2": 135},
    {"R4": 45, "L3": 45},
    {"R2": 90, "L1": 90},
    {"R4": 0, "L3": 0},
    {"R2": 45, "L1": 45},
)
_RIGHT_CYCLE_UPDATES = (
    {"R4": 45, "L3": 45},
    {"R2": 0, "L1": 0},
    {"R4": 0, "L3": 0},
    {"R2": 45, "L1": 45},
    {"R3": 135, "L4": 135},
    {"R1": 90, "L2": 90},
    {"R3": 180, "L4": 180},
    {"R1": 135, "L2": 135},
)


def _loop_from_updates(
    updates: tuple[dict[str, float], ...], *, initial: np.ndarray | None = None
) -> np.ndarray:
    from sesame_ml.constants import JOINT_INDEX

    current = STAND_ANGLES_DEG.copy() if initial is None else initial.copy()
    result: list[np.ndarray] = []
    for update in updates:
        for name, value in update.items():
            current[JOINT_INDEX[name]] = value
        result.append(current.copy())
    return np.asarray(result)


FIRMWARE_KEYFRAMES_DEG: dict[FirmwareMotion, np.ndarray] = {
    FirmwareMotion.FORWARD: np.vstack(
        [_FORWARD_INITIAL, _loop_from_updates(_FORWARD_CYCLE_UPDATES, initial=_FORWARD_INITIAL)]
    ),
    # runWalkBackward performs one frameDelay hold before its loop.
    FirmwareMotion.BACKWARD: np.vstack(
        [STAND_ANGLES_DEG, _loop_from_updates(_BACKWARD_CYCLE_UPDATES)]
    ),
    FirmwareMotion.TURN_LEFT: _loop_from_updates(_LEFT_CYCLE_UPDATES),
    FirmwareMotion.TURN_RIGHT: _loop_from_updates(_RIGHT_CYCLE_UPDATES),
}


class FirmwareSequencePolicy(BasePolicy):
    """Replay the field-tested firmware gait in Gym residual-action space.

    The source firmware uses absolute 0–180 degree commands.  ``SesameEnv`` uses
    residuals around stand with a default 0.58-radian span, so large firmware
    excursions are necessarily clipped at ±1.  ``servo_targets_rad`` retains the
    exact interpolated physical target for regression inspection or an absolute-
    target hardware runtime; ``predict`` is always the safe Gym residual action.
    """

    evaluation_name = "firmware_sequence"

    def __init__(self, config: FirmwareSequenceConfig | None = None) -> None:
        self.config = config or FirmwareSequenceConfig()
        self._motion: FirmwareMotion | None = None
        self._elapsed_s = 0.0
        self._servo_targets_rad = STAND_ANGLES_RAD.copy()

    @property
    def motion(self) -> FirmwareMotion | None:
        return self._motion

    @property
    def servo_targets_rad(self) -> np.ndarray:
        return self._servo_targets_rad.copy()

    def reset(self, *, seed: int | None = None) -> None:
        del seed
        self._motion = None
        self._elapsed_s = 0.0
        self._servo_targets_rad = STAND_ANGLES_RAD.copy()

    def _select_motion(self, observation: Observation) -> FirmwareMotion | None:
        configured = FirmwareMotion(self.config.motion)
        if configured is not FirmwareMotion.AUTO:
            return configured
        command = proprioception(observation)[33:36]
        forward, yaw = float(command[0]), float(command[2])
        if abs(yaw) > max(abs(forward), self.config.command_deadband):
            return FirmwareMotion.TURN_LEFT if yaw > 0 else FirmwareMotion.TURN_RIGHT
        if forward > self.config.command_deadband:
            return FirmwareMotion.FORWARD
        if forward < -self.config.command_deadband:
            return FirmwareMotion.BACKWARD
        return None

    def _target_degrees(self, motion: FirmwareMotion) -> np.ndarray:
        frame_position = self._elapsed_s / self.config.frame_duration_s
        lower_index = math.floor(frame_position)
        lower = self._pose_at_frame(motion, lower_index)
        if not self.config.interpolate:
            return lower
        fraction = frame_position - math.floor(frame_position)
        upper = self._pose_at_frame(motion, lower_index + 1)
        return lower + fraction * (upper - lower)

    @staticmethod
    def _pose_at_frame(motion: FirmwareMotion, frame_index: int) -> np.ndarray:
        """Resolve an unbounded frame without reintroducing the one-time prefix."""

        from sesame_ml.constants import JOINT_INDEX

        pose = STAND_ANGLES_DEG.copy()

        def apply(update: dict[str, float]) -> None:
            for name, value in update.items():
                pose[JOINT_INDEX[name]] = value

        if motion is FirmwareMotion.FORWARD:
            apply({"R3": 135, "L3": 45, "R2": 100, "L1": 25})
            # The initial step occurs once.  Firmware then repeats only the six
            # updates inside its walkCycles loop.
            for index in range(frame_index):
                apply(_FORWARD_CYCLE_UPDATES[index % len(_FORWARD_CYCLE_UPDATES)])
        elif motion is FirmwareMotion.BACKWARD:
            # Firmware intentionally holds stand for one frameDelay before its loop.
            for index in range(max(0, frame_index)):
                apply(_BACKWARD_CYCLE_UPDATES[index % len(_BACKWARD_CYCLE_UPDATES)])
        else:
            updates = (
                _LEFT_CYCLE_UPDATES
                if motion is FirmwareMotion.TURN_LEFT
                else _RIGHT_CYCLE_UPDATES
            )
            for index in range(frame_index + 1):
                apply(updates[index % len(updates)])
        return pose

    def predict(
        self, observation: Observation, *, deterministic: bool = True
    ) -> np.ndarray:
        del deterministic
        selected = self._select_motion(observation)
        if selected is None:
            self._motion = None
            self._elapsed_s = 0.0
            self._servo_targets_rad = STAND_ANGLES_RAD.copy()
            return np.zeros(8, dtype=np.float32)
        if selected is not self._motion:
            self._motion = selected
            self._elapsed_s = 0.0
        degrees = self._target_degrees(selected)
        self._servo_targets_rad = np.deg2rad(degrees)
        residual = (self._servo_targets_rad - STAND_ANGLES_RAD) / self.config.action_scale_rad
        self._elapsed_s += self.config.control_dt
        return np.clip(residual, -1.0, 1.0).astype(np.float32)
