"""Canonical robot conventions shared by simulation, datasets, and hardware."""

from __future__ import annotations

import numpy as np

# Firmware/PWM channel order. This is the only action order used on the wire or in datasets.
JOINT_NAMES: tuple[str, ...] = ("R1", "R2", "L1", "L2", "R4", "R3", "L3", "L4")
JOINT_INDEX = {name: index for index, name in enumerate(JOINT_NAMES)}

STAND_ANGLES_DEG = np.asarray([135, 45, 45, 135, 0, 180, 0, 180], dtype=np.float64)
REST_ANGLES_DEG = np.full(8, 90.0, dtype=np.float64)

# Limits already enforced by Sesame Studio for the four hips; the distal joints use their
# full nominal 180-degree servo range. Calibration can tighten these per physical robot.
JOINT_LIMITS_DEG = np.asarray(
    [
        [45, 180],  # R1
        [0, 135],   # R2
        [0, 135],   # L1
        [45, 180],  # L2
        [0, 180],   # R4
        [0, 180],   # R3
        [0, 180],   # L3
        [0, 180],   # L4
    ],
    dtype=np.float64,
)

STAND_ANGLES_RAD = np.deg2rad(STAND_ANGLES_DEG)
REST_ANGLES_RAD = np.deg2rad(REST_ANGLES_DEG)
JOINT_LIMITS_RAD = np.deg2rad(JOINT_LIMITS_DEG)

ACTION_RATE_HZ = 50.0
CONTROL_DT = 1.0 / ACTION_RATE_HZ
PHYSICS_DT = 0.002
SERVO_MAX_SPEED_RAD_S = np.deg2rad(600.0)  # MG90S nominal: about 60 degrees / 0.1 s.
SERVO_MAX_TORQUE_NM = 0.18

# Public robot-frame convention: +x front (OLED/camera), +y left, +z up.
ROBOT_FRAME = "x-forward, y-left, z-up"


def validate_joint_vector(value: np.ndarray | list[float], *, name: str = "joints") -> np.ndarray:
    """Return a finite float64 joint vector in canonical firmware order."""

    result = np.asarray(value, dtype=np.float64)
    if result.shape != (len(JOINT_NAMES),):
        raise ValueError(f"{name} must have shape ({len(JOINT_NAMES)},), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def clip_servo_targets(target_rad: np.ndarray | list[float]) -> np.ndarray:
    """Clamp physical servo targets to the conservative mechanical limits."""

    target = validate_joint_vector(target_rad, name="target_rad")
    return np.clip(target, JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1])


def normalized_to_servo(action: np.ndarray | list[float]) -> np.ndarray:
    """Map [-1, 1] to each calibrated joint's full physical range."""

    normalized = validate_joint_vector(action, name="action")
    normalized = np.clip(normalized, -1.0, 1.0)
    low, high = JOINT_LIMITS_RAD.T
    return low + 0.5 * (normalized + 1.0) * (high - low)


def servo_to_normalized(target_rad: np.ndarray | list[float]) -> np.ndarray:
    """Map physical radians to the normalized full-range action space."""

    target = clip_servo_targets(target_rad)
    low, high = JOINT_LIMITS_RAD.T
    return 2.0 * (target - low) / (high - low) - 1.0
