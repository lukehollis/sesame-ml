"""Safe standing baseline policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BasePolicy, Observation, proprioception


@dataclass(frozen=True)
class StandPolicyConfig:
    """Gains for the conservative attitude-damping stand controller."""

    attitude_gain: float = 0.20
    angular_rate_gain: float = 0.035
    maximum_correction: float = 0.18


class StandPolicy(BasePolicy):
    """Hold the calibrated stand pose with a small body-attitude correction.

    With a level, stationary observation this returns exactly zero.  Corrections
    only move joints inward from endpoint-limited distal stand angles, so the
    output is safe under the same residual semantics used for learned policies.
    """

    evaluation_name = "stand"

    def __init__(self, config: StandPolicyConfig | None = None) -> None:
        self.config = config or StandPolicyConfig()

    def predict(
        self, observation: Observation, *, deterministic: bool = True
    ) -> np.ndarray:
        del deterministic
        state = proprioception(observation)
        up = state[0:3]
        angular_velocity = state[6:9]
        # For small attitude errors, body-frame gravity x/y correspond to pitch
        # and roll.  Mirrored leg corrections lower the high side of the body.
        pitch = float(up[0])
        roll = float(up[1])
        pitch_rate = float(angular_velocity[1])
        roll_rate = float(angular_velocity[0])
        pitch_correction = (
            self.config.attitude_gain * pitch + self.config.angular_rate_gain * pitch_rate
        )
        roll_correction = (
            self.config.attitude_gain * roll + self.config.angular_rate_gain * roll_rate
        )

        action = np.zeros(8, dtype=np.float32)
        # Hip order: front-right, rear-right, front-left, rear-left.
        action[:4] = np.asarray(
            [
                -pitch_correction - roll_correction,
                pitch_correction - roll_correction,
                pitch_correction + roll_correction,
                -pitch_correction + roll_correction,
            ],
            dtype=np.float32,
        )
        return np.clip(
            action, -self.config.maximum_correction, self.config.maximum_correction
        ).astype(np.float32)
