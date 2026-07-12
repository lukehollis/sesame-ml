"""Command-conditioned central-pattern-generator locomotion baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from sesame_ml.constants import CONTROL_DT

from .base import BasePolicy, Observation, proprioception


class Gait(StrEnum):
    TROT = "trot"
    WALK = "walk"
    BOUND = "bound"


@dataclass(frozen=True)
class CPGConfig:
    """Parameters for the deployable open-loop gait with attitude feedback."""

    gait: Gait = Gait.TROT
    control_dt: float = CONTROL_DT
    base_frequency_hz: float = 1.65
    maximum_frequency_hz: float = 2.8
    hip_amplitude: float = 0.38
    lift_amplitude: float = 0.62
    idle_amplitude: float = 0.04
    turn_gain: float = 0.45
    attitude_gain: float = 0.16
    angular_rate_gain: float = 0.025
    phase_coupling_gain: float = 0.10

    def __post_init__(self) -> None:
        if self.control_dt <= 0:
            raise ValueError("control_dt must be positive")
        if self.base_frequency_hz <= 0 or self.maximum_frequency_hz < self.base_frequency_hz:
            raise ValueError("invalid oscillator frequency range")


class CPGPolicy(BasePolicy):
    """A deterministic gait controller useful as a baseline and demonstrator.

    The oscillator reads normalized commanded forward/yaw velocity from the
    observation and outputs the environment's residual joint convention.  It is
    intentionally stateful: call ``reset`` at every episode boundary.
    """

    evaluation_name = "cpg"

    # Leg order in this controller: front-right, rear-right, front-left, rear-left.
    _LOWER_INDICES = np.asarray([5, 4, 6, 7])
    # Distal joints at 0 degrees can only move positive; those at 180 only negative.
    _LOWER_DIRECTION = np.asarray([-1.0, 1.0, 1.0, -1.0])
    # Mirrored hip servo installation converts a common leg angle into firmware residuals.
    _HIP_DIRECTION = np.asarray([-1.0, 1.0, 1.0, -1.0])
    _SIDE = np.asarray([-1.0, -1.0, 1.0, 1.0])  # right=-1, left=+1

    def __init__(self, config: CPGConfig | None = None) -> None:
        self.config = config or CPGConfig()
        self._phase = 0.0
        self._rng = np.random.default_rng(0)

    @property
    def phase(self) -> float:
        return self._phase

    def reset(self, *, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        # A fixed phase makes deterministic evaluations comparable.  Supplying a
        # seed allows stochastic rollouts to randomize contact timing reproducibly.
        self._phase = 0.0 if seed is None else float(self._rng.uniform(0.0, 2.0 * math.pi))

    def _offsets(self) -> np.ndarray:
        return {
            Gait.TROT: np.asarray([0.0, math.pi, math.pi, 0.0]),
            Gait.WALK: np.asarray([0.0, math.pi, 1.5 * math.pi, 0.5 * math.pi]),
            Gait.BOUND: np.asarray([0.0, math.pi, 0.0, math.pi]),
        }[Gait(self.config.gait)]

    def predict(
        self, observation: Observation, *, deterministic: bool = True
    ) -> np.ndarray:
        state = proprioception(observation)
        command = state[33:36]
        forward = float(np.clip(command[0], -1.0, 1.0))
        yaw = float(np.clip(command[2], -1.0, 1.0))
        effort = float(np.clip(max(abs(forward), 0.55 * abs(yaw)), 0.0, 1.0))
        frequency = self.config.base_frequency_hz + effort * (
            self.config.maximum_frequency_hz - self.config.base_frequency_hz
        )
        phase_jitter = 0.0 if deterministic else float(self._rng.normal(0.0, 0.012))
        self._phase = (
            self._phase + 2.0 * math.pi * frequency * self.config.control_dt + phase_jitter
        ) % (2.0 * math.pi)

        phases = self._phase + self._offsets()
        stride = np.sin(phases)
        # Smooth half-wave swing profile with zero velocity at liftoff/touchdown.
        swing = np.maximum(np.sin(phases), 0.0) ** 2
        amplitude = self.config.idle_amplitude + (1.0 - self.config.idle_amplitude) * effort
        side_drive = forward + self.config.turn_gain * yaw * self._SIDE
        hip = self._HIP_DIRECTION * self.config.hip_amplitude * amplitude * side_drive * stride

        # Extend endpoint-limited distal servos inward during swing.  A small
        # stance component keeps the feet compliant instead of hard against 0/180°.
        lower = self._LOWER_DIRECTION * self.config.lift_amplitude * amplitude * (
            0.12 + 0.88 * swing
        )

        # Body-attitude feedback, shared with the stand controller, remains active
        # during gait generation and materially improves randomized evaluation.
        up = state[0:3]
        angular = state[6:9]
        pitch_feedback = self.config.attitude_gain * float(
            up[0]
        ) + self.config.angular_rate_gain * float(angular[1])
        roll_feedback = self.config.attitude_gain * float(
            up[1]
        ) + self.config.angular_rate_gain * float(angular[0])
        hip += np.asarray(
            [
                -pitch_feedback - roll_feedback,
                pitch_feedback - roll_feedback,
                pitch_feedback + roll_feedback,
                -pitch_feedback + roll_feedback,
            ]
        )

        action = np.zeros(8, dtype=np.float32)
        action[:4] = hip
        action[self._LOWER_INDICES] = lower
        return np.clip(action, -1.0, 1.0)
