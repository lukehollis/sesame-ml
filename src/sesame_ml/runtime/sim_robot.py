"""Run MuJoCo through the exact Wi-Fi observation/action protocol used by hardware."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

from sesame_ml.constants import STAND_ANGLES_RAD
from sesame_ml.envs import SesameEnv
from sesame_ml.transport import ActionSample, ControlState, ObservationV1


@dataclass
class SimRuntimeMetrics:
    steps: int = 0
    fallback_steps: int = 0
    episode_return: float = 0.0
    episodes: int = 0
    falls: int = 0


class SimulatedRobotRuntime:
    """Callbacks that make a MuJoCo environment behave like the Orange Pi runtime.

    ``RobotWebSocketClient`` calls ``observation_source`` at the camera rate and
    ``action_sink`` at 50 Hz. Policy-host code therefore cannot tell whether it is driving
    this simulator or a robot, which is the essential sim-to-real integration test.
    """

    def __init__(
        self,
        env: SesameEnv,
        *,
        robot_id: str = "sesame-sim-001",
        seed: int = 0,
        jpeg_quality: int = 85,
        fallback_target_rad: np.ndarray | None = None,
    ) -> None:
        if not 1 <= jpeg_quality <= 95:
            raise ValueError("jpeg_quality must be between 1 and 95")
        self.env = env
        self.robot_id = robot_id
        self.seed = seed
        self.jpeg_quality = jpeg_quality
        self.fallback_target = (
            STAND_ANGLES_RAD.copy()
            if fallback_target_rad is None
            else np.asarray(fallback_target_rad, dtype=np.float64)
        )
        if self.fallback_target.shape != (8,):
            raise ValueError("fallback_target_rad must have shape (8,)")
        self.metrics = SimRuntimeMetrics()
        self._observation, self._info = env.reset(seed=seed)

    def _camera_jpeg(self) -> bytes:
        if isinstance(self._observation, dict) and "rgb" in self._observation:
            image = self._observation["rgb"]
        else:
            image = self.env.render_camera()
        output = io.BytesIO()
        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(
            output, format="JPEG", quality=self.jpeg_quality, optimize=False
        )
        return output.getvalue()

    def observation_source(self, sequence: int, monotonic_ns: int) -> ObservationV1:
        quaternion, gyro, acceleration = self.env.deployment_imu()
        return ObservationV1(
            robot_id=self.robot_id,
            sequence=sequence,
            monotonic_ns=monotonic_ns,
            # Feedback-free deployment exposes the commanded target, not perfect qpos.
            joint_position=tuple(float(value) for value in self.env.servo_targets),
            joint_velocity=None,
            command_velocity=tuple(float(value) for value in self.env.command),
            imu_quaternion=tuple(float(value) for value in quaternion),
            imu_gyro=tuple(float(value) for value in gyro),
            imu_acceleration=tuple(float(value) for value in acceleration),
            language_instruction=self.env.language_instruction,
            battery_voltage=float(self._info["domain"]["battery_voltage"]),
            image_jpeg=self._camera_jpeg(),
            status={
                "simulated": True,
                "task": self.env.task.value,
                "control_step": self.metrics.steps,
            },
        )

    def action_sink(self, sample: ActionSample) -> None:
        if sample.target is None:
            target = self.fallback_target
            self.metrics.fallback_steps += 1
        else:
            target = np.asarray(sample.target, dtype=np.float64)
        self._observation, reward, terminated, truncated, self._info = (
            self.env.step_servo_targets(target)
        )
        self.metrics.steps += 1
        self.metrics.episode_return += reward
        if sample.state is ControlState.FALLBACK:
            self.metrics.fallback_steps += int(sample.target is not None)
        if terminated or truncated:
            self.metrics.episodes += 1
            self.metrics.falls += int(terminated and not self._info.get("success", False))
            next_seed = self.seed + self.metrics.episodes
            self._observation, self._info = self.env.reset(seed=next_seed)
            self.metrics.episode_return = 0.0

    @property
    def last_info(self) -> dict:
        return self._info.copy()

    def close(self) -> None:
        self.env.close()
