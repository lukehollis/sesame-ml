"""Bridge remote VLA clients into Sesame's latest-only Wi-Fi policy host."""

from __future__ import annotations

import io
from numbers import Integral, Real
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from sesame_ml.integrations.base import PolicyObservation, RemotePolicy
from sesame_ml.transport import ObservationV1, PolicyOutput


class RemotePolicyBridge:
    """Callable accepted directly by :class:`PolicyWebSocketServer`.

    Camera JPEG decoding intentionally happens on the GPU-host side, leaving the Orange
    Pi wire packet compact. The returned action chunk remains absolute radians all the way
    to the robot; no normalized/residual action convention crosses this boundary.
    """

    def __init__(
        self,
        policy: RemotePolicy,
        *,
        valid_for_s: float = 1.0,
        default_instruction: str | None = None,
        max_image_pixels: int = 1920 * 1080,
    ) -> None:
        if not isinstance(policy, RemotePolicy):
            raise TypeError("policy must implement infer(), reset(), and close()")
        if not np.isfinite(valid_for_s) or not 0.001 <= valid_for_s <= 10:
            raise ValueError("valid_for_s must be between 0.001 and 10 seconds")
        if default_instruction is not None and not default_instruction.strip():
            raise ValueError("default_instruction cannot be blank")
        if max_image_pixels < 1:
            raise ValueError("max_image_pixels must be positive")
        self.policy = policy
        self.valid_for_s = float(valid_for_s)
        self.default_instruction = (
            default_instruction.strip() if default_instruction is not None else None
        )
        self.max_image_pixels = int(max_image_pixels)

    def _decode_rgb(self, encoded: bytes | None) -> np.ndarray:
        if encoded is None or not encoded:
            raise ValueError("VLA inference requires a non-empty front-camera JPEG")
        try:
            with Image.open(io.BytesIO(encoded)) as image:
                width, height = image.size
                if width < 1 or height < 1 or width * height > self.max_image_pixels:
                    raise ValueError(
                        f"camera image {width}x{height} exceeds the configured pixel limit"
                    )
                image.load()
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("front-camera payload is not a valid image") from exc
        return np.ascontiguousarray(rgb)

    @staticmethod
    def _wire_metadata(chunk: Any) -> dict[str, str | int | float | bool | None]:
        metadata: dict[str, str | int | float | bool | None] = {
            "policy": str(chunk.policy_name),
            "action_horizon": int(chunk.horizon),
        }
        if chunk.inference_latency_s is not None:
            metadata["inference_ms"] = float(chunk.inference_latency_s * 1000)
        # Preserve scalar diagnostics without allowing a vendor response to exceed the
        # bounded wire metadata schema or overwrite core provenance.
        for key, value in chunk.info.items():
            wire_key = f"policy.{key}"
            if len(metadata) >= 64 or len(wire_key) > 128:
                break
            if value is None or isinstance(value, (str, bool, Integral, Real)):
                if isinstance(value, Real) and not isinstance(value, (bool, Integral)):
                    value = float(value)
                    if not np.isfinite(value):
                        continue
                metadata[wire_key] = value  # type: ignore[assignment]
        return metadata

    def __call__(self, observation: ObservationV1) -> PolicyOutput:
        if not isinstance(observation, ObservationV1):
            raise TypeError("observation must be an ObservationV1")
        instruction = observation.language_instruction or self.default_instruction
        if instruction is None:
            raise ValueError(
                "wire observation has no language instruction and no default was configured"
            )
        context = {
            "robot_id": observation.robot_id,
            "joint_velocity": observation.joint_velocity,
            "command_velocity": observation.command_velocity,
            "imu_quaternion": observation.imu_quaternion,
            "imu_gyro": observation.imu_gyro,
            "imu_acceleration": observation.imu_acceleration,
            "battery_voltage": observation.battery_voltage,
            "status": dict(observation.status),
        }
        policy_observation = PolicyObservation(
            state_rad=np.asarray(observation.joint_position, dtype=np.float32),
            rgb=self._decode_rgb(observation.image_jpeg),
            instruction=instruction,
            timestamp_ns=observation.monotonic_ns,
            observation_seq=observation.sequence,
            context=context,
        )
        chunk = self.policy.infer(policy_observation)
        if chunk.based_on_observation_seq != observation.sequence:
            raise ValueError("remote policy action chunk references the wrong observation")
        return PolicyOutput(
            targets=tuple(tuple(float(value) for value in row) for row in chunk.actions_rad),
            control_period_s=chunk.dt_s,
            valid_for_s=self.valid_for_s,
            metadata=self._wire_metadata(chunk),
        )

    def reset(self) -> None:
        self.policy.reset()

    def close(self) -> None:
        self.policy.close()
