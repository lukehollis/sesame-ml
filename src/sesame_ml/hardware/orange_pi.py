"""Production Orange Pi runtime for one-board camera/IMU/servo control.

Linux SBC GPIO is not a reliable source of eight simultaneous 50 Hz hobby-servo PWM
signals. This adapter uses an inexpensive PCA9685 over I2C, leaving the Orange Pi to run
camera capture, Wi-Fi, safety checks, and the 50 Hz action interpolation loop.
"""

from __future__ import annotations

import asyncio
import io
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import yaml
from PIL import Image

from sesame_ml.constants import JOINT_LIMITS_RAD, JOINT_NAMES, STAND_ANGLES_RAD
from sesame_ml.transport import ActionSample, ControlState, ObservationV1, RobotWebSocketClient


@runtime_checkable
class Camera(Protocol):
    def capture_rgb(self) -> np.ndarray: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class ImuReading:
    quaternion_wxyz: tuple[float, float, float, float]
    gyro_rad_s: tuple[float, float, float]
    acceleration_m_s2: tuple[float, float, float]


@runtime_checkable
class ImuReader(Protocol):
    def read(self) -> ImuReading: ...

    def close(self) -> None: ...


class V4L2Camera:
    """Persistent V4L2 camera capture through OpenCV's Linux backend."""

    def __init__(
        self,
        device: int | str = 0,
        *,
        width: int = 320,
        height: int = 240,
        fps: int = 15,
    ) -> None:
        try:
            import cv2
        except ImportError as error:  # pragma: no cover - hardware-only dependency path
            raise ImportError("V4L2 capture requires `pip install sesame-ml[hardware]`") from error
        self._cv2 = cv2
        backend = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else cv2.CAP_ANY
        self._capture = cv2.VideoCapture(device, backend)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._capture.set(cv2.CAP_PROP_FPS, fps)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"Could not open V4L2 camera {device!r}")
        self._lock = threading.Lock()

    def capture_rgb(self) -> np.ndarray:
        with self._lock:
            ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError("V4L2 camera frame capture failed")
        return self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        with self._lock:
            self._capture.release()


class BNO085ImuReader:
    """BNO085 absolute orientation, gyro, and acceleration over a Linux I2C bus."""

    def __init__(self, bus_number: int = 3, *, address: int = 0x4A) -> None:
        try:
            from adafruit_bno08x import (
                BNO_REPORT_ACCELEROMETER,
                BNO_REPORT_GYROSCOPE,
                BNO_REPORT_ROTATION_VECTOR,
            )
            from adafruit_bno08x.i2c import BNO08X_I2C
            from adafruit_extended_bus import ExtendedI2C
        except ImportError as error:  # pragma: no cover - hardware-only dependency path
            raise ImportError(
                "BNO085 support requires `pip install sesame-ml[hardware]`"
            ) from error
        self._i2c = ExtendedI2C(bus_number)
        self._sensor = BNO08X_I2C(self._i2c, address=address)
        self._sensor.enable_feature(BNO_REPORT_ROTATION_VECTOR)
        self._sensor.enable_feature(BNO_REPORT_GYROSCOPE)
        self._sensor.enable_feature(BNO_REPORT_ACCELEROMETER)

    def read(self) -> ImuReading:
        # CircuitPython exposes quaternion as (i, j, k, real), while Sesame's wire
        # convention and MuJoCo use (w, x, y, z).
        x, y, z, w = self._sensor.quaternion
        gx, gy, gz = self._sensor.gyro
        ax, ay, az = self._sensor.acceleration
        return ImuReading(
            quaternion_wxyz=(float(w), float(x), float(y), float(z)),
            gyro_rad_s=(float(gx), float(gy), float(gz)),
            acceleration_m_s2=(float(ax), float(ay), float(az)),
        )

    def close(self) -> None:
        deinit = getattr(self._i2c, "deinit", None)
        if callable(deinit):
            deinit()


@dataclass(frozen=True)
class ServoCalibration:
    minimum_rad: np.ndarray
    maximum_rad: np.ndarray
    direction: np.ndarray
    subtrim_rad: np.ndarray
    minimum_pulse_us: int = 732
    maximum_pulse_us: int = 2929
    maximum_step_rad: float = np.deg2rad(12.0)
    maximum_speed_rad_s: float = np.deg2rad(600.0)
    action_timeout_s: float = 0.250
    disable_timeout_s: float = 1.000

    def __post_init__(self) -> None:
        for name in ("minimum_rad", "maximum_rad", "direction", "subtrim_rad"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (8,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite 8-vector")
            object.__setattr__(self, name, value)
        if np.any(self.minimum_rad >= self.maximum_rad):
            raise ValueError("minimum joint limits must be below maximum limits")
        if not np.all(np.isin(self.direction, (-1, 1))):
            raise ValueError("direction entries must be -1 or +1")
        if not 300 <= self.minimum_pulse_us < self.maximum_pulse_us <= 3000:
            raise ValueError("pulse calibration is outside a plausible servo range")
        if self.maximum_step_rad <= 0:
            raise ValueError("maximum_step_rad must be positive")
        if self.maximum_speed_rad_s <= 0:
            raise ValueError("maximum_speed_rad_s must be positive")
        if not 0.05 <= self.action_timeout_s <= 2.0:
            raise ValueError("action_timeout_s must be between 0.05 and 2 seconds")
        if not self.action_timeout_s <= self.disable_timeout_s <= 10.0:
            raise ValueError("disable_timeout_s must be at least action_timeout_s and <= 10")

    @classmethod
    def defaults(cls) -> ServoCalibration:
        return cls(
            minimum_rad=JOINT_LIMITS_RAD[:, 0],
            maximum_rad=JOINT_LIMITS_RAD[:, 1],
            direction=np.ones(8),
            subtrim_rad=np.zeros(8),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ServoCalibration:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if value.get("calibrated") is not True:
            raise ValueError(
                "calibration file must set calibrated: true after physical endpoint, "
                "direction, subtrim, PWM, and safety validation"
            )
        order = value.get("joint_order")
        if order != list(JOINT_NAMES):
            raise ValueError(f"calibration joint_order must be {list(JOINT_NAMES)}")
        limits = value["limits_degrees"]
        direction = value.get("direction", {})
        subtrim = value.get("subtrim_degrees", {})
        pwm = value.get("pwm", {})
        safety = value.get("safety", {})
        return cls(
            minimum_rad=np.deg2rad([limits[name][0] for name in JOINT_NAMES]),
            maximum_rad=np.deg2rad([limits[name][1] for name in JOINT_NAMES]),
            direction=np.asarray([direction.get(name, 1) for name in JOINT_NAMES]),
            subtrim_rad=np.deg2rad([subtrim.get(name, 0) for name in JOINT_NAMES]),
            minimum_pulse_us=int(pwm.get("minimum_pulse_us", 732)),
            maximum_pulse_us=int(pwm.get("maximum_pulse_us", 2929)),
            maximum_step_rad=np.deg2rad(float(safety.get("maximum_step_degrees", 12))),
            maximum_speed_rad_s=np.deg2rad(
                float(safety.get("maximum_speed_degrees_s", 600))
            ),
            action_timeout_s=float(safety.get("action_timeout_ms", 250)) / 1_000.0,
            disable_timeout_s=float(safety.get("disable_timeout_ms", 1_000)) / 1_000.0,
        )


class PCA9685ServoController:
    """Eight-channel PCA9685 writer with calibration, slew limiting, and hard disable."""

    MODE1 = 0x00
    MODE2 = 0x01
    LED0_ON_L = 0x06
    ALL_LED_ON_L = 0xFA
    PRESCALE = 0xFE
    RESTART = 0x80
    SLEEP = 0x10
    AI = 0x20
    OUTDRV = 0x04

    def __init__(
        self,
        bus: Any,
        *,
        address: int = 0x40,
        frequency_hz: float = 50.0,
        calibration: ServoCalibration | None = None,
        oscillator_hz: float = 25_000_000.0,
    ) -> None:
        if not 0x03 <= address <= 0x77:
            raise ValueError("invalid I2C address")
        if not 40 <= frequency_hz <= 60:
            raise ValueError("hobby servo PWM frequency must be between 40 and 60 Hz")
        self.bus = bus
        self.address = address
        self.frequency_hz = float(frequency_hz)
        self.calibration = calibration or ServoCalibration.defaults()
        self.oscillator_hz = oscillator_hz
        self._last_target = STAND_ANGLES_RAD.copy()
        self._enabled = False
        self._lock = threading.Lock()
        self._initialize()

    @classmethod
    def open(
        cls,
        bus_number: int = 3,
        **kwargs: Any,
    ) -> PCA9685ServoController:
        try:
            from smbus2 import SMBus
        except ImportError as error:  # pragma: no cover - hardware-only dependency path
            raise ImportError(
                "PCA9685 control requires `pip install sesame-ml[hardware]`"
            ) from error
        return cls(SMBus(bus_number), **kwargs)

    def _initialize(self) -> None:
        with self._lock:
            self.bus.write_byte_data(self.address, self.MODE1, self.AI)
            self.bus.write_byte_data(self.address, self.MODE2, self.OUTDRV)
            old_mode = self.bus.read_byte_data(self.address, self.MODE1)
            prescale = int(round(self.oscillator_hz / (4096 * self.frequency_hz)) - 1)
            prescale = int(np.clip(prescale, 3, 255))
            self.bus.write_byte_data(self.address, self.MODE1, (old_mode & 0x7F) | self.SLEEP)
            self.bus.write_byte_data(self.address, self.PRESCALE, prescale)
            self.bus.write_byte_data(self.address, self.MODE1, old_mode & ~self.SLEEP)
            time.sleep(0.0006)
            self.bus.write_byte_data(self.address, self.MODE1, (old_mode | self.AI | self.RESTART))
            self._disable_locked()

    def _angle_to_pulse_us(self, angle_rad: np.ndarray) -> np.ndarray:
        calibrated = (
            np.pi / 2
            + self.calibration.direction * (angle_rad - np.pi / 2)
            + self.calibration.subtrim_rad
        )
        fraction = np.clip(calibrated / np.pi, 0.0, 1.0)
        return self.calibration.minimum_pulse_us + fraction * (
            self.calibration.maximum_pulse_us - self.calibration.minimum_pulse_us
        )

    def _pulse_to_count(self, pulse_us: np.ndarray) -> np.ndarray:
        period_us = 1_000_000.0 / self.frequency_hz
        return np.clip(np.rint(pulse_us / period_us * 4096), 1, 4095).astype(np.int32)

    def write_targets(self, target_rad: np.ndarray | tuple[float, ...]) -> np.ndarray:
        target = np.asarray(target_rad, dtype=np.float64)
        if target.shape != (8,) or not np.all(np.isfinite(target)):
            raise ValueError("target_rad must be a finite 8-vector")
        target = np.clip(target, self.calibration.minimum_rad, self.calibration.maximum_rad)
        with self._lock:
            target = self._last_target + np.clip(
                target - self._last_target,
                -min(
                    self.calibration.maximum_step_rad,
                    self.calibration.maximum_speed_rad_s / self.frequency_hz,
                ),
                min(
                    self.calibration.maximum_step_rad,
                    self.calibration.maximum_speed_rad_s / self.frequency_hz,
                ),
            )
            counts = self._pulse_to_count(self._angle_to_pulse_us(target))
            # Four bytes per channel: on-low, on-high, off-low, off-high. Individual writes
            # avoid depending on controller-specific SMBus block-length behavior.
            for channel, count in enumerate(counts):
                register = self.LED0_ON_L + 4 * channel
                self.bus.write_i2c_block_data(
                    self.address,
                    register,
                    [0, 0, int(count) & 0xFF, (int(count) >> 8) & 0x0F],
                )
            self._last_target = target
            self._enabled = True
            return target.copy()

    def _disable_locked(self) -> None:
        # Full-off bit in ALL_LED_OFF_H disables all outputs in one I2C transaction.
        self.bus.write_i2c_block_data(self.address, self.ALL_LED_ON_L, [0, 0, 0, 0x10])
        self._enabled = False

    def disable(self) -> None:
        with self._lock:
            self._disable_locked()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def last_target(self) -> np.ndarray:
        return self._last_target.copy()

    def close(self) -> None:
        self.disable()
        close = getattr(self.bus, "close", None)
        if callable(close):
            close()


class OrangePiRobotRuntime:
    """Camera/IMU source and software-watchdog-bounded PCA9685 action sink."""

    def __init__(
        self,
        *,
        robot_id: str,
        camera: Camera,
        servos: PCA9685ServoController,
        imu: ImuReader,
        jpeg_quality: int = 82,
        battery_reader: Any | None = None,
        fallback_behavior: str = "disable",
    ) -> None:
        if fallback_behavior not in {"disable", "stand"}:
            raise ValueError("fallback_behavior must be 'disable' or 'stand'")
        self.robot_id = robot_id
        self.camera = camera
        self.servos = servos
        self.imu = imu
        self.jpeg_quality = jpeg_quality
        self.battery_reader = battery_reader
        self.fallback_behavior = fallback_behavior
        self.command_velocity = (0.0, 0.0, 0.0)
        self.language_instruction = "Stand safely and await a command."
        self._fallback_active = True
        self._fallback_started_s: float | None = time.monotonic()
        self._last_action_sequence = -1

    def _jpeg(self) -> bytes:
        frame = self.camera.capture_rgb()
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise RuntimeError("camera must return HxWx3 uint8 RGB")
        output = io.BytesIO()
        Image.fromarray(frame).save(output, "JPEG", quality=self.jpeg_quality, optimize=False)
        return output.getvalue()

    def observation_source(self, sequence: int, monotonic_ns: int) -> ObservationV1:
        imu = self.imu.read()
        battery = float(self.battery_reader()) if self.battery_reader is not None else None
        return ObservationV1(
            robot_id=self.robot_id,
            sequence=sequence,
            monotonic_ns=monotonic_ns,
            joint_position=tuple(float(value) for value in self.servos.last_target),
            joint_velocity=None,
            command_velocity=self.command_velocity,
            imu_quaternion=imu.quaternion_wxyz,
            imu_gyro=imu.gyro_rad_s,
            imu_acceleration=imu.acceleration_m_s2,
            language_instruction=self.language_instruction,
            battery_voltage=battery,
            image_jpeg=self._jpeg(),
            status={
                "servo_outputs_enabled": self.servos.enabled,
                "fallback_active": self._fallback_active,
                "last_action_sequence": self._last_action_sequence,
            },
        )

    def action_sink(self, sample: ActionSample) -> None:
        if sample.state is ControlState.FALLBACK or sample.target is None:
            if not self._fallback_active or self._fallback_started_s is None:
                self._fallback_started_s = time.monotonic()
            self._fallback_active = True
            within_disable_grace = (
                time.monotonic() - self._fallback_started_s
                < self.servos.calibration.disable_timeout_s
            )
            if self.fallback_behavior == "stand" and within_disable_grace:
                target = sample.target if sample.target is not None else STAND_ANGLES_RAD
                self.servos.write_targets(target)
            else:
                self.servos.disable()
            return
        self.servos.write_targets(sample.target)
        self._fallback_active = False
        self._fallback_started_s = None
        self._last_action_sequence = sample.based_on_observation_sequence or 0

    def close(self) -> None:
        self.servos.close()
        self.camera.close()
        self.imu.close()


async def run_robot_client(
    runtime: OrangePiRobotRuntime,
    *,
    policy_uri: str,
    observation_hz: float = 15.0,
) -> None:
    """Run the outbound Wi-Fi client until SIGINT/SIGTERM, always disabling on exit."""

    safe_fallback = (
        tuple(float(value) for value in STAND_ANGLES_RAD)
        if runtime.fallback_behavior == "stand"
        else None
    )
    client = RobotWebSocketClient(
        uri=policy_uri,
        robot_id=runtime.robot_id,
        observation_source=runtime.observation_source,
        action_sink=runtime.action_sink,
        observation_hz=observation_hz,
        control_hz=50,
        watchdog_timeout_s=runtime.servos.calibration.action_timeout_s,
        safe_fallback=safe_fallback,
    )
    loop = asyncio.get_running_loop()
    for signal_name in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signal_name, client.stop)
        except NotImplementedError:  # pragma: no cover - Unix Orange Pi supports handlers.
            pass
    try:
        await client.run()
    finally:
        runtime.close()
