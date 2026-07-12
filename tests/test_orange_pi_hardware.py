from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sesame_ml.constants import STAND_ANGLES_RAD
from sesame_ml.hardware import (
    ImuReading,
    OrangePiRobotRuntime,
    PCA9685ServoController,
    ServoCalibration,
)
from sesame_ml.transport import ActionSample, ControlState


class FakeBus:
    def __init__(self) -> None:
        self.bytes: dict[int, int] = {}
        self.blocks: list[tuple[int, list[int]]] = []
        self.closed = False

    def write_byte_data(self, address: int, register: int, value: int) -> None:
        del address
        self.bytes[register] = value

    def read_byte_data(self, address: int, register: int) -> int:
        del address
        return self.bytes.get(register, 0)

    def write_i2c_block_data(
        self, address: int, register: int, values: list[int]
    ) -> None:
        del address
        self.blocks.append((register, values))

    def close(self) -> None:
        self.closed = True


class FakeCamera:
    def capture_rgb(self) -> np.ndarray:
        return np.zeros((24, 32, 3), dtype=np.uint8)

    def close(self) -> None:
        pass


class FakeImu:
    def read(self) -> ImuReading:
        return ImuReading((1, 0, 0, 0), (0, 0, 0), (0, 0, 9.81))

    def close(self) -> None:
        pass


def test_pca9685_writes_all_channels_and_hard_disables() -> None:
    bus = FakeBus()
    servos = PCA9685ServoController(bus)
    initial_blocks = len(bus.blocks)
    written = servos.write_targets(STAND_ANGLES_RAD)
    assert np.allclose(written, STAND_ANGLES_RAD)
    assert len(bus.blocks) == initial_blocks + 8
    assert servos.enabled
    servos.disable()
    assert bus.blocks[-1] == (servos.ALL_LED_ON_L, [0, 0, 0, 0x10])
    assert not servos.enabled


def test_orange_runtime_fails_closed_on_watchdog_sample() -> None:
    bus = FakeBus()
    servos = PCA9685ServoController(bus)
    runtime = OrangePiRobotRuntime(
        robot_id="sesame-zero",
        camera=FakeCamera(),
        servos=servos,
        imu=FakeImu(),
        fallback_behavior="disable",
    )
    runtime.action_sink(
        ActionSample(
            target=tuple(float(value) for value in STAND_ANGLES_RAD),
            state=ControlState.ACTIVE,
            reason="test",
            based_on_observation_sequence=7,
        )
    )
    assert servos.enabled
    observation = runtime.observation_source(2, 100)
    assert observation.image_jpeg is not None
    assert observation.status["last_action_sequence"] == 7
    runtime.action_sink(
        ActionSample(target=None, state=ControlState.FALLBACK, reason="watchdog_expired")
    )
    assert not servos.enabled


def test_calibration_loads_every_safety_limit_and_enforces_speed(tmp_path) -> None:
    calibration_path = tmp_path / "robot.yaml"
    calibration_path.write_text(
        """
joint_order: [R1, R2, L1, L2, R4, R3, L3, L4]
calibrated: true
limits_degrees:
  R1: [100, 160]
  R2: [20, 80]
  L1: [20, 80]
  L2: [100, 160]
  R4: [0, 40]
  R3: [140, 180]
  L3: [0, 40]
  L4: [140, 180]
direction: {R1: 1, R2: 1, L1: 1, L2: 1, R4: 1, R3: 1, L3: 1, L4: 1}
subtrim_degrees: {R1: 0, R2: 0, L1: 0, L2: 0, R4: 0, R3: 0, L3: 0, L4: 0}
pwm: {minimum_pulse_us: 900, maximum_pulse_us: 2100}
safety:
  maximum_step_degrees: 30
  maximum_speed_degrees_s: 50
  action_timeout_ms: 300
  disable_timeout_ms: 1200
""".strip()
        + "\n"
    )
    calibration = ServoCalibration.from_yaml(calibration_path)
    assert calibration.maximum_speed_rad_s == pytest.approx(np.deg2rad(50))
    assert calibration.action_timeout_s == pytest.approx(0.3)
    assert calibration.disable_timeout_s == pytest.approx(1.2)

    controller = PCA9685ServoController(FakeBus(), calibration=calibration)
    requested = STAND_ANGLES_RAD.copy()
    requested[0] += 0.5
    written = controller.write_targets(requested)
    assert written[0] - STAND_ANGLES_RAD[0] == pytest.approx(
        calibration.maximum_speed_rad_s / controller.frequency_hz
    )
    controller.close()


def test_uncalibrated_template_is_rejected() -> None:
    template = (
        Path(__file__).parents[1]
        / "src/sesame_ml/assets/calibration/default.yaml"
    )
    with pytest.raises(ValueError, match="calibrated: true"):
        ServoCalibration.from_yaml(template)
