from __future__ import annotations

import msgpack
import numpy as np
import pytest

from sesame_ml.transport import (
    ActionChunkV1,
    ObservationV1,
    PolicyOutput,
    ProtocolError,
    decode_action_chunk,
    decode_observation,
    encode_action_chunk,
    encode_observation,
)

JOINTS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def test_observation_round_trip_preserves_binary_camera_and_optional_modalities() -> None:
    observation = ObservationV1(
        robot_id="sesame-001",
        sequence=42,
        monotonic_ns=123_456_789,
        joint_position=JOINTS,
        joint_velocity=tuple(-value for value in JOINTS),
        command_velocity=(0.1, -0.2, 0.3),
        imu_quaternion=(1.0, 0.0, 0.0, 0.0),
        imu_gyro=(0.01, -0.02, 0.03),
        imu_acceleration=(0.0, 0.0, 9.81),
        language_instruction="walk to the red marker",
        battery_voltage=7.6,
        image_jpeg=b"\xff\xd8jpeg\xff\xd9",
        status={"camera_ok": True, "wifi_rssi_dbm": -54},
    )

    decoded = decode_observation(encode_observation(observation))

    assert decoded == observation
    assert decoded.image_jpeg == b"\xff\xd8jpeg\xff\xd9"


def test_action_chunk_round_trip_is_exact() -> None:
    chunk = ActionChunkV1(
        robot_id="sesame-001",
        chunk_id=9,
        based_on_observation_sequence=42,
        created_monotonic_ns=987_654_321,
        control_period_s=0.02,
        targets=(JOINTS, tuple(value + 0.05 for value in JOINTS)),
        valid_for_s=0.3,
        metadata={"policy": "pi0.5", "inference_ms": 37.2},
    )

    assert decode_action_chunk(encode_action_chunk(chunk)) == chunk


def test_policy_types_normalize_numpy_outputs_to_wire_native_numbers() -> None:
    observation = ObservationV1(
        robot_id="sesame-001",
        sequence=np.int64(1),
        monotonic_ns=np.int64(10),
        joint_position=np.arange(8, dtype=np.float32) / 10,
        command_velocity=np.asarray([0.1, 0.0, -0.1], dtype=np.float32),
        status={"temperature_c": np.float32(42.5)},
    )
    output = PolicyOutput(
        targets=np.zeros((2, 8), dtype=np.float32),
        control_period_s=np.float32(0.02),
        valid_for_s=np.float32(0.25),
    )

    assert observation.sequence == 1
    assert all(type(value) is float for value in observation.joint_position)
    assert output.targets == ((0.0,) * 8, (0.0,) * 8)


@pytest.mark.parametrize(
    "field,value",
    [
        ("joint_position", (0.0,) * 7),
        ("joint_position", (float("nan"),) + (0.0,) * 7),
        ("sequence", -1),
        ("image_jpeg", "not-bytes"),
        ("imu_quaternion", (0.0, 0.0, 0.0, 0.0)),
    ],
)
def test_observation_validation_rejects_unsafe_values(field: str, value: object) -> None:
    kwargs = {
        "robot_id": "sesame-001",
        "sequence": 0,
        "monotonic_ns": 1,
        "joint_position": JOINTS,
        field: value,
    }
    with pytest.raises(ProtocolError):
        ObservationV1(**kwargs)


def test_decoder_rejects_wrong_version_and_text_wire_messages() -> None:
    packet = msgpack.packb(
        {
            "type": "observation",
            "version": 2,
            "robot_id": "sesame-001",
            "sequence": 0,
            "monotonic_ns": 1,
            "joint_position": list(JOINTS),
        },
        use_bin_type=True,
    )
    with pytest.raises(ProtocolError, match="unsupported protocol version"):
        decode_observation(packet)
    with pytest.raises(ProtocolError, match="binary"):
        decode_observation("not binary")  # type: ignore[arg-type]
