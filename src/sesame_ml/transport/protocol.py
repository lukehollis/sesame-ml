"""Versioned wire messages for Sesame's policy transport.

The wire format deliberately contains only MessagePack-native values.  In
particular, NumPy arrays are converted to ordinary arrays and camera frames
are carried as encoded JPEG bytes.  This keeps the Orange Pi client small and
allows implementations in languages other than Python.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, TypeAlias

import msgpack

PROTOCOL_VERSION = 1
JOINT_COUNT = 8
MAX_PACKET_BYTES = 4 * 1024 * 1024
MAX_IMAGE_BYTES = 3 * 1024 * 1024
MAX_ACTION_STEPS = 512
MAX_METADATA_ENTRIES = 64

JointVector: TypeAlias = tuple[float, float, float, float, float, float, float, float]
Vector3: TypeAlias = tuple[float, float, float]
Quaternion: TypeAlias = tuple[float, float, float, float]
MetadataValue: TypeAlias = str | int | float | bool | None


class ProtocolError(ValueError):
    """Raised when a packet violates the Sesame transport contract."""


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ProtocolError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"{name} must be finite")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < 2**64:
        raise ProtocolError(f"{name} must be a non-negative integer")
    return int(value)


def _required_string(value: Any, name: str, *, max_length: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > max_length:
        raise ProtocolError(f"{name} must be a non-empty string of at most {max_length} characters")
    return value


def _joint_vector(value: Any, name: str) -> JointVector:
    if isinstance(value, (str, bytes, Mapping)):
        raise ProtocolError(f"{name} must contain exactly {JOINT_COUNT} joints")
    try:
        length = len(value)
    except TypeError as error:
        raise ProtocolError(f"{name} must contain exactly {JOINT_COUNT} joints") from error
    if length != JOINT_COUNT:
        raise ProtocolError(f"{name} must contain exactly {JOINT_COUNT} joints")
    result = tuple(_finite_float(item, f"{name}[{index}]") for index, item in enumerate(value))
    # Catch corrupt policy output while leaving room for calibrated joints whose
    # zero isn't the mechanical center.
    if any(abs(item) > 4.0 * math.pi for item in result):
        raise ProtocolError(f"{name} contains an implausible joint angle")
    return result  # type: ignore[return-value]


def _fixed_vector(value: Any, name: str, length: int) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, Mapping)):
        raise ProtocolError(f"{name} must contain exactly {length} values")
    try:
        actual_length = len(value)
    except TypeError as error:
        raise ProtocolError(f"{name} must contain exactly {length} values") from error
    if actual_length != length:
        raise ProtocolError(f"{name} must contain exactly {length} values")
    return tuple(_finite_float(item, f"{name}[{index}]") for index, item in enumerate(value))


def _metadata(value: Any, name: str) -> dict[str, MetadataValue]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or len(value) > MAX_METADATA_ENTRIES:
        raise ProtocolError(f"{name} must be a map with at most {MAX_METADATA_ENTRIES} entries")
    result: dict[str, MetadataValue] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key or len(key) > 128:
            raise ProtocolError(f"{name} keys must be non-empty strings of at most 128 characters")
        if item is not None and not isinstance(item, (str, Real, Integral, bool)):
            raise ProtocolError(f"{name}[{key!r}] has an unsupported value type")
        if isinstance(item, Real) and not isinstance(item, (bool, Integral)):
            item = float(item)
            if not math.isfinite(item):
                raise ProtocolError(f"{name}[{key!r}] must be finite")
        elif isinstance(item, Integral) and not isinstance(item, bool):
            item = int(item)
            if not -(2**63) <= item < 2**64:
                raise ProtocolError(f"{name}[{key!r}] is outside MessagePack's integer range")
        if isinstance(item, str) and len(item) > 4096:
            raise ProtocolError(f"{name}[{key!r}] is too long")
        result[key] = item
    return result


@dataclass(frozen=True, slots=True)
class ObservationV1:
    """A timestamped robot observation sent to the remote policy host.

    ``monotonic_ns`` is the capture time in the robot's local monotonic clock.
    Monotonic timestamps must never be compared across machines.  The client
    retains this capture time locally and uses the echoed sequence number in an
    action chunk to calculate end-to-end freshness without clock synchronization.
    """

    robot_id: str
    sequence: int
    monotonic_ns: int
    joint_position: JointVector
    joint_velocity: JointVector | None = None
    command_velocity: tuple[float, float, float] | None = None
    imu_quaternion: Quaternion | None = None
    imu_gyro: Vector3 | None = None
    imu_acceleration: Vector3 | None = None
    language_instruction: str | None = None
    battery_voltage: float | None = None
    image_jpeg: bytes | None = None
    status: Mapping[str, MetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "robot_id", _required_string(self.robot_id, "robot_id", max_length=128)
        )
        object.__setattr__(self, "sequence", _nonnegative_int(self.sequence, "sequence"))
        object.__setattr__(
            self, "monotonic_ns", _nonnegative_int(self.monotonic_ns, "monotonic_ns")
        )
        object.__setattr__(
            self, "joint_position", _joint_vector(self.joint_position, "joint_position")
        )
        if self.joint_velocity is not None:
            object.__setattr__(
                self, "joint_velocity", _joint_vector(self.joint_velocity, "joint_velocity")
            )
        if self.command_velocity is not None:
            command = _fixed_vector(self.command_velocity, "command_velocity", 3)
            object.__setattr__(self, "command_velocity", command)
        if self.imu_quaternion is not None:
            quaternion = _fixed_vector(self.imu_quaternion, "imu_quaternion", 4)
            norm = math.sqrt(sum(value * value for value in quaternion))
            if not 0.5 <= norm <= 1.5:
                raise ProtocolError("imu_quaternion norm is implausible")
            object.__setattr__(
                self, "imu_quaternion", tuple(value / norm for value in quaternion)
            )
        if self.imu_gyro is not None:
            object.__setattr__(self, "imu_gyro", _fixed_vector(self.imu_gyro, "imu_gyro", 3))
        if self.imu_acceleration is not None:
            object.__setattr__(
                self,
                "imu_acceleration",
                _fixed_vector(self.imu_acceleration, "imu_acceleration", 3),
            )
        if self.language_instruction is not None:
            if (
                not isinstance(self.language_instruction, str)
                or len(self.language_instruction) > 4096
            ):
                raise ProtocolError(
                    "language_instruction must be a string of at most 4096 characters"
                )
        if self.battery_voltage is not None:
            voltage = _finite_float(self.battery_voltage, "battery_voltage")
            if not 0.0 <= voltage <= 100.0:
                raise ProtocolError("battery_voltage is outside the supported range")
            object.__setattr__(self, "battery_voltage", voltage)
        if self.image_jpeg is not None:
            if not isinstance(self.image_jpeg, bytes):
                raise ProtocolError("image_jpeg must be bytes")
            if len(self.image_jpeg) > MAX_IMAGE_BYTES:
                raise ProtocolError(f"image_jpeg exceeds {MAX_IMAGE_BYTES} bytes")
        object.__setattr__(self, "status", _metadata(self.status, "status"))


@dataclass(frozen=True, slots=True)
class ActionChunkV1:
    """A finite, freshness-bounded sequence of joint-position targets.

    ``valid_for_s`` is measured from the originating observation's robot-local
    capture time.  It therefore includes capture, network, and policy latency.
    The targets begin at chunk receipt and are sampled every ``control_period_s``.
    """

    robot_id: str
    chunk_id: int
    based_on_observation_sequence: int
    created_monotonic_ns: int
    control_period_s: float
    targets: tuple[JointVector, ...]
    valid_for_s: float
    mode: str = "joint_position_rad"
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "robot_id", _required_string(self.robot_id, "robot_id", max_length=128)
        )
        object.__setattr__(self, "chunk_id", _nonnegative_int(self.chunk_id, "chunk_id"))
        object.__setattr__(
            self,
            "based_on_observation_sequence",
            _nonnegative_int(self.based_on_observation_sequence, "based_on_observation_sequence"),
        )
        object.__setattr__(
            self,
            "created_monotonic_ns",
            _nonnegative_int(self.created_monotonic_ns, "created_monotonic_ns"),
        )
        period = _finite_float(self.control_period_s, "control_period_s")
        if not 0.001 <= period <= 1.0:
            raise ProtocolError("control_period_s must be between 0.001 and 1.0 seconds")
        object.__setattr__(self, "control_period_s", period)
        try:
            target_count = len(self.targets)
        except TypeError as error:
            raise ProtocolError(
                f"targets must contain between 1 and {MAX_ACTION_STEPS} steps"
            ) from error
        if isinstance(self.targets, (str, bytes, Mapping)) or not (
            1 <= target_count <= MAX_ACTION_STEPS
        ):
            raise ProtocolError(f"targets must contain between 1 and {MAX_ACTION_STEPS} steps")
        targets = tuple(
            _joint_vector(target, f"targets[{index}]") for index, target in enumerate(self.targets)
        )
        object.__setattr__(self, "targets", targets)
        valid_for = _finite_float(self.valid_for_s, "valid_for_s")
        if not 0.001 <= valid_for <= 10.0:
            raise ProtocolError("valid_for_s must be between 0.001 and 10 seconds")
        object.__setattr__(self, "valid_for_s", valid_for)
        if self.mode != "joint_position_rad":
            raise ProtocolError("only joint_position_rad actions are supported by protocol v1")
        object.__setattr__(self, "metadata", _metadata(self.metadata, "metadata"))


@dataclass(frozen=True, slots=True)
class PolicyOutput:
    """Policy callback result before server-owned sequencing is attached."""

    targets: tuple[JointVector, ...]
    control_period_s: float
    valid_for_s: float
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Reuse ActionChunkV1's strict action validation without duplicating it.
        checked = ActionChunkV1(
            robot_id="validation",
            chunk_id=0,
            based_on_observation_sequence=0,
            created_monotonic_ns=0,
            control_period_s=self.control_period_s,
            targets=self.targets,
            valid_for_s=self.valid_for_s,
            metadata=self.metadata,
        )
        object.__setattr__(self, "targets", checked.targets)
        object.__setattr__(self, "control_period_s", checked.control_period_s)
        object.__setattr__(self, "valid_for_s", checked.valid_for_s)
        object.__setattr__(self, "metadata", checked.metadata)


def encode_observation(message: ObservationV1) -> bytes:
    """Serialize an observation as a bounded MessagePack binary packet."""

    packet = {
        "type": "observation",
        "version": PROTOCOL_VERSION,
        "robot_id": message.robot_id,
        "sequence": message.sequence,
        "monotonic_ns": message.monotonic_ns,
        "joint_position": list(message.joint_position),
        "joint_velocity": list(message.joint_velocity)
        if message.joint_velocity is not None
        else None,
        "command_velocity": list(message.command_velocity)
        if message.command_velocity is not None
        else None,
        "imu_quaternion": list(message.imu_quaternion)
        if message.imu_quaternion is not None
        else None,
        "imu_gyro": list(message.imu_gyro) if message.imu_gyro is not None else None,
        "imu_acceleration": list(message.imu_acceleration)
        if message.imu_acceleration is not None
        else None,
        "language_instruction": message.language_instruction,
        "battery_voltage": message.battery_voltage,
        "image_jpeg": message.image_jpeg,
        "status": dict(message.status),
    }
    return _pack(packet)


def decode_observation(packet: bytes) -> ObservationV1:
    """Deserialize and validate an observation packet."""

    value = _unpack(packet)
    _check_envelope(value, "observation")
    try:
        return ObservationV1(
            robot_id=value["robot_id"],
            sequence=value["sequence"],
            monotonic_ns=value["monotonic_ns"],
            joint_position=value["joint_position"],
            joint_velocity=value.get("joint_velocity"),
            command_velocity=value.get("command_velocity"),
            imu_quaternion=value.get("imu_quaternion"),
            imu_gyro=value.get("imu_gyro"),
            imu_acceleration=value.get("imu_acceleration"),
            language_instruction=value.get("language_instruction"),
            battery_voltage=value.get("battery_voltage"),
            image_jpeg=value.get("image_jpeg"),
            status=value.get("status", {}),
        )
    except KeyError as error:
        raise ProtocolError(f"observation is missing required field {error.args[0]!r}") from error


def encode_action_chunk(message: ActionChunkV1) -> bytes:
    """Serialize an action chunk as a bounded MessagePack binary packet."""

    packet = {
        "type": "action_chunk",
        "version": PROTOCOL_VERSION,
        "robot_id": message.robot_id,
        "chunk_id": message.chunk_id,
        "based_on_observation_sequence": message.based_on_observation_sequence,
        "created_monotonic_ns": message.created_monotonic_ns,
        "control_period_s": message.control_period_s,
        "targets": [list(target) for target in message.targets],
        "valid_for_s": message.valid_for_s,
        "mode": message.mode,
        "metadata": dict(message.metadata),
    }
    return _pack(packet)


def decode_action_chunk(packet: bytes) -> ActionChunkV1:
    """Deserialize and validate an action-chunk packet."""

    value = _unpack(packet)
    _check_envelope(value, "action_chunk")
    try:
        return ActionChunkV1(
            robot_id=value["robot_id"],
            chunk_id=value["chunk_id"],
            based_on_observation_sequence=value["based_on_observation_sequence"],
            created_monotonic_ns=value["created_monotonic_ns"],
            control_period_s=value["control_period_s"],
            targets=value["targets"],
            valid_for_s=value["valid_for_s"],
            mode=value.get("mode", ""),
            metadata=value.get("metadata", {}),
        )
    except KeyError as error:
        raise ProtocolError(f"action chunk is missing required field {error.args[0]!r}") from error


def _pack(value: Mapping[str, Any]) -> bytes:
    packet = msgpack.packb(value, use_bin_type=True)
    if len(packet) > MAX_PACKET_BYTES:
        raise ProtocolError(f"encoded packet exceeds {MAX_PACKET_BYTES} bytes")
    return packet


def _unpack(packet: bytes) -> dict[str, Any]:
    if not isinstance(packet, bytes):
        raise ProtocolError("wire messages must be binary")
    if len(packet) > MAX_PACKET_BYTES:
        raise ProtocolError(f"packet exceeds {MAX_PACKET_BYTES} bytes")
    try:
        value = msgpack.unpackb(
            packet,
            raw=False,
            strict_map_key=True,
            max_str_len=64 * 1024,
            max_bin_len=MAX_IMAGE_BYTES,
            max_array_len=MAX_ACTION_STEPS * JOINT_COUNT + 64,
            max_map_len=MAX_METADATA_ENTRIES + 32,
        )
    except (
        ValueError,
        TypeError,
        msgpack.ExtraData,
        msgpack.FormatError,
        msgpack.StackError,
    ) as error:
        raise ProtocolError(f"invalid MessagePack packet: {error}") from error
    if not isinstance(value, dict):
        raise ProtocolError("packet root must be a map")
    return value


def _check_envelope(value: Mapping[str, Any], expected_type: str) -> None:
    if value.get("type") != expected_type:
        raise ProtocolError(f"expected {expected_type!r} packet")
    version = value.get("version")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version {version!r}")
