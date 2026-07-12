"""Crash-tolerant, lossless on-disk episodes for Sesame.

The format is intentionally independent of any training framework. An episode is a
directory with an atomically updated JSON manifest and ordered, checksum-protected NPZ
chunks. RGB is stored as raw uint8 arrays inside lossless ZIP/DEFLATE containers; it is
never passed through JPEG or video compression. This makes the files suitable as the
source of truth from which lossy training formats can be regenerated.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from sesame_ml.constants import JOINT_LIMITS_RAD, JOINT_NAMES, validate_joint_vector

FORMAT_NAME = "sesame-lossless-episode"
SCHEMA_VERSION = 2
MANIFEST_NAME = "manifest.json"
_CHUNK_KEYS = (
    "state_rad",
    "action_rad",
    "reward",
    "terminated",
    "truncated",
    "timestamp_ns",
    "rgb",
    "imu_quaternion",
    "imu_gyro",
    "imu_acceleration",
    "command_velocity",
    "next_success",
    "next_fallen",
    "next_obstacle_contact",
    "next_out_of_bounds",
    "next_goal_distance_m",
)


@dataclass(frozen=True, slots=True)
class EpisodeStep:
    """One time-aligned robot transition in canonical firmware joint order."""

    state_rad: np.ndarray
    rgb: np.ndarray
    action_rad: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    timestamp_ns: int
    imu_quaternion: np.ndarray
    imu_gyro: np.ndarray
    imu_acceleration: np.ndarray
    command_velocity: np.ndarray
    next_success: bool
    next_fallen: bool
    next_obstacle_contact: bool
    next_out_of_bounds: bool
    next_goal_distance_m: float


@dataclass(frozen=True, slots=True)
class Episode:
    """A materialized episode. Arrays retain the exact canonical stored values."""

    state_rad: np.ndarray
    rgb: np.ndarray
    action_rad: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    timestamp_ns: np.ndarray
    imu_quaternion: np.ndarray
    imu_gyro: np.ndarray
    imu_acceleration: np.ndarray
    command_velocity: np.ndarray
    next_success: np.ndarray
    next_fallen: np.ndarray
    next_obstacle_contact: np.ndarray
    next_out_of_bounds: np.ndarray
    next_goal_distance_m: np.ndarray
    instruction: str
    metadata: Mapping[str, Any]
    episode_id: str

    def __len__(self) -> int:
        return int(self.state_rad.shape[0])


def _json_value(value: Any, *, field: str) -> Any:
    """Round-trip a value through strict JSON, rejecting ambiguous metadata."""

    try:
        encoded = json.dumps(value, allow_nan=False, sort_keys=True)
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain only finite JSON values") from exc


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        # Make the rename durable on filesystems which support directory fsync.
        try:
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            pass
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_image(image: np.ndarray) -> np.ndarray:
    result = np.asarray(image)
    if result.dtype != np.uint8:
        raise ValueError(f"rgb must have dtype uint8, got {result.dtype}")
    if result.ndim != 3 or result.shape[-1] != 3:
        raise ValueError(f"rgb must have shape (height, width, 3), got {result.shape}")
    if result.shape[0] < 1 or result.shape[1] < 1:
        raise ValueError("rgb height and width must be positive")
    return np.ascontiguousarray(result)


def _canonical_action(action_rad: np.ndarray | list[float]) -> np.ndarray:
    result = validate_joint_vector(action_rad, name="action_rad")
    low, high = JOINT_LIMITS_RAD.T
    tolerance = 1e-6
    if np.any(result < low - tolerance) or np.any(result > high + tolerance):
        raise ValueError("action_rad contains a target outside the global physical servo range")
    return np.clip(result, low, high).astype(np.float32)


def _context_vector(
    value: np.ndarray | list[float] | tuple[float, ...] | None,
    *,
    name: str,
    size: int,
    default: tuple[float, ...],
) -> np.ndarray:
    result = np.asarray(default if value is None else value, dtype=np.float32)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with shape ({size},)")
    return result.copy()


class EpisodeWriter:
    """Append a lossless episode without retaining the entire recording in RAM.

    Completed chunks and the manifest are committed atomically. If the process dies,
    already committed chunks remain checksummed and the manifest stays in ``recording``
    state instead of masquerading as a complete training episode.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        instruction: str,
        metadata: Mapping[str, Any] | None = None,
        episode_id: str | None = None,
        chunk_size: int = 128,
        timestamp_clock: str = "monotonic",
        overwrite: bool = False,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if not isinstance(timestamp_clock, str) or not timestamp_clock.strip():
            raise ValueError("timestamp_clock must be a non-empty string")
        if self.path.exists():
            if not overwrite:
                raise FileExistsError(self.path)
            if self.path.is_symlink() or not self.path.is_dir():
                raise ValueError(f"Refusing to overwrite non-directory episode path: {self.path}")
            shutil.rmtree(self.path)
        self.path.mkdir(parents=True)

        self._chunk_size = int(chunk_size)
        self._buffer: dict[str, list[Any]] = {key: [] for key in _CHUNK_KEYS}
        self._last_timestamp_ns: int | None = None
        self._terminal_seen = False
        self._closed = False
        self._manifest: dict[str, Any] = {
            "format": FORMAT_NAME,
            "schema_version": SCHEMA_VERSION,
            "status": "recording",
            "episode_id": episode_id or uuid.uuid4().hex,
            "instruction": instruction.strip(),
            "metadata": _json_value(dict(metadata or {}), field="metadata"),
            "created_at": datetime.now(UTC).isoformat(),
            "timestamp_clock": timestamp_clock.strip(),
            "joint_names": list(JOINT_NAMES),
            "state_representation": "absolute_servo_position_rad",
            "action_representation": "absolute_servo_target_rad",
            "rgb": None,
            "step_count": 0,
            "chunks": [],
        }
        _atomic_write_json(self.path / MANIFEST_NAME, self._manifest)

    @property
    def step_count(self) -> int:
        return int(self._manifest["step_count"]) + len(self._buffer["state_rad"])

    def append(
        self,
        *,
        state_rad: np.ndarray | list[float],
        rgb: np.ndarray,
        action_rad: np.ndarray | list[float],
        reward: float = 0.0,
        terminated: bool = False,
        truncated: bool = False,
        timestamp_ns: int | None = None,
        imu_quaternion: np.ndarray | list[float] | tuple[float, ...] | None = None,
        imu_gyro: np.ndarray | list[float] | tuple[float, ...] | None = None,
        imu_acceleration: np.ndarray | list[float] | tuple[float, ...] | None = None,
        command_velocity: np.ndarray | list[float] | tuple[float, ...] | None = None,
        next_success: bool = False,
        next_fallen: bool = False,
        next_obstacle_contact: bool = False,
        next_out_of_bounds: bool = False,
        next_goal_distance_m: float = 0.0,
    ) -> None:
        """Append one aligned transition and commit automatically at chunk boundaries."""

        if self._closed:
            raise RuntimeError("episode writer is closed")
        if self._terminal_seen:
            raise RuntimeError("cannot append after a terminated or truncated step")
        state = validate_joint_vector(state_rad, name="state_rad").astype(np.float32)
        action = _canonical_action(action_rad)
        image = _canonical_image(rgb)
        scalar_reward = float(reward)
        if not np.isfinite(scalar_reward):
            raise ValueError("reward must be finite")
        if not isinstance(terminated, (bool, np.bool_)):
            raise TypeError("terminated must be boolean")
        if not isinstance(truncated, (bool, np.bool_)):
            raise TypeError("truncated must be boolean")
        if terminated and truncated:
            raise ValueError("a step cannot be both terminated and truncated")

        quaternion = _context_vector(
            imu_quaternion,
            name="imu_quaternion",
            size=4,
            default=(1.0, 0.0, 0.0, 0.0),
        )
        quaternion_norm = float(np.linalg.norm(quaternion))
        if not 0.5 <= quaternion_norm <= 1.5:
            raise ValueError("imu_quaternion norm must be between 0.5 and 1.5")
        quaternion /= quaternion_norm
        gyro = _context_vector(
            imu_gyro, name="imu_gyro", size=3, default=(0.0, 0.0, 0.0)
        )
        acceleration = _context_vector(
            imu_acceleration,
            name="imu_acceleration",
            size=3,
            default=(0.0, 0.0, 0.0),
        )
        command = _context_vector(
            command_velocity,
            name="command_velocity",
            size=3,
            default=(0.0, 0.0, 0.0),
        )
        labels = {
            "next_success": next_success,
            "next_fallen": next_fallen,
            "next_obstacle_contact": next_obstacle_contact,
            "next_out_of_bounds": next_out_of_bounds,
        }
        for name, value in labels.items():
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be boolean")
        goal_distance = float(next_goal_distance_m)
        if not np.isfinite(goal_distance) or goal_distance < 0:
            raise ValueError("next_goal_distance_m must be finite and non-negative")

        timestamp = time.monotonic_ns() if timestamp_ns is None else int(timestamp_ns)
        if timestamp < 0:
            raise ValueError("timestamp_ns cannot be negative")
        if self._last_timestamp_ns is not None and timestamp <= self._last_timestamp_ns:
            raise ValueError("timestamp_ns must be strictly increasing")

        rgb_spec = self._manifest["rgb"]
        if rgb_spec is None:
            self._manifest["rgb"] = {
                "shape": list(image.shape),
                "dtype": "uint8",
                "color_space": "RGB",
                "compression": "lossless-deflate",
            }
        elif tuple(rgb_spec["shape"]) != image.shape:
            raise ValueError(
                f"rgb shape changed within episode: expected {tuple(rgb_spec['shape'])}, "
                f"got {image.shape}"
            )

        values = {
            "state_rad": state,
            "action_rad": action,
            "reward": np.float32(scalar_reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "timestamp_ns": np.int64(timestamp),
            "rgb": image,
            "imu_quaternion": quaternion,
            "imu_gyro": gyro,
            "imu_acceleration": acceleration,
            "command_velocity": command,
            "next_success": bool(next_success),
            "next_fallen": bool(next_fallen),
            "next_obstacle_contact": bool(next_obstacle_contact),
            "next_out_of_bounds": bool(next_out_of_bounds),
            "next_goal_distance_m": np.float32(goal_distance),
        }
        for key, value in values.items():
            self._buffer[key].append(value)
        self._last_timestamp_ns = timestamp
        self._terminal_seen = bool(terminated or truncated)

        if len(self._buffer["state_rad"]) >= self._chunk_size or self._terminal_seen:
            self.flush()

    def flush(self) -> None:
        """Durably commit buffered samples as one checksummed chunk."""

        if self._closed:
            raise RuntimeError("episode writer is closed")
        count = len(self._buffer["state_rad"])
        if count == 0:
            return
        chunk_index = len(self._manifest["chunks"])
        filename = f"chunk-{chunk_index:06d}.npz"
        final_path = self.path / filename
        temporary = self.path / f".{filename}.{uuid.uuid4().hex}.tmp"
        arrays = {
            "state_rad": np.stack(self._buffer["state_rad"]).astype(np.float32, copy=False),
            "action_rad": np.stack(self._buffer["action_rad"]).astype(np.float32, copy=False),
            "reward": np.asarray(self._buffer["reward"], dtype=np.float32),
            "terminated": np.asarray(self._buffer["terminated"], dtype=np.bool_),
            "truncated": np.asarray(self._buffer["truncated"], dtype=np.bool_),
            "timestamp_ns": np.asarray(self._buffer["timestamp_ns"], dtype=np.int64),
            "rgb": np.stack(self._buffer["rgb"]).astype(np.uint8, copy=False),
            "imu_quaternion": np.stack(self._buffer["imu_quaternion"]).astype(
                np.float32, copy=False
            ),
            "imu_gyro": np.stack(self._buffer["imu_gyro"]).astype(np.float32, copy=False),
            "imu_acceleration": np.stack(self._buffer["imu_acceleration"]).astype(
                np.float32, copy=False
            ),
            "command_velocity": np.stack(self._buffer["command_velocity"]).astype(
                np.float32, copy=False
            ),
            "next_success": np.asarray(self._buffer["next_success"], dtype=np.bool_),
            "next_fallen": np.asarray(self._buffer["next_fallen"], dtype=np.bool_),
            "next_obstacle_contact": np.asarray(
                self._buffer["next_obstacle_contact"], dtype=np.bool_
            ),
            "next_out_of_bounds": np.asarray(
                self._buffer["next_out_of_bounds"], dtype=np.bool_
            ),
            "next_goal_distance_m": np.asarray(
                self._buffer["next_goal_distance_m"], dtype=np.float32
            ),
        }
        try:
            with temporary.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, final_path)
        finally:
            temporary.unlink(missing_ok=True)

        start = int(self._manifest["step_count"])
        self._manifest["chunks"].append(
            {
                "path": filename,
                "start": start,
                "count": count,
                "sha256": _sha256(final_path),
            }
        )
        self._manifest["step_count"] = start + count
        for values in self._buffer.values():
            values.clear()
        _atomic_write_json(self.path / MANIFEST_NAME, self._manifest)

    def close(self) -> Path:
        """Commit remaining data and mark the episode complete."""

        if self._closed:
            return self.path
        self.flush()
        if int(self._manifest["step_count"]) == 0:
            raise ValueError("cannot complete an empty episode")
        self._manifest["status"] = "complete"
        self._manifest["completed_at"] = datetime.now(UTC).isoformat()
        _atomic_write_json(self.path / MANIFEST_NAME, self._manifest)
        self._closed = True
        return self.path

    def abort(self) -> Path:
        """Preserve committed data but mark it unusable as a complete episode."""

        if self._closed:
            return self.path
        self.flush()
        self._manifest["status"] = "aborted"
        self._manifest["aborted_at"] = datetime.now(UTC).isoformat()
        _atomic_write_json(self.path / MANIFEST_NAME, self._manifest)
        self._closed = True
        return self.path

    def __enter__(self) -> EpisodeWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


class EpisodeReader:
    """Validate and stream a canonical episode."""

    def __init__(
        self,
        path: str | Path,
        *,
        verify_checksums: bool = True,
        allow_incomplete: bool = False,
    ) -> None:
        candidate = Path(path).expanduser().resolve()
        self.path = candidate.parent if candidate.name == MANIFEST_NAME else candidate
        manifest_path = self.path / MANIFEST_NAME
        try:
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Episode manifest does not exist: {manifest_path}") from exc
        self._validate_manifest(allow_incomplete=allow_incomplete)
        if verify_checksums:
            for chunk in self.manifest["chunks"]:
                chunk_path = self.path / chunk["path"]
                if not chunk_path.is_file():
                    raise ValueError(f"episode chunk is missing: {chunk_path}")
                actual = _sha256(chunk_path)
                if actual != chunk["sha256"]:
                    raise ValueError(f"episode chunk checksum mismatch: {chunk_path.name}")

    def _validate_manifest(self, *, allow_incomplete: bool) -> None:
        manifest = self.manifest
        if manifest.get("format") != FORMAT_NAME:
            raise ValueError(f"not a {FORMAT_NAME} directory")
        if manifest.get("schema_version") != SCHEMA_VERSION:
            version = manifest.get("schema_version")
            raise ValueError(f"unsupported episode schema version: {version}")
        if not allow_incomplete and manifest.get("status") != "complete":
            raise ValueError(f"episode is not complete (status={manifest.get('status')!r})")
        if tuple(manifest.get("joint_names", ())) != JOINT_NAMES:
            raise ValueError("episode joint_names do not match canonical firmware order")
        if not isinstance(manifest.get("instruction"), str) or not manifest["instruction"]:
            raise ValueError("episode has no language instruction")
        rgb_spec = manifest.get("rgb")
        if not isinstance(rgb_spec, dict) or len(rgb_spec.get("shape", [])) != 3:
            raise ValueError("episode has invalid RGB metadata")
        expected_start = 0
        for index, chunk in enumerate(manifest.get("chunks", [])):
            if chunk.get("path") != f"chunk-{index:06d}.npz":
                raise ValueError("episode chunks are not canonically ordered")
            if chunk.get("start") != expected_start or int(chunk.get("count", 0)) < 1:
                raise ValueError("episode chunk ranges are invalid")
            expected_start += int(chunk["count"])
        if expected_start != int(manifest.get("step_count", -1)) or expected_start < 1:
            raise ValueError("episode step_count does not match chunk index")

    def __len__(self) -> int:
        return int(self.manifest["step_count"])

    @property
    def instruction(self) -> str:
        return str(self.manifest["instruction"])

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self.manifest["metadata"]

    def iter_chunks(self) -> Iterator[dict[str, np.ndarray]]:
        """Yield validated chunks while keeping peak RAM bounded by recording chunk size."""

        previous_timestamp: int | None = None
        image_shape = tuple(self.manifest["rgb"]["shape"])
        for entry in self.manifest["chunks"]:
            with np.load(self.path / entry["path"], allow_pickle=False) as archive:
                if set(archive.files) != set(_CHUNK_KEYS):
                    raise ValueError(f"chunk {entry['path']} has unexpected arrays")
                arrays = {key: np.asarray(archive[key]) for key in _CHUNK_KEYS}
            count = int(entry["count"])
            expected = {
                "state_rad": ((count, len(JOINT_NAMES)), np.float32),
                "action_rad": ((count, len(JOINT_NAMES)), np.float32),
                "reward": ((count,), np.float32),
                "terminated": ((count,), np.bool_),
                "truncated": ((count,), np.bool_),
                "timestamp_ns": ((count,), np.int64),
                "rgb": ((count, *image_shape), np.uint8),
                "imu_quaternion": ((count, 4), np.float32),
                "imu_gyro": ((count, 3), np.float32),
                "imu_acceleration": ((count, 3), np.float32),
                "command_velocity": ((count, 3), np.float32),
                "next_success": ((count,), np.bool_),
                "next_fallen": ((count,), np.bool_),
                "next_obstacle_contact": ((count,), np.bool_),
                "next_out_of_bounds": ((count,), np.bool_),
                "next_goal_distance_m": ((count,), np.float32),
            }
            for key, (shape, dtype) in expected.items():
                if arrays[key].shape != shape or arrays[key].dtype != dtype:
                    raise ValueError(
                        f"chunk {entry['path']} {key} must be {dtype} {shape}, got "
                        f"{arrays[key].dtype} {arrays[key].shape}"
                    )
            if not np.all(np.isfinite(arrays["state_rad"])):
                raise ValueError(f"chunk {entry['path']} has non-finite state")
            if not np.all(np.isfinite(arrays["action_rad"])):
                raise ValueError(f"chunk {entry['path']} has non-finite action")
            if not np.all(np.isfinite(arrays["reward"])):
                raise ValueError(f"chunk {entry['path']} has non-finite reward")
            for key in (
                "imu_quaternion",
                "imu_gyro",
                "imu_acceleration",
                "command_velocity",
                "next_goal_distance_m",
            ):
                if not np.all(np.isfinite(arrays[key])):
                    raise ValueError(f"chunk {entry['path']} has non-finite {key}")
            quaternion_norms = np.linalg.norm(arrays["imu_quaternion"], axis=1)
            if np.any(np.abs(quaternion_norms - 1.0) > 1e-5):
                raise ValueError(f"chunk {entry['path']} has a non-unit IMU quaternion")
            if np.any(arrays["next_goal_distance_m"] < 0):
                raise ValueError(f"chunk {entry['path']} has a negative goal distance")
            timestamps = arrays["timestamp_ns"]
            if np.any(np.diff(timestamps) <= 0):
                raise ValueError(f"chunk {entry['path']} timestamps are not strictly increasing")
            if previous_timestamp is not None and int(timestamps[0]) <= previous_timestamp:
                raise ValueError("episode timestamps are not strictly increasing across chunks")
            previous_timestamp = int(timestamps[-1])
            yield arrays

    def iter_steps(self) -> Iterator[EpisodeStep]:
        for arrays in self.iter_chunks():
            for index in range(arrays["state_rad"].shape[0]):
                yield EpisodeStep(
                    state_rad=arrays["state_rad"][index],
                    rgb=arrays["rgb"][index],
                    action_rad=arrays["action_rad"][index],
                    reward=float(arrays["reward"][index]),
                    terminated=bool(arrays["terminated"][index]),
                    truncated=bool(arrays["truncated"][index]),
                    timestamp_ns=int(arrays["timestamp_ns"][index]),
                    imu_quaternion=arrays["imu_quaternion"][index],
                    imu_gyro=arrays["imu_gyro"][index],
                    imu_acceleration=arrays["imu_acceleration"][index],
                    command_velocity=arrays["command_velocity"][index],
                    next_success=bool(arrays["next_success"][index]),
                    next_fallen=bool(arrays["next_fallen"][index]),
                    next_obstacle_contact=bool(
                        arrays["next_obstacle_contact"][index]
                    ),
                    next_out_of_bounds=bool(arrays["next_out_of_bounds"][index]),
                    next_goal_distance_m=float(arrays["next_goal_distance_m"][index]),
                )

    def load(self) -> Episode:
        chunks = list(self.iter_chunks())
        return Episode(
            state_rad=np.concatenate([chunk["state_rad"] for chunk in chunks]),
            rgb=np.concatenate([chunk["rgb"] for chunk in chunks]),
            action_rad=np.concatenate([chunk["action_rad"] for chunk in chunks]),
            reward=np.concatenate([chunk["reward"] for chunk in chunks]),
            terminated=np.concatenate([chunk["terminated"] for chunk in chunks]),
            truncated=np.concatenate([chunk["truncated"] for chunk in chunks]),
            timestamp_ns=np.concatenate([chunk["timestamp_ns"] for chunk in chunks]),
            imu_quaternion=np.concatenate([chunk["imu_quaternion"] for chunk in chunks]),
            imu_gyro=np.concatenate([chunk["imu_gyro"] for chunk in chunks]),
            imu_acceleration=np.concatenate(
                [chunk["imu_acceleration"] for chunk in chunks]
            ),
            command_velocity=np.concatenate(
                [chunk["command_velocity"] for chunk in chunks]
            ),
            next_success=np.concatenate([chunk["next_success"] for chunk in chunks]),
            next_fallen=np.concatenate([chunk["next_fallen"] for chunk in chunks]),
            next_obstacle_contact=np.concatenate(
                [chunk["next_obstacle_contact"] for chunk in chunks]
            ),
            next_out_of_bounds=np.concatenate(
                [chunk["next_out_of_bounds"] for chunk in chunks]
            ),
            next_goal_distance_m=np.concatenate(
                [chunk["next_goal_distance_m"] for chunk in chunks]
            ),
            instruction=self.instruction,
            metadata=self.metadata,
            episode_id=str(self.manifest["episode_id"]),
        )


def read_episode(
    path: str | Path, *, verify_checksums: bool = True, allow_incomplete: bool = False
) -> Episode:
    """Read and materialize a canonical episode."""

    return EpisodeReader(
        path, verify_checksums=verify_checksums, allow_incomplete=allow_incomplete
    ).load()
