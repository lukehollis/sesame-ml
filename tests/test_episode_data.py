from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sesame_ml.constants import STAND_ANGLES_RAD
from sesame_ml.data import EpisodeReader, EpisodeWriter, read_episode


def _image(index: int) -> np.ndarray:
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    image[..., 0] = index
    image[2:5, 3:9, 1] = 255
    return image


def _record(path: Path) -> tuple[list[np.ndarray], Path]:
    images = [_image(index) for index in range(3)]
    with EpisodeWriter(
        path,
        instruction="walk toward the green marker",
        metadata={"source": "mujoco", "seed": 7},
        episode_id="episode-test",
        chunk_size=2,
        timestamp_clock="simulation",
    ) as writer:
        for index, image in enumerate(images):
            writer.append(
                state_rad=STAND_ANGLES_RAD + index * 0.001,
                rgb=image,
                action_rad=STAND_ANGLES_RAD,
                reward=float(index),
                terminated=index == 2,
                timestamp_ns=index * 20_000_000,
                imu_quaternion=[1.0, 0.0, 0.0, 0.0],
                imu_gyro=[0.0, 0.0, float(index)],
                imu_acceleration=[0.0, 0.0, 9.81],
                command_velocity=[0.1, 0.0, -0.2],
                next_success=index == 2,
                next_obstacle_contact=index == 1,
                next_goal_distance_m=float(2 - index),
            )
    return images, path


def test_lossless_chunked_episode_round_trip(tmp_path: Path) -> None:
    images, path = _record(tmp_path / "episode-000")
    reader = EpisodeReader(path)
    episode = reader.load()

    assert len(reader) == 3
    assert episode.episode_id == "episode-test"
    assert episode.instruction == "walk toward the green marker"
    assert episode.metadata == {"seed": 7, "source": "mujoco"}
    assert episode.state_rad.dtype == np.float32
    assert episode.action_rad.shape == (3, 8)
    assert episode.rgb.dtype == np.uint8
    assert np.array_equal(episode.rgb, np.stack(images))
    assert episode.timestamp_ns.tolist() == [0, 20_000_000, 40_000_000]
    assert episode.terminated.tolist() == [False, False, True]
    assert episode.imu_quaternion.shape == (3, 4)
    assert episode.imu_gyro[:, 2].tolist() == [0.0, 1.0, 2.0]
    assert episode.imu_acceleration[:, 2].tolist() == pytest.approx([9.81] * 3)
    assert episode.command_velocity[0].tolist() == pytest.approx([0.1, 0.0, -0.2])
    assert episode.next_success.tolist() == [False, False, True]
    assert episode.next_obstacle_contact.tolist() == [False, True, False]
    assert episode.next_goal_distance_m.tolist() == [2.0, 1.0, 0.0]
    assert len(reader.manifest["chunks"]) == 2
    assert read_episode(path).reward.tolist() == [0.0, 1.0, 2.0]


def test_episode_checksum_detects_corruption(tmp_path: Path) -> None:
    _, path = _record(tmp_path / "episode-000")
    with (path / "chunk-000000.npz").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        EpisodeReader(path)


def test_episode_writer_preserves_aborted_recording_but_reader_rejects_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "episode-aborted"
    with pytest.raises(RuntimeError, match="capture failed"):
        with EpisodeWriter(path, instruction="stand") as writer:
            writer.append(
                state_rad=STAND_ANGLES_RAD,
                rgb=_image(0),
                action_rad=STAND_ANGLES_RAD,
                timestamp_ns=1,
            )
            raise RuntimeError("capture failed")
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["status"] == "aborted"
    with pytest.raises(ValueError, match="not complete"):
        EpisodeReader(path)
    assert len(EpisodeReader(path, allow_incomplete=True)) == 1


def test_episode_validation_rejects_misaligned_or_unsafe_samples(tmp_path: Path) -> None:
    writer = EpisodeWriter(tmp_path / "bad", instruction="stand")
    with pytest.raises(ValueError, match="dtype uint8"):
        writer.append(
            state_rad=STAND_ANGLES_RAD,
            rgb=np.zeros((4, 4, 3), dtype=np.float32),
            action_rad=STAND_ANGLES_RAD,
            timestamp_ns=1,
        )
    writer.append(
        state_rad=STAND_ANGLES_RAD,
        rgb=_image(0),
        action_rad=STAND_ANGLES_RAD,
        timestamp_ns=2,
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        writer.append(
            state_rad=STAND_ANGLES_RAD,
            rgb=_image(1),
            action_rad=STAND_ANGLES_RAD,
            timestamp_ns=2,
        )
    writer.abort()
