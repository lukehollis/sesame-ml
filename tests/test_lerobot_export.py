from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest

from sesame_ml.constants import JOINT_NAMES, STAND_ANGLES_RAD
from sesame_ml.data import (
    EpisodeWriter,
    export_to_lerobot,
    groot_modality,
    validate_groot_v2_dataset,
)


def _episode(path: Path, instruction: str, offset: int) -> Path:
    with EpisodeWriter(path, instruction=instruction, chunk_size=1) as writer:
        for index in range(2):
            writer.append(
                state_rad=STAND_ANGLES_RAD,
                rgb=np.full((8, 10, 3), offset + index, dtype=np.uint8),
                action_rad=STAND_ANGLES_RAD,
                reward=index,
                terminated=index == 1,
                timestamp_ns=index + 1,
            )
    return path


class _FakeLeRobotDataset:
    instance = None

    @classmethod
    def create(cls, **kwargs):
        instance = cls(kwargs)
        cls.instance = instance
        return instance

    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.root = Path(kwargs["root"])
        (self.root / "meta").mkdir(parents=True)
        self.frames = []
        self.episodes = []

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self, task):
        index = len(self.episodes)
        self.episodes.append(task)
        data = self.root / "data/chunk-000" / f"episode_{index:06d}.parquet"
        video = (
            self.root
            / "videos/chunk-000/observation.images.front"
            / f"episode_{index:06d}.mp4"
        )
        data.parent.mkdir(parents=True, exist_ok=True)
        video.parent.mkdir(parents=True, exist_ok=True)
        data.write_bytes(b"parquet-placeholder")
        video.write_bytes(b"video-placeholder")
        features = {
            key: {**value, "shape": list(value["shape"])}
            for key, value in self.kwargs["features"].items()
        }
        info = {
            "codebase_version": "v2.1",
            "robot_type": self.kwargs["robot_type"],
            "fps": self.kwargs["fps"],
            "features": features,
        }
        (self.root / "meta/info.json").write_text(json.dumps(info))
        tasks = []
        for value in self.episodes:
            if value not in tasks:
                tasks.append(value)
        (self.root / "meta/tasks.jsonl").write_text(
            "".join(
                json.dumps({"task_index": task_index, "task": value}) + "\n"
                for task_index, value in enumerate(tasks)
            )
        )
        (self.root / "meta/episodes.jsonl").write_text(
            "".join(
                json.dumps({"episode_index": episode_index, "tasks": [value], "length": 2})
                + "\n"
                for episode_index, value in enumerate(self.episodes)
            )
        )


class _FakeLeRobotV21(_FakeLeRobotDataset):
    def add_frame(self, frame):
        assert isinstance(frame.get("task"), str)
        self.frames.append(frame)

    def save_episode(self):
        self.episodes.append(self.frames[-1]["task"])


def test_lerobot_export_uses_canonical_features_and_adds_groot_metadata(tmp_path: Path) -> None:
    first = _episode(tmp_path / "first", "walk forward", 1)
    second = _episode(tmp_path / "second", "turn left", 3)
    output = tmp_path / "lerobot"

    result = export_to_lerobot(
        [first, second],
        repo_id="local/sesame",
        output_dir=output,
        dataset_class=_FakeLeRobotDataset,
    )

    assert result.dataset_root == output
    assert result.episode_count == 2
    assert result.frame_count == 4
    assert result.task_count == 2
    fake = _FakeLeRobotDataset.instance
    assert fake.kwargs["features"]["observation.state"]["names"] == list(JOINT_NAMES)
    assert fake.kwargs["features"]["action"]["shape"] == (8,)
    assert "annotation.human.task_description" not in fake.kwargs["features"]
    assert "annotation.human.task_description" not in fake.frames[0]
    assert np.array_equal(fake.frames[0]["observation.state"], STAND_ANGLES_RAD.astype(np.float32))
    assert json.loads((output / "meta/modality.json").read_text()) == groot_modality()
    validate_groot_v2_dataset(output)


def test_lerobot_loader_supports_pinned_v21_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sesame_ml.data import lerobot as module

    expected = object()

    def fake_import(name: str):
        if name == "lerobot.datasets.lerobot_dataset":
            raise ModuleNotFoundError(name="lerobot.datasets")
        assert name == "lerobot.common.datasets.lerobot_dataset"
        return types.SimpleNamespace(LeRobotDataset=expected)

    monkeypatch.setattr(module.importlib, "import_module", fake_import)
    assert module._load_lerobot_dataset_class() is expected


def test_lerobot_export_supplies_frame_task_for_pinned_v21_api(tmp_path: Path) -> None:
    episode = _episode(tmp_path / "episode", "walk forward", 1)
    result = export_to_lerobot(
        [episode],
        repo_id="local/sesame",
        output_dir=tmp_path / "v21",
        dataset_class=_FakeLeRobotV21,
        add_groot_metadata=False,
    )
    assert result.frame_count == 2
    assert [frame["task"] for frame in _FakeLeRobotV21.instance.frames] == [
        "walk forward",
        "walk forward",
    ]


def test_lerobot_loader_does_not_hide_missing_transitive_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sesame_ml.data import lerobot as module

    def fake_import(name: str):
        del name
        raise ModuleNotFoundError(name="torchcodec")

    monkeypatch.setattr(module.importlib, "import_module", fake_import)
    with pytest.raises(ModuleNotFoundError) as error:
        module._load_lerobot_dataset_class()
    assert error.value.name == "torchcodec"


def test_groot_validator_rejects_v3_dataset(tmp_path: Path) -> None:
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"codebase_version": "v3.0"}))
    for name in ("episodes.jsonl", "tasks.jsonl"):
        (meta / name).write_text("")
    (meta / "modality.json").write_text(json.dumps(groot_modality()))
    with pytest.raises(ValueError, match="requires LeRobot v2"):
        validate_groot_v2_dataset(tmp_path)
