"""Export canonical recordings through LeRobot and add GR00T v2 metadata."""

from __future__ import annotations

import importlib
import inspect
import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sesame_ml.constants import ACTION_RATE_HZ, JOINT_NAMES
from sesame_ml.data.episode import EpisodeReader


@dataclass(frozen=True, slots=True)
class LeRobotExportResult:
    dataset_root: Path
    episode_count: int
    frame_count: int
    task_count: int


def groot_modality() -> dict[str, Any]:
    """Return the exact GR00T interpretation of the canonical 8D arrays."""

    return {
        "state": {"joints": {"start": 0, "end": len(JOINT_NAMES)}},
        "action": {"joints": {"start": 0, "end": len(JOINT_NAMES)}},
        "video": {"front": {"original_key": "observation.images.front"}},
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
        },
    }


def write_groot_metadata(dataset_root: str | Path, *, overwrite: bool = True) -> Path:
    """Install the GR00T N1.7 ``meta/modality.json`` extension."""

    root = Path(dataset_root).expanduser().resolve()
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    target = meta / "modality.json"
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(groot_modality(), indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
    return target


def _version_major(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    normalized = value.lower().lstrip("v")
    try:
        return int(normalized.split(".", 1)[0])
    except ValueError:
        return None


def validate_groot_v2_dataset(dataset_root: str | Path) -> None:
    """Fail early on the structural mistakes that otherwise surface deep in training."""

    root = Path(dataset_root).expanduser().resolve()
    meta = root / "meta"
    required = ["info.json", "episodes.jsonl", "tasks.jsonl", "modality.json"]
    missing = [str(meta / name) for name in required if not (meta / name).is_file()]
    if missing:
        raise ValueError("GR00T LeRobot v2 metadata is incomplete: " + ", ".join(missing))
    info = json.loads((meta / "info.json").read_text(encoding="utf-8"))
    version = info.get("codebase_version") or info.get("format_version")
    if _version_major(version) != 2:
        raise ValueError(
            f"GR00T N1.7 currently requires LeRobot v2, but dataset reports {version!r}. "
            "Export in the Isaac-GR00T LeRobot environment or run NVIDIA's official "
            "scripts/lerobot_conversion/convert_v3_to_v2.py, then add modality.json."
        )
    features = info.get("features", {})
    for key in ("observation.state", "action", "observation.images.front"):
        if key not in features:
            raise ValueError(f"LeRobot info.json is missing required feature {key!r}")
    for key in ("observation.state", "action"):
        if list(features[key].get("shape", [])) != [len(JOINT_NAMES)]:
            raise ValueError(f"{key} must have shape [{len(JOINT_NAMES)}]")
    modality = json.loads((meta / "modality.json").read_text(encoding="utf-8"))
    if modality != groot_modality():
        raise ValueError("meta/modality.json does not match the canonical Sesame embodiment")
    if not list(root.glob("data/chunk-*/*.parquet")):
        raise ValueError("GR00T LeRobot v2 dataset contains no episode parquet files")
    if not list(root.glob("videos/chunk-*/observation.images.front/*.mp4")):
        raise ValueError("GR00T LeRobot v2 dataset contains no front-camera MP4 episodes")


def _load_lerobot_dataset_class() -> Any:
    module_names = (
        "lerobot.datasets.lerobot_dataset",
        "lerobot.common.datasets.lerobot_dataset",
    )
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            missing = exc.name or ""
            if missing == "lerobot" or module_name == missing or module_name.startswith(
                f"{missing}."
            ):
                continue
            raise
        dataset_class = getattr(module, "LeRobotDataset", None)
        if dataset_class is None:
            raise ImportError(f"{module_name} does not expose LeRobotDataset")
        return dataset_class
    raise ModuleNotFoundError(
        "LeRobot export requires a supported `lerobot` package. Use OpenPI's pinned "
        "LeRobot v2.1 environment (commit 0cf86487) for a direct GR00T-compatible v2 "
        "export, or follow the documented GR00T v3-to-v2 conversion path."
    )


def _accepts(callable_: Any, parameter: str) -> bool:
    signature = inspect.signature(callable_)
    return parameter in signature.parameters or any(
        item.kind is inspect.Parameter.VAR_KEYWORD for item in signature.parameters.values()
    )


def _dataset_features(image_shape: tuple[int, int, int], *, use_videos: bool) -> dict[str, Any]:
    return {
        "observation.images.front": {
            "dtype": "video" if use_videos else "image",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": list(JOINT_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": list(JOINT_NAMES),
        },
        "next.reward": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["reward"],
        },
        "next.done": {
            "dtype": "bool",
            "shape": (1,),
            "names": ["done"],
        },
    }


def _resolve_dataset_root(dataset: Any, requested: Path | None) -> Path:
    for owner in (dataset, getattr(dataset, "meta", None)):
        value = getattr(owner, "root", None) if owner is not None else None
        if value is not None:
            return Path(value).expanduser().resolve()
    if requested is not None:
        return requested
    raise RuntimeError(
        "The installed LeRobot API did not expose the dataset root. Pass `output_dir=` "
        "or use a supported LeRobot release."
    )


def export_to_lerobot(
    episode_paths: Iterable[str | Path],
    *,
    repo_id: str,
    output_dir: str | Path | None = None,
    fps: int = int(ACTION_RATE_HZ),
    robot_type: str = "sesame_quadruped",
    use_videos: bool = True,
    overwrite: bool = False,
    add_groot_metadata: bool = True,
    require_groot_v2: bool = True,
    image_writer_threads: int = 4,
    image_writer_processes: int = 0,
    dataset_class: Any | None = None,
) -> LeRobotExportResult:
    """Convert one or more canonical episodes using the installed LeRobot API.

    The canonical source remains lossless. The resulting video-backed dataset is optimized
    for training throughput and can be consumed by OpenPI. When ``add_groot_metadata`` is
    enabled, the function also writes the NEW_EMBODIMENT modality mapping and, by default,
    verifies that the installed LeRobot actually produced v2 rather than silently handing
    GR00T an incompatible v3 dataset.
    """

    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("repo_id must be a non-empty string")
    if fps < 1:
        raise ValueError("fps must be positive")
    readers = [EpisodeReader(path) for path in episode_paths]
    if not readers:
        raise ValueError("at least one episode is required")
    image_shape = tuple(readers[0].manifest["rgb"]["shape"])
    for reader in readers[1:]:
        if tuple(reader.manifest["rgb"]["shape"]) != image_shape:
            raise ValueError("all episodes must use the same RGB resolution")

    requested_root = Path(output_dir).expanduser().resolve() if output_dir is not None else None
    if requested_root is not None and requested_root.exists():
        if not overwrite:
            raise FileExistsError(requested_root)
        if requested_root.is_symlink() or not requested_root.is_dir():
            raise ValueError(f"Refusing to overwrite non-directory dataset path: {requested_root}")
        shutil.rmtree(requested_root)

    cls = dataset_class or _load_lerobot_dataset_class()
    create = cls.create
    kwargs: dict[str, Any] = {
        "repo_id": repo_id.strip(),
        "robot_type": robot_type,
        "fps": int(fps),
        "features": _dataset_features(image_shape, use_videos=use_videos),
    }
    optional = {
        "use_videos": use_videos,
        "image_writer_threads": int(image_writer_threads),
        "image_writer_processes": int(image_writer_processes),
    }
    kwargs.update({key: value for key, value in optional.items() if _accepts(create, key)})
    if requested_root is not None:
        if not _accepts(create, "root"):
            raise RuntimeError(
                "This LeRobot release cannot select a local root through its API. Omit "
                "`output_dir` or install the version pinned by OpenPI/Isaac-GR00T."
            )
        kwargs["root"] = requested_root
    dataset = create(**kwargs)
    dataset_root = _resolve_dataset_root(dataset, requested_root)

    task_indices: dict[str, int] = {}
    for reader in readers:
        if reader.instruction not in task_indices:
            task_indices[reader.instruction] = len(task_indices)

    add_frame_accepts_task = _accepts(dataset.add_frame, "task")
    save_accepts_task = _accepts(dataset.save_episode, "task")
    frame_count = 0
    for reader in readers:
        for step in reader.iter_steps():
            frame: dict[str, Any] = {
                "observation.images.front": step.rgb,
                "observation.state": step.state_rad,
                "action": step.action_rad,
                "next.reward": np.asarray([step.reward], dtype=np.float32),
                "next.done": np.asarray([step.terminated or step.truncated], dtype=np.bool_),
            }
            if add_frame_accepts_task:
                dataset.add_frame(frame, task=reader.instruction)
            else:
                # Older LeRobot APIs stored the task in every frame; newer releases
                # attach it once in save_episode(task=...).
                if not save_accepts_task:
                    frame["task"] = reader.instruction
                dataset.add_frame(frame)
            frame_count += 1
        if save_accepts_task:
            dataset.save_episode(task=reader.instruction)
        else:
            dataset.save_episode()

    # Newer LeRobot releases defer metadata/video finalization; older v2 releases do it
    # in save_episode and expose neither method.
    finalize = getattr(dataset, "finalize", None)
    consolidate = getattr(dataset, "consolidate", None)
    if callable(finalize):
        finalize()
    elif callable(consolidate):
        consolidate()

    if add_groot_metadata:
        if not use_videos:
            raise ValueError("GR00T LeRobot v2 requires MP4-backed video; set use_videos=True")
        write_groot_metadata(dataset_root)
        if require_groot_v2:
            validate_groot_v2_dataset(dataset_root)
    return LeRobotExportResult(
        dataset_root=dataset_root,
        episode_count=len(readers),
        frame_count=frame_count,
        task_count=len(task_indices),
    )
