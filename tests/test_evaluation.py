from __future__ import annotations

import csv
import json
from fractions import Fraction

import numpy as np
import pytest

from sesame_ml.evaluation import EvaluationConfig, _VideoWriter, evaluate_policy
from sesame_ml.policies import FirmwareSequencePolicy, StandPolicy


def test_multiseed_evaluation_writes_json_and_csv(tmp_path) -> None:
    config = EvaluationConfig(
        tasks=("stand",),
        seeds=(7, 11),
        episodes_per_seed=2,
        output_directory=tmp_path,
        run_name="stand-regression",
        domain_randomization=False,
        maximum_episode_steps=4,
    )
    result = evaluate_policy(StandPolicy(), config)
    assert len(result.episodes) == 4
    assert result.summary["overall"]["seeds"] == 2
    assert result.summary["overall"]["success_rate"] == 1.0
    assert result.json_report.exists()
    assert result.csv_report.exists()

    report = json.loads(result.json_report.read_text())
    assert report["schema_version"] == 1
    assert report["policy"] == "stand"
    assert len(report["episodes"]) == 4
    with result.csv_report.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["seed"] for row in rows} == {"7", "11"}


def test_firmware_sequence_has_stable_evaluation_identifier(tmp_path) -> None:
    config = EvaluationConfig(
        tasks=("locomotion",),
        seeds=(3,),
        output_directory=tmp_path,
        run_name="firmware",
        domain_randomization=False,
        maximum_episode_steps=3,
    )
    result = evaluate_policy(FirmwareSequencePolicy(), config)
    assert result.episodes[0].policy == "firmware_sequence"
    assert result.episodes[0].episode_length == 3


def test_locomotion_survival_without_tracking_is_not_success(tmp_path) -> None:
    config = EvaluationConfig(
        tasks=("locomotion",),
        seeds=(4,),
        output_directory=tmp_path,
        run_name="stationary-locomotion",
        domain_randomization=False,
        maximum_episode_steps=200,
    )
    result = evaluate_policy(StandPolicy(), config)
    episode = result.episodes[0]
    assert not episode.fall
    assert not episode.success
    assert episode.termination == "time_limit"
    assert episode.linear_tracking_rmse_m_s > 0.06


def test_mp4_writer_preserves_fractional_frame_rate(tmp_path) -> None:
    av = pytest.importorskip("av")
    path = tmp_path / "fractional.mp4"
    writer = _VideoWriter(path, fps=50 / 3)
    for value in range(6):
        writer.append(np.full((48, 64, 3), value * 20, dtype=np.uint8))
    writer.close()

    with av.open(path) as container:
        stream = container.streams.video[0]
        assert stream.average_rate == Fraction(50, 3)


def test_strided_video_includes_unaligned_terminal_frame(tmp_path) -> None:
    av = pytest.importorskip("av")
    config = EvaluationConfig(
        tasks=("stand",),
        seeds=(2,),
        output_directory=tmp_path,
        run_name="terminal-frame",
        domain_randomization=False,
        maximum_episode_steps=2,
        video_episodes_per_task=1,
        video_frame_stride=3,
    )
    result = evaluate_policy(StandPolicy(), config)
    with av.open(result.episodes[0].video) as container:
        frames = list(container.decode(video=0))
        assert container.streams.video[0].average_rate == Fraction(50, 3)
    assert len(frames) == 2  # initial state plus off-stride terminal state


def test_front_video_records_policy_camera_observation(tmp_path) -> None:
    av = pytest.importorskip("av")
    config = EvaluationConfig(
        tasks=("navigation",),
        seeds=(4,),
        output_directory=tmp_path,
        run_name="front-camera",
        observation_mode="pixels",
        domain_randomization=False,
        maximum_episode_steps=2,
        video_episodes_per_task=1,
        video_view="front",
    )
    result = evaluate_policy(StandPolicy(), config)
    with av.open(result.episodes[0].video) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height) == (160, 120)
        assert len(list(container.decode(video=0))) == 3
