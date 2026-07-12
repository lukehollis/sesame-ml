"""Multi-seed policy evaluation, metrics, reports, and video capture."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from PIL import Image

from sesame_ml.envs import SesameEnv, SesameEnvConfig, Task
from sesame_ml.policies import PolicyLike, SB3Policy, predict_action

EnvironmentFactory = Callable[[Task, str | None], gym.Env]


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for comparable task and domain-randomization evaluation."""

    tasks: tuple[Task | str, ...] = (Task.STAND, Task.LOCOMOTION, Task.NAVIGATION)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    episodes_per_seed: int = 1
    output_directory: str | Path = "evaluations"
    run_name: str | None = None
    domain_randomization: bool = True
    observation_mode: str = "state"
    privileged_joint_state: bool = False
    maximum_episode_steps: int | None = None
    deterministic: bool = True
    video_episodes_per_task: int = 0
    video_frame_stride: int = 1
    video_fps: int = 50
    video_extension: str = "mp4"
    video_view: str = "external"

    def __post_init__(self) -> None:
        tasks = tuple(Task(task) for task in self.tasks)
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "seeds", tuple(int(seed) for seed in self.seeds))
        if not tasks:
            raise ValueError("at least one task is required")
        if not self.seeds:
            raise ValueError("at least one evaluation seed is required")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("evaluation seeds must be non-negative")
        if self.episodes_per_seed <= 0:
            raise ValueError("episodes_per_seed must be positive")
        if self.video_episodes_per_task < 0:
            raise ValueError("video_episodes_per_task cannot be negative")
        if self.video_frame_stride <= 0 or self.video_fps <= 0:
            raise ValueError("video frame stride and fps must be positive")
        if self.video_extension.lower().lstrip(".") not in {"mp4", "gif"}:
            raise ValueError("video_extension must be 'mp4' or 'gif'")
        if self.video_view not in {"external", "front", "split"}:
            raise ValueError("video_view must be 'external', 'front', or 'split'")
        if self.maximum_episode_steps is not None and self.maximum_episode_steps <= 0:
            raise ValueError("maximum_episode_steps must be positive when supplied")


@dataclass(frozen=True)
class EpisodeMetrics:
    policy: str
    task: str
    seed: int
    episode: int
    environment_seed: int
    episode_return: float
    episode_length: int
    simulated_seconds: float
    wall_seconds: float
    realtime_factor: float
    success: bool
    fall: bool
    termination: str
    distance_m: float
    energy_j: float
    upright_mean: float
    upright_minimum: float
    height_mean_m: float
    height_minimum_m: float
    linear_tracking_rmse_m_s: float
    yaw_tracking_rmse_rad_s: float
    goal_distance_initial_m: float
    goal_distance_final_m: float
    goal_distance_minimum_m: float
    command_linear_rms_m_s: float
    command_yaw_rms_rad_s: float
    obstacle_contact_steps: int
    obstacle_contact_fraction: float
    domain_mass_scale: float
    domain_friction: float
    domain_servo_strength: float
    domain_servo_kp_scale: float
    domain_servo_time_constant_s: float
    domain_action_delay_steps: int
    domain_imu_noise_scale: float
    domain_battery_voltage: float
    action_rms: float
    action_delta_rms: float
    action_saturation_fraction: float
    inference_latency_mean_ms: float
    inference_latency_p50_ms: float
    inference_latency_p95_ms: float
    inference_latency_maximum_ms: float
    video: str


@dataclass(frozen=True)
class EvaluationResult:
    output_directory: Path
    json_report: Path
    csv_report: Path
    summary: dict[str, Any]
    episodes: tuple[EpisodeMetrics, ...]


def evaluation_name(policy: Any) -> str:
    """Return the stable identifier written to evaluation artifacts."""

    explicit = getattr(policy, "evaluation_name", None)
    if explicit:
        return str(explicit)
    name = policy.__class__.__name__
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()
    return name.removesuffix("_policy") or "policy"


def _make_environment(config: EvaluationConfig, task: Task, render_mode: str | None) -> SesameEnv:
    environment_config = SesameEnvConfig(
        task=task,
        observation_mode=config.observation_mode,
        domain_randomization=config.domain_randomization,
        privileged_joint_state=config.privileged_joint_state,
        maximum_episode_steps=config.maximum_episode_steps,
    )
    return SesameEnv(task=task, config=environment_config, render_mode=render_mode)


def _output_directory(config: EvaluationConfig, policy_name: str) -> Path:
    root = Path(config.output_directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if config.run_name:
        name = config.run_name
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        name = f"{policy_name}-{stamp}"
    output = root / name
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"evaluation directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


class _VideoWriter:
    def __init__(self, path: Path, *, fps: float) -> None:
        self.path = path
        self._fps = fps
        self._container = None
        self._stream = None
        self._writer = None
        if path.suffix == ".mp4":
            try:
                import av
            except ImportError as error:  # pragma: no cover - optional dependency path
                raise ImportError(
                    "MP4 recording requires `pip install sesame-ml[video]`"
                ) from error
            self._av = av
            self._container = av.open(str(path), mode="w")
        else:
            try:
                import imageio.v2 as imageio
            except ImportError as error:  # pragma: no cover - optional dependency path
                raise ImportError(
                    "GIF recording requires `pip install sesame-ml[video]`"
                ) from error
            self._writer = imageio.get_writer(path, mode="I", fps=fps)

    def append(self, frame: np.ndarray | None) -> None:
        if frame is None:
            return
        pixels = np.asarray(frame)
        if pixels.ndim != 3 or pixels.shape[2] not in {3, 4}:
            raise ValueError(f"rendered frame has invalid shape {pixels.shape}")
        pixels = pixels[:, :, :3].astype(np.uint8, copy=False)
        if self._container is not None:
            if self._stream is None:
                rate = Fraction(self._fps).limit_denominator(1_000)
                self._stream = self._container.add_stream("libx264", rate=rate)
                self._stream.width = pixels.shape[1]
                self._stream.height = pixels.shape[0]
                self._stream.pix_fmt = "yuv420p"
                self._stream.options = {"crf": "20", "preset": "medium"}
            video_frame = self._av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in self._stream.encode(video_frame):
                self._container.mux(packet)
        else:
            self._writer.append_data(pixels)

    def close(self) -> None:
        if self._container is not None:
            if self._stream is not None:
                for packet in self._stream.encode():
                    self._container.mux(packet)
            self._container.close()
            self._container = None
        elif self._writer is not None:
            self._writer.close()


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _episode_seed(seed: int, episode: int) -> int:
    # SeedSequence avoids correlations while keeping every row independently replayable.
    return int(np.random.SeedSequence([seed, episode]).generate_state(1, dtype=np.uint32)[0])


def _front_video_frame(environment: gym.Env, observation: Any) -> np.ndarray:
    if isinstance(observation, dict) and "rgb" in observation:
        return np.asarray(observation["rgb"], dtype=np.uint8).copy()
    unwrapped = getattr(environment, "unwrapped", environment)
    render_camera = getattr(unwrapped, "render_camera", None)
    if not callable(render_camera):
        raise TypeError("front video view requires an environment with render_camera()")
    return np.asarray(render_camera(), dtype=np.uint8)


def _video_frame(environment: gym.Env, observation: Any, view: str) -> np.ndarray | None:
    if view == "external":
        return environment.render()
    front = _front_video_frame(environment, observation)
    if view == "front":
        return front

    external = environment.render()
    if external is None:
        return front
    frame = np.asarray(external, dtype=np.uint8).copy()
    target_height = max(2, int(round(frame.shape[0] * 0.38)))
    target_width = max(2, int(round(target_height * front.shape[1] / front.shape[0])))
    target_width -= target_width % 2
    target_height -= target_height % 2
    inset = np.asarray(
        Image.fromarray(front).resize((target_width, target_height), Image.Resampling.BILINEAR)
    )
    border = max(2, frame.shape[0] // 120)
    margin = max(8, frame.shape[0] // 40)
    y0 = margin
    x0 = frame.shape[1] - margin - target_width
    frame[y0 - border : y0 + target_height + border, x0 - border : x0 + target_width + border] = 0
    frame[y0 : y0 + target_height, x0 : x0 + target_width] = inset
    return frame


def _evaluate_episode(
    policy: PolicyLike,
    *,
    policy_name: str,
    task: Task,
    seed: int,
    episode_index: int,
    environment: gym.Env,
    deterministic: bool,
    video_path: Path | None,
    video_fps: int,
    video_frame_stride: int,
    video_view: str,
) -> EpisodeMetrics:
    environment_seed = _episode_seed(seed, episode_index)
    observation, initial_info = environment.reset(seed=environment_seed)
    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset(seed=environment_seed)

    writer = (
        _VideoWriter(video_path, fps=video_fps / video_frame_stride)
        if video_path is not None
        else None
    )
    inference_latency_ms: list[float] = []
    actions: list[np.ndarray] = []
    uprights: list[float] = []
    heights: list[float] = []
    linear_errors: list[float] = []
    yaw_errors: list[float] = []
    commands: list[np.ndarray] = []
    goal_distances: list[float] = []
    episode_return = 0.0
    step_count = 0
    terminated = False
    truncated = False
    final_info = initial_info
    execution_wall_seconds = 0.0
    try:
        if writer is not None:
            writer.append(_video_frame(environment, observation, video_view))
        while not (terminated or truncated):
            execution_start = time.perf_counter()
            inference_start = time.perf_counter_ns()
            action = predict_action(policy, observation, deterministic=deterministic)
            inference_latency_ms.append((time.perf_counter_ns() - inference_start) / 1_000_000.0)
            observation, reward, terminated, truncated, final_info = environment.step(action)
            # Keep throughput comparable between recorded and unrecorded episodes. Pixel
            # observation rendering remains included because it is part of policy execution;
            # the optional third-person render and encoder below are intentionally excluded.
            execution_wall_seconds += time.perf_counter() - execution_start
            episode_return += float(reward)
            step_count += 1
            actions.append(action.copy())
            uprights.append(float(final_info.get("upright", 0.0)))
            heights.append(float(final_info.get("base_height_m", 0.0)))
            command = np.asarray(final_info.get("command", np.zeros(3)), dtype=np.float64)
            commands.append(command.copy())
            velocity = np.asarray(
                final_info.get("base_velocity_m_s", np.zeros(3)), dtype=np.float64
            )
            linear_errors.append(float(np.linalg.norm(velocity[:2] - command[:2])))
            yaw_errors.append(float(final_info.get("yaw_rate_rad_s", 0.0) - command[2]))
            if task is Task.NAVIGATION:
                goal_distances.append(float(final_info.get("goal_distance_m", math.inf)))
            if writer is not None and (
                step_count % video_frame_stride == 0 or terminated or truncated
            ):
                writer.append(_video_frame(environment, observation, video_view))
    finally:
        if writer is not None:
            writer.close()
    wall_seconds = execution_wall_seconds

    episode_info = final_info.get("episode", {})
    fall = bool(episode_info.get("fall", terminated and not final_info.get("success", False)))
    linear_tracking_rmse = (
        float(np.sqrt(np.mean(np.square(linear_errors)))) if linear_errors else 0.0
    )
    yaw_tracking_rmse = (
        float(np.sqrt(np.mean(np.square(yaw_errors)))) if yaw_errors else 0.0
    )
    upright_mean = _mean(uprights)
    if task is Task.NAVIGATION:
        success = bool(final_info.get("success", False))
    elif task is Task.RECOVERY:
        success = bool(final_info.get("success", False) and not fall)
    elif task is Task.LOCOMOTION:
        # Survival is necessary but not sufficient: a stationary controller must not earn a
        # successful locomotion label merely by waiting out the episode.
        success = bool(
            truncated
            and not fall
            and upright_mean >= 0.75
            and linear_tracking_rmse <= 0.06
            and yaw_tracking_rmse <= 0.45
        )
    else:
        success = bool(truncated and not fall)
    if success:
        termination = "success"
    elif fall:
        termination = "fall"
    elif final_info.get("out_of_bounds", False):
        termination = "out_of_bounds"
    elif truncated:
        termination = "time_limit"
    else:
        termination = "terminated"

    action_array = np.asarray(actions, dtype=np.float64)
    if len(action_array):
        action_rms = float(np.sqrt(np.mean(action_array**2)))
        saturation = float(np.mean(np.abs(action_array) >= 1.0 - 1e-6))
    else:
        action_rms = saturation = 0.0
    action_delta_rms = (
        float(np.sqrt(np.mean(np.diff(action_array, axis=0) ** 2)))
        if len(action_array) > 1
        else 0.0
    )
    unwrapped = getattr(environment, "unwrapped", environment)
    control_dt = float(getattr(unwrapped, "config", SesameEnvConfig()).control_dt)
    simulated_seconds = step_count * control_dt
    initial_goal = float(initial_info.get("goal_distance_m", 0.0))
    final_goal = float(goal_distances[-1]) if goal_distances else initial_goal
    minimum_goal = float(min(goal_distances)) if goal_distances else initial_goal
    command_array = np.asarray(commands, dtype=np.float64)
    command_linear_rms = (
        float(np.sqrt(np.mean(np.sum(np.square(command_array[:, :2]), axis=1))))
        if len(command_array)
        else 0.0
    )
    command_yaw_rms = (
        float(np.sqrt(np.mean(np.square(command_array[:, 2])))) if len(command_array) else 0.0
    )
    obstacle_contact_steps = int(final_info.get("obstacle_contact_steps", 0))
    domain = dict(initial_info.get("domain", {}))
    return EpisodeMetrics(
        policy=policy_name,
        task=task.value,
        seed=seed,
        episode=episode_index,
        environment_seed=environment_seed,
        episode_return=episode_return,
        episode_length=step_count,
        simulated_seconds=simulated_seconds,
        wall_seconds=wall_seconds,
        realtime_factor=simulated_seconds / max(wall_seconds, 1e-12),
        success=success,
        fall=fall,
        termination=termination,
        distance_m=float(episode_info.get("distance_m", 0.0)),
        energy_j=float(episode_info.get("energy_j", 0.0)),
        upright_mean=upright_mean,
        upright_minimum=float(min(uprights)) if uprights else 0.0,
        height_mean_m=_mean(heights),
        height_minimum_m=float(min(heights)) if heights else 0.0,
        linear_tracking_rmse_m_s=linear_tracking_rmse,
        yaw_tracking_rmse_rad_s=yaw_tracking_rmse,
        goal_distance_initial_m=initial_goal,
        goal_distance_final_m=final_goal,
        goal_distance_minimum_m=minimum_goal,
        command_linear_rms_m_s=command_linear_rms,
        command_yaw_rms_rad_s=command_yaw_rms,
        obstacle_contact_steps=obstacle_contact_steps,
        obstacle_contact_fraction=obstacle_contact_steps / max(step_count, 1),
        domain_mass_scale=float(domain.get("mass_scale", 1.0)),
        domain_friction=float(domain.get("friction", 1.0)),
        domain_servo_strength=float(domain.get("servo_strength", 1.0)),
        domain_servo_kp_scale=float(domain.get("servo_kp_scale", 1.0)),
        domain_servo_time_constant_s=float(domain.get("servo_time_constant_s", 0.0)),
        domain_action_delay_steps=int(domain.get("action_delay_steps", 0)),
        domain_imu_noise_scale=float(domain.get("imu_noise_scale", 1.0)),
        domain_battery_voltage=float(domain.get("battery_voltage", 0.0)),
        action_rms=action_rms,
        action_delta_rms=action_delta_rms,
        action_saturation_fraction=saturation,
        inference_latency_mean_ms=_mean(inference_latency_ms),
        inference_latency_p50_ms=_percentile(inference_latency_ms, 50),
        inference_latency_p95_ms=_percentile(inference_latency_ms, 95),
        inference_latency_maximum_ms=max(inference_latency_ms, default=0.0),
        video=str(video_path) if video_path is not None else "",
    )


def _summary(episodes: Sequence[EpisodeMetrics]) -> dict[str, Any]:
    def summarize(rows: Sequence[EpisodeMetrics]) -> dict[str, Any]:
        returns = [row.episode_return for row in rows]
        seed_returns = [
            _mean([row.episode_return for row in rows if row.seed == seed])
            for seed in sorted({row.seed for row in rows})
        ]
        successes = [float(row.success) for row in rows]
        falls = [float(row.fall) for row in rows]
        latencies = [row.inference_latency_p95_ms for row in rows]
        count = len(rows)
        return {
            "episodes": count,
            "seeds": len({row.seed for row in rows}),
            "return_mean": _mean(returns),
            "return_std": float(np.std(returns)) if returns else 0.0,
            "return_between_seed_std": (
                float(np.std(seed_returns, ddof=1)) if len(seed_returns) > 1 else 0.0
            ),
            "return_95ci_half_width": (
                1.96 * float(np.std(seed_returns, ddof=1)) / math.sqrt(len(seed_returns))
                if len(seed_returns) > 1
                else 0.0
            ),
            "success_rate": _mean(successes),
            "fall_rate": _mean(falls),
            "episode_length_mean": _mean([float(row.episode_length) for row in rows]),
            "distance_mean_m": _mean([row.distance_m for row in rows]),
            "energy_mean_j": _mean([row.energy_j for row in rows]),
            "linear_tracking_rmse_mean_m_s": _mean(
                [row.linear_tracking_rmse_m_s for row in rows]
            ),
            "yaw_tracking_rmse_mean_rad_s": _mean(
                [row.yaw_tracking_rmse_rad_s for row in rows]
            ),
            "obstacle_contact_fraction_mean": _mean(
                [row.obstacle_contact_fraction for row in rows]
            ),
            "inference_latency_p95_ms": _percentile(latencies, 95),
            "realtime_factor_mean": _mean([row.realtime_factor for row in rows]),
        }

    by_task: dict[str, Any] = {}
    for task in sorted({row.task for row in episodes}):
        by_task[task] = summarize([row for row in episodes if row.task == task])
    return {"overall": summarize(episodes), "by_task": by_task}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Task):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_reports(
    output: Path,
    *,
    config: EvaluationConfig,
    policy_name: str,
    episodes: Sequence[EpisodeMetrics],
    summary: dict[str, Any],
) -> tuple[Path, Path]:
    csv_path = output / "episodes.csv"
    csv_temporary = csv_path.with_suffix(".csv.tmp")
    with csv_temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[field.name for field in fields(EpisodeMetrics)])
        writer.writeheader()
        writer.writerows(asdict(episode) for episode in episodes)
    os.replace(csv_temporary, csv_path)

    json_path = output / "report.json"
    json_temporary = json_path.with_suffix(".json.tmp")
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "policy": policy_name,
        "config": _jsonable(asdict(config)),
        "summary": summary,
        "episodes": [asdict(episode) for episode in episodes],
    }
    json_temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(json_temporary, json_path)
    return json_path, csv_path


def evaluate_policy(
    policy: PolicyLike,
    config: EvaluationConfig,
    *,
    environment_factory: EnvironmentFactory | None = None,
) -> EvaluationResult:
    """Evaluate a policy across tasks/seeds and write machine-readable artifacts."""

    name = evaluation_name(policy)
    output = _output_directory(config, name)
    videos_directory = output / "videos"
    if config.video_episodes_per_task:
        videos_directory.mkdir(parents=True, exist_ok=True)
    make_environment = environment_factory or (
        lambda task, render_mode: _make_environment(config, task, render_mode)
    )
    episode_rows: list[EpisodeMetrics] = []
    extension = config.video_extension.lower().lstrip(".")
    try:
        for task in config.tasks:
            video_count = 0
            for seed in config.seeds:
                for episode_index in range(config.episodes_per_seed):
                    record = video_count < config.video_episodes_per_task
                    video_path = (
                        videos_directory
                        / f"{task.value}-seed{seed}-episode{episode_index}.{extension}"
                        if record
                        else None
                    )
                    environment = make_environment(task, "rgb_array" if record else None)
                    try:
                        row = _evaluate_episode(
                            policy,
                            policy_name=name,
                            task=task,
                            seed=seed,
                            episode_index=episode_index,
                            environment=environment,
                            deterministic=config.deterministic,
                            video_path=video_path,
                            video_fps=config.video_fps,
                            video_frame_stride=config.video_frame_stride,
                            video_view=config.video_view,
                        )
                    finally:
                        environment.close()
                    episode_rows.append(row)
                    video_count += int(record)
    except Exception:
        # Preserve any completed videos for diagnosis, but do not emit a report
        # that could be mistaken for a complete benchmark.
        raise
    summary = _summary(episode_rows)
    json_report, csv_report = _write_reports(
        output,
        config=config,
        policy_name=name,
        episodes=episode_rows,
        summary=summary,
    )
    return EvaluationResult(
        output_directory=output,
        json_report=json_report,
        csv_report=csv_report,
        summary=summary,
        episodes=tuple(episode_rows),
    )


def evaluate_ppo_checkpoint(
    checkpoint: str | Path,
    config: EvaluationConfig,
    *,
    device: str = "auto",
    vecnormalize_path: str | Path | None = None,
) -> EvaluationResult:
    """Load a PPO checkpoint and run the standard multi-seed benchmark."""

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if vecnormalize_path is None:
        checkpoint_match = re.fullmatch(
            r"(?P<prefix>.+)_(?P<step>\d+)_steps", checkpoint_path.stem
        )
        candidates = [
            (
                checkpoint_path.with_name(
                    f"{checkpoint_match.group('prefix')}_vecnormalize_"
                    f"{checkpoint_match.group('step')}_steps.pkl"
                )
                if checkpoint_match
                else checkpoint_path.with_name("__no_callback_normalizer__")
            ),
            checkpoint_path.parent / "vecnormalize.pkl",
            checkpoint_path.parent.parent / "vecnormalize.pkl",
        ]
        vecnormalize_path = next(
            (candidate for candidate in candidates if candidate.exists()), None
        )
    normalization_environment = None
    if vecnormalize_path is not None:
        first_task = Task(config.tasks[0])
        normalization_environment = _make_environment(config, first_task, None)
    policy = SB3Policy.load(
        checkpoint,
        algorithm="ppo",
        device=device,
        environment=normalization_environment,
        vecnormalize_path=vecnormalize_path,
    )
    policy.evaluation_name = f"ppo_{Path(checkpoint).stem}"
    try:
        return evaluate_policy(policy, config)
    finally:
        policy.close()
