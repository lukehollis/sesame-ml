"""Command-line entry point for simulation, training, planning, data, and deployment."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sesame_ml.constants import JOINT_LIMITS_RAD, STAND_ANGLES_RAD
from sesame_ml.envs import SesameEnv, SesameEnvConfig, Task
from sesame_ml.policies import (
    CPGPolicy,
    FirmwareSequencePolicy,
    PolicyLike,
    SB3Policy,
    StandPolicy,
    predict_action,
)

ENVIRONMENT_IDS = {
    "SesameStand-v0": Task.STAND,
    "SesameRecovery-v0": Task.RECOVERY,
    "SesameLocomotion-v0": Task.LOCOMOTION,
    "SesameNavigation-v0": Task.NAVIGATION,
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (Path, Task)):
        return str(value.value if isinstance(value, Task) else value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=_json_default))


def _task(environment_id: str) -> Task:
    try:
        return ENVIRONMENT_IDS[environment_id]
    except KeyError as error:
        raise ValueError(f"Unknown environment {environment_id!r}") from error


def _policy(
    name: str,
    *,
    checkpoint: str | None = None,
    vecnormalize: str | None = None,
    task: Task = Task.LOCOMOTION,
    observation_mode: str = "state",
) -> PolicyLike:
    if name == "stand":
        return StandPolicy()
    if name == "cpg":
        return CPGPolicy()
    if name == "firmware":
        return FirmwareSequencePolicy()
    if name == "ppo":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for --policy ppo")
        auxiliary = None
        if vecnormalize:
            auxiliary = SesameEnv(
                task=task,
                config=SesameEnvConfig(
                    task=task,
                    observation_mode=observation_mode,
                    domain_randomization=False,
                ),
            )
        return SB3Policy.load(
            checkpoint,
            algorithm="ppo",
            environment=auxiliary,
            vecnormalize_path=vecnormalize,
        )
    raise ValueError(f"Unknown policy {name!r}")


def _close_policy(policy: Any) -> None:
    close = getattr(policy, "close", None)
    if callable(close):
        close()


def command_validate_model(args: argparse.Namespace) -> int:
    import mujoco

    from sesame_ml.model import joint_qpos_addresses, load_model, make_data

    model = load_model()
    data = make_data(model)
    steps = round(args.seconds / model.opt.timestep)
    minimum_height = float("inf")
    for _ in range(steps):
        mujoco.mj_step(model, data)
        minimum_height = min(minimum_height, float(data.qpos[2]))
    joint_error = data.qpos[joint_qpos_addresses(model)] - STAND_ANGLES_RAD
    result = {
        "status": "ok",
        "mujoco_version": mujoco.__version__,
        "model": "sesame_quadruped",
        "nq": model.nq,
        "nv": model.nv,
        "actuators": model.nu,
        "bodies": model.nbody,
        "geometries": model.ngeom,
        "total_dynamic_mass_kg": float(model.body_mass.sum()),
        "settled_height_m": float(data.qpos[2]),
        "minimum_height_m": minimum_height,
        "maximum_joint_error_rad": float(np.max(np.abs(joint_error))),
        "contacts": int(data.ncon),
        "simulated_seconds": float(data.time),
    }
    if minimum_height < 0.04 or not np.all(np.isfinite(data.qpos)):
        result["status"] = "failed"
        _print_json(result)
        return 1
    _print_json(result)
    return 0


def _evaluation_config(args: argparse.Namespace, tasks: tuple[Task, ...]) -> Any:
    from sesame_ml.evaluation import EvaluationConfig

    seeds = tuple(range(args.seed, args.seed + args.episodes))
    output = Path(args.output)
    return EvaluationConfig(
        tasks=tasks,
        seeds=seeds,
        episodes_per_seed=1,
        output_directory=output.parent,
        run_name=output.name,
        domain_randomization=not args.no_randomization,
        observation_mode=args.observation_mode,
        video_episodes_per_task=args.videos,
        video_view=args.video_view,
    )


def command_rollout(args: argparse.Namespace) -> int:
    from sesame_ml.evaluation import evaluate_policy

    task = _task(args.env)
    policy = _policy(
        args.policy,
        checkpoint=args.checkpoint,
        vecnormalize=args.vecnormalize,
        task=task,
        observation_mode=args.observation_mode,
    )
    try:
        result = evaluate_policy(policy, _evaluation_config(args, (task,)))
    finally:
        _close_policy(policy)
    _print_json(
        {
            "report": result.json_report,
            "episodes_csv": result.csv_report,
            "summary": result.summary,
        }
    )
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    from sesame_ml.evaluation import evaluate_policy

    tasks = tuple(_task(value) for value in args.env)
    policy = _policy(
        args.policy,
        checkpoint=args.checkpoint,
        vecnormalize=args.vecnormalize,
        task=tasks[0],
        observation_mode=args.observation_mode,
    )
    try:
        result = evaluate_policy(policy, _evaluation_config(args, tasks))
    finally:
        _close_policy(policy)
    _print_json({"report": result.json_report, "summary": result.summary})
    return 0


def command_train_ppo(args: argparse.Namespace) -> int:
    from sesame_ml.training import PPOConfig, train_ppo

    output = Path(args.output).expanduser()
    config = PPOConfig(
        task=_task(args.env),
        total_timesteps=args.steps,
        number_of_environments=args.num_envs,
        seed=args.seed,
        output_directory=output.parent,
        run_name=output.name,
        resume_from=args.resume,
        resume_vecnormalize_from=args.resume_vecnormalize,
        device=args.device,
        vector_backend=args.vector_backend,
        domain_randomization=not args.no_randomization,
        observation_mode=args.observation_mode,
        privileged_joint_state=args.privileged_joint_state,
        rollout_steps=args.rollout_steps,
        batch_size=args.batch_size,
        checkpoint_frequency=args.checkpoint_frequency,
        evaluation_frequency=args.evaluation_frequency,
        evaluation_episodes=args.evaluation_episodes,
        progress_bar=args.progress,
    )
    result = train_ppo(config)
    _print_json(result)
    return 0


def command_train_mjx(args: argparse.Namespace) -> int:
    try:
        from sesame_ml.playground import PPOConfig, train
    except ImportError as error:
        raise RuntimeError(
            "MJX training requires `uv sync --extra playground-cuda` on the CUDA host"
        ) from error
    output = Path(args.output).expanduser()
    result = train(
        PPOConfig(
            num_timesteps=args.steps,
            num_envs=args.num_envs,
            seed=args.seed,
            impl=args.impl,
            checkpoint_root=str(output.parent),
            run_name=output.name,
        )
    )
    _print_json(
        {
            "run_directory": result.run_directory,
            "checkpoint_directory": result.checkpoint_directory,
            "metrics": result.metrics,
        }
    )
    return 0


def command_plan(args: argparse.Namespace) -> int:
    from sesame_ml.planning import CEMConfig, MuJoCoCEMPlanner

    task = _task(args.env)
    env = SesameEnv(task=task, domain_randomization=not args.no_randomization)
    _, info = env.reset(seed=args.seed)
    planner = MuJoCoCEMPlanner(
        env,
        CEMConfig(
            horizon=args.horizon,
            population=args.population,
            elites=args.elites,
            iterations=args.iterations,
            chunk_steps=args.chunk_steps,
            seed=args.seed,
        ),
    )
    plans = []
    terminated = truncated = False
    try:
        for _ in range(args.chunks):
            plan = planner.plan()
            for action in plan.actions:
                _, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            plans.append(
                {
                    "predicted_return": plan.predicted_return,
                    "elapsed_s": plan.elapsed_s,
                    "population_std": plan.population_std,
                }
            )
            if terminated or truncated:
                break
    finally:
        env.close()
    _print_json(
        {
            "plans": plans,
            "terminated": terminated,
            "truncated": truncated,
            "final": info,
        }
    )
    return 0


def command_view(args: argparse.Namespace) -> int:
    if sys.platform == "darwin":
        from mujoco import viewer

        if viewer._MJPYTHON is None:  # noqa: SLF001 - required by MuJoCo's macOS launcher.
            launcher = shutil.which("mjpython")
            if launcher is None:
                raise RuntimeError("MuJoCo viewer on macOS requires the mjpython launcher")
            os.execv(launcher, [launcher, "-m", "sesame_ml.cli", *sys.argv[1:]])
    task = _task(args.env)
    env = SesameEnv(task=task, render_mode="human", domain_randomization=False)
    policy = _policy(args.policy, checkpoint=args.checkpoint, task=task)
    observation, _ = env.reset(seed=args.seed)
    policy.reset(seed=args.seed)
    try:
        while True:
            started = time.perf_counter()
            action = predict_action(policy, observation)
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                observation, _ = env.reset()
                policy.reset()
            viewer = env._viewer
            if viewer is not None and not viewer.is_running():
                break
            time.sleep(max(0.0, env.config.control_dt - (time.perf_counter() - started)))
    except KeyboardInterrupt:
        pass
    finally:
        _close_policy(policy)
        env.close()
    return 0


def _decode_jpeg(value: bytes | None) -> np.ndarray | None:
    if not value:
        return None
    with Image.open(io.BytesIO(value)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _quat_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat / max(float(np.linalg.norm(quat)), 1e-12)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class _ResidualPolicyServerBridge:
    def __init__(
        self,
        policy: PolicyLike,
        *,
        action_scale_rad: float = 0.58,
        chunk_steps: int = 10,
        pixels: bool = False,
    ) -> None:
        self.policy = policy
        self.action_scale_rad = action_scale_rad
        self.chunk_steps = chunk_steps
        self.pixels = pixels
        self._last_action: dict[str, np.ndarray] = {}
        self._expected_camera_size = self._policy_camera_size(policy)

    @staticmethod
    def _policy_camera_size(policy: PolicyLike) -> tuple[int, int] | None:
        model = getattr(policy, "model", None)
        observation_space = getattr(model, "observation_space", None)
        spaces = getattr(observation_space, "spaces", None)
        image_space = spaces.get("rgb") if spaces is not None else None
        shape = tuple(getattr(image_space, "shape", ()))
        if len(shape) != 3:
            return None
        if shape[0] in {1, 3, 4}:  # SB3's VecTransposeImage advertises CHW.
            return int(shape[2]), int(shape[1])
        if shape[2] in {1, 3, 4}:
            return int(shape[1]), int(shape[0])
        return None

    def __call__(self, observation: Any) -> Any:
        from sesame_ml.transport import PolicyOutput

        quaternion = np.asarray(observation.imu_quaternion or (1, 0, 0, 0), dtype=np.float64)
        up = _quat_matrix(quaternion).T @ np.asarray([0.0, 0.0, 1.0])
        gyro = np.asarray(observation.imu_gyro or (0, 0, 0), dtype=np.float64)
        acceleration = np.asarray(
            observation.imu_acceleration or (0, 0, 9.81), dtype=np.float64
        ) / 9.81
        joints = np.asarray(observation.joint_position, dtype=np.float64)
        command = np.asarray(observation.command_velocity or (0, 0, 0), dtype=np.float64)
        last_action = self._last_action.setdefault(observation.robot_id, np.zeros(8))
        targets: list[tuple[float, ...]] = []
        for _ in range(self.chunk_steps):
            state = np.concatenate(
                [
                    up,
                    acceleration,
                    gyro,
                    joints - STAND_ANGLES_RAD,
                    np.zeros(8),
                    last_action,
                    command / np.asarray([0.25, 0.25, 1.4]),
                    np.zeros(4),
                    np.zeros(3),
                ]
            ).astype(np.float32)
            policy_observation: Any = state
            if self.pixels:
                rgb = _decode_jpeg(observation.image_jpeg)
                if rgb is None:
                    raise ValueError("pixel policy requires a camera JPEG")
                if self._expected_camera_size is not None and (
                    rgb.shape[1], rgb.shape[0]
                ) != self._expected_camera_size:
                    rgb = np.asarray(
                        Image.fromarray(rgb).resize(
                            self._expected_camera_size, Image.Resampling.BILINEAR
                        ),
                        dtype=np.uint8,
                    )
                policy_observation = {"proprio": state, "rgb": rgb}
            action = predict_action(self.policy, policy_observation)
            target = np.clip(
                STAND_ANGLES_RAD + self.action_scale_rad * action,
                JOINT_LIMITS_RAD[:, 0],
                JOINT_LIMITS_RAD[:, 1],
            )
            targets.append(tuple(float(value) for value in target))
            last_action = action
            joints = target
        self._last_action[observation.robot_id] = last_action
        return PolicyOutput(
            targets=tuple(targets),
            control_period_s=0.02,
            valid_for_s=0.35,
            metadata={"policy": self.policy.__class__.__name__},
        )


async def _serve(args: argparse.Namespace) -> None:
    from sesame_ml.transport import PolicyWebSocketServer

    close: Any = None
    if args.policy in {"openpi", "groot"}:
        from sesame_ml.integrations import (
            GrootRemotePolicy,
            OpenPIRemotePolicy,
            RemotePolicyBridge,
        )

        if args.policy == "openpi":
            remote = OpenPIRemotePolicy(
                host=args.backend_host,
                port=8000 if args.backend_port is None else args.backend_port,
            )
        else:
            remote = GrootRemotePolicy(
                host=args.backend_host,
                port=5555 if args.backend_port is None else args.backend_port,
            )
        callback = RemotePolicyBridge(remote, valid_for_s=args.valid_for)
        close = callback.close
    else:
        policy = _policy(
            args.policy,
            checkpoint=args.checkpoint,
            vecnormalize=args.vecnormalize,
            observation_mode="pixels" if args.pixels else "state",
        )
        callback = _ResidualPolicyServerBridge(
            policy, chunk_steps=args.chunk_steps, pixels=args.pixels
        )
        def close() -> None:
            _close_policy(policy)
    server = PolicyWebSocketServer(callback, host=args.host, port=args.port)
    print(f"Sesame policy server listening on ws://{args.host}:{args.port}", flush=True)
    try:
        await server.serve_forever()
    finally:
        close()


def command_serve(args: argparse.Namespace) -> int:
    try:
        asyncio.run(_serve(args))
    except KeyboardInterrupt:
        pass
    return 0


async def _sim_client(args: argparse.Namespace) -> None:
    from sesame_ml.runtime import SimulatedRobotRuntime
    from sesame_ml.transport import RobotWebSocketClient

    task = _task(args.env)
    env = SesameEnv(
        task=task,
        config=SesameEnvConfig(
            task=task,
            observation_mode="pixels",
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            domain_randomization=not args.no_randomization,
        ),
    )
    runtime = SimulatedRobotRuntime(env, robot_id=args.robot_id, seed=args.seed)
    client = RobotWebSocketClient(
        uri=args.uri,
        robot_id=args.robot_id,
        observation_source=runtime.observation_source,
        action_sink=runtime.action_sink,
        observation_hz=args.observation_hz,
        control_hz=50,
        watchdog_timeout_s=0.25,
        safe_fallback=tuple(float(value) for value in STAND_ANGLES_RAD),
    )
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, client.stop)
        except NotImplementedError:
            pass
    try:
        await client.run()
    finally:
        _print_json({"runtime": runtime.metrics, "transport": client.metrics.snapshot()})
        runtime.close()


def command_sim_client(args: argparse.Namespace) -> int:
    try:
        asyncio.run(_sim_client(args))
    except KeyboardInterrupt:
        pass
    return 0


def command_orange_client(args: argparse.Namespace) -> int:
    from sesame_ml.hardware import (
        BNO085ImuReader,
        OrangePiRobotRuntime,
        PCA9685ServoController,
        ServoCalibration,
        V4L2Camera,
    )
    from sesame_ml.hardware.orange_pi import run_robot_client

    calibration = ServoCalibration.from_yaml(args.calibration)
    camera_device: int | str = int(args.camera) if args.camera.isdigit() else args.camera
    camera = V4L2Camera(
        camera_device, width=args.camera_width, height=args.camera_height, fps=15
    )
    servos = PCA9685ServoController.open(
        args.i2c_bus, address=int(args.pca9685_address, 0), calibration=calibration
    )
    imu = BNO085ImuReader(args.i2c_bus, address=int(args.imu_address, 0))
    runtime = OrangePiRobotRuntime(
        robot_id=args.robot_id,
        camera=camera,
        servos=servos,
        imu=imu,
        fallback_behavior=args.fallback,
    )
    asyncio.run(run_robot_client(runtime, policy_uri=args.uri))
    return 0


def command_collect(args: argparse.Namespace) -> int:
    from sesame_ml.data import EpisodeWriter

    task = _task(args.env)
    policy = _policy(args.policy, checkpoint=args.checkpoint, task=task)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    written = []
    try:
        for episode_index in range(args.episodes):
            env = SesameEnv(
                task=task,
                config=SesameEnvConfig(
                    task=task,
                    observation_mode="pixels",
                    camera_width=args.camera_width,
                    camera_height=args.camera_height,
                    domain_randomization=not args.no_randomization,
                ),
            )
            observation, info = env.reset(seed=args.seed + episode_index)
            policy.reset(seed=args.seed + episode_index)
            episode_path = output / f"episode-{episode_index:06d}"
            try:
                with EpisodeWriter(
                    episode_path,
                    instruction=info["instruction"],
                    metadata={
                        "task": task.value,
                        "policy": args.policy,
                        "seed": args.seed + episode_index,
                        "simulated": True,
                        "domain": info["domain"],
                    },
                ) as writer:
                    terminated = truncated = False
                    step = 0
                    upright_sum = 0.0
                    linear_error_squared_sum = 0.0
                    yaw_error_squared_sum = 0.0
                    while not (terminated or truncated):
                        state = env.servo_targets
                        rgb = observation["rgb"]
                        imu_quaternion, imu_gyro, imu_acceleration = env.deployment_imu()
                        command_velocity = info["command"]
                        action = predict_action(policy, observation)
                        if isinstance(policy, FirmwareSequencePolicy):
                            absolute_action = policy.servo_targets_rad
                            next_observation, reward, terminated, truncated, next_info = (
                                env.step_servo_targets(absolute_action)
                            )
                        else:
                            absolute_action = np.clip(
                                STAND_ANGLES_RAD + env.config.action_scale_rad * action,
                                JOINT_LIMITS_RAD[:, 0],
                                JOINT_LIMITS_RAD[:, 1],
                            )
                            next_observation, reward, terminated, truncated, next_info = env.step(
                                action
                            )
                        command = np.asarray(next_info["command"], dtype=np.float64)
                        velocity = np.asarray(next_info["base_velocity_m_s"], dtype=np.float64)
                        upright_sum += float(next_info["upright"])
                        linear_error_squared_sum += float(
                            np.sum(np.square(velocity[:2] - command[:2]))
                        )
                        yaw_error_squared_sum += float(
                            (float(next_info["yaw_rate_rad_s"]) - command[2]) ** 2
                        )
                        strict_success = bool(next_info["success"])
                        if terminated or truncated:
                            count = step + 1
                            if task is Task.STAND:
                                strict_success = bool(truncated and not next_info["fallen"])
                            elif task is Task.RECOVERY:
                                strict_success = bool(
                                    next_info["success"] and not next_info["fallen"]
                                )
                            elif task is Task.LOCOMOTION:
                                strict_success = bool(
                                    truncated
                                    and not next_info["fallen"]
                                    and upright_sum / count >= 0.75
                                    and np.sqrt(linear_error_squared_sum / count) <= 0.06
                                    and np.sqrt(yaw_error_squared_sum / count) <= 0.45
                                )
                        writer.append(
                            state_rad=state,
                            rgb=rgb,
                            action_rad=absolute_action,
                            reward=reward,
                            terminated=terminated,
                            truncated=truncated,
                            timestamp_ns=step * round(env.config.control_dt * 1e9),
                            imu_quaternion=imu_quaternion,
                            imu_gyro=imu_gyro,
                            imu_acceleration=imu_acceleration,
                            command_velocity=command_velocity,
                            next_success=strict_success,
                            next_fallen=next_info["fallen"],
                            next_obstacle_contact=next_info["obstacle_contact"],
                            next_out_of_bounds=next_info["out_of_bounds"],
                            next_goal_distance_m=next_info["goal_distance_m"],
                        )
                        observation = next_observation
                        info = next_info
                        step += 1
                written.append(episode_path)
            finally:
                env.close()
    finally:
        _close_policy(policy)
    _print_json({"episodes": written})
    return 0


def command_export_dataset(args: argparse.Namespace) -> int:
    from sesame_ml.data import export_to_lerobot

    result = export_to_lerobot(
        args.episodes,
        repo_id=args.repo_id,
        output_dir=args.output,
        overwrite=args.overwrite,
        require_groot_v2=args.groot_v2,
        add_groot_metadata=args.groot_v2,
    )
    _print_json(result)
    return 0


def _add_policy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--policy", choices=("stand", "cpg", "firmware", "ppo"), default="cpg")
    parser.add_argument("--checkpoint")
    parser.add_argument("--vecnormalize")


def _add_eval_arguments(parser: argparse.ArgumentParser) -> None:
    _add_policy_arguments(parser)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="artifacts/evaluation")
    parser.add_argument("--videos", type=int, default=0)
    parser.add_argument(
        "--video-view", choices=("external", "front", "split"), default="external"
    )
    parser.add_argument("--observation-mode", choices=("state", "pixels"), default="state")
    parser.add_argument("--no-randomization", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sesame-ml", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-model", help="compile and settle the MJCF")
    validate.add_argument("--seconds", type=float, default=2.0)
    validate.set_defaults(handler=command_validate_model)

    rollout = subparsers.add_parser("rollout", help="evaluate one task")
    rollout.add_argument("--env", choices=tuple(ENVIRONMENT_IDS), default="SesameLocomotion-v0")
    _add_eval_arguments(rollout)
    rollout.set_defaults(handler=command_rollout)

    evaluate = subparsers.add_parser("evaluate", help="run a multi-task, multi-seed benchmark")
    evaluate.add_argument(
        "--env", choices=tuple(ENVIRONMENT_IDS), action="append", default=None
    )
    _add_eval_arguments(evaluate)
    evaluate.set_defaults(handler=command_evaluate)

    train = subparsers.add_parser("train-ppo", help="train or resume SB3 PPO")
    train.add_argument("--env", choices=tuple(ENVIRONMENT_IDS), default="SesameLocomotion-v0")
    train.add_argument("--steps", type=int, default=5_000_000)
    train.add_argument("--num-envs", type=int, default=8)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--output", default="runs/locomotion-ppo")
    train.add_argument("--resume")
    train.add_argument("--resume-vecnormalize")
    train.add_argument("--device", default="auto")
    train.add_argument("--vector-backend", choices=("auto", "dummy", "subproc"), default="auto")
    train.add_argument("--observation-mode", choices=("state", "pixels"), default="state")
    train.add_argument("--privileged-joint-state", action="store_true")
    train.add_argument("--no-randomization", action="store_true")
    train.add_argument("--rollout-steps", type=int, default=1024)
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument("--checkpoint-frequency", type=int, default=250_000)
    train.add_argument("--evaluation-frequency", type=int, default=100_000)
    train.add_argument("--evaluation-episodes", type=int, default=10)
    train.add_argument("--progress", action="store_true")
    train.set_defaults(handler=command_train_ppo)

    mjx = subparsers.add_parser("train-mjx", help="train Playground/Brax PPO on JAX or CUDA")
    mjx.add_argument("--steps", type=int, default=100_000_000)
    mjx.add_argument("--num-envs", type=int, default=4096)
    mjx.add_argument("--seed", type=int, default=0)
    mjx.add_argument("--impl", choices=("jax", "warp"), default="warp")
    mjx.add_argument("--output", default="runs/mjx-locomotion")
    mjx.set_defaults(handler=command_train_mjx)

    plan = subparsers.add_parser("plan", help="run receding-horizon MuJoCo CEM")
    plan.add_argument("--env", choices=tuple(ENVIRONMENT_IDS), default="SesameLocomotion-v0")
    plan.add_argument("--seed", type=int, default=0)
    plan.add_argument("--horizon", type=int, default=18)
    plan.add_argument("--population", type=int, default=192)
    plan.add_argument("--elites", type=int, default=24)
    plan.add_argument("--iterations", type=int, default=4)
    plan.add_argument("--chunk-steps", type=int, default=5)
    plan.add_argument("--chunks", type=int, default=10)
    plan.add_argument("--no-randomization", action="store_true")
    plan.set_defaults(handler=command_plan)

    view = subparsers.add_parser("view", help="run a policy in the interactive viewer")
    view.add_argument("--env", choices=tuple(ENVIRONMENT_IDS), default="SesameLocomotion-v0")
    view.add_argument("--seed", type=int, default=0)
    _add_policy_arguments(view)
    view.set_defaults(handler=command_view)

    serve = subparsers.add_parser("serve", help="host a local, PPO, OpenPI, or GR00T policy")
    serve.add_argument(
        "--policy", choices=("stand", "cpg", "firmware", "ppo", "openpi", "groot"), default="cpg"
    )
    serve.add_argument("--checkpoint")
    serve.add_argument("--vecnormalize")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--backend-host", default="127.0.0.1")
    serve.add_argument("--backend-port", type=int, default=None)
    serve.add_argument("--chunk-steps", type=int, default=10)
    serve.add_argument("--valid-for", type=float, default=1.0)
    serve.add_argument("--pixels", action="store_true")
    serve.set_defaults(handler=command_serve)

    sim_client = subparsers.add_parser("sim-client", help="drive MuJoCo through the Wi-Fi path")
    sim_client.add_argument("--uri", required=True)
    sim_client.add_argument("--robot-id", default="sesame-sim-001")
    sim_client.add_argument("--env", choices=tuple(ENVIRONMENT_IDS), default="SesameNavigation-v0")
    sim_client.add_argument("--seed", type=int, default=0)
    sim_client.add_argument("--observation-hz", type=float, default=15)
    sim_client.add_argument("--camera-width", type=int, default=160)
    sim_client.add_argument("--camera-height", type=int, default=120)
    sim_client.add_argument("--no-randomization", action="store_true")
    sim_client.set_defaults(handler=command_sim_client)

    orange = subparsers.add_parser("orange-client", help="run the physical Orange Pi client")
    orange.add_argument("--uri", required=True)
    orange.add_argument("--robot-id", required=True)
    orange.add_argument("--camera", default="0")
    orange.add_argument("--camera-width", type=int, default=320)
    orange.add_argument("--camera-height", type=int, default=240)
    orange.add_argument("--i2c-bus", type=int, default=3)
    orange.add_argument("--pca9685-address", default="0x40")
    orange.add_argument("--imu-address", default="0x4a")
    orange.add_argument("--calibration", required=True)
    orange.add_argument("--fallback", choices=("disable", "stand"), default="disable")
    orange.set_defaults(handler=command_orange_client)

    collect = subparsers.add_parser("collect", help="record lossless simulator demonstrations")
    collect.add_argument("--env", choices=tuple(ENVIRONMENT_IDS), default="SesameNavigation-v0")
    collect.add_argument("--episodes", type=int, default=10)
    collect.add_argument("--seed", type=int, default=0)
    collect.add_argument("--output", default="artifacts/episodes")
    collect.add_argument("--camera-width", type=int, default=160)
    collect.add_argument("--camera-height", type=int, default=120)
    collect.add_argument("--no-randomization", action="store_true")
    _add_policy_arguments(collect)
    collect.set_defaults(handler=command_collect)

    export = subparsers.add_parser("export-dataset", help="convert episodes to LeRobot")
    export.add_argument("episodes", nargs="+")
    export.add_argument("--repo-id", required=True)
    export.add_argument("--output")
    export.add_argument("--groot-v2", action="store_true")
    export.add_argument("--overwrite", action="store_true")
    export.set_defaults(handler=command_export_dataset)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "evaluate" and args.env is None:
        args.env = ["SesameStand-v0", "SesameLocomotion-v0", "SesameNavigation-v0"]
    if hasattr(args, "episodes") and isinstance(args.episodes, int) and args.episodes < 1:
        parser.error("--episodes must be positive")
    try:
        return int(args.handler(args))
    except (ValueError, FileNotFoundError, RuntimeError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
