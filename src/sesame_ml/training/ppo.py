"""Reproducible, resumable PPO training for the Sesame MuJoCo tasks."""

from __future__ import annotations

import json
import os
import random
import re
import signal
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from sesame_ml.envs import SesameEnv, SesameEnvConfig, Task

VectorBackend = Literal["auto", "dummy", "subproc"]


@dataclass(frozen=True)
class PPOConfig:
    """Complete PPO run configuration.

    ``total_timesteps`` is the number of additional samples when ``resume_from``
    is set.  Checkpoint/evaluation frequencies are measured in aggregate samples
    across all workers, independent of ``number_of_environments``.
    """

    task: Task | str = Task.LOCOMOTION
    total_timesteps: int = 5_000_000
    number_of_environments: int = 8
    seed: int = 0
    output_directory: str | Path = "runs"
    run_name: str | None = None
    resume_from: str | Path | None = None
    resume_vecnormalize_from: str | Path | None = None
    device: str = "auto"
    vector_backend: VectorBackend = "auto"
    subproc_start_method: str | None = None
    domain_randomization: bool = True
    evaluation_domain_randomization: bool = False
    observation_mode: str = "state"
    privileged_joint_state: bool = False
    maximum_episode_steps: int | None = None
    normalize_observations: bool = True
    normalize_rewards: bool = True
    normalization_clip: float = 10.0
    learning_rate: float = 3.0e-4
    rollout_steps: int = 1024
    batch_size: int = 512
    epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coefficient: float = 0.005
    value_coefficient: float = 0.5
    maximum_gradient_norm: float = 0.5
    target_kl: float | None = 0.03
    network_width: int = 256
    network_depth: int = 3
    checkpoint_frequency: int = 250_000
    evaluation_frequency: int = 100_000
    evaluation_episodes: int = 10
    log_interval: int = 1
    progress_bar: bool = False
    verbose: int = 1

    def __post_init__(self) -> None:
        task = Task(self.task)
        object.__setattr__(self, "task", task)
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.number_of_environments <= 0:
            raise ValueError("number_of_environments must be positive")
        if self.rollout_steps <= 1:
            raise ValueError("rollout_steps must be greater than one")
        rollout_size = self.rollout_steps * self.number_of_environments
        if self.batch_size <= 1 or self.batch_size > rollout_size:
            raise ValueError("batch_size must be in [2, rollout_steps * number_of_environments]")
        if rollout_size % self.batch_size:
            raise ValueError(
                "batch_size must divide rollout_steps * number_of_environments to avoid "
                "truncated minibatches"
            )
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.checkpoint_frequency <= 0 or self.evaluation_frequency <= 0:
            raise ValueError("checkpoint and evaluation frequencies must be positive")
        if self.evaluation_episodes <= 0:
            raise ValueError("evaluation_episodes must be positive")
        if self.network_width <= 0 or self.network_depth <= 0:
            raise ValueError("network dimensions must be positive")
        if self.vector_backend not in {"auto", "dummy", "subproc"}:
            raise ValueError(f"unsupported vector_backend {self.vector_backend!r}")
        if self.observation_mode not in {"state", "pixels"}:
            raise ValueError("observation_mode must be 'state' or 'pixels'")
        if self.maximum_episode_steps is not None and self.maximum_episode_steps <= 0:
            raise ValueError("maximum_episode_steps must be positive when supplied")


@dataclass(frozen=True)
class TrainingResult:
    run_directory: Path
    final_checkpoint: Path
    best_checkpoint: Path | None
    vecnormalize_path: Path | None
    samples_seen: int
    resumed: bool


def _optional_sb3() -> dict[str, Any]:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import (
            BaseCallback,
            CallbackList,
            CheckpointCallback,
            EvalCallback,
        )
        from stable_baselines3.common.utils import set_random_seed
        from stable_baselines3.common.vec_env import (
            DummyVecEnv,
            SubprocVecEnv,
            VecMonitor,
            VecNormalize,
            VecTransposeImage,
        )
    except ImportError as error:  # pragma: no cover - optional dependency path
        raise ImportError(
            "PPO training requires `pip install sesame-ml[train]`"
        ) from error
    return {
        "PPO": PPO,
        "BaseCallback": BaseCallback,
        "CallbackList": CallbackList,
        "CheckpointCallback": CheckpointCallback,
        "EvalCallback": EvalCallback,
        "DummyVecEnv": DummyVecEnv,
        "SubprocVecEnv": SubprocVecEnv,
        "VecMonitor": VecMonitor,
        "VecNormalize": VecNormalize,
        "VecTransposeImage": VecTransposeImage,
        "set_random_seed": set_random_seed,
    }


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, Torch, and SB3 from one recorded value."""

    random.seed(seed)
    np.random.seed(seed)
    dependencies = _optional_sb3()
    dependencies["set_random_seed"](seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - SB3 installations include torch
        pass


def _environment_factory(
    config: PPOConfig, *, worker_index: int, evaluation: bool
) -> Any:
    seed = config.seed + (100_000 if evaluation else 0) + worker_index

    def make() -> SesameEnv:
        environment_config = SesameEnvConfig(
            task=Task(config.task),
            observation_mode=config.observation_mode,
            domain_randomization=(
                config.evaluation_domain_randomization
                if evaluation
                else config.domain_randomization
            ),
            privileged_joint_state=config.privileged_joint_state,
            maximum_episode_steps=config.maximum_episode_steps,
        )
        environment = SesameEnv(task=config.task, config=environment_config)
        environment.action_space.seed(seed)
        return environment

    return make


def make_vector_environment(
    config: PPOConfig,
    *,
    evaluation: bool = False,
    monitor_file: str | Path | None = None,
    normalization_training: bool | None = None,
) -> Any:
    """Construct the exact vectorized environment used by training/evaluation."""

    dependencies = _optional_sb3()
    count = 1 if evaluation else config.number_of_environments
    factories = [
        _environment_factory(config, worker_index=index, evaluation=evaluation)
        for index in range(count)
    ]
    backend = config.vector_backend
    if backend == "auto":
        # Subprocesses materially improve throughput once MuJoCo has enough work
        # to amortize IPC, while a single worker is easier to debug.
        backend = "dummy" if count == 1 else "subproc"
    if backend == "subproc" and count > 1:
        start_method = config.subproc_start_method
        if start_method is None:
            start_method = "forkserver" if sys.platform.startswith("linux") else "spawn"
        vector_environment = dependencies["SubprocVecEnv"](
            factories, start_method=start_method
        )
    else:
        vector_environment = dependencies["DummyVecEnv"](factories)
    vector_environment.seed(config.seed + (100_000 if evaluation else 0))
    vector_environment = dependencies["VecMonitor"](
        vector_environment,
        filename=str(monitor_file) if monitor_file is not None else None,
        info_keywords=("success",),
    )
    if config.normalize_observations or config.normalize_rewards:
        normalization_keys = ["proprio"] if config.observation_mode == "pixels" else None
        vector_environment = dependencies["VecNormalize"](
            vector_environment,
            training=(not evaluation if normalization_training is None else normalization_training),
            norm_obs=config.normalize_observations,
            norm_reward=config.normalize_rewards and not evaluation,
            clip_obs=config.normalization_clip,
            norm_obs_keys=normalization_keys,
            gamma=config.gamma,
        )
    if config.observation_mode == "pixels":
        # Apply this explicitly to both train and eval environments.  Letting PPO
        # auto-wrap only the training side would make EvalCallback's normalization
        # synchronization fail due to mismatched wrapper stacks.
        vector_environment = dependencies["VecTransposeImage"](vector_environment)
    return vector_environment


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Task):
        return value.value
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_value(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _run_directory(config: PPOConfig) -> Path:
    root = Path(config.output_directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if config.run_name:
        name = config.run_name
    else:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        name = f"ppo-{Task(config.task).value}-seed{config.seed}-{timestamp}"
    path = root / name
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"training run directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalizer_from_resume(config: PPOConfig, training_environment: Any) -> Any:
    dependencies = _optional_sb3()
    requested = config.resume_vecnormalize_from
    if requested is None and config.resume_from is not None:
        checkpoint = Path(config.resume_from).expanduser().resolve()
        checkpoint_match = re.fullmatch(r"(?P<prefix>.+)_(?P<step>\d+)_steps", checkpoint.stem)
        callback_normalizer = (
            checkpoint.with_name(
                f"{checkpoint_match.group('prefix')}_vecnormalize_"
                f"{checkpoint_match.group('step')}_steps.pkl"
            )
            if checkpoint_match
            else checkpoint.with_name("__no_callback_normalizer__")
        )
        candidates = [
            callback_normalizer,
            checkpoint.parent / "vecnormalize.pkl",
            checkpoint.parent.parent / "vecnormalize.pkl",
        ]
        requested = next((candidate for candidate in candidates if candidate.exists()), None)
    if requested is None:
        if config.resume_from is not None and (
            config.normalize_observations or config.normalize_rewards
        ):
            raise FileNotFoundError(
                "resuming a normalized policy requires its VecNormalize .pkl; "
                "pass resume_vecnormalize_from explicitly"
            )
        return training_environment
    if not (config.normalize_observations or config.normalize_rewards):
        raise ValueError("resume VecNormalize state was supplied but normalization is disabled")
    path = Path(requested).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"VecNormalize state does not exist: {path}")
    # Replace the freshly initialized wrapper with its saved running statistics.
    # In pixel mode VecTransposeImage is outside VecNormalize and must be rebuilt
    # so the train/eval wrapper stacks remain identical.
    if config.observation_mode == "pixels":
        fresh_normalizer = training_environment.venv
        base_environment = fresh_normalizer.venv
    else:
        base_environment = training_environment.venv
    restored = dependencies["VecNormalize"].load(str(path), base_environment)
    restored.training = True
    restored.norm_reward = config.normalize_rewards
    if config.observation_mode == "pixels":
        return dependencies["VecTransposeImage"](restored)
    return restored


def _save_normalizer(environment: Any, path: Path) -> Path | None:
    dependencies = _optional_sb3()
    vecnormalize_type = dependencies["VecNormalize"]
    candidate = environment
    while candidate is not None:
        if isinstance(candidate, vecnormalize_type):
            candidate.save(str(path))
            return path
        candidate = getattr(candidate, "venv", None)
    return None


def train_ppo(config: PPOConfig) -> TrainingResult:
    """Run (or resume) PPO and persist all artifacts needed for deployment.

    The function is suitable for multi-million-step runs.  A SIGINT/keyboard
    interrupt always writes ``interrupted_model.zip`` and normalization state
    before propagating the interruption.
    """

    dependencies = _optional_sb3()
    seed_everything(config.seed)
    run_directory = _run_directory(config)
    checkpoint_directory = run_directory / "checkpoints"
    evaluation_directory = run_directory / "evaluation"
    best_directory = run_directory / "best"
    tensorboard_directory = run_directory / "tensorboard"
    for directory in (
        checkpoint_directory,
        evaluation_directory,
        best_directory,
        tensorboard_directory,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    _write_json(
        run_directory / "config.json",
        {
            **asdict(config),
            "created_at": datetime.now(UTC).isoformat(),
            "python": sys.version,
            "platform": sys.platform,
        },
    )

    training_environment = make_vector_environment(
        config, monitor_file=run_directory / "training.monitor.csv"
    )
    if config.resume_from is not None:
        training_environment = _normalizer_from_resume(config, training_environment)
    evaluation_environment = make_vector_environment(
        config, evaluation=True, normalization_training=False
    )

    policy_name = "MultiInputPolicy" if config.observation_mode == "pixels" else "MlpPolicy"
    policy_kwargs = {
        "net_arch": {
            "pi": [config.network_width] * config.network_depth,
            "vf": [config.network_width] * config.network_depth,
        }
    }
    resumed = config.resume_from is not None
    if resumed:
        model = dependencies["PPO"].load(
            str(Path(config.resume_from).expanduser()),
            env=training_environment,
            device=config.device,
            tensorboard_log=str(tensorboard_directory),
            seed=config.seed,
            learning_rate=config.learning_rate,
            n_steps=config.rollout_steps,
            batch_size=config.batch_size,
            n_epochs=config.epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.entropy_coefficient,
            vf_coef=config.value_coefficient,
            max_grad_norm=config.maximum_gradient_norm,
            target_kl=config.target_kl,
        )
        model.set_random_seed(config.seed)
    else:
        model = dependencies["PPO"](
            policy_name,
            training_environment,
            learning_rate=config.learning_rate,
            n_steps=config.rollout_steps,
            batch_size=config.batch_size,
            n_epochs=config.epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.entropy_coefficient,
            vf_coef=config.value_coefficient,
            max_grad_norm=config.maximum_gradient_norm,
            target_kl=config.target_kl,
            tensorboard_log=str(tensorboard_directory),
            policy_kwargs=policy_kwargs,
            verbose=config.verbose,
            seed=config.seed,
            device=config.device,
        )

    # EvalCallback synchronizes VecNormalize statistics before every evaluation.
    checkpoint_callback = dependencies["CheckpointCallback"](
        save_freq=max(1, config.checkpoint_frequency // config.number_of_environments),
        save_path=str(checkpoint_directory),
        name_prefix="sesame_ppo",
        save_vecnormalize=config.normalize_observations or config.normalize_rewards,
        verbose=config.verbose,
    )

    class SaveBestNormalizer(dependencies["BaseCallback"]):  # type: ignore[misc, valid-type]
        def _on_step(self) -> bool:
            _save_normalizer(self.model.get_env(), best_directory / "vecnormalize.pkl")
            return True

    evaluation_callback = dependencies["EvalCallback"](
        evaluation_environment,
        callback_on_new_best=SaveBestNormalizer(verbose=0),
        best_model_save_path=str(best_directory),
        log_path=str(evaluation_directory),
        eval_freq=max(1, config.evaluation_frequency // config.number_of_environments),
        n_eval_episodes=config.evaluation_episodes,
        deterministic=True,
        render=False,
        verbose=config.verbose,
    )
    callbacks = dependencies["CallbackList"]([checkpoint_callback, evaluation_callback])

    previous_sigterm = signal.getsignal(signal.SIGTERM)
    interrupted = False
    failed = False

    def request_stop(signum: int, frame: Any) -> None:
        del signum, frame
        nonlocal interrupted
        interrupted = True
        raise KeyboardInterrupt

    # Only the main thread may install signal handlers (normal CLI training path).
    import threading

    installed_signal = threading.current_thread() is threading.main_thread()
    if installed_signal:
        signal.signal(signal.SIGTERM, request_stop)
    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            log_interval=config.log_interval,
            tb_log_name="ppo",
            reset_num_timesteps=not resumed,
            progress_bar=config.progress_bar,
        )
    except KeyboardInterrupt:
        interrupted = True
        interrupted_checkpoint = run_directory / "interrupted_model"
        model.save(str(interrupted_checkpoint))
        _save_normalizer(training_environment, run_directory / "vecnormalize.pkl")
        _write_json(
            run_directory / "status.json",
            {
                "status": "interrupted",
                "samples_seen": model.num_timesteps,
                "finished_at": datetime.now(UTC).isoformat(),
            },
        )
        raise
    except Exception as error:
        failed = True
        _write_json(
            run_directory / "status.json",
            {
                "status": "failed",
                "samples_seen": model.num_timesteps,
                "error_type": type(error).__name__,
                "error": str(error),
                "finished_at": datetime.now(UTC).isoformat(),
            },
        )
        raise
    finally:
        if installed_signal:
            signal.signal(signal.SIGTERM, previous_sigterm)
        if interrupted or failed:
            evaluation_environment.close()
            training_environment.close()

    final_checkpoint = run_directory / "final_model"
    model.save(str(final_checkpoint))
    final_checkpoint = final_checkpoint.with_suffix(".zip")
    normalizer_path = _save_normalizer(
        training_environment, run_directory / "vecnormalize.pkl"
    )
    best_checkpoint = best_directory / "best_model.zip"
    if not best_checkpoint.exists():
        best_checkpoint = None
    _write_json(
        run_directory / "status.json",
        {
            "status": "complete",
            "samples_seen": model.num_timesteps,
            "additional_samples_requested": config.total_timesteps,
            "final_checkpoint": final_checkpoint,
            "best_checkpoint": best_checkpoint,
            "vecnormalize": normalizer_path,
            "finished_at": datetime.now(UTC).isoformat(),
        },
    )
    samples_seen = int(model.num_timesteps)
    evaluation_environment.close()
    training_environment.close()
    return TrainingResult(
        run_directory=run_directory,
        final_checkpoint=final_checkpoint,
        best_checkpoint=best_checkpoint,
        vecnormalize_path=normalizer_path,
        samples_seen=samples_seen,
        resumed=resumed,
    )
