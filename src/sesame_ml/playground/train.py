"""Brax PPO GPU training entrypoint for the Sesame Playground environment."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import functools
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import jax
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper
except ImportError as exc:  # pragma: no cover - guarded by the lazy package boundary.
    raise ImportError(
        "sesame_ml.playground.train requires a GPU JAX build, Brax, and "
        "MuJoCo Playground; install sesame-ml[playground-cuda]."
    ) from exc

from sesame_ml.playground.environment import SesameLocomotion, default_config
from sesame_ml.playground.randomize import make_domain_randomizer


@dataclass(frozen=True)
class PPOConfig:
    """Production-scale asymmetric actor/critic PPO defaults."""

    impl: str = "jax"
    seed: int = 1
    num_timesteps: int = 100_000_000
    num_envs: int = 8_192
    num_eval_envs: int = 128
    num_evals: int = 10
    episode_length: int = 1_000
    reward_scaling: float = 1.0
    normalize_observations: bool = True
    action_repeat: int = 1
    unroll_length: int = 20
    num_minibatches: int = 32
    num_updates_per_batch: int = 4
    batch_size: int = 256
    discounting: float = 0.97
    learning_rate: float = 3e-4
    entropy_cost: float = 1e-2
    max_grad_norm: float = 1.0
    clipping_epsilon: float = 0.3
    policy_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
    value_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
    domain_randomization: bool = True
    checkpoint_root: str = "runs/playground"
    run_name: str | None = None
    restore_checkpoint_path: str | None = None
    require_accelerator: bool = True

    def __post_init__(self) -> None:
        if self.impl not in {"jax", "warp"}:
            raise ValueError("impl must be 'jax' or 'warp'")
        if self.num_envs <= 0 or self.num_eval_envs <= 0:
            raise ValueError("num_envs and num_eval_envs must be positive")
        if self.num_timesteps <= 0:
            raise ValueError("num_timesteps must be positive")


@dataclass(frozen=True)
class TrainingResult:
    """Artifacts returned after PPO completes or restores a run."""

    run_directory: Path
    checkpoint_directory: Path
    make_inference_fn: Callable[..., Any]
    parameters: Any
    metrics: Mapping[str, Any]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def train(
    config: PPOConfig | None = None,
    *,
    progress_fn: Callable[[int, Mapping[str, Any]], None] | None = None,
) -> TrainingResult:
    """Train a deployable residual-action actor with Brax PPO.

    Checkpoints and exact environment/training configurations are persisted in
    a timestamped run directory.  JAX supplies MJX physics for ``impl='jax'``;
    ``impl='warp'`` selects MuJoCo Warp physics while retaining the JAX/Brax PPO
    learner.
    """

    cfg = config or PPOConfig()
    backend = jax.default_backend()
    if cfg.impl == "warp" and backend != "gpu":
        raise RuntimeError(
            f"MuJoCo Warp requires a GPU backend, but JAX selected {backend!r}. "
            "Install the CUDA Playground stack or use impl='jax'."
        )
    if cfg.require_accelerator and backend not in {"gpu", "tpu"}:
        raise RuntimeError(
            f"PPO training requires an accelerator, but JAX selected {backend!r}. "
            "Install the appropriate CUDA JAX build or explicitly set "
            "require_accelerator=False for diagnostic CPU runs."
        )

    env_cfg = default_config()
    env_cfg.impl = cfg.impl
    if cfg.impl == "warp":
        largest_batch = max(cfg.num_envs, cfg.num_eval_envs)
        env_cfg.naconmax = max(env_cfg.naconmax, 8 * largest_batch)
        env_cfg.naccdmax = max(env_cfg.naccdmax, 8 * largest_batch)
    environment = SesameLocomotion(config=env_cfg)
    eval_environment = SesameLocomotion(config=env_cfg)

    timestamp = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%d-%H%M%S")
    run_name = cfg.run_name or f"sesame-locomotion-{cfg.impl}-{timestamp}"
    run_directory = Path(cfg.checkpoint_root).expanduser().resolve() / run_name
    checkpoint_directory = run_directory / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "ppo_config.json").write_text(
        json.dumps(dataclasses.asdict(cfg), indent=2, default=_jsonable) + "\n",
        encoding="utf-8",
    )
    (run_directory / "environment_config.json").write_text(
        json.dumps(env_cfg.to_dict(), indent=2, default=_jsonable) + "\n",
        encoding="utf-8",
    )

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=cfg.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=cfg.value_hidden_layer_sizes,
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )
    randomization_fn = (
        make_domain_randomizer(environment.mj_model) if cfg.domain_randomization else None
    )

    if progress_fn is None:

        def progress_fn(step: int, metrics: Mapping[str, Any]) -> None:
            reward = metrics.get("eval/episode_reward")
            if reward is not None:
                print(f"{step}: eval/episode_reward={float(reward):.4f}")

    make_inference_fn, parameters, metrics = ppo.train(
        environment=environment,
        eval_env=eval_environment,
        num_timesteps=cfg.num_timesteps,
        num_evals=cfg.num_evals,
        reward_scaling=cfg.reward_scaling,
        episode_length=cfg.episode_length,
        normalize_observations=cfg.normalize_observations,
        action_repeat=cfg.action_repeat,
        unroll_length=cfg.unroll_length,
        num_minibatches=cfg.num_minibatches,
        num_updates_per_batch=cfg.num_updates_per_batch,
        discounting=cfg.discounting,
        learning_rate=cfg.learning_rate,
        entropy_cost=cfg.entropy_cost,
        num_envs=cfg.num_envs,
        num_eval_envs=cfg.num_eval_envs,
        batch_size=cfg.batch_size,
        max_grad_norm=cfg.max_grad_norm,
        clipping_epsilon=cfg.clipping_epsilon,
        network_factory=network_factory,
        seed=cfg.seed,
        restore_checkpoint_path=cfg.restore_checkpoint_path,
        save_checkpoint_path=checkpoint_directory,
        wrap_env_fn=functools.partial(wrapper.wrap_for_brax_training, full_reset=True),
        randomization_fn=randomization_fn,
        deterministic_eval=True,
        progress_fn=progress_fn,
    )
    return TrainingResult(
        run_directory=run_directory,
        checkpoint_directory=checkpoint_directory,
        make_inference_fn=make_inference_fn,
        parameters=parameters,
        metrics=metrics,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", choices=("jax", "warp"), default="jax")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-timesteps", type=int, default=100_000_000)
    parser.add_argument("--num-envs", type=int, default=8_192)
    parser.add_argument("--num-eval-envs", type=int, default=128)
    parser.add_argument("--num-evals", type=int, default=10)
    parser.add_argument("--checkpoint-root", default="runs/playground")
    parser.add_argument("--run-name")
    parser.add_argument("--restore-checkpoint-path")
    parser.add_argument("--no-domain-randomization", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> TrainingResult:
    args = _parser().parse_args(argv)
    config = PPOConfig(
        impl=args.impl,
        seed=args.seed,
        num_timesteps=args.num_timesteps,
        num_envs=args.num_envs,
        num_eval_envs=args.num_eval_envs,
        num_evals=args.num_evals,
        checkpoint_root=args.checkpoint_root,
        run_name=args.run_name,
        restore_checkpoint_path=args.restore_checkpoint_path,
        domain_randomization=not args.no_domain_randomization,
        require_accelerator=not args.allow_cpu,
    )
    return train(config)


if __name__ == "__main__":
    main()
