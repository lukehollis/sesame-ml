"""Optional MuJoCo Playground/MJX integration.

The base :mod:`sesame_ml` package intentionally does not depend on JAX, Brax, or
MuJoCo Playground.  Objects in this module are resolved lazily so importing the
rest of Sesame ML remains safe on the robot and on developer machines without
the accelerator training stack installed.
"""

from __future__ import annotations

from importlib import import_module, util
from typing import Any

_RUNTIME_DEPENDENCIES = ("jax", "ml_collections", "mujoco_playground")
_TRAINING_DEPENDENCIES = (*_RUNTIME_DEPENDENCIES, "brax")

_EXPORTS = {
    "SesameLocomotion": ("sesame_ml.playground.environment", "SesameLocomotion"),
    "default_config": ("sesame_ml.playground.environment", "default_config"),
    "domain_randomize": ("sesame_ml.playground.randomize", "domain_randomize"),
    "make_domain_randomizer": (
        "sesame_ml.playground.randomize",
        "make_domain_randomizer",
    ),
    "PPOConfig": ("sesame_ml.playground.train", "PPOConfig"),
    "TrainingResult": ("sesame_ml.playground.train", "TrainingResult"),
    "train": ("sesame_ml.playground.train", "train"),
}

__all__ = [*_EXPORTS, "is_available", "require_dependencies"]


def _missing_dependencies(*, training: bool) -> tuple[str, ...]:
    dependencies = _TRAINING_DEPENDENCIES if training else _RUNTIME_DEPENDENCIES
    return tuple(name for name in dependencies if util.find_spec(name) is None)


def is_available(*, training: bool = False) -> bool:
    """Return whether the optional environment or full training stack is installed."""

    return not _missing_dependencies(training=training)


def require_dependencies(*, training: bool = False) -> None:
    """Raise a focused error listing missing optional accelerator dependencies."""

    missing = _missing_dependencies(training=training)
    if missing:
        purpose = "PPO training" if training else "the Playground environment"
        raise ImportError(
            f"{purpose} requires optional packages: {', '.join(missing)}. "
            "Install sesame-ml[playground-cuda]"
            + (" for accelerator PPO training." if training else " or sesame-ml[playground].")
        )


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    require_dependencies(training=name in {"PPOConfig", "TrainingResult", "train"})
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
