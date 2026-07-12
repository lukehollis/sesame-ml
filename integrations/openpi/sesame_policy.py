"""Copy into ``openpi/src/openpi/policies/`` for Sesame training and serving.

This file deliberately follows OpenPI's dataset-specific transform pattern. It targets
the official pi0, pi0-FAST, and pi0.5 model types and uses an explicit physical-8D to
checkpoint-32D padding projection for pi0/pi0.5.
Adapted from the Apache-2.0 OpenPI policy-transform examples at commit 15a9616a.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import numpy as np
from openpi import transforms
from openpi.models import model as _model

ACTION_DIM = 8


def _image(value: object) -> np.ndarray:
    image = np.asarray(value)
    if np.issubdtype(image.dtype, np.floating):
        if not np.all(np.isfinite(image)) or np.min(image) < 0 or np.max(image) > 1:
            raise ValueError("floating front images must be finite and normalized to [0, 1]")
        image = np.rint(image * 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"front image must be uint8 HWC RGB, got {image.dtype} {image.shape}")
    return image


def _joints(value: object, *, key: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (ACTION_DIM,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{key} must be a finite ({ACTION_DIM},) absolute-radian vector")
    return result


@dataclasses.dataclass(frozen=True)
class SesameInputs(transforms.DataTransformFn):
    """Map ``state/images.front/prompt`` into an OpenPI model observation."""

    model_type: _model.ModelType
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("front",)

    def __call__(self, data: dict) -> dict:
        state = _joints(data["state"], key="state")
        images = data["images"]
        if not isinstance(images, dict) or set(images) != {"front"}:
            raise ValueError("images must contain exactly the `front` camera")
        front = _image(images["front"])
        black = np.zeros_like(front)

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                model_images = {
                    "base_0_rgb": front,
                    "left_wrist_0_rgb": black,
                    "right_wrist_0_rgb": black,
                }
                masks = {
                    "base_0_rgb": np.True_,
                    "left_wrist_0_rgb": np.False_,
                    "right_wrist_0_rgb": np.False_,
                }
            case _model.ModelType.PI0_FAST:
                model_images = {
                    "base_0_rgb": front,
                    "base_1_rgb": black,
                    "wrist_0_rgb": black,
                }
                # FAST does not mask padding images in the official transforms.
                masks = {name: np.True_ for name in model_images}
            case _:
                raise ValueError(f"unsupported OpenPI model type: {self.model_type}")

        result = {"state": state, "image": model_images, "image_mask": masks}
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM:
                raise ValueError(f"actions must have shape (horizon, {ACTION_DIM})")
            result["actions"] = actions
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            result["prompt"] = prompt
        return result


@dataclasses.dataclass(frozen=True)
class SesameOutputs(transforms.DataTransformFn):
    """Project the verified vendor head back to eight physical servo targets."""

    model_action_dim: int

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if self.model_action_dim not in {ACTION_DIM, 32}:
            raise ValueError(f"unsupported OpenPI model action dimension: {self.model_action_dim}")
        if actions.ndim != 2 or actions.shape[-1] != self.model_action_dim:
            raise ValueError(
                "OpenPI head must return "
                f"(horizon, {self.model_action_dim}), got {actions.shape}"
            )
        # Physical Sesame joints occupy the first eight dimensions. For pi0/pi0.5,
        # upstream PadStatesAndActions(32) defines the matching zero-padded training
        # projection; slicing here is its explicit inverse after denormalization.
        return {"actions": actions[:, :ACTION_DIM]}
