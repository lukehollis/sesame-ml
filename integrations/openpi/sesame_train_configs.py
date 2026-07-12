"""OpenPI config factories for the canonical Sesame LeRobot dataset.

Copy ``sesame_policy.py`` into ``openpi/policies`` and this file into your OpenPI
checkout, then add the returned configs to ``training.config._CONFIGS``.
The factory follows OpenPI's Apache-2.0 training configuration patterns at commit 15a9616a.
"""

from __future__ import annotations

import dataclasses
import pathlib

from openpi import transforms as _transforms
from openpi.models import pi0_config, pi0_fast
from openpi.policies import sesame_policy
from openpi.training import config as _config
from openpi.training import weight_loaders
from typing_extensions import override


@dataclasses.dataclass(frozen=True)
class SesameDataConfig(_config.DataConfigFactory):
    """Repack ``observation.state/action/front`` without delta-action conversion."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config):
        base = self.create_base_config(assets_dirs, model_config)
        repack = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"front": "observation.images.front"},
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data = _transforms.Group(
            inputs=[sesame_policy.SesameInputs(model_type=model_config.model_type)],
            outputs=[
                sesame_policy.SesameOutputs(model_action_dim=model_config.action_dim)
            ],
        )
        return dataclasses.replace(
            base,
            repack_transforms=repack,
            data_transforms=data,
            model_transforms=_config.ModelTransformFactory()(model_config),
            action_sequence_keys=("action",),
            prompt_from_task=True,
        )


def make_sesame_configs(repo_id: str, *, horizon: int = 16) -> tuple[_config.TrainConfig, ...]:
    """Create LoRA fine-tuning configs for pi0, pi0-FAST, and pi0.5."""

    pi0_model = pi0_config.Pi0Config(
        # pi0_base uses 32D state/action projections. Upstream's model transform
        # explicitly zero-pads Sesame's physical 8D arrays to this checkpoint shape.
        action_dim=32,
        action_horizon=horizon,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    fast_model = pi0_fast.Pi0FASTConfig(
        action_dim=8,
        action_horizon=horizon,
        max_token_len=180,
        paligemma_variant="gemma_2b_lora",
    )
    pi05_model = pi0_config.Pi0Config(
        pi05=True,
        # pi05_base is also trained with a 32D head.
        action_dim=32,
        action_horizon=horizon,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    specs = (
        ("pi0_sesame_lora", pi0_model, "pi0_base"),
        ("pi0_fast_sesame_lora", fast_model, "pi0_fast_base"),
        ("pi05_sesame_lora", pi05_model, "pi05_base"),
    )
    return tuple(
        _config.TrainConfig(
            name=name,
            model=model,
            data=SesameDataConfig(
                repo_id=repo_id,
                base_config=_config.DataConfig(prompt_from_task=True),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                f"gs://openpi-assets/checkpoints/{checkpoint}/params"
            ),
            freeze_filter=model.get_freeze_filter(),
            ema_decay=None,
            num_train_steps=30_000,
        )
        for name, model, checkpoint in specs
    )
