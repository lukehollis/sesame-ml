# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""GR00T N1.7 NEW_EMBODIMENT config for Sesame's absolute 8D servo actions.

Adapted from NVIDIA Isaac-GR00T's custom-embodiment configuration examples for Sesame ML.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

sesame_config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["front"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=["joints"]),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["joints"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(sesame_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
