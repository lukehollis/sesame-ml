from __future__ import annotations

from pathlib import Path

from sesame_ml.vendor_templates import template_directory


def test_installed_vendor_templates_match_source_repository_copies() -> None:
    repository_root = Path(__file__).parents[1]
    packaged = template_directory()
    for relative in (
        Path("openpi/sesame_policy.py"),
        Path("openpi/sesame_train_configs.py"),
        Path("groot/sesame_config.py"),
        Path("groot/modality.json"),
        Path("bridge-requirements.txt"),
    ):
        assert (packaged / relative).read_bytes() == (
            repository_root / "integrations" / relative
        ).read_bytes()
