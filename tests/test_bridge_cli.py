from __future__ import annotations

import ast
import enum
import runpy
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def test_bridge_cli_imports_without_simulator_stack() -> None:
    code = (
        "import sys; import sesame_ml.bridge_cli; "
        "assert 'mujoco' not in sys.modules; assert 'gymnasium' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_openpi_configs_preserve_base_checkpoint_dimensions() -> None:
    source = Path(__file__).parents[1] / "integrations/openpi/sesame_train_configs.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    dimensions: list[tuple[bool, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"Pi0Config", "Pi0FASTConfig"}:
            continue
        values = {item.arg: item.value for item in node.keywords if item.arg is not None}
        action_dim = values.get("action_dim")
        assert isinstance(action_dim, ast.Constant) and isinstance(action_dim.value, int)
        pi05 = values.get("pi05")
        dimensions.append((isinstance(pi05, ast.Constant) and pi05.value is True, action_dim.value))
    assert dimensions == [(False, 32), (False, 8), (True, 32)]


def test_openpi_output_projection_is_explicit_and_shape_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ModelType(enum.Enum):
        PI0 = "pi0"
        PI05 = "pi05"
        PI0_FAST = "pi0_fast"

    transforms = types.ModuleType("openpi.transforms")
    transforms.DataTransformFn = object
    model = types.ModuleType("openpi.models.model")
    model.ModelType = ModelType
    models = types.ModuleType("openpi.models")
    models.model = model
    openpi = types.ModuleType("openpi")
    openpi.transforms = transforms
    openpi.models = models
    monkeypatch.setitem(sys.modules, "openpi", openpi)
    monkeypatch.setitem(sys.modules, "openpi.transforms", transforms)
    monkeypatch.setitem(sys.modules, "openpi.models", models)
    monkeypatch.setitem(sys.modules, "openpi.models.model", model)

    source = Path(__file__).parents[1] / "integrations/openpi/sesame_policy.py"
    module = runpy.run_path(str(source))
    outputs = module["SesameOutputs"](model_action_dim=32)
    vendor_actions = np.arange(96, dtype=np.float32).reshape(3, 32)
    projected = outputs({"actions": vendor_actions})["actions"]
    assert np.array_equal(projected, vendor_actions[:, :8])
    with pytest.raises(ValueError, match="must return"):
        outputs({"actions": np.zeros((3, 8), dtype=np.float32)})
