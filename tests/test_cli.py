from __future__ import annotations

import io
import json

import numpy as np
import pytest
from gymnasium import spaces
from PIL import Image

from sesame_ml.cli import _ResidualPolicyServerBridge, build_parser, main
from sesame_ml.constants import STAND_ANGLES_RAD
from sesame_ml.policies import BasePolicy
from sesame_ml.transport import ObservationV1


class _PixelShapePolicy(BasePolicy):
    class Model:
        observation_space = spaces.Dict(
            {
                "proprio": spaces.Box(-np.inf, np.inf, (43,), dtype=np.float32),
                "rgb": spaces.Box(0, 255, (3, 120, 160), dtype=np.uint8),
            }
        )

    model = Model()

    def __init__(self) -> None:
        self.observations = []

    def predict(self, observation, *, deterministic=True):  # type: ignore[no-untyped-def]
        del deterministic
        self.observations.append(observation)
        return np.zeros(8, dtype=np.float32)


def test_parser_exposes_all_operational_stages() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    for command in (
        "validate-model",
        "train-ppo",
        "train-mjx",
        "plan",
        "serve",
        "sim-client",
        "orange-client",
        "collect",
        "export-dataset",
    ):
        assert command in help_text


def test_validate_model_command_runs_real_dynamics(capsys) -> None:
    assert main(["validate-model", "--seconds", "0.05"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ok"
    assert result["actuators"] == 8
    assert result["total_dynamic_mass_kg"] > 0.3


def test_policy_server_resizes_camera_and_uses_deployable_acceleration() -> None:
    image_bytes = io.BytesIO()
    Image.fromarray(np.zeros((240, 320, 3), dtype=np.uint8)).save(image_bytes, format="JPEG")
    policy = _PixelShapePolicy()
    bridge = _ResidualPolicyServerBridge(policy, pixels=True, chunk_steps=1)
    bridge(
        ObservationV1(
            robot_id="sesame-test",
            sequence=1,
            monotonic_ns=1,
            joint_position=tuple(float(value) for value in STAND_ANGLES_RAD),
            imu_quaternion=(1, 0, 0, 0),
            imu_gyro=(0, 0, 0),
            imu_acceleration=(9.81, 0, -9.81),
            image_jpeg=image_bytes.getvalue(),
        )
    )
    observation = policy.observations[-1]
    assert observation["rgb"].shape == (120, 160, 3)
    assert observation["proprio"][3:6] == pytest.approx([1, 0, -1])


def test_physical_client_requires_explicit_calibration() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["orange-client", "--uri", "ws://127.0.0.1:8765", "--robot-id", "robot"]
        )
