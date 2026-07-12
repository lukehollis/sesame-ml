from __future__ import annotations

import importlib
import io

import numpy as np
import pytest
from PIL import Image

from sesame_ml.constants import STAND_ANGLES_RAD
from sesame_ml.integrations import (
    GrootRemotePolicy,
    OpenPIRemotePolicy,
    PolicyActionChunk,
    PolicyObservation,
    RemotePolicyBridge,
)
from sesame_ml.integrations.groot import LANGUAGE_KEY
from sesame_ml.transport import ObservationV1


def _observation() -> PolicyObservation:
    return PolicyObservation(
        state_rad=STAND_ANGLES_RAD,
        rgb=np.full((24, 32, 3), 17, dtype=np.uint8),
        instruction="recover, then walk toward the green marker",
        timestamp_ns=100,
        observation_seq=9,
    )


class _OpenPIClient:
    def __init__(self, actions: np.ndarray) -> None:
        self.actions = actions
        self.request = None
        self.reset_count = 0

    def infer(self, request):
        self.request = request
        return {"actions": self.actions, "model": "pi0.5"}

    def reset(self):
        self.reset_count += 1


def test_openpi_adapter_maps_observation_and_strict_action_chunk() -> None:
    client = _OpenPIClient(np.tile(STAND_ANGLES_RAD, (4, 1)))
    policy = OpenPIRemotePolicy(client=client, max_chunk_steps=3)
    chunk = policy.infer(_observation())

    assert client.request["state"].shape == (8,)
    assert np.array_equal(client.request["images"]["front"], _observation().rgb)
    assert "green marker" in client.request["prompt"]
    assert chunk.actions_rad.shape == (3, 8)
    assert chunk.based_on_observation_seq == 9
    assert chunk.policy_name == "openpi"
    assert chunk.info == {"model": "pi0.5"}
    policy.reset()
    assert client.reset_count == 1


def test_openpi_adapter_rejects_wrong_action_dimension() -> None:
    policy = OpenPIRemotePolicy(client=_OpenPIClient(np.zeros((4, 32))))
    with pytest.raises(ValueError, match="physical output projection"):
        policy.infer(_observation())


def test_openpi_adapter_bounds_startup_and_closes_pinned_private_socket() -> None:
    with pytest.raises(ValueError, match="connect_timeout_s"):
        OpenPIRemotePolicy(client=_OpenPIClient(np.zeros((1, 8))), connect_timeout_s=0)

    class WebSocket:
        closed = False

        def close(self) -> None:
            self.closed = True

    client = _OpenPIClient(np.zeros((1, 8)))
    client._ws = WebSocket()
    policy = OpenPIRemotePolicy(client=client)
    policy._owns_client = True
    policy.close()
    assert client._ws.closed


class _GrootClient:
    def __init__(self, actions: np.ndarray) -> None:
        self.actions = actions
        self.request = None
        self.closed = False
        self.reset_options = "unset"

    def ping(self) -> bool:
        return True

    def get_action(self, request):
        self.request = request
        return {"joints": self.actions}, {"checkpoint": "N1.7"}

    def reset(self, options=None):
        self.reset_options = options

    def close(self):
        self.closed = True


def test_groot_adapter_maps_documented_batch_time_modalities() -> None:
    client = _GrootClient(np.tile(STAND_ANGLES_RAD, (1, 5, 1)))
    policy = GrootRemotePolicy(client=client, max_chunk_steps=2)
    chunk = policy.infer(_observation())

    assert client.request["video"]["front"].shape == (1, 1, 24, 32, 3)
    assert client.request["state"]["joints"].shape == (1, 1, 8)
    assert client.request["language"][LANGUAGE_KEY] == [
        ["recover, then walk toward the green marker"]
    ]
    assert chunk.actions_rad.shape == (2, 8)
    assert chunk.info == {"checkpoint": "N1.7"}
    policy.reset()
    policy.close()
    assert client.reset_options is None
    assert client.closed


def test_groot_adapter_rejects_missing_batch_axis_and_out_of_range_targets() -> None:
    policy = GrootRemotePolicy(
        client=_GrootClient(np.tile(STAND_ANGLES_RAD, (3, 1))), verify_connection=False
    )
    with pytest.raises(ValueError, match="batch=1"):
        policy.infer(_observation())

    unsafe = np.tile(STAND_ANGLES_RAD, (1, 2, 1))
    unsafe[..., 0] = -1
    policy = GrootRemotePolicy(client=_GrootClient(unsafe), verify_connection=False)
    with pytest.raises(ValueError, match="global physical"):
        policy.infer(_observation())


def test_common_action_chunk_rejects_nonfinite_values() -> None:
    actions = np.tile(STAND_ANGLES_RAD, (2, 1))
    actions[0, 3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        PolicyActionChunk(actions_rad=actions, based_on_observation_seq=0)


def test_optional_dependency_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = importlib.import_module

    def missing(name: str, package: str | None = None):
        if name == "openpi_client.websocket_client_policy":
            error = ModuleNotFoundError("missing")
            error.name = "openpi_client"
            raise error
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(ModuleNotFoundError, match="packages/openpi-client"):
        OpenPIRemotePolicy()


def test_remote_policy_bridge_plugs_into_wifi_server_callback() -> None:
    client = _OpenPIClient(np.tile(STAND_ANGLES_RAD, (4, 1)))
    backend = OpenPIRemotePolicy(client=client)
    bridge = RemotePolicyBridge(backend, valid_for_s=0.8)
    encoded = io.BytesIO()
    Image.fromarray(np.full((20, 30, 3), 42, dtype=np.uint8)).save(encoded, format="JPEG")
    wire_observation = ObservationV1(
        robot_id="sesame-1",
        sequence=11,
        monotonic_ns=1234,
        joint_position=STAND_ANGLES_RAD,
        language_instruction="walk forward",
        image_jpeg=encoded.getvalue(),
    )

    output = bridge(wire_observation)

    assert len(output.targets) == 4
    assert output.control_period_s == pytest.approx(0.02)
    assert output.valid_for_s == 0.8
    assert output.metadata["policy"] == "openpi"
    assert output.metadata["action_horizon"] == 4
    assert client.request["images"]["front"].shape == (20, 30, 3)


def test_remote_policy_bridge_fails_closed_without_valid_camera_or_instruction() -> None:
    bridge = RemotePolicyBridge(OpenPIRemotePolicy(client=_OpenPIClient(STAND_ANGLES_RAD)))
    observation = ObservationV1(
        robot_id="sesame-1",
        sequence=0,
        monotonic_ns=1,
        joint_position=STAND_ANGLES_RAD,
        image_jpeg=b"not-an-image",
    )
    with pytest.raises(ValueError, match="language instruction"):
        bridge(observation)

    bridge = RemotePolicyBridge(
        OpenPIRemotePolicy(client=_OpenPIClient(STAND_ANGLES_RAD)),
        default_instruction="stand",
    )
    with pytest.raises(ValueError, match="valid image"):
        bridge(observation)
