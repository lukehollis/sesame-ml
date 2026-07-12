"""Physical Intelligence OpenPI WebSocket client adapter."""

from __future__ import annotations

import importlib
import socket
import time
from typing import Any

import numpy as np

from sesame_ml.constants import CONTROL_DT
from sesame_ml.integrations.base import (
    PolicyActionChunk,
    PolicyObservation,
    canonical_action_array,
)


def _wait_for_endpoint(host: str, port: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: OSError | None = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ConnectionError(
                f"OpenPI policy server is not reachable at {host}:{port} "
                f"after {timeout_s:.1f} seconds"
            ) from last_error
        try:
            with socket.create_connection((host, port), timeout=min(0.5, remaining)):
                return
        except OSError as error:
            last_error = error
            time.sleep(min(0.1, max(deadline - time.monotonic(), 0.0)))


def _new_openpi_client(
    host: str,
    port: int | None,
    api_key: str | None,
    connect_timeout_s: float,
) -> Any:
    try:
        module = importlib.import_module("openpi_client.websocket_client_policy")
    except ModuleNotFoundError as exc:
        if exc.name == "openpi_client" or (exc.name or "").startswith("openpi_client."):
            raise ModuleNotFoundError(
                "OpenPI remote inference requires Physical Intelligence's openpi-client. "
                "From an openpi checkout run `uv pip install -e packages/openpi-client`, "
                "then run `sesame_ml.bridge_cli` from the Sesame source tree on PYTHONPATH."
            ) from None
        raise
    _wait_for_endpoint(host, 8000 if port is None else port, connect_timeout_s)
    return module.WebsocketClientPolicy(host=host, port=port, api_key=api_key)


class OpenPIRemotePolicy:
    """Translate Sesame observations to the official OpenPI WebSocket protocol.

    The corresponding ``SesameInputs`` transform lives in
    ``integrations/openpi/sesame_policy.py``. Actions are required to already be
    denormalized absolute servo positions in radians; unsafe or wrong-dimensional
    outputs are rejected unless the vendor transform has explicitly projected them to 8D.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int | None = 8000,
        *,
        api_key: str | None = None,
        action_dt_s: float = CONTROL_DT,
        max_chunk_steps: int | None = None,
        connect_timeout_s: float = 5.0,
        client: Any | None = None,
    ) -> None:
        if not np.isfinite(action_dt_s) or action_dt_s <= 0:
            raise ValueError("action_dt_s must be finite and positive")
        if max_chunk_steps is not None and max_chunk_steps < 1:
            raise ValueError("max_chunk_steps must be positive")
        if not np.isfinite(connect_timeout_s) or connect_timeout_s <= 0:
            raise ValueError("connect_timeout_s must be finite and positive")
        self._owns_client = client is None
        self._client = (
            client
            if client is not None
            else _new_openpi_client(host, port, api_key, float(connect_timeout_s))
        )
        self.action_dt_s = float(action_dt_s)
        self.max_chunk_steps = max_chunk_steps

    @staticmethod
    def encode_observation(observation: PolicyObservation) -> dict[str, Any]:
        """Build the raw dictionary expected by the provided OpenPI transform."""

        if not isinstance(observation, PolicyObservation):
            raise TypeError("observation must be a PolicyObservation")
        return {
            "state": observation.state_rad.copy(),
            "images": {"front": observation.rgb.copy()},
            "prompt": observation.instruction,
        }

    def infer(self, observation: PolicyObservation) -> PolicyActionChunk:
        request = self.encode_observation(observation)
        started = time.perf_counter()
        response = self._client.infer(request)
        latency = time.perf_counter() - started
        if not isinstance(response, dict) or "actions" not in response:
            raise ValueError("OpenPI response must be a dictionary containing `actions`")
        actions = canonical_action_array(response["actions"], source="OpenPI")
        if self.max_chunk_steps is not None:
            actions = actions[: self.max_chunk_steps]
        info = {key: value for key, value in response.items() if key != "actions"}
        return PolicyActionChunk(
            actions_rad=actions,
            based_on_observation_seq=observation.observation_seq,
            dt_s=self.action_dt_s,
            policy_name="openpi",
            inference_latency_s=latency,
            info=info,
        )

    def reset(self) -> None:
        reset = getattr(self._client, "reset", None)
        if reset is not None:
            reset()

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()
            return
        # Pinned OpenPI 15a9616a exposes no public close method. Its private
        # synchronous WebSocket is the only owned resource; keep this guarded so
        # future clients with a public method take the branch above.
        websocket = getattr(self._client, "_ws", None) if self._owns_client else None
        close_websocket = getattr(websocket, "close", None)
        if callable(close_websocket):
            close_websocket()

    def __enter__(self) -> OpenPIRemotePolicy:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
