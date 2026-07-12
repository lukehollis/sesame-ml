"""NVIDIA GR00T N1.7 ZeroMQ policy-server adapter."""

from __future__ import annotations

import importlib
import time
from typing import Any

import numpy as np

from sesame_ml.constants import CONTROL_DT
from sesame_ml.integrations.base import PolicyActionChunk, PolicyObservation

LANGUAGE_KEY = "annotation.human.task_description"


def _new_groot_client(
    host: str, port: int, timeout_ms: int, api_token: str | None
) -> Any:
    try:
        module = importlib.import_module("gr00t.policy.server_client")
    except ModuleNotFoundError as exc:
        if exc.name == "gr00t" or (exc.name or "").startswith("gr00t."):
            raise ModuleNotFoundError(
                "GR00T remote inference requires NVIDIA Isaac-GR00T N1.7. Run the "
                "MuJoCo-free `sesame_ml.bridge_cli` from the source tree on PYTHONPATH "
                "inside the Isaac-GR00T environment, then run its PolicyServer; "
                "the Orange Pi/client machine does not need the model weights."
            ) from None
        raise
    return module.PolicyClient(
        host=host,
        port=port,
        timeout_ms=timeout_ms,
        api_token=api_token,
        strict=False,
    )


class GrootRemotePolicy:
    """Translate the canonical Sesame contract to GR00T's N1.7 policy API."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        *,
        timeout_ms: int = 15_000,
        api_token: str | None = None,
        action_dt_s: float = CONTROL_DT,
        max_chunk_steps: int | None = None,
        verify_connection: bool = True,
        client: Any | None = None,
    ) -> None:
        if timeout_ms < 1:
            raise ValueError("timeout_ms must be positive")
        if not np.isfinite(action_dt_s) or action_dt_s <= 0:
            raise ValueError("action_dt_s must be finite and positive")
        if max_chunk_steps is not None and max_chunk_steps < 1:
            raise ValueError("max_chunk_steps must be positive")
        self._client = (
            client
            if client is not None
            else _new_groot_client(host, port, timeout_ms, api_token)
        )
        self.action_dt_s = float(action_dt_s)
        self.max_chunk_steps = max_chunk_steps
        if verify_connection:
            ping = getattr(self._client, "ping", None)
            if ping is None or not ping():
                self.close()
                raise ConnectionError(f"GR00T policy server is not reachable at {host}:{port}")

    @staticmethod
    def encode_observation(observation: PolicyObservation) -> dict[str, Any]:
        """Create documented GR00T [batch=1, time=1, ...] modalities."""

        if not isinstance(observation, PolicyObservation):
            raise TypeError("observation must be a PolicyObservation")
        return {
            "video": {"front": observation.rgb[None, None, ...].copy()},
            "state": {"joints": observation.state_rad[None, None, ...].copy()},
            "language": {LANGUAGE_KEY: [[observation.instruction]]},
        }

    @staticmethod
    def _decode_actions(response: Any) -> tuple[np.ndarray, dict[str, Any]]:
        if not isinstance(response, tuple) or len(response) != 2:
            raise ValueError("GR00T get_action must return (action_dict, info_dict)")
        action, info = response
        if not isinstance(action, dict) or "joints" not in action:
            raise ValueError("GR00T action dictionary must contain the `joints` modality")
        actions = np.asarray(action["joints"])
        valid_shape = (
            actions.ndim == 3
            and actions.shape[0] == 1
            and actions.shape[1] >= 1
            and actions.shape[2] == 8
        )
        if not valid_shape:
            raise ValueError(
                "GR00T `joints` must have shape (batch=1, horizon, 8); got "
                f"{actions.shape}. Use the Sesame NEW_EMBODIMENT modality config with "
                "absolute NON_EEF actions."
            )
        if info is None:
            info = {}
        if not isinstance(info, dict):
            raise ValueError("GR00T policy info must be a dictionary")
        return actions[0], info

    def infer(self, observation: PolicyObservation) -> PolicyActionChunk:
        request = self.encode_observation(observation)
        started = time.perf_counter()
        response = self._client.get_action(request)
        latency = time.perf_counter() - started
        actions, info = self._decode_actions(response)
        if self.max_chunk_steps is not None:
            actions = actions[: self.max_chunk_steps]
        return PolicyActionChunk(
            actions_rad=actions,
            based_on_observation_seq=observation.observation_seq,
            dt_s=self.action_dt_s,
            policy_name="groot-n1.7",
            inference_latency_s=latency,
            info=info,
        )

    def reset(self) -> None:
        reset = getattr(self._client, "reset", None)
        if reset is not None:
            reset(options=None)

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> GrootRemotePolicy:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
