"""Newest-only action buffering, interpolation, and watchdog behavior."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real
from threading import Lock

from .protocol import ActionChunkV1, JointVector


class InstallResult(StrEnum):
    ACCEPTED = "accepted"
    REPLACED = "replaced"
    REJECTED_STALE = "rejected_stale"
    REJECTED_SEQUENCE = "rejected_sequence"


class ControlState(StrEnum):
    ACTIVE = "active"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class ActionSample:
    """One control-loop output for a hardware or simulator adapter.

    A ``None`` target explicitly means that the adapter must execute its own
    safe fallback (normally disable output or enter a separately validated rest
    routine).  The transport does not guess safe servo angles.
    """

    target: JointVector | None
    state: ControlState
    reason: str
    chunk_id: int | None = None
    based_on_observation_sequence: int | None = None
    interpolation_fraction: float = 0.0
    entered_fallback: bool = False


class ActionInterpolationBuffer:
    """Thread-safe receding-horizon action buffer.

    New chunks atomically replace old chunks.  Chunk IDs and echoed observation
    sequences must both increase within a connection epoch, preventing delayed
    WebSocket frames from rolling the robot back to an older plan.
    """

    def __init__(
        self,
        *,
        watchdog_timeout_s: float = 0.25,
        safe_fallback: JointVector | None = None,
    ) -> None:
        if not math.isfinite(watchdog_timeout_s) or watchdog_timeout_s <= 0.0:
            raise ValueError("watchdog_timeout_s must be finite and positive")
        if safe_fallback is not None:
            if len(safe_fallback) != 8 or any(
                isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
                for value in safe_fallback
            ):
                raise ValueError("safe_fallback must contain eight finite joint angles")
            safe_fallback = tuple(float(value) for value in safe_fallback)  # type: ignore[assignment]
        self._watchdog_timeout_ns = round(watchdog_timeout_s * 1e9)
        self._safe_fallback = safe_fallback
        self._lock = Lock()
        self._chunk: ActionChunkV1 | None = None
        self._received_at_ns = 0
        self._deadline_ns = 0
        self._latest_chunk_id: int | None = None
        self._latest_observation_sequence: int | None = None
        self._fallback_reason = "no_action"
        self._last_sample_was_active = False

    @property
    def watchdog_timeout_s(self) -> float:
        return self._watchdog_timeout_ns / 1e9

    def install(
        self,
        chunk: ActionChunkV1,
        *,
        received_at_ns: int | None = None,
        originating_observation_ns: int | None = None,
    ) -> InstallResult:
        """Install a chunk if it is newer and still fresh.

        When ``originating_observation_ns`` is supplied, validity begins at that
        robot-local capture timestamp, so the deadline includes capture, network,
        and policy latency. If omitted, validity begins at receipt (useful only
        for local policies and direct buffer use).
        """

        now_ns = time.monotonic_ns() if received_at_ns is None else received_at_ns
        if not isinstance(now_ns, int) or now_ns < 0:
            raise ValueError("received_at_ns must be a non-negative integer")
        if originating_observation_ns is not None and (
            not isinstance(originating_observation_ns, int) or originating_observation_ns < 0
        ):
            raise ValueError("originating_observation_ns must be a non-negative integer")

        validity_origin_ns = (
            now_ns if originating_observation_ns is None else originating_observation_ns
        )
        validity_deadline_ns = validity_origin_ns + round(chunk.valid_for_s * 1e9)
        watchdog_deadline_ns = now_ns + self._watchdog_timeout_ns
        deadline_ns = min(validity_deadline_ns, watchdog_deadline_ns)

        with self._lock:
            latest_observation_sequence = self._latest_observation_sequence
            if self._latest_chunk_id is not None and (
                chunk.chunk_id <= self._latest_chunk_id
                or (
                    latest_observation_sequence is not None
                    and chunk.based_on_observation_sequence <= latest_observation_sequence
                )
            ):
                return InstallResult.REJECTED_SEQUENCE
            if deadline_ns <= now_ns:
                return InstallResult.REJECTED_STALE

            replaced = self._chunk is not None and self._deadline_ns > now_ns
            self._chunk = chunk
            self._received_at_ns = now_ns
            self._deadline_ns = deadline_ns
            self._latest_chunk_id = chunk.chunk_id
            self._latest_observation_sequence = chunk.based_on_observation_sequence
            self._fallback_reason = ""
            return InstallResult.REPLACED if replaced else InstallResult.ACCEPTED

    def sample(self, *, now_ns: int | None = None) -> ActionSample:
        """Return a linearly interpolated target or the configured fallback."""

        sample_ns = time.monotonic_ns() if now_ns is None else now_ns
        if not isinstance(sample_ns, int) or sample_ns < 0:
            raise ValueError("now_ns must be a non-negative integer")
        with self._lock:
            chunk = self._chunk
            if chunk is None or sample_ns >= self._deadline_ns:
                if chunk is not None:
                    self._chunk = None
                    self._fallback_reason = "watchdog_expired"
                entered = self._last_sample_was_active
                self._last_sample_was_active = False
                return ActionSample(
                    target=self._safe_fallback,
                    state=ControlState.FALLBACK,
                    reason=self._fallback_reason,
                    entered_fallback=entered,
                )

            elapsed_s = max(0.0, (sample_ns - self._received_at_ns) / 1e9)
            phase = elapsed_s / chunk.control_period_s
            lower_index = min(int(math.floor(phase)), len(chunk.targets) - 1)
            upper_index = min(lower_index + 1, len(chunk.targets) - 1)
            fraction = 0.0 if lower_index == upper_index else phase - lower_index
            lower = chunk.targets[lower_index]
            upper = chunk.targets[upper_index]
            target = tuple(a + (b - a) * fraction for a, b in zip(lower, upper, strict=True))
            self._last_sample_was_active = True
            return ActionSample(
                target=target,  # type: ignore[arg-type]
                state=ControlState.ACTIVE,
                reason="action_chunk",
                chunk_id=chunk.chunk_id,
                based_on_observation_sequence=chunk.based_on_observation_sequence,
                interpolation_fraction=fraction,
            )

    def invalidate(self, reason: str = "disconnected") -> None:
        """Immediately force fallback while preserving anti-replay sequence state."""

        if not reason:
            raise ValueError("fallback reason may not be empty")
        with self._lock:
            self._chunk = None
            self._fallback_reason = reason

    def reset_sequence_epoch(self, reason: str = "new_connection") -> None:
        """Start a new server sequencing epoch after a fresh connection."""

        self.invalidate(reason)
        with self._lock:
            self._latest_chunk_id = None
            self._latest_observation_sequence = None

    @property
    def latest_chunk_id(self) -> int | None:
        with self._lock:
            return self._latest_chunk_id
