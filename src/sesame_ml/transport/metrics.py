"""Low-overhead counters and latency summaries for the policy transport."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any


@dataclass(frozen=True, slots=True)
class TransportMetricsSnapshot:
    connections: int
    reconnects: int
    protocol_errors: int
    observations_sent: int
    observations_received: int
    observations_overwritten: int
    observation_sequence_gaps: int
    observation_out_of_order: int
    chunks_generated: int
    chunks_skipped_stale: int
    chunks_sent: int
    chunks_received: int
    chunks_accepted: int
    chunks_replaced: int
    chunks_rejected_stale: int
    chunks_rejected_sequence: int
    watchdog_entries: int
    control_sink_errors: int
    policy_errors: int
    last_round_trip_ms: float | None
    mean_round_trip_ms: float | None
    max_round_trip_ms: float | None
    last_policy_ms: float | None
    mean_policy_ms: float | None
    max_policy_ms: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class TransportMetrics:
    """Thread-safe transport instrumentation.

    Counters are intentionally dependency-free so applications can periodically
    export ``snapshot().as_dict()`` to Prometheus, logs, or an experiment tracker.
    """

    _COUNTERS = (
        "connections",
        "reconnects",
        "protocol_errors",
        "observations_sent",
        "observations_received",
        "observations_overwritten",
        "observation_sequence_gaps",
        "observation_out_of_order",
        "chunks_generated",
        "chunks_skipped_stale",
        "chunks_sent",
        "chunks_received",
        "chunks_accepted",
        "chunks_replaced",
        "chunks_rejected_stale",
        "chunks_rejected_sequence",
        "watchdog_entries",
        "control_sink_errors",
        "policy_errors",
    )

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters = {name: 0 for name in self._COUNTERS}
        self._round_trip_count = 0
        self._round_trip_sum_ms = 0.0
        self._round_trip_last_ms: float | None = None
        self._round_trip_max_ms: float | None = None
        self._policy_count = 0
        self._policy_sum_ms = 0.0
        self._policy_last_ms: float | None = None
        self._policy_max_ms: float | None = None

    def increment(self, name: str, amount: int = 1) -> None:
        if name not in self._counters:
            raise KeyError(f"unknown transport metric {name!r}")
        if not isinstance(amount, int) or amount < 0:
            raise ValueError("metric increments must be non-negative integers")
        with self._lock:
            self._counters[name] += amount

    def observe_round_trip(self, milliseconds: float) -> None:
        self._observe_latency("round_trip", milliseconds)

    def observe_policy(self, milliseconds: float) -> None:
        self._observe_latency("policy", milliseconds)

    def _observe_latency(self, kind: str, milliseconds: float) -> None:
        if not math.isfinite(milliseconds) or milliseconds < 0.0:
            raise ValueError("latency must be a finite, non-negative value")
        with self._lock:
            if kind == "round_trip":
                self._round_trip_count += 1
                self._round_trip_sum_ms += milliseconds
                self._round_trip_last_ms = milliseconds
                self._round_trip_max_ms = max(self._round_trip_max_ms or 0.0, milliseconds)
            elif kind == "policy":
                self._policy_count += 1
                self._policy_sum_ms += milliseconds
                self._policy_last_ms = milliseconds
                self._policy_max_ms = max(self._policy_max_ms or 0.0, milliseconds)
            else:  # pragma: no cover - all callers use a fixed kind
                raise KeyError(kind)

    def snapshot(self) -> TransportMetricsSnapshot:
        with self._lock:
            counters = dict(self._counters)
            return TransportMetricsSnapshot(
                **counters,
                last_round_trip_ms=self._round_trip_last_ms,
                mean_round_trip_ms=(
                    self._round_trip_sum_ms / self._round_trip_count
                    if self._round_trip_count
                    else None
                ),
                max_round_trip_ms=self._round_trip_max_ms,
                last_policy_ms=self._policy_last_ms,
                mean_policy_ms=self._policy_sum_ms / self._policy_count
                if self._policy_count
                else None,
                max_policy_ms=self._policy_max_ms,
            )
