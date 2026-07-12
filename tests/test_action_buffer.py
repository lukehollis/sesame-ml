from __future__ import annotations

import pytest

from sesame_ml.transport import (
    ActionChunkV1,
    ActionInterpolationBuffer,
    ControlState,
    InstallResult,
)


def _target(value: float) -> tuple[float, float, float, float, float, float, float, float]:
    return (value,) * 8


def _chunk(
    chunk_id: int,
    observation_sequence: int,
    *,
    values: tuple[float, ...] = (0.0, 1.0),
    valid_for_s: float = 0.5,
) -> ActionChunkV1:
    return ActionChunkV1(
        robot_id="sesame-001",
        chunk_id=chunk_id,
        based_on_observation_sequence=observation_sequence,
        created_monotonic_ns=1,
        control_period_s=0.05,
        targets=tuple(_target(value) for value in values),
        valid_for_s=valid_for_s,
    )


def test_interpolates_actions_and_enters_configured_fallback_at_watchdog() -> None:
    fallback = _target(0.25)
    buffer = ActionInterpolationBuffer(watchdog_timeout_s=0.2, safe_fallback=fallback)
    received_ns = 1_000_000_000
    assert (
        buffer.install(
            _chunk(0, 10),
            received_at_ns=received_ns,
            originating_observation_ns=950_000_000,
        )
        is InstallResult.ACCEPTED
    )

    midpoint = buffer.sample(now_ns=received_ns + 25_000_000)
    assert midpoint.state is ControlState.ACTIVE
    assert midpoint.interpolation_fraction == pytest.approx(0.5)
    assert midpoint.target == pytest.approx(_target(0.5))

    expired = buffer.sample(now_ns=received_ns + 200_000_000)
    assert expired.state is ControlState.FALLBACK
    assert expired.target == fallback
    assert expired.reason == "watchdog_expired"
    assert expired.entered_fallback

    still_expired = buffer.sample(now_ns=received_ns + 210_000_000)
    assert not still_expired.entered_fallback


def test_new_chunk_replaces_current_and_reordered_chunks_are_rejected() -> None:
    buffer = ActionInterpolationBuffer(watchdog_timeout_s=1.0)
    assert buffer.install(_chunk(3, 20), received_at_ns=1_000) is InstallResult.ACCEPTED
    assert buffer.install(_chunk(4, 22), received_at_ns=2_000) is InstallResult.REPLACED
    assert buffer.install(_chunk(3, 23), received_at_ns=3_000) is InstallResult.REJECTED_SEQUENCE
    assert buffer.install(_chunk(5, 21), received_at_ns=3_000) is InstallResult.REJECTED_SEQUENCE

    sample = buffer.sample(now_ns=2_000)
    assert sample.chunk_id == 4
    assert sample.based_on_observation_sequence == 22


def test_network_and_inference_age_are_included_in_freshness() -> None:
    buffer = ActionInterpolationBuffer(watchdog_timeout_s=1.0)
    result = buffer.install(
        _chunk(0, 1, valid_for_s=0.1),
        received_at_ns=500_000_000,
        originating_observation_ns=350_000_000,
    )
    assert result is InstallResult.REJECTED_STALE
    assert buffer.sample(now_ns=500_000_000).state is ControlState.FALLBACK


def test_new_connection_epoch_allows_server_chunk_counter_to_restart() -> None:
    buffer = ActionInterpolationBuffer(watchdog_timeout_s=1.0)
    assert buffer.install(_chunk(7, 100), received_at_ns=1_000) is InstallResult.ACCEPTED
    buffer.reset_sequence_epoch()
    assert buffer.install(_chunk(0, 101), received_at_ns=2_000) is InstallResult.ACCEPTED
