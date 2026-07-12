from __future__ import annotations

import asyncio
import time

from websockets.asyncio.client import connect

from sesame_ml.transport import (
    ControlState,
    ObservationV1,
    PolicyOutput,
    PolicyWebSocketServer,
    RobotWebSocketClient,
    decode_action_chunk,
    encode_observation,
)

ZERO_JOINTS = (0.0,) * 8


def test_websocket_runtime_streams_newest_observation_and_falls_back_on_disconnect() -> None:
    asyncio.run(_exercise_runtime())


async def _exercise_runtime() -> None:
    async def policy(observation: ObservationV1) -> PolicyOutput:
        # Deliberately slower than capture: the server must overwrite its
        # one-item waiting mailbox instead of building a latency queue.
        await asyncio.sleep(0.04)
        value = min(1.0, observation.sequence / 100.0)
        return PolicyOutput(
            targets=((value,) * 8, (value + 0.01,) * 8),
            control_period_s=0.02,
            valid_for_s=0.2,
            metadata={"source_sequence": observation.sequence},
        )

    server = PolicyWebSocketServer(policy, host="127.0.0.1", port=0)
    await server.start()

    active = asyncio.Event()
    disconnected_fallback = asyncio.Event()
    samples = []

    def source(sequence: int, monotonic_ns: int) -> ObservationV1:
        return ObservationV1(
            robot_id="sesame-integration-test",
            sequence=sequence,
            monotonic_ns=monotonic_ns,
            joint_position=ZERO_JOINTS,
            image_jpeg=b"\xff\xd8\xff\xd9",
        )

    def sink(sample) -> None:
        samples.append(sample)
        if sample.state is ControlState.ACTIVE:
            active.set()
        if sample.entered_fallback and sample.reason == "disconnected":
            disconnected_fallback.set()

    client = RobotWebSocketClient(
        uri=f"ws://127.0.0.1:{server.bound_port}",
        robot_id="sesame-integration-test",
        observation_source=source,
        action_sink=sink,
        observation_hz=100.0,
        control_hz=100.0,
        watchdog_timeout_s=0.15,
        reconnect_delay_s=0.05,
        run_sync_source_in_executor=False,
    )
    client_task = asyncio.create_task(client.run())
    try:
        await asyncio.wait_for(active.wait(), timeout=2.0)
        # Give the slower policy enough time to demonstrate mailbox overwrite.
        await asyncio.sleep(0.12)
        server_snapshot = server.metrics.snapshot()
        client_snapshot = client.metrics.snapshot()
        assert server_snapshot.observations_overwritten > 0
        assert server_snapshot.chunks_sent > 0
        assert client_snapshot.chunks_accepted > 0
        assert client_snapshot.mean_round_trip_ms is not None
        assert client_snapshot.mean_round_trip_ms >= 30.0
        assert any(
            sample.target is not None for sample in samples if sample.state is ControlState.ACTIVE
        )

        await server.close()
        await asyncio.wait_for(disconnected_fallback.wait(), timeout=1.0)
    finally:
        client.stop()
        await asyncio.wait_for(client_task, timeout=2.0)
        await server.close()
    assert samples[-1].state is ControlState.FALLBACK
    assert samples[-1].reason == "client_stopped"


def test_client_rejects_chunks_not_correlated_with_a_sent_observation() -> None:
    from sesame_ml.transport import ActionChunkV1, InstallResult

    def source(sequence: int, monotonic_ns: int) -> ObservationV1:
        return ObservationV1("sesame-test", sequence, monotonic_ns, ZERO_JOINTS)

    client = RobotWebSocketClient(
        uri="ws://127.0.0.1:9999",
        robot_id="sesame-test",
        observation_source=source,
        action_sink=lambda sample: None,
    )
    chunk = ActionChunkV1(
        robot_id="sesame-test",
        chunk_id=0,
        based_on_observation_sequence=99,
        created_monotonic_ns=time.monotonic_ns(),
        control_period_s=0.02,
        targets=(ZERO_JOINTS,),
        valid_for_s=0.2,
    )

    assert client.accept_action_chunk(chunk) is InstallResult.REJECTED_SEQUENCE
    assert client.metrics.snapshot().chunks_rejected_sequence == 1


def test_server_drops_duplicate_observations_and_accounts_for_sequence_gaps() -> None:
    asyncio.run(_exercise_observation_sequence_validation())


async def _exercise_observation_sequence_validation() -> None:
    async def policy(observation: ObservationV1) -> PolicyOutput:
        return PolicyOutput(
            targets=(ZERO_JOINTS,),
            control_period_s=0.02,
            valid_for_s=0.5,
        )

    server = PolicyWebSocketServer(policy, host="127.0.0.1", port=0)
    await server.start()
    try:
        async with connect(f"ws://127.0.0.1:{server.bound_port}") as websocket:
            capture_ns = time.monotonic_ns()
            observation = ObservationV1("sesame-sequences", 2, capture_ns, ZERO_JOINTS)
            await websocket.send(encode_observation(observation))
            assert decode_action_chunk(await websocket.recv()).based_on_observation_sequence == 2

            await websocket.send(encode_observation(observation))
            skipped = ObservationV1("sesame-sequences", 4, capture_ns + 1, ZERO_JOINTS)
            await websocket.send(encode_observation(skipped))
            assert decode_action_chunk(await websocket.recv()).based_on_observation_sequence == 4
    finally:
        await server.close()

    metrics = server.metrics.snapshot()
    assert metrics.observation_out_of_order == 1
    assert metrics.observation_sequence_gaps == 1
