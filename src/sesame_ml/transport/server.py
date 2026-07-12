"""WebSocket policy host with latest-observation scheduling."""

from __future__ import annotations

import asyncio
import inspect
import logging
import ssl
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from .metrics import TransportMetrics
from .protocol import (
    MAX_PACKET_BYTES,
    ActionChunkV1,
    ObservationV1,
    PolicyOutput,
    ProtocolError,
    decode_observation,
    encode_action_chunk,
)

LOGGER = logging.getLogger(__name__)
PolicyCallback = Callable[[ObservationV1], PolicyOutput | Awaitable[PolicyOutput]]


@dataclass(slots=True)
class _ConnectionState:
    robot_id: str | None = None
    last_observation_sequence: int | None = None
    newest_observation_sequence: int | None = None
    next_chunk_id: int = 0


class PolicyWebSocketServer:
    """Runs a remote policy behind a binary WebSocket endpoint.

    Observation ingestion and policy inference are separate tasks.  A one-item
    mailbox is overwritten when a camera stream outruns inference, ensuring the
    policy always works toward the newest observation rather than accumulating
    a dangerous stale queue.
    """

    def __init__(
        self,
        policy: PolicyCallback,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        allowed_robot_ids: set[str] | frozenset[str] | None = None,
        ssl_context: ssl.SSLContext | None = None,
        ping_interval_s: float = 10.0,
        ping_timeout_s: float = 10.0,
        run_sync_policy_in_executor: bool = True,
        metrics: TransportMetrics | None = None,
    ) -> None:
        if not callable(policy):
            raise TypeError("policy must be callable")
        if not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        self.policy = policy
        self.host = host
        self.port = port
        self.allowed_robot_ids = (
            frozenset(allowed_robot_ids) if allowed_robot_ids is not None else None
        )
        self.ssl_context = ssl_context
        self.ping_interval_s = ping_interval_s
        self.ping_timeout_s = ping_timeout_s
        self.run_sync_policy_in_executor = run_sync_policy_in_executor
        self.metrics = metrics or TransportMetrics()
        self._server: Server | None = None

    async def start(self) -> None:
        """Bind the endpoint and begin accepting clients."""

        if self._server is not None:
            raise RuntimeError("policy server is already running")
        self._server = await serve(
            self._handle_connection,
            self.host,
            self.port,
            ssl=self.ssl_context,
            ping_interval=self.ping_interval_s,
            ping_timeout=self.ping_timeout_s,
            max_size=MAX_PACKET_BYTES,
            compression=None,
        )

    async def close(self) -> None:
        """Stop accepting clients and wait for connection handlers to finish."""

        server, self._server = self._server, None
        if server is not None:
            server.close()
            await server.wait_closed()

    async def serve_forever(self) -> None:
        await self.start()
        assert self._server is not None
        try:
            await self._server.serve_forever()
        finally:
            await self.close()

    @property
    def bound_port(self) -> int:
        if self._server is None or not self._server.sockets:
            raise RuntimeError("policy server has not been started")
        return int(self._server.sockets[0].getsockname()[1])

    async def __aenter__(self) -> PolicyWebSocketServer:
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def _handle_connection(self, websocket: ServerConnection) -> None:
        self.metrics.increment("connections")
        state = _ConnectionState()
        mailbox: asyncio.Queue[ObservationV1] = asyncio.Queue(maxsize=1)
        receiver = asyncio.create_task(
            self._receive_observations(websocket, state, mailbox),
            name="sesame-policy-observation-receiver",
        )
        worker = asyncio.create_task(
            self._run_policy(websocket, state, mailbox),
            name="sesame-policy-worker",
        )
        done, pending = await asyncio.wait({receiver, worker}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            try:
                task.result()
            except ConnectionClosed:
                pass
            except asyncio.CancelledError:
                pass
            except Exception:
                LOGGER.exception("Sesame policy connection failed")
                if not websocket.close_code:
                    await websocket.close(code=1011, reason="policy server failure")

    async def _receive_observations(
        self,
        websocket: ServerConnection,
        state: _ConnectionState,
        mailbox: asyncio.Queue[ObservationV1],
    ) -> None:
        async for packet in websocket:
            if not isinstance(packet, bytes):
                self.metrics.increment("protocol_errors")
                await websocket.close(code=1003, reason="binary MessagePack packets required")
                return
            try:
                observation = decode_observation(packet)
            except ProtocolError:
                self.metrics.increment("protocol_errors")
                await websocket.close(code=1007, reason="invalid observation packet")
                return

            if (
                self.allowed_robot_ids is not None
                and observation.robot_id not in self.allowed_robot_ids
            ):
                self.metrics.increment("protocol_errors")
                await websocket.close(code=1008, reason="robot id is not authorized")
                return
            if state.robot_id is None:
                state.robot_id = observation.robot_id
            elif observation.robot_id != state.robot_id:
                self.metrics.increment("protocol_errors")
                await websocket.close(code=1008, reason="robot id changed within a connection")
                return

            previous = state.last_observation_sequence
            if previous is not None and observation.sequence <= previous:
                self.metrics.increment("observation_out_of_order")
                continue
            if previous is not None and observation.sequence > previous + 1:
                self.metrics.increment(
                    "observation_sequence_gaps", observation.sequence - previous - 1
                )
            state.last_observation_sequence = observation.sequence
            state.newest_observation_sequence = observation.sequence
            self.metrics.increment("observations_received")

            if mailbox.full():
                try:
                    mailbox.get_nowait()
                except (
                    asyncio.QueueEmpty
                ):  # pragma: no cover - single event loop makes this extremely unlikely
                    pass
                else:
                    self.metrics.increment("observations_overwritten")
            mailbox.put_nowait(observation)

    async def _run_policy(
        self,
        websocket: ServerConnection,
        state: _ConnectionState,
        mailbox: asyncio.Queue[ObservationV1],
    ) -> None:
        while True:
            observation = await mailbox.get()
            start_ns = time.monotonic_ns()
            try:
                output = await self._invoke_policy(observation)
            except asyncio.CancelledError:
                raise
            except Exception:
                self.metrics.increment("policy_errors")
                LOGGER.exception("Policy callback failed for robot %s", observation.robot_id)
                await websocket.close(code=1011, reason="policy callback failed")
                return
            elapsed_ms = (time.monotonic_ns() - start_ns) / 1e6
            self.metrics.observe_policy(elapsed_ms)
            self.metrics.increment("chunks_generated")

            # The mailbox schedules the newest waiting observation next.  A
            # result already being computed remains useful until its explicit
            # TTL expires; dropping every such result would starve a continuous
            # camera stream whenever inference is slower than capture.
            if elapsed_ms / 1000.0 >= output.valid_for_s:
                self.metrics.increment("chunks_skipped_stale")
                continue

            chunk = ActionChunkV1(
                robot_id=observation.robot_id,
                chunk_id=state.next_chunk_id,
                based_on_observation_sequence=observation.sequence,
                created_monotonic_ns=time.monotonic_ns(),
                control_period_s=output.control_period_s,
                targets=output.targets,
                valid_for_s=output.valid_for_s,
                metadata=output.metadata,
            )
            state.next_chunk_id += 1
            try:
                await websocket.send(encode_action_chunk(chunk))
            except ConnectionClosed:
                return
            self.metrics.increment("chunks_sent")

    async def _invoke_policy(self, observation: ObservationV1) -> PolicyOutput:
        if inspect.iscoroutinefunction(self.policy):
            result: Any = await self.policy(observation)
        elif self.run_sync_policy_in_executor:
            result = await asyncio.to_thread(self.policy, observation)
        else:
            result = self.policy(observation)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, PolicyOutput):
            raise TypeError("policy callbacks must return PolicyOutput")
        return result
