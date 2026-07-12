"""Outbound robot client for remote policy inference over Wi-Fi."""

from __future__ import annotations

import asyncio
import inspect
import logging
import ssl
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from typing import Any
from urllib.parse import urlsplit

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import WebSocketException

from .action_buffer import ActionInterpolationBuffer, ActionSample, InstallResult
from .metrics import TransportMetrics
from .protocol import (
    MAX_PACKET_BYTES,
    ActionChunkV1,
    ObservationV1,
    ProtocolError,
    decode_action_chunk,
    encode_observation,
)

LOGGER = logging.getLogger(__name__)
ObservationSource = Callable[[int, int], ObservationV1 | Awaitable[ObservationV1]]
ActionSink = Callable[[ActionSample], None | Awaitable[None]]


class ObservationSourceError(RuntimeError):
    pass


class ActionSinkError(RuntimeError):
    pass


class RobotWebSocketClient:
    """Streams observations out and executes fresh action chunks locally.

    The client always initiates the connection, which works with ordinary LAN
    Wi-Fi and avoids exposing a service on the robot.  A hardware integration
    supplies two callbacks: an observation source and an action sink.  This
    module intentionally contains no board- or servo-specific assumptions.
    """

    def __init__(
        self,
        *,
        uri: str,
        robot_id: str,
        observation_source: ObservationSource,
        action_sink: ActionSink,
        observation_hz: float = 15.0,
        control_hz: float = 50.0,
        watchdog_timeout_s: float = 0.25,
        safe_fallback: tuple[float, float, float, float, float, float, float, float] | None = None,
        reconnect_delay_s: float = 1.0,
        ssl_context: ssl.SSLContext | None = None,
        additional_headers: Mapping[str, str] | None = None,
        ping_interval_s: float = 10.0,
        ping_timeout_s: float = 10.0,
        max_observation_history: int = 2048,
        run_sync_source_in_executor: bool = True,
        run_sync_sink_in_executor: bool = False,
        metrics: TransportMetrics | None = None,
    ) -> None:
        scheme = urlsplit(uri).scheme
        if scheme not in {"ws", "wss"}:
            raise ValueError("uri must use ws:// or wss://")
        if not robot_id or len(robot_id) > 128:
            raise ValueError("robot_id must be a non-empty string of at most 128 characters")
        if observation_hz <= 0.0 or control_hz <= 0.0:
            raise ValueError("observation_hz and control_hz must be positive")
        if reconnect_delay_s < 0.0:
            raise ValueError("reconnect_delay_s may not be negative")
        if max_observation_history < 2:
            raise ValueError("max_observation_history must be at least two")
        if not callable(observation_source) or not callable(action_sink):
            raise TypeError("observation_source and action_sink must be callable")

        self.uri = uri
        self.robot_id = robot_id
        self.observation_source = observation_source
        self.action_sink = action_sink
        self.observation_period_s = 1.0 / observation_hz
        self.control_period_s = 1.0 / control_hz
        self.reconnect_delay_s = reconnect_delay_s
        self.ssl_context = ssl_context
        self.additional_headers = dict(additional_headers or {})
        self.ping_interval_s = ping_interval_s
        self.ping_timeout_s = ping_timeout_s
        self.max_observation_history = max_observation_history
        self.run_sync_source_in_executor = run_sync_source_in_executor
        self.run_sync_sink_in_executor = run_sync_sink_in_executor
        self.metrics = metrics or TransportMetrics()
        self.action_buffer = ActionInterpolationBuffer(
            watchdog_timeout_s=watchdog_timeout_s,
            safe_fallback=safe_fallback,
        )

        self._observation_capture_times: OrderedDict[int, int] = OrderedDict()
        self._next_observation_sequence = 0
        self._stop_event: asyncio.Event | None = None
        self._run_loop: asyncio.AbstractEventLoop | None = None
        self._running = False

    async def run(self) -> None:
        """Run reconnect, streaming, and local control loops until ``stop``."""

        if self._running:
            raise RuntimeError("robot client is already running")
        self._running = True
        self._run_loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        network_task = asyncio.create_task(self._reconnect_loop(), name="sesame-policy-network")
        control_task = asyncio.create_task(self._control_loop(), name="sesame-local-control")
        try:
            done, pending = await asyncio.wait(
                {network_task, control_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                task.result()
        finally:
            self.action_buffer.invalidate("client_stopped")
            # Explicit shutdown and callback failures must be just as safe as a
            # Wi-Fi loss: deliver the adapter's fallback contract once before
            # returning control to the application.
            final_sample = self.action_buffer.sample(now_ns=time.monotonic_ns())
            try:
                await self._invoke_sink(final_sample)
            except ActionSinkError:
                LOGGER.exception("Failed to deliver the final safe fallback action")
            self._running = False
            self._run_loop = None

    def stop(self) -> None:
        """Request a clean stop; safe to call from a different thread."""

        event, loop = self._stop_event, self._run_loop
        if event is None or loop is None:
            return
        loop.call_soon_threadsafe(event.set)

    def accept_action_chunk(
        self,
        chunk: ActionChunkV1,
        *,
        received_at_ns: int | None = None,
    ) -> InstallResult:
        """Validate and install one decoded chunk (also useful in local tests)."""

        now_ns = time.monotonic_ns() if received_at_ns is None else received_at_ns
        self.metrics.increment("chunks_received")
        if chunk.robot_id != self.robot_id:
            self.metrics.increment("protocol_errors")
            raise ProtocolError("action chunk robot_id does not match this client")
        observation_capture_ns = self._observation_capture_times.get(
            chunk.based_on_observation_sequence
        )
        if observation_capture_ns is None:
            self.metrics.increment("chunks_rejected_sequence")
            return InstallResult.REJECTED_SEQUENCE

        round_trip_ms = max(0.0, (now_ns - observation_capture_ns) / 1e6)
        self.metrics.observe_round_trip(round_trip_ms)
        result = self.action_buffer.install(
            chunk,
            received_at_ns=now_ns,
            originating_observation_ns=observation_capture_ns,
        )
        if result is InstallResult.REJECTED_STALE:
            self.metrics.increment("chunks_rejected_stale")
        elif result is InstallResult.REJECTED_SEQUENCE:
            self.metrics.increment("chunks_rejected_sequence")
        else:
            self.metrics.increment("chunks_accepted")
            if result is InstallResult.REPLACED:
                self.metrics.increment("chunks_replaced")
            self._discard_observation_history_through(chunk.based_on_observation_sequence)
        return result

    async def control_once(self, *, now_ns: int | None = None) -> ActionSample:
        """Sample and deliver one action, primarily for deterministic runtimes."""

        sample = self.action_buffer.sample(now_ns=now_ns)
        if sample.entered_fallback:
            self.metrics.increment("watchdog_entries")
        await self._invoke_sink(sample)
        return sample

    async def _reconnect_loop(self) -> None:
        first_attempt = True
        while not self._stopping:
            if not first_attempt:
                self.metrics.increment("reconnects")
            first_attempt = False
            try:
                await self._run_connection()
            except asyncio.CancelledError:
                raise
            except (OSError, TimeoutError, WebSocketException) as error:
                if not self._stopping:
                    LOGGER.warning("Policy connection unavailable (%s); retrying", error)
            finally:
                self.action_buffer.invalidate("disconnected")
            if not self._stopping:
                await self._wait_until_stop(self.reconnect_delay_s)

    async def _run_connection(self) -> None:
        async with connect(
            self.uri,
            ssl=self.ssl_context,
            additional_headers=self.additional_headers or None,
            ping_interval=self.ping_interval_s,
            ping_timeout=self.ping_timeout_s,
            max_size=MAX_PACKET_BYTES,
            compression=None,
            open_timeout=10.0,
            close_timeout=3.0,
        ) as websocket:
            self.metrics.increment("connections")
            self.action_buffer.reset_sequence_epoch("new_connection")
            self._observation_capture_times.clear()
            sender = asyncio.create_task(
                self._send_observations(websocket), name="sesame-observation-sender"
            )
            receiver = asyncio.create_task(
                self._receive_actions(websocket), name="sesame-action-receiver"
            )
            assert self._stop_event is not None
            stopper = asyncio.create_task(self._stop_event.wait(), name="sesame-connection-stop")
            done, pending = await asyncio.wait(
                {sender, receiver, stopper}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                if task is not stopper:
                    task.result()

    async def _send_observations(self, websocket: ClientConnection) -> None:
        loop = asyncio.get_running_loop()
        next_tick = loop.time()
        while not self._stopping:
            sequence = self._next_observation_sequence
            self._next_observation_sequence += 1
            requested_capture_ns = time.monotonic_ns()
            try:
                observation = await self._invoke_source(sequence, requested_capture_ns)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                raise ObservationSourceError("observation source failed") from error
            if not isinstance(observation, ObservationV1):
                raise ObservationSourceError("observation source must return ObservationV1")
            if observation.robot_id != self.robot_id:
                raise ObservationSourceError("observation source returned the wrong robot_id")
            if observation.sequence != sequence:
                raise ObservationSourceError(
                    "observation source returned sequence "
                    f"{observation.sequence}; expected {sequence}"
                )
            now_ns = time.monotonic_ns()
            if observation.monotonic_ns > now_ns:
                raise ObservationSourceError("observation capture timestamp is in the future")

            self._observation_capture_times[sequence] = observation.monotonic_ns
            while len(self._observation_capture_times) > self.max_observation_history:
                self._observation_capture_times.popitem(last=False)
            try:
                await websocket.send(encode_observation(observation))
            except BaseException:
                self._observation_capture_times.pop(sequence, None)
                raise
            self.metrics.increment("observations_sent")

            next_tick += self.observation_period_s
            delay = next_tick - loop.time()
            if delay <= 0.0:
                next_tick = loop.time()
                await asyncio.sleep(0)
            else:
                await asyncio.sleep(delay)

    async def _receive_actions(self, websocket: ClientConnection) -> None:
        async for packet in websocket:
            if not isinstance(packet, bytes):
                self.metrics.increment("protocol_errors")
                raise ProtocolError("policy host sent a non-binary packet")
            try:
                chunk = decode_action_chunk(packet)
            except ProtocolError:
                self.metrics.increment("protocol_errors")
                raise
            self.accept_action_chunk(chunk, received_at_ns=time.monotonic_ns())

    async def _control_loop(self) -> None:
        loop = asyncio.get_running_loop()
        next_tick = loop.time()
        while not self._stopping:
            await self.control_once(now_ns=time.monotonic_ns())
            next_tick += self.control_period_s
            delay = next_tick - loop.time()
            if delay <= 0.0:
                next_tick = loop.time()
                await asyncio.sleep(0)
            else:
                await asyncio.sleep(delay)

    async def _invoke_source(self, sequence: int, monotonic_ns: int) -> ObservationV1:
        if inspect.iscoroutinefunction(self.observation_source):
            result: Any = await self.observation_source(sequence, monotonic_ns)
        elif self.run_sync_source_in_executor:
            result = await asyncio.to_thread(self.observation_source, sequence, monotonic_ns)
        else:
            result = self.observation_source(sequence, monotonic_ns)
        if inspect.isawaitable(result):
            result = await result
        return result

    async def _invoke_sink(self, sample: ActionSample) -> None:
        try:
            if inspect.iscoroutinefunction(self.action_sink):
                result: Any = await self.action_sink(sample)
            elif self.run_sync_sink_in_executor:
                result = await asyncio.to_thread(self.action_sink, sample)
            else:
                result = self.action_sink(sample)
            if inspect.isawaitable(result):
                await result
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.metrics.increment("control_sink_errors")
            self.action_buffer.invalidate("control_sink_error")
            raise ActionSinkError("action sink failed") from error

    def _discard_observation_history_through(self, sequence: int) -> None:
        for old_sequence in tuple(self._observation_capture_times):
            if old_sequence <= sequence:
                self._observation_capture_times.pop(old_sequence, None)
            else:
                break

    async def _wait_until_stop(self, timeout_s: float) -> None:
        assert self._stop_event is not None
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=timeout_s)
        except TimeoutError:
            pass

    @property
    def _stopping(self) -> bool:
        return self._stop_event is not None and self._stop_event.is_set()
