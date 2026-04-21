"""
Reusable HTTP/2 client with pluggable defense strategies.

The :class:`H2Client` protocol owns everything that is about HTTP/2 itself
(connection setup, flow control, push, 103 Early Hints, pings) plus the
generic scaffolding that the defenses need to plug into (dummy-traffic
loop, ping loop, receive-side pacing, request shuffling and batching).

A :class:`Defense` object decides *what* to send as noise and *when*; the
client simply asks it at the right moments. Several defenses ship with
the project (``nop``, ``tamaraw``, ``front``, ``httpos``,
``llama``, ``h2pc``) and can be resolved by name via
:func:`get_defense`.

High-level runners :func:`send_requests` and :func:`run_test_case` are
provided so experimental code only needs to assemble requests + pick a
defense name.
"""

# future
from __future__ import annotations

# stdlib
import asyncio
from copy import deepcopy
import io
import json
import random
import ssl
import time
from typing import Any, Dict, List, Optional

# third party
from h2.config import H2Configuration
from h2.connection import H2Connection
from h2.errors import ErrorCodes
from h2.events import (
    DataReceived,
    InformationalResponseReceived,
    PingAckReceived,
    PingReceived,
    PushedStreamReceived,
    SettingsAcknowledged,
    StreamEnded,
    StreamReset,
    WindowUpdated,
)
from h2.exceptions import ProtocolError, StreamClosedError
from h2.settings import SettingCodes
from pydantic import BaseModel

# h2deflib absolute
from h2deflib.client_defenses.front import FRONT_DEFENSE
from h2deflib.client_defenses.h2pc import CLMODS_DEFENSE
from h2deflib.client_defenses.httpos import HTTPOS_DEFENSE
from h2deflib.client_defenses.llama import LLAMA_DEFENSE
from h2deflib.client_defenses.nop import NOP_DEFENSE
from h2deflib.client_defenses.tamaraw_qcsd import TAMARAW_QCSD_DEFENSE

DEFAULT_TEST_TIMEOUT = 60


# ====================================================================== #
# Data types                                                              #
# ====================================================================== #


class Request(BaseModel):
    path: str
    label: str
    data: dict = {}
    headers: dict = {}
    delay: float = 0
    expected_size: Optional[int] = None
    connection_id: Optional[str] = None


class ConnectionDetails(BaseModel):
    """Latency / size tracking for one connection, surfaced via ``stats()``."""

    response_window: int
    time_connect: float
    time_settings: Optional[float] = None
    streams_start: dict = {}
    streams_response_timestamps: dict = {}
    streams_response_sizes: dict = {}

    def stats(self) -> Dict[str, Any]:
        baseline = 0.0
        if self.time_settings is not None:
            baseline = self.time_settings - self.time_connect

        roundtrips = [baseline]
        frames = []
        for sid, ts in self.streams_response_timestamps.items():
            frames.append(1 + len(ts))
            if ts:
                roundtrips.append(ts[0] - self.streams_start[sid])
        return {
            "start_time": self.time_connect,
            "baseline": baseline,
            "frames": frames,
            "response_sizes": self.streams_response_sizes,
            "response_delay": roundtrips,
        }


# ====================================================================== #
# Defense resolver                                                        #
# ====================================================================== #


def get_defense(defense: str):
    """
    Return a defense strategy instance by name.

    The defense classes live in the ``client_defenses`` package (one
    module per strategy). Adding a new defense only requires dropping a
    file in there and extending this dispatcher.
    """

    registry = {
        "nop": NOP_DEFENSE,
        "tamaraw": TAMARAW_QCSD_DEFENSE,
        "front": FRONT_DEFENSE,
        "httpos": HTTPOS_DEFENSE,
        "llama": LLAMA_DEFENSE,
        "h2pc": CLMODS_DEFENSE,
    }
    if defense == "nop":
        return registry["nop"]
    if defense not in registry:
        raise NotImplementedError(defense)
    return deepcopy(registry[defense])


# ====================================================================== #
# H2Client                                                                #
# ====================================================================== #


class H2Client(asyncio.Protocol):
    """
    HTTP/2 client protocol driving an arbitrary :class:`Defense`.

    The experimental layer builds one of these per connection through
    :func:`send_single_request` / :func:`send_requests` / :func:`run_test_case`.
    """

    def __init__(
        self,
        connection_id: str,
        requests: List[Request],
        defense_name: str = "nop",
        request_server_defense: bool = False,
    ) -> None:
        cfg = H2Configuration(client_side=True, header_encoding="utf-8")
        self.conn: H2Connection = H2Connection(config=cfg)
        self.transport: Optional[asyncio.Transport] = None

        self.requests = requests
        self.dummy_requests = deepcopy(requests)
        self.connection_id = connection_id
        self.server_hints: List[str] = []

        defense = get_defense(defense_name)
        initial_window = defense.initial_window_size()
        self.window_size = initial_window
        self.defense = defense
        self.request_server_defense = request_server_defense

        self.conn_settings = {
            SettingCodes.INITIAL_WINDOW_SIZE: initial_window,
            SettingCodes.MAX_CONCURRENT_STREAMS: 9999,
        }

        self.max_dummy_time = 2  # seconds
        self.connection_start_time = time.time()
        self.padding_char = b"\x00"

        self.stream_data: Dict[int, io.BytesIO] = {}
        self.pending_streams: set = set()
        self.exit_event = asyncio.Event()
        # NB: ``get_event_loop`` is deprecated when no loop is running, but
        # this class is always constructed from inside ``create_connection``
        # i.e. with a running loop. We fall back to ``get_event_loop`` only
        # to support direct construction from tests.
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.get_event_loop()

        # Per-stream futures resolved when WINDOW_UPDATE arrives; prevents
        # ``_send_data`` from busy-looping when the flow control window is 0.
        self.flow_control_futures: Dict[int, asyncio.Future] = {}

        self.conn_traits = ConnectionDetails(
            response_window=initial_window,
            time_connect=time.time(),
        )
        self.conn_lock = asyncio.Lock()

        print(self.defense.summary())
        print(
            f"[conn] id={self.connection_id} t={time.time()}\n"
            f"       settings={self.conn_settings}\n"
            f"       request_server_defense={self.request_server_defense}"
        )

    # ------------------------------------------------------------------ #
    # asyncio.Protocol                                                   #
    # ------------------------------------------------------------------ #

    def connection_made(self, transport):
        self.transport = transport
        self.conn.initiate_connection()
        self.conn.update_settings(self.conn_settings)
        self._flush()
        self.loop.create_task(self._send_ping_loop())
        self.loop.create_task(self._send_dummy_traffic_loop())

    def connection_lost(self, exc):
        # Wake up any coroutines still blocked on WINDOW_UPDATE so they
        # return immediately instead of hanging until the test timeout.
        for f in self.flow_control_futures.values():
            if not f.done():
                f.cancel()
        self.flow_control_futures.clear()
        self.exit_event.set()

    def data_received(self, data):
        try:
            events = self.conn.receive_data(data)
        except ProtocolError as e:
            print(f"[h2deflib][client] protocol error: {e}")
            self._flush(swallow=True)
            self.transport.close()
            return

        self._flush(swallow=True)
        for event in events:
            self._dispatch_event(event)
        self._flush(swallow=True)

    def _dispatch_event(self, event):
        if isinstance(event, SettingsAcknowledged):
            self.conn_traits.time_settings = time.time()
            print(
                f"[traits] roundtrip = "
                f"{self.conn_traits.time_settings - self.conn_traits.time_connect:.4f}s"
            )
            asyncio.create_task(self._handle_next_request())
        elif isinstance(event, DataReceived):
            self._receive_data(event.data, event.stream_id)
        elif isinstance(event, StreamEnded):
            self._on_stream_end(event.stream_id)
        elif isinstance(event, PushedStreamReceived):
            print(f"[push] conn={self.connection_id} hdrs={event.headers}")
            self._log_push(
                event.headers, event.parent_stream_id, event.pushed_stream_id
            )
        elif isinstance(event, InformationalResponseReceived):
            self._on_informational(event.headers)
        elif isinstance(event, WindowUpdated):
            # Wake up any ``_send_data`` coroutine waiting on this stream.
            self._window_updated(event.stream_id, event.delta)
        elif isinstance(event, StreamReset):
            # Server aborted the stream — don't leave it dangling in
            # ``pending_streams`` or it'll block the exit condition forever.
            self._on_stream_reset(event.stream_id)
        elif isinstance(event, PingAckReceived):
            asyncio.create_task(self._handle_next_request())
        elif isinstance(event, PingReceived):
            pass

    # ------------------------------------------------------------------ #
    # Request sending                                                    #
    # ------------------------------------------------------------------ #

    async def _send_request(self, base_request: Request, is_noise: bool = False):
        def _prepare_single(request: Request, user_agent: str, data: bytes):
            hdrs = dict(request.headers)
            hdrs["label"] = request.label
            hdrs["connection_id"] = self.connection_id
            hdrs["defend_connection"] = str(int(self.request_server_defense))

            request_headers = [
                (":method", "GET"),
                (":scheme", "https"),
                (":path", request.path),
                (":authority", "localhost"),
                ("user-agent", user_agent),
                ("content-length", str(len(data))),
            ]
            for k, v in hdrs.items():
                request_headers.append((k.lower(), v))

            stream_id = self.conn.get_next_available_stream_id()
            print(f"[stream] {self.connection_id}-{stream_id} path={request.path}")
            self.conn.send_headers(stream_id, request_headers)

            now = time.time()
            self.conn_traits.streams_start[stream_id] = now
            self.conn_traits.streams_response_timestamps[stream_id] = []
            self.conn_traits.streams_response_sizes[stream_id] = request.expected_size
            self.pending_streams.add(stream_id)
            asyncio.create_task(self._send_data(stream_id, data))
            return (stream_id, request_headers, data)

        def _release_batch(reqs):
            results = [
                _prepare_single(r, ua, json.dumps({}).encode("utf8"))
                for (r, ua) in reqs
            ]
            self._flush()
            return results

        def _get_dummies():
            if is_noise or base_request.expected_size is None:
                return []
            if base_request.expected_size < 1000:
                return []
            dummies = self.defense.send_dummy_packet(
                self.dummy_requests,
                previous_request=base_request,
                window_size=self.window_size,
                stream_stats=self.conn_traits.stats(),
            )
            if not dummies:
                return []
            out = []
            for d in dummies:
                ua = self.defense.user_agent(is_noise=True)
                out.append((d, ua))
                print(f"[defense] extra path={d.path} size={d.expected_size} ua={ua}")
            return out

        pending: List = _get_dummies()

        packet_size = self.defense.send_packet_size()
        user_agent = self.defense.user_agent(is_noise=is_noise)
        # Pad the UA to a fixed on-the-wire size if requested.
        if packet_size > len(base_request.path) + len(user_agent):
            user_agent += "a" * (packet_size - len(base_request.path) - len(user_agent))

        # Ranged-request splitting (some defenses chop one request into N).
        if self.defense.use_ranged_requests():
            split = self.defense.split_for_ranged_requests(base_request)
            if split:
                pending.extend((r, user_agent) for r in split)
            else:
                pending.append((base_request, user_agent))
        else:
            pending.append((base_request, user_agent))

        if is_noise:
            return _release_batch(pending)

        delay = self.defense.should_delay_request()
        if delay > 0:
            await asyncio.sleep(delay)

        pending.extend(_get_dummies())
        return _release_batch(pending)

    async def _handle_next_request(
        self, custom_path: Optional[str] = None, is_noise: bool = False
    ):
        if self.defense.should_shuffle() and self.conn_traits.streams_start:
            random.shuffle(self.requests)

        next_idx = None
        if custom_path is not None:
            for i, req in enumerate(self.requests):
                if req.path == custom_path:
                    next_idx = i
                    break
            if next_idx is None:
                next_request = Request(path=custom_path, label="mock")
                await self._send_request(next_request, is_noise=is_noise)
                return

        if next_idx is None:
            if not self.requests:
                return
            next_idx = 0

        try:
            next_request = self.requests.pop(next_idx)
        except Exception:
            return
        await self._send_request(next_request, is_noise=is_noise)

    async def _send_data(self, stream_id: int, data: bytes):
        while data:
            while self.conn.local_flow_control_window(stream_id) < 1:
                try:
                    await self._wait_for_flow_control(stream_id)
                except asyncio.CancelledError:
                    return

            chunk_size = min(
                self.conn.local_flow_control_window(stream_id),
                len(data),
                self.conn.max_outbound_frame_size,
                self.window_size,
            )
            print(f"  >>>> [SEND FRAME] {stream_id} --> {chunk_size}")
            try:
                self.conn.send_data(
                    stream_id, data[:chunk_size], end_stream=(chunk_size == len(data))
                )
            except (StreamClosedError, ProtocolError):
                break
            self._flush()
            data = data[chunk_size:]

    # ------------------------------------------------------------------ #
    # Receive-side handling                                              #
    # ------------------------------------------------------------------ #

    def _receive_data(self, data: bytes, stream_id: int):
        try:
            stream_buf = self.stream_data.get(stream_id)
            if stream_buf is None:
                stream_buf = io.BytesIO()
                self.stream_data[stream_id] = stream_buf
        except KeyError:
            self.conn.reset_stream(stream_id, error_code=ErrorCodes.PROTOCOL_ERROR)
            return

        stream_buf.write(data)
        print(
            f"  >>>> [RECV FRAME] {stream_id} --> {len(data)} "
            f"pending={self.pending_streams} t={time.time()}"
        )
        self.conn_traits.streams_response_timestamps[stream_id].append(time.time())

        # Receive-side pacing via the defense.
        interval = self.defense.recv_interval(
            stream_stats=self.conn_traits.stats(), cumul_data=len(data)
        )
        if (time.time() - self.connection_start_time) > self.max_dummy_time:
            interval = 0

        if interval > 0:

            async def _delayed_window():
                await asyncio.sleep(interval)
                try:
                    self._send_window_update(stream_id)
                except Exception:
                    pass

            asyncio.create_task(_delayed_window())
        else:
            try:
                self._send_window_update(stream_id)
            except Exception:
                pass

    def _on_stream_end(self, stream_id: int):
        self._log_data(stream_id)
        self.pending_streams.discard(stream_id)

        if not self.pending_streams:
            asyncio.create_task(self._handle_next_request())

        if self.defense.should_batch():
            for _ in range(random.randrange(0, 4)):
                asyncio.create_task(self._handle_next_request())

        if not self.pending_streams and not self.requests:
            self.exit_event.set()

    def _on_informational(self, headers):
        """Handle 103 Early Hints — enqueue the hinted resources as requests."""
        self.server_hints = []
        for key, hint in headers:
            if key != "link":
                continue
            try:
                hinted = hint.split(";")[0].split("<")[-1].split(">")[0]
                self.server_hints.append(hinted)
            except Exception as e:
                print(f"[103] parse failure: {e}")

        if self.server_hints:
            asyncio.create_task(self._handle_hints_async(list(self.server_hints)))

    async def _handle_hints_async(self, hinted_paths: List[str]):
        for p in hinted_paths:
            print(f"[103][{self.connection_id}] handle {p}")
            await self._handle_next_request(custom_path=p)
            await asyncio.sleep(0.001)

    def _log_push(self, headers, parent_stream_id, stream_id):
        path = None
        for k, v in headers:
            if k == ":path":
                path = v
                break

        self.conn_traits.streams_start[stream_id] = time.time()
        self.conn_traits.streams_response_sizes[stream_id] = 0
        self.conn_traits.streams_response_timestamps[stream_id] = []

        if not self.requests:
            self.pending_streams.add(stream_id)
            return

        index_to_remove = None
        for i, req in enumerate(self.requests):
            if req.path == path:
                index_to_remove = i
                break

        print(f" >> [RECV PUSH] stream={stream_id} path={path}")
        if index_to_remove is not None:
            del self.requests[index_to_remove]
        self.pending_streams.add(stream_id)

    def _log_data(self, stream_id: int):
        buf = self.stream_data.get(stream_id)
        if buf is None:
            return
        buf.seek(0)
        print(
            f"  >>>> [RECV FULL] conn={self.connection_id} stream={stream_id} "
            f"len={len(buf.read())} pending={self.pending_streams}"
        )

    # ------------------------------------------------------------------ #
    # Flow control / pings / dummy traffic                               #
    # ------------------------------------------------------------------ #

    def _send_window_update(self, stream_id: int):
        inc = self.window_size
        self.conn.increment_flow_control_window(inc, stream_id=stream_id)
        self.conn.increment_flow_control_window(inc)
        self._flush()

    async def _wait_for_flow_control(self, stream_id: int):
        """
        Block until we receive a WINDOW_UPDATE frame that (may have) opened
        the send window. Resolved by ``_window_updated`` below. Previously
        this was a ``sleep(0)`` that busy-looped whenever the window was 0.
        """
        f: asyncio.Future = self.loop.create_future()
        self.flow_control_futures[stream_id] = f
        try:
            await f
        except asyncio.CancelledError:
            self.flow_control_futures.pop(stream_id, None)
            raise

    def _window_updated(self, stream_id: Optional[int], delta: int):
        """
        Resolve any coroutine waiting on this stream's send window. Stream id
        0 (``None`` here) is a connection-level window update and wakes up
        every waiter.
        """
        if stream_id and stream_id in self.flow_control_futures:
            f = self.flow_control_futures.pop(stream_id)
            if not f.done():
                f.set_result(delta)
        elif not stream_id:
            futures = list(self.flow_control_futures.values())
            self.flow_control_futures.clear()
            for f in futures:
                if not f.done():
                    f.set_result(delta)

    def _on_stream_reset(self, stream_id: int):
        """Clean up state when the server resets a stream."""
        self.pending_streams.discard(stream_id)
        f = self.flow_control_futures.pop(stream_id, None)
        if f is not None and not f.done():
            f.cancel()
        if not self.pending_streams and not self.requests:
            self.exit_event.set()

    async def _send_ping_loop(self):
        while not self.exit_event.is_set():
            count = self.defense.should_send_random_pings()
            if count > 0:
                print(f"[defense][{self.connection_id}] noise pings x{count}")
                for _ in range(count):
                    self._send_ping()
                    await asyncio.sleep(random.uniform(0, 0.01))
            else:
                await asyncio.sleep(0.5)
                self._send_ping()

    def _send_ping(self):
        self.conn.ping(b"\x00" * 8)
        self._flush()

    async def _send_dummy_traffic_loop(self):
        while not self.exit_event.is_set():
            if not self.defense.send_dummy_packet_loop:
                break
            if time.time() - self.connection_start_time > self.max_dummy_time:
                break

            dummies = self.defense.send_dummy_packet(
                self.dummy_requests,
                window_size=self.window_size,
                stream_stats=self.conn_traits.stats(),
            )
            if not dummies:
                await asyncio.sleep(0.01)
                continue

            await asyncio.sleep(self.defense.send_dummy_interval())
            for d in dummies:
                print(f"[defense] dummy path={d.path}")
                await self._send_request(d, is_noise=True)

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    async def wait_for_exit(self, timeout: float = DEFAULT_TEST_TIMEOUT):
        try:
            await asyncio.wait_for(self.exit_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            print(f"[timeout] conn {self.connection_id} after {timeout}s")
            self.stop()

    def stop(self):
        if self.transport:
            self.transport.close()
        self.exit_event.set()

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _flush(self, swallow: bool = False):
        buf = self.conn.data_to_send()
        if not buf or self.transport is None:
            return
        try:
            self.transport.write(buf)
        except (BrokenPipeError, ConnectionResetError, OSError):
            if not swallow:
                raise


# ====================================================================== #
# TLS + runners                                                           #
# ====================================================================== #


def make_client_ssl_context() -> ssl.SSLContext:
    """Build a client SSL context for HTTP/2 over TLS with ALPN."""
    # ``SSLContext()`` with no argument is deprecated; use the explicit
    # client protocol. We disable cert verification because the experiment
    # server uses a self-signed cert.
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    ctx.set_alpn_protocols(["h2"])
    return ctx


async def connect_h2_client(client_factory, host: str, port: int):
    ctx = make_client_ssl_context()
    loop = asyncio.get_running_loop()
    transport, protocol = await loop.create_connection(
        client_factory, host=host, port=port, ssl=ctx
    )
    return transport, protocol


async def send_single_request(
    connection_id: str,
    server_ip: str,
    server_port: int,
    requests: List[Request],
    defense_name: str = "nop",
    timeout: float = DEFAULT_TEST_TIMEOUT,
    request_server_defense: bool = False,
):
    ctx = make_client_ssl_context()
    loop = asyncio.get_running_loop()

    protocol = H2Client(
        connection_id=connection_id,
        requests=requests,
        defense_name=defense_name,
        request_server_defense=request_server_defense,
    )
    _, protocol = await loop.create_connection(
        lambda: protocol, host=server_ip, port=server_port, ssl=ctx
    )
    try:
        await protocol.wait_for_exit(timeout=timeout)
    finally:
        protocol.stop()


async def send_requests(
    server_ip: str,
    server_port: int,
    requests_by_connection: Dict[str, List[Request]],
    request_server_defenses: Dict[str, bool],
    defense_name: str = "nop",
):
    """
    Fire off one connection per key in ``requests_by_connection`` concurrently.
    """
    coros = []
    for conn_id, reqs in requests_by_connection.items():
        print(f"[run] new connection {conn_id}")
        coros.append(
            send_single_request(
                conn_id,
                server_ip,
                server_port,
                reqs,
                defense_name=defense_name,
                request_server_defense=request_server_defenses.get(conn_id, False),
            )
        )
    await asyncio.gather(*coros)


def run_test_case(
    server_ip: str,
    server_port: int,
    requests_by_connection: Dict[str, List[Request]],
    request_server_defenses: Dict[str, bool],
    defense_name: str = "nop",
):
    """Synchronous entry point. Spins up an event loop and runs one test case."""
    return asyncio.run(
        send_requests(
            server_ip,
            server_port,
            requests_by_connection=requests_by_connection,
            request_server_defenses=request_server_defenses,
            defense_name=defense_name,
        )
    )
