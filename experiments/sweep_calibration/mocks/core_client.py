# stdlib
import asyncio
import io
import json
import os
import random
import ssl
import time
from copy import deepcopy
from typing import List, Optional

# CLIENT DEFENSES
# get_defense(name, level) returns a deepcopy, so each Client gets its own
# per-connection defense state.  The individual client_defenses/*.py modules
# are unchanged; levels.py derives the intensity ladder from them.
from client_defenses.levels import get_defense

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
)
from h2.exceptions import ProtocolError, StreamClosedError
from h2.settings import SettingCodes
from pydantic import BaseModel

TEST_TIMEOUT = 60

# A PING has two unrelated jobs here: it paces request dispatch (the ACK
# triggers handle_next_request) and it is defensive padding.  Distinct 8-byte
# payloads keep them apart, so raising the padding volume no longer raises
# request concurrency -- without this, a defense could load a page FASTER
# than the undefended baseline.
PUMP_PING = b"\x00" * 8
NOISE_PING = b"\xa5" * 8
PUMP_PING_INTERVAL = 0.5  # unchanged from the submitted behaviour

# Marks a send_dummy_packet call triggered by an arriving response rather
# than an outgoing request.
_RESPONSE_TRIGGER = object()

# Frame-level prints are very expensive across a multi-cell sweep.
# Set H2_VERBOSE=0 to silence them; default keeps the previous behaviour.
VERBOSE = os.environ.get("H2_VERBOSE", "1") != "0"


def _log(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


class Request(BaseModel):
    path: str
    label: str
    data: dict = {}
    headers: dict = {}
    delay: float = 0
    expected_size: Optional[int] = None
    content_type: Optional[str] = None
    connection_id: Optional[str] = None


class ConnectionDetails(BaseModel):
    response_window: int
    time_connect: float
    time_settings: Optional[float] = None
    streams_start: dict = {}
    streams_response_timestamps: dict = {}
    streams_response_sizes: dict = {}

    def stats(self):
        baseline_roundtrip = 0
        if self.time_settings is not None:
            baseline_roundtrip = self.time_settings - self.time_connect

        roundtrips = [baseline_roundtrip]
        frames = []
        for stream_id in self.streams_response_timestamps:
            frames.append(1 + len(self.streams_response_timestamps[stream_id]))
            if len(self.streams_response_timestamps[stream_id]) == 0:
                continue
            roundtrips.append(
                self.streams_response_timestamps[stream_id][0]
                - self.streams_start[stream_id]
            )
        return {
            "start_time": self.time_connect,
            "baseline": baseline_roundtrip,
            "frames": frames,
            "response_sizes": self.streams_response_sizes,
            "response_delay": roundtrips,
        }


class Client(asyncio.Protocol):
    def __init__(
        self,
        connection_id: str,
        requests: List[Request],
        defense_name="nop",
        request_server_defense: bool = False,
        defense_level: str = "mid1",
    ) -> None:
        config = H2Configuration(client_side=True, header_encoding="utf-8")
        self.conn = H2Connection(config=config)
        self.transport = None
        self.requests = requests
        self.dummy_requests = deepcopy(requests)
        self.connection_id = connection_id
        self.server_hints = []

        defense = get_defense(defense_name, defense_level)
        self.defense_level = defense_level
        initial_window_size = defense.initial_window_size()
        self.window_size = initial_window_size

        self.conn_settings = {
            SettingCodes.INITIAL_WINDOW_SIZE: initial_window_size,
            SettingCodes.MAX_CONCURRENT_STREAMS: 9999,
        }
        self.defense = defense
        self.request_server_defense = request_server_defense

        self.max_dummy_time = defense.max_dummy_time
        self.connection_start_time = time.time()

        self.padding_char = b"\x00"
        _log(self.defense.summary())
        _log(
            f"""
            [CONN] {self.connection_id} {time.time()}
            [DEFENSE] {defense_name}/{defense_level} {self.conn_settings}
            [SERVER DEFENSE] {self.request_server_defense}
              """
        )
        self.stream_data = {}
        self.pending_streams = set()
        self.exit_event = asyncio.Event()

        self.loop = asyncio.get_event_loop()

        self.conn_traits = ConnectionDetails(
            response_window=initial_window_size,
            time_connect=time.time(),
        )
        self.conn_lock = asyncio.Lock()

        # --- overhead tracking ---
        self.real_stream_ids = set()  # stream IDs for non-dummy requests

        self.t_real_end = None  # wall time when last real stream ends

        self.bytes_rx_real = 0  # download bytes on real streams
        self.bytes_rx_total = 0  # download bytes on ALL streams (real + push + hints)
        self.bytes_rx_at_real_end = None  # snapshot of total download at t_real_end

        self.bytes_tx_real = 0  # upload bytes on real streams
        self.bytes_tx_total = 0  # upload bytes on all streams (real + dummy)
        self.bytes_tx_at_real_end = None  # snapshot of total upload at t_real_end

    def connection_made(self, transport):
        self.transport = transport
        self.conn.initiate_connection()
        self.conn.update_settings(self.conn_settings)
        self.transport.write(self.conn.data_to_send())
        self.loop.create_task(self.send_ping_in_loop())
        self.loop.create_task(self.send_noise_pings())
        self.loop.create_task(self.send_dummy_traffic())

    async def send_data(self, stream_id, data):
        """
        Send data according to the flow control rules.
        """
        while data:
            while self.conn.local_flow_control_window(stream_id) < 1:
                try:
                    await self.wait_for_flow_control(stream_id)
                except asyncio.CancelledError:
                    return

            chunk_size = min(
                self.conn.local_flow_control_window(stream_id),
                len(data),
                self.conn.max_outbound_frame_size,
                self.window_size,
            )

            chunk = data[:chunk_size]

            _log("   >>>> [SEND FRAME] ", stream_id, "-->", len(chunk))
            try:
                self.conn.send_data(
                    stream_id, chunk, end_stream=(chunk_size == len(data))
                )
            except (StreamClosedError, ProtocolError):
                break

            # Count the actual bytes going on the wire (DATA frame header + payload),
            # and attribute only to streams that actually issued the send.
            frame_bytes = self.conn.data_to_send()
            self.bytes_tx_total += len(frame_bytes)
            if stream_id in self.real_stream_ids:
                self.bytes_tx_real += len(frame_bytes)
            self.transport.write(frame_bytes)

            data = data[chunk_size:]

    async def send_request(self, base_request, is_noise=False):
        def _prepare_single_request(request, user_agent, data, tag_as_noise=False):
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
            _log(f"STREAM = {self.connection_id}-{stream_id} PATH={request.path}")

            pending_buf = self.conn.data_to_send()
            if pending_buf:
                self.bytes_tx_total += len(
                    pending_buf
                )  # counts as connection overhead, not real
                self.transport.write(pending_buf)

            self.conn.send_headers(stream_id, request_headers)

            header_frame = self.conn.data_to_send()
            if header_frame:
                self.bytes_tx_total += len(header_frame)
                if not tag_as_noise:
                    self.bytes_tx_real += len(header_frame)
                self.transport.write(header_frame)

            now = time.time()
            self.conn_traits.streams_start[stream_id] = now

            self.conn_traits.streams_response_timestamps[stream_id] = []
            self.conn_traits.streams_response_sizes[stream_id] = request.expected_size
            self.pending_streams.add(stream_id)

            # tag real vs dummy streams — use explicit flag, not closure
            if not tag_as_noise:
                self.real_stream_ids.add(stream_id)

            asyncio.create_task(self.send_data(stream_id, data))

            return (stream_id, request_headers, data)

        def _release_batch(requests, noise_flags):
            _log(f"[DBG] Handle batch sz={len(requests)} noise={is_noise}")
            results = []
            data = json.dumps({}).encode("utf8")

            for (req, user_agent), flag in zip(requests, noise_flags):
                result = _prepare_single_request(
                    req, user_agent, data, tag_as_noise=flag
                )
                results.append(result)

            self.transport.write(self.conn.data_to_send())
            return results

        def _get_dummies():
            if is_noise:
                return []

            floor = self.defense.dummy_min_resource_size
            if floor > 0 and (
                base_request.expected_size is None or base_request.expected_size < floor
            ):
                return []

            _log("[DBG] dummies for ", base_request.expected_size)
            dummy_requests = self.defense.send_dummy_packet(
                self.dummy_requests,
                previous_request=base_request,
                window_size=self.window_size,
                stream_stats=self.conn_traits.stats(),
            )
            if dummy_requests is None:
                return []

            noise = []
            for dummy_request in dummy_requests:
                user_agent = self.defense.user_agent(is_noise=True)
                noise.append((dummy_request, user_agent))
                _log(
                    f" >>> [DEFENSE] Extra path = {dummy_request.path} {dummy_request.headers} size = {dummy_request.expected_size} user-agent = {user_agent}"
                )
            return noise

        # collect all requests with explicit noise flags
        pending_requests = []
        noise_flags = []

        # pre-request dummies — always noise
        for item in _get_dummies():
            pending_requests.append(item)
            noise_flags.append(True)

        packet_size = self.defense.send_packet_size()
        user_agent = self.defense.user_agent(is_noise=is_noise)
        _log(f"[DEFENSE] user-agent = {user_agent}")
        if packet_size > len(base_request.path) + len(user_agent):
            _log(f"[DEFENSE] Send packets with padding {packet_size} ")
            user_agent += "a" * (packet_size - len(base_request.path) - len(user_agent))

        if self.defense.use_ranged_requests():
            ranged_requests = self.defense.split_for_ranged_requests(base_request)
            if len(ranged_requests) > 0:
                _log(f"[DEFENSE] Using ranged requests N = {len(ranged_requests)}")
                for req in ranged_requests:
                    pending_requests.append((req, user_agent))
                    noise_flags.append(is_noise)
            else:
                pending_requests.append((base_request, user_agent))
                noise_flags.append(is_noise)
        else:
            pending_requests.append((base_request, user_agent))
            noise_flags.append(is_noise)

        if is_noise:
            return _release_batch(pending_requests, noise_flags)

        delay_req = self.defense.should_delay_request()
        if delay_req > 0:
            _log("[DEFENSE] Delay request")
            await asyncio.sleep(delay_req)

        # post-request dummies — always noise
        for item in _get_dummies():
            pending_requests.append(item)
            noise_flags.append(True)

        return _release_batch(pending_requests, noise_flags)

    async def handle_next_request(self, custom_path=None, is_noise: bool = False):
        if self.defense.should_shuffle() and len(self.conn_traits.streams_start) > 0:
            random.shuffle(self.requests)

        next_idx = None

        if custom_path is not None:
            for idx, req in enumerate(self.requests):
                if req.path == custom_path:
                    next_idx = idx
                    break
        if next_idx is None and custom_path is not None:
            next_request = Request(
                **{
                    "path": custom_path,
                    "label": "mock",
                }
            )
        else:
            if next_idx is None:
                next_idx = 0
            try:
                next_request = self.requests.pop(next_idx)
            except BaseException:
                return

        await self.send_request(next_request, is_noise=is_noise)

    def data_received(self, data):
        try:
            events = self.conn.receive_data(data)
        except ProtocolError as e:
            _log(" ERR: data_received proto error", e)
            self.transport.write(self.conn.data_to_send())
            self.transport.close()
        else:
            self.transport.write(self.conn.data_to_send())
            for event in events:
                if isinstance(event, SettingsAcknowledged):
                    self.conn_traits.time_settings = time.time()
                    _log(
                        f"[TRAITS] Roundtrip = {self.conn_traits.time_settings - self.conn_traits.time_connect} s"
                    )
                    asyncio.create_task(self.handle_next_request())

                elif isinstance(event, DataReceived):
                    self.receive_data(event.data, event.stream_id)

                elif isinstance(event, StreamEnded):
                    self.log_data(event.stream_id)

                    was_real = event.stream_id in self.real_stream_ids
                    if was_real:
                        self.real_stream_ids.remove(event.stream_id)
                        if self.defense.dummy_on_response:
                            asyncio.create_task(self.send_response_dummy())

                    if event.stream_id in self.pending_streams:
                        self.pending_streams.remove(event.stream_id)

                    # Snapshot only when ALL real streams have finished AND no more real
                    # requests are waiting to be issued. Without the second clause we'd
                    # snapshot between sequential requests (real_stream_ids transiently empty).
                    if (
                        len(self.real_stream_ids) == 0
                        and len(self.requests) == 0
                        and self.t_real_end is None
                    ):
                        self.t_real_end = time.time()
                        self.bytes_rx_at_real_end = self.bytes_rx_total
                        self.bytes_tx_at_real_end = self.bytes_tx_total

                    if len(self.pending_streams) == 0:
                        asyncio.create_task(self.handle_next_request())

                    if (
                        self.defense.should_batch()
                        and was_real
                        and len(self.requests) > 1
                    ):
                        if random.random() < 0.3:
                            _log("[DEFENSE] Batch one extra request")
                            asyncio.create_task(self.handle_next_request())

                    if len(self.pending_streams) == 0 and len(self.requests) == 0:
                        self.exit_event.set()

                elif isinstance(event, PushedStreamReceived):
                    _log(
                        "[HTTP2][PUSH] Server pushed", self.connection_id, event.headers
                    )
                    self.log_push(
                        event.headers, event.parent_stream_id, event.pushed_stream_id
                    )

                elif isinstance(event, PingAckReceived):
                    # padding pings must not pump the request queue
                    if bytes(getattr(event, "ping_data", PUMP_PING)) == PUMP_PING:
                        asyncio.create_task(self.handle_next_request())

                elif isinstance(event, PingReceived):
                    self.respond_ping(event)

                elif isinstance(event, InformationalResponseReceived):
                    self.server_hints = []
                    _log("Received HINTS", len(event.headers))
                    for key, hint in event.headers:
                        if key != "link":
                            continue
                        try:
                            hinted_path = (
                                hint.split(";")[0].split("<")[-1].split(">")[0]
                            )
                            self.server_hints.append(hinted_path)
                        except BaseException as e:
                            _log("Early Hints failure", e)
                            continue

                    if len(self.server_hints) == 0:
                        continue

                    _log("HANDLE HINTS ASYNC")
                    asyncio.create_task(self.handle_hints_async(self.server_hints))

        self.transport.write(self.conn.data_to_send())

    async def handle_hints_async(self, hinted_paths):
        for hinted_path in hinted_paths:
            _log(f"[HTTP2][HINTS103][{self.connection_id}] Handle {hinted_path}")
            await self.handle_next_request(custom_path=hinted_path, is_noise=True)
            await asyncio.sleep(0.001)

    def log_push(self, headers, pid, stream_id):
        path = None
        for header in headers:
            if header[0] == ":path":
                path = header[1]

        self.conn_traits.streams_start[stream_id] = time.time()
        self.conn_traits.streams_response_sizes[stream_id] = 0
        self.conn_traits.streams_response_timestamps[stream_id] = []

        index_to_remove = None
        if len(self.requests) == 0:
            return

        for idx, pending_req in enumerate(self.requests):
            pending_path = pending_req.path
            if path == pending_path:
                index_to_remove = idx
                break

        _log(" >> [RECV PUSH]", stream_id, path, idx)

        if index_to_remove is not None:
            del self.requests[index_to_remove]

        self.pending_streams.add(stream_id)

    def receive_data(self, data, stream_id):
        try:
            if stream_id in self.stream_data:
                stream_data = self.stream_data[stream_id]
            else:
                stream_data = io.BytesIO()
                self.stream_data[stream_id] = stream_data
        except KeyError:
            self.conn.reset_stream(stream_id, error_code=ErrorCodes.PROTOCOL_ERROR)
        else:
            stream_data.write(data)

        # track download bytes — only count real streams
        self.bytes_rx_total += len(data)
        if stream_id in self.real_stream_ids:
            self.bytes_rx_real += len(data)

        _log(
            "   >>>> [RECV FRAME] ",
            stream_id,
            "-->",
            len(data),
            self.pending_streams,
            time.time(),
        )
        self.conn_traits.streams_response_timestamps[stream_id].append(time.time())

        recv_interval = self.defense.recv_interval(
            stream_stats=self.conn_traits.stats(),
            cumul_data=len(data),
        )
        if (time.time() - self.connection_start_time) > self.max_dummy_time:
            recv_interval = 0
        if recv_interval > 0:
            _log(f"[DEFENSE] Recv interval {recv_interval} ")

            async def _handle_recv_delay(recv_interval):
                start_time = time.time()
                await asyncio.sleep(recv_interval)
                try:
                    self.send_window_update(stream_id)
                    _log(
                        "update frame",
                        start_time,
                        time.time(),
                        recv_interval,
                        stream_id,
                    )
                except BaseException:
                    pass

            asyncio.create_task(_handle_recv_delay(recv_interval))
        else:
            try:
                self.send_window_update(stream_id)
            except BaseException:
                pass

    def send_window_update(self, stream_id):
        window_size_increment = self.window_size
        self.conn.increment_flow_control_window(
            window_size_increment, stream_id=stream_id
        )
        self.conn.increment_flow_control_window(window_size_increment)
        self.transport.write(self.conn.data_to_send())

    def log_data(self, stream_id):
        data = self.stream_data[stream_id]
        data.seek(0)
        _log(
            "   >>>> [RECV FULL]",
            self.connection_id,
            stream_id,
            "--> data len",
            len(data.read()),
            self.pending_streams,
        )

    async def wait_for_exit(self, timeout=TEST_TIMEOUT):
        try:
            await asyncio.wait_for(self.exit_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            _log(
                f"[TIMEOUT] Connection {self.connection_id} timed out after {timeout} seconds."
            )
            self.stop()

    def stop(self):
        self.transport.close()
        self.exit_event.set()

    def send_ping(self, noise: bool = False):
        self.conn.ping(NOISE_PING if noise else PUMP_PING)
        self.transport.write(self.conn.data_to_send())

    def respond_ping(self, event):
        pass

    async def send_ping_in_loop(self):
        """Request pump.  Fixed cadence for every defense and every level, so
        dispatch concurrency is not a function of padding volume."""
        while not self.exit_event.is_set():
            await asyncio.sleep(PUMP_PING_INTERVAL)
            self.send_ping()

    async def send_noise_pings(self):
        """Defensive PING padding.  Never drives request dispatch."""
        while not self.exit_event.is_set():
            should_send_pings = self.defense.should_send_random_pings()
            if should_send_pings > 0:
                _log(
                    f"[DEFENSE][{self.connection_id}] Send {should_send_pings} noise PINGS"
                )
                for pingid in range(should_send_pings):
                    self.send_ping(noise=True)
                    await asyncio.sleep(random.uniform(0, 0.01))
            # paced regardless of probability: the old loop lost its only
            # sleep once ping_probability reached 1.0 and became a spin
            await asyncio.sleep(getattr(self.defense, "ping_burst_interval", 0.05))

    async def send_response_dummy(self):
        dummies = self.defense.send_dummy_packet(
            self.dummy_requests,
            previous_request=_RESPONSE_TRIGGER,
            window_size=self.window_size,
            stream_stats=self.conn_traits.stats(),
        )
        for dummy_request in dummies or []:
            await self.send_request(dummy_request, is_noise=True)

    async def send_dummy_traffic(self):
        while not self.exit_event.is_set():
            if not self.defense.send_dummy_packet_loop:
                break
            dummy_requests = self.defense.send_dummy_packet(
                self.dummy_requests,
                window_size=self.window_size,
                stream_stats=self.conn_traits.stats(),
            )

            if time.time() - self.connection_start_time > self.max_dummy_time:
                break

            if dummy_requests is None:
                await asyncio.sleep(0.01)
                continue

            send_dummy_interval = self.defense.send_dummy_interval()
            await asyncio.sleep(send_dummy_interval)

            for dummy_request in dummy_requests:
                _log(
                    f"[DEFENSE] Send dummy traffic {send_dummy_interval} {dummy_request.path}"
                )
                await self.send_request(dummy_request, is_noise=True)


async def send_single_request(
    connection_id,
    server_ip,
    server_port,
    requests,
    defense_name="nop",
    timeout=TEST_TIMEOUT,
    request_server_defense: bool = False,
    defense_level: str = "mid1",
) -> dict:
    ctx = ssl.SSLContext()
    ctx.set_alpn_protocols(["h2"])
    ctx.check_hostname = False

    loop = asyncio.get_running_loop()

    protocol = Client(
        connection_id=connection_id,
        requests=requests,
        defense_name=defense_name,
        request_server_defense=request_server_defense,
        defense_level=defense_level,
    )
    coro = loop.create_connection(
        lambda: protocol, host=server_ip, port=server_port, ssl=ctx
    )
    transport, protocol = await coro

    try:
        await protocol.wait_for_exit(timeout=timeout)
    finally:
        protocol.stop()

    return {
        "connection_id": connection_id,
        "t_real_end": protocol.t_real_end,
        "t_start": protocol.conn_traits.time_connect,
        # download: real bytes only (dummy excluded throughout)
        "bytes_rx_real": protocol.bytes_rx_real,
        "bytes_rx_at_real_end": protocol.bytes_rx_at_real_end,
        # upload: real bytes + total snapshot at real-traffic-end
        "bytes_tx_real": protocol.bytes_tx_real,
        "bytes_tx_at_real_end": protocol.bytes_tx_at_real_end,
    }


async def send_requests(
    server_ip,
    server_port,
    testcase,
    requests: dict,
    request_server_defenses: dict,
    defense_name="nop",
    defense_level: str = "mid1",
) -> dict:
    pending_connections = []
    for conn_id in requests:
        _log("create new connection ", conn_id)
        pending_connections.append(
            send_single_request(
                conn_id,
                server_ip,
                server_port,
                requests[conn_id],
                defense_name=defense_name,
                request_server_defense=request_server_defenses[conn_id],
                defense_level=defense_level,
            )
        )

    results = await asyncio.gather(*pending_connections)

    valid = [r for r in results if r["t_real_end"] is not None]
    if not valid:
        return {
            "t_real_end": None,
            "t_start": None,
            "latency": None,
            "bytes_rx_real": None,
            "bytes_rx_at_real_end": None,
            "bytes_tx_real": None,
            "bytes_tx_at_real_end": None,
            "defense": defense_name,
            "level": defense_level,
        }

    t_real_end = max(r["t_real_end"] for r in valid)

    total_bytes_rx_real = sum(r["bytes_rx_real"] for r in valid)
    total_bytes_rx_at_end = sum(r["bytes_rx_at_real_end"] or 0 for r in valid)

    total_bytes_tx_real = sum(r["bytes_tx_real"] for r in valid)
    total_bytes_tx_at_end = sum(r["bytes_tx_at_real_end"] or 0 for r in valid)

    bw_rx_overhead = (
        total_bytes_rx_at_end / total_bytes_rx_real - 1
        if total_bytes_rx_real > 0
        else None
    )
    bw_tx_overhead = (
        total_bytes_tx_at_end / total_bytes_tx_real - 1
        if total_bytes_tx_real > 0
        else None
    )

    t_start = min(r["t_start"] for r in valid)

    return {
        "t_real_end": t_real_end,
        "t_start": t_start,
        "latency": t_real_end - t_start,
        "bytes_rx_real": total_bytes_rx_real,
        "bytes_rx_at_real_end": total_bytes_rx_at_end,
        "bytes_tx_real": total_bytes_tx_real,
        "bytes_tx_at_real_end": total_bytes_tx_at_end,
        "bw_rx_overhead": bw_rx_overhead,
        "bw_tx_overhead": bw_tx_overhead,
        # carried into the CSV so calibrate.py never has to parse the scenario
        "defense": defense_name,
        "level": defense_level,
    }


def run_test_case(
    server_ip,
    server_port,
    testcase: str,
    requests: dict,
    request_server_defenses: dict,
    defense_name="nop",
    defense_level: str = "mid1",
) -> dict:
    return asyncio.run(
        send_requests(
            server_ip,
            server_port,
            testcase,
            requests=requests,
            request_server_defenses=request_server_defenses,
            defense_name=defense_name,
            defense_level=defense_level,
        )
    )
