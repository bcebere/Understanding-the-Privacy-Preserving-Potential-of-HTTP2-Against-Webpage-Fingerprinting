# stdlib
import asyncio
from copy import deepcopy
import io
import json
import random
import ssl
import time
from typing import List, Optional

# third party
# DEFENSES
from defenses.front import FRONT_DEFENSE
from defenses.httpos import HTTPOS_DEFENSE
from defenses.llama import LLAMA_DEFENSE
from defenses.mod_all import CLMODS_DEFENSE
from defenses.nop import NOP_DEFENSE
from defenses.tamaraw import TAMARAW_DEFENSE
from defenses.tamaraw_qcsd import TAMARAW_QCSD_DEFENSE
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


def get_defense(defense: str):
    if defense == "nop":
        return NOP_DEFENSE
    # DEFENSES
    elif defense == "tamaraw":
        return deepcopy(TAMARAW_DEFENSE)
    elif defense == "tamaraw_qcsd":
        return deepcopy(TAMARAW_QCSD_DEFENSE)
    elif defense == "front":
        return deepcopy(FRONT_DEFENSE)
    elif defense == "httpos":
        return deepcopy(HTTPOS_DEFENSE)
    elif defense == "llama":
        return deepcopy(LLAMA_DEFENSE)
    # MODALITIES
    elif defense == "mod_all":
        return deepcopy(CLMODS_DEFENSE)
    else:
        raise NotImplementedError(defense)


class Request(BaseModel):
    path: str
    label: str
    data: dict = {}
    headers: dict = {}
    delay: float = 0
    expected_size: Optional[int] = None
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
    ) -> None:
        config = H2Configuration(client_side=True, header_encoding="utf-8")
        self.conn = H2Connection(config=config)
        self.transport = None
        self.requests = requests
        self.dummy_requests = deepcopy(requests)
        self.connection_id = connection_id
        self.server_hints = []

        if request_server_defense:
            defense_name = "mod_clping"

        defense = get_defense(defense_name)
        initial_window_size = defense.initial_window_size()
        self.window_size = initial_window_size

        self.conn_settings = {
            SettingCodes.INITIAL_WINDOW_SIZE: initial_window_size,
            SettingCodes.MAX_CONCURRENT_STREAMS: 9999,
        }
        self.defense = defense
        self.request_server_defense = request_server_defense

        self.max_dummy_time = 2  # seconds
        self.connection_start_time = time.time()

        self.padding_char = b"\x00"
        print(self.defense.summary())
        print(
            f"""
            [CONN] {self.connection_id} {time.time()}
            [DEFENSE] {self.conn_settings}
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
        self.conn_lock = asyncio.Lock()  # one lock per TCP connection

    def connection_made(self, transport):
        self.transport = transport
        self.conn.initiate_connection()
        self.conn.update_settings(self.conn_settings)
        self.transport.write(self.conn.data_to_send())
        self.loop.create_task(self.send_ping_in_loop())
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

            # Extract the data chunk and pad it to the fixed packet size
            chunk = data[:chunk_size]
            # if len(chunk) < packet_size:
            #    chunk = chunk.ljust(packet_size, self.padding_char)

            print(
                "   >>>> [SEND FRAME] ",
                stream_id,
                "-->",
                len(chunk),
            )
            try:
                self.conn.send_data(
                    stream_id, chunk, end_stream=(chunk_size == len(data))
                )
            except (StreamClosedError, ProtocolError):
                # The stream got closed and we didn't get told. We're done
                # here.
                break

            self.transport.write(self.conn.data_to_send())
            data = data[chunk_size:]

    async def send_request(self, base_request, is_noise=False):
        def _prepare_release_request_old(request, user_agent):
            data = json.dumps({}).encode("utf8")

            request.headers["label"] = request.label
            request.headers["connection_id"] = self.connection_id
            request.headers["defend_connection"] = str(int(self.request_server_defense))

            request_headers = [
                (":method", "GET"),
                (":scheme", "https"),
                (":path", request.path),
                (":authority", "localhost"),
                ("user-agent", user_agent),
                ("content-length", str(len(data))),
            ]
            for key in request.headers:
                request_headers.append((key.lower(), request.headers[key]))

            next_stream = self.conn.get_next_available_stream_id()
            print(f"STREAM = {self.connection_id}-{next_stream} PATH={request.path}")

            self.conn.send_headers(next_stream, request_headers)
            self.transport.write(self.conn.data_to_send())

            self.conn_traits.streams_start[next_stream] = time.time()
            self.conn_traits.streams_response_timestamps[next_stream] = []
            self.conn_traits.streams_response_sizes[next_stream] = request.expected_size

            self.pending_streams.add(next_stream)

            asyncio.ensure_future(self.send_data(next_stream, data))

            return (next_stream, request_headers, data)

        def _prepare_single_request(request, user_agent, data):
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
            print(f"STREAM = {self.connection_id}-{stream_id} PATH={request.path}")

            self.conn.send_headers(stream_id, request_headers)

            # Book-keeping
            now = time.time()
            self.conn_traits.streams_start[stream_id] = now
            self.conn_traits.streams_response_timestamps[stream_id] = []
            self.conn_traits.streams_response_sizes[stream_id] = request.expected_size
            self.pending_streams.add(stream_id)

            # Schedule the body send
            asyncio.create_task(self.send_data(stream_id, data))

            return (stream_id, request_headers, data)

        def _release_batch(requests):
            print(f"[DBG] Handle batch sz={len(requests)} noise={is_noise}")
            results = []
            data = json.dumps({}).encode("utf8")

            for req, user_agent in requests:
                result = _prepare_single_request(req, user_agent, data)
                results.append(result)

            self.transport.write(self.conn.data_to_send())
            return results

        def _get_dummies():
            if is_noise:
                return []

            if base_request.expected_size < 1000:
                return []

            print("[DBG] dummies for ", base_request.expected_size)
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
                print(
                    f" >>> [DEFENSE] Extra path = {dummy_request.path} {dummy_request.headers} size = {dummy_request.expected_size} user-agent = {user_agent}"
                )
            return noise

        pending_requests = _get_dummies()

        # Add random UA
        packet_size = self.defense.send_packet_size()  # Use max frame size for padding
        user_agent = self.defense.user_agent(is_noise=is_noise)
        print(f"[DEFENSE] user-agent = {user_agent}")
        if packet_size > len(base_request.path) + len(user_agent):
            print(f"[DEFENSE] Send packets with padding {packet_size} ")
            user_agent += "a" * (packet_size - len(base_request.path) - len(user_agent))

        # Handle main request
        if self.defense.use_ranged_requests():
            ranged_requests = self.defense.split_for_ranged_requests(base_request)
            if len(ranged_requests) > 0:
                print(f"[DEFENSE] Using ranged requests N = {len(ranged_requests)}")
                for req in ranged_requests:
                    pending_requests.append((req, user_agent))
            else:
                pending_requests.append((base_request, user_agent))
        else:
            pending_requests.append((base_request, user_agent))

        if is_noise:  # already handled noise
            return _release_batch(pending_requests)

        delay_req = self.defense.should_delay_request()
        if delay_req > 0:
            print("[DEFENSE] Delay request")
            await asyncio.sleep(delay_req)

        pending_requests.extend(_get_dummies())

        return _release_batch(pending_requests)

    async def handle_next_request(self, custom_path=None, is_noise: bool = False):
        if self.defense.should_shuffle() and len(self.conn_traits.streams_start) > 0:
            # print(f"[DEFENSE] Shuffle request order ")
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
            print(" ERR: data_received proto error", e)
            self.transport.write(self.conn.data_to_send())
            self.transport.close()
        else:
            self.transport.write(self.conn.data_to_send())
            for event in events:
                if isinstance(event, SettingsAcknowledged):
                    self.conn_traits.time_settings = time.time()
                    print(
                        f"[TRAITS] Roundtrip = {self.conn_traits.time_settings - self.conn_traits.time_connect} s"
                    )
                    # self.handle_next_request()
                    asyncio.create_task(self.handle_next_request())
                elif isinstance(event, DataReceived):
                    self.receive_data(event.data, event.stream_id)
                elif isinstance(event, StreamEnded):
                    self.log_data(event.stream_id)

                    if event.stream_id in self.pending_streams:
                        self.pending_streams.remove(event.stream_id)

                    if len(self.pending_streams) == 0:
                        # self.handle_next_request()
                        asyncio.create_task(self.handle_next_request())

                    if self.defense.should_batch():
                        print("[DEFENSE] Batch requests ")
                        batch_more_requests = random.randrange(0, 4)
                        for _ in range(batch_more_requests):
                            # self.handle_next_request()
                            asyncio.create_task(self.handle_next_request())

                    if len(self.pending_streams) == 0:
                        self.exit_event.set()  # Signal to exit

                elif isinstance(event, PushedStreamReceived):
                    print(
                        "[HTTP2][PUSH] Server pushed", self.connection_id, event.headers
                    )
                    self.log_push(
                        event.headers, event.parent_stream_id, event.pushed_stream_id
                    )
                elif isinstance(event, PingAckReceived):
                    # self.handle_next_request()
                    asyncio.create_task(self.handle_next_request())
                elif isinstance(event, PingReceived):
                    self.respond_ping(event)
                elif isinstance(event, InformationalResponseReceived):
                    # print(
                    #    "[HTTP2][HINTS103] Server hinted",
                    #    self.connection_id,
                    #    event.headers,
                    # )
                    self.server_hints = []

                    print("Received HINTS", len(event.headers))
                    for key, hint in event.headers:
                        if key != "link":
                            continue
                        try:
                            hinted_path = (
                                hint.split(";")[0].split("<")[-1].split(">")[0]
                            )
                            self.server_hints.append(hinted_path)
                        except BaseException as e:
                            print("Early Hints failure", e)
                            continue

                    if len(self.server_hints) == 0:
                        continue

                    print("HANDLE HINTS ASYNC")
                    asyncio.create_task(self.handle_hints_async(self.server_hints))

        self.transport.write(self.conn.data_to_send())

    async def handle_hints_async(self, hinted_paths):
        for hinted_path in hinted_paths:
            print(f"[HTTP2][HINTS103][{self.connection_id}] Handle {hinted_path}")
            await self.handle_next_request(custom_path=hinted_path)
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

        print(" >> [RECV PUSH]", stream_id, path, idx)

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

        print(
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
            print(f"[DEFENSE] Recv interval {recv_interval} ")

            async def _handle_recv_delay(recv_interval):
                start_time = time.time()
                await asyncio.sleep(recv_interval)
                try:
                    self.send_window_update(stream_id)
                    print(
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
        window_size_increment = self.window_size  # 1MB, for example
        self.conn.increment_flow_control_window(
            window_size_increment, stream_id=stream_id
        )
        self.conn.increment_flow_control_window(window_size_increment)
        self.transport.write(self.conn.data_to_send())

    def log_data(self, stream_id):
        data = self.stream_data[stream_id]
        data.seek(0)
        print(
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
            print(
                f"[TIMEOUT] Connection {self.connection_id} timed out after {timeout} seconds."
            )
            self.stop()
        # await self.exit_event.wait()

    def stop(self):
        self.transport.close()
        self.exit_event.set()

    def send_ping(self):
        # print("SEND PING")
        ping_data = b"\x00" * 8  # 8 bytes of arbitrary data
        self.conn.ping(ping_data)
        self.transport.write(self.conn.data_to_send())

    def respond_ping(self, event):
        # self.conn.acknowledge_ping(event.ping_data)
        # self.conn.ping(event.ping_data)
        # self.transport.write(self.conn.data_to_send())
        pass

    async def send_ping_in_loop(self):
        while not self.exit_event.is_set():
            should_send_pings = self.defense.should_send_random_pings()
            if should_send_pings > 0:
                print(
                    f"[DEFENSE][{self.connection_id}] Send {should_send_pings} noise PINGS"
                )
                for pingid in range(should_send_pings):
                    self.send_ping()
                    timeout = random.uniform(0, 0.01)
                    await asyncio.sleep(timeout)  # Send a ping every 10 ms
            else:
                await asyncio.sleep(0.5)  # Send a ping every 500 ms
                self.send_ping()

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
                print(
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
):
    ctx = ssl.SSLContext()
    ctx.set_alpn_protocols(["h2"])
    ctx.check_hostname = False

    loop = asyncio.get_running_loop()

    protocol = Client(
        connection_id=connection_id,
        requests=requests,
        defense_name=defense_name,
        request_server_defense=request_server_defense,
    )
    coro = loop.create_connection(
        lambda: protocol, host=server_ip, port=server_port, ssl=ctx
    )
    transport, protocol = await coro

    try:
        await protocol.wait_for_exit(timeout=timeout)
    finally:
        protocol.stop()


async def send_requests(
    server_ip,
    server_port,
    testcase,
    requests: dict,
    request_server_defenses: dict,
    defense_name="nop",
):
    pending_connections = []
    for conn_id in requests:
        print("create new connection ", conn_id)

        pending_connections.append(
            send_single_request(
                conn_id,
                server_ip,
                server_port,
                requests[conn_id],
                defense_name=defense_name,
                request_server_defense=request_server_defenses[conn_id],
            )
        )

    await asyncio.gather(*pending_connections)


def run_test_case(
    server_ip,
    server_port,
    testcase: str,
    requests: dict,
    request_server_defenses: dict,
    defense_name="nop",
):
    results = asyncio.run(
        send_requests(
            server_ip,
            server_port,
            testcase,
            requests=requests,
            request_server_defenses=request_server_defenses,
            defense_name=defense_name,
        )
    )
    return results
