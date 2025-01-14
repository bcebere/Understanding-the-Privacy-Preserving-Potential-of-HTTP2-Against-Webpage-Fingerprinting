# stdlib
import asyncio
from copy import deepcopy
import io
import json
import random
import ssl
import time
from typing import Dict, List, Optional

# third party
from defenses.adaptive import ADAPTIVE_DEFENSE
from defenses.front import FRONT_DEFENSE
from defenses.httpos import HTTPOS_DEFENSE

# DEFENSES
from defenses.nop import NOP_DEFENSE
from defenses.tamaraw import TAMARAW_DEFENSE
from defenses.wtfpad import WTFPAD_DEFENSE
from h2.config import H2Configuration
from h2.connection import H2Connection
from h2.errors import ErrorCodes
from h2.events import (
    DataReceived,
    InformationalResponseReceived,
    PingAckReceived,
    PushedStreamReceived,
    SettingsAcknowledged,
    StreamEnded,
)
from h2.exceptions import ProtocolError, StreamClosedError
from h2.settings import SettingCodes
from pydantic import BaseModel

# abc


def get_defense(defense: str):
    if defense == "nop":
        return NOP_DEFENSE
    elif defense == "tamaraw":
        return TAMARAW_DEFENSE
    elif defense == "wtfpad":
        return WTFPAD_DEFENSE
    elif defense == "front":
        return FRONT_DEFENSE
    elif defense == "httpos":
        return HTTPOS_DEFENSE
    elif defense == "adaptive":
        return ADAPTIVE_DEFENSE
    else:
        raise NotImplementedError(defense)


class Request(BaseModel):
    path: str
    data: dict = {}
    headers: dict = {}
    delay: float = 0
    expected_size: Optional[int] = None


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
        requests: List[Request],
        defense_name="nop",
    ) -> None:
        config = H2Configuration(client_side=True, header_encoding="utf-8")
        self.conn = H2Connection(config=config)
        self.transport = None
        self.requests = requests
        self.dummy_requests = deepcopy(requests)

        defense = get_defense(defense_name)
        initial_window_size = defense.initial_window_size()
        self.window_size = initial_window_size

        self.conn_settings = {
            SettingCodes.INITIAL_WINDOW_SIZE: initial_window_size,
            SettingCodes.MAX_CONCURRENT_STREAMS: 9999,
        }
        self.defense = defense
        self.max_dummy_time = 1  # one second
        self.connection_start_time = time.time()

        self.padding_char = b"\x00"
        print(self.defense.summary())
        print(
            f"""
            [Connection config] {self.conn_settings}
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

    def connection_made(self, transport):
        self.transport = transport
        self.conn.initiate_connection()
        self.conn.update_settings(self.conn_settings)
        self.transport.write(self.conn.data_to_send())
        self.loop.create_task(self.send_ping())
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

    def send_request(self, base_request, is_noise=False):
        def _handle_request(request, user_agent):
            data = json.dumps({}).encode("utf8")
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

            self.conn_traits.streams_start[next_stream] = time.time()
            self.conn_traits.streams_response_timestamps[next_stream] = []
            self.conn_traits.streams_response_sizes[next_stream] = request.expected_size

            self.conn.send_headers(next_stream, request_headers)
            asyncio.ensure_future(self.send_data(next_stream, data))

            self.pending_streams.add(next_stream)

            return next_stream

        packet_size = self.defense.send_packet_size()  # Use max frame size for padding
        user_agent = self.defense.user_agent()
        if packet_size > len(base_request.path) + len(user_agent):
            print(f"[DEFENSE] Send packets with padding {packet_size} ")
            user_agent += "a" * (packet_size - len(base_request.path) - len(user_agent))

        # Handle main request
        if self.defense.use_ranged_requests():
            ranged_requests = self.defense.split_for_ranged_requests(base_request)
            print(f"[DEFENSE] Using ranged requests N = {len(ranged_requests)}")
            for req in ranged_requests:
                _handle_request(req, user_agent)
            return

        stream_id = _handle_request(base_request, user_agent)

        if is_noise:  # already handled noise
            return

        # Adjust padding with dummy requests
        print(
            f" >>> [REQ] path = {base_request.path} stream = {stream_id} window = {self.window_size} HEAD = {base_request.expected_size}"
        )
        dummy_requests = self.defense.send_dummy_packet(
            self.dummy_requests,
            previous_request=base_request,
            window_size=self.window_size,
            stream_stats=self.conn_traits.stats(),
        )

        if dummy_requests is None:
            return

        for dummy_request in dummy_requests:
            stream_id = _handle_request(dummy_request, user_agent)
            print(
                f" >>> [DEFENSE] Extra path = {dummy_request.path} stream = {stream_id} {dummy_request.headers} size = {dummy_request.expected_size}"
            )
            user_agent = self.defense.user_agent()

    def handle_next_request(self, custom_path=None, is_noise: bool = False):
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
                }
            )
        else:
            if next_idx is None:
                next_idx = 0
            try:
                next_request = self.requests.pop(next_idx)
            except BaseException:
                return

        self.send_request(next_request, is_noise=is_noise)

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
                    self.handle_next_request()
                elif isinstance(event, DataReceived):
                    self.receive_data(event.data, event.stream_id)
                elif isinstance(event, StreamEnded):
                    self.log_data(event.stream_id)

                    if event.stream_id in self.pending_streams:
                        self.pending_streams.remove(event.stream_id)

                    if len(self.pending_streams) == 0:
                        self.handle_next_request()

                    if self.defense.should_batch():
                        print("[DEFENSE] Batch requests ")
                        batch_more_requests = random.randrange(0, 4)
                        for _ in range(batch_more_requests):
                            self.handle_next_request()

                    if len(self.pending_streams) == 0:
                        self.exit_event.set()  # Signal to exit

                elif isinstance(event, PushedStreamReceived):
                    self.log_push(
                        event.headers, event.parent_stream_id, event.pushed_stream_id
                    )
                elif isinstance(event, PingAckReceived):
                    self.handle_next_request()
                elif isinstance(event, InformationalResponseReceived):
                    take_hints = random.randrange(len(event.headers))
                    random.shuffle(event.headers)
                    for key, hint in event.headers:
                        if key != "link":
                            continue
                        try:
                            hinted_path = (
                                hint.split(";")[0].split("<")[-1].split(">")[0]
                            )
                            self.handle_next_request(custom_path=hinted_path)
                            take_hints -= 1
                            if take_hints <= 0:
                                break

                        except BaseException as e:
                            print("Early Hints failure", e)
                            continue

        self.transport.write(self.conn.data_to_send())

    def log_push(self, headers, pid, sid):
        path = None
        for header in headers:
            if header[0] == ":path":
                path = header[1]
        index_to_remove = None
        if len(self.requests) == 0:
            return

        for idx, pending_req in enumerate(self.requests):
            pending_path = pending_req.path
            if path == pending_path:
                index_to_remove = idx
                break

        print(" >> [RECV PUSH]", sid, path, idx)
        if index_to_remove is not None:
            del self.requests[index_to_remove]

        self.pending_streams.add(sid)

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
            "   >>>> [RECV FRAME] ", stream_id, "-->", len(data), self.pending_streams
        )
        self.conn_traits.streams_response_timestamps[stream_id].append(time.time())

        try:
            self.conn.increment_flow_control_window(
                self.window_size, stream_id=stream_id
            )
            self.conn.increment_flow_control_window(self.window_size)
            self.send_window_update(stream_id)
        except BaseException:
            pass

        send_interval = self.defense.send_interval(
            stream_stats=self.conn_traits.stats()
        )
        if (time.time() - self.connection_start_time) > self.max_dummy_time:
            send_interval = 0
        if send_interval > 0:
            print(f"[DEFENSE] Send interval {send_interval} ")
            time.sleep(send_interval)

        # print(self.conn_traits.stats())

    def send_window_update(self, stream_id):
        window_size_increment = 128  # self.window_size # 1MB, for example
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
            stream_id,
            "--> data len",
            len(data.read().decode("UTF-8")),
            self.pending_streams,
        )

    async def wait_for_exit(self):
        await self.exit_event.wait()

    def stop(self):
        self.transport.close()
        self.exit_event.set()

    async def send_ping(self):
        while not self.exit_event.is_set():
            await asyncio.sleep(0.01)  # Send a ping every 10 seconds
            ping_data = b"\x00" * 8  # 8 bytes of arbitrary data
            self.conn.ping(ping_data)
            self.transport.write(self.conn.data_to_send())

    async def send_dummy_traffic(self):
        while not self.exit_event.is_set():
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

            send_interval = self.defense.send_interval()
            await asyncio.sleep(send_interval)

            for dummy_request in dummy_requests:
                print(
                    f"[DEFENSE] Send dummy traffic {send_interval} {dummy_request.path}"
                )
                self.send_request(dummy_request, is_noise=True)


async def send_requests(
    server_ip,
    server_port,
    testcase,
    requests,
    defense_name="nop",
):
    ctx = ssl.SSLContext()
    ctx.set_alpn_protocols(["h2"])
    ctx.check_hostname = False

    loop = asyncio.get_running_loop()

    protocol = Client(requests, defense_name=defense_name)
    coro = loop.create_connection(
        lambda: protocol, host=server_ip, port=server_port, ssl=ctx
    )
    transport, protocol = await coro

    try:
        await protocol.wait_for_exit()
    finally:
        protocol.stop()


def run_test_case(
    server_ip,
    server_port,
    testcase: str,
    requests: List[Dict],
    defense_name="nop",
):
    results = asyncio.run(
        send_requests(
            server_ip,
            server_port,
            testcase,
            requests=requests,
            defense_name=defense_name,
        )
    )
    return results
