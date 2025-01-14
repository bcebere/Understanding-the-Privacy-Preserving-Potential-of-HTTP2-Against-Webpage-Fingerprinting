# stdlib
from argparse import ArgumentParser
import asyncio
from copy import deepcopy
import io
import json
from pathlib import Path
import random
import ssl
import time
from typing import Dict, List

# third party
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
from scapy.config import conf
from scapy.sendrecv import AsyncSniffer
from scapy.utils import wrpcap
import sslkeylog
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("-dst_ip", "--dst_ip", dest="dst_ip", help="Destination IP")
parser.add_argument("-dst_port", "--dst_port", dest="dst_port", help="Destination Port")
parser.add_argument(
    "-repeats", "--repeats", dest="repeats", default=30, help="Subpage repeats"
)
parser.add_argument(
    "-ifname", "--ifname", dest="ifname", help="Interface to use for capture"
)
parser.add_argument(
    "-use_random_window",
    "--use_random_window",
    dest="use_random_window",
    default=False,
    help="Enable Control Flow Randomization",
)


args = parser.parse_args()

assert args.dst_ip is not None
assert args.dst_port is not None
assert args.ifname is not None

conf.route_autoload = False
conf.route6_autoload = False
conf.bufsize = 50 * 1024 * 1024  # 50 MB buffer size


DATA_PATH = Path("data")
TRACE_PATH = Path("traces")
OUTPUT_CSV = Path("output_csv_single")
startup_wait_sec = 0.2

sslkeylog.set_keylog("ssllogkey.log")
IFACE = args.ifname  # "lo"
SERVER_PORT = args.dst_port  # 8443
SERVER_IP = args.dst_ip  # "127.0.0.1"
REPEATS = int(args.repeats)
USE_RANDOM_WINDOW = bool(args.use_random_window)

TRACE_PATH.mkdir(parents=True, exist_ok=True)

with open(DATA_PATH / "client_db.json") as f:
    TESTCASES = json.load(f)


class AsyncEventWithTimeout(asyncio.Event):
    async def wait(self, timeout=None) -> bool:
        try:
            await asyncio.wait_for(super().wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return False
        return True


class H2Protocol(asyncio.Protocol):
    def __init__(self, requests: list) -> None:
        config = H2Configuration(client_side=True, header_encoding="utf-8")
        self.conn = H2Connection(config=config)
        self.transport = None
        self.requests = requests
        self.stream_data = {}
        self.pending_streams = set()
        self.exit_event = AsyncEventWithTimeout()

        self.loop = asyncio.get_event_loop()

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
            )

            try:
                self.conn.send_data(
                    stream_id, data[:chunk_size], end_stream=(chunk_size == len(data))
                )
            except (StreamClosedError, ProtocolError):
                # The stream got closed and we didn't get told. We're done
                # here.
                break

            self.transport.write(self.conn.data_to_send())
            data = data[chunk_size:]

    def send_request(self, path, body):
        data = json.dumps({}).encode("utf8")
        request_headers = [
            (":method", "GET"),
            (":scheme", "https"),
            (":path", path),
            (":authority", "localhost"),
            ("user-agent", "hyper-h2/1.0.0"),
            ("content-length", str(len(data))),
        ]
        next_stream = self.conn.get_next_available_stream_id()
        self.conn.send_headers(next_stream, request_headers)
        asyncio.ensure_future(self.send_data(next_stream, data))

        self.pending_streams.add(next_stream)

    def connection_made(self, transport):
        self.transport = transport
        self.conn.initiate_connection()
        if USE_RANDOM_WINDOW:
            # Random Window Size
            random_window_size = (
                random.randint(1, 32) * 128
            )  # Random size between 128 bytes and 4KB
            print(f"Setting random WINDOW_SIZE: {random_window_size}")
            self.conn.update_settings(
                {
                    SettingCodes.INITIAL_WINDOW_SIZE: random_window_size,
                    SettingCodes.MAX_CONCURRENT_STREAMS: 9999,
                }
            )
        self.transport.write(self.conn.data_to_send())
        self.loop.create_task(self.send_ping())

    def handle_next_request(self, custom_path=None):
        next_idx = 0

        if custom_path is not None:
            for idx, req in enumerate(self.requests):
                if req["path"] == custom_path:
                    next_idx = idx
                    break
        try:
            next_request = self.requests.pop(next_idx)
        except BaseException:
            return

        print("executing next req", next_idx, next_request)
        path = next_request["path"]
        body = next_request["data"]
        self.send_request(path, body)

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
                    self.handle_next_request()
                elif isinstance(event, DataReceived):
                    self.receive_data(event.data, event.stream_id)
                elif isinstance(event, StreamEnded):
                    self.log_data(event.stream_id)

                    if event.stream_id in self.pending_streams:
                        self.pending_streams.remove(event.stream_id)

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
                            print("next hinted path", hinted_path)
                            self.handle_next_request(custom_path=hinted_path)
                            take_hints -= 1
                            if take_hints <= 0:
                                break

                        except BaseException as e:
                            print("Early Hints failure", e)
                            raise
                            continue

        self.transport.write(self.conn.data_to_send())

    def log_push(self, headers, pid, sid):
        path = None
        for header in headers:
            if header[0] == ":path":
                path = header[1]
        index_to_remove = None
        for idx, pending_req in enumerate(self.requests):
            pending_path = pending_req["path"]
            if path == pending_path:
                index_to_remove = idx
                break

        print(" >> Server pushed", sid, path, idx)
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

        try:
            self.conn.increment_flow_control_window(len(data), stream_id=stream_id)
            self.conn.increment_flow_control_window(len(data))
            self.send_window_update(stream_id)
        except BaseException:
            pass

    def send_window_update(self, stream_id):
        window_size_increment = 1024 * 1024  # 1MB, for example
        self.conn.increment_flow_control_window(
            window_size_increment, stream_id=stream_id
        )
        self.conn.increment_flow_control_window(window_size_increment)
        self.transport.write(self.conn.data_to_send())

    def log_data(self, stream_id):
        data = self.stream_data[stream_id]
        data.seek(0)
        print(
            "   >>>> ",
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
        while self.exit_event.wait():
            await asyncio.sleep(0.1)  # Send a ping every 10 seconds
            ping_data = b"\x00" * 8  # 8 bytes of arbitrary data
            self.conn.ping(ping_data)
            self.transport.write(self.conn.data_to_send())


async def send_requests(testcase, requests):
    ctx = ssl.SSLContext()
    ctx.set_alpn_protocols(["h2"])
    ctx.check_hostname = False

    loop = asyncio.get_running_loop()

    protocol = H2Protocol(requests)
    coro = loop.create_connection(
        lambda: protocol, host=SERVER_IP, port=SERVER_PORT, ssl=ctx
    )
    transport, protocol = await coro

    try:
        await protocol.wait_for_exit()
    finally:
        protocol.stop()


def run_test_case(testcase: str, requests: List[Dict]):
    try:
        results = asyncio.run(send_requests(testcase, requests=requests))
    except BaseException as e:
        print("test case crash", e)
        return None
    return results


keys = list(TESTCASES.keys())
random.shuffle(keys)

for testcase in tqdm(keys):
    test_repeats = 1
    if "test_repeats" in TESTCASES[testcase]:
        test_repeats = TESTCASES[testcase]["test_repeats"]

    requests = TESTCASES[testcase]["requests"]
    for repeat in range(test_repeats):
        trace_file = TRACE_PATH / f"test_{testcase[1:]}_{repeat}.pcap"
        if trace_file.exists():
            print("already collected", trace_file)
            continue

        # PCAP Collection
        tracer = AsyncSniffer(
            iface=IFACE,
            # filter=BPF_FILTER,
        )
        tracer.start()
        for retry in range(10):
            if not hasattr(tracer, "stop_cb"):
                print(f"Tracer not ready yet {retry}")
                time.sleep(startup_wait_sec)
            else:
                break

        # Run the test
        run_test_case(testcase, deepcopy(requests))

        # Save trace
        network_trace = tracer.stop()
        with open(trace_file, "wb") as outfile:
            wrpcap(outfile, network_trace)

        del network_trace
        del tracer
