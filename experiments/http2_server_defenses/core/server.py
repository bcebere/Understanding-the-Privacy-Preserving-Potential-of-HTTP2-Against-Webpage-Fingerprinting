# stdlib
from argparse import ArgumentParser
import asyncio
import collections
import copy
import io
import json
from pathlib import Path
import random
import ssl
import string
import time
from typing import List, Tuple

# third party
import h2
from h2.config import H2Configuration
from h2.connection import H2Connection
from h2.errors import ErrorCodes
from h2.events import (
    ConnectionTerminated,
    DataReceived,
    PingReceived,
    RemoteSettingsChanged,
    RequestReceived,
    StreamEnded,
    StreamReset,
    WindowUpdated,
)
from h2.exceptions import ProtocolError, StreamClosedError
from h2.settings import SettingCodes
import numpy as np

parser = ArgumentParser()
parser.add_argument("-dst_port", "--dst_port", dest="dst_port", help="Destination Port")
parser.add_argument(
    "-defense_batch",
    "--defense_batch",
    dest="defense_batch",
    help="Activate batching defense",
    default=0,
)
parser.add_argument(
    "-defense_inject",
    "--defense_inject",
    dest="defense_inject",
    help="Activate inject defense",
    default=0,
)
parser.add_argument(
    "-defense_delay_random",
    "--defense_delay_random",
    dest="defense_delay_random",
    help="Activate delay defense at random. Used for WTFPAD",
    default=0,
)
parser.add_argument(
    "-defense_delay_constant",
    "--defense_delay_constant",
    dest="defense_delay_constant",
    help="Activate delay defense with constant value. Used for Tamaraw",
    default=0,
)
parser.add_argument(
    "-defense_pad_random",
    "--defense_pad_random",
    dest="defense_pad_random",
    help="Activate pad defense at random. used for WTFPAD",
    default=0,
)
parser.add_argument(
    "-defense_pad_constant",
    "--defense_pad_constant",
    dest="defense_pad_constant",
    help="Activate pad defense with constant value. used for Tamaraw",
    default=0,
)

parser.add_argument(
    "-defense_morph",
    "--defense_morph",
    dest="defense_morph",
    help="Activate traffic morphing.",
    default=0,
)

parser.add_argument(
    "-defense_adaptive",
    "--defense_adaptive",
    dest="defense_adaptive",
    help="Activate adaptive defense.",
    default=0,
)

parser.add_argument(
    "-defense_adaptive_hints",
    "--defense_adaptive_hints",
    dest="defense_adaptive_hints",
    help="Activate adaptive defense with early padding hints.",
    default=0,
)
args = parser.parse_args()

assert args.dst_port is not None

SERVER_IP = "0.0.0.0"
SERVER_PORT = args.dst_port  # 8443
DATA_PATH = Path("data")
SERVER_PATH = Path(__file__).resolve().parent


DEFENSE_BATCH = int(args.defense_batch)  # int(os.environ.get("SRV_DEFENSE_BATCH", 0))
DEFENSE_INJECT = int(
    args.defense_inject
)  # int(os.environ.get("SRV_DEFENSE_INJECT", 0))
DEFENSE_DELAY_RANDOM = int(args.defense_delay_random)
DEFENSE_DELAY_CONSTANT = int(args.defense_delay_constant)
DEFENSE_PAD_RANDOM = int(args.defense_pad_random)
DEFENSE_PAD_CONSTANT = int(args.defense_pad_constant)
DEFENSE_MORPH = int(args.defense_morph)
DEFENSE_ADAPTIVE = int(args.defense_adaptive)
DEFENSE_ADAPTIVE_HINTS = int(args.defense_adaptive_hints)

RequestData = collections.namedtuple("RequestData", ["headers", "data"])

with open(DATA_PATH / "server_db.json") as f:
    SERVER_DB = json.load(f)
with open(DATA_PATH / "client_db.json") as f:
    CLIENT_DB = json.load(f)

print(
    f"""
      Defense config:
        Defense 1 Response batching: {DEFENSE_BATCH}
        Defense 2 Response injection: {DEFENSE_INJECT}
        Defense 3 Response delay random: {DEFENSE_DELAY_RANDOM}
        Defense 4 Response delay constant: {DEFENSE_DELAY_CONSTANT}
        Defense 5 Response padding random: {DEFENSE_PAD_RANDOM}
        Defense 6 Response padding constant: {DEFENSE_PAD_CONSTANT}
        Defense 7 Response morphing: {DEFENSE_MORPH}
        Defense 8 Adaptive defense: {DEFENSE_ADAPTIVE}
        Defense 9 Adaptive Hints defense: {DEFENSE_ADAPTIVE_HINTS}
"""
)


def _add_injection_hints(database, low=100, high=5000, step=517):
    hints = []
    for padding_size in range(low, high, step):
        key = f"inject_{padding_size}B"
        database[key] = {
            "data_delay": 0,
            "data_size": padding_size,
        }
        hints.append(
            {
                "path": key,
            }
        )

    return hints, database


def _generate_noise(data: list, epsilon=0.5, delta=1e-5, use_gaussian=False):
    # Calculate the running mean
    cumulative_sum = np.sum(data)
    mean_N = cumulative_sum / len(data)

    # Calculate sensitivity based on the deviation from the mean
    sensitivity = abs(data[-1] - mean_N)

    if use_gaussian:
        # Gaussian noise
        sigma = (sensitivity / epsilon) * np.sqrt(2 * np.log(1.25 / delta))
        noise = np.random.normal(0, sigma)
    else:
        # Laplace noise
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale)

    # Ensure noise results in a non-negative delay
    return max(0, noise)


class H2Protocol(asyncio.Protocol):
    def __init__(self):
        config = H2Configuration(client_side=False, header_encoding="utf-8")
        self.conn = H2Connection(config=config)
        self.transport = None
        self.stream_data = {}
        self.flow_control_futures = {}
        self.pending_streams = []
        self.pending_requests_cnt = 99999
        self.pending_streams_flushed = False
        self.table_size = 4096
        self.server_name = "http2-mock;"
        self.server_db = SERVER_DB

        self.response_start = {}
        self.response_done = {}

        # constants
        self.delay_constant = None
        self.pad_constant = None

        if DEFENSE_INJECT:
            self.inject_hints, self.server_db = _add_injection_hints(self.server_db)

        if DEFENSE_ADAPTIVE:
            self.pad_constant = random.randint(128, 1024)
            print(f"[DEFENSE ADAPTIVE] pad_constant = {self.pad_constant}. ")

        if DEFENSE_ADAPTIVE_HINTS:
            self.inject_hints, self.server_db = _add_injection_hints(
                self.server_db,
                low=4 * self.pad_constant,
                high=8 * self.pad_constant,
                step=self.pad_constant,
            )
            print(f"[DEFENSE ADAPTIVE_HINTS] hints = {len(self.inject_hints)}")
        # print("Headers", self.table_size, self.server_name)

    def connection_made(self, transport: asyncio.Transport):
        self.transport = transport
        # self.conn.update_settings({
        #    SettingCodes.MAX_CONCURRENT_STREAMS: 9999
        # })
        self.conn.initiate_connection()
        self.conn.update_settings(
            {h2.settings.SettingCodes.HEADER_TABLE_SIZE: self.table_size}
        )
        self.transport.write(self.conn.data_to_send())

    def connection_lost(self, exc):
        for future in self.flow_control_futures.values():
            future.cancel()
        self.flow_control_futures = {}

    def data_received(self, data: bytes):
        try:
            events = self.conn.receive_data(data)
        except ProtocolError:
            self.transport.write(self.conn.data_to_send())
            self.transport.close()
        else:
            self.transport.write(self.conn.data_to_send())
            for event in events:
                if isinstance(event, RequestReceived):
                    self.request_received(event.headers, event.stream_id)
                elif isinstance(event, DataReceived):
                    self.receive_data(event.data, event.stream_id)
                elif isinstance(event, StreamEnded):
                    if DEFENSE_BATCH:
                        self.pending_streams.append(event.stream_id)
                        self._check_handle_multiplexing()
                    else:
                        self.stream_complete(event.stream_id)

                elif isinstance(event, ConnectionTerminated):
                    self.transport.close()
                elif isinstance(event, StreamReset):
                    self.stream_reset(event.stream_id)
                elif isinstance(event, WindowUpdated):
                    self.window_updated(event.stream_id, event.delta)
                elif isinstance(event, RemoteSettingsChanged):
                    if SettingCodes.INITIAL_WINDOW_SIZE in event.changed_settings:
                        self.window_updated(None, 0)
                elif isinstance(event, PingReceived):
                    self.handle_ping(event)

                self.transport.write(self.conn.data_to_send())

    def handle_ping(self, event):
        self.conn.ping(event.ping_data)
        self.transport.write(self.conn.data_to_send())

    def request_received(self, headers: List[Tuple[str, str]], stream_id: int):
        self.response_start[stream_id] = time.time()
        headers = collections.OrderedDict(headers)

        # Store off the request data.
        request_data = RequestData(headers, io.BytesIO())
        self.stream_data[stream_id] = request_data

        path = headers[":path"]
        if stream_id == 1:
            self._check_prepare_multiplexing(stream_id, path)
            if DEFENSE_INJECT or DEFENSE_ADAPTIVE_HINTS:
                self._generate_103_early_hints(stream_id, path)

    def _generate_response_data(self, path: str, headers={}):
        if path not in self.server_db:
            print("missing path", path)
            return {}

        response_size = 0
        if "range" in headers:
            range_bytes = headers["range"].split("bytes=")[-1]
            start, end = range_bytes.split("-")
            response_size = int(end) - int(start)

        db_data = self.server_db[path]
        timeout = 0
        if "data_delay" in db_data:
            if isinstance(db_data["data_delay"], (int, float)):
                timeout = db_data["data_delay"]
            elif db_data["data_delay"] == "random":
                timeout = random.uniform(0.001, 0.1)
            else:
                timeout = 0
        if DEFENSE_DELAY_RANDOM:
            # Insert additional small delay
            add_delay = random.uniform(0, 0.1)
            timeout += add_delay
            print(f"[DEFENSE_DELAY_RANDOM] : insert rnd delay {add_delay}")
        elif DEFENSE_DELAY_CONSTANT:
            if self.delay_constant is None:
                self.delay_constant = random.uniform(0, 0.1)
            add_delay = self.delay_constant
            timeout += add_delay
            print(f"[DEFENSE_DELAY_CONSTANT] : insert constant delay {add_delay}")
        elif DEFENSE_ADAPTIVE:
            should_delay = random.choice([True, False])
            if should_delay:
                latencies = []
                for prev_stream in self.response_done:
                    latencies.append(
                        self.response_done[prev_stream]
                        - self.response_start[prev_stream]
                    )
                if len(latencies) > 0:
                    timeout = _generate_noise(latencies)
                    print(f"[DEFENSE_ADAPTIVE] : insert delay {timeout}")

        body = {}
        if response_size > 0:
            body = "".join(
                random.choices(string.ascii_uppercase + string.digits, k=response_size)
            )
        elif "data" in db_data:
            raise NotImplementedError("data support")
            # body = db_data["data"]
        elif "data_size" in db_data:  # generate random data of len
            if db_data["data_size"] == "random":
                response_size = random.randint(100, 10000)
            elif isinstance(db_data["data_size"], (int, float)):
                response_size = int(db_data["data_size"])
            else:
                response_size = 1

            body = "".join(
                random.choices(string.ascii_uppercase + string.digits, k=response_size)
            )
        else:
            body = ""
        if DEFENSE_PAD_RANDOM:
            pad_size = 2 ** random.randint(4, 13)
            pad_size = (int(len(body) / pad_size) + 1) * pad_size
            padding = "".join(
                random.choices(
                    string.ascii_uppercase + string.digits, k=pad_size - len(body)
                )
            )
            old_size = len(body)
            body += padding
            print(
                f"[DEFENSE_PAD_RANDOM] adding costant response padding = {pad_size} body len = {len(body)} old body size = {old_size}"
            )
        elif DEFENSE_PAD_CONSTANT:
            if self.pad_constant is None:
                self.pad_constant = 2 ** random.randint(9, 13)

            pad_size = (int(len(body) / self.pad_constant) + 1) * self.pad_constant
            padding = "".join(
                random.choices(
                    string.ascii_uppercase + string.digits, k=pad_size - len(body)
                )
            )
            old_size = len(body)
            body += padding
            print(
                f"[DEFENSE_PAD_CONSTANT] adding costant response padding = {self.pad_constant} body len = {len(body)} old body size = {old_size}"
            )
        elif DEFENSE_MORPH:
            old_size = len(body)
            morph_path = random.choice(list(self.server_db.keys()))
            morph_data = self.server_db[morph_path]
            if morph_data["data_size"] == "random":
                morph_size = random.randint(old_size, 2 * old_size)
            elif isinstance(morph_data["data_size"], (int, float)):
                morph_size = int(morph_data["data_size"])
            else:
                morph_size = len(body)

            morph_size = (int(len(body) / morph_size) + 1) * morph_size
            padding = "".join(
                random.choices(
                    string.ascii_uppercase + string.digits, k=morph_size - len(body)
                )
            )
            body += padding
            print(
                f"[DEFENSE_MORPH] morphing {path} in {morph_path}. morph size = {morph_size}. body len = {len(body)} old body size = {old_size}"
            )
        elif DEFENSE_ADAPTIVE:
            should_pad = random.choice([True, False])
            if should_pad and len(body) % self.pad_constant != 0:
                pad_size = (int(len(body) / self.pad_constant) + 1) * self.pad_constant
                padding = "".join(
                    random.choices(
                        string.ascii_uppercase + string.digits, k=pad_size - len(body)
                    )
                )
                old_size = len(body)
                body += padding
                print(
                    f"[DEFENSE_ADAPTIVE] adding response padding = {self.pad_constant} body len = {len(body)} old body size = {old_size}"
                )

        print(f"[DATA] path={path} timeout={timeout} size={len(body)}")

        return body, timeout

    def _prepare_multiplexing_strategy(self):
        if DEFENSE_BATCH:
            strategies = [
                {"ckp_pct": [0.1, 0.3, 1]},
                {"ckp_pct": [0.2, 0.4, 1]},
                {"ckp_pct": [0.1, 0.5, 1]},
                {"ckp_pct": [0.2, 0.6, 1]},
                {"ckp_pct": [0.1, 0.7, 1]},
                {"ckp_pct": [0.2, 0.8, 1]},
                {"ckp_pct": [0.1, 0.3, 0.5, 1]},
                {"ckp_pct": [0.2, 0.4, 0.6, 1]},
                {"ckp_pct": [0.1, 0.3, 0.7, 1]},
                {"ckp_pct": [0.2, 0.4, 0.8, 1]},
                {"ckp_pct": [0.1, 0.3, 0.5, 0.7, 1]},
                {"ckp_pct": [0.2, 0.4, 0.6, 0.8, 1]},
                {"ckp_pct": "all"},
            ]
        else:
            raise RuntimeError()

        selected_idx = random.randint(0, len(strategies) - 1)
        selected = copy.deepcopy(strategies[selected_idx])

        selected["progress"] = []

        return selected

    def _check_prepare_multiplexing(self, parent_stream_id: int, prev_path: str):
        if not DEFENSE_BATCH:
            return

        self.pending_requests_cnt = len(CLIENT_DB[prev_path]["requests"])
        mult_strategy = self._prepare_multiplexing_strategy()
        if mult_strategy["ckp_pct"] == "all":
            mult_ckp = list(range(self.pending_requests_cnt + 1))
        else:
            mult_ckp = []
            for pct in mult_strategy["ckp_pct"]:
                mult_ckp.append(int(pct * self.pending_requests_cnt))

        mult_ckp = np.asarray(mult_ckp)
        mult_ckp = mult_ckp[mult_ckp != 0]
        mult_ckp = list(set(mult_ckp))
        mult_ckp.sort()
        mult_strategy["checkpoints"] = mult_ckp
        self.multiplexing_strategy = mult_strategy
        print("[DEFENSE_BATCH]", self.multiplexing_strategy)

    def _flush_pending_streams(self, flush_all=True):
        pending_cnt = len(self.pending_streams)
        for pidx in range(pending_cnt):
            try:
                stream_id = self.pending_streams.pop(0)
            except BaseException:
                break

            self.stream_complete(stream_id)

    def _check_handle_multiplexing(self):
        last_ckp_idx = len(self.multiplexing_strategy["progress"])
        ckp_already_flushed = 0
        if last_ckp_idx > 0:
            ckp_already_flushed = self.multiplexing_strategy["progress"][-1]
        current_avail = ckp_already_flushed + len(self.pending_streams)

        if last_ckp_idx >= len(self.multiplexing_strategy["checkpoints"]):
            return self._flush_pending_streams()

        assert (
            self.multiplexing_strategy["checkpoints"][last_ckp_idx]
            <= self.pending_requests_cnt
        )
        next_ckp = self.multiplexing_strategy["checkpoints"][last_ckp_idx]

        if current_avail != next_ckp:
            return
            # self._flush_pending_streams(flush_all = True)

        self.multiplexing_strategy["progress"].append(next_ckp)
        random.shuffle(self.pending_streams)
        return self._flush_pending_streams()

    def _generate_head_response(self, stream_id, path):
        headers = [
            (":status", "200"),
            ("content-type", "text/plain"),
            ("content-length", "0"),  # No body content for HEAD
        ]
        if path not in self.server_db:
            return headers

        db_data = self.server_db[path]
        if db_data["data_size"] == "random":
            return headers

        headers = [
            (":status", "200"),
            ("content-type", "text/plain"),
            ("content-length", f"{db_data['data_size']}"),  # No body content for HEAD
        ]
        return headers

    def _generate_103_early_hints(self, stream_id, prev_path: str):
        pending_requests = self.inject_hints
        hints_headers = [
            (":status", "103"),
        ]
        for idx, req in enumerate(pending_requests):
            req_path = req["path"]
            hints_headers.append(("link", f"<{req_path}>; rel=preload; as=image"))

        print(f"[DEFENSE_INJECT] potential noise requests = {len(pending_requests)}")
        self.conn.send_headers(stream_id, hints_headers)

    def stream_complete(self, stream_id: int):
        """
        When a stream is complete, we can send our response.
        """
        try:
            request_data = self.stream_data[stream_id]
        except KeyError:
            # Just return, we probably 405'd this already
            return

        headers = request_data.headers
        path = headers[":path"]

        body, delay = self._generate_response_data(path, headers)

        req_body = request_data.data.getvalue().decode("utf-8")
        print(f"request stream_id = {stream_id} path = {path} req_body = {req_body}")

        data = json.dumps(
            {
                # "headers": headers,
                "body": body
            },
            indent=4,
        ).encode("utf8")

        response_headers = (
            (":status", "200"),
            ("content-type", " application/json"),
            ("content-length", str(len(data))),
            ("server", self.server_name),
        )
        self.conn.send_headers(stream_id, response_headers)

        asyncio.ensure_future(self.send_data(data, stream_id, delay))

    def receive_data(self, data: bytes, stream_id: int):
        """
        We've received some data on a stream. If that stream is one we're
        expecting data on, save it off. Otherwise, reset the stream.
        """
        try:
            stream_data = self.stream_data[stream_id]
        except KeyError:
            self.conn.reset_stream(stream_id, error_code=ErrorCodes.PROTOCOL_ERROR)
        else:
            stream_data.data.write(data)

    def stream_reset(self, stream_id):
        """
        A stream reset was sent. Stop sending data.
        """
        if stream_id in self.flow_control_futures:
            future = self.flow_control_futures.pop(stream_id)
            future.cancel()

    async def send_data(self, data, stream_id, delay):
        """
        Send data according to the flow control rules.
        """
        if delay > 0:
            await asyncio.sleep(delay)

        while data:
            while self.conn.local_flow_control_window(stream_id) < 1:
                # print(
                #    "waiting flow control",
                #    stream_id,
                #    self.conn.local_flow_control_window(stream_id),
                # )
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
                print("stream got closed", stream_id)
                # The stream got closed and we didn't get told. We're done
                # here.
                break

            self.transport.write(self.conn.data_to_send())
            data = data[chunk_size:]

        self.response_done[stream_id] = time.time()

    async def wait_for_flow_control(self, stream_id):
        """
        Waits for a Future that fires when the flow control window is opened.
        """
        f = asyncio.Future()
        self.flow_control_futures[stream_id] = f
        await f

    def window_updated(self, stream_id, delta):
        """
        A window update frame was received. Unblock some number of flow control
        Futures.
        """
        if stream_id and stream_id in self.flow_control_futures:
            f = self.flow_control_futures.pop(stream_id)
            f.set_result(delta)
        elif not stream_id:
            for f in self.flow_control_futures.values():
                f.set_result(delta)

            self.flow_control_futures = {}


ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
ssl_context.options |= ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1 | ssl.OP_NO_TLSv1_2
ssl_context.load_cert_chain(
    certfile=SERVER_PATH / "cert.pem", keyfile=SERVER_PATH / "key.pem"
)
ssl_context.set_alpn_protocols(["h2"])

loop = asyncio.get_event_loop()
# Each client connection will create a new protocol instance
coro = loop.create_server(H2Protocol, SERVER_IP, SERVER_PORT, ssl=ssl_context)
# coro = loop.create_server(H2Protocol, "127.0.0.1", 8443)
server = loop.run_until_complete(coro)

# Serve requests until Ctrl+C is pressed
print(f"Serving on {server.sockets[0].getsockname()}")
try:
    loop.run_forever()
except KeyboardInterrupt:
    pass

# Close the server
server.close()
loop.run_until_complete(server.wait_closed())
loop.close()
