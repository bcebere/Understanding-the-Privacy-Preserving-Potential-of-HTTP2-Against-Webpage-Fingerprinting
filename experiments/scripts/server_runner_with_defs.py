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
from urllib.parse import urlparse

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

SERVER_PATH = Path(__file__).parent

parser = ArgumentParser()
parser.add_argument("-dst_port", "--dst_port", dest="dst_port", help="Destination Port")
parser.add_argument(
    "-http2_server_push",
    "--http2_server_push",
    dest="http2_server_push",
    default=0,
    help="Use HTTP2 Server Push",
)
parser.add_argument(
    "-http2_rnd_server_push",
    "--http2_rnd_server_push",
    dest="http2_rnd_server_push",
    default=0,
    help="Use Random HTTP2 Server Push",
)
parser.add_argument(
    "-http2_batch",
    "--http2_batch",
    dest="http2_batch",
    default=0,
    help="Use HTTP2 Multiplexing",
)
parser.add_argument(
    "-http2_hpack",
    "--http2_hpack",
    dest="http2_hpack",
    default=0,
    help="Use HTTP2 HPACK",
)
parser.add_argument(
    "-http2_rnd_pad",
    "--http2_rnd_pad",
    dest="http2_rnd_pad",
    default=0,
    help="Use HTTP2 Random padding",
)
parser.add_argument(
    "-http2_fixed_pad",
    "--http2_fixed_pad",
    dest="http2_fixed_pad",
    default=0,
    help="Use HTTP2 Fixed padding",
)

parser.add_argument(
    "-http2_hints103",
    "--http2_hints103",
    dest="http2_hints103",
    default=0,
    help="Use HTTP2 HINTS 103",
)
parser.add_argument(
    "-http2_rnd_hints103",
    "--http2_rnd_hints103",
    dest="http2_rnd_hints103",
    default=0,
    help="Use Random HTTP2 HINTS 103",
)

parser.add_argument(
    "-defense_tamaraw",
    "--defense_tamaraw",
    dest="defense_tamaraw",
    help="Activate Tamaraw defense.",
    default=0,
)
parser.add_argument(
    "-defense_alpaca",
    "--defense_alpaca",
    dest="defense_alpaca",
    help="Activate ALPaCA defense.",
    default=0,
)

args = parser.parse_args()

assert args.dst_port is not None

SERVER_IP = "0.0.0.0"
SERVER_PORT = args.dst_port


USE_SERVER_PUSH = int(args.http2_server_push)
USE_RND_SERVER_PUSH = int(args.http2_rnd_server_push)
USE_MULTIPLEXING_RANDOM = int(args.http2_batch)
USE_RANDOM_HPACK = int(args.http2_hpack)
USE_HINTS103 = int(args.http2_hints103)
USE_RND_HINTS103 = int(args.http2_rnd_hints103)
USE_RND_PADDING = int(args.http2_rnd_pad)
USE_FIXED_PADDING = int(args.http2_fixed_pad)

DEFENSE_TAMARAW = int(args.defense_tamaraw)
DEFENSE_ALPACA = int(args.defense_alpaca)

OUT_WINDOW_SIZE = None
SEND_DELAY = None
SEND_DELAY_THRESHOLD = None

INJECT_HI_LIMIT = 100
INJECT_LO_LIMIT = 4

PAD_CONSTANT = 512

if DEFENSE_ALPACA:
    USE_RND_SERVER_PUSH = 1
    USE_RND_PADDING = 1
elif DEFENSE_TAMARAW:
    USE_FIXED_PADDING = 1
    SEND_DELAY = 0.001
    SEND_DELAY_THRESHOLD = 4096
    OUT_WINDOW_SIZE = 2048
    PAD_CONSTANT = 8092
    USE_RND_SERVER_PUSH = 1


RequestData = collections.namedtuple("RequestData", ["headers", "data"])

DATA_PATH = Path("data")
SRV_DB_PATH = DATA_PATH / "server_trace"
CLIENT_DB_PATH = DATA_PATH / "client_trace"
BIN_DB_PATH = DATA_PATH / "bin"

print(
    f"""
      Test config:
        Server Push: {USE_SERVER_PUSH}
        Noise Server Push: {USE_RND_SERVER_PUSH}
        Multiplexing Agg: {USE_MULTIPLEXING_RANDOM}
        RANDOM HPACK: {USE_RANDOM_HPACK}
        HINTS103: {USE_HINTS103}
        Noise HINTS103: {USE_RND_HINTS103}
        RND PAD: {USE_RND_PADDING}
        Fixed PAD: {USE_FIXED_PADDING}

      Defenses:
        Defense 1 Tamaraw defense: {DEFENSE_TAMARAW}
        Defense 3 ALPACA defense: {DEFENSE_ALPACA}
"""
)


def _add_injection_hints(database, low=100, high=5000):
    hints = []
    padding_size = low
    while padding_size < high:
        key = f"/inject_{padding_size}B"
        database[key] = {
            "data_size": padding_size,
            "url_local": key,
            "content_type": "image/png",
            "headers": {},
            "timeout_s": 0.0,
        }
        hints.append(
            {
                "url_local": key,
            }
        )
        step = random.randint(512, 10000)
        padding_size += step

    return hints, database


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
        self.server_name = "mock"
        self.conn_database = None
        self.full_conn_database = None
        self.conn_client_database = None

        # conn default behaviour : inherit globals. Can be overwritten via client headers.
        if USE_RND_PADDING:
            self.pad_constant = random.randint(1024, 8000)
        else:
            self.pad_constant = PAD_CONSTANT

        self.defense_active_per_connection = True
        # HTTP2 mods
        self.use_server_push = USE_SERVER_PUSH
        self.use_rnd_server_push = USE_RND_SERVER_PUSH
        self.use_multiplexing = USE_MULTIPLEXING_RANDOM
        self.use_random_hpack = USE_RANDOM_HPACK
        self.use_hints103 = USE_HINTS103
        self.use_rnd_hints103 = USE_RND_HINTS103
        # Tamaraw/Alpaca related
        self.use_random_padding = bool(USE_RND_PADDING or USE_FIXED_PADDING)
        self.always_pad = bool(USE_FIXED_PADDING)
        self.send_data_delay_threshold = None
        self.send_data_delay = None
        self.send_data_counter = 0

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
                    if self.use_multiplexing:
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

    def _configure_connection_defenses(self, headers):
        if "defend_connection" in headers:
            self.defense_active_per_connection = bool(int(headers["defend_connection"]))

        if not self.defense_active_per_connection:
            self.use_server_push = False
            self.use_rnd_server_push = False
            self.use_multiplexing = False
            self.use_random_hpack = False
            self.use_hints103 = False
            self.use_rnd_hints103 = False
            self.use_random_padding = False
            self.always_pad = False
            self.send_data_delay_threshold = None
            self.send_data_delay = None
            self.send_data_counter = 0
        else:
            if OUT_WINDOW_SIZE is not None:
                self.conn.max_outbound_frame_size = OUT_WINDOW_SIZE

            if SEND_DELAY_THRESHOLD is not None:
                self.send_data_delay_threshold = SEND_DELAY_THRESHOLD
            if SEND_DELAY_THRESHOLD is not None:
                self.send_data_delay = SEND_DELAY
            self.send_data_counter = 0

        if self.use_random_hpack and self.server_name == "mock":
            should_break_cache = random.choice([True, False])

            if should_break_cache:
                big_header = random.randint(2**12 + 1, 2**14)
                self.server_name = "#" * big_header
                print(
                    f"[HTTP2][HPACK] Prepare HPACK padding. Big Header = {big_header}"
                )

    def request_received(self, headers: List[Tuple[str, str]], stream_id: int):
        headers = collections.OrderedDict(headers)
        # method = headers[":method"]
        testcase = headers["label"]
        connection_id = None
        if "connection_id" in headers:
            connection_id = headers["connection_id"]

        self._configure_connection_defenses(headers)

        def _get_domain(url):
            parsed_url = urlparse(url)
            return parsed_url.netloc

        print(
            f"[CONN {id(self)}] testcase = {testcase} connection_id = {connection_id} defend = {self.defense_active_per_connection}"
        )
        if self.conn_database is None:
            print(
                f"""Server HTTP2 config:
                            Server-Push: {self.use_server_push}
                            RND Server-Push: {self.use_rnd_server_push}
                            Multiplexing: {self.use_multiplexing}
                            HPACK: {self.use_random_hpack}
                            Hints103: {self.use_hints103}
                            RND Hints103: {self.use_rnd_hints103}
                            Random Padding: {self.use_random_padding}
                  """
            )

            with open(SRV_DB_PATH / f"{testcase}.json") as f:
                full_conn_database = json.load(f)
                if connection_id is None:
                    self.conn_database = full_conn_database
                else:
                    self.conn_database = {}
                    for srvkey in full_conn_database:
                        keydata = full_conn_database[srvkey]
                        fullurl = keydata["url"]
                        urlkey = _get_domain(fullurl)
                        if urlkey == connection_id:
                            self.conn_database[srvkey] = full_conn_database[srvkey]
                self.full_conn_database = full_conn_database
            with open(CLIENT_DB_PATH / f"{testcase}.json") as f:
                full_client_conn_client_database = json.load(f)
                if connection_id is None:
                    self.conn_client_database = full_client_conn_client_database
                else:
                    self.conn_client_database = []
                    for clkey in full_client_conn_client_database:
                        fullurl = clkey["url"]
                        urlkey = _get_domain(fullurl)
                        if urlkey == connection_id:
                            self.conn_client_database.append(clkey)

            self.inject_hints = self.conn_client_database[1:]
            if self.use_rnd_hints103 or self.use_rnd_server_push:
                self.inject_hints, self.conn_database = _add_injection_hints(
                    self.conn_database,
                    low=INJECT_LO_LIMIT * self.pad_constant,
                    high=INJECT_HI_LIMIT * self.pad_constant,
                    # step=self.pad_constant,
                )

        print(
            f"[CONN {id(self)}]",
            connection_id,
            len(self.conn_database),
            len(self.conn_client_database),
        )

        # Store off the request data.
        request_data = RequestData(headers, io.BytesIO())
        self.stream_data[stream_id] = request_data

        if stream_id == 1:
            self._check_prepare_multiplexing(stream_id)
            if self.use_hints103 or self.use_rnd_hints103:
                print(f"[CONN {id(self)}][HTTP2] Prepare HINTS103")
                self._generate_103_early_hints(stream_id)

    def _generate_response_data(self, path: str, request_headers={}):
        if path in self.conn_database:
            db_data = self.conn_database[path]
        elif path in self.full_conn_database:
            # last attempt for various redirects
            db_data = self.full_conn_database[path]
        else:
            print("path missing in database !!!!", path, self.conn_database)
            return b"", {}, "application/text"

        if "body_path" in db_data:
            content_path = db_data["body_path"]
            if not Path(content_path).exists():
                print("Missing content", path, content_path)
                return b"", {}, "application/text"
            body = open(content_path, "rb").read()
        elif "data_size" in db_data:  # generate random data of len
            if db_data["data_size"] == "random":
                response_size = random.randint(100, 10000)
            elif isinstance(db_data["data_size"], (int, float)):
                response_size = int(db_data["data_size"])
            else:
                response_size = 1

            body = "".join(
                random.choices(string.ascii_uppercase + string.digits, k=response_size)
            ).encode("utf-8")
        else:
            raise NotImplementedError()

        if "timeout_ts" in db_data:
            timeout = db_data["timeout_s"]
        else:
            timeout = 0
        if "headers" in db_data:
            response_headers = db_data["headers"]
        else:
            response_headers = {}
        if "content_type" in db_data:
            content_type = db_data["content_type"]
        else:
            content_type = "application/json"

        if "range" in request_headers:
            range_bytes = request_headers["range"].split("bytes=")[-1]
            start, end = range_bytes.split("-")
            print("ranged-request", path, start, end, len(body))
            body = body[int(start) : int(end)]

        if self.use_random_padding:
            pad_size = (int(len(body) / self.pad_constant) + 1) * self.pad_constant
            padding = "".join(
                random.choices(
                    string.ascii_uppercase + string.digits, k=pad_size - len(body)
                )
            )
            old_size = len(body)
            body += padding.encode("utf-8")
            print(
                f"[HTTP2] adding response padding = {self.pad_constant} body len = {len(body)} old body size = {old_size}"
            )

        if timeout > 0:
            time.sleep(float(timeout / 1000))

        print(
            f"[DATA] path={path} timeout={timeout} size={len(body)} content_type = {content_type}"
        )

        return body, response_headers, content_type

    def stream_complete(self, stream_id: int):
        """
        When a stream is complete, we can send our response.
        """
        try:
            request_data = self.stream_data[stream_id]
        except KeyError:
            print("ignore stream!!!", stream_id)
            # Just return, we probably 405'd this already
            return

        headers = request_data.headers
        path = headers[":path"]

        body, _, content_type = self._generate_response_data(
            path, request_headers=headers
        )
        # print(f"request stream_id = {stream_id} path = {path} req_body = {body}")

        response_headers = (
            (":status", "200"),
            ("content-type", content_type),
            # ("content-type", "application/json"),
            ("content-length", str(len(body))),
            ("server", self.server_name),
        )
        self.conn.send_headers(stream_id, response_headers)
        push_streams = []
        if stream_id == 1:
            push_streams = self._prepare_server_push(stream_id)
        elif self.use_rnd_server_push:
            should_push_noise = random.choice([True, False])
            if should_push_noise:
                push_streams = self._prepare_server_push(stream_id)

        asyncio.ensure_future(self.send_data(body, stream_id))

        if len(push_streams) > 0:
            self._send_server_push_data(stream_id, push_streams)

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

    async def send_data(self, data, stream_id, delay=0.01):
        """
        Send data according to the flow control rules.
        """
        if delay > 0:
            # print(f"[HTTP2] Stream={stream_id} Timeout={delay}", time.time())
            await asyncio.sleep(delay)

        # print(f"SEND stream_id={stream_id} data len={len(data)}",)
        if len(data) == 0:
            try:
                self.conn.send_data(stream_id, data, end_stream=True)
                self.transport.write(self.conn.data_to_send())
            except BaseException as e:
                print("failed to send data", e)
            return

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
            # print(f"SEND chunk size = {chunk_size}")
            self.send_data_counter += chunk_size

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

            # Traffic-shaping logic (Tamaraw)
            if (
                self.send_data_delay_threshold is not None
                and self.send_data_counter > self.send_data_delay_threshold
            ):
                self.send_data_counter = 0
                # print(f"[HTTP2] Stream={stream_id} Sleep on data threshold")
                await asyncio.sleep(self.send_data_delay)

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

    # HTTP2: Batching/multiplexing simulation
    def _prepare_multiplexing_strategy(self):
        if self.use_multiplexing:
            strategies = [
                {"ckp_pct": [1]},
                {"ckp_pct": [0.1, 1]},
                {"ckp_pct": [0.3, 1]},
                {"ckp_pct": [0.5, 1]},
                {"ckp_pct": [0.7, 1]},
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
        print(f"[CONN {id(self)}][HTTP2] Using batching strategy", selected)

        return selected

    def _check_prepare_multiplexing(self, parent_stream_id: int):
        if not self.use_multiplexing:
            return

        self.pending_requests_cnt = len(self.conn_client_database)
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
        print("[HTTP2] Batching", self.multiplexing_strategy)

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

    # HTTP2: Proactive resource suggestion
    def _generate_103_early_hints(self, stream_id):
        pending_requests = self.inject_hints  # self.conn_client_database[1:]
        if len(pending_requests) == 0:
            return

        hints_headers = [
            (":status", "103"),
        ]
        for idx, req in enumerate(pending_requests):
            req_path = req["url_local"]
            hints_headers.append(("link", f"<{req_path}>; rel=preload; as=image"))

        print(f"[CONN {id(self)}][HTTP2] HINTS103 {len(hints_headers)} at random ")
        self.conn.send_headers(stream_id, hints_headers)

    def _generate_server_push_headers(self, parent_stream_id: int, path: str):
        # Push a resource
        promised_stream_id = self.conn.get_next_available_stream_id()

        data, headers, content_type = self._generate_response_data(path)

        self.conn.push_stream(
            parent_stream_id,
            promised_stream_id,
            [
                (":method", "GET"),
                (":authority", "localhost"),
                (":scheme", "http"),
                (":path", path),
            ],
        )
        self.conn.send_headers(
            promised_stream_id,
            [
                (":status", "200"),
                ("content-type", content_type),
                ("server", self.server_name),
                ("content-length", str(len(data))),
            ],
        )
        return path, promised_stream_id, data

    def _generate_server_push_data(
        self, parent_stream_id: int, promised_stream_id: int, path: str, data: bytes
    ):
        asyncio.ensure_future(self.send_data(data, promised_stream_id))
        print(
            f" >>> push data path = {path} stream={promised_stream_id} data={len(data)}"
        )
        return promised_stream_id

    def _prepare_server_push(self, parent_stream_id: int):
        if self.use_server_push:
            pending_requests = copy.deepcopy(self.conn_client_database[1:])
        elif self.use_rnd_server_push:
            pending_requests = self.inject_hints
        else:
            return []

        if len(pending_requests) < 2:
            return []

        push_streams_cnt = random.randrange(1, min(10, len(pending_requests)))
        random.shuffle(pending_requests)
        print(
            f"[CONN {id(self)}][HTTP2] Pushing {push_streams_cnt}/{len(pending_requests)} at random "
        )

        push_streams = []
        for req in pending_requests:
            req_path = req["url_local"]
            push_streams.append(
                self._generate_server_push_headers(parent_stream_id, req_path)
            )
            if len(push_streams) >= push_streams_cnt:
                break

        return push_streams

    def _send_server_push_data(self, parent_stream_id: int, push_streams: list):
        if not (self.use_server_push or self.use_rnd_server_push):
            return

        for req_path, push_stream_id, push_data in push_streams:
            self._generate_server_push_data(
                parent_stream_id, push_stream_id, req_path, push_data
            )


ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
ssl_context.options |= ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1 | ssl.OP_NO_TLSv1_2
ssl_context.load_cert_chain(
    certfile=SERVER_PATH / "keys/cert.pem", keyfile=SERVER_PATH / "keys/key.pem"
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
