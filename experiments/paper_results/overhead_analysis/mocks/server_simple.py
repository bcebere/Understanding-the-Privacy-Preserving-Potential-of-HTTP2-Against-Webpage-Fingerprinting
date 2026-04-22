import asyncio
import collections
import copy
import io
import json
import random
import socket
import ssl
import string
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urlparse

import h2
import h2.connection
import h2.events
import numpy as np
from h2.config import H2Configuration
from h2.connection import H2Connection
from h2.errors import ErrorCodes
from h2.events import (
    ConnectionTerminated,
    DataReceived,
    PingAckReceived,
    PingReceived,
    RemoteSettingsChanged,
    RequestReceived,
    StreamEnded,
    StreamReset,
    WindowUpdated,
)
from h2.exceptions import ProtocolError, StreamClosedError
from h2.settings import SettingCodes

SERVER_PATH = Path(__file__).parent.parent / "mocks"
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
    "-http2_rnd_out_window",
    "--http2_rnd_out_window",
    dest="http2_rnd_out_window",
    default=0,
    help="Use Random HTTP2 Send Window",
)
parser.add_argument(
    "-http2_batch",
    "--http2_batch",
    dest="http2_batch",
    default=0,
    help="Use HTTP2 Multiplexing",
)
parser.add_argument(
    "-http2_rnd_delay",
    "--http2_rnd_delay",
    dest="http2_rnd_delay",
    default=0,
    help="Use HTTP2 Stream Delays",
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
    "-http2_hints103",
    "--http2_hints103",
    dest="http2_hints103",
    default=0,
    help="Use HTTP2 HINTS 103",
)
parser.add_argument(
    "-http2_hints103_lo_limit",
    "--http2_hints103_lo_limit",
    dest="http2_hints103_lo_limit",
    default=1,
    help="Use HTTP2 HINTS 103 at random at least for...",
)

parser.add_argument(
    "-http2_hints103_hi_limit",
    "--http2_hints103_hi_limit",
    dest="http2_hints103_hi_limit",
    default=5,
    help="Use HTTP2 HINTS 103 at random at max for...",
)
parser.add_argument(
    "-http2_rnd_pings",
    "--http2_rnd_pings",
    dest="http2_rnd_pings",
    default=0,
    help="Send HTTP2 Ping",
)
parser.add_argument(
    "-http2_global_hints103",
    "--http2_global_hints103",
    dest="http2_global_hints103",
    default=0,
    help="Use HTTP2 Global HINTS 103 From the first connection",
)
parser.add_argument(
    "-http2_rnd_hints103",
    "--http2_rnd_hints103",
    dest="http2_rnd_hints103",
    default=0,
    help="Use Random HTTP2 HINTS 103",
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
USE_GLOBAL_HINTS103 = int(args.http2_global_hints103)
USE_RND_HINTS103 = int(args.http2_rnd_hints103)
USE_RND_PADDING = int(args.http2_rnd_pad)
USE_RND_PINGS = int(args.http2_rnd_pings)
USE_RND_DELAY = int(args.http2_rnd_delay)
USE_RND_OUT_WINDOW = int(args.http2_rnd_out_window)

HINTS_LO_LIMIT = int(args.http2_hints103_lo_limit)
HINTS_HI_LIMIT = int(args.http2_hints103_hi_limit)

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
        Global HINTS103: {USE_GLOBAL_HINTS103}
        Noise HINTS103: {USE_RND_HINTS103}
        RND PAD: {USE_RND_PADDING}
        RND PINGS: {USE_RND_PINGS}
        RND DELAY: {USE_RND_DELAY}
        RND OUT WINDOW: {USE_RND_OUT_WINDOW}

"""
)


def _add_injection_hints(database, cnt=15, existing_hints=[]):
    hints = []
    existing_sizes = []
    for db_key in database:
        content_path = database[db_key]["body_path"]
        if not Path(content_path).exists():
            continue
        body = open(content_path, "rb").read()

        existing_sizes.append(len(body))

    print("EXISTING HINTS", existing_sizes)

    min_sample = 1001
    max_sample = 10000

    if len(existing_sizes) > 0:
        min_sample = int(max(existing_sizes) / 2)
        if 2 * max(existing_sizes) > max_sample:
            max_sample = 2 * max(existing_sizes) + 1

        if len(existing_sizes) > cnt:
            cnt = len(existing_sizes)

    print("HINTS PARAMS", cnt, min_sample, max_sample)
    for path in range(cnt):
        padding_size = random.randint(min_sample, max_sample)
        key = f"/inject_{padding_size}B"
        database[key] = {
            "data_size": padding_size,
            "url_local": key,
            "content_type": "image/png",
            "headers": {},
            "timeout_s": random.uniform(0, 1),
        }
        hints.append(
            {
                "url_local": key,
            }
        )

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
        self.connection_id = None

        # flow-control sanity
        self._can_write = asyncio.Event()
        self._can_write.set()

        # conn default behaviour : inherit globals. Can be overwritten via client headers.
        self.pad_constant = random.randint(128, 1024)
        self.defense_active_per_connection = True
        self.use_server_push = USE_SERVER_PUSH
        self.use_rnd_server_push = USE_RND_SERVER_PUSH
        self.use_multiplexing = USE_MULTIPLEXING_RANDOM
        self.use_random_hpack = USE_RANDOM_HPACK
        self.use_hints103 = bool(USE_HINTS103)
        self.use_global_hints103 = bool(USE_GLOBAL_HINTS103)
        self.use_rnd_hints103 = bool(USE_RND_HINTS103)
        self.use_random_padding = bool(USE_RND_PADDING)
        self.use_random_pings = bool(USE_RND_PINGS)
        self.max_outbound_frame_size = 16384
        if USE_RND_OUT_WINDOW:
            self.max_outbound_frame_size = random.randint(2**12, 2**14)
            print(f"[DEFENSE] Out Window = {self.max_outbound_frame_size}")
        # Traffic shaping
        self.send_data_delay_threshold = None
        self.send_data_counter = 0
        self.use_random_delay = bool(USE_RND_DELAY)

    def connection_made(self, transport: asyncio.Transport):
        self.transport = transport
        sock = transport.get_extra_info("socket")
        if sock:
            sock.setsockopt(
                socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024 * 1024
            )  # 8 MiB

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

    def pause_writing(self):
        self._can_write.clear()

    def resume_writing(self):
        self._can_write.set()

    def data_received(self, data: bytes):
        try:
            events = self.conn.receive_data(data)
        except ProtocolError:
            self.transport.write(self.conn.data_to_send())
            self.transport.close()
        else:
            try:
                self.transport.write(self.conn.data_to_send())
            except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                print("RECV failed on stream %d: %s", exc)
                return

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
                    self.respond_ping(event)
                elif isinstance(event, PingAckReceived):
                    pass

                try:
                    self.transport.write(self.conn.data_to_send())
                except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                    print("FLUSH failed on stream %d: %s", exc)
                    return

    def request_received(self, headers: List[Tuple[str, str]], stream_id: int):
        headers = collections.OrderedDict(headers)
        # method = headers[":method"]
        testcase = headers["label"]
        connection_id = None
        if "connection_id" in headers:
            connection_id = headers["connection_id"]
            self.connection_id = connection_id

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
            self.use_random_pings = False
            self.use_random_delay = False

        if self.use_global_hints103:
            self.use_hints103 = True

        if self.use_random_delay:
            self.send_data_delay_threshold = random.randint(2**8, 2**10)
            print(
                f"[HTTP2][DELAY][{self.connection_id}] Threshold = {self.send_data_delay_threshold} bytes"
            )

        if self.use_random_hpack and self.server_name == "mock":
            should_break_cache = random.choice([True, False])

            if should_break_cache:
                big_header = random.randint(2**12 + 1, 2**14)
                self.server_name = "#" * big_header
                print(
                    f"[HTTP2][HPACK] Prepare HPACK padding. Big Header = {big_header}"
                )

        def _get_domain(url):
            parsed_url = urlparse(url)
            return parsed_url.netloc

        print(
            f"[CONN {id(self)}] testcase = {testcase} connection_id = {connection_id} stream_id = {stream_id} defend = {self.defense_active_per_connection}"
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
                            Random Delays: {self.use_random_delay}
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
                    existing_hints=self.inject_hints,
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

        path = headers[":path"]
        if stream_id == 1:
            self._check_prepare_multiplexing(stream_id, path)

            if self.use_hints103 or self.use_rnd_hints103:
                print(f"[CONN {id(self)}][HTTP2] Prepare HINTS103")
                self._generate_103_early_hints(stream_id, path)

    def _generate_response_data(self, path: str, request_headers={}):
        if path in self.conn_database:
            db_data = self.conn_database[path]
        elif path in self.full_conn_database:
            # last attempt for various redirects
            db_data = self.full_conn_database[path]
        else:
            print("path missing in database !!!!", path, self.conn_database)
            return b"", {}, "application/text", 0

        if "body_path" in db_data:
            content_path = db_data["body_path"]
            if not Path(content_path).exists():
                print("Missing content", path, content_path)
                return b"", {}, "application/text", 0
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

        if "timeout_s" in db_data:
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

        frame_delay = 0
        if self.use_random_delay:
            should_delay = random.choice([True, False])
            if should_delay:
                frame_delay = random.uniform(0.00001, 0.001)

        if "range" in request_headers:
            range_bytes = request_headers["range"].split("bytes=")[-1]
            start, end = range_bytes.split("-")
            print("ranged-request", path, start, end, len(body))
            body = body[int(start) : int(end)]

        if self.use_random_padding:
            should_pad = random.choice([True, False])
            if should_pad and len(body) % self.pad_constant != 0:
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

        print(
            f"[DATA] path={path} resp.delay={timeout} frame.delay={frame_delay} size={len(body)} content_type = {content_type}"
        )

        return body, response_headers, content_type, timeout, frame_delay

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

    def _check_prepare_multiplexing(self, parent_stream_id: int, prev_path: str):
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

    def _generate_103_early_hints(self, stream_id, prev_path: str):
        pending_requests = self.inject_hints  # self.conn_client_database[1:]
        if len(pending_requests) == 0:
            return

        hints_headers = [
            (":status", "103"),
        ]

        sample_size = random.randint(
            HINTS_LO_LIMIT, HINTS_HI_LIMIT
        )  # random size between HINTS_LO_LIMIT and HINTS_LI_LIMIT
        sampled_requests = random.choices(pending_requests, k=sample_size)

        # hints_streams_limit = random.randrange(0, len(pending_requests))
        # random.shuffle(pending_requests)

        for idx, req in enumerate(sampled_requests):
            req_path = req["url_local"]
            hints_headers.append(("link", f"<{req_path}>; rel=preload; as=image"))

        print(f"[HTTP2][{self.connection_id}] HINTS = {len(hints_headers)}")
        print(f"[CONN {id(self)}][HTTP2] HINTS103 {len(hints_headers)} at random ")
        self.conn.send_headers(stream_id, hints_headers)

    def _generate_server_push_headers(self, parent_stream_id: int, path: str):
        # Push a resource
        promised_stream_id = self.conn.get_next_available_stream_id()

        (
            data,
            headers,
            content_type,
            response_delay,
            frame_delay,
        ) = self._generate_response_data(path)

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
        return path, promised_stream_id, data, response_delay, frame_delay

    def _generate_server_push_data(
        self,
        parent_stream_id: int,
        promised_stream_id: int,
        path: str,
        data: bytes,
        response_delay: float = 0,
        frame_delay: float = 0,
    ):
        asyncio.ensure_future(
            self.send_data(
                data,
                promised_stream_id,
                response_delay=response_delay,
                frame_delay=frame_delay,
            )
        )
        print(
            f" >>> push data path = {path} stream={promised_stream_id} data={len(data)}"
        )
        return promised_stream_id

    def _prepare_server_push(self, parent_stream_id: int, prev_path: str):
        if self.use_server_push:
            pending_requests = copy.deepcopy(self.conn_client_database[1:])
        elif self.use_rnd_server_push:
            pending_requests = self.inject_hints
        else:
            return []

        if len(pending_requests) < 2:
            return []

        push_streams_cnt = random.randrange(1, len(pending_requests))
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

    def _send_server_push_data(
        self, parent_stream_id: int, prev_path: str, push_streams: list
    ):
        if not self.use_server_push and not self.use_rnd_server_push:
            return

        for (
            req_path,
            push_stream_id,
            push_data,
            response_delay,
            frame_delay,
        ) in push_streams:
            self._generate_server_push_data(
                parent_stream_id,
                push_stream_id,
                req_path,
                push_data,
                response_delay=response_delay,
                frame_delay=frame_delay,
            )

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

        (
            body,
            _,
            content_type,
            response_delay,
            frame_delay,
        ) = self._generate_response_data(path, request_headers=headers)
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
            push_streams = self._prepare_server_push(stream_id, path)
        elif self.use_rnd_server_push:
            should_push_noise = random.choice([True, False])
            if should_push_noise:
                print("Pushing resources at random !!!!")
                push_streams = self._prepare_server_push(stream_id, path)

        asyncio.ensure_future(
            self.send_data(
                body, stream_id, response_delay=response_delay, frame_delay=frame_delay
            )
        )

        if len(push_streams) > 0:
            self._send_server_push_data(stream_id, path, push_streams)

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

    async def send_data(
        self, data, stream_id, response_delay: bool = 0, frame_delay: bool = 0
    ):
        """
        Send data according to the flow control rules.
        """
        if response_delay > 0:
            await asyncio.sleep(response_delay)
            print(
                f"[HTTP2][{self.connection_id}] Stream={stream_id} Timeout={response_delay}",
                time.time(),
                flush=True,
            )

        if len(data) == 0:
            try:
                self.conn.send_data(stream_id, data, end_stream=True)
                self.transport.write(self.conn.data_to_send())
            except BaseException as e:
                print("failed to send data", e)
            return

        print("SEND DATA for stream ", stream_id, len(data))
        while data:
            await self._can_write.wait()  # <-- real back-pressure

            while self.conn.local_flow_control_window(stream_id) < 1:
                try:
                    await self.wait_for_flow_control(stream_id)
                except asyncio.CancelledError:
                    return

            chunk_size = min(
                self.conn.local_flow_control_window(stream_id),
                len(data),
                self.conn.max_outbound_frame_size,
                self.max_outbound_frame_size,
            )
            self.send_data_counter += chunk_size

            try:
                self.conn.send_data(
                    stream_id, data[:chunk_size], end_stream=(chunk_size == len(data))
                )
            except (StreamClosedError, ProtocolError):
                # The stream got closed and we didn't get told. We're done
                # here.
                break

            try:
                self.transport.write(self.conn.data_to_send())
            except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                print("write failed on stream %d: %s", stream_id, exc)
                break

            data = data[chunk_size:]

            # Traffic-shaping logic
            if (
                self.use_random_delay
                and frame_delay > 0
                and self.send_data_delay_threshold is not None
                and self.send_data_counter > self.send_data_delay_threshold
            ):
                print(
                    f"[HTTP2][{self.connection_id}] Stream={stream_id} Sleep {frame_delay} on data threshold: {self.send_data_counter}/{self.send_data_delay_threshold}"
                )
                self.send_data_counter = 0
                await asyncio.sleep(frame_delay)

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

    def respond_ping(self, event):
        pass


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
