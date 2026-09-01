# server_with_defs.py

import asyncio
import collections
import copy
import io
import json
import random
import socket
import ssl
import string
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urlparse

import h2
import numpy as np
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
from server_defenses.levels import params

SERVER_PATH = Path(__file__).parent.parent / "mocks"
parser = ArgumentParser()
parser.add_argument("-dst_port", "--dst_port", dest="dst_port", help="Destination Port")
parser.add_argument(
    "-level",
    "--level",
    dest="level",
    default="mid1",
    help="Defense intensity from server_defenses/levels.py",
)
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
LEVEL = args.level

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
INJECT_MAX_COUNT = 100  # cap the noise pool; see _add_injection_hints

PAD_CONSTANT = 512
PAD_LO, PAD_HI = 1024, 8000
PUSH_MAX = 10
PUSH_PROB = 0.5

if DEFENSE_ALPACA:
    # Figure 6: PAD ~ U{1024..8000} per connection, probabilistic padding,
    # probabilistic PUSH_PROMISE of synthetic resources.
    USE_RND_SERVER_PUSH = 1
    USE_RND_PADDING = 1
    _p = params("alpaca", LEVEL)
    PAD_LO, PAD_HI = _p["pad_lo"], _p["pad_hi"]
    PUSH_MAX, PUSH_PROB = _p["push_max"], _p["push_prob"]
elif DEFENSE_TAMARAW:
    # Figure 7: W = 2048 B, tau = 0.001 s, dLIM = 4096 B, PAD = 8092 B,
    # padding applied to every response.
    USE_FIXED_PADDING = 1
    USE_RND_SERVER_PUSH = 1
    _p = params("tamaraw", LEVEL)
    PAD_CONSTANT = _p["pad_constant"]
    OUT_WINDOW_SIZE = _p["out_window"]
    SEND_DELAY = _p["send_delay"]
    SEND_DELAY_THRESHOLD = _p["delay_threshold"]
    PUSH_MAX, PUSH_PROB = _p["push_max"], _p["push_prob"]

RequestData = collections.namedtuple("RequestData", ["headers", "data"])

DATA_PATH = Path("data")
SRV_DB_PATH = DATA_PATH / "server_trace"
CLIENT_DB_PATH = DATA_PATH / "client_trace"
BIN_DB_PATH = DATA_PATH / "bin"

print(
    f"""
      Test config:
        Level: {LEVEL}
        Server Push: {USE_SERVER_PUSH}
        Noise Server Push: {USE_RND_SERVER_PUSH}
        Multiplexing Agg: {USE_MULTIPLEXING_RANDOM}
        RANDOM HPACK: {USE_RANDOM_HPACK}
        HINTS103: {USE_HINTS103}
        Noise HINTS103: {USE_RND_HINTS103}
        RND PAD: {USE_RND_PADDING}  (range {PAD_LO}-{PAD_HI})
        Fixed PAD: {USE_FIXED_PADDING}  (constant {PAD_CONSTANT})
        Push: max={PUSH_MAX} prob={PUSH_PROB}

      Defenses:
        Tamaraw: {DEFENSE_TAMARAW}  window={OUT_WINDOW_SIZE} delay={SEND_DELAY} thresh={SEND_DELAY_THRESHOLD}
        ALPaCA : {DEFENSE_ALPACA}
"""
)


def _add_injection_hints(database, low=100, high=5000, max_count=INJECT_MAX_COUNT):
    """Synthetic noise resources sized in [low, high].

    The step is randomized, so the pool size scales with the padding constant.
    max_count bounds it: at high padding levels the unbounded loop produced
    hundreds of multi-megabyte entries, which made the noise pool itself an
    uncontrolled second intensity dimension.
    """
    hints = []
    padding_size = low
    while padding_size < high and len(hints) < max_count:
        key = f"/inject_{padding_size}B"
        database[key] = {
            "data_size": padding_size,
            "url_local": key,
            "content_type": "image/png",
            "headers": {},
            "timeout_s": 0.0,
        }
        hints.append({"url_local": key})
        padding_size += random.randint(512, 10000)

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
        self.multiplexing_strategy = None
        self.table_size = 4096
        self.server_name = "mock"
        self.conn_database = None
        self.full_conn_database = None
        self.conn_client_database = None
        self.inject_hints = []

        # flow-control back-pressure
        self._can_write = asyncio.Event()
        self._can_write.set()

        # conn defaults: inherit globals, overridable via client headers
        if USE_RND_PADDING:
            self.pad_constant = random.randint(PAD_LO, PAD_HI)
        else:
            self.pad_constant = PAD_CONSTANT

        self.defense_active_per_connection = True
        self.use_server_push = USE_SERVER_PUSH
        self.use_rnd_server_push = USE_RND_SERVER_PUSH
        self.use_multiplexing = USE_MULTIPLEXING_RANDOM
        self.use_random_hpack = USE_RANDOM_HPACK
        self.use_hints103 = USE_HINTS103
        self.use_rnd_hints103 = USE_RND_HINTS103

        # ALPaCA pads probabilistically (Fig. 6); Tamaraw pads every response
        # (Fig. 7).  always_pad was previously set and never read, so ALPaCA
        # padded unconditionally.
        self.use_random_padding = bool(USE_RND_PADDING or USE_FIXED_PADDING)
        self.always_pad = bool(USE_FIXED_PADDING)

        # h2 derives max_outbound_frame_size from the peer's SETTINGS, so it
        # cannot be assigned; keep our own cap and apply it when chunking.
        self.max_outbound_frame_size = OUT_WINDOW_SIZE or 16384

        self.send_data_delay_threshold = None
        self.send_data_delay = None
        self.send_data_counter = 0

    # ------------------------------------------------------------------
    # connection lifecycle
    # ------------------------------------------------------------------

    def connection_made(self, transport: asyncio.Transport):
        self.transport = transport
        sock = transport.get_extra_info("socket")
        if sock:
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024 * 1024)
            except OSError:
                pass
        self.conn.initiate_connection()
        self.conn.update_settings(
            {h2.settings.SettingCodes.HEADER_TABLE_SIZE: self.table_size}
        )
        self._flush()

    def connection_lost(self, exc):
        for future in self.flow_control_futures.values():
            future.cancel()
        self.flow_control_futures = {}

    def pause_writing(self):
        self._can_write.clear()

    def resume_writing(self):
        self._can_write.set()

    def _flush(self):
        """Write pending frames, tolerating a peer that has gone away."""
        try:
            data = self.conn.data_to_send()
            if data:
                self.transport.write(data)
            return True
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            print("write failed:", exc)
            return False

    def data_received(self, data: bytes):
        try:
            events = self.conn.receive_data(data)
        except ProtocolError as exc:
            print("proto error:", exc)
            self._flush()
            self.transport.close()
            return

        if not self._flush():
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
                return
            elif isinstance(event, StreamReset):
                self.stream_reset(event.stream_id)
            elif isinstance(event, WindowUpdated):
                self.window_updated(event.stream_id, event.delta)
            elif isinstance(event, RemoteSettingsChanged):
                if SettingCodes.INITIAL_WINDOW_SIZE in event.changed_settings:
                    self.window_updated(None, 0)
            elif isinstance(event, PingReceived):
                self.handle_ping(event)

            if not self._flush():
                return

    def handle_ping(self, event):
        """h2 queues the PING ACK itself.  Sending conn.ping() here would emit
        a NEW ping in reply, doubling ping traffic on every client probe."""
        pass

    # ------------------------------------------------------------------
    # per-connection defense config
    # ------------------------------------------------------------------

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
            self.max_outbound_frame_size = 16384
            self.send_data_delay_threshold = None
            self.send_data_delay = None
            self.send_data_counter = 0
        else:
            if OUT_WINDOW_SIZE is not None:
                self.max_outbound_frame_size = OUT_WINDOW_SIZE
            if SEND_DELAY_THRESHOLD is not None:
                self.send_data_delay_threshold = SEND_DELAY_THRESHOLD
            if SEND_DELAY is not None:
                self.send_data_delay = SEND_DELAY
            self.send_data_counter = 0

        if self.use_random_hpack and self.server_name == "mock":
            if random.choice([True, False]):
                big_header = random.randint(2**12 + 1, 2**14)
                self.server_name = "#" * big_header
                print(f"[HTTP2][HPACK] Big Header = {big_header}")

    def request_received(self, headers: List[Tuple[str, str]], stream_id: int):
        headers = collections.OrderedDict(headers)
        testcase = headers["label"]
        connection_id = headers.get("connection_id")

        self._configure_connection_defenses(headers)

        def _get_domain(url):
            return urlparse(url).netloc

        print(
            f"[CONN {id(self)}] testcase={testcase} connection_id={connection_id} "
            f"defend={self.defense_active_per_connection}"
        )
        if self.conn_database is None:
            print(
                f"""Server HTTP2 config:
                            Level: {LEVEL}
                            Server-Push: {self.use_server_push}
                            RND Server-Push: {self.use_rnd_server_push}
                            Multiplexing: {self.use_multiplexing}
                            HPACK: {self.use_random_hpack}
                            Hints103: {self.use_hints103}
                            RND Hints103: {self.use_rnd_hints103}
                            Padding: {self.use_random_padding} always={self.always_pad} const={self.pad_constant}
                  """
            )

            with open(SRV_DB_PATH / f"{testcase}.json") as f:
                full_conn_database = json.load(f)
                if connection_id is None:
                    self.conn_database = full_conn_database
                else:
                    self.conn_database = {
                        k: v
                        for k, v in full_conn_database.items()
                        if _get_domain(v["url"]) == connection_id
                    }
                self.full_conn_database = full_conn_database

            with open(CLIENT_DB_PATH / f"{testcase}.json") as f:
                full_client = json.load(f)
                if connection_id is None:
                    self.conn_client_database = full_client
                else:
                    self.conn_client_database = [
                        c for c in full_client if _get_domain(c["url"]) == connection_id
                    ]

            self.inject_hints = self.conn_client_database[1:]
            if self.use_rnd_hints103 or self.use_rnd_server_push:
                self.inject_hints, self.conn_database = _add_injection_hints(
                    self.conn_database,
                    low=INJECT_LO_LIMIT * self.pad_constant,
                    high=INJECT_HI_LIMIT * self.pad_constant,
                )

        print(
            f"[CONN {id(self)}]",
            connection_id,
            len(self.conn_database),
            len(self.conn_client_database),
        )

        self.stream_data[stream_id] = RequestData(headers, io.BytesIO())

        if stream_id == 1:
            self._check_prepare_multiplexing(stream_id)
            if self.use_hints103 or self.use_rnd_hints103:
                print(f"[CONN {id(self)}][HTTP2] Prepare HINTS103")
                self._generate_103_early_hints(stream_id)

    # ------------------------------------------------------------------
    # responses
    # ------------------------------------------------------------------

    def _generate_response_data(self, path: str, request_headers={}):
        if path in self.conn_database:
            db_data = self.conn_database[path]
        elif self.full_conn_database and path in self.full_conn_database:
            db_data = self.full_conn_database[path]  # redirects
        else:
            print("path missing in database !!!!", path)
            return b"", {}, "application/text", 0

        if "body_path" in db_data:
            content_path = db_data["body_path"]
            if not Path(content_path).exists():
                print("Missing content", path, content_path)
                return b"", {}, "application/text", 0
            body = open(content_path, "rb").read()
        elif "data_size" in db_data:
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

        # was `if "timeout_ts" in db_data` -- a typo, so the recorded response
        # delay never applied.  It is now honoured, and awaited in send_data
        # rather than slept on the event loop.
        timeout = db_data.get("timeout_s", 0) or 0
        response_headers = db_data.get("headers", {})
        content_type = db_data.get("content_type", "application/json")

        if "range" in request_headers:
            range_bytes = request_headers["range"].split("bytes=")[-1]
            start, end = range_bytes.split("-")
            body = body[int(start) : int(end)]

        if self.use_random_padding:
            should_pad = self.always_pad or random.choice([True, False])
            if should_pad and len(body) % self.pad_constant != 0:
                pad_size = (len(body) // self.pad_constant + 1) * self.pad_constant
                padding = "".join(
                    random.choices(
                        string.ascii_uppercase + string.digits, k=pad_size - len(body)
                    )
                )
                old_size = len(body)
                body += padding.encode("utf-8")
                print(f"[HTTP2] pad={self.pad_constant} {old_size} -> {len(body)}")

        print(f"[DATA] path={path} delay={timeout} size={len(body)} ct={content_type}")
        return body, response_headers, content_type, timeout

    def stream_complete(self, stream_id: int):
        try:
            request_data = self.stream_data[stream_id]
        except KeyError:
            print("ignore stream!!!", stream_id)
            return

        headers = request_data.headers
        path = headers[":path"]

        body, _, content_type, response_delay = self._generate_response_data(
            path, request_headers=headers
        )

        try:
            self.conn.send_headers(
                stream_id,
                (
                    (":status", "200"),
                    ("content-type", content_type),
                    ("content-length", str(len(body))),
                    ("server", self.server_name),
                ),
            )
        except (StreamClosedError, ProtocolError) as exc:
            print("send_headers failed", stream_id, exc)
            return

        push_streams = []
        if stream_id == 1:
            push_streams = self._prepare_server_push(stream_id)
        elif self.use_rnd_server_push and random.random() < PUSH_PROB:
            push_streams = self._prepare_server_push(stream_id)

        asyncio.ensure_future(
            self.send_data(body, stream_id, response_delay=response_delay)
        )

        if push_streams:
            self._send_server_push_data(stream_id, push_streams)

    def receive_data(self, data: bytes, stream_id: int):
        try:
            stream_data = self.stream_data[stream_id]
        except KeyError:
            self.conn.reset_stream(stream_id, error_code=ErrorCodes.PROTOCOL_ERROR)
        else:
            stream_data.data.write(data)

    def stream_reset(self, stream_id):
        if stream_id in self.flow_control_futures:
            self.flow_control_futures.pop(stream_id).cancel()

    async def send_data(self, data, stream_id, response_delay: float = 0):
        """Send according to flow control.  The default delay was 10 ms, which
        applied an undocumented pause to every response including pushes."""
        if response_delay and response_delay > 0:
            await asyncio.sleep(float(response_delay))

        if len(data) == 0:
            try:
                self.conn.send_data(stream_id, data, end_stream=True)
                self._flush()
            except (StreamClosedError, ProtocolError) as exc:
                print("failed to send empty data", exc)
            return

        while data:
            await self._can_write.wait()

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
                break

            if not self._flush():
                break
            data = data[chunk_size:]

            # Tamaraw pacing
            if (
                self.send_data_delay_threshold is not None
                and self.send_data_delay is not None
                and self.send_data_counter > self.send_data_delay_threshold
            ):
                self.send_data_counter = 0
                await asyncio.sleep(self.send_data_delay)

    async def wait_for_flow_control(self, stream_id):
        f = asyncio.Future()
        self.flow_control_futures[stream_id] = f
        await f

    def window_updated(self, stream_id, delta):
        if stream_id and stream_id in self.flow_control_futures:
            self.flow_control_futures.pop(stream_id).set_result(delta)
        elif not stream_id:
            for f in self.flow_control_futures.values():
                if not f.done():
                    f.set_result(delta)
            self.flow_control_futures = {}

    # ------------------------------------------------------------------
    # multiplexing
    # ------------------------------------------------------------------

    def _prepare_multiplexing_strategy(self):
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
        selected = copy.deepcopy(random.choice(strategies))
        selected["progress"] = []
        print(f"[CONN {id(self)}][HTTP2] batching strategy", selected)
        return selected

    def _check_prepare_multiplexing(self, parent_stream_id: int):
        if not self.use_multiplexing:
            return

        self.pending_requests_cnt = len(self.conn_client_database)
        strategy = self._prepare_multiplexing_strategy()
        if strategy["ckp_pct"] == "all":
            ckp = list(range(self.pending_requests_cnt + 1))
        else:
            ckp = [int(p * self.pending_requests_cnt) for p in strategy["ckp_pct"]]

        ckp = np.asarray(ckp)
        ckp = sorted(set(ckp[ckp != 0].tolist()))
        strategy["checkpoints"] = ckp
        self.multiplexing_strategy = strategy
        print("[HTTP2] Batching", strategy)

    def _flush_pending_streams(self):
        for _ in range(len(self.pending_streams)):
            if not self.pending_streams:
                break
            self.stream_complete(self.pending_streams.pop(0))

    def _check_handle_multiplexing(self):
        # a connection whose first request disabled defenses has no strategy
        if not self.multiplexing_strategy:
            return self._flush_pending_streams()

        progress = self.multiplexing_strategy["progress"]
        checkpoints = self.multiplexing_strategy["checkpoints"]
        last_idx = len(progress)
        already = progress[-1] if progress else 0
        current = already + len(self.pending_streams)

        if last_idx >= len(checkpoints):
            return self._flush_pending_streams()

        next_ckp = checkpoints[last_idx]
        # was an assert; a client that opens more streams than the trace
        # predicted would abort the connection instead of just flushing
        if next_ckp > self.pending_requests_cnt or current > next_ckp:
            return self._flush_pending_streams()
        if current != next_ckp:
            return

        progress.append(next_ckp)
        random.shuffle(self.pending_streams)
        return self._flush_pending_streams()

    # ------------------------------------------------------------------
    # proactive resource suggestion
    # ------------------------------------------------------------------

    def _generate_103_early_hints(self, stream_id):
        pending_requests = self.inject_hints
        if not pending_requests:
            return

        hints_headers = [(":status", "103")]
        for req in pending_requests:
            hints_headers.append(
                ("link", f"<{req['url_local']}>; rel=preload; as=image")
            )

        print(f"[CONN {id(self)}][HTTP2] HINTS103 {len(hints_headers) - 1} links")
        try:
            self.conn.send_headers(stream_id, hints_headers)
        except (StreamClosedError, ProtocolError) as exc:
            print("103 hints failed", exc)

    def _generate_server_push_headers(self, parent_stream_id: int, path: str):
        promised_stream_id = self.conn.get_next_available_stream_id()
        data, _, content_type, _ = self._generate_response_data(path)

        self.conn.push_stream(
            parent_stream_id,
            promised_stream_id,
            [
                (":method", "GET"),
                (":authority", "localhost"),
                (":scheme", "https"),
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
        self, parent_stream_id, promised_stream_id, path, data
    ):
        asyncio.ensure_future(self.send_data(data, promised_stream_id))
        print(f" >>> push {path} stream={promised_stream_id} data={len(data)}")
        return promised_stream_id

    def _prepare_server_push(self, parent_stream_id: int):
        if self.use_server_push:
            pending_requests = copy.deepcopy(self.conn_client_database[1:])
        elif self.use_rnd_server_push:
            pending_requests = list(self.inject_hints)
        else:
            return []

        if len(pending_requests) < 2:
            return []

        push_streams_cnt = random.randrange(1, min(PUSH_MAX, len(pending_requests)))
        random.shuffle(pending_requests)
        print(
            f"[CONN {id(self)}][HTTP2] pushing {push_streams_cnt}/"
            f"{len(pending_requests)} at random"
        )

        push_streams = []
        for req in pending_requests:
            try:
                push_streams.append(
                    self._generate_server_push_headers(
                        parent_stream_id, req["url_local"]
                    )
                )
            except (StreamClosedError, ProtocolError) as exc:
                print("push failed", exc)
                break
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
coro = loop.create_server(H2Protocol, SERVER_IP, SERVER_PORT, ssl=ssl_context)
server = loop.run_until_complete(coro)

print(f"Serving on {server.sockets[0].getsockname()}")
try:
    loop.run_forever()
except KeyboardInterrupt:
    pass

server.close()
loop.run_until_complete(server.wait_closed())
loop.close()
