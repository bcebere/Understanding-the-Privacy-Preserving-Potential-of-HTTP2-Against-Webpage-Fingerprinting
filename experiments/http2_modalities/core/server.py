# stdlib
from argparse import ArgumentParser
import asyncio
import collections
import copy
import io
import json
import os
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
args = parser.parse_args()

assert args.dst_port is not None

SERVER_IP = "0.0.0.0"
SERVER_PORT = args.dst_port  # 8443
DATA_PATH = Path("data")
SERVER_PATH = Path(__file__).resolve().parent
print(SERVER_PATH)

try:
    USE_SERVER_PUSH = int(os.environ.get("HTTP2_WF_USE_SERVER_PUSH", 0))
except BaseException:
    USE_SERVER_PUSH = 0

try:
    USE_MULTIPLEXING_RANDOM = int(os.environ.get("HTTP2_WF_USE_MULTIPLEXING_RANDOM", 0))
except BaseException:
    USE_MULTIPLEXING_RANDOM = 0

try:
    USE_RANDOM_HPACK = int(os.environ.get("HTTP2_WF_USE_RANDOM_HPACK", 0))
except BaseException:
    USE_RANDOM_HPACK = 0

try:
    USE_HINTS103 = int(os.environ.get("HTTP2_WF_USE_HINTS103", 0))
except BaseException:
    USE_HINTS103 = 0


RequestData = collections.namedtuple("RequestData", ["headers", "data"])

with open(DATA_PATH / "server_db.json") as f:
    SERVER_DB = json.load(f)
with open(DATA_PATH / "client_db.json") as f:
    CLIENT_DB = json.load(f)

print(
    f"""
      Test config:
        Server Push: {USE_SERVER_PUSH}
        Multiplexing Agg: {USE_MULTIPLEXING_RANDOM}
        RANDOM HPACK: {USE_RANDOM_HPACK}
        HINTS103: {USE_HINTS103}
"""
)


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
        if USE_RANDOM_HPACK:
            self.table_size = 2 ** random.randrange(14)
            self.server_name = "http2-mock;" * (int(self.table_size / 100) + 1)
        else:
            self.table_size = 4096
            self.server_name = "http2-mock;"
        print("Headers", self.table_size, self.server_name)

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
                    if USE_MULTIPLEXING_RANDOM:
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
                        print(event.changed_settings)
                        self.window_updated(None, 0)
                elif isinstance(event, PingReceived):
                    self.handle_ping(event)

                self.transport.write(self.conn.data_to_send())

    def handle_ping(self, event):
        self.conn.ping(event.ping_data)
        self.transport.write(self.conn.data_to_send())

    def request_received(self, headers: List[Tuple[str, str]], stream_id: int):
        headers = collections.OrderedDict(headers)

        # Store off the request data.
        request_data = RequestData(headers, io.BytesIO())
        self.stream_data[stream_id] = request_data

        path = headers[":path"]
        if stream_id == 1:
            self._check_prepare_multiplexing(stream_id, path)
            if USE_HINTS103:
                self._generate_103_early_hints(stream_id, path)

    def _generate_response_data(self, path: str, headers={}):
        if path not in SERVER_DB:
            return {}

        response_size = 0
        if "range" in headers:
            range_bytes = headers["range"].split("bytes=")[-1]
            start, end = range_bytes.split("-")
            response_size = int(end) - int(start)

        db_data = SERVER_DB[path]
        timeout = 0
        if "data_delay" in db_data:
            if isinstance(db_data["data_delay"], (int, float)):
                timeout = db_data["data_delay"]
            elif db_data["data_delay"] == "random":
                timeout = random.uniform(0.001, 0.1)
            else:
                timeout = 0
        if timeout > 0:
            time.sleep(timeout)

        body = {}
        if response_size > 0:
            body = "".join(
                random.choices(string.ascii_uppercase + string.digits, k=response_size)
            )
        elif "data" in db_data:
            body = db_data["data"]
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
            body = {}
        print(f"[DATA] path={path} timeout={timeout} size={response_size}")

        return body

    def _generate_server_push_headers(self, parent_stream_id: int, path: str):
        # Push a resource
        promised_stream_id = self.conn.get_next_available_stream_id()

        push_data = self._generate_response_data(path)
        data = json.dumps(
            {"body": push_data},
            indent=4,
        ).encode("utf8")

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
                ("content-type", " application/json"),
                ("server", self.server_name),
                ("content-length", str(len(data))),
            ],
        )
        return promised_stream_id, data

    def _prepare_multiplexing_strategy(self):
        if USE_MULTIPLEXING_RANDOM:
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

        return selected

    def _check_prepare_multiplexing(self, parent_stream_id: int, prev_path: str):
        if not USE_MULTIPLEXING_RANDOM:
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
        print("Multiplexing", self.multiplexing_strategy)

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
        if path not in SERVER_DB:
            return headers

        db_data = SERVER_DB[path]
        if db_data["data_size"] == "random":
            return headers

        headers = [
            (":status", "200"),
            ("content-type", "text/plain"),
            ("content-length", f"{db_data['data_size']}"),  # No body content for HEAD
        ]
        return headers

    def _generate_103_early_hints(self, stream_id, prev_path: str):
        pending_requests = CLIENT_DB[prev_path]["requests"][1:]
        hints_headers = [
            (":status", "103"),
            # ('link', '</static/style.css>; rel=preload; as=style'),
            # ('link', '</static/script.js>; rel=preload; as=script')
        ]
        for idx, req in enumerate(pending_requests):
            req_path = req["path"]
            hints_headers.append(("link", f"<{req_path}>; rel=preload; as=image"))

        self.conn.send_headers(stream_id, hints_headers)

    def _generate_server_push_data(
        self, parent_stream_id: int, promised_stream_id: int, path: str, data: bytes
    ):
        asyncio.ensure_future(self.send_data(data, promised_stream_id))
        print(f" >>> push data {path} stream={promised_stream_id} data={len(data)}")
        return promised_stream_id

    def _check_prepare_server_push(self, parent_stream_id: int, prev_path: str):
        if USE_SERVER_PUSH != 1:
            return []

        if prev_path not in CLIENT_DB:
            return []

        pending_requests = CLIENT_DB[prev_path]["requests"][1:]
        push_streams = []
        for req in pending_requests:
            req_path = req["path"]
            push_streams.append(
                self._generate_server_push_headers(parent_stream_id, req_path)
            )

        return push_streams

    def _check_server_push_data(
        self, parent_stream_id: int, prev_path: str, push_streams: list
    ):
        if not USE_SERVER_PUSH:
            return

        if prev_path not in CLIENT_DB:
            return

        pending_requests = CLIENT_DB[prev_path]["requests"][1:]
        for idx, req in enumerate(pending_requests):
            req_path = req["path"]
            push_stream_id, push_data = push_streams[idx]
            self._generate_server_push_data(
                parent_stream_id, push_stream_id, req_path, push_data
            )

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

        body = self._generate_response_data(path, headers)

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
        push_streams = []
        if stream_id == 1:
            push_streams = self._check_prepare_server_push(stream_id, path)

        asyncio.ensure_future(self.send_data(data, stream_id))

        if len(push_streams) > 0:
            self._check_server_push_data(stream_id, path, push_streams)

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

    async def send_data(self, data, stream_id):
        """
        Send data according to the flow control rules.
        """
        while data:
            while self.conn.local_flow_control_window(stream_id) < 1:
                print(
                    "waiting flow control",
                    stream_id,
                    self.conn.local_flow_control_window(stream_id),
                )
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
