"""
Reusable HTTP/2 server with optional traffic-shaping / defense features.
All experimental behaviour is exposed through :class:`H2ServerConfig`:

  * server push — real (driven by the resource store) or noise (synthetic)
  * 103 Early Hints — real or noise
  * response padding — fixed or random block sizes
  * HPACK cache-busting via a huge ``server`` header
  * stream multiplexing / batching strategies
  * per-frame random delays and data-volume traffic shaping
  * random outbound frame size
  * connection-scoped opt-out via the ``defend_connection`` request header
  * Tamaraw / ALPaCA presets for convenience
"""

# future
from __future__ import annotations

# stdlib
import abc
import asyncio
import collections
import copy
from dataclasses import dataclass, field
import io
from pathlib import Path
import random
import socket
import ssl
import string
import time
from typing import Callable, Dict, List, Optional, Tuple

# third party
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

# ====================================================================== #
# Data types                                                              #
# ====================================================================== #


@dataclass
class ResponseSpec:
    """What the resource store returns for a given path."""

    body: bytes
    content_type: str = "application/json"
    headers: Dict[str, str] = field(default_factory=dict)
    response_delay: float = 0.0  # seconds to sleep before first byte


RequestData = collections.namedtuple("RequestData", ["headers", "data"])


# ====================================================================== #
# Resource store                                                          #
# ====================================================================== #


class ResourceStore(abc.ABC):
    """
    Abstract map of ``path -> ResponseSpec`` that the server serves from.

    Implementations must be able to:

      * look up the response for a path (``get``)
      * list paths related to a given scope so the server can push/hint them
        (``related_paths``). The scope is typically a ``connection_id`` but
        the library treats it opaquely.
      * dynamically register synthetic resources (``put``) — the server uses
        this to inject noise when the random-push / random-hints features
        are enabled.
    """

    @abc.abstractmethod
    def get(self, path: str) -> Optional[ResponseSpec]:
        ...

    @abc.abstractmethod
    def put(self, path: str, spec: ResponseSpec) -> None:
        ...

    @abc.abstractmethod
    def related_paths(self, scope: Optional[str] = None) -> List[str]:
        ...


class InMemoryResourceStore(ResourceStore):
    """Trivial ``dict``-backed store, useful for simple servers and tests."""

    def __init__(self, initial: Optional[Dict[str, ResponseSpec]] = None):
        self._resources: Dict[str, ResponseSpec] = dict(initial or {})

    def get(self, path: str) -> Optional[ResponseSpec]:
        return self._resources.get(path)

    def put(self, path: str, spec: ResponseSpec) -> None:
        self._resources[path] = spec

    def related_paths(self, scope: Optional[str] = None) -> List[str]:
        return list(self._resources.keys())


#: Signature of a factory that builds a store from the first request's headers.
#: The experimental layer typically provides one of these so the server can
#: pick the right JSON database based on the ``label`` / ``connection_id``.
StoreFactory = Callable[[Dict[str, str]], ResourceStore]


# ====================================================================== #
# Configuration                                                           #
# ====================================================================== #


@dataclass
class H2ServerConfig:
    """Toggles for every experimental / defensive feature."""

    # ---- Server push ------------------------------------------------- #
    enable_server_push: bool = False
    enable_random_server_push: bool = False

    # ---- 103 Early Hints -------------------------------------------- #
    enable_103_hints: bool = False
    enable_random_103_hints: bool = False
    # If True, enables 103 hints even when the client did not ask for them
    # (equivalent to the old ``http2_global_hints103`` flag).
    enable_global_103_hints: bool = False
    hints_count_lo: int = 1
    hints_count_hi: int = 5

    # ---- Multiplexing / response batching --------------------------- #
    enable_multiplexing_batching: bool = False

    # ---- HPACK cache-busting ---------------------------------------- #
    enable_hpack_cache_bust: bool = False

    # ---- Response padding ------------------------------------------- #
    # ``enable_fixed_padding`` always pads; ``enable_random_padding`` pads
    # with 50% probability per request. Set at most one. ``pad_constant``
    # is the block size; if None a random value is chosen per connection.
    enable_random_padding: bool = False
    enable_fixed_padding: bool = False
    pad_constant: Optional[int] = None
    random_pad_range: Tuple[int, int] = (128, 1024)

    # ---- Frame-level traffic shaping -------------------------------- #
    # If ``enable_random_frame_delay`` is on, each response randomly picks
    # a per-frame delay and per-connection byte threshold (old behaviour).
    # Alternatively, ``fixed_frame_delay`` + ``fixed_frame_threshold`` set
    # both statically (used by the Tamaraw preset).
    enable_random_frame_delay: bool = False
    fixed_frame_delay: Optional[float] = None
    fixed_frame_threshold: Optional[int] = None

    # ---- Flow control ----------------------------------------------- #
    enable_random_out_window: bool = False
    fixed_out_window_size: Optional[int] = None

    # ---- Pings ------------------------------------------------------ #
    enable_random_pings: bool = False

    # ---- Noise injection sizing (for random push / random hints) ---- #
    noise_hint_count: int = 15
    noise_hint_size_lo: int = 1001
    noise_hint_size_hi: int = 10_000

    # ---- Per-connection control ------------------------------------- #
    # If True, the request header ``defend_connection: 0`` disables every
    # feature above for that connection. If False, headers are ignored.
    respect_defend_connection_header: bool = True

    # ------------------------------------------------------------------ #
    # Presets                                                            #
    # ------------------------------------------------------------------ #
    @classmethod
    def tamaraw(cls) -> "H2ServerConfig":
        """Tamaraw: constant-rate send with fixed padding and random push."""
        return cls(
            enable_random_server_push=True,
            enable_fixed_padding=True,
            pad_constant=8092,
            fixed_frame_delay=0.001,
            fixed_frame_threshold=4096,
            fixed_out_window_size=2048,
        )

    @classmethod
    def alpaca(cls) -> "H2ServerConfig":
        """ALPaCA: random padding + random server push."""
        return cls(
            enable_random_server_push=True,
            enable_random_padding=True,
        )


# ====================================================================== #
# H2Server                                                                #
# ====================================================================== #


class H2Server(asyncio.Protocol):
    """
    HTTP/2 server protocol. Construct via a factory::

        factory = H2Server.factory(store_factory, config)
        loop.create_server(factory, host, port, ssl=ssl_ctx)

    or the convenience helper :func:`run_server`.
    """

    # Tunables that rarely need adjustment
    HEADER_TABLE_SIZE = 4096
    SO_SNDBUF = 8 * 1024 * 1024

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def factory(
        cls,
        store_factory: StoreFactory,
        config: Optional[H2ServerConfig] = None,
    ) -> Callable[[], "H2Server"]:
        """Build a zero-arg protocol factory for ``loop.create_server``."""
        cfg = config or H2ServerConfig()
        return lambda: cls(store_factory, cfg)

    def __init__(
        self,
        store_factory: StoreFactory,
        config: Optional[H2ServerConfig] = None,
    ):
        self.store_factory = store_factory
        self.config = config or H2ServerConfig()

        h2cfg = H2Configuration(client_side=False, header_encoding="utf-8")
        self.conn: H2Connection = H2Connection(config=h2cfg)
        self.transport: Optional[asyncio.Transport] = None

        # Stream bookkeeping
        self.stream_data: Dict[int, RequestData] = {}
        self.flow_control_futures: Dict[int, asyncio.Future] = {}

        # Transport backpressure
        self._can_write = asyncio.Event()
        self._can_write.set()

        # Lazily populated on first request
        self.store: Optional[ResourceStore] = None
        self.connection_id: Optional[str] = None
        self.defense_active: bool = True  # toggled by defend_connection header
        self._noise_paths: List[str] = []

        # ---- Connection-scoped state from config --------------------- #
        self.server_name = "mock"  # may be inflated by HPACK bust
        self.pad_constant = (
            self.config.pad_constant
            if self.config.pad_constant is not None
            else random.randint(*self.config.random_pad_range)
        )

        # Outbound frame-size choice
        if self.config.fixed_out_window_size is not None:
            self.max_outbound_frame_size = self.config.fixed_out_window_size
        elif self.config.enable_random_out_window:
            self.max_outbound_frame_size = random.randint(2**12, 2**14)
            print(f"[h2deflib][out-window] {self.max_outbound_frame_size}")
        else:
            self.max_outbound_frame_size = 16384

        # Traffic-shaping state
        self.frame_delay_threshold: Optional[int] = self.config.fixed_frame_threshold
        self.frame_delay: Optional[float] = self.config.fixed_frame_delay
        self.send_data_counter = 0

        # Multiplexing / batching state
        self.pending_streams: List[int] = []
        self.pending_requests_cnt = 0
        self.multiplexing_strategy: Optional[dict] = None

    # ------------------------------------------------------------------ #
    # asyncio.Protocol                                                   #
    # ------------------------------------------------------------------ #

    def connection_made(self, transport: asyncio.Transport):
        self.transport = transport
        sock = transport.get_extra_info("socket")
        if sock is not None:
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.SO_SNDBUF)
            except OSError:
                pass
        self.conn.initiate_connection()
        self.conn.update_settings(
            {SettingCodes.HEADER_TABLE_SIZE: self.HEADER_TABLE_SIZE}
        )
        self._flush(swallow=True)

    def connection_lost(self, exc):
        for f in self.flow_control_futures.values():
            f.cancel()
        self.flow_control_futures.clear()

    def pause_writing(self):
        self._can_write.clear()

    def resume_writing(self):
        self._can_write.set()

    def data_received(self, data: bytes):
        try:
            events = self.conn.receive_data(data)
        except ProtocolError:
            self._flush(swallow=True)
            self.transport.close()
            return

        self._flush(swallow=True)
        for event in events:
            try:
                self._dispatch_event(event)
            except Exception as exc:  # noqa: BLE001
                print(f"[h2deflib] error handling {type(event).__name__}: {exc}")
            self._flush(swallow=True)

    # ------------------------------------------------------------------ #
    # Event dispatch                                                     #
    # ------------------------------------------------------------------ #

    def _dispatch_event(self, event):
        if isinstance(event, RequestReceived):
            self._on_request_received(event.stream_id, event.headers)
        elif isinstance(event, DataReceived):
            self._on_request_data(event.stream_id, event.data)
        elif isinstance(event, StreamEnded):
            if self.defense_active and self.config.enable_multiplexing_batching:
                self.pending_streams.append(event.stream_id)
                self._check_handle_multiplexing()
            else:
                self._stream_complete(event.stream_id)
        elif isinstance(event, StreamReset):
            self._cancel_stream(event.stream_id)
        elif isinstance(event, WindowUpdated):
            self._window_updated(event.stream_id, event.delta)
        elif isinstance(event, RemoteSettingsChanged):
            if SettingCodes.INITIAL_WINDOW_SIZE in event.changed_settings:
                self._window_updated(None, 0)
        elif isinstance(event, ConnectionTerminated):
            self.transport.close()
        elif isinstance(event, (PingReceived, PingAckReceived)):
            pass  # h2 auto-acks pings

    def _on_request_received(self, stream_id: int, raw_headers):
        headers = collections.OrderedDict(raw_headers)

        # Per-connection config: first request sets it up.
        if self.store is None:
            self._configure_from_first_request(headers)

        # Per-connection defense opt-out via header
        if (
            self.config.respect_defend_connection_header
            and "defend_connection" in headers
        ):
            self.defense_active = bool(int(headers["defend_connection"]))

        if not self.defense_active:
            # For this connection, every feature is off.
            pass  # handled by guards throughout

        request_data = RequestData(headers=headers, data=io.BytesIO())
        self.stream_data[stream_id] = request_data

        if stream_id == 1:
            self._on_first_stream(stream_id)

    def _configure_from_first_request(self, headers: Dict[str, str]):
        self.connection_id = headers.get("connection_id")
        self.store = self.store_factory(headers)

        # HPACK cache busting: inflate the ``server:`` header once per conn
        if (
            self.defense_active
            and self.config.enable_hpack_cache_bust
            and random.choice([True, False])
        ):
            big = random.randint(2**12 + 1, 2**14)
            self.server_name = "#" * big
            print(f"[h2deflib][hpack] big server header = {big}")

        # Random frame-delay threshold (if enabled dynamically)
        if self.defense_active and self.config.enable_random_frame_delay:
            self.frame_delay_threshold = random.randint(2**8, 2**10)
            print(f"[h2deflib][shape] threshold = {self.frame_delay_threshold}")

        # Inject synthetic noise resources for random-push / random-hints.
        if self.defense_active and (
            self.config.enable_random_server_push or self.config.enable_random_103_hints
        ):
            self._noise_paths = self._inject_noise_resources()

    def _on_first_stream(self, stream_id: int):
        if self.defense_active and self.config.enable_multiplexing_batching:
            self._prepare_multiplexing()

        hints_on = self.config.enable_103_hints or self.config.enable_global_103_hints
        if self.defense_active and (hints_on or self.config.enable_random_103_hints):
            self._send_103_early_hints(stream_id)

    def _on_request_data(self, stream_id: int, data: bytes):
        stream = self.stream_data.get(stream_id)
        if stream is None:
            self.conn.reset_stream(stream_id, error_code=ErrorCodes.PROTOCOL_ERROR)
            return
        stream.data.write(data)

    # ------------------------------------------------------------------ #
    # Stream completion / response                                       #
    # ------------------------------------------------------------------ #

    def _stream_complete(self, stream_id: int):
        request_data = self.stream_data.get(stream_id)
        if request_data is None:
            return

        path = request_data.headers[":path"]
        (
            body,
            resp_headers,
            content_type,
            response_delay,
            frame_delay,
        ) = self._build_response(path, request_data.headers)

        response_headers = [
            (":status", "200"),
            ("content-type", content_type),
            ("content-length", str(len(body))),
            ("server", self.server_name),
        ]
        for k, v in resp_headers.items():
            response_headers.append((k, v))

        self.conn.send_headers(stream_id, response_headers)

        # Server push: always on stream 1, randomly on other streams if noise mode.
        push_streams: List[tuple] = []
        if stream_id == 1 and self.defense_active:
            push_streams = self._prepare_server_push(stream_id)
        elif (
            self.defense_active
            and self.config.enable_random_server_push
            and random.choice([True, False])
        ):
            push_streams = self._prepare_server_push(stream_id)

        asyncio.ensure_future(
            self._send_data(body, stream_id, response_delay, frame_delay)
        )
        for req_path, promised_id, data, r_delay, f_delay in push_streams:
            asyncio.ensure_future(self._send_data(data, promised_id, r_delay, f_delay))

    def _build_response(
        self, path: str, request_headers: Dict[str, str]
    ) -> Tuple[bytes, Dict[str, str], str, float, float]:
        spec = self.store.get(path) if self.store else None
        if spec is None:
            print(f"[h2deflib] path not in store: {path}")
            return b"", {}, "application/text", 0.0, 0.0

        body = spec.body
        content_type = spec.content_type
        response_delay = spec.response_delay
        resp_headers = dict(spec.headers or {})

        # HTTP Range support
        if "range" in request_headers:
            try:
                start, end = request_headers["range"].split("bytes=")[-1].split("-")
                body = body[int(start) : int(end)]
            except ValueError:
                pass  # malformed range; ignore

        # Response padding
        if self.defense_active and (
            self.config.enable_fixed_padding or self.config.enable_random_padding
        ):
            should_pad = self.config.enable_fixed_padding or random.choice(
                [True, False]
            )
            if should_pad and (
                self.config.enable_fixed_padding or len(body) % self.pad_constant != 0
            ):
                pad_size = (len(body) // self.pad_constant + 1) * self.pad_constant
                extra = pad_size - len(body)
                padding = "".join(
                    random.choices(string.ascii_uppercase + string.digits, k=extra)
                ).encode("utf-8")
                body = body + padding

        # Per-frame random delay
        frame_delay = 0.0
        if (
            self.defense_active
            and self.config.enable_random_frame_delay
            and random.choice([True, False])
        ):
            frame_delay = random.uniform(0.001, 0.01)
        elif self.defense_active and self.config.fixed_frame_delay is not None:
            frame_delay = self.config.fixed_frame_delay

        print(
            f"[h2deflib][data] path={path} delay={response_delay} "
            f"frame_delay={frame_delay} size={len(body)}"
        )
        return body, resp_headers, content_type, response_delay, frame_delay

    # ------------------------------------------------------------------ #
    # 103 Early Hints                                                    #
    # ------------------------------------------------------------------ #

    def _send_103_early_hints(self, stream_id: int):
        candidates = self._hint_candidates()
        if not candidates:
            return

        k = random.randint(self.config.hints_count_lo, self.config.hints_count_hi)
        sampled = random.choices(candidates, k=k)

        hints_headers = [(":status", "103")]
        for path in sampled:
            hints_headers.append(("link", f"<{path}>; rel=preload; as=image"))

        print(f"[h2deflib][103] sending {len(sampled)} hints")
        self.conn.send_headers(stream_id, hints_headers)

    def _hint_candidates(self) -> List[str]:
        if self.config.enable_random_103_hints:
            return list(self._noise_paths) or self._related_paths_minus_first()
        return self._related_paths_minus_first()

    # ------------------------------------------------------------------ #
    # Server push                                                        #
    # ------------------------------------------------------------------ #

    def _prepare_server_push(self, parent_stream_id: int) -> List[tuple]:
        if self.config.enable_server_push:
            candidates = self._related_paths_minus_first()
        elif self.config.enable_random_server_push:
            candidates = list(self._noise_paths)
        else:
            return []

        if len(candidates) < 2:
            return []

        count = random.randrange(1, len(candidates))
        candidates = list(candidates)
        random.shuffle(candidates)
        print(f"[h2deflib][push] pushing {count}/{len(candidates)}")

        streams: List[tuple] = []
        for path in candidates[:count]:
            pushed = self._push_headers(parent_stream_id, path)
            if pushed is not None:
                streams.append(pushed)
        return streams

    def _push_headers(self, parent_stream_id: int, path: str) -> Optional[tuple]:
        promised_id = self.conn.get_next_available_stream_id()
        body, resp_hdrs, content_type, r_delay, f_delay = self._build_response(path, {})
        try:
            self.conn.push_stream(
                parent_stream_id,
                promised_id,
                [
                    (":method", "GET"),
                    (":authority", "localhost"),
                    (":scheme", "http"),
                    (":path", path),
                ],
            )
            self.conn.send_headers(
                promised_id,
                [
                    (":status", "200"),
                    ("content-type", content_type),
                    ("server", self.server_name),
                    ("content-length", str(len(body))),
                ],
            )
        except ProtocolError:
            return None
        return (path, promised_id, body, r_delay, f_delay)

    # ------------------------------------------------------------------ #
    # Multiplexing / batching                                            #
    # ------------------------------------------------------------------ #

    _MUX_STRATEGIES = [
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

    def _prepare_multiplexing(self):
        related = self._related_paths_minus_first()
        # "pending requests" is the number of related client requests we
        # anticipate (matches the original code's use of conn_client_database)
        self.pending_requests_cnt = len(related) + 1
        strategy = copy.deepcopy(random.choice(self._MUX_STRATEGIES))
        strategy["progress"] = []

        if strategy["ckp_pct"] == "all":
            ckp = list(range(self.pending_requests_cnt + 1))
        else:
            ckp = [int(p * self.pending_requests_cnt) for p in strategy["ckp_pct"]]

        ckp = sorted({c for c in ckp if c != 0})
        strategy["checkpoints"] = ckp
        self.multiplexing_strategy = strategy
        print(f"[h2deflib][mux] {strategy}")

    def _check_handle_multiplexing(self):
        if self.multiplexing_strategy is None:
            return self._flush_pending_streams()

        progress = self.multiplexing_strategy["progress"]
        checkpoints = self.multiplexing_strategy["checkpoints"]

        if len(progress) >= len(checkpoints):
            return self._flush_pending_streams()

        already_flushed = progress[-1] if progress else 0
        avail = already_flushed + len(self.pending_streams)
        next_ckp = checkpoints[len(progress)]

        if avail != next_ckp:
            return

        progress.append(next_ckp)
        random.shuffle(self.pending_streams)
        self._flush_pending_streams()

    def _flush_pending_streams(self):
        while self.pending_streams:
            self._stream_complete(self.pending_streams.pop(0))

    # ------------------------------------------------------------------ #
    # Noise injection                                                    #
    # ------------------------------------------------------------------ #

    def _inject_noise_resources(self) -> List[str]:
        existing_sizes = []
        for p in self._related_paths_minus_first():
            spec = self.store.get(p)
            if spec is not None:
                existing_sizes.append(len(spec.body))

        lo = self.config.noise_hint_size_lo
        hi = self.config.noise_hint_size_hi
        count = self.config.noise_hint_count
        if existing_sizes:
            lo = max(lo, max(existing_sizes) // 2)
            hi = max(hi, 2 * max(existing_sizes) + 1)
            count = max(count, len(existing_sizes))

        injected: List[str] = []
        for _ in range(count):
            size = random.randint(lo, hi)
            path = f"/inject_{size}B"
            body = "".join(
                random.choices(string.ascii_uppercase + string.digits, k=size)
            ).encode("utf-8")
            self.store.put(
                path,
                ResponseSpec(
                    body=body,
                    content_type="image/png",
                    response_delay=random.uniform(0, 1),
                ),
            )
            injected.append(path)
        return injected

    # ------------------------------------------------------------------ #
    # Flow-controlled data send                                          #
    # ------------------------------------------------------------------ #

    async def _send_data(
        self,
        data: bytes,
        stream_id: int,
        response_delay: float = 0.0,
        frame_delay: float = 0.0,
    ):
        if response_delay > 0:
            await asyncio.sleep(response_delay)
            print(
                f"[h2deflib][stream {stream_id}] delay {response_delay} at {time.time()}"
            )

        if len(data) == 0:
            try:
                self.conn.send_data(stream_id, data, end_stream=True)
                self._flush(swallow=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[h2deflib] empty send failed: {exc}")
            return

        while data:
            await self._can_write.wait()

            while self.conn.local_flow_control_window(stream_id) < 1:
                try:
                    await self._wait_for_flow_control(stream_id)
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
                return

            try:
                self._flush()
            except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                print(f"[h2deflib] write failed on stream {stream_id}: {exc}")
                return

            data = data[chunk_size:]

            # Traffic shaping
            if (
                self.defense_active
                and frame_delay > 0
                and self.frame_delay_threshold is not None
                and self.send_data_counter > self.frame_delay_threshold
            ):
                self.send_data_counter = 0
                await asyncio.sleep(frame_delay)

    async def _wait_for_flow_control(self, stream_id: int):
        f: asyncio.Future = asyncio.Future()
        self.flow_control_futures[stream_id] = f
        await f

    def _window_updated(self, stream_id: Optional[int], delta: int):
        if stream_id and stream_id in self.flow_control_futures:
            self.flow_control_futures.pop(stream_id).set_result(delta)
        elif not stream_id:
            for f in self.flow_control_futures.values():
                f.set_result(delta)
            self.flow_control_futures.clear()

    def _cancel_stream(self, stream_id: int):
        f = self.flow_control_futures.pop(stream_id, None)
        if f is not None:
            f.cancel()

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _related_paths_minus_first(self) -> List[str]:
        if self.store is None:
            return []
        paths = self.store.related_paths(self.connection_id)
        # The first path is the "main" resource the client is about to request,
        # so it should not be offered back as a push/hint.
        return list(paths[1:]) if len(paths) > 1 else []

    def _flush(self, swallow: bool = False):
        """Push any pending bytes to the transport."""
        buf = self.conn.data_to_send()
        if not buf:
            return
        try:
            self.transport.write(buf)
        except (BrokenPipeError, ConnectionResetError, OSError):
            if not swallow:
                raise


# ====================================================================== #
# TLS + server runner                                                     #
# ====================================================================== #


def make_server_ssl_context(
    certfile: str | Path, keyfile: str | Path
) -> ssl.SSLContext:
    """Build an H2-ready server SSL context (TLS 1.3+)."""
    ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    # Require TLS 1.3. Replaces the deprecated
    # ``options |= OP_NO_TLSv1 | OP_NO_TLSv1_1 | OP_NO_TLSv1_2`` bitmask.
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.load_cert_chain(certfile=str(certfile), keyfile=str(keyfile))
    ctx.set_alpn_protocols(["h2"])
    return ctx


def run_server(
    protocol_factory: Callable[[], H2Server],
    host: str,
    port: int,
    ssl_context: ssl.SSLContext,
):
    """Blocking helper that runs an H2 server until Ctrl-C."""

    async def _serve():
        loop = asyncio.get_running_loop()
        server = await loop.create_server(protocol_factory, host, port, ssl=ssl_context)
        print(f"Serving on {server.sockets[0].getsockname()}")
        try:
            await server.serve_forever()
        except asyncio.CancelledError:
            pass
        finally:
            server.close()
            await server.wait_closed()

    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        pass
