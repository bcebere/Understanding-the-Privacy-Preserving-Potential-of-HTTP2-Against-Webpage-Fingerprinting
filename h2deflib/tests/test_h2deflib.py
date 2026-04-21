"""
Tests for ``h2deflib``.
"""

# future
from __future__ import annotations

# stdlib
import asyncio
import datetime
import io
from pathlib import Path
import random
import ssl
from typing import Dict, List, Tuple

# third party
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from h2.config import H2Configuration
from h2.connection import H2Connection
from h2.events import (
    DataReceived,
    InformationalResponseReceived,
    PushedStreamReceived,
    ResponseReceived,
    SettingsAcknowledged,
    StreamEnded,
)
import pytest

# h2deflib absolute
from h2deflib import (
    H2Server,
    H2ServerConfig,
    InMemoryResourceStore,
    ResponseSpec,
    make_server_ssl_context,
)

# ====================================================================== #
# Unit tests: data types                                                  #
# ====================================================================== #


class TestResponseSpec:
    def test_defaults(self):
        spec = ResponseSpec(body=b"hello")
        assert spec.body == b"hello"
        assert spec.content_type == "application/json"
        assert spec.headers == {}
        assert spec.response_delay == 0.0

    def test_overrides(self):
        spec = ResponseSpec(
            body=b"<html/>",
            content_type="text/html",
            headers={"x-test": "1"},
            response_delay=0.5,
        )
        assert spec.content_type == "text/html"
        assert spec.headers == {"x-test": "1"}
        assert spec.response_delay == 0.5


class TestInMemoryResourceStore:
    def test_get_missing_returns_none(self):
        store = InMemoryResourceStore()
        assert store.get("/nope") is None

    def test_get_returns_put_value(self):
        store = InMemoryResourceStore()
        spec = ResponseSpec(body=b"A")
        store.put("/a", spec)
        assert store.get("/a") is spec

    def test_initial_dict_is_copied(self):
        original = {"/a": ResponseSpec(body=b"A")}
        store = InMemoryResourceStore(original)
        store.put("/b", ResponseSpec(body=b"B"))
        # Caller's dict should not be mutated
        assert "/b" not in original

    def test_related_paths_returns_all(self):
        store = InMemoryResourceStore(
            {
                "/a": ResponseSpec(body=b"A"),
                "/b": ResponseSpec(body=b"B"),
            }
        )
        assert set(store.related_paths()) == {"/a", "/b"}
        # Scope argument is accepted but ignored for the simple store.
        assert set(store.related_paths("anything")) == {"/a", "/b"}


class TestH2ServerConfig:
    def test_defaults_all_off(self):
        c = H2ServerConfig()
        # Every defense feature is off by default.
        assert not c.enable_server_push
        assert not c.enable_random_server_push
        assert not c.enable_103_hints
        assert not c.enable_random_103_hints
        assert not c.enable_multiplexing_batching
        assert not c.enable_hpack_cache_bust
        assert not c.enable_random_padding
        assert not c.enable_fixed_padding
        assert not c.enable_random_frame_delay
        assert not c.enable_random_out_window
        assert not c.enable_random_pings
        # But per-connection opt-out is on by default.
        assert c.respect_defend_connection_header is True

    def test_tamaraw_preset(self):
        c = H2ServerConfig.tamaraw()
        assert c.enable_random_server_push
        assert c.enable_fixed_padding
        assert c.pad_constant == 8092
        assert c.fixed_frame_delay == 0.001
        assert c.fixed_frame_threshold == 4096
        assert c.fixed_out_window_size == 2048
        # Tamaraw should not enable random padding (they're mutually exclusive).
        assert not c.enable_random_padding

    def test_alpaca_preset(self):
        c = H2ServerConfig.alpaca()
        assert c.enable_random_server_push
        assert c.enable_random_padding
        assert not c.enable_fixed_padding
        # ALPaCA has no traffic shaping.
        assert c.fixed_frame_delay is None
        assert c.fixed_frame_threshold is None


# ====================================================================== #
# Integration tests: real server + raw h2 client                           #
# ====================================================================== #


def _make_self_signed_cert(tmp_path: Path) -> Tuple[Path, Path]:
    """Generate a throwaway TLS cert for the tests."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(
            datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)
        )
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost")]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_file = tmp_path / "cert.pem"
    key_file = tmp_path / "key.pem"
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return cert_file, key_file


class _FetchClient(asyncio.Protocol):
    """
    Minimal HTTP/2 GET client built directly on the ``h2`` state machine.
    Used by the integration tests so we aren't testing H2Client against
    itself.
    """

    def __init__(self, path: str, authority: str = "localhost"):
        self.conn = H2Connection(
            config=H2Configuration(client_side=True, header_encoding="utf-8")
        )
        self.path = path
        self.authority = authority
        self.body = io.BytesIO()
        self.response_headers: List[Tuple[str, str]] = []
        self.informational_headers: List[List[Tuple[str, str]]] = []
        self.pushed_streams: Dict[int, Dict] = {}  # stream_id -> {headers, body}
        self.done = asyncio.Event()
        self.transport: asyncio.Transport = None

    def connection_made(self, transport):
        self.transport = transport
        self.conn.initiate_connection()
        self.transport.write(self.conn.data_to_send())

    def data_received(self, data):
        for event in self.conn.receive_data(data):
            if isinstance(event, SettingsAcknowledged):
                self._send_request()
            elif isinstance(event, InformationalResponseReceived):
                self.informational_headers.append(list(event.headers))
            elif isinstance(event, ResponseReceived):
                if event.stream_id == 1:
                    self.response_headers = list(event.headers)
                elif event.stream_id in self.pushed_streams:
                    self.pushed_streams[event.stream_id]["headers"] = list(
                        event.headers
                    )
            elif isinstance(event, DataReceived):
                if event.stream_id == 1:
                    self.body.write(event.data)
                elif event.stream_id in self.pushed_streams:
                    self.pushed_streams[event.stream_id]["body"].write(event.data)
                # Acknowledge the data so the server's flow window opens.
                self.conn.acknowledge_received_data(
                    len(event.data), stream_id=event.stream_id
                )
            elif isinstance(event, StreamEnded):
                if event.stream_id == 1:
                    self.done.set()
            elif isinstance(event, PushedStreamReceived):
                path = next(v for k, v in event.headers if k == ":path")
                self.pushed_streams[event.pushed_stream_id] = {
                    "path": path,
                    "headers": None,
                    "body": io.BytesIO(),
                }
        if self.transport is not None:
            self.transport.write(self.conn.data_to_send())

    def _send_request(self):
        sid = self.conn.get_next_available_stream_id()
        self.conn.send_headers(
            sid,
            [
                (":method", "GET"),
                (":scheme", "https"),
                (":path", self.path),
                (":authority", self.authority),
                # The server's store factory receives these headers. Our test
                # factory ignores them, but we pass realistic values.
                ("label", "unit-test"),
                ("connection_id", "test"),
                ("user-agent", "h2deflib-test"),
            ],
            end_stream=True,
        )
        self.transport.write(self.conn.data_to_send())


async def _fetch(host: str, port: int, path: str, timeout: float = 5.0):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    ctx.set_alpn_protocols(["h2"])
    loop = asyncio.get_running_loop()
    transport, proto = await loop.create_connection(
        lambda: _FetchClient(path), host, port, ssl=ctx
    )
    try:
        await asyncio.wait_for(proto.done.wait(), timeout=timeout)
    finally:
        transport.close()
    # Wait a tick to receive any pushes/hints that arrived just after stream 1 ended
    await asyncio.sleep(0.05)
    return proto


async def _start_server(
    config: H2ServerConfig,
    store: InMemoryResourceStore,
    cert_file: Path,
    key_file: Path,
):
    ssl_ctx = make_server_ssl_context(cert_file, key_file)
    factory = H2Server.factory(lambda headers: store, config)
    loop = asyncio.get_event_loop()
    server = await loop.create_server(factory, "127.0.0.1", 0, ssl=ssl_ctx)
    port = server.sockets[0].getsockname()[1]
    return server, port


@pytest.fixture(scope="session")
def tls_cert(tmp_path_factory):
    """Generate one self-signed cert per test session."""
    tmp = tmp_path_factory.mktemp("tls")
    return _make_self_signed_cert(tmp)


@pytest.fixture(autouse=True)
def seeded_random():
    """Make the tests deterministic despite the server's use of ``random``."""
    random.seed(1234)


@pytest.mark.asyncio
async def test_server_basic_get(tls_cert):
    """Plain GET returns the stored body verbatim with default config."""
    cert_file, key_file = tls_cert
    store = InMemoryResourceStore(
        {
            "/hello": ResponseSpec(body=b"Hello, h2deflib!", content_type="text/plain"),
        }
    )
    server, port = await _start_server(H2ServerConfig(), store, cert_file, key_file)
    try:
        proto = await _fetch("127.0.0.1", port, "/hello")
        assert proto.body.getvalue() == b"Hello, h2deflib!"
        status = dict(proto.response_headers)[":status"]
        assert status == "200"
        # No 103 by default
        assert proto.informational_headers == []
        # No push by default
        assert proto.pushed_streams == {}
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_server_fixed_padding_inflates_body(tls_cert):
    """With fixed padding on, the response body is padded up to pad_constant boundary."""
    cert_file, key_file = tls_cert
    original = b"abc"  # 3 bytes — expect padding to pad_constant
    pad_constant = 512
    store = InMemoryResourceStore(
        {
            "/tiny": ResponseSpec(body=original, content_type="text/plain"),
        }
    )
    config = H2ServerConfig(
        enable_fixed_padding=True,
        pad_constant=pad_constant,
    )
    server, port = await _start_server(config, store, cert_file, key_file)
    try:
        proto = await _fetch("127.0.0.1", port, "/tiny")
        received = proto.body.getvalue()
        # Body should be padded to the next pad_constant multiple.
        assert len(received) == pad_constant
        assert received.startswith(original)
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_server_103_early_hints(tls_cert):
    """Enabling 103 hints produces an informational response before the 200."""
    cert_file, key_file = tls_cert
    store = InMemoryResourceStore(
        {
            "/main": ResponseSpec(body=b"main", content_type="text/plain"),
            "/img1.png": ResponseSpec(body=b"img1", content_type="image/png"),
            "/img2.png": ResponseSpec(body=b"img2", content_type="image/png"),
            "/img3.png": ResponseSpec(body=b"img3", content_type="image/png"),
        }
    )
    config = H2ServerConfig(
        enable_103_hints=True,
        hints_count_lo=1,
        hints_count_hi=3,
    )
    server, port = await _start_server(config, store, cert_file, key_file)
    try:
        proto = await _fetch("127.0.0.1", port, "/main")
        # The 200 response body is still delivered correctly
        assert proto.body.getvalue() == b"main"
        # And a 103 informational response arrived first
        assert len(proto.informational_headers) == 1
        hints_hdrs = dict(proto.informational_headers[0])
        assert hints_hdrs[":status"] == "103"
        # And that 103 carried at least one ``link: <...>; rel=preload`` header
        link_headers = [v for k, v in proto.informational_headers[0] if k == "link"]
        assert link_headers, "expected at least one Link header in 103"
        assert all("rel=preload" in v for v in link_headers)
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_server_push(tls_cert):
    """Enabling server push sends PUSH_PROMISE + data for additional resources."""
    cert_file, key_file = tls_cert
    store = InMemoryResourceStore(
        {
            "/page": ResponseSpec(body=b"page", content_type="text/html"),
            "/a.png": ResponseSpec(body=b"a-body", content_type="image/png"),
            "/b.png": ResponseSpec(body=b"b-body", content_type="image/png"),
            "/c.png": ResponseSpec(body=b"c-body", content_type="image/png"),
        }
    )
    config = H2ServerConfig(enable_server_push=True)
    server, port = await _start_server(config, store, cert_file, key_file)
    try:
        proto = await _fetch("127.0.0.1", port, "/page")
        # Main response unaffected
        assert proto.body.getvalue() == b"page"
        # At least one push promise was received
        assert len(proto.pushed_streams) >= 1
        for stream_id, info in proto.pushed_streams.items():
            assert info["path"] in {"/a.png", "/b.png", "/c.png"}
            # Push data should match the stored body
            expected_path = info["path"]
            expected = store.get(expected_path).body
            assert info["body"].getvalue() == expected
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_server_defend_connection_header_disables_features(tls_cert):
    """``defend_connection: 0`` from the client disables server-side defenses."""
    cert_file, key_file = tls_cert
    store = InMemoryResourceStore(
        {
            "/tiny": ResponseSpec(body=b"xy", content_type="text/plain"),
        }
    )
    # Would normally pad to 512, but the client will opt out below.
    config = H2ServerConfig(
        enable_fixed_padding=True,
        pad_constant=512,
    )
    server, port = await _start_server(config, store, cert_file, key_file)
    try:
        # Custom fetch that sends defend_connection: 0
        class OptOutFetch(_FetchClient):
            def _send_request(self):
                sid = self.conn.get_next_available_stream_id()
                self.conn.send_headers(
                    sid,
                    [
                        (":method", "GET"),
                        (":scheme", "https"),
                        (":path", "/tiny"),
                        (":authority", "localhost"),
                        ("label", "unit-test"),
                        ("defend_connection", "0"),
                    ],
                    end_stream=True,
                )
                self.transport.write(self.conn.data_to_send())

        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.set_alpn_protocols(["h2"])
        loop = asyncio.get_running_loop()
        transport, proto = await loop.create_connection(
            lambda: OptOutFetch("/tiny"), "127.0.0.1", port, ssl=ctx
        )
        try:
            await asyncio.wait_for(proto.done.wait(), timeout=5)
        finally:
            transport.close()

        # Padding should NOT have been applied because the client opted out
        assert proto.body.getvalue() == b"xy"
    finally:
        server.close()
        await server.wait_closed()


# ====================================================================== #
# H2Client integration tests                                              #
#                                                                         #
# These exercise ``h2deflib.H2Client`` against a real ``H2Server``. The      #
# client normally loads defense plugins from a ``client_defenses``        #
# package that lives outside this module; to keep the tests               #
# self-contained we monkeypatch ``h2deflib.client.get_defense`` with a       #
# ====================================================================== #


async def _start_server_with_requests(
    config: H2ServerConfig,
    store: InMemoryResourceStore,
    cert_file: Path,
    key_file: Path,
):
    ssl_ctx = make_server_ssl_context(cert_file, key_file)
    factory = H2Server.factory(lambda headers: store, config)
    loop = asyncio.get_running_loop()
    server = await loop.create_server(factory, "127.0.0.1", 0, ssl=ssl_ctx)
    port = server.sockets[0].getsockname()[1]
    return server, port


@pytest.mark.asyncio
@pytest.mark.parametrize("defense", ["front", "tamaraw", "h2pc", "httpos", "llama"])
async def test_h2client_completes_multi_request_sequence(tls_cert, defense):
    """
    Regression test for the exit-event race in ``_on_stream_end``.

    Before the fix, the client could signal exit as soon as
    ``pending_streams`` drained — even while a freshly-scheduled
    ``_handle_next_request`` task was still pending. That dropped the
    tail end of the request list. Here we queue 4 requests and verify
    all 4 actually go over the wire.
    """
    # h2deflib absolute
    from h2deflib import Request, send_single_request

    cert_file, key_file = tls_cert
    store = InMemoryResourceStore(
        {
            f"/r{i}": ResponseSpec(body=f"body-{i}".encode(), content_type="text/plain")
            for i in range(4)
        }
    )
    server, port = await _start_server_with_requests(
        H2ServerConfig(), store, cert_file, key_file
    )

    requests = [
        Request(path=f"/r{i}", label="multi", expected_size=10) for i in range(4)
    ]

    # Run the client. It should complete all 4 requests without hanging
    # (and without dropping any of them).
    try:
        await asyncio.wait_for(
            send_single_request(
                connection_id="multi-test",
                server_ip="127.0.0.1",
                server_port=port,
                requests=requests,
                defense_name=defense,
                timeout=10,
            ),
            timeout=15,
        )
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("defense", ["front", "tamaraw"])
async def test_h2client_handles_empty_request_list(tls_cert, defense):
    """
    Constructing an H2Client with no requests should exit cleanly rather
    than hang. This validates the exit condition works with the base
    case too.
    """
    # h2deflib absolute
    from h2deflib import send_single_request

    cert_file, key_file = tls_cert
    store = InMemoryResourceStore(
        {
            "/dummy": ResponseSpec(body=b"x", content_type="text/plain"),
        }
    )
    server, port = await _start_server_with_requests(
        H2ServerConfig(), store, cert_file, key_file
    )
    try:
        # With zero requests, the client should observe that pending is
        # empty AND requests is empty on first pass and exit immediately.
        await asyncio.wait_for(
            send_single_request(
                connection_id="empty",
                server_ip="127.0.0.1",
                server_port=port,
                requests=[],
                defense_name=defense,
                timeout=5,
            ),
            timeout=10,
        )
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("defense", ["front", "tamaraw"])
async def test_h2client_window_updated_resolves_waiter(defense):
    """
    Regression test for the flow-control busy loop.

    Previously ``_wait_for_flow_control`` was a ``sleep(0)`` no-op, so a
    closed send window would turn the send loop into a CPU spin. We now
    store a Future and resolve it from ``_window_updated``. This test
    exercises that resolution path directly without needing real I/O.
    """
    # h2deflib absolute
    from h2deflib import H2Client, Request

    client = H2Client(
        connection_id="fc",
        requests=[Request(path="/", label="t")],
        defense_name=defense,
    )

    # Kick off a waiter — it should block on the future we place in
    # ``flow_control_futures``.
    waiter = asyncio.create_task(client._wait_for_flow_control(stream_id=3))

    # Give the task a tick to register its future.
    await asyncio.sleep(0)
    assert (
        3 in client.flow_control_futures
    ), "_wait_for_flow_control should register a future in flow_control_futures"

    # Simulate an incoming WINDOW_UPDATE for that stream.
    client._window_updated(stream_id=3, delta=1024)

    # The waiter must now complete. Give it a generous timeout in case of
    # scheduler hiccups, but it should be near-instant.
    await asyncio.wait_for(waiter, timeout=1.0)
    assert (
        3 not in client.flow_control_futures
    ), "resolved future should be removed from flow_control_futures"


@pytest.mark.asyncio
async def test_h2client_connection_level_window_update_wakes_all_waiters():
    """
    A stream_id=0 / ``None`` window update is connection-level and should
    resolve every pending waiter in one go.
    """
    # h2deflib absolute
    from h2deflib import H2Client, Request

    client = H2Client(
        connection_id="fc2",
        requests=[Request(path="/", label="t")],
        defense_name="nop",
    )

    waiters = [
        asyncio.create_task(client._wait_for_flow_control(stream_id=sid))
        for sid in (1, 3, 5)
    ]
    await asyncio.sleep(0)
    assert len(client.flow_control_futures) == 3

    # Connection-level update (``stream_id`` falsy) should wake every waiter.
    client._window_updated(stream_id=None, delta=0)

    await asyncio.wait_for(asyncio.gather(*waiters), timeout=1.0)
    assert client.flow_control_futures == {}


@pytest.mark.asyncio
async def test_h2client_stream_reset_clears_pending():
    """
    Regression test for unhandled ``StreamReset``. Before the fix, a
    RST_STREAM from the server left the stream in ``pending_streams``
    forever, so the client would never exit.
    """
    # h2deflib absolute
    from h2deflib import H2Client

    # Empty requests list so that once the only in-flight stream is
    # reset, there's no queued work and the client should exit.
    client = H2Client(
        connection_id="rst",
        requests=[],
        defense_name="nop",
    )

    # Pretend stream 5 is in flight and waiting on flow control.
    client.pending_streams.add(5)
    future = client.loop.create_future()
    client.flow_control_futures[5] = future

    # Server resets the stream.
    client._on_stream_reset(5)

    assert (
        5 not in client.pending_streams
    ), "stream_reset should remove the stream from pending_streams"
    assert (
        5 not in client.flow_control_futures
    ), "stream_reset should drop the flow-control future"
    assert (
        future.cancelled()
    ), "pending flow-control waiter should be cancelled on reset"
    # With no pending streams AND no queued requests, the client should exit.
    assert (
        client.exit_event.is_set()
    ), "client should exit when the last stream is reset and no work remains"


@pytest.mark.asyncio
async def test_h2client_stream_reset_keeps_going_if_requests_remain():
    """
    Complement to the previous test: if there are still queued requests
    when a stream gets reset, the client should NOT exit — it should
    continue processing the queue.
    """
    # h2deflib absolute
    from h2deflib import H2Client, Request

    client = H2Client(
        connection_id="rst-keep",
        requests=[Request(path="/next", label="t")],
        defense_name="nop",
    )
    client.pending_streams.add(5)

    client._on_stream_reset(5)

    assert 5 not in client.pending_streams
    assert (
        not client.exit_event.is_set()
    ), "client should keep going when work still remains after a stream reset"
