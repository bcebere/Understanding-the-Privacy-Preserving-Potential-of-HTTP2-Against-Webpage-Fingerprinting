# future
from __future__ import annotations

# stdlib
import asyncio
from pathlib import Path

# third party
import pytest

# h2deflib absolute
from h2deflib import (
    H2Server,
    H2ServerConfig,
    InMemoryResourceStore,
    Request,
    ResponseSpec,
    make_server_ssl_context,
    send_single_request,
)


async def _start_server(store, cert_file: Path, key_file: Path):
    ssl_ctx = make_server_ssl_context(cert_file, key_file)
    factory = H2Server.factory(lambda headers: store, H2ServerConfig())
    loop = asyncio.get_running_loop()
    server = await loop.create_server(factory, "127.0.0.1", 0, ssl=ssl_ctx)
    port = server.sockets[0].getsockname()[1]
    return server, port


def _store():
    return InMemoryResourceStore(
        {
            f"/r{i}": ResponseSpec(body=f"body-{i}".encode(), content_type="text/plain")
            for i in range(4)
        }
    )


def _requests(n=4):
    return [Request(path=f"/r{i}", label="t", expected_size=10) for i in range(n)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "defense",
    ["nop", "tamaraw", "front", "httpos", "llama", "h2pc"],
)
async def test_defense_completes(tls_cert, defense):
    """Every defense runs a small sequence end-to-end without hanging."""
    cert_file, key_file = tls_cert
    server, port = await _start_server(_store(), cert_file, key_file)
    try:
        await asyncio.wait_for(
            send_single_request(
                connection_id="t",
                server_ip="127.0.0.1",
                server_port=port,
                requests=_requests(),
                defense_name=defense,
                timeout=8,
            ),
            timeout=12,
        )
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_front_sends_extra_dummies(tls_cert, capsys):
    """FRONT's dummy loop fires within the first ~2 s — look for its log line."""
    cert_file, key_file = tls_cert
    server, port = await _start_server(_store(), cert_file, key_file)
    try:
        await send_single_request(
            connection_id="t",
            server_ip="127.0.0.1",
            server_port=port,
            requests=_requests(),
            defense_name="front",
            timeout=6,
        )
    finally:
        server.close()
        await server.wait_closed()

    out = capsys.readouterr().out
    assert "[FRONT]" in out, "expected FRONT to log its sampled dummy count"


@pytest.mark.asyncio
async def test_httpos_splits_binary_with_range(tls_cert, capsys):
    """HTTPOS splits binary >=10KB into ranged requests — check its [RANGED] log."""
    cert_file, key_file = tls_cert
    store = InMemoryResourceStore(
        {
            "/big.png": ResponseSpec(body=b"X" * 20_000, content_type="image/png"),
        }
    )
    server, port = await _start_server(store, cert_file, key_file)
    try:
        await send_single_request(
            connection_id="t",
            server_ip="127.0.0.1",
            server_port=port,
            requests=[Request(path="/big.png", label="t", expected_size=20_000)],
            defense_name="httpos",
            timeout=6,
        )
    finally:
        server.close()
        await server.wait_closed()

    out = capsys.readouterr().out
    assert "[RANGED] USE binary" in out, "expected HTTPOS to log the split"
    # The split should produce multiple streams for the same path.
    stream_lines = [
        ln for ln in out.splitlines() if "path=/big.png" in ln and "[stream]" in ln
    ]
    assert (
        len(stream_lines) >= 2
    ), f"expected multiple streams for /big.png after split, got {len(stream_lines)}"


@pytest.mark.asyncio
async def test_tamaraw_sets_small_window():
    """tamaraw advertises INITIAL_WINDOW_SIZE=4096."""
    # third party
    from h2.settings import SettingCodes

    # h2deflib absolute
    from h2deflib import H2Client

    c = H2Client(connection_id="t", requests=[], defense_name="tamaraw")
    assert c.window_size == 4096
    assert c.conn_settings[SettingCodes.INITIAL_WINDOW_SIZE] == 4096


@pytest.mark.asyncio
async def test_h2pc_uses_random_initial_window():
    """h2pc picks a random window in [2^10, 2^14]."""
    # h2deflib absolute
    from h2deflib import H2Client

    c = H2Client(connection_id="t", requests=[], defense_name="h2pc")
    assert 2**10 <= c.window_size <= 2**14


@pytest.mark.asyncio
async def test_nop_uses_default_window():
    """nop leaves the window at the h2 default (16384)."""
    # h2deflib absolute
    from h2deflib import H2Client

    c = H2Client(connection_id="t", requests=[], defense_name="nop")
    assert c.window_size == 16384


@pytest.mark.asyncio
async def test_llama_is_configured_for_batch_shuffle_delay():
    """llama opts into request batching, shuffling and delay."""
    # h2deflib absolute
    from h2deflib.client import get_defense

    d = get_defense("llama")
    assert d.should_batch()
    assert d.should_shuffle()
    # request_delay is probabilistic — we only assert the flag is set.
    assert d.request_delay is True
