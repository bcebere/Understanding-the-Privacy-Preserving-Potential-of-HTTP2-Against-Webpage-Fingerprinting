"""
Fetch a page from a real public HTTP/2 site using h2deflib's own client.

Usage:

    python examples/fetch_real_site.py                     # defaults to nghttp2.org
    python examples/fetch_real_site.py https://example.com/
    python examples/fetch_real_site.py https://nghttp2.org/ tamaraw

The second positional argument picks a defense to run with. Any name
that ``h2deflib.get_defense`` accepts works ('nop', 'front', 'llama',
'httpos', 'tamaraw', 'h2pc').
"""

# stdlib
import asyncio
import ssl
import sys
from urllib.parse import urlparse

from h2deflib import H2Client, Request


async def fetch(url: str, defense: str = "nop", timeout: float = 10.0):
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or 443
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    # Real TLS this time (not cert-checked-off like the localhost tests).
    ctx = ssl.create_default_context()
    ctx.set_alpn_protocols(["h2"])

    loop = asyncio.get_running_loop()
    client = H2Client(
        connection_id=host,
        requests=[Request(path=path, label="fetch", expected_size=0)],
        defense_name=defense,
    )
    transport, _ = await loop.create_connection(
        lambda: client,
        host=host,
        port=port,
        ssl=ctx,
        server_hostname=host,
    )

    # Verify ALPN picked h2.
    ssl_obj = transport.get_extra_info("ssl_object")
    if ssl_obj and ssl_obj.selected_alpn_protocol() != "h2":
        transport.close()
        raise RuntimeError(
            f"{host} did not negotiate HTTP/2 "
            f"(got ALPN = {ssl_obj.selected_alpn_protocol()!r})"
        )

    try:
        await client.wait_for_exit(timeout=timeout)
    finally:
        client.stop()
        transport.close()
    return client


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "https://nghttp2.org/"
    defense = sys.argv[2] if len(sys.argv) > 2 else "nop"
    print(f"GET {url} (defense={defense})")

    client = asyncio.run(fetch(url, defense=defense))

    for sid, buf in client.stream_data.items():
        body = buf.getvalue()
        print(f"\nStream {sid}: {len(body)} bytes")
        preview = body[:300].decode("utf-8", errors="replace")
        print(preview + ("..." if len(body) > 300 else ""))


if __name__ == "__main__":
    main()
