import json
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

from h2deflib import (
    H2Server,
    H2ServerConfig,
    ResourceStore,
    ResponseSpec,
    make_server_ssl_context,
    run_server,
)

SERVER_PATH = Path(__file__).parent
DATA_PATH = Path("data")
SRV_DB_PATH = DATA_PATH / "server_trace"
CLIENT_DB_PATH = DATA_PATH / "client_trace"


# ====================================================================== #
# CLI                                                                     #
# ====================================================================== #


def _parse_args():
    p = ArgumentParser()
    p.add_argument("-dst_port", "--dst_port", dest="dst_port", required=True)

    # Preset
    p.add_argument("--preset", choices=["tamaraw", "alpaca", "h2ps"], default=None)

    # Defense toggles — each maps 1:1 to an H2ServerConfig field
    def flag(name, help_):
        p.add_argument(f"-{name}", f"--{name}", default=0, help=help_)

    flag("http2_server_push", "Use HTTP2 Server Push")
    flag("http2_rnd_server_push", "Use Random HTTP2 Server Push")
    flag("http2_rnd_out_window", "Use Random HTTP2 Send Window")
    flag("http2_batch", "Use HTTP2 Multiplexing")
    flag("http2_rnd_delay", "Use HTTP2 Stream Delays")
    flag("http2_hpack", "Use HTTP2 HPACK")
    flag("http2_rnd_pad", "Use HTTP2 Random padding")
    flag("http2_fixed_pad", "Use HTTP2 Fixed padding")
    flag("http2_hints103", "Use HTTP2 HINTS 103")
    flag("http2_rnd_hints103", "Use Random HTTP2 HINTS 103")
    flag("http2_global_hints103", "Use HTTP2 Global HINTS 103")
    flag("http2_rnd_pings", "Send HTTP2 Ping")
    p.add_argument(
        "-http2_hints103_lo_limit",
        "--http2_hints103_lo_limit",
        default=1,
        help="HINTS 103 lower count",
    )
    p.add_argument(
        "-http2_hints103_hi_limit",
        "--http2_hints103_hi_limit",
        default=5,
        help="HINTS 103 upper count",
    )
    return p.parse_args()


def _config_from_args(args) -> H2ServerConfig:
    if args.preset == "tamaraw":
        return H2ServerConfig.tamaraw()
    if args.preset == "alpaca":
        return H2ServerConfig.alpaca()
    if args.preset == "h2ps":
        return H2ServerConfig.h2ps()

    def i(x):
        return bool(int(x))

    return H2ServerConfig(
        enable_server_push=i(args.http2_server_push),
        enable_random_server_push=i(args.http2_rnd_server_push),
        enable_103_hints=i(args.http2_hints103),
        enable_random_103_hints=i(args.http2_rnd_hints103),
        enable_global_103_hints=i(args.http2_global_hints103),
        hints_count_lo=int(args.http2_hints103_lo_limit),
        hints_count_hi=int(args.http2_hints103_hi_limit),
        enable_multiplexing_batching=i(args.http2_batch),
        enable_hpack_cache_bust=i(args.http2_hpack),
        enable_random_padding=i(args.http2_rnd_pad),
        enable_fixed_padding=i(args.http2_fixed_pad),
        enable_random_frame_delay=i(args.http2_rnd_delay),
        enable_random_out_window=i(args.http2_rnd_out_window),
        enable_random_pings=i(args.http2_rnd_pings),
    )


def _get_domain(url: str) -> str:
    return urlparse(url).netloc


class JsonResourceStore(ResourceStore):
    """
    Loads the experiment's ``server_trace`` and ``client_trace`` JSON files
    for a given testcase label.

    - ``get(path)`` reads the body from ``body_path`` or generates random
      bytes of the requested ``data_size``.
    - ``related_paths(connection_id)`` returns only the client-ordered
      paths that belong to the given connection (URL netloc match).
    - ``put(path, spec)`` is used by the server for injected noise.
    """

    def __init__(
        self,
        server_trace_path: Path,
        client_trace_path: Path,
        connection_id: Optional[str] = None,
    ):
        with open(server_trace_path) as f:
            full_server = json.load(f)
        with open(client_trace_path) as f:
            full_client: List[dict] = json.load(f)

        if connection_id is None:
            self._server = dict(full_server)
            self._client = list(full_client)
        else:
            self._server = {
                k: v
                for k, v in full_server.items()
                if _get_domain(v["url"]) == connection_id
            }
            self._client = [
                c for c in full_client if _get_domain(c["url"]) == connection_id
            ]

        # Fall back to the complete server DB for redirect resolution
        self._full_server = full_server
        self._synthetic: Dict[str, ResponseSpec] = {}

    # -- ResourceStore interface ------------------------------------- #

    def get(self, path: str) -> Optional[ResponseSpec]:
        if path in self._synthetic:
            return self._synthetic[path]

        entry = self._server.get(path) or self._full_server.get(path)
        if entry is None:
            return None

        if "body_path" in entry:
            bp = Path(entry["body_path"])
            if not bp.exists():
                return None
            body = bp.read_bytes()
        elif "data_size" in entry:
            size = entry["data_size"]
            if size == "random":
                size = random.randint(100, 10000)
            elif isinstance(size, (int, float)):
                size = int(size)
            else:
                size = 1
            import string

            body = "".join(
                random.choices(string.ascii_uppercase + string.digits, k=size)
            ).encode()
        else:
            return None

        return ResponseSpec(
            body=body,
            content_type=entry.get("content_type", "application/json"),
            headers=dict(entry.get("headers", {})),
            response_delay=float(entry.get("timeout_s", 0)),
        )

    def put(self, path: str, spec: ResponseSpec) -> None:
        self._synthetic[path] = spec

    def related_paths(self, scope: Optional[str] = None) -> List[str]:
        """Return client-ordered paths. ``scope`` is ignored because the
        JsonResourceStore was already filtered per-connection at __init__."""
        return [c["url_local"] for c in self._client]


def make_store_factory():
    """
    Returns a factory ``(headers) -> ResourceStore`` suitable for H2Server.
    Uses the request's ``label`` header to pick the testcase and
    ``connection_id`` to scope the store to that connection.
    """

    def _factory(headers):
        label = headers["label"]
        connection_id = headers.get("connection_id")
        return JsonResourceStore(
            server_trace_path=SRV_DB_PATH / f"{label}.json",
            client_trace_path=CLIENT_DB_PATH / f"{label}.json",
            connection_id=connection_id,
        )

    return _factory


def main():
    args = _parse_args()
    config = _config_from_args(args)

    print(f"[experimental_server] config = {config}")

    factory = H2Server.factory(make_store_factory(), config)
    ssl_ctx = make_server_ssl_context(
        certfile=SERVER_PATH / "keys/cert.pem",
        keyfile=SERVER_PATH / "keys/key.pem",
    )
    run_server(factory, "0.0.0.0", int(args.dst_port), ssl_ctx)


if __name__ == "__main__":
    main()
