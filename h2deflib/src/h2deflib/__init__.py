"""
h2deflib — A reusable HTTP/2 client and server with optional defensive
traffic-shaping features.

Typical use:

    # Server side
    from h2deflib import H2Server, H2ServerConfig, make_server_ssl_context, run_server
    config = H2ServerConfig(enable_random_padding=True, enable_random_103_hints=True)
    factory = H2Server.factory(my_store_factory, config)
    run_server(factory, "0.0.0.0", 8443, make_server_ssl_context("cert.pem", "key.pem"))

    # Client side
    from h2deflib import Request, run_test_case
    run_test_case(
        server_ip="127.0.0.1", server_port=8443,
        requests_by_connection={"example.com": [Request(path="/", label="t1")]},
        request_server_defenses={"example.com": True},
        defense_name="tamaraw",
    )
"""

# h2deflib relative
# wfaudit relative
from .client import (
    ConnectionDetails,
    H2Client,
    Request,
    connect_h2_client,
    get_defense,
    make_client_ssl_context,
    run_test_case,
    send_requests,
    send_single_request,
)
from .server import (
    H2Server,
    H2ServerConfig,
    InMemoryResourceStore,
    RequestData,
    ResourceStore,
    ResponseSpec,
    make_server_ssl_context,
    run_server,
)

__all__ = [
    # Server
    "H2Server",
    "H2ServerConfig",
    "ResourceStore",
    "InMemoryResourceStore",
    "ResponseSpec",
    "RequestData",
    "make_server_ssl_context",
    "run_server",
    # Client
    "H2Client",
    "Request",
    "ConnectionDetails",
    "get_defense",
    "make_client_ssl_context",
    "connect_h2_client",
    "send_single_request",
    "send_requests",
    "run_test_case",
]
