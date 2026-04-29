# HTTP/2 Defense Library layout

```
src/h2deflib/
├── __init__.py        # public API
├── server.py          # H2Server, H2ServerConfig, ResourceStore, ResponseSpec
└── client.py          # H2Client, Request, ConnectionDetails, get_defense, run_test_case

client_defenses/       # one defense strategy per file
```

## What lives where

### `h2deflib/server.py`
- **`H2Server`** — asyncio HTTP/2 server protocol. Handles connection setup,
  flow control, stream lifecycle, pings, window updates.
- **`H2ServerConfig`** — single dataclass holding every traffic-shaping knob:
  server push (real / noise), 103 Early Hints (real / noise), HPACK cache
  busting, multiplexing batching, padding (fixed / random), per-frame and
  per-volume delays, random outbound frame size, random pings. Has
  `.tamaraw()`, `.alpaca()`, and `.h2ps()` presets.
- **`ResourceStore`** — abstract interface the server reads responses from.
  An in-memory implementation ships for simple use.
- **`ResponseSpec`** — what the store returns: body, content-type, headers,
  pre-send delay.
- **`make_server_ssl_context`**, **`run_server`** — TLS + event-loop helpers.

### `h2deflib/client.py`
- **`H2Client`** — asyncio HTTP/2 client protocol. Drives a pluggable
  `Defense` object for dummy traffic, pings, receive-side pacing,
  request shuffling, batching, ranged-request splitting.
- **`Request`**, **`ConnectionDetails`** — pydantic models used by the
  client and reported back via `ConnectionDetails.stats()`.
- **`get_defense(name)`** — resolves `"nop" | "tamaraw" |
  "front" | "httpos" | "llama" | "h2pc"` into a defense instance.
- **`send_single_request`**, **`send_requests`**, **`run_test_case`** —
  runners for the experimental code.


More usage examples are available in the [experiments folder](../experiments/example_replay).
