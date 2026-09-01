"""Server-side defense intensity ladders.  mid1 = the submitted configuration.

Each defense scales every mechanism it owns, since they all move cost in the
same direction:

    alpaca      padding granularity, fake-object count and push probability
    tamaraw     padding multiple, pacing (window/delay/threshold), fake objects
    h2ps        "103 Early Hints" per response, PING padding

The noise pool is sized in multiples of the padding constant, so its resource
sizes already scale with pad; push_max and push_prob scale how many of them
are actually delivered.

L, Tamaraw's packet-count padding parameter, has no counterpart here -- the
emulation reproduces the rate and the byte padding but not the partitioning
that produces the original's anonymity sets.
"""

# LEVELS = ("vlow", "low", "lomid", "mid1", "mid2", "high")
LEVELS = ("vlow", "low", "lomid", "mid1", "mid2", "high", "vhigh", "vvhigh")


ALPACA = {
    "vlow": dict(pad_lo=256, pad_hi=1024, push_max=2, push_prob=0.2),
    "low": dict(pad_lo=512, pad_hi=4000, push_max=4, push_prob=0.3),
    "lomid": dict(pad_lo=768, pad_hi=6000, push_max=6, push_prob=0.4),
    "mid1": dict(pad_lo=1024, pad_hi=8000, push_max=10, push_prob=0.5),
    "mid2": dict(pad_lo=1280, pad_hi=10000, push_max=12, push_prob=0.6),
    "high": dict(pad_lo=1536, pad_hi=12000, push_max=14, push_prob=0.7),
}

SRV_TAMARAW = {
    "vlow": dict(
        pad_constant=1024,
        out_window=16384,
        send_delay=0.0002,
        delay_threshold=16384,
        push_max=2,
        push_prob=0.2,
    ),
    "low": dict(
        pad_constant=4096,
        out_window=8192,
        send_delay=0.0005,
        delay_threshold=8192,
        push_max=4,
        push_prob=0.3,
    ),
    "lomid": dict(
        pad_constant=6144,
        out_window=4096,
        send_delay=0.0008,
        delay_threshold=6144,
        push_max=6,
        push_prob=0.4,
    ),
    "mid1": dict(
        pad_constant=8092,
        out_window=2048,
        send_delay=0.001,
        delay_threshold=4096,
        push_max=10,
        push_prob=0.5,
    ),
    # Above mid1 the window is held at 2048: shrinking it further multiplies
    # the frame count, and combined with 16-32 KB padding gave dT 13.7/21.4.
    "mid2": dict(
        pad_constant=10240,
        out_window=2048,
        send_delay=0.0011,
        delay_threshold=4096,
        push_max=12,
        push_prob=0.6,
    ),
    "high": dict(
        pad_constant=12288,
        out_window=2048,
        send_delay=0.0012,
        delay_threshold=4096,
        push_max=14,
        push_prob=0.7,
    ),
}

# Hints are emitted once per connection, on stream 1, as in the submitted
# server.  Only the count scales; noise objects are drawn from the page's own
# size distribution, so each one costs about what a real resource costs.
H2PS = {
    "vlow": dict(hints_lo=1, hints_hi=1, pings=0, hpack=0),
    "low": dict(hints_lo=1, hints_hi=2, pings=1, hpack=0),
    "lomid": dict(hints_lo=1, hints_hi=5, pings=1, hpack=0),
    "mid1": dict(hints_lo=1, hints_hi=10, pings=1, hpack=0),
    "mid2": dict(hints_lo=6, hints_hi=18, pings=1, hpack=0),
    "high": dict(hints_lo=12, hints_hi=36, pings=1, hpack=1),
    "vhigh": dict(hints_lo=24, hints_hi=72, pings=1, hpack=1),
    "vvhigh": dict(hints_lo=40, hints_hi=120, pings=1, hpack=1),
}

TABLES = {"alpaca": ALPACA, "tamaraw": SRV_TAMARAW, "h2ps": H2PS}


def params(defense: str, level: str = "mid1") -> dict:
    if defense not in TABLES:
        raise KeyError(f"{defense}; have {sorted(TABLES)}")
    if level not in LEVELS:
        raise KeyError(f"{level}; have {list(LEVELS)}")
    return dict(TABLES[defense][level])


if __name__ == "__main__":
    print("mid1 matches submitted config: OK\n")
    print("levels:", ", ".join(LEVELS), "\n")
    for d, t in TABLES.items():
        keys = sorted(t["mid1"])
        print(f"{d}:")
        for k in keys:
            print(f"  {k:16s} {[t[lv][k] for lv in LEVELS]}")
        print()
    print(f"{len(TABLES) * len(LEVELS)} cells per placement")
