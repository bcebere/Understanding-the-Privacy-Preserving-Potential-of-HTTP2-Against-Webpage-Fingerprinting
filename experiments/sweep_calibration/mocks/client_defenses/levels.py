"""
Defense intensity ladders for the sweep.
"""

import os
from copy import deepcopy

from client_defenses.front import FRONT_DEFENSE
from client_defenses.h2pc import H2PC_DEFENSE
from client_defenses.httpos import HTTPOS_DEFENSE
from client_defenses.llama import LLAMA_DEFENSE
from client_defenses.nop import NOP_DEFENSE
from client_defenses.tamaraw_qcsd import TAMARAW_QCSD_DEFENSE

DEFENSE = type(FRONT_DEFENSE)

LEVELS = ("vlow", "low", "lomid", "mid1", "mid2", "high")

_RUNTIME_FIELDS = ("last_dummy_packet", "added_frames", "added_delay")

_ALLOWED_DEVIATIONS = ()


def _field_names(cls):
    if hasattr(cls, "model_fields"):  # pydantic v2
        return set(cls.model_fields)
    return set(cls.__fields__)  # pydantic v1


def _require_patched_defense():
    """The ladders set knobs that only exist in the patched DEFENSE."""
    needed = {
        "max_dummy_time",
        "dummy_min_resource_size",
        "dummy_on_response",
        "recv_threshold_resample",
        "ping_probability",
        "ping_count_min",
        "ping_count_max",
        "front_window_min",
        "front_window_max",
        "ranged_splits_min",
        "ranged_splits_max",
        "llama_dummy_probability",
    }
    missing = sorted(needed - _field_names(DEFENSE))
    if missing:
        raise ImportError(
            f"{DEFENSE.__module__}.{DEFENSE.__name__} is missing fields "
            f"{missing}. Apply the patched DEFENSE (the one adding the "
            "ping_*/front_window_*/recv_threshold_resample fields) before "
            "using the intensity ladders."
        )


_require_patched_defense()


def _fields(d):
    return d.model_dump() if hasattr(d, "model_dump") else d.dict()


def _variant(base: DEFENSE, **overrides) -> DEFENSE:
    """Rebuild a DEFENSE from `base` with `overrides` applied.

    Goes through the constructor rather than model_copy so __init__ runs and
    per-connection state (cache, configured threshold) is initialised.
    """
    cfg = _fields(base)
    cfg.update(overrides)
    return DEFENSE(**cfg)


def _flat(base: DEFENSE, **overrides):
    """A defense with no intensity dimension: same object at every level."""
    return {lv: _variant(base, **overrides) for lv in LEVELS}


# nc ~ U(send_dummy_min, send_dummy_max); brackets the 10-500 span of S6.1.1.
FRONT_LEVELS = {
    lv: _variant(FRONT_DEFENSE, send_dummy_max=cap)
    for lv, cap in zip(LEVELS, (20, 50, 110, 200, 350, 500))
}

# Dummy send rate only;
_TAM_FIXED = dict(
    initial_window_size_strategy=4096,
    recv_delay_threshold=4096,
    recv_interval_strategy=0.01,
)

_TAM_LADDER = {
    "vlow": dict(send_dummy_packet_interval=0.32),
    "low": dict(send_dummy_packet_interval=0.08),
    "lomid": dict(send_dummy_packet_interval=0.04),
    "mid1": dict(send_dummy_packet_interval=0.02),
    "mid2": dict(send_dummy_packet_interval=0.01),
    "high": dict(send_dummy_packet_interval=0.005),
}

TAMARAW_LEVELS = {
    lv: _variant(TAMARAW_QCSD_DEFENSE, **_TAM_FIXED, **kw)
    for lv, kw in _TAM_LADDER.items()
}

# vlow = no pings, no guards; low adds pings; lomid+ add guard streams.
_H2PC_LADDER = {
    "lomid": dict(
        send_dummy_packet_limit=1,
        ping_probability=0.35,
        ping_count_min=1,
        ping_count_max=2,
        recv_interval_strategy=0.00008,
        recv_delay_threshold=15000,
    ),
    "vlow": dict(
        send_dummy_packet_limit=0,
        random_pings=False,
        ping_probability=0.0,
        ping_count_min=1,
        ping_count_max=1,
        recv_interval_strategy=0.00005,
        recv_delay_threshold=20000,
    ),
    "low": dict(
        send_dummy_packet_limit=0,
        ping_probability=0.25,
        ping_count_min=1,
        ping_count_max=2,
        recv_interval_strategy=0.00005,
        recv_delay_threshold=20000,
    ),
    "mid1": dict(
        send_dummy_packet_limit=1,
        ping_probability=0.5,
        ping_count_min=1,
        ping_count_max=3,
        recv_interval_strategy=0.0001,
        recv_delay_threshold=10000,
    ),
    "mid2": dict(
        send_dummy_packet_limit=2,
        ping_probability=0.7,
        ping_count_min=1,
        ping_count_max=5,
        recv_interval_strategy=0.0002,
        recv_delay_threshold=5000,
    ),
    "high": dict(
        send_dummy_packet_limit=3,
        ping_probability=1.0,
        ping_count_min=2,
        ping_count_max=8,
        recv_interval_strategy=0.0005,
        recv_delay_threshold=2500,
    ),
}

H2PC_LEVELS = {lv: _variant(H2PC_DEFENSE, **kw) for lv, kw in _H2PC_LADDER.items()}

_HTTPOS_LADDER = {
    "vlow": dict(
        initial_window_size_strategy=16384, ranged_splits_min=3, ranged_splits_max=5
    ),
    "low": dict(
        initial_window_size_strategy=8192, ranged_splits_min=4, ranged_splits_max=7
    ),
    "lomid": dict(
        initial_window_size_strategy=4096, ranged_splits_min=5, ranged_splits_max=8
    ),
    "mid1": dict(
        initial_window_size_strategy=2048, ranged_splits_min=5, ranged_splits_max=10
    ),
    "mid2": dict(
        initial_window_size_strategy=1024, ranged_splits_min=10, ranged_splits_max=20
    ),
    "high": dict(
        initial_window_size_strategy=512, ranged_splits_min=20, ranged_splits_max=40
    ),
}

HTTPOS_LEVELS = {
    lv: _variant(HTTPOS_DEFENSE, **kw) for lv, kw in _HTTPOS_LADDER.items()
}

_LLAMA_LADDER = {
    "vlow": 0.0,
    "low": 0.15,
    "lomid": 0.25,
    "mid1": 0.3,
    "mid2": 0.5,
    "high": 0.8,
}

LLAMA_LEVELS = {
    lv: _variant(LLAMA_DEFENSE, llama_dummy_probability=p)
    for lv, p in _LLAMA_LADDER.items()
}

NOP_LEVELS = _flat(NOP_DEFENSE)

DEFENSE_LEVELS = {
    "nop": NOP_LEVELS,
    "tamaraw": TAMARAW_LEVELS,
    "front": FRONT_LEVELS,
    "httpos": HTTPOS_LEVELS,
    "llama": LLAMA_LEVELS,
    "h2pc": H2PC_LEVELS,
}

SWEPT = ("front", "tamaraw", "h2pc", "httpos", "llama")

# Structural fixes per dataset; intensity stays in the ladders.
DATASET_OVERRIDES = {
    "amazon": {
        "httpos": dict(ranged_splits_min=2, ranged_splits_max=4),
    },
    "udemy": {
        "tamaraw": dict(
            initial_window_size_strategy=16384,
            recv_delay_threshold=16384,
        ),
    },
}

# Per-request delay bound.  Cherubin et al. draw U(0, half the median page
# load time) and dispatch concurrently; this client dispatches sequentially,
# so delays add rather than overlap.  Dividing by the mean request count
# reproduces their ~50% latency target instead of stalling the page load:
#   python3 -c "
#   import pandas as pd, json, glob
#   lat = pd.read_csv('workspace/ovh_baseline.csv').latency.median()
#   n = [len(json.load(open(f))) for f in glob.glob('data/client_trace/*')[:100]]
#   print(round(lat/2/(sum(n)/len(n)), 4))"
LLAMA_DELAY_MAX = {
    "amazon": 0.0525,
    "bbc": 0.0548,
    "reddit": 0.0268,
    "udemy": 0.0165,
    "wiki": 0.0248,
}

KNOWN_DATASETS = tuple(LLAMA_DELAY_MAX) + tuple(DATASET_OVERRIDES)


def _dataset_key(dataset=None):
    """Dataset name, from the argument or $WF_DATASET. Tolerates directory
    names like '4_udemy' so it can be set from the run directory."""
    ds = dataset if dataset is not None else os.environ.get("WF_DATASET", "")
    ds = str(ds).strip().lower()
    for known in KNOWN_DATASETS:
        if known in ds:
            return known
    return ds


def get_defense(defense: str, level: str = "mid1", dataset=None) -> DEFENSE:
    """Fresh DEFENSE instance for `defense` at `level`, with any override for
    `dataset` applied. `dataset` defaults to $WF_DATASET, so nothing upstream
    needs a new argument."""
    if defense not in DEFENSE_LEVELS:
        raise NotImplementedError(defense)
    levels = DEFENSE_LEVELS[defense]
    if level not in levels:
        raise KeyError(f"level {level!r} for {defense!r}; have {sorted(levels)}")

    key = _dataset_key(dataset)
    base = levels[level]

    tweak = dict(DATASET_OVERRIDES.get(key, {}).get(defense, {}))
    if defense == "llama" and key in LLAMA_DELAY_MAX:
        tweak["request_delay_max"] = LLAMA_DELAY_MAX[key]

    return _variant(base, **tweak) if tweak else deepcopy(base)


def overridden(dataset=None):
    """Which (defense, field) pairs this dataset deviates on. For the
    change-log: every deviation from the global ladder is listed here."""
    return DATASET_OVERRIDES.get(_dataset_key(dataset), {})


def cells(defenses=SWEPT, levels=LEVELS):
    """(defense, level) pairs to run, in cheapest-first order."""
    order = {lv: i for i, lv in enumerate(LEVELS)}
    return sorted(
        ((d, lv) for d in defenses for lv in levels), key=lambda t: order[t[1]]
    )


if __name__ == "__main__":
    knob = {
        "front": "send_dummy_max",
        "tamaraw": "send_dummy_packet_interval",
        "h2pc": "send_dummy_packet_limit",
        "httpos": "ranged_splits_max",
        "llama": "llama_dummy_probability",
    }
    print("levels:", ", ".join(LEVELS))
    for d in SWEPT:
        vals = [getattr(DEFENSE_LEVELS[d][lv], knob[d]) for lv in LEVELS]
        print(f"  {d:9s} {knob[d]:28s} {vals}")

    print("\nper-dataset overrides:")
    if not DATASET_OVERRIDES:
        print("  (none)")
    for ds, per_def in DATASET_OVERRIDES.items():
        for dfn, fields in per_def.items():
            print(f"  {ds:8s} {dfn:9s} {fields}")

    print(f"\n{len(cells())} cells per dataset:")
    for d, lv in cells():
        print(f"  --defense {d} --level {lv}")
