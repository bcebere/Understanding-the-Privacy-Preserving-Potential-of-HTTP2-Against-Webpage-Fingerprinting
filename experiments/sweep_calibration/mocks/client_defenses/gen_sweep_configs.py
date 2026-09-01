#!/usr/bin/env python3
"""
Generates four defense configuration files at increasing overhead intensity.

    sweep_low.json   sweep_mid1.json   sweep_mid2.json   sweep_high.json

mid1 == the configuration used in the submitted paper, byte for byte.
Every level moves ONE intensity dimension per defense so the resulting
points lie on a single trade-off curve (Cai et al. CCS'14 Fig. 1 style).

Requires the small DEFENSE patch (ping_* fields, front_window_*,
recv_threshold_resample) -- see accompanying notes.
"""

import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")

# ----------------------------------------------------------------------
# Level-independent settings: everything that defines WHAT the defense is,
# as opposed to HOW HARD it is applied.  These never change across levels.
# ----------------------------------------------------------------------

BASE = {
    "front": {
        "name": "front",
        "send_dummy_packet_strategy": "front",
        "send_dummy_packet_interval": "front",
        "send_dummy_packet_loop": True,
        "send_dummy_min": 1,
        # original FRONT draws the Rayleigh scale from U(1,14) s; the HTTP/2
        # emulation uses U(0,1) s.  Exported so the deviation is explicit in
        # the config rather than buried in send_dummy_packet().
        "front_window_min": 0.0,
        "front_window_max": 1.0,
    },
    "tamaraw_qcsd": {
        "name": "tamaraw_qcsd",
        "send_dummy_packet_strategy": "random_per_connection",
        "send_dummy_packet_loop": True,
        # configured pacing threshold must actually persist -- see notes
        "recv_threshold_resample": False,
    },
    "all_mods": {
        "name": "all_mods",
        "initial_window_size_strategy": "random",
        "request_batch": False,
        "request_shuffle": True,
        "send_dummy_packet_strategy": "random_batch",
        "send_dummy_packet_loop": False,
        "recv_threshold_resample": False,
    },
}

# ----------------------------------------------------------------------
# Intensity ladders.  One scalar dimension per defense.
# ----------------------------------------------------------------------

LADDER = {
    # FRONT: dummy-request COUNT ceiling.  nc ~ U(send_dummy_min, send_dummy_max),
    # so mean dummies ~ max/2.  Range chosen to bracket the 10 -> 500 span the
    # paper already probes in Section 6.1.1 (F1 0.76 -> 0.48 on Udemy).
    "front": {
        "low": {"send_dummy_max": 50},
        "mid1": {"send_dummy_max": 200},
        "mid2": {"send_dummy_max": 350},
        "high": {"send_dummy_max": 500},
    },
    # CL-TAM: joint rate knob, mirroring Tamaraw's rho.  Rising intensity =
    # dummies more often, window updates delayed longer, smaller frames.
    # dummy interval halves, recv interval doubles, window/threshold halve.
    "tamaraw_qcsd": {
        "low": {
            "initial_window_size_strategy": 16384,
            "recv_delay_threshold": 16384,
            "recv_interval_strategy": 0.0025,
            "send_dummy_packet_interval": 0.08,
        },
        "mid1": {
            "initial_window_size_strategy": 4096,
            "recv_delay_threshold": 4096,
            "recv_interval_strategy": 0.01,
            "send_dummy_packet_interval": 0.02,
        },
        "mid2": {
            "initial_window_size_strategy": 2048,
            "recv_delay_threshold": 2048,
            "recv_interval_strategy": 0.02,
            "send_dummy_packet_interval": 0.01,
        },
        "high": {
            "initial_window_size_strategy": 1024,
            "recv_delay_threshold": 1024,
            "recv_interval_strategy": 0.04,
            "send_dummy_packet_interval": 0.005,
        },
    },
    # H2PC: guarding-noise volume + probing volume.
    # low = Client Opportunity 1 only (multiplexing/flow-control/pings, NO
    # guard streams: limit 0 makes send_dummy_packet() return None).  That is
    # the paper's own Opportunity-1 vs Opportunity-2 split, so the lowest
    # point doubles as a free ablation.
    "all_mods": {
        "low": {
            "send_dummy_packet_limit": 0,
            "random_pings": True,
            "ping_probability": 0.25,
            "ping_count_min": 1,
            "ping_count_max": 2,
            "recv_interval_strategy": 0.00005,
            "recv_delay_threshold": 20000,
        },
        "mid1": {
            "send_dummy_packet_limit": 1,
            "random_pings": True,
            "ping_probability": 0.5,
            "ping_count_min": 1,
            "ping_count_max": 3,
            "recv_interval_strategy": 0.0001,
            "recv_delay_threshold": 10000,
        },
        "mid2": {
            "send_dummy_packet_limit": 2,
            "random_pings": True,
            "ping_probability": 0.7,
            "ping_count_min": 1,
            "ping_count_max": 5,
            "recv_interval_strategy": 0.0002,
            "recv_delay_threshold": 5000,
        },
        "high": {
            "send_dummy_packet_limit": 3,
            "random_pings": True,
            "ping_probability": 1.0,
            "ping_count_min": 2,
            "ping_count_max": 8,
            "recv_interval_strategy": 0.0005,
            "recv_delay_threshold": 2500,
        },
    },
}

LEVELS = ["low", "mid1", "mid2", "high"]

NOTE = {
    "low": "lightest configuration; H2PC has guard streams disabled (features only)",
    "mid1": "AS SUBMITTED -- reproduces the paper's Tables 3-10 exactly",
    "mid2": "one step above the submitted configuration",
    "high": "heaviest configuration on each ladder",
}


def build(level: str) -> dict:
    defenses = {}
    for key, base in BASE.items():
        cfg = dict(base)
        cfg.update(LADDER[key][level])
        defenses[key] = cfg
    return {
        "level": level,
        "note": NOTE[level],
        "measured_overhead": {k: None for k in BASE},  # filled by calibration
        "defenses": defenses,
    }


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    for level in LEVELS:
        path = os.path.join(OUT, f"sweep_{level}.json")
        with open(path, "w") as fh:
            json.dump(build(level), fh, indent=2)
        print("wrote", path)


# ----------------------------------------------------------------------
# loader -- import this from the runner
# ----------------------------------------------------------------------


def load_defense(name: str, level: str, config_dir: str = OUT):
    """Return a FRESH DEFENSE instance.  Always construct per connection:
    DEFENSE carries mutable per-connection state (_cache, recv_delay_threshold,
    added_frames, last_dummy_packet) and reusing one instance across
    connections silently changes the defense -- see notes."""
    from h2deflib.defenses import DEFENSE  # adjust import to your layout

    with open(os.path.join(config_dir, f"sweep_{level}.json")) as fh:
        blob = json.load(fh)
    if name not in blob["defenses"]:
        raise KeyError(f"{name} not in {level}; have {list(blob['defenses'])}")
    return DEFENSE(**blob["defenses"][name])


if __name__ == "__main__":
    main()

    # checks: mid1 must equal the submitted values, and each ladder must be
    # strictly monotone in its intensity dimension.
    mid1 = build("mid1")["defenses"]
    submitted = {
        "front": {"send_dummy_max": 200},
        "tamaraw_qcsd": {
            "initial_window_size_strategy": 4096,
            "recv_delay_threshold": 4096,
            "recv_interval_strategy": 0.01,
            "send_dummy_packet_interval": 0.02,
        },
        "all_mods": {
            "send_dummy_packet_limit": 1,
            "recv_interval_strategy": 0.0001,
            "recv_delay_threshold": 10000,
        },
    }
    for d, expected in submitted.items():
        for k, v in expected.items():
            assert mid1[d][k] == v, f"mid1 drift: {d}.{k} = {mid1[d][k]} != {v}"
    print("\nmid1 matches submitted values: OK")

    monotone = {
        "front": ("send_dummy_max", 1),
        "tamaraw_qcsd": ("send_dummy_packet_interval", -1),
        "all_mods": ("send_dummy_packet_limit", 1),
    }
    for d, (knob, direction) in monotone.items():
        vals = [LADDER[d][lv][knob] for lv in LEVELS]
        ok = all((b - a) * direction > 0 or (b == a) for a, b in zip(vals, vals[1:]))
        assert ok, f"{d}.{knob} not monotone: {vals}"
        print(f"{d:14s} {knob:28s} {vals}")
