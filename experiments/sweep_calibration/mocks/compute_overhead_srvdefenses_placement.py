#!/usr/bin/env python3
"""Overhead sweep: every server defense, every level, every placement.

    python3 compute_overhead_placements.py <server_ip> [--pages 25]
    python3 compute_overhead_placements.py <ip> --defense alpaca --levels mid1,high

Run from a dataset dir in the calibration tree.  Placements are defined here
rather than imported, so this does not depend on main_table_config.

Scenario tags carry both placement and level, so nothing overwrites:
    ovh_srvalpaca_1st_mid1.csv
    ovh_srvalpaca_3rd_1_mid1.csv
    ovh_srvalpaca_all_mid1.csv
"""
import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path

PLACEMENTS = {
    "1_amazon": {"1st": "www.amazon.com", "3rd_1": "m.media-amazon.com", "all": "all"},
    "2_bbc": {"1st": "www.bbc.com", "3rd_1": "static.files.bbci.co.uk", "all": "all"},
    "3_reddit": {
        "1st": "www.reddit.com",
        "3rd_1": "www.redditstatic.com",
        "all": "all",
    },
    "4_udemy": {
        "1st": "www.udemy.com",
        "3rd_1": "challenges.cloudflare.com",
        "all": "all",
    },
    "5_wiki": {
        "1st": "en.wikipedia.org",
        "3rd_1": "upload.wikimedia.org",
        "all": "all",
    },
}

# must match server_defenses/levels.py, since the port encodes the indices
DEFENSE_ORDER = ["alpaca", "tamaraw", "h2ps"]
LEVELS = ["vlow", "low", "lomid", "mid1", "mid2", "high", "vhigh", "vvhigh"]
# levels that exist per defense
DEFENSE_LEVELS = {
    "alpaca": LEVELS[:6],
    "tamaraw": LEVELS[:6],
    "h2ps": LEVELS,
}

DATASET = Path.cwd().name
CATEGORY = Path.cwd().parent.name
WORKSPACE = Path(f"/http2/experiments/{CATEGORY}/{DATASET}/overhead")

parser = argparse.ArgumentParser()
parser.add_argument("ip")
parser.add_argument("--pages", type=int, default=25)
parser.add_argument("--defense", default=None, help="comma-separated subset")
parser.add_argument("--levels", default=None, help="comma-separated subset")
parser.add_argument("--placements", default=None, help="comma-separated subset")
parser.add_argument("--force", action="store_true")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

defenses = args.defense.split(",") if args.defense else DEFENSE_ORDER
places = PLACEMENTS.get(DATASET)
if not places:
    sys.exit(f"no placements defined for {DATASET}")
if args.placements:
    keep = set(args.placements.split(","))
    places = {k: v for k, v in places.items() if k in keep}

WORKSPACE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("WF_DATASET", DATASET)
os.environ.setdefault("H2_VERBOSE", "0")


def port_for(defense, level):
    return (
        9000
        + DEFENSE_ORDER.index(defense) * 100
        + LEVELS.index(level) * 10
        + int(DATASET.split("_")[0])
    )


def reachable(tcp_port):
    try:
        with socket.create_connection((args.ip, tcp_port), timeout=2):
            return True
    except OSError:
        return False


plan = []
for defense in defenses:
    levels = DEFENSE_LEVELS.get(defense, LEVELS)
    if args.levels:
        levels = [lv for lv in levels if lv in args.levels.split(",")]
    for level in levels:
        for name, target in places.items():
            # h2ps defends the origin it runs on; other placements are moot
            if defense == "h2ps" and name != "1st":
                continue
            plan.append((defense, level, name, target, port_for(defense, level)))

print(
    f"{DATASET}: {len(plan)} cells "
    f"({' '.join(defenses)} x {' '.join(places)}, {args.pages} pages)\n"
)

done = skipped = failed = 0
for defense, level, name, target, tcp_port in plan:
    tag = f"srv{defense}_{name}_{level}"
    out = WORKSPACE / f"ovh_{tag}.csv"

    if out.exists() and not args.force:
        print(f"skip  {tag}")
        skipped += 1
        continue
    if args.dry_run:
        print(f"would run {tag}  port {tcp_port}  target {target}")
        continue
    if not reachable(tcp_port):
        print(f"!!! {tcp_port}  {defense} {level}  not reachable, skipping")
        failed += 1
        continue

    print(f"=== {tag}  port {tcp_port}  target {target}", flush=True)
    rc = subprocess.run(
        [
            sys.executable,
            "./approximate_overhead.py",
            "--dst_ip",
            args.ip,
            "--dst_port",
            str(tcp_port),
            "--pages",
            str(args.pages),
            "--workspace",
            str(WORKSPACE),
            "--request_server_defense",
            target,
            "--tag",
            tag,
        ]
    ).returncode
    if rc == 0:
        done += 1
    else:
        print(f"!!! {tag} failed")
        failed += 1

print(f"\n{done} run, {skipped} skipped, {failed} failed")
if not args.dry_run and done:
    subprocess.run(
        [
            sys.executable,
            "./calibrate.py",
            "--workspace",
            str(WORKSPACE),
            "--baseline",
            "nop",
        ]
    )
