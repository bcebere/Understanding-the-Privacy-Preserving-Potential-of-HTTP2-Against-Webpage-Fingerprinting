#!/usr/bin/env python3
"""Collect per-dataset results and plot client and server defenses separately.

    python3 plot_all.py

Client and server defenses answer different questions -- who deploys, and
which connections are covered -- and h2ps is measured 1st-party-only while
alpaca and tamaraw are measured at "all", so they get separate trade-off
figures.  nop appears in both as the undefended reference.
"""
import csv
import subprocess
import sys
from pathlib import Path

DATASETS = ["1_amazon", "2_bbc", "3_reddit", "4_udemy", "5_wiki"]
CLIENT_DEFENSES = ["front", "h2pc", "tamaraw", "httpos", "llama"]
SERVER_DEFENSES = ["srvalpaca", "srvtamaraw", "srvh2ps1p"]

HERE = Path.cwd()
DEFENSES = HERE / "defenses"
OUT = HERE / "out"
DEFENSES.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)


def run(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        print(f"!!! failed: {' '.join(str(c) for c in cmd)}")
    return r.returncode == 0


# ---- collect ---------------------------------------------------------
for dataset in DATASETS:
    src = HERE.parent / dataset
    if not src.is_dir():
        print(f"skip {dataset} (no such directory)")
        continue
    run(
        [
            sys.executable,
            "track_calibration_results.py",
            "--csv",
            str(DEFENSES / f"{dataset}.csv"),
        ],
        cwd=src,
    )

# ---- split -----------------------------------------------------------
server = set(SERVER_DEFENSES)
for src in sorted(DEFENSES.glob("*.csv")):
    if src.stem.endswith(("_client", "_server")):
        continue
    rows = list(csv.DictReader(open(src)))
    if not rows:
        print(f"  {src.name}: empty")
        continue
    fields = list(rows[0])
    for kind, keep in (
        ("client", lambda d: d not in server),
        ("server", lambda d: d in server),
    ):
        sel = [r for r in rows if keep(r["defense"]) or r["defense"] == "nop"]
        out = src.with_name(f"{src.stem}_{kind}.csv")
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(sel)
        print(f"  {out.name}: {len(sel)} rows")

# ---- plot ------------------------------------------------------------
client_csvs = sorted(str(p) for p in DEFENSES.glob("*_client.csv"))
server_csvs = sorted(str(p) for p in DEFENSES.glob("*_server.csv"))

for defense in CLIENT_DEFENSES:
    run(
        [
            sys.executable,
            "plot_defense.py",
            defense,
            *client_csvs,
            "--out",
            str(OUT / f"cldef_calibration_{defense}"),
        ]
    )

for defense in SERVER_DEFENSES:
    run(
        [
            sys.executable,
            "plot_defense.py",
            defense,
            *server_csvs,
            "--out",
            str(OUT / f"srvdef_calibration_{defense}"),
        ]
    )

if client_csvs:
    run(
        [
            sys.executable,
            "plot_tradeoff.py",
            *client_csvs,
            "--out",
            str(OUT / "cldef_tradeoff"),
        ]
    )
if server_csvs:
    run(
        [
            sys.executable,
            "plot_tradeoff.py",
            *server_csvs,
            "--out",
            str(OUT / "srvdef_tradeoff"),
        ]
    )

print(f"\nwrote {len(list(OUT.glob('*.png')))} figures to {OUT}")
