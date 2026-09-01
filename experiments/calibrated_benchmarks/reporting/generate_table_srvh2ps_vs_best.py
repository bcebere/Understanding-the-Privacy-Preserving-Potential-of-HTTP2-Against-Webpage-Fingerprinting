#!/usr/bin/env python3
"""H2PS against the best single-server deployment of a competing defense.

    python3 generate_table_h2ps_vs_best.py

H2PS is first-party only by design, so the fair comparison is against ALPaCA
and SRV-Tamaraw deployed on ONE server -- either the first party or a CDN,
whichever leaks least for that competitor.  All-servers deployments are
excluded: they require every origin to cooperate, which is the assumption
H2PS avoids.

Overhead comes from the calibration sweep at the level main_table_config
selected, and at the matching placement.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd().parent / "mocks"))
from helpers import ml_models, security_results  # noqa: E402
from main_table_config import SERVER  # noqa: E402

workspace_name = Path.cwd().parent.name
WORKSPACE = f"/http2/experiments/{workspace_name}/"


CALIBRATION = Path("/http2/experiments/sweep_calibration")

DATASETS = ["1_amazon", "2_bbc", "3_reddit", "4_udemy", "5_wiki"]
PRETTY_DATASET = {
    "1_amazon": "Amazon",
    "2_bbc": "BBC",
    "3_reddit": "Reddit",
    "4_udemy": "Udemy",
    "5_wiki": "Wiki",
}
PRETTY_DEFENSE = {
    "srvdef_alpaca_html": "ALP (1st)",
    "srvdef_alpaca_cdn1": "ALP (CDN)",
    "srvdef_tamaraw_html": "TAM (1st)",
    "srvdef_tamaraw_cdn1": "TAM (CDN)",
    "srvdef_h2ps": "H2PS",
}

# security_results name -> (overhead defense name, main_table_config key)
OVH_NAME = {
    "srvdef_alpaca_html": ("srvalpaca_1st", "alpaca"),
    "srvdef_alpaca_cdn1": ("srvalpaca_3rd_1", "alpaca"),
    "srvdef_tamaraw_html": ("srvtamaraw_1st", "tamaraw"),
    "srvdef_tamaraw_cdn1": ("srvtamaraw_3rd_1", "tamaraw"),
    "srvdef_h2ps": ("srvh2ps_1st", "h2ps"),
}
OURS = "srvdef_h2ps"
COMPETITORS = [
    "srvdef_alpaca_html",
    "srvdef_alpaca_cdn1",
    "srvdef_tamaraw_html",
    "srvdef_tamaraw_cdn1",
]

res = security_results(WORKSPACE)


def overhead_table(dataset):
    path = CALIBRATION / dataset / "overhead" / "overhead_summary.csv"
    if not path.exists():
        print(f"WARNING: no overhead summary for {dataset}")
        return {}
    return {(r["defense"], r["level"]): r for r in csv.DictReader(open(path))}


def best_attacker(row, prefix):
    scores = {}
    for m in ml_models:
        mean = f"{prefix}_{m}_mean"
        if mean in row and not pd.isna(row[mean]):
            scores[m] = (float(row[mean]), row.get(f"{prefix}_{m}_std", np.nan))
    if not scores:
        return np.nan, np.nan
    return scores[max(scores, key=lambda m: scores[m][0])]


def cell_for(dataset, defense, ovh):
    hit = res[(res["dataset"] == dataset) & (res["defense"] == defense)]
    if hit.empty:
        return None
    r = hit.iloc[0]
    f1, f1_err = best_attacker(r, "f1_score_macro")
    if np.isnan(f1):
        return None
    top5, _ = best_attacker(r, "acc_top5")

    ovh_name, short = OVH_NAME.get(defense, (defense, defense))
    level = SERVER.get(dataset, {}).get(short)
    row = ovh.get((ovh_name, level), {})

    def num(col):
        try:
            return float(row[col])
        except (KeyError, TypeError, ValueError):
            return np.nan

    return dict(
        defense=defense,
        level=level,
        f1=f1,
        f1_err=f1_err,
        top5=top5,
        kstar=float(r["anon_cons"]) if not pd.isna(r.get("anon_cons")) else np.nan,
        dUpB=num("dUpB"),
        dDownB=num("dDownB"),
        dT=num("dT"),
    )


rows = []
for dataset in DATASETS:
    ovh = overhead_table(dataset)
    ours = cell_for(dataset, OURS, ovh)
    rivals = [c for c in (cell_for(dataset, d, ovh) for d in COMPETITORS) if c]
    if ours is None or not rivals:
        print(
            f"WARNING: {PRETTY_DATASET[dataset]} incomplete "
            f"(h2ps={'yes' if ours else 'no'}, {len(rivals)} competitors)"
        )
        continue
    best = min(rivals, key=lambda c: c["f1"])
    rows.append((PRETTY_DATASET[dataset], best, ours))

if not rows:
    sys.exit("nothing to report")


def show(v, d=2):
    return "--" if v is None or np.isnan(v) else f"{v:.{d}f}"


print(
    f"\n{'Dataset':8s} {'Defense':14s} {'Lvl':6s} {'F1':>16s} "
    f"{'Top-5':>7s} {'K*':>7s} {'dT':>6s} {'dDown':>7s}"
)
for pretty, best, ours in rows:
    for i, c in enumerate((best, ours)):
        print(
            f"{pretty if i == 0 else '':8s} "
            f"{PRETTY_DEFENSE.get(c['defense'], c['defense']):14s} "
            f"{str(c['level']):6s} "
            f"{c['f1']:8.3f} +-{0 if np.isnan(c['f1_err']) else c['f1_err']:.3f} "
            f"{show(c['top5']):>7s} {show(c['kstar']):>7s} "
            f"{show(c['dT'], 1):>6s} {show(c['dDownB'], 1):>7s}"
        )
    print()


def tex(v, err=None, digits=2):
    if v is None or np.isnan(v):
        return "--"
    s = f"{v:.{digits}f}"
    if err is not None and not np.isnan(err):
        s += f" \\pm {err:.2f}"
    return f"${s}$"


print("=" * 60)
print("LATEX body")
print("=" * 60)
for i, (pretty, best, ours) in enumerate(rows):
    print(f"    \\multirow{{2}}{{*}}{{{pretty}}}")
    for c in (best, ours):
        name = PRETTY_DEFENSE.get(c["defense"], c["defense"])
        print(
            f"{'':28} & {name:<14} "
            f"& {tex(c['f1'], c['f1_err'])} "
            f"& {tex(c['top5'])} "
            f"& {tex(c['kstar'])} "
            f"& {tex(c['dDownB'], digits=1)} "
            f"& {tex(c['dT'], digits=1)} \\\\"
        )
    print(r"    \midrule" if i < len(rows) - 1 else "")
