#!/usr/bin/env python3
"""H2PC against two reference points per dataset.

    python3 generate_table_h2pc_vs_best.py

Three rows per dataset:
  * strongest -- the competing defense with the lowest attacker Macro-F1,
    regardless of what it costs
  * closest cost -- the competing defense whose download overhead is nearest
    H2PC's, which is the matched-overhead comparison
  * H2PC

Overhead is used to pick the closest-cost row but is not reported here; the
cost figures live in the separate overhead table, so they are not duplicated.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_here = Path(__file__).parent  # main_table_config.py symlink sits next to it
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.resolve().parent / "mocks"))  # or ../mocks from reporting/
from helpers import ml_models, security_results  # noqa: E402
from main_table_config import CLIENT  # noqa: E402

WORKSPACE = (
    Path(sys.argv[1])
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
    else _here / "workspace"
)


CALIBRATION = WORKSPACE  # <workspace>/<dataset>/overhead/overhead_summary.csv

DATASETS = ["1_amazon", "2_bbc", "3_reddit", "4_udemy", "5_wiki"]
PRETTY_DATASET = {
    "1_amazon": "Amazon",
    "2_bbc": "BBC",
    "3_reddit": "Reddit",
    "4_udemy": "Udemy",
    "5_wiki": "Wiki",
}
PRETTY_DEFENSE = {
    "cldef_httpos": "HTTPOS",
    "cldef_llama": "LLaMA",
    "cldef_front": "FRONT",
    "cldef_tamaraw": "CL-TAM",
    "cldef_h2pc": "H2PC",
}
# security_results name -> the name used in main_table_config / overhead
SHORT = {
    "cldef_httpos": "httpos",
    "cldef_llama": "llama",
    "cldef_front": "front",
    "cldef_tamaraw": "tamaraw",
    "cldef_h2pc": "h2pc",
}
OURS = "cldef_h2pc"
COMPETITORS = ["cldef_httpos", "cldef_llama", "cldef_front", "cldef_tamaraw"]

res = security_results(WORKSPACE)


def overhead_table(dataset):
    """-> {(defense, level): row} from that dataset's calibration summary."""
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

    short = SHORT.get(defense, defense)
    level = CLIENT.get(dataset, {}).get(short)
    row = ovh.get((short, level), {})

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
        print(f"WARNING: {PRETTY_DATASET[dataset]} incomplete")
        continue

    strongest = min(rivals, key=lambda c: c["f1"])
    priced = [c for c in rivals if not np.isnan(c["dDownB"])]
    closest = (
        min(priced, key=lambda c: abs(c["dDownB"] - ours["dDownB"]))
        if priced and not np.isnan(ours["dDownB"])
        else None
    )
    rows.append((PRETTY_DATASET[dataset], strongest, closest, ours))

if not rows:
    sys.exit("nothing to report")


def show(v, digits=2):
    return "--" if v is None or np.isnan(v) else f"{v:.{digits}f}"


print(
    f"\n{'Dataset':8s} {'Role':10s} {'Defense':8s} {'Lvl':6s} "
    f"{'dDown':>7s} {'dT':>6s} {'F1':>16s} {'Top-5':>7s} {'K*':>7s}"
)
for pretty, strongest, closest, ours in rows:
    entries = [("strongest", strongest)]
    if closest is not None and closest["defense"] != strongest["defense"]:
        entries.append(("closest cost", closest))
    entries.append(("ours", ours))
    for i, (role, c) in enumerate(entries):
        print(
            f"{pretty if i == 0 else '':8s} {role:10s} "
            f"{PRETTY_DEFENSE.get(c['defense'], c['defense']):8s} "
            f"{str(c['level']):6s} {show(c['dDownB']):>7s} {show(c['dT']):>6s} "
            f"{c['f1']:8.3f} +-{0 if np.isnan(c['f1_err']) else c['f1_err']:.3f} "
            f"{show(c['top5'], 3):>7s} {show(c['kstar']):>7s}"
        )
    print()


def tex(v, err=None, digits=3):
    if v is None or np.isnan(v):
        return "--"
    s = f"{v:.{digits}f}"
    if err is not None and not np.isnan(err):
        s += f" \\pm {err:.2f}"
    return f"${s}$"


print("=" * 60)
print("LATEX body")
print("=" * 60)
for i, (pretty, strongest, closest, ours) in enumerate(rows):
    entries = [strongest]
    if closest is not None and closest["defense"] != strongest["defense"]:
        entries.append(closest)
    entries.append(ours)

    print(f"    \\multirow{{{len(entries)}}}{{*}}{{{pretty}}}")
    for c in entries:
        name = PRETTY_DEFENSE.get(c["defense"], c["defense"])
        print(
            f"{'':28} & {name:<8} "
            f"& {tex(c['f1'], c['f1_err'], digits=2)} "
            f"& {tex(c['top5'], digits=2)} "
            f"& {tex(c['kstar'], digits=2)}"
            f"& {tex(c['dUpB'], digits=1)}"
            f"& {tex(c['dDownB'], digits=1)}"
            f"& {tex(c['dT'], digits=1)} \\\\"
        )
        # f"& {tex(c['dT'], digits=1)} \\\\")
    print(r"    \midrule" if i < len(rows) - 1 else "")
