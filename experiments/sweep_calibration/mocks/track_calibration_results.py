#!/usr/bin/env python3
"""Calibration sweep status: best attacker F1 per cell, joined with overhead.

    python3 track_calibration_results.py [arch ...]
    python3 track_calibration_results.py --csv results.csv

Run from a dataset dir.  Reads scores from results/<cell>/tcp_repr/eval_ml*
and overhead from <workspace>/overhead/overhead_summary.csv.
"""

import csv
import re
import sys
from pathlib import Path

from wfaudit.helpers_ml import load_from_file

argv = [a for a in sys.argv[1:] if not a.startswith("--")]
out_csv = None
if "--csv" in sys.argv:
    i = sys.argv.index("--csv")
    out_csv = sys.argv[i + 1] if i + 1 < len(sys.argv) else "results.csv"
    argv = [a for a in argv if a != out_csv]

DEBUG = "--debug" in sys.argv
ARCHS = argv or ["robustfp", "xgboost", "kfp"]
EVAL_DIRS = ("eval_ml_nn", "eval_ml")
ORDER = ["vlow", "low", "lomid", "mid1", "mid2", "high", "vhigh", "vvhigh"]

ds = Path.cwd().name
cat = Path.cwd().parent.name
WORKSPACE = Path(f"/http2/experiments/{cat}/{ds}")
RESULTS = WORKSPACE / "results"
OVERHEAD = WORKSPACE / "overhead"

OVH = {}
summary = OVERHEAD / "overhead_summary.csv"
if summary.exists():
    for r in csv.DictReader(open(summary)):
        OVH[(r["defense"], r["level"])] = r


OVH_ALIAS = {
    "srvalpaca": "srvalpaca_all",
    "srvtamaraw": "srvtamaraw_all",
    "srvh2ps1p": "srvh2ps_1st",
}


def overhead_for(defense, level):
    return OVH.get((OVH_ALIAS.get(defense, defense), level), {})


def num(row, col):
    try:
        return float(row[col])
    except (KeyError, TypeError, ValueError):
        return None


def key(name):
    d, _, lv = name.rpartition("_")
    return (d, lv) if lv in ORDER else (name, "mid1")


def _pair(v):
    """-> (mean, ci95).  Scores are stored as (mean, ci) tuples; accept a bare
    number or a "0.412 +/- 0.01" string too."""
    if isinstance(v, (list, tuple)):
        if not v:
            return None, None
        m = _pair(v[0])[0]
        c = _pair(v[1])[0] if len(v) > 1 else None
        return m, c
    if isinstance(v, (int, float)):
        return float(v), None
    if isinstance(v, str):
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", v)
        if nums:
            return float(nums[0]), (float(nums[1]) if len(nums) > 1 else None)
    return None, None


def metrics_of(path):
    """-> {metric: (mean, ci95)}."""
    s = load_from_file(path)
    src = s.get("raw", s) if hasattr(s, "get") else s
    if not hasattr(src, "get"):
        return {}
    out = {}
    for k, v in src.items():
        m, c = _pair(v)
        if m is not None:
            out[k] = (m, c)
    if DEBUG and "f1_score_macro" not in out:
        print(f"  ! no f1_score_macro in {path.name}: {list(src)[:8]}", file=sys.stderr)
    return out


rows, pending = [], []
for cell in sorted(RESULTS.iterdir()):
    if not (cell / "tcp_repr").is_dir():
        continue
    per_arch, all_m = {}, {}
    for a in ARCHS:
        for sub in EVAL_DIRS:
            f_ = cell / "tcp_repr" / sub / f"scores_rawts_{a}_topk.bkp"
            if f_.exists():
                m = metrics_of(f_)
                if "f1_score_macro" in m:
                    per_arch[a] = m["f1_score_macro"]
                    all_m[a] = m
                break
    if not per_arch:
        pending.append(cell.name)
        continue

    best = max(per_arch, key=lambda a: per_arch[a][0])
    bm = all_m[best]
    f1, f1_ci = per_arch[best]
    top5, _ = bm.get("acc_top5", (None, None))
    d, lv = key(cell.name)
    # o = OVH.get((d, lv), OVH.get((cell.name, "mid1"), {}))
    o = overhead_for(d, lv)

    rows.append(
        dict(
            defense=d,
            level=lv,
            f1=f1,
            f1_ci=f1_ci,
            by=best,
            n=len(per_arch),
            top5=top5,
            top10=bm.get("acc_top10", (None,))[0],
            entropy=bm.get("entropy_mi", (None,))[0],
            dDownB=num(o, "dDownB"),
            dDownB_q1=num(o, "dDownB_q1"),
            dDownB_q3=num(o, "dDownB_q3"),
            dUpB=num(o, "dUpB"),
            dT=num(o, "dT"),
            dT_q1=num(o, "dT_q1"),
            dT_q3=num(o, "dT_q3"),
            n_pages=num(o, "n_pages"),
        )
    )

rows.sort(
    key=lambda r: (r["defense"], ORDER.index(r["level"]) if r["level"] in ORDER else 9)
)


def f(v, w=8, p=2):
    return f"{v:{w}.{p}f}" if isinstance(v, float) else " " * w


def overlaps(a, b):
    """Do two (mean, ci95) intervals overlap?"""
    if None in (a["f1"], b["f1"], a["f1_ci"], b["f1_ci"]):
        return False
    return (abs(a["f1"] - b["f1"])) <= (a["f1_ci"] + b["f1_ci"])


print(f"archs: {' '.join(ARCHS)}   {len(rows)} scored, {len(pending)} pending")
if not OVH:
    print("(no overhead/overhead_summary.csv -- overhead columns blank)")
print()
print(
    f"{'defense':9s} {'level':6s} {'maxF1':>16s} {'top5':>6s} {'by':>9s} {'n':>2s} "
    f"{'dDownB':>8s} {'(Q1-Q3)':>15s} {'dUpB':>7s} {'dT':>7s}"
)

last, prev = None, None
for r in rows:
    if last and r["defense"] != last:
        print()
        prev = None
    last = r["defense"]

    score = (
        f"{r['f1']:7.3f} +-{r['f1_ci']:.3f}"
        if isinstance(r["f1_ci"], float)
        else f"{r['f1']:7.3f}        "
    )
    iqr = ""
    if isinstance(r["dDownB_q1"], float) and isinstance(r["dDownB_q3"], float):
        iqr = f"({r['dDownB_q1']:.2f}-{r['dDownB_q3']:.2f})"
    t5 = f"{r['top5']:6.3f}" if isinstance(r["top5"], float) else " " * 6

    # flags: ~ level is within CI of the previous one; ^ F1 rose with cost
    flag = ""
    if prev is not None:
        if overlaps(prev, r):
            flag += " ~"
        elif r["f1"] > prev["f1"]:
            flag += " ^"
    if r["n"] != len(ARCHS):
        flag += " *"

    print(
        f"{r['defense']:9s} {r['level']:6s} {score:>16s} {t5} {r['by']:>9s} {r['n']:2d} "
        f"{f(r['dDownB'])} {iqr:>15s} {f(r['dUpB'], 7)} {f(r['dT'], 7)}{flag}"
    )
    prev = r

notes = []
if any(r["n"] != len(ARCHS) for r in rows):
    notes.append("*  not all architectures scored; maxF1 may still rise")
notes.append("~  within CI95 of the previous level (not a real difference)")
notes.append("^  F1 rose with cost, outside CI")
print("\n" + "\n".join(notes))

if pending:
    print("\npending: " + " ".join(pending))

if out_csv:
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else [])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv}")
