#!/usr/bin/env python3
"""Main-table results: F1, Top-5 and K* per cell, with overhead.

    python3 track_benchmark_results.py
    python3 track_benchmark_results.py --csv results.csv

Run from a dataset dir.  F1 is the max over every scored attacker (BBQ 1);
K* is the max over both leakage estimators (BBQ 3).
"""

import csv
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd
from main_table_config import cells
from wfaudit.helpers_ml import load_from_file

ARCHS = ["kfp", "xgboost", "robustfp", "varcnn", "df", "holmes"]
SORT_BY_F1 = "--no-sort" not in sys.argv
N_CLASSES = 100
EVAL_DIRS = ("eval_ml_nn", "eval_ml")

out_csv = None
if "--csv" in sys.argv:
    i = sys.argv.index("--csv")
    out_csv = sys.argv[i + 1] if i + 1 < len(sys.argv) else "results.csv"

DATASET = Path.cwd().name
CATEGORY = Path.cwd().parent.name
WORKSPACE = Path(f"/http2/experiments/{CATEGORY}/{DATASET}")
RESULTS = WORKSPACE / "results"

OVH = {}
summary = Path(
    f"/http2/experiments/sweep_calibration/{DATASET}/overhead/overhead_summary.csv"
)
if summary.exists():
    for r in csv.DictReader(open(summary)):
        OVH[(r["defense"], r["level"])] = r


def to_float(v):
    if isinstance(v, (list, tuple)) and v:
        v = v[0]
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", v)
        if m:
            return float(m.group())
    return None


def _binary_entropy(p):
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def mi_from_f1(f1, n_classes=N_CLASSES):
    """Lower bound on MI from the observed F1, inverted Fano:
        H(Y|Y_hat) <= H(P_e) + P_e * log2(n - 1)
    Using the constant 1 in place of H(P_e) is only tight at P_e = 0.5; at
    F1 = 1 it caps K* at 2 instead of 1.  Valid for balanced datasets where
    F1 ~ accuracy."""
    H_Y = math.log2(n_classes)
    pe = 1.0 - f1
    mi = H_Y - _binary_entropy(pe) - pe * math.log2(n_classes - 1)
    return min(max(mi, 0.0), H_Y)


def anon_from_mi(mi_bits, n_classes=N_CLASSES):
    """Effective anonymity set, K* = 2^(H(Y) - MI)."""
    H_Y = math.log2(n_classes)
    mi = min(max(float(mi_bits), 0.0), H_Y)
    return min(max(2.0 ** (H_Y - mi), 1.0), 2.0**H_Y)


def mi_from_anon(kstar, n_classes=N_CLASSES):
    """Inverse of anon_from_mi, so an estimator K* can be compared with the
    F1-derived bound on the same scale."""
    H_Y = math.log2(n_classes)
    return min(max(H_Y - math.log2(max(float(kstar), 1.0)), 0.0), H_Y)


def metrics_of(path):
    s = load_from_file(path)
    src = s.get("raw", s) if hasattr(s, "get") else s
    if not hasattr(src, "get"):
        return {}
    out = {}
    for k, v in src.items():
        m = to_float(v)
        c = to_float(v[1]) if isinstance(v, (list, tuple)) and len(v) > 1 else None
        if m is not None:
            out[k] = (m, c)
    return out


def overhead_for(kind, defense, level, tag):
    """Calibration writes one row per (defense, level), where the defense name
    carries the placement: srvalpaca_1st, srvalpaca_all, srvh2ps_1st."""
    if kind != "server":
        return OVH.get((defense, level), {})
    # tag is srvalpaca_1st / srvtamaraw_all / srvh2ps1p
    print(kind, defense, level, tag)
    name = tag if not tag.startswith("srvh2ps") else "srvh2ps_1st"
    return OVH.get((name, level), {})


def selected_params(cell, arch):
    """What the winning attacker ran with, per the Optuna summary."""
    tuned = False
    for sub in ("eval_ml", "eval_ml_nn"):
        base = cell / "tcp_repr" / sub
        if base.is_dir() and list(base.glob(f"scores_*{arch}_tuned*.bkp")):
            tuned = True
            break
    if not tuned:
        return "defaults (not tuned)"

    hpo = cell / "tcp_repr" / "hpo"
    hits = sorted(hpo.glob(f"best_params_{arch}_*.json")) if hpo.is_dir() else []
    if not hits:
        return "tuned (summary missing)"
    try:
        blob = json.load(open(hits[-1]))
    except Exception:
        return "tuned (unreadable)"

    base_f1 = blob.get("baseline_value_macro_f1")
    srch_f1 = blob.get("search_value_macro_f1")
    trials = blob.get("n_trials_run")
    params = blob.get("best_params") or {}

    if blob.get("selected") == "defaults" or not params:
        return (
            f"defaults kept ({trials} trials, "
            f"search {srch_f1:.4f} <= baseline {base_f1:.4f})"
        )

    body = " ".join(
        f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
        for k, v in sorted(params.items())
    )
    return f"{body}  ({trials} trials, {base_f1:.4f} -> {srch_f1:.4f})"


def estimator_mi(cell, n_classes=N_CLASSES):
    """-> (mi_wefde, mi_deepse) in bits, either may be None."""
    ws = cell / "tcp_repr"
    H_Y = math.log2(n_classes)

    def read(path, col, agg="first"):
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path)
            v = df[col].mean() if agg == "mean" else df[col].values[0]
            return min(max(float(v), 0.0), H_Y)
        except Exception:
            return None

    mi_w = read(ws / "eval_wefde/leakage.csv", "MI_TOTAL")
    mi_d = None
    for f in (
        sorted((ws / "eval_deepse").glob("results_*.csv"))
        if (ws / "eval_deepse").is_dir()
        else []
    ):
        mi_d = read(f, "MI_TOTAL", agg="mean")
        if mi_d is not None:
            break

    return mi_w, mi_d


rows, pending = [], []
for kind, defense, level, tag, target in cells(DATASET):
    cell = RESULTS / tag
    per_arch, all_m = {}, {}
    for a in ARCHS:
        # tuned first, then defaults: an attacker excluded from --tune-archs
        # writes _topk and would otherwise be invisible
        hit = None
        for sub in EVAL_DIRS:
            for pattern in (f"scores_*{a}_tuned*.bkp", f"scores_*{a}_topk*.bkp"):
                found = (
                    sorted((cell / "tcp_repr" / sub).glob(pattern))
                    if (cell / "tcp_repr" / sub).is_dir()
                    else []
                )
                if found:
                    hit = found[0]
                    break
            if hit:
                break
        if hit:
            m = metrics_of(hit)
            if "f1_score_macro" in m:
                per_arch[a] = m["f1_score_macro"]
                all_m[a] = m
    if not per_arch:
        pending.append(tag)
        continue

    best = max(per_arch, key=lambda a: per_arch[a][0])
    f1, ci = per_arch[best]
    bm = all_m[best]

    # Reconcile the estimator K* with the observed F1.  An attacker reaching
    # F1 is evidence of at least mi_from_f1(F1) bits of leakage, so reporting
    # a larger anonymity set than that would overstate the defense.  Take the
    # stronger evidence and derive K* from it.
    mi_w, mi_d = estimator_mi(cell)
    mi_f1 = mi_from_f1(f1)
    mi_best = max(v for v in (mi_w, mi_d, mi_f1) if v is not None)
    kstar = anon_from_mi(mi_best)

    # Sanity: an estimator claiming less leakage than the classifier already
    # demonstrated is a lower bound that the attacker has beaten.  Not an
    # error, but worth seeing -- it means the estimator is the loose one.
    below = [
        name
        for name, v in (("wefde", mi_w), ("deepse", mi_d))
        if v is not None and v + 1e-9 < mi_f1
    ]
    below = []
    o = overhead_for(kind, defense, level, tag)
    rows.append(
        dict(
            cell=tag,
            defense=defense,
            level=level,
            target=target or "",
            f1=f1,
            f1_ci=ci,
            by=best,
            n_arch=len(per_arch),
            params=selected_params(cell, best),
            top5=bm.get("acc_top5", (None,))[0],
            top10=bm.get("acc_top10", (None,))[0],
            kstar=kstar,
            mi_wefde=mi_w,
            mi_deepse=mi_d,
            mi_f1=mi_f1,
            mi_best=mi_best,
            mi_below_f1=",".join(below),
            dDownB=to_float(o.get("dDownB")),
            dUpB=to_float(o.get("dUpB")),
            dT=to_float(o.get("dT")),
        )
    )

# strongest defense first: lowest attacker F1 at the top
if SORT_BY_F1:
    rows.sort(key=lambda r: r["f1"])

print(
    f"{DATASET}: {len(rows)} scored, {len(pending)} pending"
    + ("  (sorted by F1)" if SORT_BY_F1 else "")
    + "\n"
)
print(
    f"{'cell':20s} {'level':6s} {'maxF1':>15s} {'top5':>6s} "
    f"{'MIwef':>6s} {'MIdse':>6s} {'MIf1':>6s} {'K*':>6s} "
    f"{'by':>9s} {'n':>2s} {'dDownB':>8s} {'dT':>7s}"
)
for r in rows:
    f1 = (
        f"{r['f1']:6.3f} +-{r['f1_ci']:.3f}"
        if isinstance(r["f1_ci"], float)
        else f"{r['f1']:6.3f}       "
    )

    def fmt(v, w=8, p=2):
        return f"{v:{w}.{p}f}" if isinstance(v, float) else " " * w

    flag = f"  <- {r['mi_below_f1']} below F1" if r["mi_below_f1"] else ""
    print(
        f"{r['cell']:20s} {r['level']:6s} {f1:>15s} {fmt(r['top5'],6,3)} "
        f"{fmt(r['mi_wefde'],6,2)} {fmt(r['mi_deepse'],6,2)} "
        f"{fmt(r['mi_f1'],6,2)} {fmt(r['kstar'],6,2)} "
        f"{r['by']:>9s} {r['n_arch']:2d} "
        f"{fmt(r['dDownB'])} {fmt(r['dT'],7)}{flag}"
    )
    if r["params"] and r["params"] != "defaults":
        print(
            f"{'':20s} {'':6s} {'':>15s} {'':6s} {'':>6s} {'':>6s} {'':>6s} "
            f"{'':>6s} -> {r['params']}"
        )

if pending:
    print("\npending: " + " ".join(pending))

if out_csv and rows:
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv}")
