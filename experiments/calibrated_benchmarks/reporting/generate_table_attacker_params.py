#!/usr/bin/env python3
"""Strongest attacker and its selected hyperparameters, per dataset/defense.

    python3 generate_table_attacker_params.py
    python3 generate_table_attacker_params.py --dataset 5_wiki

Reads the score files to find which attacker won each cell, then the Optuna
summary in hpo/ for the parameters it ran with.  A cell whose winning attacker
was not tuned reports "defaults"; one where the search lost to the defaults
reports "defaults kept", which is the distinction the rebuttal needs.
"""
import json
import sys
from pathlib import Path

from main_table_config import cells
from wfaudit.helpers_ml import load_from_file

DATASETS = ["1_amazon", "2_bbc", "3_reddit", "4_udemy", "5_wiki"]
if "--dataset" in sys.argv:
    DATASETS = [sys.argv[sys.argv.index("--dataset") + 1]]

ARCHS = ["kfp", "xgboost", "robustfp", "varcnn", "df", "holmes"]
EVAL_DIRS = ("eval_ml_nn", "eval_ml")
ROOT = Path("/http2/experiments/calibrated_benchmarks")

PRETTY_DATASET = {
    "1_amazon": "Amazon",
    "2_bbc": "BBC",
    "3_reddit": "Reddit",
    "4_udemy": "Udemy",
    "5_wiki": "Wikipedia",
}
PRETTY_MODEL = {
    "robustfp": "RobustFP-CNN",
    "holmes": "Holmes",
    "kfp": "k-FP",
    "xgboost": "XGBoost",
    "varcnn": "Var-CNN",
    "df": "DF",
}
PRETTY_DEFENSE = {
    "nop": "Undefended",
    "front": "FRONT",
    "h2pc": "H2PC",
    "httpos": "HTTPOS",
    "llama": "LLaMA",
    "tamaraw": "CL-Tamaraw",
    "srvalpaca_1st": "SRV-ALPaCA (1st)",
    "srvalpaca_3rd_1": "SRV-ALPaCA (3rd)",
    "srvalpaca_all": "SRV-ALPaCA (all)",
    "srvtamaraw_1st": "SRV-Tamaraw (1st)",
    "srvtamaraw_3rd_1": "SRV-Tamaraw (3rd)",
    "srvtamaraw_all": "SRV-Tamaraw (all)",
    "srvh2ps1p": "H2PS (1st)",
}
# short names for the table, in the order they should appear
SHORT_PARAM = {
    "batch_size": "bs",
    "dropout": "do",
    "dropout_conv": "do",
    "lr": "lr",
    "weight_decay": "wd",
    "n_estimators": "trees",
    "n_neighbours": "k",
    "max_depth": "depth",
    "eta": "eta",
    "subsample": "sub",
}
PARAM_ORDER = ["bs", "do", "lr", "wd", "trees", "k", "depth", "eta", "sub"]


def score_of(path):
    s = load_from_file(path)
    raw = s.get("raw", s) if hasattr(s, "get") else s
    v = raw.get("f1_score_macro") if hasattr(raw, "get") else None
    if isinstance(v, (list, tuple)):
        v = v[0]
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def winner_of(cell):
    """-> (arch, was_tuned) for the strongest attacker on this cell."""
    best, best_v, tuned = None, -1.0, False
    for arch in ARCHS:
        for sub in EVAL_DIRS:
            base = cell / "tcp_repr" / sub
            if not base.is_dir():
                continue
            for label in ("tuned", "topk"):
                hits = sorted(base.glob(f"scores_*{arch}_{label}*.bkp"))
                if not hits:
                    continue
                v = score_of(hits[0])
                if v is not None and v > best_v:
                    best, best_v, tuned = arch, v, label == "tuned"
                break
            else:
                continue
            break
    return best, best_v, tuned


def params_of(cell, arch, tuned):
    if not tuned:
        return "defaults"
    hpo = cell / "tcp_repr" / "hpo"
    hits = sorted(hpo.glob(f"best_params_{arch}_*.json")) if hpo.is_dir() else []
    if not hits:
        return "tuned (summary missing)"
    try:
        blob = json.load(open(hits[-1]))
    except Exception:
        return "tuned"

    params = blob.get("best_params") or {}
    if blob.get("selected") == "defaults" or not params:
        return "defaults kept"

    named = {}
    for k, v in params.items():
        short = SHORT_PARAM.get(k, k)
        if isinstance(v, float):
            if v == 0:
                named[short] = "0"
            elif abs(v) < 0.01:
                named[short] = f"{v:.1e}".replace("e-0", "e-")
            else:
                named[short] = f"{v:.3f}".rstrip("0").rstrip(".").lstrip("0") or "0"
        else:
            named[short] = str(v)
    ordered = [f"{k} {named[k]}" for k in PARAM_ORDER if k in named]
    ordered += [f"{k} {v}" for k, v in named.items() if k not in PARAM_ORDER]
    return ", ".join(ordered)


rows = []
for dataset in DATASETS:
    results = ROOT / dataset / "results"
    if not results.is_dir():
        print(f"WARNING: no results for {dataset}")
        continue
    for kind, _, _, tag, _ in cells(dataset):
        cell = results / tag
        if not (cell / "tcp_repr").is_dir():
            continue
        arch, f1, tuned = winner_of(cell)
        if arch is None:
            continue
        rows.append(
            dict(
                dataset=dataset,
                kind=kind,
                tag=tag,
                arch=arch,
                f1=f1,
                params=params_of(cell, arch, tuned),
            )
        )

if not rows:
    sys.exit("no scored cells")

width = max(len(PRETTY_DEFENSE.get(r["tag"], r["tag"])) for r in rows)
for dataset in DATASETS:
    sub = [r for r in rows if r["dataset"] == dataset]
    if not sub:
        continue
    print(f"\n{PRETTY_DATASET.get(dataset, dataset)}")
    for r in sub:
        print(
            f"  {PRETTY_DEFENSE.get(r['tag'], r['tag']):{width}s} "
            f"{PRETTY_MODEL.get(r['arch'], r['arch']):13s} "
            f"{r['f1']:.3f}  {r['params']}"
        )

print("\n" + "=" * 70)
print("LATEX")
print("=" * 70)
for dataset in DATASETS:
    sub = [r for r in rows if r["dataset"] == dataset]
    if not sub:
        continue
    print(
        f"    \\multicolumn{{3}}{{@{{}}l}}{{\\emph{{{PRETTY_DATASET[dataset]}}}}} \\\\"
    )
    for kind in ("client", "server"):
        block = [r for r in sub if r["kind"] == kind]
        if not block:
            continue
        for r in block:
            name = PRETTY_DEFENSE.get(r["tag"], r["tag"])
            model = PRETTY_MODEL.get(r["arch"], r["arch"])
            print(f"    {name:<18} & {model:<13} & {r['params']} \\\\")
        if kind == "client" and any(r["kind"] == "server" for r in sub):
            print(r"    \addlinespace")
    print(r"    \addlinespace")
