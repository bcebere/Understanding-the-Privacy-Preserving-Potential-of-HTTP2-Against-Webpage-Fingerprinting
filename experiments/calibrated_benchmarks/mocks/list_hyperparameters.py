#!/usr/bin/env python3
"""Hyperparameters selected per (cell, attacker).

    python3 list_hyperparams.py                    # this dataset
    python3 list_hyperparams.py --all              # every dataset
    python3 list_hyperparams.py --csv hp.csv       # machine readable
    python3 list_hyperparams.py --latex            # appendix table

Reads the Optuna best_params_*.json files written by wfaudit tuning.  A cell
with no file ran at the attacker's defaults, which is reported as "defaults"
rather than omitted -- that distinction is the point of the table.
"""

import csv
import json
import sys
from pathlib import Path

ARCHS = ["kfp", "kfpv2", "xgboost", "robustfp", "varcnn", "df", "holmes"]
DATASETS = ["1_amazon", "2_bbc", "3_reddit", "4_udemy", "5_wiki"]

args = sys.argv[1:]
ALL = "--all" in args
LATEX = "--latex" in args
out_csv = None
if "--csv" in args:
    i = args.index("--csv")
    out_csv = args[i + 1] if i + 1 < len(args) else "hyperparams.csv"

HERE = Path.cwd()
CATEGORY = HERE.parent.name if not ALL else None
ROOT = Path("/http2/experiments")


def datasets():
    if not ALL:
        return [(HERE.name, ROOT / HERE.parent.name / HERE.name / "results")]
    out = []
    for cat in sorted(p.name for p in ROOT.iterdir() if p.is_dir()):
        for ds in DATASETS:
            r = ROOT / cat / ds / "results"
            if r.is_dir():
                out.append((f"{cat}/{ds}", r))
    return out


def scored_archs(cell):
    """Which attackers produced a score, and whether each was tuned."""
    found = {}
    for arch in ARCHS:
        for sub in ("eval_ml", "eval_ml_nn"):
            base = cell / "tcp_repr" / sub
            if not base.is_dir():
                continue
            for label in ("tuned", "topk"):
                if list(base.glob(f"scores_*{arch}_{label}*.bkp")):
                    found[arch] = label == "tuned"
                    break
            if arch in found:
                break
    return found


def best_params(cell, arch):
    """-> (params, source) from the Optuna study, if one ran."""
    hpo = cell / "tcp_repr" / "hpo"
    if not hpo.is_dir():
        return None, None
    hits = sorted(hpo.glob(f"best_params_{arch}_*.json"))
    if not hits:
        return None, None
    path = hits[-1]
    try:
        blob = json.load(open(path))
    except Exception:
        return None, path.name
    params = blob.get("params", blob.get("best_params", blob))
    if not isinstance(params, dict):
        return None, path.name
    # drop bookkeeping keys, keep the actual hyperparameters
    params = {
        k: v
        for k, v in params.items()
        if k not in ("study_name", "dataset_tag", "arch", "value")
    }
    return params, path.name


rows = []
for label, results in datasets():
    if not results.is_dir():
        continue
    for cell in sorted(results.iterdir()):
        if not (cell / "tcp_repr").is_dir():
            continue
        for arch, was_tuned in sorted(scored_archs(cell).items()):
            params, src = best_params(cell, arch)
            # A study can exist while the reported score used the defaults --
            # that happens when the attacker was dropped from --tune-archs
            # after a study had already run.  Report what was USED.
            orphan = bool(params) and not was_tuned
            rows.append(
                dict(
                    dataset=label,
                    cell=cell.name,
                    arch=arch,
                    tuned=bool(was_tuned and params),
                    params=params if was_tuned else {},
                    orphan_study=orphan,
                    source=src or "",
                )
            )

if not rows:
    sys.exit("no scored cells found")

width = max(len(f"{r['dataset']}/{r['cell']}") for r in rows)
print(f"{'dataset/cell':{width}s} {'attacker':9s} {'tuned':6s} parameters")
last = None
for r in rows:
    key = f"{r['dataset']}/{r['cell']}"
    if last and key != last:
        print()
    last = key
    if r["params"]:
        p = "  ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in sorted(r["params"].items())
        )
    else:
        p = "defaults"
    note = "   (study exists, unused)" if r["orphan_study"] else ""
    print(f"{key:{width}s} {r['arch']:9s} {'yes' if r['tuned'] else 'no':6s} {p}{note}")

n_tuned = sum(1 for r in rows if r["tuned"])
print(
    f"\n{len(rows)} (cell, attacker) pairs; {n_tuned} tuned, "
    f"{len(rows) - n_tuned} at defaults"
)

if out_csv:
    keys = sorted({k for r in rows for k in r["params"]})
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["dataset", "cell", "arch", "tuned", "orphan_study", "source"]
            + keys,
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "dataset": r["dataset"],
                    "cell": r["cell"],
                    "arch": r["arch"],
                    "tuned": r["tuned"],
                    "orphan_study": r["orphan_study"],
                    "source": r["source"],
                    **r["params"],
                }
            )
    print(f"wrote {out_csv}")

if LATEX:
    print("\n\\begin{tabular}{lllp{7cm}}")
    print("\\toprule")
    print("Dataset & Defense & Attacker & Selected hyperparameters \\\\")
    print("\\midrule")
    for r in rows:
        p = (
            ", ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in sorted(r["params"].items())
            )
            or "defaults"
        )
        p = p.replace("_", "\\_")
        print(
            f"{r['dataset'].replace('_', ' ')} & "
            f"{r['cell'].replace('_', ' ')} & {r['arch']} & {p} \\\\"
        )
    print("\\bottomrule")
    print("\\end{tabular}")
