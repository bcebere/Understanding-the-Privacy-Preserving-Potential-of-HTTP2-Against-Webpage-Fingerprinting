#!/usr/bin/env python3
"""Leakage estimators for the main tables: WeFDE and DeepSE-WF.

    python3 benchmark_process_4_mi_estimators.py [workspace]
    python3 benchmark_process_4_mi_estimators.py --only deepse --device cuda
    python3 benchmark_process_4_mi_estimators.py --cell h2pc --dataset 4_udemy

Per cell it reads <defense>/wefdetraces and <defense>/deepsetraces/real/dataset.npz;
writes results next to eval_ml / eval_ml_nn, creating benchmarks/ if missing:

    <defense>/benchmarks/eval_wefde/leakage.csv       MI_TOTAL, per-category MI
    <defense>/benchmarks/eval_deepse/results_df.csv   MI_TOTAL, BER_LO, BER_HI, ACC

evaluate_leakage_from_wefde returns a DataFrame and writes nothing, so the
result is saved here.
"""

import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from wfaudit import evaluate_leakage_from_deepse, evaluate_leakage_from_wefde

parser = ArgumentParser()
parser.add_argument(
    "workspace",
    nargs="?",
    default=None,
    help="workspace root holding <dataset>/<defense>/ (default: ./workspace "
    "next to this script)",
)
parser.add_argument("-cell", "--cell", dest="cell", default=None)
parser.add_argument("-dataset", "--dataset", dest="dataset", default=None)
parser.add_argument(
    "-only",
    "--only",
    dest="only",
    default="deepse",
    choices=["both", "wefde", "deepse"],
)
parser.add_argument(
    "-device",
    "--device",
    dest="device",
    default="cuda" if torch.cuda.is_available() else "cpu",
)
parser.add_argument("-model", "--model", dest="model", default="df")
parser.add_argument(
    "-epochs", "--epochs", dest="epochs", type=int, default=2
)  # TODO 1000
parser.add_argument("-n_procs", "--n_procs", dest="n_procs", type=int, default=10)
args = parser.parse_args()

WORKSPACE = (
    Path(args.workspace) if args.workspace else Path(__file__).parent / "workspace"
)


def find_cells(workspace, dataset=None, cell=None):
    """-> [<workspace>/<dataset>/<defense>] that hold input traces.

    Keyed on the trace dirs, not on benchmarks/, so cells whose results have
    not been produced (or extracted) yet are still picked up.
    """
    out = []
    for ds in sorted(p for p in workspace.iterdir() if p.is_dir()):
        if dataset and ds.name != dataset:
            continue
        for d in sorted(p for p in ds.iterdir() if p.is_dir()):
            if cell and d.name != cell:
                continue
            if (d / "deepsetraces").is_dir() or (d / "wefdetraces").is_dir():
                out.append(d)
    return out


if not WORKSPACE.is_dir():
    sys.exit(f"no workspace at {WORKSPACE}")
cells = find_cells(WORKSPACE, args.dataset, args.cell)
if not cells:
    sys.exit(f"no cells under {WORKSPACE}")

print(f"{len(cells)} cells  ({args.only}, device={args.device})\n")


def get_cpu_count():
    cpus_cnt = []
    for retry in range(3):
        # Get the CPU usage percentage for each CPU core
        cpu_usage = psutil.cpu_percent(interval=1, percpu=True)

        # Count available (idle) CPUs
        available_cpus_local = sum(1 for usage in cpu_usage if usage < 50) - 1

        cpus_cnt.append(available_cpus_local)

        time.sleep(1)

    available_cpus = int(np.mean(cpus_cnt))
    print(f"Available CPUs: {np.mean(available_cpus)}")
    return available_cpus


for i, cell in enumerate(cells, 1):
    bench = cell / "benchmarks"
    label = f"{cell.parent.name}/{cell.name}"
    tag = f"[{i}/{len(cells)}] {label:24s}"

    if args.only in ("both", "wefde"):
        feats = cell / "wefdetraces"
        target = bench / "eval_wefde/leakage.csv"
        available_cpus = get_cpu_count()

        if target.exists():
            mi = pd.read_csv(target)["MI_TOTAL"].values[0]
            print(f"{tag} wefde  cached MI_TOTAL={mi:.3f}", flush=True)
        elif not (feats / "FeaturePositions.json").exists():
            print(f"{tag} wefde  SKIP - no wefdetraces", flush=True)
        else:
            print(f"{tag} wefde", flush=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            df = evaluate_leakage_from_wefde(
                wefde_output_folder=bench / "eval_wefde",
                wefde_feats_folder=feats,
                n_procs=min(max(10, available_cpus), 60),
                topn=20,
                nmi_threshold=0.7,
            )
            assert df is not None

            df.to_csv(target, index=False)
            mi = df["MI_TOTAL"].values[0] if "MI_TOTAL" in df else float("nan")
            print(f"    wefde MI_TOTAL={mi:.3f} -> {target}", flush=True)

    if args.only in ("both", "deepse"):
        dataset = cell / "deepsetraces/real/dataset.npz"
        target = bench / f"eval_deepse/results_{args.model}.csv"
        if target.exists():
            df = pd.read_csv(target)
            print(
                f"{tag} deepse cached MI={df['MI_TOTAL'].mean():.3f} "
                f"BER={df['BER_LO'].mean():.3f}-{df['BER_HI'].mean():.3f}",
                flush=True,
            )

        elif not dataset.exists():
            print(f"{tag} deepse SKIP - no dataset.npz", flush=True)
        else:
            print(f"{tag} deepse", flush=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            evaluate_leakage_from_deepse(
                deepse_dataset=dataset,
                deepse_output=str(target),
                model=args.model,
                device=args.device,
                epochs=args.epochs,
            )
            df = pd.read_csv(target)
            print(
                f"    deepse MI={df['MI_TOTAL'].mean():.3f} "
                f"BER={df['BER_LO'].mean():.3f}-{df['BER_HI'].mean():.3f}",
                flush=True,
            )
