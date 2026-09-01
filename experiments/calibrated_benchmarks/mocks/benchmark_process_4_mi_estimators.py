#!/usr/bin/env python3
"""Leakage estimators for the main tables: WeFDE and DeepSE-WF.

    python3 benchmark_process_4_mi_estimators.py
    python3 benchmark_process_4_mi_estimators.py --only deepse --device cuda
    python3 benchmark_process_4_mi_estimators.py --cell h2pc

Reads tcp_repr/output_features and tcp_repr/output_deepse/real/dataset.npz;
writes results next to eval_ml / eval_ml_nn:

    tcp_repr/eval_wefde/leakage.csv          MI_TOTAL, per-category MI
    tcp_repr/eval_deepse/results_df.csv      MI_TOTAL, BER_LO, BER_HI, ACC

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
from wfaudit import evaluate_leakage_from_deepse, evaluate_leakage_from_wefde

parser = ArgumentParser()
parser.add_argument("-cell", "--cell", dest="cell", default=None)
parser.add_argument(
    "-only",
    "--only",
    dest="only",
    default="deepse",
    choices=["both", "wefde", "deepse"],
)
parser.add_argument("-device", "--device", dest="device", default="cuda")
parser.add_argument("-model", "--model", dest="model", default="df")
parser.add_argument("-epochs", "--epochs", dest="epochs", type=int, default=1000)
parser.add_argument("-n_procs", "--n_procs", dest="n_procs", type=int, default=10)
args = parser.parse_args()

testcase = Path(__file__).parent.name
cat = Path(__file__).parent.parent.name
RESULTS = Path(f"/http2/experiments/{cat}/{testcase}/results")

cells = sorted(d for d in RESULTS.iterdir() if (d / "tcp_repr").is_dir())
if args.cell:
    cells = [d for d in cells if d.name == args.cell]
if not cells:
    sys.exit(f"no cells under {RESULTS}")

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
    ws = cell / "tcp_repr"
    tag = f"[{i}/{len(cells)}] {cell.name:18s}"

    if args.only in ("both", "wefde"):
        feats = ws / "output_features"
        target = ws / "eval_wefde/leakage.csv"
        available_cpus = get_cpu_count()

        if target.exists():
            print(f"{tag} wefde  cached", flush=True)
        elif not (feats / "FeaturePositions.json").exists():
            print(f"{tag} wefde  SKIP - no output_features", flush=True)
        else:
            print(f"{tag} wefde", flush=True)
            df = evaluate_leakage_from_wefde(
                wefde_output_folder=ws / "eval_wefde",
                wefde_feats_folder=feats,
                n_procs=min(max(10, available_cpus), 60),
                topn=20,
                nmi_threshold=0.7,
            )
            assert df is not None

            target.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(target, index=False)
            mi = df["MI_TOTAL"].values[0] if "MI_TOTAL" in df else float("nan")
            print(f"    wefde MI_TOTAL={mi:.3f} -> {target}", flush=True)

    if args.only in ("both", "deepse"):
        dataset = ws / "output_deepse/real/dataset.npz"
        target = ws / f"eval_deepse/results_{args.model}.csv"
        if target.exists():
            print(f"{tag} deepse cached", flush=True)
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
