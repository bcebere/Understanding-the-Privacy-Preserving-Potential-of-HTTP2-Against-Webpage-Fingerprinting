#!/usr/bin/env python3
"""Attacker evaluation for the main tables.

    python3 benchmark_process_3_evaluate.py                  # kfp + robustfp, tuned
    python3 benchmark_process_3_evaluate.py --arch all
    python3 benchmark_process_3_evaluate.py --no-tune
    python3 benchmark_process_3_evaluate.py --cell h2pc

Tuning uses wfaudit.helpers_ml.tuning: an Optuna study per (arch, dataset),
scoped by a content hash of the data, so every cell gets its own search and
studies resume across invocations.  The winning configuration is then passed
to evaluate_multiclass on the full data.
"""

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from wfaudit.helpers_ml import evaluate_multiclass, load_from_file, save_to_file
from wfaudit.helpers_ml.tuning import load_best_params, top_trials, tune
from wfaudit.helpers_wefde.analysis.data_utils import load_wefde_features

ARCH_2D = {"kfp", "xgboost"}

TUNE_ARCHS = {"kfp", "robustfp", "holmes", "varcnn", "df"}
NN_ARCHS = {"varcnn", "holmes", "robustfp", "df"}
PRESETS = {
    "fast": ["kfp", "robustfp", "holmes"],
    "all": ["kfp", "robustfp", "varcnn", "df", "holmes"],
}

parser = ArgumentParser()
parser.add_argument(
    "-arch",
    "--arch",
    dest="arch",
    default="all",
    help="fast | all | comma-separated list",
)
parser.add_argument("-cell", "--cell", dest="cell", default=None)
parser.add_argument(
    "--no-tune",
    dest="tune",
    action="store_false",
    help="run every attacker at its defaults",
)
parser.add_argument(
    "-tune_archs",
    "--tune-archs",
    dest="tune_archs",
    default=",".join(sorted(TUNE_ARCHS)),
    help="comma-separated attackers to tune; the rest run " "at their defaults",
)
parser.add_argument("-trials", "--trials", dest="trials", type=int, default=10)
parser.add_argument(
    "-per_class",
    "--per_class",
    dest="per_class",
    type=int,
    default=150,
    help="traces per class used for the search",
)
parser.add_argument(
    "-epoch_budget",
    "--epoch_budget",
    dest="epoch_budget",
    type=int,
    default=60,
    help="epoch cap for NN trials",
)
parser.add_argument(
    "-timeout",
    "--timeout",
    dest="timeout",
    type=float,
    default=None,
    help="seconds per (cell, arch) search",
)
args = parser.parse_args()

ARCHS = PRESETS.get(args.arch, args.arch.split(","))
TUNE_ARCHS = set(a.strip() for a in args.tune_archs.split(",") if a.strip())

testcase = Path(__file__).parent.name
cat = Path(__file__).parent.parent.name
RESULTS = Path(f"/http2/experiments/{cat}/{testcase}/results")


def load_deepse_data(path):
    data = np.load(path)
    return data["traces"].astype("float32"), data["labels"].astype("float32")


def tuned_params(arch, X, Y, hpo_dir, tag):
    """Run or resume the study for this cell, returning the best parameters."""
    try:
        return load_best_params(arch, workspace=hpo_dir, X=X, y=Y, dataset_tag=tag)
    except BaseException:
        pass

    summary = tune(
        arch,
        X,
        Y,
        n_trials=args.trials,
        per_class=args.per_class,
        epoch_budget=args.epoch_budget,
        timeout=args.timeout,
        workspace=hpo_dir,
        dataset_tag=tag,
    )
    print(
        f"      defaults={summary.get('baseline_value_macro_f1')} "
        f"search={summary.get('search_value_macro_f1')} "
        f"-> {summary.get('selected')} "
        f"({summary.get('n_trials_run')} trials, "
        f"{summary.get('n_pruned')} pruned)",
        flush=True,
    )
    return load_best_params(arch, workspace=hpo_dir, X=X, y=Y, dataset_tag=tag)


cells = sorted(d for d in RESULTS.iterdir() if (d / "tcp_repr").is_dir())
if args.cell:
    cells = [d for d in cells if d.name == args.cell]
if not cells:
    sys.exit(f"no cells under {RESULTS}")

if args.tune:
    tuned = [a for a in ARCHS if a in TUNE_ARCHS]
    plain = [a for a in ARCHS if a not in TUNE_ARCHS]
    mode = (
        f"tuned: {' '.join(tuned) or 'none'} "
        f"({args.trials} trials, {args.per_class}/class)"
        + (f"; defaults: {' '.join(plain)}" if plain else "")
    )
else:
    mode = "defaults for every attacker"
print(f"{len(cells)} cells x {len(ARCHS)} attackers  ({mode})\n")

log = []

for i, cell in enumerate(cells, 1):
    workspace = cell / "tcp_repr"
    hpo_dir = workspace / "hpo"

    for arch in ARCHS:
        is2d = arch in ARCH_2D
        out = workspace / ("eval_ml" if is2d else "eval_ml_nn")
        out.mkdir(parents=True, exist_ok=True)
        do_tune = args.tune and arch in TUNE_ARCHS
        suffix = "tuned" if do_tune else "topk"
        backup = out / f"scores_rawts_{arch}_{suffix}.bkp"

        if backup.exists():
            score = load_from_file(backup)
            print(
                f"[{i}/{len(cells)}] {cell.name:18s} {arch:9s} cached  "
                f"{score.get('str', {}).get('f1_score_macro', '?')}",
                flush=True,
            )
            continue

        try:
            if is2d:
                features = workspace / "output_features"
                if not (features / "FeaturePositions.json").exists():
                    print(
                        f"[{i}/{len(cells)}] {cell.name:18s} {arch:9s} "
                        f"SKIP - no features",
                        flush=True,
                    )
                    continue
                X, Y = load_wefde_features(features)
            else:
                dataset = workspace / "output_deepse/real/dataset.npz"
                if not dataset.exists():
                    print(
                        f"[{i}/{len(cells)}] {cell.name:18s} {arch:9s} "
                        f"SKIP - no dataset.npz",
                        flush=True,
                    )
                    continue
                X, Y = load_deepse_data(dataset)

            print(f"[{i}/{len(cells)}] {cell.name:18s} {arch:9s} {X.shape}", flush=True)

            best = {}
            if do_tune:
                hpo_dir.mkdir(parents=True, exist_ok=True)
                best = tuned_params(arch, X, Y, hpo_dir, f"{testcase}_{cell.name}")
                print(f"      -> {best}", flush=True)

            score = evaluate_multiclass(
                arch=arch,
                label=suffix,
                data=X,
                labels=Y,
                workspace=out,
                use_cache=is2d,
                **best,
            )
            save_to_file(backup, score)

            if do_tune:
                try:
                    shortlist = top_trials(
                        arch,
                        k=3,
                        workspace=hpo_dir,
                        X=X,
                        y=Y,
                        dataset_tag=f"{testcase}_{cell.name}",
                    )
                except BaseException:
                    shortlist = []
                log.append(
                    dict(cell=cell.name, arch=arch, selected=best, top_trials=shortlist)
                )

            print(
                f"    {cell.name:18s} {arch:9s} "
                f"{score.get('str', {}).get('f1_score_macro', '?')}",
                flush=True,
            )

        except BaseException as exc:
            print(f"    {cell.name:18s} {arch:9s} FAILED: {exc}", flush=True)

if log:
    path = RESULTS.parent / "tuning_log.json"
    existing = []
    if path.exists():
        try:
            existing = json.load(open(path))
        except Exception:
            pass
    with open(path, "w") as fh:
        json.dump(existing + log, fh, indent=2, default=str)
    print(f"\nwrote {path}  ({len(existing) + len(log)} searches)")
