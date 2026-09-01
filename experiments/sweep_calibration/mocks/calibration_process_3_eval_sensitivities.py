import hashlib
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from wfaudit.helpers_ml import (
    evaluate_multiclass,
    load_best_params,
    load_from_file,
    save_to_file,
    tune,
)
from wfaudit.helpers_wefde.analysis.data_utils import load_wefde_features

parser = ArgumentParser()
parser.add_argument("-cell", "--cell", dest="cell", default=None)
parser.add_argument("--tune", action="store_true", default=False)
parser.add_argument("--cuda", action="store_true", default=True)
args = parser.parse_args()

ARCH_2D = ["kfp", "xgboost"]

if bool(args.cuda):
    ARCH_3D = ["robustfp"]
else:
    ARCH_3D = []

STRENGTH_ORDER = ["vlow", "low", "lomid", "mid1", "mid2", "high", "vhigh", "vvhigh"]
TUNE_STRENGTH = "lomid"
TUNE_TRIALS = 10
TUNE_TIMEOUT = None  # seconds
TUNE_PER_CLASS = 150
TUNE_EPOCHS = 60

testcase = Path(__file__).parent.name
cat = Path(__file__).parent.parent.name
WORKSPACE = Path(f"/http2/experiments/{cat}/{testcase}/")
RESULTS = WORKSPACE / "results"
TUNING = WORKSPACE / "tuning"


def params_sig(kwargs):
    """Suffix identifying a configuration, so re-tuning does not reuse the old
    result. Empty parameters give an empty suffix, keeping untuned filenames."""
    if not kwargs:
        return ""
    blob = json.dumps(kwargs, sort_keys=True, default=str)
    return "_" + hashlib.md5(blob.encode()).hexdigest()[:8]


def load_deepse_data(data_path):
    data = np.load(data_path)
    return data["traces"].astype("float32"), data["labels"].astype("float32")


def load_data(arch, cell):
    workspace = cell / "tcp_repr"
    if arch in ARCH_2D:
        path = workspace / "output_features"
        return load_wefde_features(path) if path.is_dir() else None
    path = workspace / "output_deepse/real/dataset.npz"
    return load_deepse_data(path) if path.exists() else None


def family(cell):
    return cell.name.split("_")[0]


def preference(cell):
    name = cell.name.partition("_")[2]
    if name == TUNE_STRENGTH:
        return len(STRENGTH_ORDER)
    return STRENGTH_ORDER.index(name) if name in STRENGTH_ORDER else -1


all_cells = sorted(d for d in RESULTS.iterdir() if (d / "tcp_repr").is_dir())
if not all_cells:
    sys.exit(f"no cells under {RESULTS}")

references = {}
for cell in all_cells:
    fam = family(cell)
    if fam not in references or preference(cell) > preference(references[fam]):
        references[fam] = cell

cells = all_cells
if args.cell:
    cells = [c for c in all_cells if c.name == args.cell]
    if not cells:
        sys.exit(f"no cell named {args.cell}")

params = {}
if args.tune:
    TUNING.mkdir(parents=True, exist_ok=True)
    for fam, ref in sorted(references.items()):
        for arch in ARCH_2D + ARCH_3D:
            data = load_data(arch, ref)
            if data is None:
                print(f"tuning {fam}/{arch}: SKIP - no data in {ref.name}", flush=True)
                continue
            X, Y = data
            try:
                params[fam, arch] = load_best_params(arch, TUNING, X=X, y=Y)
            except FileNotFoundError:
                print(f"tuning {fam}/{arch} on {ref.name} {X.shape}", flush=True)
                res = tune(
                    arch,
                    X,
                    Y,
                    n_trials=TUNE_TRIALS,
                    timeout=TUNE_TIMEOUT,
                    per_class=TUNE_PER_CLASS,
                    epoch_budget=TUNE_EPOCHS,
                    workspace=TUNING,
                    dataset_tag=ref.name,
                )
                print(
                    f"    baseline={res['baseline_value_macro_f1']} "
                    f"best={res['best_value_macro_f1']:.4f}",
                    flush=True,
                )
                params[fam, arch] = load_best_params(arch, TUNING, X=X, y=Y)
            print(f"{fam}/{arch}: {params[fam, arch]}", flush=True)

suffix = "_tuned" if args.tune else ""

for i, cell in enumerate(cells, 1):
    path_output = cell / "tcp_repr" / "eval_ml"
    path_output.mkdir(parents=True, exist_ok=True)

    for arch in ARCH_2D + ARCH_3D:
        if args.tune and (family(cell), arch) not in params:
            print(
                f"[{i}/{len(cells)}] {cell.name:16s} {arch:9s} SKIP - not tuned",
                flush=True,
            )
            continue

        kwargs = dict(params.get((family(cell), arch), {}))
        backup_file = (
            path_output / f"scores_rawts_{arch}_topk{suffix}{params_sig(kwargs)}.bkp"
        )
        if backup_file.exists():
            score = load_from_file(backup_file)
            print(
                f"[{i}/{len(cells)}] {cell.name:16s} {arch:9s} cached  {score['str']}",
                flush=True,
            )
            continue

        data = load_data(arch, cell)
        if data is None:
            print(
                f"[{i}/{len(cells)}] {cell.name:16s} {arch:9s} SKIP - no data",
                flush=True,
            )
            continue
        X, Y = data
        print(
            f"[{i}/{len(cells)}] {cell.name:16s} {arch:9s} {X.shape} {kwargs}",
            flush=True,
        )
        score = evaluate_multiclass(
            arch=arch,
            label=f"topk{suffix}",
            data=X,
            labels=Y,
            workspace=path_output,
            use_cache=True,
            **kwargs,
        )
        save_to_file(backup_file, score)
        print(f"    {cell.name:16s}--{arch} {score['str']}", flush=True)
