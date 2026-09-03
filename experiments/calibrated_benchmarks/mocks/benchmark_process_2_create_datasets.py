import sys
from argparse import ArgumentParser
from pathlib import Path

from wfaudit import prepare_all_datasets

parser = ArgumentParser()
parser.add_argument(
    "workspace",
    nargs="?",
    default=None,
    help="workspace root holding <dataset>/<cell>/tcp_repr " "(default: ./workspace)",
)
parser.add_argument("-dataset", "--dataset", dest="dataset", default=None)
parser.add_argument(
    "-cell",
    "--cell",
    dest="cell",
    default=None,
    help="Process one cell only, e.g. front_low",
)
args = parser.parse_args()

WORKSPACE = Path(args.workspace or "workspace")
if not WORKSPACE.is_dir():
    sys.exit(f"no workspace at {WORKSPACE}")
DATASETS = (
    [args.dataset]
    if args.dataset
    else sorted(p.name for p in WORKSPACE.iterdir() if p.is_dir())
)
N_TRACES = 500

cells = sorted(
    d
    for ds in DATASETS
    for d in (WORKSPACE / ds).iterdir()
    if (d / "tcp_repr").is_dir()
)
if args.cell:
    cells = [d for d in cells if d.name == args.cell]
if not cells:
    sys.exit(f"no cells under {WORKSPACE}")

for i, cell in enumerate(cells, 1):
    workspace = cell / "tcp_repr"

    if (
        (workspace / "output_ml/X_1C.npy").exists()
        and (workspace / "output_features_global/FeaturePositions.json").exists()
        and (workspace / "output_features_multi/FeaturePositions.json").exists()
    ):
        print(f"[{i}/{len(cells)}] skip {cell.parent.name}/{cell.name}", flush=True)
        continue

    print(f"[{i}/{len(cells)}] {cell.parent.name}/{cell.name}", flush=True)
    prepare_all_datasets(workspace=workspace, n_traces=N_TRACES)
