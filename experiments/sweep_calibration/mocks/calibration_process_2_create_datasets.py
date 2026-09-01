import sys
from argparse import ArgumentParser
from pathlib import Path

from wfaudit import prepare_all_datasets

parser = ArgumentParser()
parser.add_argument(
    "-cell",
    "--cell",
    dest="cell",
    default=None,
    help="Process one cell only, e.g. front_low",
)
args = parser.parse_args()

testcase = Path(__file__).parent.name
cat = Path(__file__).parent.parent.name
RESULTS = Path(f"/http2/experiments/{cat}/{testcase}/results")
N_TRACES = 100

cells = sorted(d for d in RESULTS.iterdir() if (d / "tcp_repr").is_dir())
if args.cell:
    cells = [d for d in cells if d.name == args.cell]
if not cells:
    sys.exit(f"no cells under {RESULTS}")

for i, cell in enumerate(cells, 1):
    workspace = cell / "tcp_repr"

    if (
        (workspace / "output_ml/X_1C.npy").exists()
        and (workspace / "output_features_global/FeaturePositions.json").exists()
        and (workspace / "output_features_multi/FeaturePositions.json").exists()
    ):
        print(f"[{i}/{len(cells)}] skip {cell.name}", flush=True)
        continue

    print(f"[{i}/{len(cells)}] {cell.name}", flush=True)
    prepare_all_datasets(workspace=workspace, n_traces=N_TRACES)
