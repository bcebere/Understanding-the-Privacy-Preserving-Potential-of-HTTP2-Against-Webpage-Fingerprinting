# stdlib
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from random import shuffle

import psutil
from wfaudit import process_raw_pcaps

parser = ArgumentParser()
parser.add_argument(
    "workspace",
    nargs="?",
    default=None,
    help="workspace root holding <dataset>/<cell>/traces " "(default: ./workspace)",
)
parser.add_argument("-dataset", "--dataset", dest="dataset", default=None)
parser.add_argument("-use_json", "--use_json", dest="use_json", default=1)
parser.add_argument("-jobs", "--jobs", dest="jobs", type=int, default=0)
parser.add_argument(
    "-watch",
    "--watch",
    dest="watch",
    type=int,
    default=0,
    help="Seconds between scans. 0 = single pass.",
)
args = parser.parse_args()

USE_JSON = int(args.use_json)

if args.jobs > 0:
    N_JOBS = args.jobs
else:
    usage = psutil.cpu_percent(interval=1, percpu=True)
    N_JOBS = max(20, min(50, sum(1 for u in usage if u < 40) - 1))
print(f"Total CPUs: {psutil.cpu_count()}  Using: {N_JOBS}")

WORKSPACE = Path(args.workspace or "workspace")
if not WORKSPACE.is_dir():
    sys.exit(f"no workspace at {WORKSPACE}")
DATASETS = (
    [args.dataset]
    if args.dataset
    else sorted(p.name for p in WORKSPACE.iterdir() if p.is_dir())
)


def process(traces):
    ws = traces.parent / "tcp_repr"
    ws.mkdir(parents=True, exist_ok=True)
    process_raw_pcaps(
        traces=traces,
        unlink_after_processing=1,
        buffer_tcp=False,
        n_jobs=N_JOBS,
        workspace=ws,
        use_json=USE_JSON,
    )


while True:
    files = [t for ds in DATASETS for t in (WORKSPACE / ds).glob("*/traces")]
    shuffle(files)
    for cell in files:
        n = len(list(cell.glob("*.pcap")))
        if not n:
            continue
        print(f"=== {cell.parent.name}: {n} pcaps", flush=True)
        try:
            process(cell)
        except BaseException as e:
            print(f"failed on {cell.parent.name}: {e}")

    if args.watch <= 0:
        break
    time.sleep(args.watch)
