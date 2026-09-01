import json
import os
import random
import time
from argparse import ArgumentParser
from copy import deepcopy
from glob import glob
from pathlib import Path
from urllib.parse import urlparse

import sslkeylog
from scapy.config import conf
from scapy.sendrecv import AsyncSniffer
from scapy.utils import wrpcap
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("-dst_ip", "--dst_ip", dest="dst_ip", help="Destination IP")
parser.add_argument("-dst_port", "--dst_port", dest="dst_port", help="Destination Port")
parser.add_argument(
    "-repeats",
    "--repeats",
    dest="repeats",
    default=100,
    help="Subpage repeats. 500 for the main tables; 100 for the calibration sweep",
)
parser.add_argument(
    "-subpage_limit",
    "--subpage_limit",
    dest="subpage_limit",
    default=100,
    help="Subpage limit",
)
parser.add_argument(
    "-ifname", "--ifname", dest="ifname", help="Interface to use for capture"
)
parser.add_argument(
    "-defense",
    "--defense",
    dest="defense",
    default="nop",
    help="front | tamaraw | h2pc | httpos | llama | nop",
)
parser.add_argument(
    "-level",
    "--level",
    dest="level",
    default="mid1",
    choices=["vlow", "low", "lomid", "mid1", "mid2", "high", "vhigh", "vvhigh"],
    help="Defense intensity. mid1 = the submitted configuration.",
)
parser.add_argument(
    "-dataset",
    "--dataset",
    dest="dataset",
    default=None,
    help="Dataset name for per-dataset defense overrides; "
    "defaults to the containing directory (e.g. 5_udemy).",
)
parser.add_argument(
    "-tag",
    "--tag",
    dest="tag",
    default=None,
    help="Override the output cell name.  Needed when the client is nop and "
    "the cell identity comes from the server defense, e.g. srvalpaca_mid1.",
)
parser.add_argument(
    "-http2_all",
    "--http2_all",
    dest="http2_all",
    default=0,
    help="Legacy alias for --defense h2pc",
)
parser.add_argument(
    "-request_server_defense",
    "--request_server_defense",
    dest="request_server_defense",
    help="Request server defense for a connection",
    default=None,
)

args = parser.parse_args()

assert args.dst_ip is not None
assert args.dst_port is not None
assert args.ifname is not None

# ----------------------------------------------------------------------
# identity: dataset name comes from the directory, NOT from resolving
# __file__ (this script is a symlink into mocks/, so resolving it would
# report "mocks" for every dataset).
# ----------------------------------------------------------------------
testcase = Path(__file__).parent.name  # e.g. 5_udemy
cat = Path(__file__).parent.parent.name  # e.g. calibration

DATASET = args.dataset or testcase
# read by client_defenses.levels.get_defense() at call time
os.environ["WF_DATASET"] = DATASET

# ----------------------------------------------------------------------
# defense naming.  The old front_abl10/50/100/250/500 ablations are
# superseded by --level; the ladder lives in client_defenses/levels.py.
# ----------------------------------------------------------------------
DEFENSE_ALIASES = {
    "h2pc": "h2pc",
    "mod_all": "h2pc",
    "tamaraw_qcsd": "tamaraw",
    "tamaraw": "tamaraw",
    "front": "front",
    "httpos": "httpos",
    "llama": "llama",
    "nop": "nop",
}
NO_LADDER = ("nop",)

raw_defense = "h2pc" if int(args.http2_all) else args.defense
if raw_defense not in DEFENSE_ALIASES:
    raise SystemExit(
        f"unknown defense {raw_defense!r}; expected one of {sorted(set(DEFENSE_ALIASES))}"
    )

USE_DEFENSE = DEFENSE_ALIASES[raw_defense]
USE_LEVEL = "mid1" if USE_DEFENSE in NO_LADDER else args.level
CELL = f"{raw_defense}_{USE_LEVEL}" if USE_DEFENSE not in NO_LADDER else raw_defense
if args.tag:
    CELL = args.tag

# import after WF_DATASET is set
from core_client import Request, run_test_case  # noqa: E402

conf.route_autoload = False
conf.route6_autoload = False
conf.bufsize = 50 * 1024 * 1024  # 50 MB buffer size

WORKSPACE = Path(f"/http2/experiments/{cat}/{testcase}")

# one directory per (defense, level) cell
CELL_PATH = WORKSPACE / "results" / CELL
TRACE_PATH = CELL_PATH / "traces"
OUTPUT_CSV = CELL_PATH / "tcp_repr/output_csv_single"
OUTPUT_CSV_ARCH = CELL_PATH / f"tcp_repr/{CELL}_rawtraces.tar.zst"
TRACE_PATH.mkdir(parents=True, exist_ok=True)

startup_wait_sec = 0.2

sslkeylog.set_keylog("ssllogkey.log")
IFACE = args.ifname
SERVER_PORT = int(args.dst_port)
SERVER_IP = args.dst_ip
REPEATS = int(args.repeats)
PAGE_LIMIT = int(args.subpage_limit)

DATA_PATH = Path("data")
SRV_DB_PATH = DATA_PATH / "server_trace"
CLIENT_DB_PATH = DATA_PATH / "client_trace"
BIN_DB_PATH = DATA_PATH / "bin"

USE_SRV_REQ_DEFENSE = args.request_server_defense

print(
    f"""
    Collection config:
        Dataset   : {DATASET}
        Defense   : {USE_DEFENSE}  (level {USE_LEVEL})
        Cell      : {CELL}
        Traces    : {TRACE_PATH}
        Repeats   : {REPEATS} x {PAGE_LIMIT} pages = {REPEATS * PAGE_LIMIT} captures
        Request Server Defense: {USE_SRV_REQ_DEFENSE}
"""
)

TESTCASES = glob(str(CLIENT_DB_PATH / "*"))


def get_domain(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc


# the server trace for a testcase was re-read once per request; every request
# in a page shares the label, so cache it
_SRV_DB_CACHE = {}


def server_db(label):
    if label not in _SRV_DB_CACHE:
        with open(SRV_DB_PATH / f"{label}.json") as f:
            _SRV_DB_CACHE[label] = json.load(f)
    return _SRV_DB_CACHE[label]


def prepare_requests(requests):
    out = {}
    req_defenses = {}

    max_exp_size = 0
    has_binary = False
    for ridx, request in enumerate(requests):
        label = request["label"]
        server_database = server_db(label)

        fullurl = request["url"]
        connection_id = get_domain(fullurl)
        path = request["url_local"]

        req_path = path.lower().split("?")[0]
        binary_exts = tuple(["gif", "png", "svg", "jpg", "jpeg", "pdf", "webp"])
        if req_path.endswith(binary_exts):
            has_binary = True

        file_path = server_database[path]["body_path"]
        expected_size = os.path.getsize(file_path)
        max_exp_size = max(max_exp_size, expected_size)

        if connection_id not in out:
            out[connection_id] = []

            if USE_SRV_REQ_DEFENSE == "all":
                req_defenses[connection_id] = True
            elif connection_id == USE_SRV_REQ_DEFENSE:
                req_defenses[connection_id] = True
            else:
                req_defenses[connection_id] = False

        delay = 0.001
        if len(out[connection_id]) == 0:
            delay = (
                0.01 * ridx
            )  # hack to simulate first stream on each connection as real as possible

        out[connection_id].append(
            Request(
                **{
                    "path": path,
                    "label": label,
                    "data": {},
                    "headers": {},
                    "delay": delay,
                    "expected_size": expected_size,
                    "content_type": server_database[path].get("content_type"),
                    "connection_id": connection_id,
                }
            )
        )

    if (
        USE_SRV_REQ_DEFENSE is not None
        and USE_SRV_REQ_DEFENSE not in out
        and USE_SRV_REQ_DEFENSE != "all"
    ):
        print(
            f"Server defense request for {USE_SRV_REQ_DEFENSE}, but I have only {out.keys()}"
        )
        return {}, {}, False

    return out, req_defenses, has_binary


VALID_TESTCASES = []
for testcase_path in tqdm(TESTCASES, desc="validating"):
    if len(VALID_TESTCASES) >= PAGE_LIMIT:
        break

    testcase_label = Path(testcase_path).stem
    raw_requests = json.load(open(testcase_path))
    if len(raw_requests) < 3:
        continue
    for idx, _ in enumerate(raw_requests):
        raw_requests[idx]["label"] = testcase_label

    requests, request_server_defenses, has_binary = prepare_requests(
        deepcopy(raw_requests)
    )

    if len(requests) == 0:
        continue

    VALID_TESTCASES.append(testcase_path)

RUN_TESTCASES = VALID_TESTCASES[:PAGE_LIMIT]
random.shuffle(RUN_TESTCASES)
repeat_order = random.sample(range(REPEATS), k=REPEATS)

print(f"{len(RUN_TESTCASES)} pages x {REPEATS} repeats in cell {CELL}\n")

captured = 0
skipped = 0
t_cell_start = time.time()

for pass_idx, repeat in enumerate(repeat_order):
    t_pass = time.time()

    for testcase_path in tqdm(
        RUN_TESTCASES, desc=f"{CELL} pass {pass_idx + 1}/{REPEATS}"
    ):
        testcase_label = Path(testcase_path).stem
        raw_requests = json.load(open(testcase_path))
        assert len(raw_requests) >= 3

        for idx, _ in enumerate(raw_requests):
            raw_requests[idx]["label"] = testcase_label

        requests, request_server_defenses, has_binary = prepare_requests(
            deepcopy(raw_requests)
        )
        assert len(requests) != 0

        testcase_save_label = f"test_{testcase_label}_{repeat}"
        trace_file = TRACE_PATH / f"{testcase_save_label}.pcap"
        out_csv_file = OUTPUT_CSV / f"temporal_data_{testcase_save_label}.csv"

        if OUTPUT_CSV_ARCH.exists():
            print("defense done")
            break

        # already captured, or already parsed and the pcap reclaimed
        if out_csv_file.exists() or trace_file.exists():
            skipped += 1
            continue

        # PCAP Collection
        tracer = AsyncSniffer(
            iface=IFACE,
            # filter=BPF_FILTER,
        )
        tracer.start()
        for retry in range(10):
            if not hasattr(tracer, "stop_cb"):
                print(f"Tracer not ready yet {retry}")
                time.sleep(startup_wait_sec)
            else:
                break

        run_test_case(
            SERVER_IP,
            SERVER_PORT,
            testcase_label,
            requests,
            request_server_defenses=request_server_defenses,
            defense_name=USE_DEFENSE,
            defense_level=USE_LEVEL,
        )

        # Save trace.  Write to a sidecar name and rename: rename is atomic on
        # the same filesystem, so a parallel processor never sees a partial
        # .pcap.  Without this it will silently parse a truncated capture.
        network_trace = tracer.stop()
        part_file = trace_file.with_suffix(f".pcap.{os.getpid()}.part")
        with open(part_file, "wb") as outfile:
            wrpcap(outfile, network_trace)
        part_file.rename(trace_file)

        captured += 1
        del network_trace
        del tracer

    # measured throughput -> decide the sweep size before committing to it
    if captured:
        elapsed = time.time() - t_cell_start
        per_capture = elapsed / captured
        remaining = (REPEATS * len(RUN_TESTCASES)) - captured - skipped
        size_gb = sum(f.stat().st_size for f in TRACE_PATH.glob("*.pcap")) / 1e9
        print(
            f"\n[{CELL}] pass {pass_idx + 1}/{REPEATS} done in {time.time() - t_pass:.0f}s"
            f" | {per_capture:.2f}s/capture"
            f" | {captured} captured, {skipped} skipped"
            f" | cell ETA {remaining * per_capture / 3600:.1f}h"
            f" | {size_gb:.1f} GB so far"
            f" -> ~{size_gb / max(captured, 1) * REPEATS * len(RUN_TESTCASES):.0f} GB for the cell\n",
            flush=True,
        )

print(f"[{CELL}] complete: {captured} captured, {skipped} skipped")
