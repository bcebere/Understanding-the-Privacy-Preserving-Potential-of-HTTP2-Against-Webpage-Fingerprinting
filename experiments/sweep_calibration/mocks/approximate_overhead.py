import json
import os
from argparse import ArgumentParser
from copy import deepcopy
from glob import glob
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from core_client import Request, run_test_case
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("-dst_ip", "--dst_ip", dest="dst_ip", help="Destination IP")
parser.add_argument(
    "-scenario",
    "--scenario",
    dest="scenario",
    default=None,
    help="Scenario label; defaults to <defense>_<level>",
)
parser.add_argument("-dst_port", "--dst_port", dest="dst_port", help="Destination Port")
parser.add_argument(
    "-http2_all",
    "--http2_all",
    dest="http2_all",
    help="HTTP2 all features",
    default=0,
)
parser.add_argument(
    "-defense", "--defense", dest="defense", help="HTTP2 defense", default=None
)
parser.add_argument(
    "-level",
    "--level",
    dest="level",
    default="mid1",
    choices=["vlow", "low", "lomid", "mid1", "mid2", "high", "vhigh", "vvhigh"],
    help="Defense intensity level",
)
parser.add_argument(
    "-workspace",
    "--workspace",
    dest="workspace",
    default="workspace",
    help="Directory for the ovh_<scenario>.csv output",
)
parser.add_argument(
    "-tag",
    "--tag",
    dest="tag",
    default=None,
    help="Scenario name AND defense/level labels for the output rows.  Needed "
    "for server-side cells, where the client is nop and would otherwise label "
    "every cell 'nop'.  e.g. srvalpaca_mid1",
)
parser.add_argument(
    "-pages",
    "--pages",
    dest="pages",
    type=int,
    default=100,
    help="Number of pages to run; use ~25 for overhead calibration",
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

WORKSPACE = Path(args.workspace)
WORKSPACE.mkdir(parents=True, exist_ok=True)


SERVER_PORT = args.dst_port  # 8443
SERVER_IP = args.dst_ip  # "127.0.0.1"

DATA_PATH = Path("data")
SRV_DB_PATH = DATA_PATH / "server_trace"
CLIENT_DB_PATH = DATA_PATH / "client_trace"
BIN_DB_PATH = DATA_PATH / "bin"

USE_HTTP2_ALL = int(args.http2_all)
USE_SRV_REQ_DEFENSE = args.request_server_defense
USE_LEVEL = args.level
PAGE_LIMIT = args.pages

# ``str(None)`` used to turn a missing --defense into the literal string
# "None", which then fell through the dispatch by accident.
if args.defense:
    USE_DEFENSE = args.defense
elif USE_HTTP2_ALL:
    USE_DEFENSE = "h2pc"
else:
    USE_DEFENSE = "nop"

# nop has no ladder; pin it to mid1 so the CSV and filename stay consistent
if USE_DEFENSE == "nop":
    USE_LEVEL = "mid1"

SCENARIO = (
    args.tag
    or args.scenario
    or (USE_DEFENSE if USE_DEFENSE == "nop" else f"{USE_DEFENSE}_{USE_LEVEL}")
)

# label rows by the tag, not by the (undefended) client defense
LEVELS = ("vlow", "low", "lomid", "mid1", "mid2", "high", "vhigh", "vvhigh")
LABEL_DEFENSE, LABEL_LEVEL = USE_DEFENSE, USE_LEVEL
if args.tag:
    head, _, tail = args.tag.rpartition("_")
    LABEL_DEFENSE, LABEL_LEVEL = (head, tail) if tail in LEVELS else (args.tag, "mid1")

print(
    f"""
    Test config:
        Client Defense : {USE_DEFENSE}
        Defense Level  : {USE_LEVEL}
        Scenario       : {SCENARIO}
        Pages          : {PAGE_LIMIT}
        Request Server Defense: {USE_SRV_REQ_DEFENSE}
"""
)
TESTCASES = glob(str(CLIENT_DB_PATH / "*"))[:1024]


def get_domain(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc


# The server trace for a testcase was re-read from disk once per request.
# Every request in a page shares the same label, so cache it.
_SRV_DB_CACHE = {}


def server_db(testcase):
    if testcase not in _SRV_DB_CACHE:
        with open(SRV_DB_PATH / f"{testcase}.json") as f:
            _SRV_DB_CACHE[testcase] = json.load(f)
    return _SRV_DB_CACHE[testcase]


def prepare_requests(requests):
    out = {}
    req_defenses = {}

    max_exp_size = 0
    for ridx, request in enumerate(requests):
        testcase = request["label"]
        server_database = server_db(testcase)

        fullurl = request["url"]
        connection_id = get_domain(fullurl)
        path = request["url_local"]
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
                    "label": testcase,
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
        return {}, {}

    print("MAX download size", max_exp_size)
    return out, req_defenses


# Validation used to walk all 1024 testcases even when only a handful of pages
# were going to be run. Stop as soon as PAGE_LIMIT valid ones are found.
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

    requests, request_server_defenses = prepare_requests(deepcopy(raw_requests))

    if len(requests) == 0:
        continue

    VALID_TESTCASES.append(testcase_path)

keys = VALID_TESTCASES[:PAGE_LIMIT]
print(f"{len(keys)} valid testcases")

load_stats = []

for testcase_path in tqdm(keys, desc=SCENARIO):
    testcase_label = Path(testcase_path).stem
    raw_requests = json.load(open(testcase_path))
    for idx, _ in enumerate(raw_requests):
        raw_requests[idx]["label"] = testcase_label

    if len(raw_requests) < 3:
        continue

    requests, request_server_defenses = prepare_requests(raw_requests)
    if len(requests) == 0:
        continue

    print("TESTCASE", testcase_label)
    stats = run_test_case(
        SERVER_IP,
        SERVER_PORT,
        testcase_label,
        requests,
        request_server_defenses=request_server_defenses,
        defense_name=USE_DEFENSE,
        defense_level=USE_LEVEL,
    )

    stats["testcase"] = testcase_label
    stats["scenario"] = SCENARIO
    stats["defense"] = LABEL_DEFENSE
    stats["level"] = LABEL_LEVEL
    load_stats.append(stats)

load_stats = pd.DataFrame(load_stats)
out_path = WORKSPACE / f"ovh_{SCENARIO}.csv"
load_stats.to_csv(out_path, index=None)
print(f"\nwrote {out_path}  ({len(load_stats)} pages)")

# quick read of this cell; calibrate.py does the pooled table and the
# baseline-joined latency overhead across every scenario
if len(load_stats) and "bw_rx_overhead" in load_stats:
    for col, label in (("bw_tx_overhead", "dUp"), ("bw_rx_overhead", "dDown")):
        s = pd.to_numeric(load_stats[col], errors="coerce").dropna()
        if not s.empty:
            print(
                f"  {label}: median {s.median():.2f} "
                f"(Q1 {s.quantile(0.25):.2f} - Q3 {s.quantile(0.75):.2f})"
            )
