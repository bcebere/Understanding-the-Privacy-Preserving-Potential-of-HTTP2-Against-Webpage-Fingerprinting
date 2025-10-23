# stdlib
from argparse import ArgumentParser
from copy import deepcopy
from glob import glob
import json
import os
from pathlib import Path
import random
import time
from urllib.parse import urlparse

# third party
from core_client import Request, run_test_case
from scapy.config import conf
from scapy.sendrecv import AsyncSniffer
from scapy.utils import wrpcap
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("-dst_ip", "--dst_ip", dest="dst_ip", help="Destination IP")
parser.add_argument("-dst_port", "--dst_port", dest="dst_port", help="Destination Port")
parser.add_argument(
    "-repeats", "--repeats", dest="repeats", default=500, help="Subpage repeats"
)  # MUST BE GREATER THAN TOP_N in WeFDE Analysis
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
    "-http2_all",
    "--http2_all",
    dest="http2_all",
    help="HTTP2 : all features",
    default=0,
)
parser.add_argument(
    "-defense", "--defense", dest="defense", help="HTTP2 defense", default=None
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

conf.route_autoload = False
conf.route6_autoload = False
conf.bufsize = 50 * 1024 * 1024  # 50 MB buffer size

testcase = Path(__file__).parent.name
cat = Path(__file__).parent.parent.name

WORKSPACE = Path("workspace")
WORKSPACE.mkdir(parents=True, exist_ok=True)

TRACE_PATH = Path(WORKSPACE / "traces")
TRACE_PATH.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = Path(WORKSPACE / "tcp_repr/output_csv_single")
startup_wait_sec = 0.2

IFACE = args.ifname  # "lo"
SERVER_PORT = int(args.dst_port)  # 8443
SERVER_IP = args.dst_ip  # "127.0.0.1"
REPEATS = int(args.repeats)
PAGE_LIMIT = int(args.subpage_limit)

DATA_PATH = Path("data")
SRV_DB_PATH = DATA_PATH / "server_trace"
CLIENT_DB_PATH = DATA_PATH / "client_trace"
BIN_DB_PATH = DATA_PATH / "bin"

USE_HTTP2_ALL = int(args.http2_all)

USE_DEFENSE = args.defense
USE_SRV_REQ_DEFENSE = args.request_server_defense

print(
    f"""
    Test config:
        Defense: {USE_DEFENSE}
        Request Server Defense: {USE_SRV_REQ_DEFENSE}
"""
)


TESTCASES = glob(str(CLIENT_DB_PATH / "*"))


def get_domain(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc


def prepare_requests(requests):
    out = {}
    req_defenses = {}

    DATA_PATH = Path("data")
    SRV_DB_PATH = DATA_PATH / "server_trace"

    max_exp_size = 0
    has_binary = False
    for ridx, request in enumerate(requests):
        testcase = request["label"]
        with open(SRV_DB_PATH / f"{testcase}.json") as f:
            server_database = json.load(f)

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
                    "label": testcase,
                    "data": {},
                    "headers": {},
                    "delay": delay,
                    "expected_size": expected_size,
                    "connection_id": connection_id,
                }
            )
        )
    print(out.keys())
    if (
        USE_SRV_REQ_DEFENSE is not None
        and USE_SRV_REQ_DEFENSE not in out
        and USE_SRV_REQ_DEFENSE != "all"
    ):
        print(
            f"Server defense request for {USE_SRV_REQ_DEFENSE}, but I have only {out.keys()}"
        )
        return {}, {}, False

    # print("max expected size", max_exp_size)
    return out, req_defenses, has_binary


VALID_TESTCASES = []
for testcase_path in tqdm(TESTCASES):
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
    total_streams = 0

RUN_TESTCASES = VALID_TESTCASES[:PAGE_LIMIT]
random.shuffle(RUN_TESTCASES)
repeat_order = random.sample(range(REPEATS), k=REPEATS)

for repeat in repeat_order:
    for testcase_path in tqdm(RUN_TESTCASES):
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

        if out_csv_file.exists():
            continue

        if trace_file.exists():
            continue

        # PCAP Collection
        tracer = AsyncSniffer(
            iface=IFACE,
        )
        tracer.start()
        for retry in range(10):
            if not hasattr(tracer, "stop_cb"):
                print(f"Tracer not ready yet {retry}")
                time.sleep(startup_wait_sec)
            else:
                break

        # DEFENSES
        if USE_DEFENSE in [
            "tamaraw_qcsd",
            "front",
            "httpos",
            "llama",
        ]:
            run_test_case(
                SERVER_IP,
                SERVER_PORT,
                testcase_label,
                requests,
                request_server_defenses=request_server_defenses,
                defense_name=USE_DEFENSE,
            )
        elif USE_DEFENSE is not None:
            raise NotImplementedError()
        # MODALITIES
        elif USE_HTTP2_ALL:
            run_test_case(
                SERVER_IP,
                SERVER_PORT,
                testcase_label,
                requests,
                request_server_defenses=request_server_defenses,
                defense_name="mod_all",
            )
        else:
            run_test_case(
                SERVER_IP,
                SERVER_PORT,
                testcase_label,
                requests,
                request_server_defenses=request_server_defenses,
                defense_name="nop",
            )

        # Save trace
        network_trace = tracer.stop()
        with open(trace_file, "wb") as outfile:
            wrpcap(outfile, network_trace)

        del network_trace
        del tracer
