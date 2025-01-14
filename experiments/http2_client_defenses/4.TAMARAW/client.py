# stdlib
from argparse import ArgumentParser
from copy import deepcopy
import json
from pathlib import Path
import random
import time

# third party
from core_client import Request, run_test_case
from scapy.config import conf
from scapy.sendrecv import AsyncSniffer
from scapy.utils import wrpcap
import sslkeylog
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("-dst_ip", "--dst_ip", dest="dst_ip", help="Destination IP")
parser.add_argument("-dst_port", "--dst_port", dest="dst_port", help="Destination Port")
parser.add_argument(
    "-repeats", "--repeats", dest="repeats", default=30, help="Subpage repeats"
)
parser.add_argument(
    "-ifname", "--ifname", dest="ifname", help="Interface to use for capture"
)


args = parser.parse_args()

assert args.dst_ip is not None
assert args.dst_port is not None
assert args.ifname is not None

conf.route_autoload = False
conf.route6_autoload = False
conf.bufsize = 50 * 1024 * 1024  # 50 MB buffer size


DATA_PATH = Path("data")
TRACE_PATH = Path("traces")
OUTPUT_CSV = Path("output_csv_single")
startup_wait_sec = 0.2

sslkeylog.set_keylog("ssllogkey.log")
IFACE = args.ifname  # "lo"
SERVER_PORT = args.dst_port  # 8443
SERVER_IP = args.dst_ip  # "127.0.0.1"
REPEATS = args.repeats

TRACE_PATH.mkdir(parents=True, exist_ok=True)

with open(DATA_PATH / "client_db.json") as f:
    TESTCASES = json.load(f)


def prepare_requests(requests):
    out = []
    for request in requests:
        out.append(
            Request(
                **{
                    "path": request["path"],
                    "data": request["data"],
                    "headers": request["headers"] if "headers" in request else {},
                    "delay": request["delay"] if "delay" in request else 0,
                }
            )
        )
    return out


keys = list(TESTCASES.keys())
random.shuffle(keys)

for testcase in tqdm(keys):
    raw_requests = TESTCASES[testcase]["requests"]
    for repeat in range(REPEATS):
        requests = prepare_requests(deepcopy(raw_requests))

        testcase_save_label = f"test_{testcase[1:]}_{repeat}"
        trace_file = TRACE_PATH / f"{testcase_save_label}.pcap"
        out_csv_file = OUTPUT_CSV / f"temporal_data_{testcase_save_label}.csv"

        if out_csv_file.exists():
            print("already processed", out_csv_file)
            continue

        if trace_file.exists():
            print("already collected", trace_file)
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

        # Run the test
        # DEFENSE: TAMARAW HTTP2
        run_test_case(
            SERVER_IP, SERVER_PORT, testcase, requests, defense_name="tamaraw"
        )

        # Save trace
        network_trace = tracer.stop()
        with open(trace_file, "wb") as outfile:
            wrpcap(outfile, network_trace)

        del network_trace
        del tracer
