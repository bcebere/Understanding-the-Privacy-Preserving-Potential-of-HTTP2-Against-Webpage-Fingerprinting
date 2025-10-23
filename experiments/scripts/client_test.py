# stdlib
from argparse import ArgumentParser
from glob import glob
import json
import os
from pathlib import Path
from urllib.parse import urlparse

# third party
from core_client import Request, run_test_case
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("-dst_ip", "--dst_ip", dest="dst_ip", help="Destination IP")
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
    "-request_server_defense",
    "--request_server_defense",
    dest="request_server_defense",
    help="Request server defense for a connection",
    default=None,
)


args = parser.parse_args()

assert args.dst_ip is not None
assert args.dst_port is not None

DATA_PATH = Path("data")

SERVER_PORT = args.dst_port  # 8443
SERVER_IP = args.dst_ip  # "127.0.0.1"

DATA_PATH = Path("data")
SRV_DB_PATH = DATA_PATH / "server_trace"
CLIENT_DB_PATH = DATA_PATH / "client_trace"
BIN_DB_PATH = DATA_PATH / "bin"

USE_HTTP2_WND = int(args.http2_wnd)
USE_HTTP2_BATCH = int(args.http2_batch)
USE_HTTP2_PING = int(args.http2_ping)
USE_HTTP2_ALL = int(args.http2_all)
USE_DEFENSE = str(args.defense)
USE_SRV_REQ_DEFENSE = args.request_server_defense

print(
    f"""
    Test config:
        Client Defense : {USE_DEFENSE}
        Request Server Defense: {USE_SRV_REQ_DEFENSE}
"""
)
TESTCASES = glob(str(CLIENT_DB_PATH / "*"))[:1024]


def get_domain(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc


def prepare_requests(requests):
    out = {}
    req_defenses = {}

    DATA_PATH = Path("data")
    SRV_DB_PATH = DATA_PATH / "server_trace"

    max_exp_size = 0
    for ridx, request in enumerate(requests):
        testcase = request["label"]
        with open(SRV_DB_PATH / f"{testcase}.json") as f:
            server_database = json.load(f)

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


keys = TESTCASES
# random.shuffle(keys)

for testcase_path in tqdm(keys):
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
    break
