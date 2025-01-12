# synthetic_datasets_burst_variations/syn_2res

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument(
    "-cnt", "--cnt", dest="cnt", help="Count of fingerprintiable resources"
)
args = parser.parse_args()

assert args.cnt is not None

data_path = Path("data")
data_path.mkdir(parents=True, exist_ok=True)

LABELS = 256
TEST_REPEATS = 100
DATA_SIZE = 1000
DATA_DELAY = 0.0001


def generate_size_leak(testcase: int):
    return {
        "data_size": (testcase + 1) * 50,
        "data_delay": 0,
    }


def generate_time_leak(testcase: int):
    return {
        "data_size": 17,
        "data_delay": DATA_DELAY * (testcase + 1),
    }


def generate_with_res_cnt(cnt: int):
    SERVER_DB = {}
    CLIENT_DB = {}
    for testcase in range(LABELS):
        entrypoint = f"/entrypoint{testcase:04d}"
        SERVER_DB[entrypoint] = {
            "data_size": 111,
            "data_delay": 0,
        }

        client_reqs = [{"path": entrypoint, "data": {}}]
        for off in range(cnt):
            local_path_no = testcase + off
            local_path = f"/label{local_path_no:04d}"
            SERVER_DB[local_path] = generate_size_leak(local_path_no)
            client_reqs.append({"path": local_path, "data": {}})

        CLIENT_DB[entrypoint] = {
            "test_repeats": TEST_REPEATS,
            "requests": client_reqs,
        }

    with open(data_path / "server_db.json", "w") as f:
        f.write(json.dumps(SERVER_DB, sort_keys=True, indent=4))

    with open(data_path / "client_db.json", "w") as f:
        f.write(json.dumps(CLIENT_DB, sort_keys=True, indent=4))


generate_with_res_cnt(cnt=int(args.cnt))
