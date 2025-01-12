# burst_1res2dummy_rand
import json
import random
import sys
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument(
    "-cnt", "--cnt", dest="cnt", help="Count of fingerprintiable resources"
)
parser.add_argument(
    "-cnt_dummy", "--cnt_dummy", dest="cnt_dummy", help="Count of dummy resources"
)
args = parser.parse_args()

assert args.cnt is not None
assert args.cnt_dummy is not None

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


def generate_with_res_cnt(cnt: int, cnt_dummy: int):
    SERVER_DB = {}
    CLIENT_DB = {}
    SERVER_DB["/dummy_small"] = {
        "data_size": 11,
        "data_delay": 0,
    }
    SERVER_DB["/dummy_med"] = {
        "data_size": 1111,
        "data_delay": 0,
    }
    SERVER_DB["/dummy_big"] = {
        "data_size": 111111,
        "data_delay": 0,
    }
    for testcase in range(LABELS):
        entrypoint = f"/entrypoint{testcase:04d}"
        SERVER_DB[entrypoint] = {
            "data_size": 111,
            "data_delay": 0,
        }

        client_reqs = [
            {"path": entrypoint, "data": {}},
        ]
        for dcnt in range(cnt_dummy):
            # {"path": "/dummy_small", "data": {}},
            didx = random.randint(0, 2)
            if didx == 0:
                client_reqs.append({"path": "/dummy_small", "data": {}})
            elif didx == 1:
                client_reqs.append({"path": "/dummy_med", "data": {}})
            else:
                client_reqs.append({"path": "/dummy_big", "data": {}})

        for off in range(cnt):
            local_path_no = testcase + off
            local_path = f"/label{local_path_no:04d}"

            coin = random.choice([True, False])
            if coin:
                SERVER_DB[local_path] = generate_time_leak(local_path_no)
            else:
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


generate_with_res_cnt(cnt=int(args.cnt), cnt_dummy=int(args.cnt_dummy))
