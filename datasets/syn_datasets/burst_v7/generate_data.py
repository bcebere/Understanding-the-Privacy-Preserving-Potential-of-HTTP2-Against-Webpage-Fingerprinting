# orig: mixture_random/mix_size_pkt_2

# stdlib
import json
from pathlib import Path

data_path = Path("data")
data_path.mkdir(parents=True, exist_ok=True)

LABELS = 256
TEST_REPEATS = 100
DATA_SIZE = 123
DATA_DELAY = 0.0001

SERVER_DB = {}
CLIENT_DB = {}


def _gen_const_sample(testcase):
    return {
        "data_size": DATA_SIZE,
        "data_delay": 0,
    }


def _gen_srv_time_sample(testcase):
    return {
        "data_size": DATA_SIZE,
        "data_delay": DATA_DELAY * (testcase + 1),  # random, none, int
    }


def _gen_srv_size_sample(testcase):
    return {
        "data_size": (testcase + 1) * 10,
        "data_delay": 0,
    }


for testcase in range(LABELS):
    path = f"/label{testcase:04d}"
    entrypoint = f"/entrypoint{testcase:04d}"

    repeats = 1
    if testcase % 2 == 0:
        SERVER_DB[path] = _gen_const_sample(testcase)
        repeats = testcase + 1
    else:
        SERVER_DB[path] = _gen_srv_size_sample(testcase)

    CLIENT_DB[entrypoint] = {
        "test_repeats": TEST_REPEATS,
        "requests": [{"path": entrypoint, "data": {}} for i in range(2)],
    }
    for i in range(repeats):
        CLIENT_DB[entrypoint]["requests"].extend(
            [{"path": path, "data": {}}, {"path": "/dummy", "data": {}}]
        )


with open(data_path / "server_db.json", "w") as f:
    f.write(json.dumps(SERVER_DB, sort_keys=True, indent=4))

with open(data_path / "client_db.json", "w") as f:
    f.write(json.dumps(CLIENT_DB, sort_keys=True, indent=4))
