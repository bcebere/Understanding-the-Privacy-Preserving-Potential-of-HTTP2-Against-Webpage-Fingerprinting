# orig: ../../synthetic_datasets/synthetic_datasets_baseline/4_first_packets/generate_data.py

# stdlib
import json
from pathlib import Path

data_path = Path("data")
data_path.mkdir(parents=True, exist_ok=True)

LABELS = 256
TEST_REPEATS = 100
DATA_SIZE = 1000
DATA_DELAY = 0.005

SERVER_DB = {}
CLIENT_DB = {}

for testcase in range(20):
    path = f"/dummy{testcase:04d}"
    SERVER_DB[path] = {
        "data_size": "random",  # "random", int,
        "data_delay": 0,  # random, none, int, float
    }

for testcase in range(LABELS):
    # path = f"/label{testcase}"
    path = f"/label{testcase:04d}"
    SERVER_DB[path] = {
        "data_size": testcase + 10,  # "random", int,
        "data_delay": 0,  # random, none, int, float
    }
    CLIENT_DB[path] = {
        "test_repeats": TEST_REPEATS,
        "requests": [
            {"path": path, "data": {}},
        ]
        + [{"path": f"/dummy{i:04d}", "data": {}} for i in range(15)],
    }

with open(data_path / "server_db.json", "w") as f:
    f.write(json.dumps(SERVER_DB, sort_keys=True, indent=4))

with open(data_path / "client_db.json", "w") as f:
    f.write(json.dumps(CLIENT_DB, sort_keys=True, indent=4))
