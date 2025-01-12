# orig: ../../../synthetic_datasets/synthetic_datasets_mixtures/mixtures_burst_pkt/generate_data.py

import json
from pathlib import Path

data_path = Path("data")
data_path.mkdir(parents=True, exist_ok=True)

LABELS = 256
TEST_REPEATS = 100
DATA_SIZE = 1000
DATA_DELAY = 0.0005


SERVER_DB = {}
CLIENT_DB = {}

for testcase in range(LABELS):
    # path = f"/label{testcase}"
    path = f"/label{testcase:04d}"
    entrypoint = f"/entrypoint{testcase:04d}"
    SERVER_DB[path] = {
        # "data_size": (testcase + 1) * 100,
        # "data_delay": DATA_DELAY * (testcase + 1),  # random, none, int
    }
    CLIENT_DB[entrypoint] = {
        "test_repeats": TEST_REPEATS,
        "requests": [{"path": entrypoint, "data": {}} for i in range(30)]
        + [{"path": path, "data": {}} for i in range(testcase + 1)]
        + [{"path": "/dummy", "data": {}}]
        + [{"path": path, "data": {}} for i in range(testcase + 2)]
        + [{"path": "/dummy", "data": {}} for i in range(10)],
    }

with open(data_path / "server_db.json", "w") as f:
    f.write(json.dumps(SERVER_DB, sort_keys=True, indent=4))

with open(data_path / "client_db.json", "w") as f:
    f.write(json.dumps(CLIENT_DB, sort_keys=True, indent=4))
