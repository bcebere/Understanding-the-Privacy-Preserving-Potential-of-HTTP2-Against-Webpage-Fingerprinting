# orig : 2_baseline_time_stats

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

for testcase in range(LABELS):
    path = f"/label{testcase}"
    path = f"/label{testcase:04d}"
    SERVER_DB[path] = {
        "data_size": DATA_SIZE,  # "random", int,
        "data_delay": DATA_DELAY * (testcase + 1),  # random, none, int, float
    }
    CLIENT_DB[path] = {
        "test_repeats": TEST_REPEATS,
        "requests": [{"path": path, "data": {}} for i in range(1)],
    }

with open(data_path / "server_db.json", "w") as f:
    f.write(json.dumps(SERVER_DB, sort_keys=True, indent=4))

with open(data_path / "client_db.json", "w") as f:
    f.write(json.dumps(CLIENT_DB, sort_keys=True, indent=4))
