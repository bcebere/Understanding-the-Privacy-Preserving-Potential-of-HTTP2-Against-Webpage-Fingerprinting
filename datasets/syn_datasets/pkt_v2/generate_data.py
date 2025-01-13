# orig: 1_baseline_packet_counts

# stdlib
import json
from pathlib import Path

data_path = Path("data")
data_path.mkdir(parents=True, exist_ok=True)

LABELS = 256
TEST_REPEATS = 100
DATA_SIZE = 0
DATA_DELAY = 1

SERVER_DB = {}
CLIENT_DB = {}

for testcase in range(LABELS):
    path = f"/label{testcase:04d}"
    SERVER_DB[path] = {
        # "data_size": "none",
        # "data_delay": "none",
    }
    CLIENT_DB[path] = {
        "test_repeats": TEST_REPEATS,
        "requests": [{"path": path, "data": {}} for i in range(testcase + 1)],
    }

with open(data_path / "server_db.json", "w") as f:
    f.write(json.dumps(SERVER_DB, sort_keys=True, indent=4))

with open(data_path / "client_db.json", "w") as f:
    f.write(json.dumps(CLIENT_DB, sort_keys=True, indent=4))
