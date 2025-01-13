# stdlib
import json
from pathlib import Path

data_path = Path("data")
data_path.mkdir(parents=True, exist_ok=True)

LABELS = 256
TEST_REPEATS = 100
DATA_SIZE = 1000


SERVER_DB = {}
CLIENT_DB = {}

for testcase in range(LABELS):
    # path = f"/label{testcase}"
    path = f"/label{testcase:04d}"
    entrypoint = f"/entrypoint{testcase:04d}"
    SERVER_DB[path] = {
        "data_size": (testcase + 1) * 100,
        "data_delay": "random",  # random, none, int
    }
    CLIENT_DB[entrypoint] = {
        "test_repeats": TEST_REPEATS,
        "requests": [{"path": entrypoint, "data": {}} for i in range(5)]
        + [{"path": path, "data": {}} for i in range(2)]
        + [{"path": "/dummy", "data": {}}]
        + [{"path": path, "data": {}} for i in range(3)]
        + [{"path": "/dummy", "data": {}} for i in range(2)],
    }

with open(data_path / "server_db.json", "w") as f:
    f.write(json.dumps(SERVER_DB, sort_keys=True, indent=4))

with open(data_path / "client_db.json", "w") as f:
    f.write(json.dumps(CLIENT_DB, sort_keys=True, indent=4))
