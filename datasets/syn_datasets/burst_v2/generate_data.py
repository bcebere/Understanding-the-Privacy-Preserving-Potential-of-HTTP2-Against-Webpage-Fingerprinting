# orig : ../../synthetic_datasets/synthetic_datasets_mixtures_wikipat/wikipat_small_mixres_3

# stdlib
import json
from pathlib import Path
import random

data_path = Path("data")
data_path.mkdir(parents=True, exist_ok=True)

LABELS = 256
TEST_REPEATS = 100
DATA_SIZE = 1000
DATA_DELAY = 0.0005


SERVER_DB = {}
CLIENT_DB = {}

# Server Config
for testcase in range(1000):
    path = f"/label{testcase:04d}"
    entrypoint = f"/entrypoint{testcase:04d}"
    SERVER_DB[path] = {
        "data_size": (testcase + 1) * 17,
        # "data_delay": DATA_DELAY * (testcase + 1),  # random, none, int
    }
    SERVER_DB[entrypoint] = {
        "data_size": 10,
    }
    SERVER_DB["/dummy"] = {
        "data_size": 275,
    }


# Client Config
for testcase in range(LABELS):
    path = f"/label{testcase:04d}"
    entrypoint = f"/entrypoint{testcase:04d}"

    repeats = random.randint(1, 3)
    contents_idx = random.sample(range(1, 1000), repeats)
    contents = [path]
    for cidx in contents_idx:
        contents.append(f"/label{cidx:04d}")

    CLIENT_DB[entrypoint] = {
        "test_repeats": TEST_REPEATS,
        "requests": [
            {"path": entrypoint, "data": {}},
            {"path": "/dummy", "data": {}},
        ],
    }
    for cpath in contents:
        CLIENT_DB[entrypoint]["requests"].extend(
            [
                {"path": cpath, "data": {}},
                {"path": "/dummy", "data": {}},
            ]
        )

with open(data_path / "server_db.json", "w") as f:
    f.write(json.dumps(SERVER_DB, sort_keys=True, indent=4))

with open(data_path / "client_db.json", "w") as f:
    f.write(json.dumps(CLIENT_DB, sort_keys=True, indent=4))
