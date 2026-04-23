# stdlib
import json
from copy import deepcopy
from glob import glob

files = glob("./*.json")

print(files)
for res in files:
    with open(res) as f:
        data = json.load(f)

    clean_data = deepcopy(data)
    for key in data:
        clean_data[key]["headers"] = {}

    print(clean_data)

    with open(res, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, indent=4, ensure_ascii=False)
