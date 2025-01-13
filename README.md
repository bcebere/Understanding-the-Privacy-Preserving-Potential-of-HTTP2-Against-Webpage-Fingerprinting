# Understanding the Privacy-Preserving Potential of HTTP2 Against Subpage Fingerprinting
In this repository, we provide the code to reproduce the results in the "Understanding the Privacy-Preserving Potential of HTTP2 Against Subpage Fingerprinting" paper.

## 🚀 Website Fingerprinting Audit Tool

We provide the library for measuring the information leakage and fingerprinting accuracy in [wfaudit](wfaudit).

The library can be installed from source using
```bash
cd wfaudit
pip install .
# TODO: publish to PyPI
```

Example usage:

```python
# stdlib
import json
from pathlib import Path

# wfaudit absolute
from wfaudit import create_datasets, evaluate_leakage, evaluate_ml, prepare_features
from wfaudit.helpers_ml import print_score

traces = ... # folder with collected PCAPS from the interaction client-server. Each pcap name should have the format '<subpage_label>_<repeat_count>.pcap'
# See wfaudit/tests/test_benchmarks.py for a data collection example.

workspace = Path("workspace")

# Process the PCAPs in the `traces` folder
create_datasets(
    traces=Path("traces"),
    workspace=tmp_path,
    unlink_after_processing=False,
)

# Extract the information leakage features
output_features = workspace / "output_features"
features_range = prepare_features(
    time_series_traces=tmp_path / "output_wefde",
    output=output_features,
)

# Compute information leakage
output_leakage = workspace / "output_leakage"
leakage = evaluate_leakage(
    features, workspace=output_leakage, wefde_features_dir=output_features
)
print("Information Leakage ", leakage)

# Compute F1 score for fingerprinting
ml_score = evaluate_ml(
    workspace=output_ml, wefde_features_dir=output_features
)  # returns a list of F1 scores, for each label
print("ML F1-score", print_score(ml_score))

```

## 💥 HTTP/2 experiments
TODO

## :hammer: Tests

Install the testing dependencies using
```bash
pip install .[testing]
```
The tests can be executed using
```bash
pytest -vsx
```
## Citing

If you use this code, please cite the associated paper:

```
...
```
