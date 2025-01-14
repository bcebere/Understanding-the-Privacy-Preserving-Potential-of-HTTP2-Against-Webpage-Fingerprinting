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
## 🔑 Evaluation Datasets

### Synthetic Datasets (Worst-case Scenarios)
The synthetic datasets aim to spotlight a specific source of leakage in order to evaluate the privacy-preserving potential of the defenses.
The generators and sample datasets are available in [datasets](datasets).

The following scenarios are available:
* Packet count/rate leakages: [pkt_v1](datasets/syn_datasets/pkt_v1), [pkt_v2](datasets/syn_datasets/pkt_v2), [pkt_v3](datasets/syn_datasets/pkt_v3).
* Timing leakage: [time_v1](datasets/syn_datasets/time_v1), [time_v2](datasets/syn_datasets/time_v2), [time_v3](datasets/syn_datasets/time_v3), [time_v4](datasets/syn_datasets/time_v4), [time_v5](datasets/syn_datasets/time_v5).
* Burst/CUMUL Leakage: [burst_v1](datasets/syn_datasets/burst_v1),[burst_v2](datasets/syn_datasets/burst_v2),[burst_v3](datasets/syn_datasets/burst_v3),[burst_v4](datasets/syn_datasets/burst_v4),[burst_v5](datasets/syn_datasets/burst_v5),[burst_v6](datasets/syn_datasets/burst_v6),[burst_v7](datasets/syn_datasets/burst_v7),[burst_v8](datasets/syn_datasets/burst_v8).
* Joint Leakage: [mix_v1](datasets/syn_datasets/mix_v1), [mix_v2](datasets/syn_datasets/mix_v2),[mix_v3](datasets/syn_datasets/mix_v3),[mix_v4](datasets/syn_datasets/mix_v4),[mix_v5](datasets/syn_datasets/mix_v5),[mix_v6](datasets/syn_datasets/mix_v6),[mix_v7](datasets/syn_datasets/mix_v7).

### Real-world datasets
For the real-world datasets, we provide the source URLs and the [playwright](https://playwright.dev/) script for collecting the resources.
The following datasets are available:
* [Amazon](datasets/realworld_datasets/1_amazon/).
* [BBC](datasets/realworld_datasets/2_bbc/).
* [Reddit](datasets/realworld_datasets/3_reddit/).
* [DailyStar](datasets/realworld_datasets/4_dailystar/).
* [Udemy](datasets/realworld_datasets/5_udemy/).
* [Wikipedia](datasets/realworld_datasets/6_wikipedia/).


## 💥 HTTP/2 Experiments


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
