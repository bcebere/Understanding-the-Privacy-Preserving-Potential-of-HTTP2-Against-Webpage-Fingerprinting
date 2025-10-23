# 🚀 Website Fingerprinting Audit Tool

The [wfaudit](./wfaudit) library provides a framework for evaluating the security of a WF defense, through fingerprinting accuracy, information leakage, and feature importance measurements.


## 🔑 Installation
The library can be installed from source using
```bash
cd wfaudit
pip install .
# TODO: publish to PyPI
```

## 💥 Example usage:

```python
# stdlib
from pathlib import Path

# wfaudit absolute
from wfaudit import audit, prepare_all_datasets, process_raw_pcaps

traces = ... # path to the collected PCAPS
# See experiments/ for a full example on how to simulate the PCAPs

workspace = Path("workspace")
pcaps = process_raw_pcaps(
    traces=traces,
    workspace=workspace,
    unlink_after_processing=False,
)

# Create the evaluation datasets
prepare_all_datasets(
    workspace=workspace,
    n_websites=...,  # Number of labels in the dataset
    n_traces=...,  # Number of samples per label
)

# Evaluate the security of the dataset
ml_output_folder = workspace / "eval_ml"
wefde_output_folder = workspace / "eval_wefde"
deepse_output = workspace / "eval_deepse/results.csv"
xai_output_folder = workspace / "eval_xai"

wefde_feats_folder = workspace / "output_features"
deepse_dataset = workspace / "output_deepse" / "real" / "dataset.npz"

assert wefde_feats_folder.exists()  # created by prepare_all_datasets
assert deepse_dataset.exists()  # created by prepare_all_datasets

scores = audit(
    # ML
    ml_output_folder=ml_output_folder,
    wefde_feats_folder=wefde_feats_folder,
    deepse_dataset=deepse_dataset,
    ml_arch_2D=["xgboost"],
    ml_arch_3D=[],
    # leakage
    wefde_output_folder=wefde_output_folder,
    deepse_output=deepse_output,
    # xai
    xai_output_folder=xai_output_folder,
)

print("ML scores ---> ", scores["ML"])
print("Leakage scores ---> ", scores["leakage"])
print("XAI scores ---> ", scores["xai"])
```

## :hammer: Tests
For more examples on how to use the library, please refer to the [unit tests](./tests).

Install the testing dependencies for `wfaudit` using
```bash
pip install .[testing]
```
The tests can be executed using
```bash
pytest -vsx
```

``
