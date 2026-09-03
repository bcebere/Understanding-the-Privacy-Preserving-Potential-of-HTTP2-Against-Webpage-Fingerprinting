# 🚀 Website Fingerprinting Audit Tool

The `wfaudit` library provides a framework for evaluating the security of a website-fingerprinting (WF) defense through fingerprinting accuracy, information leakage, and feature-importance measurements.

## 🔑 Installation

The library can be installed from source using:

```bash
cd wfaudit
python -m pip install -e .
```

For development and testing, install the optional testing dependencies:

```bash
python -m pip install -e '.[testing]'
```

## 💥 Example usage

```python
# stdlib
from pathlib import Path

# wfaudit
from wfaudit import audit, prepare_all_datasets, process_raw_pcaps

traces = ...  # Path to the collected PCAPs
# See experiments/example_replay for a complete example of collecting PCAPs.

workspace = Path("workspace")

# Parse the collected PCAP traces
pcaps = process_raw_pcaps(
    traces=traces,
    workspace=workspace,
    unlink_after_processing=False,
)

# Create the evaluation datasets
prepare_all_datasets(
    workspace=workspace,
    n_websites=...,  # Number of labels in the dataset
    n_traces=...,    # Number of samples per label
)

# Evaluation outputs
ml_output_folder = workspace / "eval_ml"
wefde_output_folder = workspace / "eval_wefde"
deepse_output = workspace / "eval_deepse/results.csv"

# Evaluation inputs created by prepare_all_datasets
wefde_feats_folder = workspace / "output_features"
deepse_dataset = workspace / "output_deepse" / "real" / "dataset.npz"

assert wefde_feats_folder.exists()
assert deepse_dataset.exists()

# Evaluate the dataset
scores = audit(
    # Machine-learning attacks
    ml_output_folder=ml_output_folder,
    wefde_feats_folder=wefde_feats_folder,
    deepse_dataset=deepse_dataset,
    ml_arch_2D=["kfp"],
    ml_arch_3D=[],

    # Information-leakage estimators
    wefde_output_folder=wefde_output_folder,
    deepse_output=deepse_output,
)

print("ML scores ---> ", scores["ML"])
print("Leakage scores ---> ", scores["leakage"])
```

Additional usage examples are available in the [unit tests](./tests).

## 🌀 Available estimators

The machine-learning estimators can be configured through:

- `ml_arch_2D`
  - `kfp`: k-Fingerprinting
  - `rf`: Random Forest
  - `xgboost`: XGBoost

- `ml_arch_3D`
  - `holmes`: Holmes
  - `varcnn`: VarCNN
  - `df`: Deep Fingerprinting
  - `robustfp`: RobustFP-CNN

The library provides information-leakage evaluation using:

- WeFDE
- DeepSE-WF

## 🎛️ Hyperparameter tuning

`wfaudit` supports hyperparameter tuning of the fingerprinting models using
Optuna.

The tuning pipeline supports searching for model parameters, storing the selected configuration, and reusing the selected parameters during evaluation.

See [`test_tuning.py`](./tests/test_tuning.py) for examples covering:

- hyperparameter search;
- loading selected parameters;
- evaluating tuned models;
- resuming existing tuning studies; and
- comparing default and tuned configurations.

The complete paper-evaluation workflow, including reuse of the published HPO outputs and rerunning individual attacks, is documented in the [benchmark example](../experiments/example_benchmark/README.md).

## :hammer: Tests

Install the testing dependencies:

```bash
python -m pip install -e '.[testing]'
```

The tests use resources stored relative to the `tests` directory, so run them from there:

```bash
cd tests
python -m pip check
python -m pytest -vvsx
```

To run the tuning tests without the slower neural-network tests:

```bash
python -m pytest test_tuning.py -v -m "not slow"
```

For additional end-to-end examples of replaying traces and evaluating the published datasets, see:
- [Replay example](../experiments/example_replay/README.md)
- [Benchmark example](../experiments/example_benchmark/README.md)
