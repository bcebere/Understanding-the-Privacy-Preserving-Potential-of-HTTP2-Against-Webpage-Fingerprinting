# Understanding the Privacy-Preserving Potential of HTTP2 Against Webpage Fingerprinting (WF)
In this repository, we provide the code to reproduce the results in the "Understanding the Privacy-Preserving Potential of HTTP2 Against Webpage Fingerprinting (WF)" paper.


## 💡 Experiments and Examples
The experiments folder contains details examples for [replaying a website under various defenses (client and server)](experiments/example_replay/README.md), as well as step-by-step example for [evaluatiang the security of the defenses and creating reports](experiments/example_benchmarks/README.md)


## 🛡️ Defenses and Calibration
The available WF defenses are:
 - Client-side defenses: FRONT, HTTPOS, Llama, Tamaraw, H2PC (detailed [here](experiments/calibrated_benchmarks/mocks/client_defenses/)).
 - Server-side: Alpaca, Tamaraw, H2PS (detailed [here](experiments/calibrated_benchmarks/mocks/server_defenses/)).


The [defense sweep_calibration](experiments/sweep_calibration) folder contains the logic for calibrating both the client-side and server-side defenses.
The calibration intensities are implemented as follows: [here for the client-side defenses](experiments/sweep_calibration/mocks/client_defenses/levels.py), and [here for the server-side defenses](experiments/sweep_calibration/mocks/server_defenses/levels.py).

The calibrated defenses are integrated in the [calibrated_benchmarks](experiments/calibrated_benchmarks/) folder. The selected configurations for each (dataset, defense) pair are available [here](experiments/calibrated_benchmarks/mocks/main_table_config.py).


## 🚀 Website Fingerprinting Audit Tool

The [wfaudit](./wfaudit/README.md) library provides a framework for evaluating the security of a WF defense, through fingerprinting accuracy, information leakage, and feature importance measurements.

The library can be installed from source using
```bash
cd wfaudit
pip install .
```

Example usage:

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
)

print("ML scores ---> ", scores["ML"])
print("Leakage scores ---> ", scores["leakage"])
```

The Machine-Learning estimators can be configured through the parameters:
 - `ml_arch_2D`. Available values: `kfp` (k-Fingerprinting), `rf` (Random Forest), `xgboost` (XGBoost).
 - `ml_arch_3D`. Available values: `holmes` (Holmes), `varcnn` (VarCNN), `df` (Deep-Fingerprinting), `robustfp` (Robust Fingerprinting).

The Information-Leakage estimators can be configured using the parameter:
 - `leakage_estimators`. Available values: `wefde` (WeFDE), `deepse` (DeepSE-WF).

Refer to the [experiments benchmark example](experiments/example_benchmarks/) for more usage examples.

## ⚡ Proof-of-Concept HTTP/2 client/servers

The [h2deflib](./h2deflib) folder contains the HTTP/2 client and servers used to replay the browser traces with various defenses enabled.

The library can be installed from source using
```bash
cd h2deflib
pip install .
```

The library includes unit tests in [tests](h2deflib/tests) and a [client defense demo](h2deflib/tests/demo/).


Refer to the [experiments replay examples](experiments/example_replay/) for more usage examples for configuring the client and the server defenses using the `h2deflib` library.

## :hammer: Tests

Install the testing dependencies `wfaudit` or `h2deflib` using
```bash
pip install .[testing]
```
The tests can be executed using
```bash
pytest -vsx
```

## 🔑 Datasets

Refer to [datasets section](./datasets/README.md) for the necessary steps preparing the resources for the experiments.


## 💥 Experiments

Refer to [experiments section](./experiments/README.md) for examples in replaying the collected datasets using various defenses, as well as evaluating the security of the defenses.

## Citing

If you use this code, please cite the associated paper:

```
@article{
    TODO
}
```
