# Understanding the Privacy-Preserving Potential of HTTP/2 Against Webpage Fingerprinting

This repository provides the code and artifacts for reproducing the results in **"Understanding the Privacy-Preserving Potential of HTTP/2 Against Webpage Fingerprinting."**

## 💡 Experiments, Datasets and Examples

The [datasets section](./datasets/README.md) describes the resources used by the experiments.

The experiments folder contains detailed examples for [replaying websites under the client- and server-side defenses](experiments/example_replay/README.md), as well as a step-by-step example for [evaluating defense security and generating the reported results](experiments/example_benchmark/README.md).

The published processed datasets and benchmark outputs can be used directly without regenerating the raw PCAP traces:
1. Download the artifacts from the accompanying [Zenodo record](https://zenodo.org/records/22229611).
2. Set up `wfaudit` as described below.
3. Prepare an evaluation workspace. For example:

   ```bash
   cd experiments/example_benchmark

   bash ./prepare_workspace.sh 4_udemy srvtamaraw_all <ZENODO_ARCHIVES_PATH> ./workspace --benchmarks
   ```
4. Generate the result tables or rerun an individual evaluation following [the benchmark example](experiments/example_benchmark/README.md).

To regenerate network traces from the browser inputs, follow the [replay example](experiments/example_replay/README.md).



## 🛡️ Defenses and Calibration
The available WF defenses are:
 - Client-side defenses: FRONT, HTTPOS, Llama, Tamaraw, H2PC (detailed [here](experiments/sweep_calibration/mocks/client_defenses/)).
 - Server-side: Alpaca, Tamaraw, H2PS (detailed [here](experiments/sweep_calibration/mocks/server_defenses/)).


The [defense sweep_calibration](experiments/sweep_calibration) folder contains the logic for calibrating both the client-side and server-side defenses.
The calibration intensities are implemented as follows: [here for the client-side defenses](experiments/sweep_calibration/mocks/client_defenses/levels.py), and [here for the server-side defenses](experiments/sweep_calibration/mocks/server_defenses/levels.py).

The calibrated defenses are integrated in the [calibrated_benchmarks](experiments/calibrated_benchmarks/) folder. The selected configurations for each (dataset, defense) pair are available [here](experiments/calibrated_benchmarks/mocks/main_table_config.py).


## 🚀 Website Fingerprinting Audit Tool

The [wfaudit](./wfaudit/README.md) library provides a framework for evaluating the security of a WF defense, through fingerprinting accuracy, information leakage, and feature importance measurements.

The library can be installed from source using
```bash
cd wfaudit
python -m pip install -e .
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
    ml_arch_2D=["kfp"],
    ml_arch_3D=["robustfp"],
    # leakage
    wefde_output_folder=wefde_output_folder,
    deepse_output=deepse_output,
)

print("ML scores ---> ", scores["ML"])
print("Leakage scores ---> ", scores["leakage"])
```

The Machine-Learning estimators can be configured through the parameters:
 - `ml_arch_2D`. Available values: `kfp` (k-Fingerprinting), `xgboost` (XGBoost).
 - `ml_arch_3D`. Available values: `holmes` (Holmes), `varcnn` (VarCNN), `df` (DF), `robustfp` (RF-CNN).


`wfaudit` also supports hyperparameter tuning of the fingerprinting models. The published benchmark artifacts include the corresponding HPO outputs; see the [benchmark example](experiments/example_benchmark/README.md) for instructions on reusing the published parameters or rerunning the tuning and evaluation.


## ⚡ Proof-of-Concept HTTP/2 client/servers

The [h2deflib](./h2deflib) folder contains the HTTP/2 client and servers used to replay the browser traces with various defenses enabled.

The library can be installed from source using
```bash
cd h2deflib
python -m pip install -e .
```

The library includes unit tests in [tests](h2deflib/tests) and a [client defense demo](h2deflib/tests/demo/).


Refer to the [experiments replay examples](experiments/example_replay/) for more usage examples for configuring the client and the server defenses using the `h2deflib` library.

## :hammer: Tests

Install the testing dependencies `wfaudit` or `h2deflib` using
```bash
python -m pip install -e '.[testing]'
```
The tests can be executed using
```bash
cd tests
python -m pip check
python -m pytest -vvsx
```


## Citing

If you use this code, please cite the associated paper:

```
@article{
    TODO
}
```
