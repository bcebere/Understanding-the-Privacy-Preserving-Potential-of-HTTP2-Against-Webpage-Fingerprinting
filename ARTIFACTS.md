# Artifact Evaluation Guide

This document provides guidelines for evaluating the artifacts accompanying **"Understanding the Privacy-Preserving Potential of HTTP/2 Against Webpage Fingerprinting."**

The repository contains the software used to replay webpage traces under the evaluated defenses, construct the evaluation datasets, run the machine-learning and information-leakage analyses, reproduce the reported aggregate results, and perform defense calibration.

## Artifact locations

- **Source code:** https://github.com/bcebere/Understanding-the-Privacy-Preserving-Potential-of-HTTP2-Against-Webpage-Fingerprinting
- **Published evaluation datasets and supporting artifacts:** https://zenodo.org/records/22229611
- **Full raw replay traces and calibration sweeps:** https://drive.google.com/drive/folders/1WAHwUJq44IQcHiTpsyAl8bgUvORlD2xR?usp=sharing
- **Permanent source-code snapshot:** TODO

The Zenodo record contains the processed artifacts needed for the recommended evaluation path. The substantially larger raw replay corpus is provided separately because it is not necessary for inspecting or recomputing the published evaluation results.

## Repository overview

The main artifact components are:

- `datasets/`
  - documentation and scripts for collecting the browser traces;
  - description of the published datasets and raw replay corpus.
- `experiments/example_benchmark/`
  - preparation of published evaluation artifacts;
  - rerunning ML and mutual-information analyses;
  - generation of aggregate tables and reports.
- `experiments/example_replay/`
  - end-to-end example for replaying webpage content under client- and
    server-side defenses and collecting PCAP traces.
- `experiments/calibrated_benchmarks/`
  - calibrated defense configurations and evaluation scripts used for the
    paper experiments.
- `experiments/sweep_calibration/`
  - defense calibration sweeps and calibration-processing scripts.
- `experiments/docker_image/`
  - Docker setup used for the replay experiments.
- `wfaudit/`
  - website-fingerprinting audit library for ML attacks, information-leakage
    estimation, feature-importance analysis, and hyperparameter tuning.
- `h2deflib/`
  - HTTP/2 client/server implementation and defense instrumentation.

## Published datasets

The Zenodo record contains artifacts for five datasets:

```text
1_amazon
2_bbc
3_reddit
4_udemy
5_wiki
```

For every dataset, the calibrated-defense evaluation artifacts are provided as:

```text
datasets_calibrated_defenses_{dataset}_benchmarks.tar.zst
datasets_calibrated_defenses_{dataset}_deepsetraces.tar.zst
datasets_calibrated_defenses_{dataset}_wefdetraces.tar.zst
```

The archive types are:

- `benchmarks`: published benchmark-result artifacts;
- `deepsetraces`: neural-network trace representation used by DF, VarCNN, Holmes, RobustFP-CNN, and DeepSE-WF;
- `wefdetraces`: handcrafted-feature representation used by k-FP and WeFDE.

The record additionally contains:

```text
browser_traces.tar.zst
overhead_analysis.tar.zst
```

See [`datasets/README.md`](datasets/README.md) for the complete archive
description.

## Recommended artifact-evaluation path

The artifact has been tested on Linux x86-64.
- Python 3.10 is used for the benchmark/evaluation environment.
- Docker is required only for regenerating replay traces.
- A GPU is not required for the main artifact-evaluation path. CPU execution is sufficient for dataset preparation, reporting, k-FP evaluation, WeFDE analysis, and the non-neural tests.
- A GPU is recommended for neural-model evaluation and tuning.

The recommended evaluation path uses the published processed datasets. It does **not** require regenerating the complete raw replay corpus or rerunning the full calibration sweeps.

### 1. Set up `wfaudit`

A clean Python environment is recommended:

```bash
conda create -n pubhttp2-clean python=3.10 -y
conda activate pubhttp2-clean

export PYTHONNOUSERSITE=1
unset PYTHONPATH

python -m pip install --upgrade pip
```

On a CPU-only machine, install the CPU build of PyTorch before installing
`wfaudit`:

```bash
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.8.0
```

Then install the library:

```bash
cd wfaudit
python -m pip install -e .
cd ..
```

### 2. Download the Zenodo artifacts

Download the required `.tar.zst` files from:

https://zenodo.org/records/22229611

Place them in a local directory, for example:

```text
archives/
```

For the full benchmark workflow, download:

```text
overhead_analysis.tar.zst
datasets_calibrated_defenses_{dataset}_benchmarks.tar.zst
datasets_calibrated_defenses_{dataset}_deepsetraces.tar.zst
datasets_calibrated_defenses_{dataset}_wefdetraces.tar.zst
```

for the dataset(s) to be evaluated.

### 3. Prepare a workspace

From:

```bash
cd experiments/example_benchmark
```

prepare one defense together with its published benchmark outputs:

```bash
bash ./prepare_workspace.sh 4_udemy srvtamaraw_all /path/to/archives ./workspace --benchmarks
```

Multiple defenses can be selected using a comma-separated list:

```bash
bash ./prepare_workspace.sh 4_udemy front,tamaraw /path/to/archives ./workspace --benchmarks
```

All available defenses can be prepared using:

```bash
bash ./prepare_workspace.sh 4_udemy all /path/to/archives ./workspace --benchmarks
```

See [`experiments/example_benchmark/README.md`](experiments/example_benchmark/README.md) for the detailed workspace structure.

## Reproducing reported results

### Aggregate published results

After preparing a workspace with `--benchmarks`, the following scripts aggregate the available published results:

```bash
python3 generate_table_clientdefs_perf.py
python3 generate_table_serverdefs_perf.py
python3 generate_table_overheads.py
python3 generate_table_strongest_attacker.py
python3 generate_table_attacker_params.py
```

These scripts respectively summarize:

- client-side defense security;
- server-side defense security;
- defense bandwidth/latency overhead;
- the strongest fingerprinting attacker;
- tuned attacker parameters.

### Recompute an ML result

Prepare the workspace without `--benchmarks`, or remove the corresponding cached benchmark result as documented in the benchmark example.

Then run, for example:

```bash
python3 benchmark_process_3_evaluate.py --dataset 4_udemy --cell srvtamaraw_all --arch kfp
```

Generic form:

```bash
python3 benchmark_process_3_evaluate.py --dataset DATASET --cell DEFENSE --arch MODEL
```

### Recompute mutual-information results

For example:

```bash
python3 benchmark_process_4_mi_estimators.py  --dataset 4_udemy --cell srvtamaraw_all
```

Generic form:

```bash
python3 benchmark_process_4_mi_estimators.py --dataset DATASET --cell DEFENSE
```

## Hyperparameter tuning

The fingerprinting models support hyperparameter tuning through `wfaudit`.
The published benchmark artifacts contain the corresponding HPO outputs.

The benchmark workflow supports both:
1. reusing the published tuning outputs; and
2. rerunning the tuning before reevaluating an attacker.

The exact cache files/directories that must be removed to force reevaluation or retuning are documented in [`experiments/example_benchmark/README.md`](experiments/example_benchmark/README.md).

The tuning implementation is additionally exercised by `wfaudit/tests/test_tuning.py`, including search, resumption, parameter loading, tuned evaluation, cache separation, and baseline-versus-tuned reporting.

## Replay evaluation

The raw network traces can also be regenerated from the browser inputs.

The complete workflow is documented in:

[`experiments/example_replay/README.md`](experiments/example_replay/README.md)
