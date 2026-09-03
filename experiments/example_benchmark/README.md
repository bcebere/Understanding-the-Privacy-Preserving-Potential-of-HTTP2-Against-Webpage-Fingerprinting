# Benchmarks example

This folder provides an example for running the evaluation and reporting the defense performance results.

## `wfaudit` setup

First, install the auditing tool. For example, create a clean Conda environment:

```bash
# Conda environment
conda create -n pubhttp2-clean python=3.10 -y
conda activate pubhttp2-clean

# Prevent packages from the user-level Python installation from leaking
# into the environment.
export PYTHONNOUSERSITE=1
unset PYTHONPATH

python -m pip install --upgrade pip

# Install wfaudit
cd wfaudit
python -m pip install -e .

# Optional: install the test dependencies and run the unit tests
python -m pip install -e '.[testing]'
cd tests
python -m pip check
python -m pytest -vvsx
cd ../../
```

## Existing datasets

The published evaluation datasets are available from the accompanying [Zenodo dataset](https://zenodo.org/records/22229611).

Existing evaluation datasets and benchmark results can be prepared for visualization or re-evaluation using `prepare_workspace.sh`.

Examples:

```bash
# Prepare all defended datasets for Udemy, including published benchmark results
bash ./prepare_workspace.sh 4_udemy all <ZENODO_ARCHIVES_PATH> --benchmarks

# Prepare one specific defense
bash ./prepare_workspace.sh 4_udemy front <ZENODO_ARCHIVES_PATH> --benchmarks

# Explicitly specify the output workspace
bash ./prepare_workspace.sh 4_udemy srvtamaraw_all \
  <ZENODO_ARCHIVES_PATH> ./workspace --benchmarks

# Generic form
bash ./prepare_workspace.sh <dataset> <defense> \
  <ZENODO_ARCHIVES_PATH> ./workspace --benchmarks
```

Valid datasets are:

```text
1_amazon
2_bbc
3_reddit
4_udemy
5_wiki
```

Valid defenses are:

```text
nop
front
httpos
llama
tamaraw
h2pc
srvalpaca_1st
srvalpaca_3rd_1
srvalpaca_all
srvtamaraw_1st
srvtamaraw_3rd_1
srvtamaraw_all
srvh2ps1p
```

`nop` denotes the undefended baseline.

The `<defense>` argument can be a single defense, a comma-separated list of defenses, or `all`.

Without `--benchmarks`, the script extracts only the evaluation inputs (`deepsetraces/` and `wefdetraces/`) together with the dataset-specific overhead measurements.
This is sufficient when recomputing the evaluation results rather than inspecting the published benchmark outputs.

With `--benchmarks`, a prepared defense directory looks like:

```text
workspace/4_udemy/srvtamaraw_all/
├── deepsetraces/
│   └── real/dataset.npz              input: DF, VarCNN, Holmes, RobustFP-CNN, DeepSE
├── wefdetraces/
│   ├── output_features/              input: k-FP, XGBoost, WeFDE
│   └── output_wefde/                 intermediate WeFDE traces
└── benchmarks/                       published evaluation outputs
    ├── eval_ml/
    ├── eval_ml_nn/
    ├── eval_wefde/
    ├── eval_deepse/
    └── hpo/
```

The dataset-specific overhead measurements are extracted separately under:

```text
workspace/4_udemy/overhead/
```

## Generate tables and plots

This folder contains helper scripts for aggregating and reporting the security and overhead results of the available defenses.

**Note:** These scripts aggregate only benchmark results that are present in the `workspace` directory.

```bash
# Summarize the performance of the available client-side defenses
python3 generate_table_clientdefs_perf.py

# Summarize the performance of the available server-side defenses
python3 generate_table_serverdefs_perf.py

# Summarize defense overhead per dataset and across datasets
python3 generate_table_overheads.py

# Report the strongest attacker and its tuned hyperparameters
python3 generate_table_strongest_attacker.py
python3 generate_table_attacker_params.py
```

## Run evaluation from scratch

To rerun an evaluation for a model, dataset, and defense, first remove the corresponding cached result from:

```text
WORKSPACE=workspace/<DATASET>/<DEFENSE>/
```

| To rerun | Delete |
|---|---|
| One attacker while keeping its tuned parameters | `<WORKSPACE>/benchmarks/eval_ml/scores_rawts_<arch>_tuned.bkp` |
| One attacker and redo hyperparameter search | the `.bkp` file and `<WORKSPACE>/benchmarks/hpo/` |
| An untuned attacker | `<WORKSPACE>/benchmarks/eval_ml*/scores_rawts_<arch>_topk.bkp` |
| WeFDE leakage | `<WORKSPACE>/benchmarks/eval_wefde/leakage.csv` |
| DeepSE leakage | `<WORKSPACE>/benchmarks/eval_deepse/results_df.csv` |

Then rerun the desired ML or mutual-information evaluation.

### ML evaluation

```bash
python3 benchmark_process_3_evaluate.py \
  --dataset <dataset> --cell <defense> --arch <model>

# Example
python3 benchmark_process_3_evaluate.py \
  --dataset 4_udemy --cell srvtamaraw_all --arch kfp
```

### Mutual-information evaluation

```bash
python3 benchmark_process_4_mi_estimators.py \
  --dataset <dataset> --cell <defense>

# Example
python3 benchmark_process_4_mi_estimators.py \
  --dataset 4_udemy --cell srvtamaraw_all
```
