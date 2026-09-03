## Benchmarks example

In this folder, we provide an example for running and reporting the defense performance numbers.

### wfaudit setup

First, make sure to install the auditing tool. Example for setting up the env.

```bash
# Conda env
conda create -n pubhttp2-clean python=3.10 -y
conda activate pubhttp2-clean

export PYTHONNOUSERSITE=1
unset PYTHONPATH

python -m pip install --upgrade pip

# wfaudit
cd wfaudit
python -m pip install -e .

# Optionally, check that the unit tests pass
python -m pip install -e .[testing]
cd tests
python -m pip check
python -m pytest -vvsx
cd ../../
```
### Existing Datasets

Our datasets are available at the following share: TODO.

The existing collected datasets or benchmarks can be reused for visualization/re-evalution using the `prepare_workspace.sh` script.

Examples:

```bash
# Get all defended datasets for the Udemy dataset
bash ./prepare_workspace.sh 4_udemy all <ZENODO ARCHIVES LOCAL PATH> --benchmarks

# Get only the defense-specific datasets for the Udemy dataset
bash ./prepare_workspace.sh 4_udemy front <ARCHIVES LOCAL PATH> --benchmarks
bash ./prepare_workspace.sh 4_udemy srvtamaraw_all <ARCHIVES PATH> ./workspace --benchmarks
bash ./prepare_workspace.sh 4_udemy <defense> <ARCHIVES PATH> ./workspace --benchmarks
#
```

Valid datasets: `1_amazon`, `2_bbc`, `3_reddit`, `4_udemy`, `5_wiki`.
Valid defenses: `nop` (undefended), `front`, `httpos`, `llama`, `tamaraw`, `h2pc`, `srvalpaca_1st`, `srvalpaca_3rd_1`, `srvalpaca_all`, `srvtamaraw_1st`, `srvtamaraw_3rd_1`, `srvtamaraw_all`, `srvh2ps1p`.


`<defense>` is one name, a comma-separated list, or `all`. Without `--benchmarks` you get only the inputs (`deepsetraces/`, `wefdetraces/`), which
is what you want if you intend to recompute the results rather than inspect the published ones.

A prepared cell looks like:

```
workspace/4_udemy/srvtamaraw_all/
├── deepsetraces/real/dataset.npz        input: DF, VarCNN, Holmes, RobustFP, DeepSE
├── wefdetraces/output_features/         input: k-FP, XGBoost, WeFDE
├── wefdetraces/output_wefde/            raw traces, not read by steps 3-4
└── benchmarks/                          outputs: eval_ml, eval_ml_nn, eval_wefde,
                                                  eval_deepse, hpo
```


### Generate tables and plots

The folder contains helper scripts for aggregating and reporting the security and overhead of each defense.

❗Note: The script aggregate only the available benchmark results in the 'workspace' folder.


```bash

# Summarize the performance of each client-side defense (available in the "workspace" folder)
python generate_table_clientdefs_perf.py

# Summarize the performance of each server-side defense (available in the "workspace" folder)
python generate_table_serverdefs_perf.py

# Summarize the overhead of each defense: per dataset and aggregated.
python generate_table_overheads.py

# Summarize bests attacker, and its hypertuned parameters
python generate_table_strongest_attacker.py
python generate_table_attacker_params.py

```


### Run evaluation from scratch
In order to run evaluation for a model, dataset and defense from scratch, first cleanup any caches in the workspace `WORKSPACE = workspace/<DATASET>/<DEFENSE>/`.
| To rerun | Delete |
|---|---|
| one attacker, keeping its tuned params | `<WORKSPACE>/benchmarks/eval_ml/scores_rawts_<arch>_tuned.bkp` |
| one attacker, redoing the search too | the `.bkp` **and** `<WORKSPACE>/benchmarks/hpo/` |
| an untuned attacker | `<WORKSPACE>/benchmarks/eval_ml*/scores_rawts_<arch>_topk.bkp` |
| WeFDE leakage | `<WORKSPACE>/benchmarks/eval_wefde/leakage.csv` |
| DeepSE leakage | `<WORKSPACE>/benchmarks/eval_deepse/results_df.csv` |


Next, in order to re-evaluate the ML or MI experiments

```
# ML
python3 benchmark_process_3_evaluate.py --dataset <dataset> --cell <defense> --arch <model>
python3 benchmark_process_3_evaluate.py --dataset 4_udemy --cell srvtamaraw_all --arch kfp

# Mutual information
python3 benchmark_process_4_mi_estimators.py --dataset <dataset> --cell <defense>
python3 benchmark_process_4_mi_estimators.py --dataset 4_udemy --cell srvtamaraw_all

```
