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
bash ./prepare_workspace.sh 4_udemy all <ZENODO SHARE PATH> --benchmarks

# Get only the FRONT-defended datasets for the Udemy dataset
bash ./prepare_workspace.sh 4_udemy front <ZENODO SHARE PATH> --benchmarks

```

