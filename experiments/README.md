# Experiment examples

Helper scripts and examples for reproducing the experiments from **"Understanding the Privacy-Preserving Potential of HTTP/2 Against Webpage Fingerprinting."**

## Reproduction examples

The repository provides two end-to-end examples:

- [example_replay](example_replay/README.md) contains the complete workflow for replaying webpage content under the evaluated client- and server-side defenses and collecting the resulting PCAP traces.
- [example_benchmark](example_benchmark/README.md) contains the workflow for preparing the published datasets, running the machine-learning and mutual-information evaluations, and generating the reported security and overhead results.

## Defense evaluation

The core scripts for evaluating already calibrated defenses are available in [calibrated_benchmarks](calibrated_benchmarks/mocks/).

The per-dataset directory structure follows the same general organization as the calibration experiments, but uses the calibrated-defense scripts from `calibrated_benchmarks/mocks/`.

To collect traces, first start the appropriate server(s):

- Client-side defenses:

  ```bash
  bash run_server.sh <SERVER_PORT>
  ```

- Server-side defenses:

  ```bash
  python3 start_defended_servers.py start --replicas 4 --dataset <DATASET>
  ```

Then collect the PCAP traces:

```bash
# Client-side defenses
python3 collect_calibrated_traces.py client \
  <SERVER_IP> <SERVER_PORT> <CAPTURE_INTERFACE> \
  --dataset <DATASET>

# Server-side defenses
python3 collect_calibrated_traces.py server \
  <SERVER_IP> <CAPTURE_INTERFACE> \
  --replicas 4 --dataset <DATASET>
```

See [example_replay](example_replay/README.md) for the complete replay and trace collection procedure.

## Defense calibration

The core scripts for calibrating defenses are available in [sweep_calibration](sweep_calibration/mocks/).

To prepare a calibration workspace for a dataset, create a dataset directory, for example `1_amazon`, and add symlinks to:

1. the scripts in `sweep_calibration/mocks/`; and
2. the corresponding `data` directory extracted from the `browser_traces` dataset.

Start the appropriate server(s):

- Client-side defense calibration:

  ```bash
  bash run_server.sh <SERVER_PORT>
  ```

- Server-side defense calibration:

  ```bash
  bash start_calibration_servers.sh
  ```

Run the calibration sweeps:

```bash
# Client-side defenses
bash calibration_collect_all_cldefenses.sh \
  <SERVER_IP> <SERVER_PORT> <CAPTURE_INTERFACE>

# Server-side defenses
bash calibration_collect_all_srvdefenses.sh \
  <SERVER_IP> <CAPTURE_INTERFACE>
```

Once collection is complete, process the resulting PCAP traces in three steps:

1. Convert PCAP traces to CSV:

   ```bash
   python3 calibration_process_1_parse_traces.py
   ```

2. Create the evaluation datasets:

   ```bash
   python3 calibration_process_2_create_datasets.py
   ```

3. Evaluate the calibration sweep:

   ```bash
   python3 calibration_process_3_eval_sensitivities.py
   ```

The calibrated configurations used in the paper are recorded in [main_table_config.py](calibrated_benchmarks/mocks/main_table_config.py).
