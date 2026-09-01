# Experiments examples
Helpers scripts for creating the traces and datasets required for the  `Understanding the Privacy-Preserving Potential of HTTP/2 Against Webpage Fingerprinting` paper.

## Defense calibration

The code for calibrating a defense is available in [sweep_calibration](sweep_calibration/mocks).

In order to calibrate a defense, first create the workspace for a dataset, e.g. `1_amazon` folder.

Then, in the `1_amazon` folder, create symlinks to (1) the scripts inside the sweepo_calibration/mocks folder, and (2) to the `data` folder from the `browser_traces` dataset.

Example folder structure for reference

```
1_amazon/
├── approximate_overhead.py -> ../mocks/approximate_overhead.py
├── calibrate.py -> ../mocks/calibrate.py
├── calibrate.sh -> ../mocks/calibrate.sh
├── calibration_collect_all_cldefenses.sh
│   -> ../mocks/calibration_collect_all_cldefenses.sh
├── calibration_collect_all_srvdefenses.sh
│   -> ../mocks/calibration_collect_all_srvdefenses.sh
├── calibration_process_1_parse_traces.py
│   -> ../mocks/calibration_process_1_parse_traces.py
├── calibration_process_2_create_datasets.py
│   -> ../mocks/calibration_process_2_create_datasets.py
├── calibration_process_3_eval_sensitivities.py
│   -> ../mocks/calibration_process_3_eval_sensitivities.py
├── client_defenses -> ../mocks/client_defenses
├── code_selfcheck.py -> ../mocks/code_selfcheck.py
├── collect_srvdefenses.sh -> ../mocks/collect_srvdefenses.sh
├── collect_traces.py -> ../mocks/collect_traces.py
├── compute_overhead_all_srvdefenses.sh
│   -> ../mocks/compute_overhead_all_srvdefenses.sh
├── compute_overhead_cldefenses.sh
│   -> ../mocks/compute_overhead_cldefenses.sh
├── compute_overhead_srvdefenses.sh
│   -> ../mocks/compute_overhead_srvdefenses.sh
├── compute_overhead_srvdefenses_placement.py
│   -> ../mocks/compute_overhead_srvdefenses_placement.py
├── core_client.py -> ../mocks/core_client.py
├── data -> ../../realworld_datasets/1_amazon/data
│   ├── bin/
│   ├── client_trace/
│   └── server_trace/
├── requirements.txt -> ../mocks/requirements.txt
├── run_server.sh -> ../mocks/run_server.sh
├── run_server_level.sh -> ../mocks/run_server_level.sh
├── server_defenses -> ../mocks/server_defenses/
├── server_simple.py -> ../mocks/server_simple.py
├── server_with_defs.py -> ../mocks/server_with_defs.py
├── start_calibration_servers.sh
│   -> ../mocks/start_calibration_servers.sh
├── track_calibration_collect_status.sh
│   -> ../mocks/track_calibration_collect_status.sh
├── track_calibration_results.py
│   -> ../mocks/track_calibration_results.py

```

Then, in order to run the experiments, first start the server(s):
 - for client defenses, the standard server `run_server.sh`
 - for server defenses, the calibration servers using `start_calibration_servers.sh`.

Next, one can do the calibration sweeps:
 - for client defenses : `bash ./calibration_collect_all_cldefenses.sh <SRV IP> <SRV PORT> <CAPTURE INTERFACE>`.
 - for server defenses: `bash calibration_collect_all_srvdefenses.sh <SRV IP> <CAPTURE INTERFACE>`.

Once finished, the raw PCAP traces can be analyzed by following the steps:
1. Convert to CSVs: `calibration_process_1_parse_traces.py`.
2. Convert to ML datasets: `calibration_process_2_create_datasets.py`.
3. Benchmark `calibration_process_3_eval_sensitivities.py`.


## Defense evaluation

The code for running an already calibrated defense is available in [calibrated_benchmarks](calibrated_benchmarks/mocks/).

The dataset folder structure is similar to the calibration sweep example, the only difference being the scripts from [calibrated_benchmarks](calibrated_benchmarks/mocks/).


First, start the server(s):
 - For client defenses, using `bash run_server.sh`
 - For server defenses, using `python ./start_defended_servers.py`

Collect the PCAP traces using either
 - for client defenses, `python3 collect_calibrated_traces.py client <SRV IP> <SRV PORT> <CAPTURE INTERFACE>`
 - for server defenses, `python3 collect_calibrated_traces.py server <SRV IP> <CAPTURE INTERFACE> --replicas 4`
