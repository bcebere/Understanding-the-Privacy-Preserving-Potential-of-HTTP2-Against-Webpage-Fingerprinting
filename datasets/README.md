# Datasets

This directory provides the scripts used to create the datasets for **"Understanding the Privacy-Preserving Potential of HTTP/2 Against Webpage Fingerprinting."**

The evaluation datasets and supporting artifacts are available from the accompanying [Zenodo record](https://zenodo.org/records/22229611).
The full raw replay traces and calibration sweeps are available from [Google Drive](https://drive.google.com/drive/folders/1WAHwUJq44IQcHiTpsyAl8bgUvORlD2xR?usp=sharing).


## Browser traces

[browser_crawlers](browser_crawlers) contains the scripts used to collect the browser traces.


## Zenodo datasets

The Zenodo record provides the browser traces together with the processed evaluation datasets, benchmark results, and overhead measurements needed to reproduce the evaluation.

### Browser traces

`browser_traces.tar.zst` contains the raw browser traces used as input to the
replay framework. These traces contain the browser-collected inputs required to
regenerate the replay-based datasets.

### Calibrated-defense evaluation artifacts

For each of the five datasets, three archives are provided:

```text
datasets_calibrated_defenses_{dataset}_benchmarks.tar.zst
datasets_calibrated_defenses_{dataset}_deepsetraces.tar.zst
datasets_calibrated_defenses_{dataset}_wefdetraces.tar.zst
```

Here, `{dataset}` is one of:

```text
1_amazon
2_bbc
3_reddit
4_udemy
5_wiki
```

For example:

```text
datasets_calibrated_defenses_4_udemy_benchmarks.tar.zst
datasets_calibrated_defenses_4_udemy_deepsetraces.tar.zst
datasets_calibrated_defenses_4_udemy_wefdetraces.tar.zst
```

The archive types are:

- `benchmarks`: benchmark-result artifacts for the evaluated fingerprinting attacks.
- `deepsetraces`: neural-network trace representation used by DF, VarCNN, Holmes, RobustFP-CNN, and DeepSE-WF.
- `wefdetraces`: handcrafted-feature representation used by k-FP and WeFDE.

Each archive is further organized by defense. For example, a WeFDE archive contains entries such as:

```text
front/tcp_repr/front_wefdetraces.tar.zst
h2pc/tcp_repr/h2pc_wefdetraces.tar.zst
httpos/tcp_repr/httpos_wefdetraces.tar.zst
llama/tcp_repr/llama_wefdetraces.tar.zst
...
```

Each nested archive contains the evaluation data for the corresponding defense.

### Overhead analysis

`overhead_analysis.tar.zst` contains the measured bandwidth and latency overhead data used in the privacy-overhead analysis.

The archive contains one nested archive per dataset:

```text
1_amazon.tar.zst
2_bbc.tar.zst
3_reddit.tar.zst
4_udemy.tar.zst
5_wiki.tar.zst
```

Each dataset archive contains the corresponding per-defense overhead measurements, including:

```text
overhead/overhead_summary.csv
overhead/tradeoff_yvalues.json
overhead/ovh_*.csv
```

### Extracting the data

The datasets use Zstandard-compressed tar archives (`.tar.zst`).

An archive can be listed without extraction using:

```bash
tar --zstd -tf ARCHIVE.tar.zst
```

For example, to inspect the Udemy WeFDE archive:

```bash
tar --zstd -tf datasets_calibrated_defenses_4_udemy_wefdetraces.tar.zst
```

For evaluation, [prepare_workspace.sh](../experiments/example_benchmark/prepare_workspace.sh) automatically prepares a workspace for a selected dataset and either one defense, multiple defenses, or all available defenses.

See the [benchmark example](../experiments/example_benchmark/README.md) for instructions on preparing the published datasets and reproducing the ML, mutual-information, and overhead analyses.

See the [replay example](../experiments/example_replay/README.md) for instructions on regenerating replay traces from the browser traces.
