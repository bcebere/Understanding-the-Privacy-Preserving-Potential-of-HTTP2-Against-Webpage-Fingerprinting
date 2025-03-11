# stdlib
import glob
import hashlib
import json
from pathlib import Path
from random import shuffle
from typing import Optional

# third party
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# wfaudit absolute
from wfaudit.helpers_wefde.analysis.data_utils import load_wefde_features
from wfaudit.helpers_wefde.preprocess.extract import prepare_wefde_features
import wfaudit.logger as log
from wfaudit.processing import process_pcap

np.set_printoptions(suppress=True)


def process_raw_pcaps(
    traces=Path("traces"),
    workspace=Path("workspace"),
    unlink_after_processing=True,
    n_jobs=8,
):
    """
    Args:
        - traces: Folder with the PCAPs to be parsed. --- traces / "*.pcap"
        - workspace : Folder where to store intermediary and final CSVs.
        - unlink_after_processing: Delete the PCAP after processing. Useful for low-space devices.
    """
    if not traces.exists():
        log.error("missing traces folder")
        return
    workspace.mkdir(parents=True, exist_ok=True)
    output = workspace / "output_csv_single"
    output.mkdir(parents=True, exist_ok=True)

    files = glob.glob(str(traces / "*.pcap"))
    shuffle(files)

    def _parse_single_pcap(filename):
        filename = Path(filename)
        stem = filename.stem
        output_csv_static = output / f"static_data_{stem}.csv"
        output_csv_temporal = output / f"temporal_data_{stem}.csv"
        if not filename.exists():
            return

        if output_csv_temporal.exists():
            if unlink_after_processing:
                log.debug(f"dropping  {filename}")
                filename.unlink()
            return
        log.debug(f"Parsing {filename}")
        try:
            session = process_pcap(filename)
        except BaseException as e:
            log.error(
                f"failed to parse pcap. moving to graveyard {filename}, error = {e}"
            )
            filename.unlink()
            return

        label = stem.split("_")[1]
        static_data, temporal_data = session.temporal_stats_per_flow()
        if len(static_data) == 0:
            log.error(f"empty dataset {filename}")
            filename.unlink()
            return

        static_data["label"] = label
        static_data.to_csv(output_csv_static, index=False)
        temporal_data.to_csv(output_csv_temporal, index=False)

        if unlink_after_processing:
            filename.unlink()

    Parallel(n_jobs=n_jobs)(delayed(_parse_single_pcap)(filename) for filename in files)


def merge_pcap_csvs(workspace=Path("workspace"), pd_lim: int = 10000) -> None:
    """
    Args:
        workspace: The folder which contains the post-processed pcaps --- output_csv_single.
        pd_lim: how often to batch the CSVs.
    """
    in_workspace = workspace / Path("output_csv_single")
    if not in_workspace.exists():
        log.error("Missing output_csv_single folder")
        return

    output = workspace / Path("output_csv_full")
    output.mkdir(parents=True, exist_ok=True)

    full_static_csv: Optional[pd.DataFrame] = None
    full_temporal_csv: Optional[pd.DataFrame] = None
    cnt = 0
    batch_idx = 0

    static_files = glob.glob(str(in_workspace / "static*.csv"))
    print("static files", len(static_files))
    for fidx, filename in enumerate(static_files):
        static_filename = Path(filename)
        base = static_filename.name.split("static_")[1]
        temporal_base = "temporal_" + base
        temporal_filename = in_workspace / temporal_base

        assert static_filename.exists()
        assert temporal_filename.exists()
        try:
            local_static_csv = pd.read_csv(static_filename)
            local_temporal_csv = pd.read_csv(temporal_filename)
        except BaseException as e:
            print("failed to read csv", e)
            continue

        original_ids = local_static_csv["id"].values[0]
        original_label = local_static_csv["label"].values[0]
        total_duration = local_temporal_csv["relative_timestamp"].sum()
        new_id = f"{original_ids}-{original_label}-{total_duration}-{fidx}"
        hashed_id = hashlib.sha1(new_id.encode())
        hashed_id = hashed_id.hexdigest()

        local_static_csv["file_order"] = fidx
        local_temporal_csv["file_order"] = fidx

        local_static_csv["full_id"] = hashed_id
        local_temporal_csv["full_id"] = hashed_id

        if full_static_csv is None:
            full_static_csv = local_static_csv
            full_temporal_csv = local_temporal_csv
        else:
            full_static_csv = pd.concat(
                [full_static_csv, local_static_csv], ignore_index=True
            )
            full_temporal_csv = pd.concat(
                [full_temporal_csv, local_temporal_csv], ignore_index=True
            )

        if cnt % 1000 == 0:
            log.debug(f"merge  batch {cnt}, {full_static_csv.shape}")

        if len(full_static_csv) > pd_lim:
            assert full_static_csv is not None
            assert full_temporal_csv is not None

            full_static_csv.to_csv(
                output / f"static_data_batch{batch_idx}.csv",
                index=False,
            )
            full_temporal_csv.to_csv(
                output / f"temporal_data_batch{batch_idx}.csv",
                index=False,
            )
            batch_idx += 1
            full_static_csv = None
            full_temporal_csv = None

        cnt += 1

    if full_temporal_csv is not None:
        full_static_csv.to_csv(
            output / f"static_data_batch{batch_idx}.csv",
            index=False,
        )
        full_temporal_csv.to_csv(
            output / f"temporal_data_batch{batch_idx}.csv",
            index=False,
        )

    full_data_static = None
    full_data_temporal = None

    for batch in range(0, batch_idx + 1):
        static_batch = Path(output / f"static_data_batch{batch}.csv")
        temporal_batch = Path(output / f"temporal_data_batch{batch}.csv")
        if not static_batch.exists():
            break

        batch_data_static = pd.read_csv(static_batch)
        batch_data_temporal = pd.read_csv(temporal_batch)

        if full_data_static is None:
            full_data_static = batch_data_static
            full_data_temporal = batch_data_temporal
        else:
            full_data_static = pd.concat(
                [full_data_static, batch_data_static], ignore_index=True
            )
            full_data_temporal = pd.concat(
                [full_data_temporal, batch_data_temporal], ignore_index=True
            )

    full_data_static.to_csv(
        output / "static_data.csv",
        index=False,
    )
    full_data_temporal.to_csv(
        output / "temporal_data.csv",
        index=False,
    )


def _discrete_columns(
    dataframe: pd.DataFrame, max_classes: int = 10, return_counts: bool = False
) -> list:
    """
    Find columns containing discrete values in a pandas dataframe.
    """
    return [
        (col, cnt) if return_counts else col
        for col, vals in dataframe.items()
        for cnt in [vals.nunique()]
        if cnt <= max_classes
    ]


def _constant_columns(dataframe: pd.DataFrame) -> list:
    """
    Find constant value columns in a pandas dataframe.
    """
    return _discrete_columns(dataframe, 1)


def _prepare_time_series(
    static_data: pd.DataFrame,
    ts_data: pd.DataFrame,
    ts_limit: Optional[int] = None,
    ts_pad: int = 0,
    ID_COL="file_order",  # id, full_id
):
    ts_data_clean = []

    groups = ts_data.groupby(ID_COL)
    ids = []

    static_data = static_data.drop_duplicates(ID_COL)
    static_ids = set(static_data[ID_COL].values)

    lens = []
    sizes = []
    rel_times = []
    for idx, group in tqdm(groups):
        if idx not in static_ids:
            continue

        lens.append(len(group))
        local_data = group.drop(columns=["id", "full_id", "file_order", "duration"])
        sizes.extend(local_data["length"].values.tolist())
        rel_times.extend(local_data["relative_timestamp"].values.tolist())
        if ts_limit is not None:
            if len(local_data) > ts_limit:
                local_data = local_data.head(ts_limit)
            else:
                padded = ts_pad * np.ones(
                    ((ts_limit - len(local_data)), len(local_data.columns))
                )
                local_data = pd.concat(
                    [local_data, pd.DataFrame(padded, columns=local_data.columns)],
                    ignore_index=True,
                )

        ts_data_clean.append(local_data)
        ids.append(idx)

    static_data = static_data.set_index(ID_COL)
    static_data = static_data.reindex(ids)
    log.debug(
        f"""
            TS info total={len(lens)}, mean len={np.mean(lens)}, median len{np.median(lens)},
              min len={np.min(lens)}, max len={np.max(lens)}
              """
    )
    return static_data, ts_data_clean, (lens, sizes, rel_times)


def prepare_ts_datasets(
    workspace=Path("workspace"),
    ID_COL="file_order",
    domain_limit=50,
    class_cnt_limit=1024,
):  # id, full_id
    in_workspace = workspace / Path("output_csv_full")
    if not in_workspace.exists():
        log.error("Missing output_csv_full data")
        return

    output = workspace / Path("output_wefde")
    output.mkdir(parents=True, exist_ok=True)

    full_data_static = pd.read_csv(in_workspace / "static_data.csv")
    full_data_temporal = pd.read_csv(in_workspace / "temporal_data.csv")

    static_data = full_data_static
    constant = _constant_columns(static_data)
    static_data = static_data.drop(columns=constant)

    static_ids = static_data[ID_COL].values

    temporal_data = full_data_temporal[full_data_temporal[ID_COL].isin(static_ids)]
    static_data = static_data[static_data[ID_COL].isin(temporal_data[ID_COL].values)]

    (clean_static_data, clean_ts_data, _) = _prepare_time_series(
        static_data,
        temporal_data,
        ID_COL=ID_COL,
    )

    real_idx = 0
    domain_repeats = {}
    domain_label = {}

    experiment_labels = []
    for ridx, static_row in clean_static_data.iterrows():
        encoded_label = static_row["label"]
        experiment_labels.append(encoded_label)

    experiment_labels = list(sorted(list(set(experiment_labels))))
    for ridx, static_row in clean_static_data.iterrows():
        local_token = static_row["label"]
        encoded_label = experiment_labels.index(local_token)

        if local_token not in domain_repeats:
            if len(domain_repeats) > class_cnt_limit:
                real_idx += 1
                continue
            domain_repeats[local_token] = 0
            domain_label[local_token] = encoded_label

        if domain_repeats[local_token] >= domain_limit:
            real_idx += 1
            continue

        domain_repeats[local_token] += 1
        outfile = f"{domain_label[local_token]}-{domain_repeats[local_token]}"

        if (output / outfile).exists():
            real_idx += 1
            continue

        local_sizes = (
            clean_ts_data[real_idx]["length"].values
            * clean_ts_data[real_idx]["direction"].values
        )
        timestamps = clean_ts_data[real_idx]["relative_timestamp"].copy()
        timestamps[timestamps < 0] = 0  # WTF
        local_ts = timestamps.values
        assert len(local_ts) == len(local_sizes)
        assert (local_ts >= 0).all(), timestamps.values

        new_local_data = pd.DataFrame(np.asarray([local_ts, local_sizes]).T)
        new_local_data.to_csv(output / outfile, sep=" ", index=False, header=False)

        real_idx += 1


def prepare_features(
    workspace=Path("workspace"),
    conn_limit: int = 5,
):
    time_series_traces = workspace / Path("output_wefde")
    if not time_series_traces.exists():
        log.error("Missing output_wefde data. Call prepare_ts_datasets first!")
        return

    output = workspace / Path("eval_features")
    output.mkdir(parents=True, exist_ok=True)

    return prepare_wefde_features(
        trace_path=time_series_traces,
        out_path=output,
        conn_limit=conn_limit,
    )


def prepare_ts_datasets_for_nns_1C(
    workspace=Path("workspace"),
):
    wefde_path = workspace / Path("eval_features")

    if not wefde_path.exists():
        log.error("Missing output_wefde features. Run prepare_features first!")
        return

    with open(wefde_path / "FeaturePositions.json", "r") as f:
        features = json.load(f)

    output = workspace / Path("output_ml")
    output.mkdir(parents=True, exist_ok=True)

    X, y = load_wefde_features(wefde_path)
    start_off = 0
    for feat in features:
        end_off = features[feat]
        X[:, start_off:end_off] = StandardScaler().fit_transform(
            X[:, start_off:end_off]
        )
        start_off = end_off

    X = np.expand_dims(X, axis=1)

    X = np.asarray(X)
    y = np.asarray(y)

    with open(output / "X_1C.npy", "wb") as f:
        np.save(f, X)
    with open(output / "y_1C.npy", "wb") as f:
        np.save(f, y)

    return X, y


def prepare_ts_datasets_for_nns_3C(
    workspace=Path("workspace"),
    ID_COL="file_order",
    domain_limit=50,
    class_cnt_limit=1024,
):  # id, full_id
    in_workspace = workspace / Path("output_csv_full")
    if not in_workspace.exists():
        log.error("Missing output_csv_full data. Run merge_pcaps_csv first!")
        return

    output = workspace / Path("output_ml")
    output.mkdir(parents=True, exist_ok=True)

    static_data = pd.read_csv(in_workspace / "static_data.csv")
    full_data_temporal = pd.read_csv(in_workspace / "temporal_data.csv")

    constant = _constant_columns(static_data)
    static_data = static_data.drop(columns=constant)

    static_ids = static_data[ID_COL].values

    temporal_data = full_data_temporal[full_data_temporal[ID_COL].isin(static_ids)]
    static_data = static_data[static_data[ID_COL].isin(temporal_data[ID_COL].values)]

    (
        clean_static_data,
        clean_ts_data,
        (lens, pkt_sizes, pkt_ts),
    ) = _prepare_time_series(
        static_data,
        temporal_data,
        ID_COL=ID_COL,
    )

    padlimit = min(max(lens), 1000)

    def _preprocess(arr):
        return np.log1p(arr + 1e-6)

    def _pad(arr, arrsize: int = padlimit):
        arr = np.asarray(arr).tolist()
        if len(arr) > arrsize:
            arr = arr[:arrsize]
        else:
            arr = np.pad(arr, (0, arrsize - len(arr)), "constant", constant_values=0)
        assert len(arr) == arrsize

        return np.asarray(arr)

    real_idx = 0
    domain_repeats = {}
    domain_label = {}

    experiment_labels = []
    for ridx, static_row in clean_static_data.iterrows():
        encoded_label = static_row["label"]
        experiment_labels.append(encoded_label)

    experiment_labels = list(sorted(list(set(experiment_labels))))

    X = []
    y = []

    for ridx, static_row in clean_static_data.iterrows():
        local_token = static_row["label"]
        encoded_label = experiment_labels.index(local_token)

        if local_token not in domain_repeats:
            if len(domain_repeats) > class_cnt_limit:
                real_idx += 1
                continue
            domain_repeats[local_token] = 0
            domain_label[local_token] = encoded_label

        if domain_repeats[local_token] >= domain_limit:
            real_idx += 1
            continue

        domain_repeats[local_token] += 1

        local_sizes = clean_ts_data[real_idx]["length"].values
        local_sizes = _preprocess(local_sizes)
        local_sizes = _pad(local_sizes)

        local_dir = clean_ts_data[real_idx]["direction"].values
        local_dir = _pad(local_dir)

        timestamps = clean_ts_data[real_idx]["relative_timestamp"].copy()
        timestamps[timestamps < 0] = 0  # WTF
        local_ts = timestamps.values
        local_ts = _pad(local_ts)

        assert len(local_ts) == len(local_sizes)

        X.append([local_dir, local_ts, local_sizes])
        y.append(encoded_label)

        real_idx += 1

    X = np.asarray(X)
    y = np.asarray(y)

    with open(output / "X_3C.npy", "wb") as f:
        np.save(f, X)
    with open(output / "y_3C.npy", "wb") as f:
        np.save(f, y)

    return X, y


def create_datasets(
    traces=Path("traces"),
    workspace=Path("workspace"),
    unlink_after_processing=True,
):
    workspace.mkdir(parents=True, exist_ok=True)

    # Parse raw pcaps
    process_raw_pcaps(
        traces=traces,
        workspace=workspace,
        unlink_after_processing=unlink_after_processing,
    )

    # Merge CSV in a single dataset
    merge_pcap_csvs(workspace=workspace)

    # Create Time-Series datasets
    prepare_ts_datasets(workspace=workspace)
    prepare_features(workspace=workspace)

    # Create datasets for NNs
    prepare_ts_datasets_for_nns_1C(workspace=workspace)
    prepare_ts_datasets_for_nns_3C(workspace=workspace)
