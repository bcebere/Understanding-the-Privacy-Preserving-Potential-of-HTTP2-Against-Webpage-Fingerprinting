# stdlib
import glob
import hashlib
from pathlib import Path
from random import shuffle
from typing import Optional

# third party
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from tqdm import tqdm

# wfaudit absolute
from wfaudit.helpers_wefde.analysis.data_utils import load_wefde_features
from wfaudit.helpers_wefde.preprocess.extract import prepare_wefde_features
import wfaudit.logger as log
from wfaudit.processing import process_pcap, process_pcap_via_json

np.set_printoptions(suppress=True)


def process_raw_pcaps(
    traces=Path("traces"),
    workspace=Path("workspace"),
    unlink_after_processing=True,
    buffer_tcp: bool = True,
    n_jobs=8,
    files=None,
    use_json=False,
):
    """
    Args:
        - traces: Folder with the PCAPs to be parsed. --- traces / "*.pcap"
        - workspace : Folder where to store intermediary and final CSVs.
        - unlink_after_processing: Delete the PCAP after processing. Useful for low-space devices.
    """
    if not traces.exists():
        log.error("missing traces folder")
        return []
    workspace.mkdir(parents=True, exist_ok=True)
    output = workspace / "output_csv_single"
    output.mkdir(parents=True, exist_ok=True)

    if files is None:
        files = glob.glob(str(traces / "*.pcap"))

    print(f"Parsing {len(files)} PCAPS to {workspace}")
    shuffle(files)

    def _parse_single_pcap(filename):
        filename = Path(filename)
        stem = filename.stem
        output_csv_static = output / f"static_data_{stem}.csv"
        output_csv_temporal = output / f"temporal_data_{stem}.csv"
        if not filename.exists():
            print("Missing file !!!", filename)
            return

        if output_csv_temporal.exists():
            if unlink_after_processing:
                log.debug(f"dropping  {filename}")
                if unlink_after_processing:
                    filename.unlink()
            print("temporal file already done", filename)
            return
        log.debug(f"Parsing {filename}")
        try:
            if use_json:
                session = process_pcap_via_json(filename, buffer_tcp=buffer_tcp)
            else:
                session = process_pcap(filename, buffer_tcp=buffer_tcp)
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

        print("Done parsing", filename)
        if unlink_after_processing:
            filename.unlink()

    Parallel(n_jobs=n_jobs)(delayed(_parse_single_pcap)(filename) for filename in files)

    return files


def merge_pcap_csvs(
    workspace=Path("workspace"),
    pd_lim: int = 1000,
    temporal_lim: int = None,
    cache: bool = False,
) -> None:
    """
    Args:
        workspace: The folder which contains the post-processed pcaps --- output_csv_single.
        pd_lim: how often to batch the CSVs.
    """
    in_workspace = workspace / Path("output_csv_single")
    if not in_workspace.exists():
        log.error("Missing output_csv_single folder")
        return
    static_files = glob.glob(str(in_workspace / "static*.csv"))

    print(in_workspace)
    print("static files", len(static_files))
    buffer_static = []
    buffer_temporal = []
    temporal_total_len = 0

    for fidx, filename in tqdm(enumerate(static_files)):
        static_filename = Path(filename)
        base = static_filename.name.split("static_")[1]
        temporal_base = "temporal_" + base
        temporal_filename = in_workspace / temporal_base

        assert static_filename.exists()
        assert temporal_filename.exists()
        try:
            local_static_csv = pd.read_csv(static_filename, engine="pyarrow")
            if temporal_lim is None:
                local_temporal_csv = pd.read_csv(temporal_filename, engine="pyarrow")
            else:
                local_temporal_csv = pd.read_csv(
                    temporal_filename, engine="pyarrow"
                ).head(temporal_lim)
        except BaseException as e:
            print("failed to read", filename, e)
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

        buffer_static.append(local_static_csv)
        buffer_temporal.append(local_temporal_csv)

        temporal_total_len += len(local_temporal_csv)

        if len(buffer_temporal) % 1000 == 0:
            print(
                "processed temporal series ",
                temporal_total_len,
                temporal_total_len / (len(buffer_temporal) + 1),
            )

    print(f"static {len(buffer_static)} CSVs")
    full_data_static = pd.concat(buffer_static, ignore_index=True, copy=False)
    print(f"temporal {len(buffer_temporal)} CSVs")
    full_data_temporal = pd.concat(buffer_temporal, ignore_index=True, copy=False)

    return full_data_static, full_data_temporal

    # print("saving !!")
    if cache:
        output = workspace / Path("output_csv_full")
        output.mkdir(parents=True, exist_ok=True)

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
    print("processing time series")
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
    static_data,
    temporal_data,
    workspace=Path("workspace"),
    ID_COL="file_order",
    domain_limit=1000,
    class_cnt_limit=1024,
):  # id, full_id
    output = workspace / Path("output_wefde")
    output.mkdir(parents=True, exist_ok=True)

    print("process time series")
    (clean_static_data, clean_ts_data, _) = _prepare_time_series(
        static_data,
        temporal_data,
        ID_COL=ID_COL,
    )

    real_idx = 0
    domain_repeats = {}
    domain_label = {}

    print("process labels")
    experiment_labels = []
    for ridx, static_row in clean_static_data.iterrows():
        encoded_label = static_row["label"]
        experiment_labels.append(encoded_label)

    print("process wefde features")
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
        timestamps[timestamps > 1000] = 0  # Parsing bug

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

    output = workspace / Path("output_features")
    output.mkdir(parents=True, exist_ok=True)

    return prepare_wefde_features(
        trace_path=time_series_traces,
        out_path=output,
        conn_limit=conn_limit,
    )


def prepare_ts_datasets_for_nns_1C(
    workspace=Path("workspace"),
):
    wefde_path = workspace / Path("output_features")

    if not wefde_path.exists():
        log.error("Missing output_wefde features. Run prepare_features first!")
        raise

    # with open(wefde_path / "FeaturePositions.json", "r") as f:
    #    features = json.load(f)

    output = workspace / Path("output_ml")
    output.mkdir(parents=True, exist_ok=True)

    X, y = load_wefde_features(wefde_path)
    # start_off = 0
    # for feat in features:
    #    end_off = features[feat]
    #    X[:, start_off:end_off] = StandardScaler().fit_transform(
    #        X[:, start_off:end_off]
    #    )
    #    start_off = end_off

    X = np.expand_dims(X, axis=1)

    X = np.asarray(X)
    y = np.asarray(y)

    with open(output / "X_1C.npy", "wb") as f:
        np.save(f, X)
    with open(output / "y_1C.npy", "wb") as f:
        np.save(f, y)


def prepare_ts_datasets_for_nns_3C(
    static_data,
    temporal_data,
    workspace=Path("workspace"),
    ID_COL="file_order",
    domain_limit=1000,
    class_cnt_limit=1024,
):  # id, full_id
    output = workspace / Path("output_ml")
    output.mkdir(parents=True, exist_ok=True)

    (
        clean_static_data,
        clean_ts_data,
        (lens, pkt_sizes, pkt_ts),
    ) = _prepare_time_series(
        static_data,
        temporal_data,
        ID_COL=ID_COL,
    )

    padlimit = min(int(np.median(lens)) + 10, 500)

    def _pad(arr, arrsize: int = padlimit):
        arr = np.asarray(arr).tolist()
        if len(arr) > arrsize:
            arr = arr[:arrsize]
        else:
            arr = np.pad(arr, (0, arrsize - len(arr)), "constant", constant_values=0)
        assert len(arr) == arrsize

        return np.asarray(arr)

    def _stats(arr):
        return [
            len(arr),
            float(np.max(arr)),
            float(np.mean(arr)),
            float(np.std(arr)),
            float(np.sum(arr)),
        ]

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
        local_sizes = local_sizes / 1024  # KB
        local_sizes_stats = _stats(local_sizes)
        local_sizes = np.asarray(list(local_sizes_stats) + list(local_sizes))
        local_sizes = _pad(local_sizes)

        local_dir = clean_ts_data[real_idx]["direction"].values
        local_dir_stats = _stats(local_dir)
        local_dir = np.asarray(list(local_dir_stats) + list(local_dir))
        local_dir = _pad(local_dir)

        timestamps = clean_ts_data[real_idx]["relative_timestamp"].copy()
        timestamps[timestamps < 0] = 0  # WTF
        timestamps[timestamps > 1000] = 0  # Parsing bug

        local_ts = timestamps.values
        local_ts_stats = _stats(local_ts)
        local_ts = np.asarray(list(local_ts_stats) + list(local_ts))
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


def prepare_datasets(
    workspace=Path("workspace"),
    conn_limit: int = 5,
):
    print("merge datasets")
    static_data, ts_data = merge_pcap_csvs(workspace=workspace)
    print("prepare wefde data")
    prepare_ts_datasets(static_data, ts_data, workspace=workspace)

    print("prepare wefde features")
    prepare_features(workspace=workspace, conn_limit=conn_limit)

    print("prepare NN features")
    prepare_ts_datasets_for_nns_1C(workspace=workspace)

    prepare_ts_datasets_for_nns_3C(static_data, ts_data, workspace=workspace)
