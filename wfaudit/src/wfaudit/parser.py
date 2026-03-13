# stdlib
from collections import defaultdict
import glob
import hashlib
from pathlib import Path
from random import shuffle
from typing import List, Optional, Tuple

# third party
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tqdm import tqdm

# wfaudit absolute
from wfaudit.helpers_deepse import prepare_deepse_dataset
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
            return

        label = stem.split("_")[1]
        static_data, temporal_data = session.temporal_stats_per_flow()
        if len(static_data) == 0:
            log.error(f"empty dataset {filename}")
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
    temporal_lim: int = 50_000,
    cache: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stream merge the individual per PCAP CSVs into two partitioned Parquet files.
    """
    in_workspace = workspace / "output_csv_single"

    out_workspace = workspace / "output_csv_merged"
    out_workspace.mkdir(parents=True, exist_ok=True)

    static_files = glob.glob(str(in_workspace / "static*.csv"))
    print("static files", len(static_files))

    static_parquet = out_workspace / "static_data.parquet"
    temporal_parquet = out_workspace / "temporal_data.parquet"
    if static_parquet.exists() and temporal_parquet.exists():
        print("Parquet cache found skipping CSV merge.")
        static_ds = ds.dataset(static_parquet, format="parquet")
        temporal_ds = ds.dataset(temporal_parquet, format="parquet")

        return static_ds, temporal_ds

    pq_static: pq.ParquetWriter = None
    pq_temporal: pq.ParquetWriter = None

    buf_static, buf_temporal = [], []
    for fidx, static_filename in enumerate(tqdm(static_files, desc="merge-csvs")):
        try:
            s_df = pd.read_csv(
                static_filename, dtype={"label": "category"}, engine="pyarrow"
            )
            t_df = pd.read_csv(
                in_workspace
                / Path(static_filename).name.replace("static_", "temporal_"),
                engine="pyarrow",
                dtype={
                    "relative_timestamp": "float32",
                    "length": "int32",
                    "direction": "int8",
                },
            )
            if temporal_lim is not None:
                t_df = t_df.head(temporal_lim)
        except BaseException as e:
            print("Failed to read csv", e, static_filename)
            continue

        # ------ post‑processing that the original version performed ------
        original_id, original_label = s_df["id"].iloc[0], s_df["label"].iloc[0]
        total_dur = t_df["relative_timestamp"].sum()
        hashed_id = hashlib.sha1(
            f"{original_id}-{original_label}-{total_dur}-{fidx}".encode()
        ).hexdigest()

        for df in (s_df, t_df):
            df["file_order"] = fidx
            df["full_id"] = hashed_id
            df["id"] = df["id"].astype(str)

        # ------------------- batch‑to‑disk -------------------            ➌
        buf_static.append(pa.Table.from_pandas(s_df, preserve_index=False))
        buf_temporal.append(pa.Table.from_pandas(t_df, preserve_index=False))

        if (fidx + 1) % pd_lim == 0:
            if pq_static is None:  # first batch → open writers
                pq_static = pq.ParquetWriter(
                    out_workspace / "static_data.parquet", buf_static[0].schema
                )
                pq_temporal = pq.ParquetWriter(
                    out_workspace / "temporal_data.parquet", buf_temporal[0].schema
                )
            for tbl_static, tbl_temp in zip(buf_static, buf_temporal):
                if len(tbl_static) == 0 or len(tbl_temp) == 0:
                    continue
                pq_static.write_table(tbl_static)
                pq_temporal.write_table(tbl_temp)
            buf_static.clear(), buf_temporal.clear()

    # ---------- flush leftovers & close ----------
    if pq_static is None:  # we never triggered the batch flush
        pq_static = pq.ParquetWriter(
            out_workspace / "static_data.parquet", buf_static[0].schema
        )
        pq_temporal = pq.ParquetWriter(
            out_workspace / "temporal_data.parquet", buf_temporal[0].schema
        )

    for tbl_static, tbl_temp in zip(buf_static, buf_temporal):
        if len(tbl_static) == 0 or len(tbl_temp) == 0:
            continue
        pq_static.write_table(tbl_static)
        pq_temporal.write_table(tbl_temp)
    pq_static.close(), pq_temporal.close()

    # ---------- return *lazy* DataFrames backed by the parquet files ----------
    static_ds = pa.dataset.dataset(
        out_workspace / "static_data.parquet", format="parquet"
    )
    temporal_ds = pa.dataset.dataset(
        out_workspace / "temporal_data.parquet", format="parquet"
    )
    return static_ds, temporal_ds


def _prepare_time_series_arrow(
    static_ds: ds.Dataset,
    ts_ds: ds.Dataset,
    ts_limit: Optional[int] = 50_000,
    ts_pad: int = 0,
    ID_COL: str = "file_order",  # id, full_id
    batch_rows: int = 10000,
) -> Tuple[
    pd.DataFrame, List[pd.DataFrame], Tuple[List[int], List[float], List[float]]
]:
    """
    Streaming, low RAM version of `_prepare_time_series` that works on
    `pyarrow.dataset.Dataset` inputs and **guarantees that each ID is gathered
    in full even when it spans multiple record batches**.
    """
    # 1)  materialise the tiny static table                              #
    static_df = (
        static_ds.to_table(columns=[ID_COL, "label"])
        .to_pandas(split_blocks=True, self_destruct=True)
        .drop_duplicates(ID_COL)
    )
    static_ids = set(static_df[ID_COL].values)

    # 2)  streaming scan with a per ID stash                             #
    pending = defaultdict(list)  # id  -> list[pd.DataFrame] (fragments)
    ts_data_clean, final_ids = [], []

    lens, sizes, rel_times = [], [], []
    print("processing time series (Arrow batches)")

    for batch in tqdm(ts_ds.to_batches(batch_size=batch_rows)):
        pdf = batch.to_pandas(split_blocks=True, self_destruct=True)

        for idx, grp in pdf.groupby(ID_COL):
            if idx not in static_ids:
                continue

            pending[idx].append(
                grp.drop(columns=["id", "full_id", "duration"], errors="ignore")
            )

            # If ts_limit is reached, or we know the group is complete, flush now
            if ts_limit is not None and sum(len(x) for x in pending[idx]) >= ts_limit:
                _ts_helper_flush_id(
                    idx,
                    pending,
                    ts_data_clean,
                    final_ids,
                    lens,
                    sizes,
                    rel_times,
                    ts_limit,
                    ts_pad,
                )
        # (otherwise we wait for more fragments)

    # 3)  flush any IDs that ended in the last batch                     #
    for idx in list(pending):  # iterate over *copy* we mutate inside
        _ts_helper_flush_id(
            idx,
            pending,
            ts_data_clean,
            final_ids,
            lens,
            sizes,
            rel_times,
            ts_limit,
            ts_pad,
        )

    # 4)  re-index the static DF and log stats                           #
    static_df = static_df.set_index(ID_COL).reindex(final_ids)

    log.debug(
        f"TS info total={len(lens)}, "
        f"mean len={np.mean(lens):.2f}, median len={np.median(lens):.0f}, "
        f"min len={np.min(lens)}, max len={np.max(lens)}"
    )

    return static_df, ts_data_clean, (lens, sizes, rel_times)


def _ts_helper_flush_id(
    idx, pending, ts_data_clean, final_ids, lens, sizes, rel_times, ts_limit, ts_pad
):
    parts = pending.pop(idx)  # remove from stash
    local = pd.concat(parts, ignore_index=True)

    lens.append(len(local))
    sizes.extend(local["length"].values.tolist())
    rel_times.extend(local["relative_timestamp"].values.tolist())

    # optional trim/pad exactly like the legacy version
    if ts_limit is not None:
        if len(local) > ts_limit:
            local = local.head(ts_limit)
        elif len(local) < ts_limit:
            pad_rows = ts_pad * np.ones((ts_limit - len(local), len(local.columns)))
            local = pd.concat(
                [local, pd.DataFrame(pad_rows, columns=local.columns)],
                ignore_index=True,
            )

    ts_data_clean.append(local)
    final_ids.append(idx)


def _prepare_time_series_pandas(
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


def prepare_wefde_raw(
    static_data,
    temporal_data,
    workspace=Path("workspace"),
    ID_COL="file_order",
    domain_limit=1000,
    class_cnt_limit=1024,
    wefde_folder: str = "output_wefde",
):  # id, full_id
    output = workspace / wefde_folder
    output.mkdir(parents=True, exist_ok=True)

    print("process time series")
    (clean_static_data, clean_ts_data, _) = _prepare_time_series_arrow(
        static_data,
        temporal_data,
        ID_COL=ID_COL,
    )
    print("processed time series", clean_static_data.shape)

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
    for ridx, static_row in tqdm(clean_static_data.iterrows()):
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


def prepare_wefde_dataset(
    workspace=Path("workspace"),
    conn_limit=3,
    wefde_folder: str = "output_wefde",
    wefde_feats_folder: str = "output_features",
):
    time_series_traces = workspace / wefde_folder
    if not time_series_traces.exists():
        log.error(f"Missing {wefde_folder} data. Call prepare_wefde_raw first!")
        return

    output = workspace / wefde_feats_folder
    output.mkdir(parents=True, exist_ok=True)

    return prepare_wefde_features(
        trace_path=time_series_traces,
        out_path=output,
        conn_limit=conn_limit,
    )


def prepare_all_datasets(
    workspace=Path("workspace"),
    n_websites: int = 100,
    n_traces: int = 500,
    feature_length: int = 5000,
    deepse_testtypes=["real", "sanity"],
    wefde_folder: str = "output_wefde",
    wefde_feats_folder: str = "output_features",
):
    print("merge raw datasets")
    static_data, ts_data = merge_pcap_csvs(workspace=workspace)

    print("prepare WeFDE data")
    prepare_wefde_raw(
        static_data, ts_data, workspace=workspace, wefde_folder=wefde_folder
    )
    prepare_wefde_dataset(
        workspace=workspace,
        wefde_folder=wefde_folder,
        wefde_feats_folder=wefde_feats_folder,
    )

    print("prepare DeepSE-WF features")
    for testtype in deepse_testtypes:
        prepare_deepse_dataset(
            path_wefde=workspace / wefde_folder,
            path_out=workspace / "output_deepse" / testtype / "dataset.npz",
            n_websites=n_websites,
            n_traces=n_traces,
            feature_length=feature_length,
            debug_mode=(testtype != "real"),
        )
