# stdlib
import argparse
from pathlib import Path

# third party
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser("Prepare traces for DeepSE-WF", add_help=False)

    # Data and Setup
    parser.add_argument(
        "--in_path",
        type=str,
        required=True,
        help="Path to data folder which should contain traces.npy and labels.npy",
    )

    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Path to .npz file containing the traces and labels",
    )

    parser.add_argument(
        "--n_traces",
        type=int,
        required=True,
        help="Number of traces to use per website",
    )

    parser.add_argument(
        "--n_websites",
        type=int,
        required=True,
        help="Number of websites in the dataset",
    )

    parser.add_argument(
        "--debug_mode",
        type=int,
        default=0,
        help="Run sanity checks with shuffled labels",
    )

    parser.add_argument(
        "--feature_length",
        default=10000,
        type=int,
        help="Length of each packet sequence",
    )

    return parser


def pad_or_truncate(some_list, target_len):
    return np.concatenate(
        (some_list[:target_len], np.array([0] * (target_len - len(some_list))))
    ).astype(float)


def interleaved_label_shuffle(y):
    y = np.array(y)
    classes = np.unique(y)
    per_class = {cls: np.where(y == cls)[0] for cls in classes}

    # Shuffle indices within each class
    for idxs in per_class.values():
        np.random.shuffle(idxs)

    # Determine how many evenly interleaved rounds we can do
    min_count = min(len(idxs) for idxs in per_class.values())

    # Interleave evenly
    interleaved_indices = []
    for i in range(min_count):
        for cls in classes:
            interleaved_indices.append(per_class[cls][i])

    # Collect the leftovers
    leftover_indices = []
    for cls in classes:
        leftover_indices.extend(per_class[cls][min_count:])

    # Shuffle leftovers and append
    np.random.shuffle(leftover_indices)
    final_indices = interleaved_indices + leftover_indices

    shuffled_y = y[final_indices]

    # Sanity checks
    assert len(shuffled_y) == len(y)
    # assert np.all(np.bincount(shuffled_y) == np.bincount(y))
    print("Label correlation:", np.corrcoef(y, shuffled_y)[0, 1])

    return shuffled_y


def prepare_deepse_dataset(
    path_wefde: str,  # WeFDE dataset path
    path_out: str,  # DeepSE dataset path
    n_websites: int = 100,  # Maximum website count
    n_traces: int = 500,  # Maximum samples per website
    feature_length=5000,  # Maximum time-series length to keep
    debug_mode: bool = False,  # Shuffle labels for sanity checks
):
    print("prepare dataset")
    if not str(path_out).endswith(".npz"):
        raise ValueError("Data path must end with .npz")

    path_out = Path(path_out)
    path_wefde = Path(path_wefde)
    folder = path_out.parent
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
        print(f"[*] Created the directory(s): {folder}")

    print("create dataset path")
    X = []
    y = []

    lens = []

    for w in tqdm(range(n_websites)):
        for t in range(1, n_traces + 1):
            trace_file = f"{w}-{t}"
            trace = np.loadtxt(path_wefde / trace_file)

            # Count trailing zero rows
            count_trailing_zeros = 0
            for row in reversed(trace):
                if np.all(row == 0):
                    count_trailing_zeros += 1
                else:
                    break

            lens.append(len(trace) - count_trailing_zeros)
            break

    print("Trace stats", np.mean(lens), np.median(lens), np.max(lens))
    feature_length = min(feature_length, max(100, int(np.median(lens)) + 50))

    for w in tqdm(range(n_websites)):
        available = 0
        for t in range(1, n_traces + 1):
            trace_file = f"{w}-{t}"
            trace_path = Path(path_wefde) / trace_file

            if not trace_path.exists():  # hack
                continue

            available += 1

        if available < n_traces - 100:
            print("[x] Ignore low trace count for website=", w, available, flush=True)
            continue

        for t in range(1, n_traces + 1):
            trace_file = f"{w}-{t}"
            trace_path = path_wefde / trace_file

            if not trace_path.exists():  # hack
                print("[x] Ignore missing", trace_path)
                continue

            trace = np.loadtxt(trace_path)

            lens.append(len(trace))
            assert (
                len(np.shape(trace)) > 1
            ), "Trace should be in time 'tab' direction format."

            # direction (+1 outgoing, -1 incoming)
            sign = np.sign(trace[:, 1])

            # timing
            ch_timing = sign * trace[:, 0]  # convert to signed time
            ch_timing = pad_or_truncate(
                np.asarray(ch_timing, dtype=float), feature_length
            )

            # bytes
            scale = 2000
            ch_sizes = sign * (np.abs(trace[:, 1]) / scale)
            ch_sizes = pad_or_truncate(
                np.asarray(ch_sizes, dtype=float), feature_length
            )

            trace = np.stack([ch_timing, ch_sizes], axis=0)
            X.append(trace)
            y.append(w)

    # prepare X
    X = np.array(X, dtype=np.float32)
    assert len(X) > 0
    mu = X.mean(axis=(0, 2), keepdims=True)  # shape (1,2,1)
    sigma = X.std(axis=(0, 2), keepdims=True) + 1e-8
    X = (X - mu) / sigma
    assert np.isnan(X).sum() == 0

    # prepare y
    y = np.array(y, dtype=np.int32)
    if debug_mode:
        print("DANGER !!! Corrupting y")
        # Corrupt Y
        y = interleaved_label_shuffle(y)

    print(f"[*] Data shape {X.shape}")
    assert X.shape[2] == feature_length
    assert X.shape[1] == 2
    assert X.shape[0] > 0

    cls_count = pd.Series(y).value_counts()
    assert min(cls_count) > 0, cls_count

    np.savez_compressed(path_out, traces=X, labels=y)
