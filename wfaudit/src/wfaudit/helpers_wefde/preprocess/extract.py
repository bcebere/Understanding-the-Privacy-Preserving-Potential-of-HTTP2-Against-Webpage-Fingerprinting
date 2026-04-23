# Adapted from https://github.com/notem/reWeFDE
# future
from __future__ import division

# stdlib
from collections import OrderedDict
import json
import os
from pathlib import Path
import re

# third party
from joblib import Parallel, delayed
import pandas as pd

# wfaudit absolute
import wfaudit.helpers_wefde.preprocess.features.Burst as Burst
import wfaudit.helpers_wefde.preprocess.features.Cumul as Cumul
import wfaudit.helpers_wefde.preprocess.features.PktNum as PktNum
import wfaudit.helpers_wefde.preprocess.features.Time as Time
from wfaudit.helpers_wefde.preprocess.util import FEATURE_EXT


def enumerate_files(dir, splitter="-", extension=""):
    """
    recursively enumerate files in a directory root
    """
    file_list = []
    for dirname, dirnames, filenames in os.walk(dir):
        # filter out invalid file names
        filenames = [
            filename
            for filename in filenames
            if re.fullmatch(f"\\d+{splitter}\\d+{extension}", filename)
        ]
        for filename in filenames:
            file_list.append(os.path.join(dirname, filename))
    return file_list


def extract(times, sizes, debug_path: str, conn_limit: int = 1):
    """
    extract features from a parsed website trace
    """
    feature_pos = OrderedDict()
    features = []

    # Transmission size features
    features.extend(PktNum.get_packet_counts(times, sizes, conn_limit=conn_limit))
    feature_pos["MI_PKT_COUNT"] = len(features)

    # inter packet time + transmission time feature
    features.extend(Time.get_time_features(times, sizes, conn_limit=conn_limit))
    feature_pos["MI_PKT_TIME"] = len(features)

    # Bursts (knn)
    features.extend(Burst.get_burst_features(times, sizes, conn_limit=conn_limit))
    feature_pos["MI_BURST"] = len(features)

    # CUMUL features
    features.extend(Cumul.get_cumul_features(times, sizes, conn_limit=conn_limit))
    feature_pos["MI_CUMUL"] = len(features)

    # output FeaturePos
    with open(os.path.join(debug_path, "FeaturePositions.json"), "w") as fd:
        fd.write(json.dumps(feature_pos))

    return features


def task_handler(filepath: str, out_path: str, conn_limit: int):
    """
    handle feature extraction for each trace instance assigned to batch
    """
    # load trace file
    x = pd.read_csv(filepath, sep=" ", header=None)

    mask_nonzero = (x != 0).any(axis=1)  # True where there's at least one non-zero
    last_nonzero_idx = mask_nonzero[::-1].idxmax()  # index of last non-zero row
    x = x.loc[:last_nonzero_idx]

    if len(x) == 0:
        print("Ignore empty dataset")
        return

    # print("Before after ", orig_len, len(x))

    times = x.iloc[:, 0].astype(float).values.tolist()

    HTTP2_DEF_WINDOW_SIZE = 65535
    sizes = (x.iloc[:, 1].astype(float).values / HTTP2_DEF_WINDOW_SIZE).tolist()

    # extract features (saving feature positions only for the first trace)
    if len(times) < 4:
        return

    # print("Processing ", filepath)
    features = extract(
        times,
        sizes,
        debug_path=out_path,
        conn_limit=conn_limit,
    )

    # print(f"Features {filepath} --> {len(features)} = {features}")

    # save features to file
    dest = os.path.join(out_path, os.path.basename(filepath) + FEATURE_EXT)
    with open(dest, "w") as fout:
        for x in features:
            if isinstance(x, str):
                if "\n" in x:
                    fout.write(x)
                else:
                    fout.write(x + " ")
            else:
                fout.write(repr(x) + " ")


def prepare_wefde_features(trace_path, out_path, conn_limit: int = 1):
    """
    start batches to handle feature extraction
    """
    file_list = enumerate_files(trace_path)
    Parallel(n_jobs=20)(
        delayed(task_handler)(f, out_path, conn_limit) for f in file_list
    )

    features = json.load(open(Path(out_path) / "FeaturePositions.json"))
    # print("Features -> ", features)
    return features
