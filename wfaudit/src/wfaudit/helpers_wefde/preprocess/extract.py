# Adapted from https://github.com/notem/reWeFDE
# future
from __future__ import division

# stdlib
from collections import OrderedDict
import itertools
import json
from multiprocessing import Pool
import os
import re

# third party
import pandas as pd
from tqdm import tqdm

# wfaudit absolute
import wfaudit.helpers_wefde.preprocess.features.Burst as Burst
import wfaudit.helpers_wefde.preprocess.features.Cumul as Cumul
import wfaudit.helpers_wefde.preprocess.features.PktLen as PktLen
import wfaudit.helpers_wefde.preprocess.features.PktNum as PktNum
import wfaudit.helpers_wefde.preprocess.features.PktSec as PktSec
import wfaudit.helpers_wefde.preprocess.features.Time as Time
from wfaudit.helpers_wefde.preprocess.util import FEATURE_EXT, featureCount


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


def extract(times, sizes, debug_path="./"):
    """
    extract features from a parsed website trace
    """
    feature_pos = OrderedDict()
    features = []

    # Transmission size features
    features.extend(PktNum.PacketNumFeature(times, sizes))
    feature_pos["PACKET_NUMBER"] = len(features)

    # inter packet time + transmission time feature
    features.extend(Time.TimeFeature(times, sizes))
    feature_pos["PKT_TIME"] = len(features)

    # Unique packet lengths
    features.extend(PktLen.PktLenFeature(times, sizes))
    feature_pos["UNIQUE_PACKET_LENGTH"] = len(features)

    # Bursts (knn)
    features.extend(Burst.BurstFeature(times, sizes))
    feature_pos["BURST"] = len(features)

    # packets per second (k-anonymity)
    # plus alternative list
    features.extend(PktSec.PktSecFeature(times, sizes))
    feature_pos["PKT_PER_SECOND"] = len(features)

    # CUMUL features
    features.extend(Cumul.CumulFeatures(sizes, featureCount))
    feature_pos["CUMUL"] = len(features)

    # output FeaturePos
    with open(os.path.join(debug_path, "FeaturePositions.json"), "w") as fd:
        fd.write(json.dumps(feature_pos))

    return features


def task_handler(args):
    """
    handle feature extraction for each trace instance assigned to batch
    """
    filepath, out_path = args

    # load trace file
    x = pd.read_csv(filepath, sep=" ", header=None)
    times = x.iloc[:, 0].astype(float).values.tolist()
    sizes = x.iloc[:, 1].astype(int).values.tolist()

    # extract features (saving feature positions only for the first trace)
    if len(times) < 4:
        return

    features = extract(
        times,
        sizes,
        debug_path=out_path,
    )

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


def prepare_wefde_features(trace_path, out_path):
    """
    start batches to handle feature extraction
    """
    file_list = enumerate_files(trace_path)

    # start BATCH_NUM processes for computation
    pool = Pool()
    for _ in tqdm(
        pool.imap(task_handler, zip(file_list, itertools.repeat(out_path))),
        total=len(file_list),
    ):
        pass
