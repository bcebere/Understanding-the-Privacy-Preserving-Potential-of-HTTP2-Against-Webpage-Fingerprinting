# stdlib
import csv
import os
from pathlib import Path

# third party
import numpy as np

# wfaudit absolute
from wfaudit.helpers_ml import (
    _evaluate_by_domain,
    generate_score,
    load_from_file,
    print_score,
    save_to_file,
)
import wfaudit.logger as log


def _load_data(
    data_dir: str,
    extension=".features",
    delimiter=" ",
    split_at="-",
    max_classes=99999,
    min_instances=10,
    max_instances=500,
    pack_dataset=False,
):
    """
    Load feature files from a directory.

    Parameters
    ----------
    extension : str
        File extension used to identify feature files.
    delimiter : str
        Character string used to split features in the feature files.
    split_at : str
        Character string used to split feature file names.
        First substring identifies the class, while the second substring identifies the instance number.
        Instance number is ignored.
    max_classes : int
        Maximum number of classes to load.
    max_instances : int
        Minimum number of instances acceptable per class.
        If a class has less than this number of instances, the all instances of the class are discarded.
    max_instances : int
        Maximum number of instances to load per class.
    pack_dataset : bool
        Determines whether or not ascii feature files should be packed into a condensed pickle file.
        If True, the function will attempt to load from the packed feature file as well.
        The packed feature file is saved in the root of the same directory as the feature files.

    Returns
    -------
    ndarray
        Numpy array of Nxf containing site visit feature instances.
    ndarray
        Numpy array of Nx1 containing the labels for site visits.

    """
    X = []  # feature instances
    Y = []  # site labels
    for root, dirs, files in os.walk(data_dir):

        # filter for feature files
        files = [fi for fi in files if fi.endswith(extension)]

        def isfloat(element):
            """
            Simple function to reliably determine if a string element is a float.
            Used for feature file filtering.
            """
            try:
                float(element)
                return True
            except ValueError:
                return False

        # read each feature file as CSV
        class_counter = dict()  # track number of instances per class
        for file in files:

            # feature files are of name
            cls, ins = file.split(split_at)
            cls = int(cls)

            # skip if maximum number of instances reached
            if class_counter.get(int(cls), 0) >= max_instances:
                continue

            # skip if maximum number of classes reached
            if int(cls) >= max_classes:
                continue

            with open(os.path.join(root, file), "r") as csvFile:

                # load the csv file and parse it into a data instance
                features = list(csv.reader(csvFile, delimiter=delimiter))
                features = [
                    [float(f) if isfloat(f) else 0 for f in instance if f]
                    for instance in features
                ]

                # cut off instance count is above the maximum
                features = features[: max_instances - class_counter.get(int(cls), 0)]

                X.extend(features)
                Y.extend([int(cls) - 1 for _ in range(len(features))])
                class_counter[int(cls)] = class_counter.get(int(cls), 0) + len(features)

        # trim data to minimum instance count
        counts = {y: Y.count(y) for y in set(Y)}
        new_X, new_Y = [], []
        for x, y in zip(X, Y):
            if counts[y] >= min_instances:
                new_Y.append(y)
                new_X.append(x)
        X, Y = new_X, new_Y

        # adjust labels such that they are assigned a number from 0..N
        # (required when labels are non-numerical or does not start at 0)
        # try to keep the class numbers the same if numerical
        labels = list(set(Y))
        labels.sort()
        d = dict()
        for i in range(len(labels)):
            d[labels[i]] = i
        Y = list(map(lambda x: d[x], Y))

    # return X and Y as numpy arrays
    return np.asarray(X), np.asarray(Y)


def evaluate_ml(
    workspace=Path("output_ml"),
    data_dir=Path("output_features"),
):
    workspace.mkdir(parents=True, exist_ok=True)

    metric_key = "f1_score_macro"

    for sample_limit in [None]:
        for arch in ["xgboost"]:
            if sample_limit is None:
                bkp_file = workspace / f"eval_ts_full_{arch}_{metric_key}.json"
            else:
                bkp_file = (
                    workspace
                    / f"eval_ts_full_{arch}_{metric_key}_samplelimit{sample_limit}.json"
                )
            if bkp_file.exists():
                scores = load_from_file(bkp_file)
                if len(scores) == 0:
                    bkp_file.unlink()
                    continue

            if not bkp_file.exists():
                if sample_limit is None:
                    X, y = _load_data(data_dir)
                    scores = _evaluate_by_domain(
                        arch,
                        "full_data",
                        X,
                        y,
                        metric_key=metric_key,
                        workspace=workspace,
                    )
                else:
                    X, y = _load_data(data_dir, max_instances=sample_limit)
                    scores = _evaluate_by_domain(
                        arch,
                        f"full_data_samplelim{sample_limit}",
                        X,
                        y,
                        metric_key=metric_key,
                        workspace=workspace,
                    )
                if len(scores) == 0:
                    continue
                save_to_file(bkp_file, scores)
            else:
                scores = load_from_file(bkp_file)

            final_score = generate_score(scores)
            log.info(f"[ML perf] arch = {arch}, score={print_score(final_score)}")
