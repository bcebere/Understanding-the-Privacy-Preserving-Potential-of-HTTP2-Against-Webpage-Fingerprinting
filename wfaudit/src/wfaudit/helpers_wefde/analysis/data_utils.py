# Adapted from https://github.com/notem/reWeFDE

# stdlib
import csv
import json
import os
from pathlib import Path

# third party
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# wfaudit absolute
import wfaudit.logger as log


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

    # ✅ Sanity checks
    assert len(shuffled_y) == len(y)
    # assert np.all(np.bincount(shuffled_y) == np.bincount(y))
    print("Label correlation:", np.corrcoef(y, shuffled_y)[0, 1])

    return shuffled_y


class WebsiteData(object):
    """
    Object-wrapper to conveniently manage dataset
    """

    def __init__(
        self,
        directory,
        debug_correctness: bool = False,
        dataset_split: bool = False,
        max_instances: int = 1000,
    ):
        features_range_path = Path(directory) / "FeaturePositions.json"
        if not features_range_path.exists():
            raise RuntimeError("Missing feature ranges")

        with open(features_range_path, "r") as f:
            self._features_range = json.load(f)

        X, Y = load_wefde_features(directory, max_instances=max_instances)
        print("X ", directory, max_instances, X.shape)

        if debug_correctness:  # Sanity checks for checking the WefDe correctnes
            print(
                "DANGER !!! Running WefDE with shuffled labels. ---> Must return 0 bits leakage !!!"
            )

            # Corrupt Y
            Y = interleaved_label_shuffle(Y)

        if dataset_split:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, Y, test_size=0.3, stratify=Y, random_state=42
            )

            self._X = X_tr
            self._Y = y_tr

            self._X_test = X_te
            self._Y_test = y_te
            print("Train-test split", len(self._X), len(self._X_test))
        else:
            self._X = X
            self._Y = Y

            self._X_test = None
            self._Y_test = None
            print("Full data split", len(self._X))

        self.features = list(range(self._X.shape[1]))
        self.sites = list(range(len(np.unique(self._Y))))
        log.info(f"total samples = {len(self._X)} unique labels = {len(self.sites)}")

    def __len__(self):
        return self._X.shape[0]

    def get_test_data(self):
        return self._X_test, self._Y_test

    def get_labels(self):
        """
        Return Y

        Returns
        -------
        ndarray

        """
        return self._Y

    def get_site(self, label, feature=None):
        """
        Return X for given site.

        Parameters
        ----------
        label : int
            The site label to load
        feature : int
            The feature number to load.
            Load all features if None.

        Returns
        -------
        ndarray

        """
        f = [True if y == label else False for y in self._Y]
        if feature is not None:
            return self._X[f, feature]
        return self._X[f, :]

    def get_feature(self, feature, site=None):
        """
        Return all X for a specific feature

        Parameters
        ----------
        feature : int
            The feature which to load.
        site : int
            The site which to load.
            Load from all sites if None.

        Returns
        -------
        ndarray

        """
        if site is not None:
            f = [True if y == site else False for y in self._Y]
            return self._X[f, feature]
        return self._X[:, feature]


def load_wefde_features(
    directory,
    extension=".features",
    delimiter=" ",
    split_at="-",
    max_classes=1000,
    min_instances=100,
    max_instances=1000,
):
    """
    Load feature files from a directory.


    Each file name is expected to look like  ``<class><split_at><instance><extension>``,
    e.g. ``42-007.features``.

    Parameters
    ----------
    directory : str
        System file path to a directory containing feature files.
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
    min_instances : int
        Minimum number of instances acceptable per class.
        If a class has less than this number of instances, the all instances of the class are discarded.
    max_instances : int
        Maximum number of instances to load per class.
    Returns
    -------
    ndarray
        Numpy array of Nxf containing site visit feature instances.
    ndarray
        Numpy array of Nx1 containing the labels for site visits.

    """
    X = []  # feature instances
    Y = []  # site labels

    for root, dirs, files in os.walk(directory):
        # filter for feature files
        files = [fi for fi in files if fi.endswith(extension)]

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
                features = [[float(f) for f in instance if f] for instance in features]

                # cut off instance count is above the maximum
                features = features[: max_instances - class_counter.get(int(cls), 0)]

                X.extend(features)
                Y.extend([int(cls) - 1 for _ in range(len(features))])
                class_counter[int(cls)] = class_counter.get(int(cls), 0) + len(features)

    print("loaded", len(X))
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
    le = LabelEncoder()
    Y = le.fit_transform(Y)

    Y = np.asarray(Y)
    X = np.asarray(X, dtype=float)

    # return X and Y as numpy arrays
    return X, Y
