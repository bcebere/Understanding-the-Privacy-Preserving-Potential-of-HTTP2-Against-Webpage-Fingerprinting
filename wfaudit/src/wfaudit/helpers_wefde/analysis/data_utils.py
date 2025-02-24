# Adapted from https://github.com/notem/reWeFDE

# stdlib
import csv
import os

# third party
import numpy as np

# wfaudit absolute
import wfaudit.logger as log


class WebsiteData_v2(object):
    """
    Object-wrapper to conveniently manage dataset
    """

    def __init__(self, X, y, **kwargs):
        self._X, self._Y = X, y
        self.features = list(range(self._X.shape[1]))
        self.sites = list(range(len(np.unique(self._Y))))
        log.info(f"total samples = {len(self._X)} unique labels = {len(self.sites)}")

    def __len__(self):
        return self._X.shape[0]

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


class WebsiteData(object):
    """
    Object-wrapper to conveniently manage dataset
    """

    def __init__(self, directory, **kwargs):
        self._X, self._Y = load_wefde_features(directory, **kwargs)
        self.features = list(range(self._X.shape[1]))
        self.sites = list(range(len(np.unique(self._Y))))
        log.info(f"total samples = {len(self._X)} unique labels = {len(self.sites)}")

    def __len__(self):
        return self._X.shape[0]

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
    max_classes=99999,
    min_instances=10,
    max_instances=500,
):
    """
    Load feature files from a directory.

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
    max_instances : int
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
