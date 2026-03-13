# stdlib
import logging

# third party
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# wfaudit absolute
from wfaudit.helpers_deepse.datasets.dataset import DefaultDataset


def load_data(
    data_path: str,
    feature_length: int,
    n_traces: int = None,
    n_websites: int = None,
):
    """Load the Dataset.

    Args:
        data_path: Path to the .npz file containing the traces and labels

    Returns:
        x: Matrix (n_traces*n_classes x feature_length) containing the traces
        y: Array (n_traces*n_classes) containing the labels
    """
    # Load data
    logging.info("Loading data...")
    data = np.load(data_path)
    x = data["traces"]
    y = data["labels"]

    # Convert data as float32 type
    x = x.astype("float32")
    assert len(x.shape) == 3
    x = x[:, :, :feature_length]

    y = y.astype("int64")

    # reduce dataset if necessary
    if (
        n_traces is not None
        and n_websites is not None
        and n_traces * n_websites < len(y)
    ):
        x, _, y, _ = train_test_split(
            x,
            y,
            train_size=n_traces * n_websites,
            stratify=y,
            random_state=42,
        )
    logging.info("\tdone.")

    return x, y


def get_split(x, y, train_idx, test_idx):
    """Get the validation splits of x and y.

    Args:
        x: traces matrix
        y: label array
        train_idx: index values for train data
        test_idx: index values for test data

    Returns:
        data: Dictionary containing train, test1 and test2
              data where a new axis is added for the traces
    """
    # get correct split of data
    x_train = x[train_idx].astype("float32")
    x_test = x[test_idx].astype("float32")

    y_train = y[train_idx].astype("int64")
    y_test = y[test_idx].astype("int64")

    # split test into tes1 and test2
    x_test1, x_test2, y_test1, y_test2 = train_test_split(
        x_test, y_test, test_size=0.5, stratify=y_test, shuffle=True, random_state=42
    )

    # we need a [Length x 1] x n shape as input to the CNN (Tensorflow)
    # x_train = x_train[:, np.newaxis, :].astype("float32")
    # x_test1 = x_test1[:, np.newaxis, :].astype("float32")
    # x_test2 = x_test2[:, np.newaxis, :].astype("float32")

    data = {
        "x_train": x_train,
        "x_test1": x_test1,
        "x_test2": x_test2,
        "y_train": y_train,
        "y_test1": y_test1,
        "y_test2": y_test2,
    }

    logging.debug(f"Train shape: {x_train.shape}")
    logging.debug(f"Test1 shape: {x_test1.shape}")
    logging.debug(f"Test2 shape: {x_test2.shape}")

    return data


def get_dataloader(
    traces,
    labels,
    is_training: bool,
    batch_size: int = 200,
    num_workers: int = 4,
):
    """Get the dataloader for the given data.

    Args:
        traces: Traces matrix
        labels: Label array
        is_training: True if the model is for training
        model: str. The model used for creating embeddings
        batch_size: int. Training batch size
        num_workers: int. Number of workers for loading the data


    Returns:
        dataloader: The dataloader
    """

    dataset = DefaultDataset(data=traces, labels=labels)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=0,
        pin_memory=True,
    )
