"""DeepSE-WF BER and MI estimation."""

# stdlib
import argparse
from datetime import timedelta
import logging
import os
from pathlib import Path
import sys
from timeit import default_timer as timer
from typing import Any, Union

# third party
import cloudpickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold as CV
from tabulate import tabulate
import torch

# wfaudit absolute
from wfaudit.helpers_deepse.datasets.data_utils import get_split, load_data
from wfaudit.helpers_deepse.models.model_utils import train_models
from wfaudit.helpers_deepse.utils.knn import compute_distance, knn_ber, knn_mi
from wfaudit.helpers_deepse.utils.utils import get_args_parser


def save_to_file(path: Union[str, Path], model: Any) -> Any:
    path = Path(path)
    ppath = path.absolute().parent

    if not ppath.exists():
        ppath.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        return cloudpickle.dump(model, f)


def load_from_file(path: Union[str, Path]) -> Any:
    with open(path, "rb") as f:
        return cloudpickle.load(f)


LOG_LVL = logging.INFO


def cache_fits(embeddings, data, embedding_size):
    """A cached (embeddings, history) pair is only usable if it was produced by
    the same split and the same embedding size as the current run."""
    for half in ("test1", "test2"):
        emb = np.asarray(embeddings[half])
        if len(emb) != len(data[f"y_{half}"]):
            return False
        if emb.ndim > 1 and emb.shape[1] != embedding_size:
            return False
    return True


def estimate_security(
    data,
    embeddings,
    knn_measure: str = "squared_l2",
    ber_k: int = 1,
    mi_k: int = 5,
):
    """
    Returns tight DeepSE BER bounds
    """
    # test1 --> test2
    d12 = compute_distance(embeddings["test1"], embeddings["test2"], knn_measure)
    lb12, ub12 = knn_ber(d12, data["y_test1"], data["y_test2"], ber_k)
    mi12 = knn_mi(d12, data["y_test1"], data["y_test2"], mi_k)

    # test2 --> test1
    d21 = compute_distance(embeddings["test2"], embeddings["test1"], knn_measure)
    lb21, ub21 = knn_ber(d21, data["y_test2"], data["y_test1"], ber_k)
    mi21 = knn_mi(d21, data["y_test2"], data["y_test1"], mi_k)

    # Per-representation bounds on error: strongest LB and UB across directions
    lb_err = max(lb12, lb21)  # tightest lower bound on Bayes error
    ub_err = min(ub12, ub21)  # tightest upper bound on Bayes error
    mi = 0.5 * (mi12 + mi21)

    return dict(lb_err=lb_err, ub_err=ub_err, mi=mi)


def estimate_mi_ber(
    data_path: str,
    results_file: str,
    n_websites: int = None,
    n_traces: int = None,
    feature_length: int = 5000,
    k_fold: int = 5,
    random_state: int = 42,
    embedding_size: int = 512,
    model: str = "df",
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    epochs: int = 100,
    batch_size: int = 200,
    num_workers: int = 4,
    knn_measure: str = "squared_l2",
    ber_k: int = 1,
    mi_k: int = 5,
):
    x, y = load_data(
        data_path=data_path,
        feature_length=feature_length,
        n_traces=n_traces,
        n_websites=n_websites,
    )

    if n_websites is None:
        n_websites = len(np.unique(y))
    if n_traces is None:
        n_traces = int(len(x) / n_websites)

    cv = CV(n_splits=k_fold, shuffle=True, random_state=random_state)

    workspace = Path(results_file).parent

    results = {"ACC": [], "BER_LO": [], "BER_HI": [], "MI_TOTAL": []}
    for cv_count, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
        start_cv = timer()
        logging.info(
            f"------------------- CV RUN {cv_count} OF {k_fold} -------------------"
        )
        data = get_split(x, y, train_idx, test_idx)

        # The cache key carries everything that determines the contents, so
        # runs with different k_fold (or model / embedding size / seed) no
        # longer overwrite or silently reuse each other's embeddings.
        bkp_path = workspace / (
            f"cache_{model}_k{k_fold}_e{embedding_size}"
            f"_r{random_state}_{cv_count}.bkp"
        )
        # caches written before the key was parameterised
        legacy_path = workspace / f"cache_{cv_count}.bkp"

        embeddings, history = None, None
        for path in (bkp_path, legacy_path):
            if not path.exists():
                continue
            cached, cached_history = load_from_file(path)
            if cache_fits(cached, data, embedding_size):
                logging.info(f"Loading cached embeddings {path}")
                embeddings, history = cached, cached_history
                break
            logging.info(
                f"Ignoring {path}: {len(cached['test1'])} cached embeddings vs "
                f"{len(data['y_test1'])} labels in this split"
            )

        if embeddings is None:
            embeddings, history = train_models(
                data=data,
                n_websites=n_websites,
                embedding_size=embedding_size,
                model_name=model,
                epochs=epochs,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            logging.info(f"Caching {bkp_path}")
            save_to_file(bkp_path, (embeddings, history))

        results["ACC"].append(history["test_acc"][-1])

        table = [
            [
                history["train_acc"][-1],
                history["val_acc"][-1],
                history["test_acc"][-1],
            ]
        ]
        table = tabulate(
            table,
            headers=["Train Acc", "Val Acc", "Test Acc"],
            tablefmt="github",
            floatfmt=".4f",
        )
        logging.info(f"Model performance:\n\n{table}\n")

        logging.info("Estimate Security:")
        table = []
        start = timer()
        estimates = estimate_security(
            data,
            embeddings,
            knn_measure=knn_measure,
            ber_k=ber_k,
            mi_k=mi_k,
        )
        results["BER_LO"].append(estimates["lb_err"])
        results["BER_HI"].append(estimates["ub_err"])
        results["MI_TOTAL"].append(estimates["mi"])

        table.append([estimates["lb_err"], estimates["ub_err"], estimates["mi"]])

        end = timer()
        logging.info(f"Done after {timedelta(seconds=end - start)}")

        table = tabulate(
            table,
            headers=["BER LO", "BER HI", "MI"],
            tablefmt="github",
            floatfmt=".4f",
        )
        logging.info(f"Results for CV {cv_count}:\n\n{table}\n")

        end_cv = timer()

        logging.info(f"Time CV {cv_count}: {timedelta(seconds=end_cv-start_cv)}\n")

    df = pd.DataFrame.from_dict(results)
    df.to_csv(results_file, index=None)

    return df


def main(args):
    results = estimate_mi_ber(
        data_path=args.data_path,
        results_file=args.results_file,
        n_websites=args.n_websites,
        n_traces=args.n_traces,
        feature_length=args.feature_length,
        k_fold=args.k_fold,
        embedding_size=args.embedding_size,
        model=args.model,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        knn_measure=args.knn_measure,
        ber_k=args.ber_k,
        mi_k=args.mi_k,
    )

    logging.info(f"Results:\n\n{results}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DeepSE-WF",
        parents=[get_args_parser()],
    )
    args = parser.parse_args()

    if args.log_file:
        logging.basicConfig(
            filename=args.log_file,
            level=LOG_LVL,
            format="%(asctime)s %(levelname)s %(message)s",
        )
    else:
        logging.basicConfig(
            stream=sys.stdout,
            level=LOG_LVL,
            format="%(asctime)s %(levelname)s %(message)s",
        )

    if args.gpu_id is not None:
        gpuid = args.gpu_id
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ[
            "CUDA_VISIBLE_DEVICES"
        ] = f"{gpuid}"  # select ID of GPU that shall be used

    if args.device == "cuda":
        logging.info(f"Using GPU: {gpuid} ({torch.cuda.is_available()})")

    hyperparams = tabulate(
        [
            [
                args.model,
                args.data_path.split("/")[-1],
                args.n_traces,
                args.epochs,
                args.dropout,
                args.embedding_size,
                args.knn_measure,
                args.ber_k,
                args.mi_k,
                args.device,
            ]
        ],
        headers=[
            "model",
            "data",
            "n_traces",
            "epochs",
            "dropout",
            "embedding_size",
            "measure",
            "ber_k",
            "mi_k",
            "device",
        ],
        tablefmt="github",
        floatfmt=".4f",
    )
    logging.info(f"Hyperparameters:\n\n{hyperparams}\n")

    main(args)
