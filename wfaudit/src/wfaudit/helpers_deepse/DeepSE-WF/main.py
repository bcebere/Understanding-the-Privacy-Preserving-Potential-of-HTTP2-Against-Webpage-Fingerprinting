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
from datasets.data_utils import get_split, load_data
from models.model_utils import train_models
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold as CV
from tabulate import tabulate
import torch
from utils.knn import compute_distance, knn_ber, knn_mi
from utils.utils import get_args_parser


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


def estimate_security(data, embeddings):
    """
    Returns tight DeepSE BER bounds
    """
    # test1 --> test2
    d12 = compute_distance(embeddings["test1"], embeddings["test2"], args.knn_measure)
    lb12, ub12 = knn_ber(d12, data["y_test1"], data["y_test2"], args.ber_k)
    mi12 = knn_mi(d12, data["y_test1"], data["y_test2"], args.mi_k)

    # test2 --> test1
    d21 = compute_distance(embeddings["test2"], embeddings["test1"], args.knn_measure)
    lb21, ub21 = knn_ber(d21, data["y_test2"], data["y_test1"], args.ber_k)
    mi21 = knn_mi(d21, data["y_test2"], data["y_test1"], args.mi_k)

    # Per-representation bounds on error: strongest LB and UB across directions
    lb_err = max(lb12, lb21)  # tightest lower bound on Bayes error
    ub_err = min(ub12, ub21)  # tightest upper bound on Bayes error
    mi = 0.5 * (mi12 + mi21)

    return dict(lb_err=lb_err, ub_err=ub_err, mi=mi)


def main(args):
    total_start = timer()
    x, y = load_data(args.data_path, args)

    cv = CV(n_splits=args.k_fold, shuffle=True, random_state=42)

    workspace = Path(args.results_file).parent

    results = {"acc": [], "ber_lo": [], "ber_hi": [], "mi": []}
    for cv_count, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
        start_cv = timer()
        logging.info(
            f"------------------- CV RUN {cv_count} OF {args.k_fold} -------------------"
        )
        data = get_split(x, y, train_idx, test_idx)

        bkp_path = workspace / f"cache_{cv_count}.bkp"
        if bkp_path.exists():
            logging.info(f"Loading cached embeddings {bkp_path}")
            embeddings, history = load_from_file(bkp_path)
        else:
            embeddings, history = train_models(data=data, args=args)
            logging.info(f"Caching {bkp_path}")
            save_to_file(bkp_path, (embeddings, history))

        results["acc"].append(history["test_acc"][-1])

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
        estimates = estimate_security(data, embeddings)
        results["ber_lo"].append(estimates["lb_err"])
        results["ber_hi"].append(estimates["ub_err"])
        results["mi"].append(estimates["mi"])

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

    total_end = timer()

    logging.info(f"Total time: {timedelta(seconds=total_end-total_start)}")

    df = pd.DataFrame.from_dict(results)
    # DataFrame(
    #    [(k, np.mean(v), np.std(v)) for k, v in results.items()],
    #    columns=["Value", "Mean", "Std"],
    # )
    df.to_csv(args.results_file, index=None)

    table = [[k, np.mean(v), np.std(v)] for k, v in results.items()]
    table = tabulate(
        table,
        headers=["Value", "Mean", "Std"],
        tablefmt="github",
        floatfmt=".4f",
    )
    logging.info(f"Results:\n\n{table}\n")


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
