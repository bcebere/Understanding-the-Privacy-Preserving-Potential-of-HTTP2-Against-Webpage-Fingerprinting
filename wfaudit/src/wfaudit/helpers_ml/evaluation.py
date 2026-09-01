# stdlib
import copy
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Tuple, Union

# third party
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, label_binarize

# wfaudit absolute
from wfaudit.helpers_ml.df import DFClassifier
from wfaudit.helpers_ml.holmes import HolmesClassifier
from wfaudit.helpers_ml.kfpv2 import KFingerprintingForestClassifier
from wfaudit.helpers_ml.lr import LinearClassifier
from wfaudit.helpers_ml.rf import RFClassifier
from wfaudit.helpers_ml.robustfp import RobustFingerprintingClassifier
from wfaudit.helpers_ml.serialization import load_from_file, save_to_file
from wfaudit.helpers_ml.varcnn import VarCNNClassifier
from wfaudit.helpers_ml.xgb import XGBoostClassifier
import wfaudit.logger as log

clf_extras = [
    "acc_top5",
    "acc_top10",
    "acc_top20",
    "f1_top5",
    "f1_top10",
    "f1_top20",
    "rank_mean",
    "rank_median",
    "confidence_mean",
    "entropy_mi",
    "entropy_uncert",
]
clf_supported_metrics = [
    "f1_score_macro",
    "precision_macro",
    "recall_macro",
    "mcc",
] + clf_extras


class classifier_metrics:
    def __init__(self, metric: Union[str, list] = clf_supported_metrics) -> None:
        if isinstance(metric, str):
            self.metrics = [metric]
        else:
            self.metrics = metric

    def get_metric(self) -> Union[str, list]:
        return self.metrics

    def score_proba(
        self, y_test: np.ndarray, y_pred_proba: np.ndarray, classes: list
    ) -> Dict[str, float]:
        if y_test is None or y_pred_proba is None:
            raise RuntimeError("Invalid input for score_proba")

        results = {}
        y_pred = np.argmax(np.asarray(y_pred_proba), axis=1)

        if len(classes) > 2:  # multiclass
            # The label space is defined by the score columns, not by the labels
            # present in this fold; a fold missing a class would otherwise raise.
            proba_labels = np.arange(np.asarray(y_pred_proba).shape[1])
            for k in [5, 10, 20]:
                results[f"acc_top{k}"] = top_k_accuracy_score(
                    y_test, y_pred_proba, k=k, labels=proba_labels
                )
                results[f"f1_top{k}"] = self.topk_recall(y_test, y_pred_proba, k=k)
            (
                H_cond,
                MI_bits,
                avg_max_conf,
                mean_rank,
                median_rank,
            ) = self.entropy_metrics(y_test, y_pred_proba)
            results["rank_mean"] = mean_rank
            results["rank_median"] = median_rank
            results["confidence_mean"] = avg_max_conf
            results["entropy_mi"] = MI_bits
            results["entropy_uncert"] = H_cond

        for metric in self.metrics:
            if metric in results:
                continue

            if metric == "f1_score_macro":
                results[metric] = f1_score(
                    y_test, y_pred, average="macro", zero_division=0
                )
            elif metric == "recall_macro":
                results[metric] = recall_score(
                    y_test, y_pred, average="macro", zero_division=0
                )
            elif metric == "precision_macro":
                results[metric] = precision_score(
                    y_test, y_pred, average="macro", zero_division=0
                )
            elif metric == "mcc":
                results[metric] = matthews_corrcoef(y_test, y_pred)
            elif metric in clf_extras:
                continue
            else:
                raise ValueError(f"invalid metric {metric}")

        # log.debug(f"evaluate_classifier: {results}")
        return results

    def topk_recall(self, y_true, y_proba, k: int):
        topk_preds = np.argsort(y_proba, axis=1)[:, -k:]  # top k class indices
        y_pred_top1 = topk_preds[:, -1]  # fallback: top-1 prediction

        # Mask: 1 if true label in top-k preds
        topk_hit_mask = np.array([y in preds for y, preds in zip(y_true, topk_preds)])

        # Create new predicted labels: if in topk, leave as is, else assign wrong label
        y_pred_topk = np.where(topk_hit_mask, y_true, y_pred_top1)

        return f1_score(y_true, y_pred_topk, average="macro")  # or 'micro'/'weighted'

    def topk_f1_score(self, y_true, y_proba, k, average="macro"):
        n, C = y_proba.shape
        topk = np.argpartition(y_proba, -k, axis=1)[:, -k:]
        Y_true = label_binarize(y_true, classes=np.arange(C))
        Y_pred = np.zeros_like(Y_true, dtype=int)
        Y_pred[np.arange(n)[:, None], topk] = 1
        return f1_score(Y_true, Y_pred, average=average, zero_division=0)

    def entropy_metrics(self, y_true, probs):
        # Per-trace prediction entropy  H_i = -Σ p_i log2 p_i  ----------
        eps = 1e-12  # to avoid log(0)
        entropy_per_trace = -(probs * np.log2(probs + eps)).sum(axis=1)  # shape (M,)

        # Average (conditional) entropy  H(Y|X)
        H_cond = entropy_per_trace.mean()

        # Prior entropy  H(Y)  (uniform case)
        N = probs.shape[1]
        H_prior = np.log2(N)

        # Mutual information  I(Y;X) = H(Y) - H(Y|X)
        MI_bits = H_prior - H_cond

        # Confidence metrics --------------------------------------------
        max_confidence = probs.max(axis=1)  # attacker top-1 probability
        avg_max_conf = max_confidence.mean()

        # Rank of the correct class ------------------------------------
        ordered = np.argsort(-probs, axis=1)  # descending order
        true_ranks = np.array(
            [
                np.where(row == y)[0][0] + 1  # +1 rank starts at 1
                for row, y in zip(ordered, y_true)
            ]
        )
        median_rank = np.median(true_ranks)
        mean_rank = true_ranks.mean()

        print(f"H(Y|X)  : {H_cond:.3f} bits (prior={H_prior:.2f})")
        print(f"MI      : {MI_bits:.3f} bits")
        print(f"Avg max confidence : {avg_max_conf:.3f}")
        print(f"Mean / median true-label rank : {mean_rank:.1f} / {median_rank}")
        return H_cond, MI_bits, avg_max_conf, mean_rank, median_rank


def enable_reproducible_results(random_state: int = 0) -> None:
    np.random.seed(random_state)
    random.seed(random_state)


def generate_score(metric: np.ndarray) -> Tuple[float, float]:
    percentile_val = 1.96
    return (np.mean(metric), percentile_val * np.std(metric) / np.sqrt(len(metric)))


def print_score(score: Tuple[float, float]) -> str:
    return str(round(score[0], 3)) + " +/- " + str(round(score[1], 2))


def evaluate_classifier(
    estimator: Any,
    X: np.ndarray,
    Y: np.ndarray,
    n_folds: int = 3,
    seed: int = 0,
    classes: Any = None,
) -> Dict:
    """Helper for evaluating classifiers.

    Args:
        estimator:
            Baseline model to evaluate. if pretrained == False, it must not be fitted.
        X: np.ndarray:
            The covariates
        Y: np.ndarray or list:
            The labels
        n_folds: int
            cross-validation folds
        seed: int
            Random seed

    Returns:
        Dict containing "raw" and "str" nodes. The "str" node contains prettified metrics, while the raw metrics includes tuples of form (`mean`, `std`) for each metric.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")

    X = np.asarray(X)

    Y = LabelEncoder().fit_transform(Y)
    Y = np.asarray(Y)

    if classes is None:
        classes = np.ravel(Y)
    classes = set(classes)

    results = {}

    evaluator = classifier_metrics()
    for metric in clf_supported_metrics:
        results[metric] = np.zeros(n_folds)

    indx = 0
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for train_index, test_index in skf.split(X, Y):
        Y_train = Y[train_index]
        Y_test = Y[test_index]

        # Per-fold normalization for 3D sequence data (VarCNN, DF).
        # Stats computed on train fold only to prevent leakage into test.
        if X.ndim == 3:  # (N, channels, length)
            mu = X[train_index].mean(axis=(0, 2), keepdims=True)
            sigma = X[train_index].std(axis=(0, 2), keepdims=True) + 1e-8
            X_train = (X[train_index] - mu) / sigma
            X_test = (X[test_index] - mu) / sigma
        else:
            X_train = X[train_index]
            X_test = X[test_index]

        model = copy.deepcopy(estimator)
        model.fit(X_train, Y_train)

        preds = model.predict_proba(X_test)
        pred_labels = set(np.ravel(model.predict(X_test)))
        classes = classes.union(pred_labels)

        scores = evaluator.score_proba(Y_test, preds, list(sorted(classes)))
        del model

        for metric in scores:
            results[metric][indx] = scores[metric]
        indx += 1

    output_clf = {}
    output_clf_str = {}

    for key in results:
        key_out = generate_score(results[key])
        output_clf[key] = key_out
        output_clf_str[key] = print_score(key_out)

    return {
        "raw": output_clf,
        "str": output_clf_str,
    }


def _get_arch_mode(arch: str, **kwargs):
    if arch == "xgboost":
        return XGBoostClassifier(**kwargs)
    elif arch == "lr":
        return LinearClassifier(**kwargs)
    elif arch == "rf":
        return RFClassifier(**kwargs)
    elif arch == "kfp":
        return KFingerprintingForestClassifier(**kwargs)
    elif arch == "varcnn":
        return VarCNNClassifier(**kwargs)
    elif arch == "holmes":
        return HolmesClassifier(**kwargs)
    elif arch == "robustfp":
        return RobustFingerprintingClassifier(**kwargs)
    elif arch == "df":
        return DFClassifier(**kwargs)
    else:
        raise RuntimeError(arch)


def _dataframe_hash(df: pd.DataFrame) -> str:
    # Ensure column order is stable
    cols = sorted(df.columns)

    # Normalize index (if you don't care about it)
    df_normalized = df[cols].fillna(0).copy()
    df_normalized.index = range(len(df_normalized))  # reset index

    # Use hash with index=False
    hashes = pd.util.hash_pandas_object(df_normalized, index=False)

    return str(abs(hashes.sum()))


def _array_hash(data: np.ndarray, labels) -> str:
    """Order-dependent content hash, computed without copying the array.

    Used for sequence data, where building a DataFrame of the flattened traces
    costs roughly three times the size of the data itself.
    """
    h = hashlib.blake2b(digest_size=8)
    for part in (np.ascontiguousarray(data), np.ascontiguousarray(labels)):
        h.update(f"{part.shape}|{part.dtype.str}|".encode())
        view = memoryview(part.reshape(-1)).cast("B")
        chunk = 1 << 24
        for i in range(0, len(view), chunk):
            h.update(view[i : i + chunk])
    return h.hexdigest()


def _params_signature(params: Dict[str, Any]) -> str:
    """Stable suffix identifying a hyper-parameter configuration.

    Returns the empty string for an empty configuration, so cache files written
    before hyper-parameters were part of the key remain valid.
    """
    if not params:
        return ""
    unstable = sorted(k for k, v in params.items() if callable(v))
    if unstable:
        # ``str`` of a function embeds its memory address, which would change
        # the cache key on every run and silently disable the cache.
        raise ValueError(
            f"cannot build a stable cache key from callable parameters {unstable}; "
            f"remove them before evaluating"
        )
    blob = json.dumps(params, sort_keys=True, default=str)
    return "_" + hashlib.md5(blob.encode()).hexdigest()[:8]


def _evaluate_static_models_cv(
    arch: str,  # = "xgboost"
    testname: str,
    input_data: List,
    labels: pd.Series,
    workspace=Path("workspace"),
    use_cache: bool = True,
    **kwargs,
):
    array_data = np.asarray(input_data)
    if array_data.ndim == 3:
        # Sequence data is hashed directly; flattening it into a DataFrame first
        # would allocate several times its own size.
        data_hash = _array_hash(array_data, np.asarray(labels))
    else:
        hash_data = pd.DataFrame(array_data.reshape(len(array_data), -1)).copy()
        hash_data["label"] = labels
        hash_data.columns = hash_data.columns.astype(str)
        data_hash = _dataframe_hash(hash_data)
    bkp_file = (
        workspace
        / f"eval_{data_hash}_{arch}_{testname}{_params_signature(kwargs)}.json"
    )

    score = None
    if bkp_file.exists() and use_cache:
        score = load_from_file(bkp_file)
    else:
        model = _get_arch_mode(
            arch,
            **kwargs,
        )
        score = evaluate_classifier(
            model, X=np.asarray(input_data), Y=np.asarray(labels)
        )
        if use_cache:
            save_to_file(bkp_file, score)
    log.info(
        f" >>> test = {testname}, datalen = {len(input_data)} score = {score['str']['f1_score_macro']}"
    )
    return score


def evaluate_multiclass(
    arch: str,  # = "xgboost"
    label: str,
    data: np.ndarray,
    labels: np.ndarray,
    workspace=Path("workspace"),
    use_cache: bool = True,
    **kwargs,
):
    score = _evaluate_static_models_cv(
        arch,
        f"{label}_multiclass",
        data,
        labels,
        workspace=workspace,
        use_cache=use_cache,
        **kwargs,
    )
    log.debug(f" >>> [{arch}][multiclass] Score = {score}")

    return score
