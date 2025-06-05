# stdlib
import copy
from pathlib import Path
import random
from typing import Any, Dict, List, Tuple, Union

# third party
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# wfaudit absolute
from wfaudit.helpers_ml.holmes import HolmesClassifier
from wfaudit.helpers_ml.kfp import kFingerprinting
from wfaudit.helpers_ml.kfpv2 import KFingerprintingForestClassifier
from wfaudit.helpers_ml.lr import LinearClassifier
from wfaudit.helpers_ml.rf import RFClassifier
from wfaudit.helpers_ml.robustfp import RobustFingerprintingClassifier
from wfaudit.helpers_ml.serialization import load_from_file, save_to_file
from wfaudit.helpers_ml.svm import SVMClassifier
from wfaudit.helpers_ml.varcnn import VarCNNClassifier
from wfaudit.helpers_ml.varcnn3d import VarCNN3DClassifier
from wfaudit.helpers_ml.xgb import XGBoostClassifier
import wfaudit.logger as log

clf_supported_metrics = [
    "f1_score_macro",
    "precision_macro",
    "recall_macro",
    "mcc",
]


class classifier_metrics:
    """Helper class for evaluating the performance of the classifier.

    Args:
        metric: list, default=["f1_score_macro", "precision_macro", "recall_macro",  "mcc",]
            The type of metric to use for evaluation.
            Potential values:
                - "f1_score_macro": F1 score is a harmonic mean of the precision and recall. This version uses the "macro" average: calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
                - "precision_macro": Precision is defined as the number of true positives over the number of true positives plus the number of false positives. This version(macro) calculates metrics for each label, and finds their unweighted mean.
                - "recall_macro": Recall is defined as the number of true positives over the number of true positives plus the number of false negatives. This version(macro) calculates metrics for each label, and finds their unweighted mean.
                - "mcc": The Matthews correlation coefficient is used in machine learning as a measure of the quality of binary and multiclass classifications. It takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes.
    """

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

        for metric in self.metrics:
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
            else:
                raise ValueError(f"invalid metric {metric}")

        # log.debug(f"evaluate_classifier: {results}")
        return results


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

    if classes is None:
        classes = np.ravel(Y)
    classes = set(classes)

    enable_reproducible_results(seed)

    X = np.asarray(X)
    Y = LabelEncoder().fit_transform(Y)
    Y = np.asarray(Y)
    # log.debug(f"evaluate_estimator shape x:{X.shape} y:{Y.shape}")

    results = {}

    evaluator = classifier_metrics()
    for metric in clf_supported_metrics:
        results[metric] = np.zeros(n_folds)

    indx = 0
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for train_index, test_index in skf.split(X, Y):
        X_train = X[train_index]
        Y_train = Y[train_index]
        X_test = X[test_index]
        Y_test = Y[test_index]

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
        return XGBoostClassifier()
    elif arch == "svm":
        return SVMClassifier()
    elif arch == "lr":
        return LinearClassifier()
    elif arch == "rf":
        return RFClassifier()
    elif arch == "kfp":
        return kFingerprinting()
    elif arch == "kfpv2":
        return KFingerprintingForestClassifier()
    elif arch == "varcnn":
        return VarCNNClassifier(**kwargs)
    elif arch == "varcnn3d":
        return VarCNN3DClassifier(**kwargs)
    elif arch == "tam":
        return RobustFingerprintingClassifier(**kwargs)
    elif arch == "holmes":
        return HolmesClassifier(**kwargs)
    else:
        raise RuntimeError(arch)


def _dataframe_hash(df: pd.DataFrame) -> str:
    cols = sorted(list(df.columns))
    return str(abs(pd.util.hash_pandas_object(df[cols].fillna(0)).sum()))


def _evaluate_static_models_cv(
    arch: str,  # = "xgboost"
    testname: str,
    input_data: List,
    labels: pd.Series,
    workspace=Path("workspace"),
    use_cache: bool = True,
    **kwargs,
):
    hash_data = pd.DataFrame(np.asarray(input_data).reshape(len(input_data), -1)).copy()
    hash_data["label"] = labels
    hash_data.columns = hash_data.columns.astype(str)
    data_hash = _dataframe_hash(hash_data)
    bkp_file = workspace / f"eval_{data_hash}_{arch}_{testname}.json"

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
    log.error(
        f" >>> test = {testname}, datalen = {len(input_data)} score = {score['str']['f1_score_macro']}"
    )
    return score


def evaluate_by_domain(
    arch: str,  # = "xgboost"
    label: str,
    data: np.ndarray,
    labels: np.ndarray,
    metric_key: str = "f1_score_macro",
    workspace=Path("workspace"),
    filtered_labels=None,
    use_cache: bool = True,
    **kwargs,
):
    scores = []
    scores_by_domain = {}
    if filtered_labels is None:
        filtered_labels = np.unique(labels)
    # print("evaluating filtered labels", filtered_labels)

    for domain in tqdm(filtered_labels):
        horizon_labels = pd.Series(labels).copy()
        horizon_labels[horizon_labels != domain] = -1
        horizon_labels[horizon_labels == domain] = 1
        horizon_labels[horizon_labels == -1] = 0

        score = _evaluate_static_models_cv(
            arch,
            f"{label}_{domain}",
            data,
            horizon_labels,
            workspace=workspace,
            use_cache=use_cache,
            **kwargs,
        )
        if score is None:
            continue
        scores.append(score["raw"][metric_key][0])
        scores_by_domain[int(domain)] = float(score["raw"][metric_key][0])
        log.debug(
            f" >>> [{arch}][{domain}] {metric_key} = {scores_by_domain[int(domain)]}"
        )

    return scores, scores_by_domain


def evaluate_multiclass(
    arch: str,  # = "xgboost"
    label: str,
    data: np.ndarray,
    labels: np.ndarray,
    metric_key: str = "f1_score_macro",
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
