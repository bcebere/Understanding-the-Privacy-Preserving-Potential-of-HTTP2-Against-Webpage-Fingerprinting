# stdlib
import copy
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple, Union

# third party
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# wfaudit absolute
from wfaudit.helpers_ml.kfp import kFingerprinting
from wfaudit.helpers_ml.lr import LinearClassifier
from wfaudit.helpers_ml.rf import RFClassifier
from wfaudit.helpers_ml.serialization import load_from_file, save_to_file
from wfaudit.helpers_ml.svm import SVMClassifier
from wfaudit.helpers_ml.varcnn import VarCNNClassifier
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

        log.debug(f"evaluate_classifier: {results}")
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
    X: Union[pd.DataFrame, np.ndarray],
    Y: Union[pd.Series, np.ndarray, List],
    n_folds: int = 3,
    seed: int = 0,
    pretrained: bool = False,
    group_ids: Optional[pd.Series] = None,
    classes: Any = None,
) -> Dict:
    """Helper for evaluating classifiers.

    Args:
        estimator:
            Baseline model to evaluate. if pretrained == False, it must not be fitted.
        X: pd.DataFrame or np.ndarray:
            The covariates
        Y: pd.Series or np.ndarray or list:
            The labels
        n_folds: int
            cross-validation folds
        seed: int
            Random seed
        pretrained: bool
            If the estimator was already trained or not.
        group_ids: pd.Series
            The group_ids to use for stratified cross-validation

    Returns:
        Dict containing "raw" and "str" nodes. The "str" node contains prettified metrics, while the raw metrics includes tuples of form (`mean`, `std`) for each metric.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")

    if classes is None:
        classes = np.ravel(Y)
    classes = set(classes)

    enable_reproducible_results(seed)

    X = pd.DataFrame(X).reset_index(drop=True)
    Y = LabelEncoder().fit_transform(Y)
    Y = pd.Series(Y).reset_index(drop=True)
    if group_ids is not None:
        group_ids = pd.Series(group_ids).reset_index(drop=True)

    log.debug(f"evaluate_estimator shape x:{X.shape} y:{Y.shape}")

    results = {}

    evaluator = classifier_metrics()
    for metric in clf_supported_metrics:
        results[metric] = np.zeros(n_folds)

    indx = 0
    if group_ids is not None:
        skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # group_ids is always ignored for StratifiedKFold so safe to pass None
    for train_index, test_index in skf.split(X, Y, groups=group_ids):
        X_train = X.loc[X.index[train_index]]
        Y_train = Y.loc[Y.index[train_index]]
        X_test = X.loc[X.index[test_index]]
        Y_test = Y.loc[Y.index[test_index]]

        if pretrained:
            model = estimator[indx]
        else:
            model = copy.deepcopy(estimator)
            model.fit(X_train, Y_train)

        preds = model.predict_proba(X_test)
        pred_labels = set(np.ravel(model.predict(X_test)))
        classes = classes.union(pred_labels)

        scores = evaluator.score_proba(Y_test, preds, list(sorted(classes)))

        for metric in scores:
            if "fpr" not in metric and "tpr" not in metric:
                results[metric][indx] = scores[metric]
            else:
                results[metric] = scores[metric]

        indx += 1

    output_clf = {}
    output_clf_str = {}

    for key in results:
        if "fpr" not in key and "tpr" not in key:
            key_out = generate_score(results[key])
            output_clf[key] = key_out
            output_clf_str[key] = print_score(key_out)
        else:
            output_clf[key] = results[key]

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
    elif arch == "varcnn":
        return VarCNNClassifier(**kwargs)
    else:
        raise RuntimeError(arch)


def _evaluate_static_models_cv(
    arch: str,  # = "xgboost"
    testname: str,
    input_data: List,
    labels: pd.Series,
    workspace=Path("workspace"),
    **kwargs,
):
    bkp_file = workspace / f"eval_ts_flow_ho_{len(input_data)}_{arch}_{testname}.json"

    score = None
    if bkp_file.exists():
        score = load_from_file(bkp_file)
    else:
        model = _get_arch_mode(
            arch,
            **kwargs,
        )
        score = evaluate_classifier(
            model, X=np.asarray(input_data), Y=np.asarray(labels)
        )
        save_to_file(bkp_file, score)
    # log.error(f" >>> test = {testname}, score = {score['str']['f1_score_macro']}")
    return score


def evaluate_by_domain(
    arch: str,  # = "xgboost"
    label: str,
    data: np.ndarray,
    labels: np.ndarray,
    metric_key: str = "f1_score_macro",
    workspace=Path("workspace"),
    filtered_labels=None,
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
            **kwargs,
        )
        if score is None:
            continue
        scores.append(score["raw"][metric_key][0])
        scores_by_domain[int(domain)] = float(score["raw"][metric_key][0])

    return scores, scores_by_domain
