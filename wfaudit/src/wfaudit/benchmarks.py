# stdlib
from pathlib import Path

# third party
import pandas as pd

# wfaudit absolute
from wfaudit.helpers_ml.evaluation import (
    _dataframe_hash,
    evaluate_by_domain,
    generate_score,
    print_score,
)
from wfaudit.helpers_wefde.analysis.data_utils import load_wefde_features
from wfaudit.helpers_wefde.analysis.info_leak import evaluate_info_leakage
import wfaudit.logger as log


def evaluate_ml_from_wefde(
    workspace=Path("output_ml"),
    wefde_features_dir=Path("wefde_features"),
    metric_key="f1_score_macro",
    arch: str = "xgboost",
    filtered_labels=None,
    use_cache: bool = True,
):
    if not wefde_features_dir.exists():
        log.error("WeFDE features not extracted")
        return None

    workspace.mkdir(parents=True, exist_ok=True)

    X, y = load_wefde_features(wefde_features_dir)
    print("X HASH", _dataframe_hash(pd.DataFrame(X)))
    print("Label distribution", pd.Series(y).value_counts())

    return evaluate_ml(
        X,
        y,
        workspace=workspace,
        metric_key=metric_key,
        arch=arch,
        filtered_labels=filtered_labels,
        use_cache=use_cache,
    )


def evaluate_ml(
    X,
    y,
    workspace=Path("output_ml"),
    metric_key="f1_score_macro",
    arch: str = "xgboost",
    filtered_labels=None,
    use_cache: bool = True,
):
    assert len(X) > 0
    assert len(X) == len(y)

    _, scores_by_domain = evaluate_by_domain(
        arch,
        "stats",
        X,
        y,
        metric_key=metric_key,
        workspace=workspace,
        filtered_labels=filtered_labels,
        use_cache=use_cache,
    )

    scores = list(scores_by_domain.values())
    final_score = generate_score(scores)
    log.info(f"[ML perf with stats] arch = {arch}, F1 score={print_score(final_score)}")

    return final_score, scores_by_domain


def evaluate_ml_rawts(
    X,
    y,
    workspace=Path("output_ml"),
    metric_key="f1_score_macro",
    arch: str = "xgboost",
    use_cache: bool = True,
    limit_domains: int = None,
    **kwargs,
):
    workspace.mkdir(parents=True, exist_ok=True)

    _, scores_by_domain = evaluate_by_domain(
        arch,
        "rawts",
        X,
        y,
        metric_key=metric_key,
        workspace=workspace,
        use_cache=use_cache,
        limit_domains=limit_domains,
        **kwargs,
    )

    scores = list(scores_by_domain.values())
    final_score = generate_score(scores)
    log.info(f"[ML perf rawts] arch = {arch}, F1 score={print_score(final_score)}")

    return final_score, scores_by_domain


def evaluate_leakage(
    features_range: dict,
    workspace=Path("output_wefde"),
    wefde_features_dir=Path("wefde_features"),
    n_procs=0,
    n_samples=50000,
    topn=20,
    nmi_threshold=0.5,
    discrete_threshold=100000,
    max_instances=1000,
    compute_joint: bool = True,
    compress_results: bool = True,
    debug_correctness: bool = False,
    dataset_split: bool = False,
):
    if not wefde_features_dir.exists():
        log.error("WeFDE features not extracted")
        return
    workspace.mkdir(parents=True, exist_ok=True)

    return evaluate_info_leakage(
        features_path=wefde_features_dir,
        output_path=workspace,
        features_range=features_range,
        n_procs=n_procs,
        n_samples=n_samples,
        topn=topn,
        nmi_threshold=nmi_threshold,
        discrete_threshold=discrete_threshold,
        max_instances=max_instances,
        compute_joint=compute_joint,
        compress_results=compress_results,
        debug_correctness=debug_correctness,
        dataset_split=dataset_split,
    )
