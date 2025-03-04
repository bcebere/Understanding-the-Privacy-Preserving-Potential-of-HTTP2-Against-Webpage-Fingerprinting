# stdlib
import hashlib
from pathlib import Path

# third party
import numpy as np
import pandas as pd

# wfaudit absolute
from wfaudit.helpers_ml import evaluate_by_domain, generate_score, print_score
from wfaudit.helpers_wefde.analysis.data_utils import load_wefde_features
from wfaudit.helpers_wefde.analysis.info_leak import (
    evaluate_info_leakage,
    evaluate_info_leakage_v2,
    exploratory_analysis,
)
from wfaudit.helpers_wefde.preprocess.extract import prepare_wefde_features
import wfaudit.logger as log


def prepare_features(
    time_series_traces=Path("output_wefde"), output=Path("output_features")
):
    output.mkdir(parents=True, exist_ok=True)
    return prepare_wefde_features(
        trace_path=time_series_traces,
        out_path=output,
    )


def evaluate_ml(
    workspace=Path("output_ml"),
    wefde_features_dir=Path("output_features"),
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
    filtered_labels=None,
    use_cache: bool = True,
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
        filtered_labels=filtered_labels,
        use_cache=use_cache,
        **kwargs,
    )

    scores = list(scores_by_domain.values())
    final_score = generate_score(scores)
    log.info(f"[ML perf rawts] arch = {arch}, F1 score={print_score(final_score)}")

    return final_score, scores_by_domain


def evaluate_leakage(
    features_range: dict,
    workspace=Path("output_leakage"),
    wefde_features_dir=Path("output_features"),
    n_procs=0,
    n_samples=50000,
    topn=20,
    nmi_threshold=0.9,
    discrete_threshold=100000,
    max_instances=100,
    compute_joint: bool = True,
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
    )


def evaluate_leakage_v2(
    X,
    y,
    features_range: dict = None,
    workspace=Path("output_leakage"),
    n_procs=0,
    n_samples=50000,
    topn=40,
    nmi_threshold=0.9,
    discrete_threshold=100000,
    max_instances=100,
):
    workspace.mkdir(parents=True, exist_ok=True)

    return evaluate_info_leakage_v2(
        X,
        y,
        output_path=workspace,
        features_range=features_range,
        n_procs=n_procs,
        n_samples=n_samples,
        topn=topn,
        nmi_threshold=nmi_threshold,
        discrete_threshold=discrete_threshold,
        max_instances=max_instances,
    )


def evaluate_exploratory(
    X,
    y,
    min_cluster_size: int,
    features_range: dict = None,
    workspace=Path("output_leakage"),
    n_procs=0,
    n_samples=50000,
    topn=40,
    nmi_threshold=0.9,
    discrete_threshold=100000,
    max_instances=100,
):
    workspace.mkdir(parents=True, exist_ok=True)

    return exploratory_analysis(
        X,
        y,
        min_cluster_size=min_cluster_size,
        output_path=workspace,
        features_range=features_range,
        n_procs=n_procs,
        n_samples=n_samples,
        topn=topn,
        nmi_threshold=nmi_threshold,
        discrete_threshold=discrete_threshold,
        max_instances=max_instances,
    )


def evaluate_exploratory_ml(
    X: pd.DataFrame,
    y: pd.DataFrame,
    min_cluster_size: int,
    features_range: dict = None,
    workspace=Path("output_leakage"),
    n_procs=0,
    n_samples=50000,
    topn=40,
    nmi_threshold=0.9,
    discrete_threshold=100000,
    max_instances=100,
):
    top_feats, clusters = evaluate_exploratory(
        np.asarray(X),
        np.asarray(y),
        min_cluster_size=min_cluster_size,
        workspace=workspace,
        features_range=features_range,
        n_procs=n_procs,
        n_samples=n_samples,
        topn=topn,
        nmi_threshold=nmi_threshold,
        discrete_threshold=discrete_threshold,
        max_instances=max_instances,
    )

    for idx, (cluster, cluster_leak) in enumerate(clusters):
        # print(f"Evaluate cluster {idx}. Bits leaked {cluster_leak}. Features = {X.columns[cluster]}")
        cluster_hash = "_".join(map(str, sorted(cluster)))
        arch = "xgboost"
        md5_hash = hashlib.md5()
        md5_hash.update(cluster_hash.encode("utf-8"))
        cluster_hash = md5_hash.hexdigest()
        scores, scores_by_domain = evaluate_by_domain(
            arch,
            f"cluster_{arch}_{cluster_hash}",
            X[X.columns[cluster]],
            y,
            metric_key="f1_score_macro",
            workspace=Path("output_ml"),
        )
        print("Evaluate", idx, cluster, print_score(generate_score(scores)))


def evaluate_all(
    time_series_traces=Path("output_wefde"),
    output_features=Path("output_features"),
    output_leakage=Path("output_leakage"),
    output_ml=Path("output_ml"),
):
    features_range = prepare_features(
        time_series_traces=time_series_traces, output=output_features
    )

    leakage = evaluate_leakage(
        features_range, workspace=output_leakage, wefde_features_dir=output_features
    )

    score = evaluate_ml(workspace=output_ml, wefde_features_dir=output_features)

    return features_range, leakage, score
