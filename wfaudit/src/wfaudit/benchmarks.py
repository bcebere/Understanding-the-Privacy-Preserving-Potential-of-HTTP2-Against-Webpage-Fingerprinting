# stdlib
from pathlib import Path

# wfaudit absolute
from wfaudit.helpers_ml import (
    _evaluate_by_domain,
    generate_score,
    load_from_file,
    print_score,
    save_to_file,
)
from wfaudit.helpers_wefde.analysis.data_utils import load_wefde_features
from wfaudit.helpers_wefde.analysis.info_leak import (
    evaluate_info_leakage,
    evaluate_info_leakage_v2,
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
):
    if not wefde_features_dir.exists():
        log.error("WeFDE features not extracted")
        return None

    workspace.mkdir(parents=True, exist_ok=True)

    arch = "xgboost"

    bkp_file = workspace / f"eval_ts_full_{arch}_{metric_key}.json"
    if bkp_file.exists():
        scores = load_from_file(bkp_file)
        return scores

    if not bkp_file.exists():
        X, y = load_wefde_features(wefde_features_dir)
        scores = _evaluate_by_domain(
            arch,
            "full_data",
            X,
            y,
            metric_key=metric_key,
            workspace=workspace,
        )
        save_to_file(bkp_file, scores)
    else:
        scores = load_from_file(bkp_file)

    final_score = generate_score(scores)
    log.info(f"[ML perf] arch = {arch}, F1 score={print_score(final_score)}")

    return final_score


def evaluate_leakage(
    features_range: dict,
    workspace=Path("output_leakage"),
    wefde_features_dir=Path("output_features"),
):
    if not wefde_features_dir.exists():
        log.error("WeFDE features not extracted")
        return
    workspace.mkdir(parents=True, exist_ok=True)

    return evaluate_info_leakage(
        features_path=wefde_features_dir,
        output_path=workspace,
        features_range=features_range,
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
