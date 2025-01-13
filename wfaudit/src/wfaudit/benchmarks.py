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
from wfaudit.helpers_wefde.analysis.info_leak import evaluate_info_leakage
from wfaudit.helpers_wefde.preprocess.extract import prepare_wefde_features
import wfaudit.logger as log


def prepare_features(
    time_series_traces=Path("output_wefde"), output=Path("output_features")
):
    prepare_wefde_features(
        trace_path=time_series_traces,
        out_path=output,
    )


def evaluate_ml(
    workspace=Path("output_ml"),
    wefde_features_dir=Path("output_features"),
):
    if not wefde_features_dir.exists():
        log.error("WeFDE features not extracted")
        return

    workspace.mkdir(parents=True, exist_ok=True)

    metric_key = "f1_score_macro"

    for sample_limit in [None]:
        for arch in ["xgboost"]:
            if sample_limit is None:
                bkp_file = workspace / f"eval_ts_full_{arch}_{metric_key}.json"
            else:
                bkp_file = (
                    workspace
                    / f"eval_ts_full_{arch}_{metric_key}_samplelimit{sample_limit}.json"
                )
            if bkp_file.exists():
                scores = load_from_file(bkp_file)
                if len(scores) == 0:
                    bkp_file.unlink()
                    continue

            if not bkp_file.exists():
                if sample_limit is None:
                    X, y = load_wefde_features(wefde_features_dir)
                    scores = _evaluate_by_domain(
                        arch,
                        "full_data",
                        X,
                        y,
                        metric_key=metric_key,
                        workspace=workspace,
                    )
                else:
                    X, y = load_wefde_features(
                        wefde_features_dir, max_instances=sample_limit
                    )
                    scores = _evaluate_by_domain(
                        arch,
                        f"full_data_samplelim{sample_limit}",
                        X,
                        y,
                        metric_key=metric_key,
                        workspace=workspace,
                    )
                if len(scores) == 0:
                    continue
                save_to_file(bkp_file, scores)
            else:
                scores = load_from_file(bkp_file)

            final_score = generate_score(scores)
            log.info(f"[ML perf] arch = {arch}, score={print_score(final_score)}")


def evaluate_leakage(
    workspace=Path("output_leakage"),
    wefde_features_dir=Path("output_features"),
):
    if not wefde_features_dir.exists():
        log.error("WeFDE features not extracted")
        return

    evaluate_info_leakage(
        features_path=wefde_features_dir,
        output_path=workspace,
    )


def evaluate_all(
    time_series_traces=Path("output_wefde"),
    output_features=Path("output_features"),
    output_leakage=Path("output_leakage"),
    output_ml=Path("output_ml"),
):
    prepare_features(time_series_traces=time_series_traces, output=output_features)

    evaluate_leakage(workspace=output_leakage, wefde_features_dir=output_features)

    evaluate_ml(workspace=output_ml, wefde_features_dir=output_features)
