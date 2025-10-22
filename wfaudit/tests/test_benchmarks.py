# stdlib
from pathlib import Path

# third party
import pytest

# wfaudit absolute
from wfaudit import (
    audit,
    evaluate_leakage_from_deepse,
    evaluate_leakage_from_wefde,
    evaluate_ml_from_deepse,
    evaluate_ml_from_wefde,
    evaluate_shap,
)


@pytest.mark.parametrize("arch", ["xgboost", "kfp"])
def test_eval_ml_2D(tmp_path, arch):
    ml_folder = tmp_path / "eval_ml"
    wefde_feats_folder = Path("datasets") / "wefde_feats"
    scores = evaluate_ml_from_wefde(
        ml_output_folder=ml_folder,
        wefde_feats_folder=wefde_feats_folder,
        arch=arch,
    )

    assert "raw" in scores
    assert "str" in scores
    print(" >>> Score for ", arch, scores["str"])

    expected_metrics = [
        "f1_score_macro",
        "precision_macro",
        "recall_macro",
        "mcc",
        "acc_top5",
        "acc_top10",
        "rank_mean",
        "rank_median",
        "entropy_mi",
        "entropy_uncert",
    ]

    for metric in expected_metrics:
        assert metric in scores["raw"]
        assert metric in scores["str"]


@pytest.mark.parametrize("arch", ["varcnn", "df"])
def test_eval_ml_3D(tmp_path, arch):
    ml_folder = tmp_path / "eval_ml"
    deepse_dataset = Path("datasets") / "deepse" / "dataset.npz"
    assert deepse_dataset.exists()

    scores = evaluate_ml_from_deepse(
        ml_output_folder=ml_folder,
        deepse_dataset=deepse_dataset,
        arch=arch,
        # model params,
        epochs=2,
    )

    assert "raw" in scores
    assert "str" in scores
    print(" >>> Score for ", arch, scores["str"])

    expected_metrics = [
        "f1_score_macro",
        "precision_macro",
        "recall_macro",
        "mcc",
        "acc_top5",
        "acc_top10",
        "rank_mean",
        "rank_median",
        "entropy_mi",
        "entropy_uncert",
    ]

    for metric in expected_metrics:
        assert metric in scores["raw"]
        assert metric in scores["str"]


def test_eval_leakage_wefde(tmp_path):
    out_folder = tmp_path / "eval_leakage"
    wefde_feats_folder = Path("datasets") / "wefde_feats"
    assert wefde_feats_folder.exists()

    scores = evaluate_leakage_from_wefde(
        wefde_output_folder=out_folder,
        wefde_feats_folder=wefde_feats_folder,
    )

    print(scores)
    assert "MI_PKT_COUNT" in scores  # pkt count statistics leakage
    assert "MI_PKT_TIME" in scores  # pkt timing leakage
    assert "MI_BURST" in scores  # burst patterns leakage
    assert "MI_CUMUL" in scores  # cumulative statistics leakage
    assert "MI_TOTAL" in scores  # joint leakage


def test_eval_leakage_deepse(tmp_path):
    deepse_dataset = Path("datasets") / "deepse" / "dataset.npz"
    assert deepse_dataset.exists()

    results = evaluate_leakage_from_deepse(
        deepse_dataset=deepse_dataset,
        deepse_output=tmp_path / "results.csv",
        # dummy small values,
        epochs=2,
        feature_length=100,
        k_fold=2,
        embedding_size=100,
    )

    assert results is not None
    assert len(results) == 2  # k_fold size

    assert (tmp_path / "results.csv").exists()

    assert "MI_TOTAL" in results
    assert "BER_LO" in results
    assert "BER_HI" in results


def test_eval_xai(tmp_path):
    out_folder = tmp_path / "eval_xai"
    wefde_feats_folder = Path("datasets") / "wefde_feats"
    assert wefde_feats_folder.exists()

    scores = evaluate_shap(
        xai_output_folder=out_folder,
        wefde_feats_folder=wefde_feats_folder,
    )

    assert len(scores) > 0
    assert (scores["importance"] > 0).sum() > 0
    print(scores.sort_values("importance", ascending=False))


def test_audit_e2e(tmp_path):
    ml_output_folder = tmp_path / "eval_ml"
    wefde_output_folder = tmp_path / "eval_wefde"
    deepse_output = tmp_path / "eval_deepse/results.csv"
    xai_output_folder = tmp_path / "eval_xai"

    wefde_feats_folder = Path("datasets") / "wefde_feats"
    deepse_dataset = Path("datasets") / "deepse" / "dataset.npz"

    ml_arch_2D = ["xgboost", "kfp"]
    ml_arch_3D = ["varcnn"]

    scores = audit(
        # ML
        ml_output_folder=ml_output_folder,
        wefde_feats_folder=wefde_feats_folder,
        deepse_dataset=deepse_dataset,
        ml_arch_2D=ml_arch_2D,
        ml_arch_3D=ml_arch_3D,
        ml_kwargs={
            "varcnn": {
                "epochs": 2,
            },
            "df": {
                "epochs": 2,
            },
        },
        # leakage
        wefde_output_folder=wefde_output_folder,
        wefde_kwargs={
            "topn": 2,
        },
        deepse_output=deepse_output,
        deepse_kwargs={
            "epochs": 2,
            "feature_length": 100,
            "k_fold": 2,
            "embedding_size": 100,
        },
        # xai
        xai_output_folder=xai_output_folder,
    )

    assert ml_output_folder.exists()

    assert "ML" in scores
    for arch in ml_arch_2D + ml_arch_3D:
        assert arch in scores["ML"]
        assert "raw" in scores["ML"][arch]
        assert "f1_score_macro" in scores["ML"][arch]["raw"]

    assert wefde_output_folder.exists()
    assert deepse_output.exists()
    assert "leakage" in scores
    assert "wefde" in scores["leakage"]
    assert "deepse" in scores["leakage"]

    assert xai_output_folder.exists()
    assert "xai" in scores
    assert "shap" in scores["xai"]
