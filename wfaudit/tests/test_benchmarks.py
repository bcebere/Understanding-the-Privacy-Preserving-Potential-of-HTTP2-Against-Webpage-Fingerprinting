# stdlib
import json
from pathlib import Path

# third party
import numpy as np
import pytest

# wfaudit absolute
from wfaudit import (
    create_datasets,
    evaluate_leakage,
    evaluate_ml,
    evaluate_ml_rawts,
    prepare_features,
)
from wfaudit.helpers_ml import print_score


def test_leakage(tmp_path):
    create_datasets(
        traces=Path("traces"),
        workspace=tmp_path,
        unlink_after_processing=False,
    )

    output_features = tmp_path / "eval_features"
    features = prepare_features(
        workspace=tmp_path,
        conn_limit=1,
    )

    assert output_features.exists()
    num_files = sum(1 for file in output_features.iterdir() if file.is_file())
    assert num_files == 6 + 1

    saved_features = json.load(open(output_features / "FeaturePositions.json"))
    assert features == saved_features

    # Test info leakage
    output_leakage = tmp_path / "output_leakage"
    leakage = evaluate_leakage(
        features,
        workspace=output_leakage,
        wefde_features_dir=output_features,
        topn=1,
        n_procs=4,
    )
    assert output_leakage.exists()

    for feat in features:
        assert feat in leakage.columns
    assert "leakage_topfeats" in leakage.columns


@pytest.mark.parametrize("arch", ["kfpv2", "kfp", "xgboost", "svm", "lr", "rf"])
def test_ml_benchmarks_stats(tmp_path, arch):
    create_datasets(
        traces=Path("traces"),
        workspace=tmp_path,
        unlink_after_processing=False,
    )

    output_features = tmp_path / "eval_features"
    features = prepare_features(
        workspace=tmp_path,
        conn_limit=1,
    )

    assert output_features.exists()
    num_files = sum(1 for file in output_features.iterdir() if file.is_file())
    assert num_files == 6 + 1

    saved_features = json.load(open(output_features / "FeaturePositions.json"))
    assert features == saved_features

    # Test ML benchmark
    output_ml = tmp_path / f"output_ml_{arch}"
    ml_score, scores_by_domain = evaluate_ml(
        workspace=output_ml,
        wefde_features_dir=output_features,
        arch=arch,
    )
    assert output_ml.exists()
    assert max(ml_score) <= 1
    print("ML", arch, print_score(ml_score))
    assert len(scores_by_domain.keys()) == 3
    for label in [0, 1, 2]:
        assert label in scores_by_domain


@pytest.mark.parametrize("arch", ["xgboost", "varcnn"])
def test_ml_benchmarks_nn_1C(tmp_path, arch):
    create_datasets(
        traces=Path("traces"),
        workspace=tmp_path,
        unlink_after_processing=False,
    )

    workspace = tmp_path / "output_ml"
    with open(workspace / "X_1C.npy", "rb") as f:
        X = np.load(f)
    with open(workspace / "y_1C.npy", "rb") as f:
        y = np.load(f)

    output_ml = tmp_path / f"output_ml_{arch}"
    ml_score, scores_by_domain = evaluate_ml_rawts(
        X, y, workspace=output_ml, arch=arch, epochs=10
    )
    assert output_ml.exists()
    assert max(ml_score) <= 1
    print("ML", arch, X.shape, print_score(ml_score))
    assert len(scores_by_domain.keys()) == 3
    for label in [0, 1, 2]:
        assert label in scores_by_domain


@pytest.mark.parametrize("arch", ["holmes", "tam"])
def test_ml_benchmarks_nn_3C(tmp_path, arch):
    create_datasets(
        traces=Path("traces"),
        workspace=tmp_path,
        unlink_after_processing=False,
    )

    workspace = tmp_path / "output_ml"
    with open(workspace / "X_3C.npy", "rb") as f:
        X = np.load(f)
    with open(workspace / "y_3C.npy", "rb") as f:
        y = np.load(f)

    output_ml = tmp_path / f"output_ml_{arch}"
    ml_score, scores_by_domain = evaluate_ml_rawts(
        X, y, workspace=output_ml, arch=arch, epochs=10
    )
    assert output_ml.exists()
    assert max(ml_score) <= 1
    print("ML", arch, X.shape, print_score(ml_score))
    assert len(scores_by_domain.keys()) == 3
    for label in [0, 1, 2]:
        assert label in scores_by_domain
