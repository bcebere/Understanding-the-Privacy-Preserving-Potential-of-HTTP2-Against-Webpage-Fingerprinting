# stdlib
import json
from pathlib import Path

# third party
import numpy as np

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

    output_features = tmp_path / "output_features"
    features = prepare_features(
        time_series_traces=tmp_path / "output_wefde",
        output=output_features,
    )

    assert output_features.exists()
    num_files = sum(1 for file in output_features.iterdir() if file.is_file())
    assert num_files == 6 + 1

    saved_features = json.load(open(output_features / "FeaturePositions.json"))
    print(features)
    assert features == saved_features

    # Test info leakage
    output_leakage = tmp_path / "output_leakage"
    leakage = evaluate_leakage(
        features, workspace=output_leakage, wefde_features_dir=output_features
    )
    assert output_leakage.exists()

    for feat in features:
        assert feat in leakage.columns
    assert "leakage_topfeats" in leakage.columns


def test_ml_benchmarks_stats(tmp_path):
    create_datasets(
        traces=Path("traces"),
        workspace=tmp_path,
        unlink_after_processing=False,
    )

    output_features = tmp_path / "output_features"
    features = prepare_features(
        time_series_traces=tmp_path / "output_wefde",
        output=output_features,
    )

    assert output_features.exists()
    num_files = sum(1 for file in output_features.iterdir() if file.is_file())
    assert num_files == 6 + 1

    saved_features = json.load(open(output_features / "FeaturePositions.json"))
    print(features)
    assert features == saved_features

    # Test ML benchmark
    for arch in ["kfp", "xgboost", "svm", "lr", "rf"]:
        output_ml = tmp_path / f"output_ml_{arch}"
        ml_score = evaluate_ml(
            workspace=output_ml, wefde_features_dir=output_features, arch=arch
        )
        assert output_ml.exists()
        assert max(ml_score) <= 1
        print("ML", arch, print_score(ml_score))


def test_ml_benchmarks_raw(tmp_path):
    create_datasets(
        traces=Path("traces"),
        workspace=tmp_path,
        unlink_after_processing=False,
    )

    workspace = tmp_path / "output_ml"
    with open(workspace / "X.npy", "rb") as f:
        X = np.load(f)
    with open(workspace / "y.npy", "rb") as f:
        y = np.load(f)

    # Test ML benchmark
    for arch in ["varcnn", "xgboost"]:
        output_ml = tmp_path / f"output_ml_{arch}"
        ml_score = evaluate_ml_rawts(
            X,
            y,
            workspace=output_ml,
            arch=arch,
            train_epochs=10,
        )
        print(ml_score)
        assert output_ml.exists()
        assert max(ml_score) <= 1
        print("ML", arch, print_score(ml_score))
