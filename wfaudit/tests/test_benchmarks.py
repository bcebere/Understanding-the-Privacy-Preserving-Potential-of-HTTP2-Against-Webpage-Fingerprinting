# stdlib
import json
from pathlib import Path

# wfaudit absolute
from wfaudit import create_datasets, evaluate_leakage, evaluate_ml, prepare_features
from wfaudit.helpers_ml import print_score


def test_sanity(tmp_path):
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
    assert "joint" in leakage.columns

    print(leakage)

    # Test ML benchmark
    for arch in ["xgboost", "svm", "lr", "rf"]:
        output_ml = tmp_path / f"output_ml_{arch}"
        ml_score = evaluate_ml(
            workspace=output_ml, wefde_features_dir=output_features, arch=arch
        )
        assert output_ml.exists()
        assert max(ml_score) <= 1
        print("ML", arch, print_score(ml_score))
