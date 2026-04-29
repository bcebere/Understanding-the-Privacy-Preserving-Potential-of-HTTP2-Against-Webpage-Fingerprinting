# stdlib
import json
from pathlib import Path

# third party
import numpy as np
import pandas as pd
import shap
import wfaudit.logger as log
from sklearn.model_selection import train_test_split

# wfaudit absolute
from wfaudit.helpers_deepse import estimate_mi_ber
from wfaudit.helpers_ml import evaluate_multiclass
from wfaudit.helpers_ml.xgb import XGBoostClassifier
from wfaudit.helpers_wefde.analysis.data_utils import load_wefde_features
from wfaudit.helpers_wefde.analysis.info_leak import evaluate_info_leakage


def evaluate_ml_from_wefde(
    ml_output_folder=Path("output_ml"),
    wefde_feats_folder=Path("wefde_features"),
    arch: str = "xgboost",
    use_cache: bool = False,
):
    if not wefde_feats_folder.exists():
        log.error("WeFDE features not extracted")
        return None

    ml_output_folder.mkdir(parents=True, exist_ok=True)

    X, y = load_wefde_features(wefde_feats_folder)

    assert len(X) > 0
    assert len(X) == len(y)

    scores = evaluate_multiclass(
        arch=arch,
        label="topk",
        data=X,
        labels=y,
        workspace=ml_output_folder,
        use_cache=use_cache,
    )

    return scores


def evaluate_ml_from_deepse(
    ml_output_folder=Path("output_ml"),
    deepse_dataset=Path("deepse/dataset.npz"),
    arch: str = "varcnn",
    use_cache: bool = False,
    **kwargs,
):
    if not deepse_dataset.exists():
        log.error("DeepSE-WF features not extracted")
        return None

    ml_output_folder.mkdir(parents=True, exist_ok=True)

    data = np.load(deepse_dataset)
    X = data["traces"].astype("float32")
    y = data["labels"].astype("float32")

    assert len(X) > 0
    assert len(X) == len(y)

    scores = evaluate_multiclass(
        arch=arch,
        label="topk",
        data=X,
        labels=y,
        workspace=ml_output_folder,
        use_cache=use_cache,
        **kwargs,
    )

    return scores


def evaluate_leakage_from_wefde(
    wefde_output_folder=Path("output_wefde"),
    wefde_feats_folder=Path("wefde_features"),
    n_procs=0,
    n_samples=50000,
    topn=10,
    nmi_threshold=0.5,
    discrete_threshold=100000,
    max_instances=1000,
    compute_joint: bool = True,
    compress_results: bool = True,
    debug_correctness: bool = False,
    dataset_split: bool = False,
):
    if not wefde_feats_folder.exists():
        log.error("WeFDE features not extracted")
        return
    wefde_output_folder.mkdir(parents=True, exist_ok=True)

    with open(wefde_feats_folder / "FeaturePositions.json", "r") as f:
        features_range = json.load(f)

    return evaluate_info_leakage(
        features_path=wefde_feats_folder,
        output_path=wefde_output_folder,
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


def evaluate_leakage_from_deepse(
    deepse_dataset: Path,
    deepse_output: str = "results.csv",
    n_websites: int = None,  # if None, use all
    n_traces: int = None,  # if None, use max
    feature_length: int = 5000,
    k_fold: int = 5,
    random_state: int = 42,
    embedding_size: int = 512,
    model: str = "df",
    device: str = "cpu",
    epochs: int = 100,
    batch_size: int = 200,
    num_workers: int = 4,
    knn_measure: str = "squared_l2",
    ber_k: int = 1,
    mi_k: int = 5,
):

    results = estimate_mi_ber(
        data_path=deepse_dataset,
        results_file=deepse_output,
        n_websites=n_websites,
        n_traces=n_traces,
        feature_length=feature_length,
        k_fold=k_fold,
        random_state=random_state,
        embedding_size=embedding_size,
        model=model,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        knn_measure=knn_measure,
        ber_k=ber_k,
        mi_k=mi_k,
    )
    return results


def evaluate_shap(
    xai_output_folder=Path("output_xai"),
    wefde_feats_folder=Path("wefde_features"),
):
    if not wefde_feats_folder.exists():
        log.error("WeFDE features not extracted")
        return None

    xai_output_folder.mkdir(parents=True, exist_ok=True)

    X, y = load_wefde_features(wefde_feats_folder)

    assert len(X) > 0
    assert len(X) == len(y)

    with open(wefde_feats_folder / "FeaturePositions.json", "r") as f:
        features_range = json.load(f)

    granular_features = {
        "MI_PKT_COUNT": [
            "pkt_total",
            "pkt_out",
            "pkt_in",
            "pkt_uniq_out",
            "pkt_uniq_in",
        ],
        "MI_PKT_TIME": [f"time_{cat}" for cat in ["mean", "std", "sum"]],
        "MI_BURST": [f"burst_uniq_topn{i}" for i in range(20)]
        + [f"burst_uniq_{cat}" for cat in ["mean", "std", "sum"]],
        "MI_CUMUL": ["cumul_all_out", "cumul_all_in"]
        + [f"cumul_all_interp{i}" for i in range(20)],
    }
    dataframe_cols = []
    prev_offset = 0
    for cat in features_range:
        assert cat in granular_features, cat
        cat_range = [prev_offset, features_range[cat]]
        assert (features_range[cat] - prev_offset) % len(granular_features[cat]) == 0
        conn_cnt = (features_range[cat] - prev_offset) / len(granular_features[cat])
        local_df_cols = []
        for conn_idx in range(int(conn_cnt)):
            for local_feat in granular_features[cat]:
                local_df_cols.append(f"{local_feat}_conn{conn_idx}")

        assert len(local_df_cols) == cat_range[1] - cat_range[0]
        dataframe_cols.extend(local_df_cols)
        prev_offset = cat_range[-1]

    assert len(dataframe_cols) == X.shape[-1]

    X = pd.DataFrame(X, columns=dataframe_cols)
    y = pd.Series(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    n_classes = len(np.unique(y_train))
    model = XGBoostClassifier()
    model.fit(X_train, y_train)

    def shap_to_tensor(shap_out, n_classes):
        """
        Normalize SHAP outputs to (n_samples, n_classes, n_features).

        - list of length C with arrays (n, f) -> stack to (n, C, f)
        - 2-D array (n, f) with n_classes == 2 -> expand to (n, 2, f) as [-a, a]
        - 3-D array -> move class axis to axis=1 if needed
        """
        # 1) List case (typical for multiclass)
        if isinstance(shap_out, list):
            return np.stack(shap_out, axis=1)  # (n, C, f)

        # 2) NumPy array cases
        if shap_out.ndim == 2:
            # Binary classification: SHAP gives only the positive class.
            if n_classes == 2:
                a = shap_out  # (n, f) for positive class
                return np.stack([-a, a], axis=1)  # (n, 2, f)
            raise ValueError(
                f"Got a 2-D array {shap_out.shape} but n_classes={n_classes}. "
                "Only binary classification should return 2-D SHAP values."
            )

        if shap_out.ndim != 3:
            raise ValueError(
                f"Expecting 3-D array for multiclass, got {shap_out.ndim}D."
            )

        # 3) 3-D array: find the class axis and move it to axis 1
        n, s1, s2 = shap_out.shape
        if s1 == n_classes:  # (n, C, f)
            return shap_out
        if s2 == n_classes:  # (n, f, C) -> (n, C, f)
            return np.moveaxis(shap_out, 2, 1)

        raise ValueError(
            f"Could not find class axis in shape {shap_out.shape} with n_classes={n_classes}."
        )

    explainer = shap.TreeExplainer(model.model)
    raw_shap = explainer.shap_values(X_test)
    print("raw shap", raw_shap.shape)

    sv = shap_to_tensor(raw_shap, n_classes)

    global_imp = np.abs(sv).mean(axis=(0, 1))

    imp_tbl = pd.DataFrame({"feature": X_train.columns, "importance": global_imp})
    imp_tbl.to_csv(xai_output_folder / "shap.csv", index=None)

    return imp_tbl


def audit(
    wefde_feats_folder=Path("wefde_features"),
    deepse_dataset=Path("deepse/dataset.npz"),
    # ML area
    ml_output_folder=Path("output_ml"),
    ml_arch_2D=["xgboost", "kfp"],
    ml_arch_3D=["varcnn", "df"],
    ml_kwargs={},
    # Leakage area
    leakage_estimators=["wefde", "deepse"],
    wefde_output_folder=Path("output_wefde"),
    wefde_kwargs={},
    deepse_output=Path("output_deepse/results.csv"),
    deepse_kwargs={},
    # XAI area
    xai_output_folder=Path("output_xai"),
):
    assert wefde_feats_folder.exists(), wefde_feats_folder
    assert deepse_dataset.exists(), deepse_dataset

    scores = {}

    # ML estimators
    ml_scores = {}
    for arch in ml_arch_2D:
        ml_scores[arch] = evaluate_ml_from_wefde(
            ml_output_folder=ml_output_folder,
            wefde_feats_folder=wefde_feats_folder,
            arch=arch,
        )

    for arch in ml_arch_3D:
        kwargs = {}
        if arch in ml_kwargs:
            kwargs = ml_kwargs[arch]

        ml_scores[arch] = evaluate_ml_from_deepse(
            ml_output_folder=ml_output_folder,
            deepse_dataset=deepse_dataset,
            arch=arch,
            **kwargs,
        )
    scores["ML"] = ml_scores

    # Leakage estimators
    leak_scores = {}

    if "wefde" in leakage_estimators:
        leak_scores["wefde"] = evaluate_leakage_from_wefde(
            wefde_output_folder=wefde_output_folder,
            wefde_feats_folder=wefde_feats_folder,
            **wefde_kwargs,
        )
    if "deepse" in leakage_estimators:
        leak_scores["deepse"] = evaluate_leakage_from_deepse(
            deepse_dataset=deepse_dataset,
            deepse_output=deepse_output,
            **deepse_kwargs,
        )
    scores["leakage"] = leak_scores

    # XAI
    xai_scores = {}

    xai_scores["shap"] = evaluate_shap(
        xai_output_folder=xai_output_folder,
        wefde_feats_folder=wefde_feats_folder,
    )
    scores["xai"] = xai_scores

    return scores
