import glob
from pathlib import Path
from typing import Any, Tuple, Union

import cloudpickle

# from wfaudit.helpers_ml import load_from_file, print_score, generate_score
import numpy as np
import pandas as pd


def generate_score(metric: np.ndarray) -> Tuple[float, float]:
    percentile_val = 1.96
    return (np.mean(metric), percentile_val * np.std(metric) / np.sqrt(len(metric)))


def print_score(score: Tuple[float, float]) -> str:
    return str(round(score[0], 3)) + " +/- " + str(round(score[1], 3))


def load_from_file(path: Union[str, Path]) -> Any:
    with open(path, "rb") as f:
        return cloudpickle.load(f)


ml_models = ["xgboost", "kfp", "varcnn", "df", "holmes", "robustfp"]


def mi_from_f1(f1, n_classes=100):
    """
    Lower bound on MI from observed F1 via inverted Fano inequality.
    Valid for balanced datasets where F1 ≈ accuracy.
    """
    H_Y = np.log2(n_classes)
    pe = float(np.clip(1.0 - f1, 0.0, 1.0))
    # H(P_e), not the constant 1: the constant is only tight at P_e = 0.5 and
    # caps K* at 2 for a perfect attacker instead of 1
    h_pe = (
        0.0
        if pe <= 0.0 or pe >= 1.0
        else float(-pe * np.log2(pe) - (1 - pe) * np.log2(1 - pe))
    )
    mi_lower = H_Y - h_pe - pe * np.log2(n_classes - 1)
    return float(np.clip(mi_lower, 0.0, H_Y))


def anon_from_mi(mi_bits, n_classes=100):
    """
    Effective anonymity set A = 2^{H(Y) - I}.
    """
    mi = np.asarray(mi_bits, dtype=float)
    H_Y_bits = np.log2(n_classes)
    mi = np.clip(mi, 0.0, H_Y_bits)
    A = 2.0 ** (H_Y_bits - mi)
    A = np.clip(A, 1.0, 2.0**H_Y_bits)
    return A if A.shape != () else float(A)


# Cell names under <dataset>/results/ map to the defense labels the table
# generators expect.  Server placements keep their suffix.
CELL_TO_DEFENSE = {
    "nop": "undefended",
    "front": "cldef_front",
    "httpos": "cldef_httpos",
    "llama": "cldef_llama",
    "tamaraw": "cldef_tamaraw",
    "h2pc": "cldef_h2pc",
    "srvh2ps1p": "srvdef_h2ps",
}
SERVER_PLACEMENT = {"1st": "html", "3rd_1": "cdn1", "3rd_2": "cdn2", "all": "all"}


def cell_to_defense(cell: str) -> str:
    if cell in CELL_TO_DEFENSE:
        return CELL_TO_DEFENSE[cell]
    for prefix, name in (("srvalpaca_", "alpaca"), ("srvtamaraw_", "tamaraw")):
        if cell.startswith(prefix):
            place = SERVER_PLACEMENT.get(cell[len(prefix) :], cell[len(prefix) :])
            return f"srvdef_{name}_{place}"
    return cell


def find_cells(root, datasets=None):
    """-> [(dataset, cell, workspace_path)] for the per-dataset results tree:
    <root>/<dataset>/results/<cell>/tcp_repr/
    """
    root = Path(root)
    out = []
    print("find cells", root)
    for ds_dir in sorted(root.iterdir()):
        if not ds_dir.is_dir() or not (ds_dir / "results").is_dir():
            continue
        if datasets and ds_dir.name not in datasets:
            continue
        for cell in sorted((ds_dir / "results").iterdir()):
            if (cell / "tcp_repr").is_dir():
                out.append((ds_dir.name, cell.name, cell))
    return out


def security_results(workspace, setup: str = "tcp_repr", n_classes=100, datasets=None):
    summaries = []

    for dataset, cell, testcase_workspace in find_cells(workspace, datasets):
        print("prcessing dataset", cell, testcase_workspace)
        data = {}
        mod = cell_to_defense(cell)

        H_Y = np.log2(n_classes)

        # --- WeFDE: MI and K* only (no BER) ---
        null_miwfd = 0.0
        leakage_path = testcase_workspace / f"{setup}/eval_wefde/sanity_0/leakage.csv"
        if leakage_path.exists():
            null_miwfd = float(pd.read_csv(leakage_path)["lk_syn"].values[0])

        leakage_path = testcase_workspace / f"{setup}/eval_wefde/leakage.csv"
        if leakage_path.exists():
            leakage_res = pd.read_csv(leakage_path)
            # column is MI_TOTAL in the current library; lk_syn in older runs
            col = "MI_TOTAL" if "MI_TOTAL" in leakage_res.columns else "lk_syn"
            mi_wefde = float(leakage_res[col].values[0]) - float(null_miwfd)
            mi_wefde = float(np.clip(mi_wefde, 0.0, H_Y))
            data["mi_wefde"] = [mi_wefde]
            data["anon_wefde"] = [anon_from_mi(mi_wefde, n_classes=n_classes)]
        else:
            data["mi_wefde"] = [np.nan]
            data["anon_wefde"] = [np.nan]

        # --- DeepSE: MI, K*, and A* band ---
        acc_lo = np.nan
        acc_hi = np.nan

        path_real = testcase_workspace / f"{setup}/eval_deepse/results_df.csv"
        if not path_real.exists():
            path_real = (
                testcase_workspace / f"{setup}/eval_deepse/real/results_hl_df.csv"
            )
        if path_real.exists():
            dse = pd.read_csv(path_real)
            dse = dse.rename(
                columns={"MI_TOTAL": "mi", "BER_LO": "ber_lo", "BER_HI": "ber_hi"}
            )

            mi_deepse = float(dse["mi"].mean())  # - null_midse
            mi_deepse = float(np.clip(mi_deepse, 0.0, H_Y))
            data["mi_dse"] = [mi_deepse]

            A_list = anon_from_mi(dse["mi"].values, n_classes=n_classes)
            data["anon_dse_min"] = [float(np.min(A_list))]
            data["anon_dse_max"] = [float(np.max(A_list))]
            data["anon_dse_mean"] = [float(np.mean(A_list))]

            # BER band from DeepSE only
            dse_lower_err = float(np.mean(dse["ber_lo"]))
            dse_upper_err = float(np.mean(dse["ber_hi"]))

            data["ber_dse_lo"] = [dse_lower_err]
            data["ber_dse_hi"] = [dse_upper_err]

            acc_lo = float(np.clip(1.0 - dse_upper_err, 0.0, 1.0))
            acc_hi = float(np.clip(1.0 - dse_lower_err, 0.0, 1.0))

            data["acc_dse_lo"] = [acc_lo]
            data["acc_dse_hi"] = [acc_hi]
        else:
            data["mi_dse"] = [np.nan]
            data["anon_dse_min"] = [np.nan]
            data["anon_dse_max"] = [np.nan]
            data["anon_dse_mean"] = [np.nan]

        # Final band
        data["acc_lo"] = [acc_lo]
        data["acc_hi"] = [acc_hi]
        data["band_inconsistent"] = [
            bool(
                not np.isnan(acc_lo) and not np.isnan(acc_hi) and acc_lo > acc_hi + 1e-9
            )
        ]

        # --- ML models ---
        def _first_existing(*paths):
            for p in paths:
                if Path(p).exists():
                    return p
            return paths[0]

        def _load_results(res_path, key):
            if Path(res_path).exists():
                domain_scores = load_from_file(res_path)
                if "raw" not in domain_scores:
                    return generate_score(list(domain_scores.values()))
                elif "raw" in domain_scores:
                    # a metric absent from this run must not abort the whole table
                    return domain_scores["raw"].get(key)
            return None

        for ml_model in ml_models:
            # Tabular models
            ml_workspace = testcase_workspace / f"{setup}/eval_ml"
            if ml_model == "xgboost":
                model_results_path = _first_existing(
                    ml_workspace / f"scores_rawts_{ml_model}_tuned.bkp",
                    ml_workspace / f"scores_rawts_{ml_model}_topk.bkp",
                )
                results = _load_results(model_results_path, key="f1_score_macro")
                if results is not None:
                    data["sanity_f1_mean"] = [results[0]]
                    data["sanity_f1_std"] = [results[1]]

            keys = [
                "f1_score_macro",
                "acc_top5",
                "acc_top10",
                "rank_median",
                "rank_mean",
                "entropy_mi",
                "entropy_uncert",
                "confidence_mean",
            ]
            model_results_path = _first_existing(
                ml_workspace / f"scores_rawts_{ml_model}_tuned.bkp",
                ml_workspace / f"scores_rawts_{ml_model}_topk.bkp",
            )
            for key in keys:
                results = _load_results(model_results_path, key=key)
                if results is not None:
                    data[f"{key}_{ml_model}_mean"] = [results[0]]
                    data[f"{key}_{ml_model}_std"] = [results[1]]

            # Neural network models
            ml_workspace = testcase_workspace / f"{setup}/eval_ml_nn"
            model_results_path = _first_existing(
                ml_workspace / f"scores_rawts_{ml_model}_tuned.bkp",
                ml_workspace / f"scores_rawts_{ml_model}_topk.bkp",
            )
            for key in keys:
                results = _load_results(model_results_path, key=key)
                if results is not None:
                    data[f"{key}_{ml_model}_mean"] = [results[0]]
                    data[f"{key}_{ml_model}_std"] = [results[1]]

        # --- MI from best observed F1 ---
        f1_values = []
        for ml_model in ml_models:
            f1_key = f"f1_score_macro_{ml_model}_mean"
            if f1_key in data and not np.isnan(data[f1_key][0]):
                f1_values.append(data[f1_key][0])

        if f1_values:
            best_f1 = max(f1_values)
            mi_f1 = mi_from_f1(best_f1, n_classes=n_classes)
            data["mi_f1"] = [mi_f1]
            data["best_f1"] = [best_f1]
        else:
            best_f1 = np.nan
            data["mi_f1"] = [np.nan]
            data["best_f1"] = [np.nan]

        # --- Final MI: maximum across all estimators ---
        mi_candidates = [
            # data.get("mi_wefde", [np.nan])[0],
            data.get("mi_dse", [np.nan])[0],
            data.get("mi_f1", [np.nan])[0],
        ]
        mi_best = float(np.nanmax(mi_candidates))
        data["mi_best"] = [mi_best]

        # K* derived from best MI — guaranteed compatible with F1
        data["anon_cons"] = [anon_from_mi(mi_best, n_classes=n_classes)]

        # --- Tighten A* band using best F1 ---
        # F1 is a lower bound on acc_hi: a practical attacker achieves F1,
        # so the theoretical best must be at least as good
        if not np.isnan(best_f1):
            if np.isnan(acc_hi):
                acc_hi = best_f1
            else:
                acc_hi = max(acc_hi, best_f1)

        # acc_lo cannot exceed acc_hi
        if not np.isnan(acc_lo) and not np.isnan(acc_hi):
            acc_lo = min(acc_lo, acc_hi)

        acc_lo = float(np.clip(acc_lo, 0.0, 1.0)) if not np.isnan(acc_lo) else np.nan
        acc_hi = float(np.clip(acc_hi, 0.0, 1.0)) if not np.isnan(acc_hi) else np.nan

        data["acc_lo"] = [acc_lo]
        data["acc_hi"] = [acc_hi]

        # Flag inconsistency only if band is inverted after all corrections
        data["band_inconsistent"] = [
            bool(
                not np.isnan(acc_lo) and not np.isnan(acc_hi) and acc_lo > acc_hi + 1e-9
            )
        ]

        data["dataset"] = [dataset]
        data["defense"] = [mod]
        summaries.append(pd.DataFrame(data))

    return (
        pd.concat(summaries, ignore_index=True)
        .sort_values(["dataset", "defense"])
        .reset_index(drop=True)
        .sort_index(axis=1)
    )


def count_testcases(workspace):
    files = glob.glob(f"{str(workspace)}/output_wefde/*", recursive=True)
    uniq_pages = set()
    for f in files:
        fname = Path(f).name
        uniq_pages.add(fname.split("-")[0])

    return len(uniq_pages)


# Function to calculate the 95% confidence interval
def confidence_interval_95(series):
    mean = series.mean()
    std = series.std()
    n = len(series)
    margin_of_error = 1.96 * (std / np.sqrt(n))
    return f"${np.round(mean, 3)} \\pm {np.round(margin_of_error, 3)}$"
