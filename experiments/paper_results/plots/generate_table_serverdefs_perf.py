from pathlib import Path

import numpy as np
import pandas as pd
from helpers import security_results

workspace = Path.cwd().parent / "benchmarks"

enabled_datasets = ["1_amazon", "2_bbc", "3_reddit", "4_udemy", "5_wiki"]
enabled_defenses = [
    f"srvdef_{defense}_{server}"
    for defense in ["alpaca", "tamaraw"]
    for server in ["html", "cdn1", "all"]
] + ["srvdef_h2ps"]

pretty_datasets = {
    "5_wiki": "Wikipedia",
    "1_amazon": "Amazon",
    "3_reddit": "Reddit",
    "2_bbc": "BBC",
    "4_udemy": "Udemy",
}
pretty_models = {
    "xgboost": "XGBoost",
    "holmes": "Holmes",
    "varcnn": "VarCNN",
    "robustfp": "RobustFP",
    "kfpv2": "k-Fingerprinting",
}
pretty_defenses = {}

sec_results = security_results(workspace)
sec_results = sec_results[sec_results["dataset"].isin(enabled_datasets)]
sec_results = sec_results[sec_results["defense"].isin(enabled_defenses)]
sec_results = (
    sec_results.replace(pretty_datasets).replace(pretty_models).replace(pretty_defenses)
)

ml_keys = ["dataset", "defense"]
ml_f1_results = []
for ridx, row in sec_results.iterrows():
    for model in ["kfpv2", "xgboost", "df", "varcnn", "robustfp", "holmes"]:
        row_info = [row["dataset"], row["defense"], model]
        for measure in ["mean", "std"]:
            model_key = f"f1_score_macro_{model}_{measure}"
            ml_keys.append(model_key)
            row_info.append(row[model_key])
        ml_f1_results.append(row_info)

ml_f1_results = (
    pd.DataFrame(
        ml_f1_results, columns=["Dataset", "Defense", "Model", "Value", "Error"]
    )
    .replace(pretty_datasets)
    .replace(pretty_models)
)
ml_f1_results["Value ± Error"] = ml_f1_results.apply(
    lambda row: f"${row['Value']:.2f} \\pm {row['Error']:.2f}$", axis=1
)

ml_f1_results = ml_f1_results.dropna()

ml_f1_results = (
    ml_f1_results.loc[ml_f1_results.groupby(["Dataset", "Defense"])["Value"].idxmax()]
    .sort_values(["Dataset", "Defense"])
    .reset_index(drop=True)
)

ml_def_results = (
    ml_f1_results.pivot(index="Dataset", columns="Defense", values="Value ± Error")
    .reset_index()
    .rename_axis(None, axis=1)
)
print(ml_f1_results)

# F1 score
print("F1-score")
print(
    ml_def_results[
        [
            "Dataset",
            "srvdef_alpaca_html",
            "srvdef_alpaca_cdn1",
            "srvdef_alpaca_all",
            "srvdef_tamaraw_html",
            "srvdef_tamaraw_cdn1",
            "srvdef_tamaraw_all",
            "srvdef_h2ps",
        ]
    ]
)

# Top-5 Acc

ml_keys = ["dataset", "defense"]
ml_topk_results = []
for ridx, row in sec_results.iterrows():
    for model in ["kfpv2", "xgboost", "df", "varcnn", "robustfp", "holmes"]:
        row_info = [row["dataset"], row["defense"], model]
        for measure in ["mean", "std"]:
            model_key = f"acc_top5_{model}_{measure}"
            ml_keys.append(model_key)
            row_info.append(row[model_key])
        ml_topk_results.append(row_info)

ml_topk_results = (
    pd.DataFrame(
        ml_topk_results, columns=["Dataset", "Defense", "Model", "Value", "Error"]
    )
    .replace(pretty_datasets)
    .replace(pretty_models)
)
ml_topk_results["Value ± Error"] = ml_topk_results.apply(
    lambda row: f"${row['Value']:.2f} \\pm {row['Error']:.2f}$", axis=1
)

ml_topk_results = ml_topk_results.dropna()

ml_topk_results = (
    ml_topk_results.loc[
        ml_topk_results.groupby(["Dataset", "Defense"])["Value"].idxmax()
    ]
    .sort_values(["Dataset", "Defense"])
    .reset_index(drop=True)
)

ml_topk_results
mltopk_def_results = (
    ml_topk_results.pivot(index="Dataset", columns="Defense", values="Value ± Error")
    .reset_index()
    .rename_axis(None, axis=1)
)

print("Top-5 acc")
print(
    mltopk_def_results[
        [
            "Dataset",
            "srvdef_alpaca_html",
            "srvdef_alpaca_cdn1",
            "srvdef_alpaca_all",
            "srvdef_tamaraw_html",
            "srvdef_tamaraw_cdn1",
            "srvdef_tamaraw_all",
            "srvdef_h2ps",
        ]
    ]
)

# Anon sets
anon_keys = ["dataset", "defense"]
anon_results = []

for ridx, row in sec_results.iterrows():
    if row["defense"] == "0. Baseline":
        continue
    row_info = [row["dataset"], row["defense"]]
    for measure in [
        "anon_wefde",
        "anon_dse_min",
        "anon_dse_mean",
        "anon_dse_max",
        "anon_cons",
    ]:
        row_info.append(row.get(measure, np.nan))
    anon_results.append(row_info)

anon_results = (
    pd.DataFrame(
        anon_results,
        columns=[
            "Dataset",
            "Defense",
            "K*_WeFDE",
            "K*_DSE_min",
            "K*_DSE_mean",
            "K*_DSE_max",
            "K*",
        ],
    )
    .replace(pretty_datasets)
    .replace(pretty_models)
)

anon_results = anon_results.dropna(subset=["K*"])
anon_results = anon_results.sort_values(["Dataset", "Defense"]).reset_index(drop=True)

# pivot on K* (conservative, MI-consistent)
anon_pivot = (
    anon_results.pivot(index="Dataset", columns="Defense", values="K*")
    .reset_index()
    .rename_axis(None, axis=1)
).round(2)

print("Anon sets")
print(
    anon_pivot[
        [
            "Dataset",
            "srvdef_alpaca_html",
            "srvdef_alpaca_cdn1",
            "srvdef_alpaca_all",
            "srvdef_tamaraw_html",
            "srvdef_tamaraw_cdn1",
            "srvdef_tamaraw_all",
            "srvdef_h2ps",
        ]
    ]
)
