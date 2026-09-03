import sys
from pathlib import Path

import numpy as np
import pandas as pd
from helpers import security_results

_here = Path(__file__).parent  # dir of the (possibly symlinked) script
workspace = (
    Path(sys.argv[1])
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
    else _here / "workspace"
)


enabled_datasets = ["1_amazon", "2_bbc", "3_reddit", "4_udemy", "5_wiki"]
enabled_defenses = [
    "undefended",
    "cldef_tamaraw",
    "cldef_front",
    "cldef_httpos",
    "cldef_llama",
    "cldef_h2pc",
]

pretty_datasets = {
    "5_wiki": "Wikipedia",
    "1_amazon": "Amazon",
    "3_reddit": "Reddit",
    "2_bbc": "BBC",
    "4_udemy": "Udemy",
}
pretty_models = {
    "xgboost": "XGBoost",
    "df": "DF",
    "holmes": "Holmes",
    "varcnn": "VarCNN",
    "robustfp": "RobustFP-CNN",
    "kfp": "k-Fingerprinting",
}

pretty_defenses = {
    "cldef_front": "3. FRONT",
    "undefended": "0. Baseline",
    "cldef_httpos": "1. HTTPOS",
    "cldef_llama": "2. LLaMA",
    "cldef_tamaraw": "4. CL-Tamaraw",
    "cldef_h2pc": "5. H2PC",
}

sec_results = security_results(workspace)
sec_results = sec_results[sec_results["dataset"].isin(enabled_datasets)]
sec_results = sec_results[sec_results["defense"].isin(enabled_defenses)]
sec_results = (
    sec_results.replace(pretty_datasets).replace(pretty_models).replace(pretty_defenses)
)

ml_keys = ["dataset", "defense"]
ml_f1_results = []
for ridx, row in sec_results.iterrows():
    for model in ["kfp", "df", "varcnn", "robustfp", "holmes"]:
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
    lambda row: f"${row['Value']:.3f} \\pm {row['Error']:.3f}$", axis=1
)

ml_f1_results = ml_f1_results.dropna()
ml_f1_results_full = ml_f1_results.copy()

ml_f1_results = (
    ml_f1_results.loc[ml_f1_results.groupby(["Dataset", "Defense"])["Value"].idxmax()]
    .sort_values(["Dataset", "Defense"])
    .reset_index(drop=True)
)

print("F1 Score")
ml_def_results = (
    ml_f1_results.pivot(index="Dataset", columns="Defense", values="Value ± Error")
    .reset_index()
    .rename_axis(None, axis=1)
)

print(ml_def_results)

# Top-5

ml_keys = ["dataset", "defense"]
ml_topk_results = []
for ridx, row in sec_results.iterrows():
    for model in ["kfp", "df", "varcnn", "robustfp", "holmes"]:
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
    lambda row: f"${row['Value']:.3f} \\pm {row['Error']:.3f}$", axis=1
)

ml_topk_results = ml_topk_results.dropna()

ml_topk_results = (
    ml_topk_results.loc[
        ml_topk_results.groupby(["Dataset", "Defense"])["Value"].idxmax()
    ]
    .sort_values(["Dataset", "Defense"])
    .reset_index(drop=True)
)

print("Top - 5 acc")
mltopk_def_results = (
    ml_topk_results.pivot(index="Dataset", columns="Defense", values="Value ± Error")
    .reset_index()
    .rename_axis(None, axis=1)
)

print(mltopk_def_results)


# Anonimity sets
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

anon_pivot = (
    anon_results.pivot(index="Dataset", columns="Defense", values="K*")
    .reset_index()
    .rename_axis(None, axis=1)
).round(2)

print("Anonymity Sets")
print(anon_pivot)


# Joined
defense_order = ["1. HTTPOS", "2. LLaMA", "3. FRONT", "4. CL-Tamaraw", "5. H2PC"]
dataset_order = ["Amazon", "BBC", "Reddit", "Udemy", "Wikipedia"]

f1_lookup = ml_f1_results.set_index(["Dataset", "Defense"])["Value"].to_dict()
topk_lookup = ml_topk_results.set_index(["Dataset", "Defense"])["Value"].to_dict()
anon_lookup = anon_results.set_index(["Dataset", "Defense"])["K*"].to_dict()


def fmt(val, bold=False):
    if np.isnan(val):
        return "--"
    inner = f"{val:.2f}"
    return f"$\\mathbf{{{inner}}}$" if bold else f"${inner}$"


rows = []
rows.append(r"\begin{tabular}{@{}c@{\,}c@{\,}cccc@{}}")
rows.append(r"\toprule")
rows.append(
    r"\multicolumn{1}{c}{\textbf{Dataset}} & \textbf{Metric} "
    r"& \multicolumn{1}{c}{\textbf{HTTPOS}} & \multicolumn{1}{c}{\textbf{LLaMA}} "
    r"& \multicolumn{1}{c}{\textbf{FRONT}} & \multicolumn{1}{c}{\textbf{CL-TAM}} "
    r"& \multicolumn{1}{c}{\textbf{H2PC}} \\ \midrule"
)

for dataset in dataset_order:
    f1_vals = {d: f1_lookup.get((dataset, d), np.nan) for d in defense_order}
    topk_vals = {d: topk_lookup.get((dataset, d), np.nan) for d in defense_order}
    anon_vals = {d: anon_lookup.get((dataset, d), np.nan) for d in defense_order}

    # Best defense = lowest attack accuracy, highest anonymity set
    f1_best = min((v for v in f1_vals.values() if not np.isnan(v)), default=np.nan)
    topk_best = min((v for v in topk_vals.values() if not np.isnan(v)), default=np.nan)
    anon_best = max((v for v in anon_vals.values() if not np.isnan(v)), default=np.nan)

    f1_cells = " & ".join(fmt(f1_vals[d], f1_vals[d] == f1_best) for d in defense_order)
    topk_cells = " & ".join(
        fmt(topk_vals[d], topk_vals[d] == topk_best) for d in defense_order
    )
    anon_cells = " & ".join(
        fmt(anon_vals[d], anon_vals[d] == anon_best) for d in defense_order
    )

    rows.append(
        f"\\multirow{{3}}{{*}}{{{dataset}}} & Macro-F1          & {f1_cells} \\\\"
    )
    rows.append(
        f"                                   & Top-5             & {topk_cells} \\\\"
    )
    rows.append(
        f"                                   & $\\mathcal{{K}}^{{*}}$ & {anon_cells} \\\\ \\midrule"
    )

rows.append(r"\bottomrule")
rows.append(r"\end{tabular}")

print("\n".join(rows))
