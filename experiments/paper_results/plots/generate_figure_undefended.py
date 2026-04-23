from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from helpers import security_results

TICKS_FONT_SIZE = 16
FONT_SIZE = 18
sns.set(style="whitegrid")
prop_cycle = plt.rcParams["axes.prop_cycle"]
custom_colors = prop_cycle.by_key()["color"]
custom_colors.extend(
    [
        "#c5b0d5",
        "#ffc500",
        "#700548",
    ]
)
hue_palette = [
    custom_colors[0],
    custom_colors[11],
    custom_colors[3],
    custom_colors[12],
    custom_colors[1],
    custom_colors[10],
    custom_colors[8],
]
hatches = ["///", "\\\\\\", "||", "xx", "--", "ooo", "**"]
DIAGRAMS = Path("diagrams")
DIAGRAMS.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


###################################
# Load Data
####################################


workspace = Path.cwd().parent / "benchmarks"

enabled_datasets = ["1_amazon", "2_bbc", "3_reddit", "4_udemy", "5_wiki"]


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
    "robustfp": "RobustFP",
    "varcnn": "VarCNN",
    "kfpv2": "k-FP",
}

sec_results = security_results(workspace)
sec_results = sec_results[sec_results["dataset"].isin(enabled_datasets)]
sec_results = sec_results.replace(pretty_datasets).replace(pretty_models)
baseline_sec_results = sec_results[sec_results["defense"] == "undefended"]

ml_keys = ["dataset", "defense"]
ml_f1_results = []
for ridx, row in baseline_sec_results.iterrows():
    for model in ["kfpv2", "df", "varcnn", "holmes", "robustfp"]:
        row_info = [row["dataset"], model]
        for measure in ["mean", "std"]:
            model_key = f"f1_score_macro_{model}_{measure}"
            ml_keys.append(model_key)
            row_info.append(row[model_key])
        ml_f1_results.append(row_info)

ml_f1_results = (
    pd.DataFrame(ml_f1_results, columns=["Dataset", "Model", "Value", "Error"])
    .replace(pretty_datasets)
    .replace(pretty_models)
)
ml_f1_results["Value ± Error"] = ml_f1_results.apply(
    lambda row: f"{row['Value']:.6f} ± {row['Error']:.6f}", axis=1
)

print(ml_f1_results)


# DATA
uniq_dataset_cnt = len(np.unique(ml_f1_results["Dataset"].values))
uniq_models_cnt = len(np.unique(ml_f1_results["Model"].values))


# Plots
def plot_ml(
    data,
    ax=None,  # plt.gca(),
    show_legend=True,
    show_xlabel=True,
    font_size=FONT_SIZE,
    font_size_ticks=TICKS_FONT_SIZE,
    ylabel="Macro-F1",
    legend_bbox_to_anchor=(0.9, 1.17),
):
    figsize = (11, 5)

    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    datasets = data["Dataset"].unique()
    models = data["Model"].unique()

    x = np.arange(len(datasets))
    m = len(models)
    width = 0.85 / m  # total bar group width ~0.8

    for j, model in enumerate(models):
        sub = (
            data[data["Model"] == model]
            .set_index("Dataset")
            .loc[datasets]  # ensure order matches x
        )
        xj = x + (j - (m - 1) / 2) * width
        ax.bar(xj, sub["Value"], width, label=model)
        ax.errorbar(
            xj,
            sub["Value"],
            yerr=sub["Error"],
            fmt="none",
            ecolor="black",
            capsize=4,
            elinewidth=1,
        )

    for i, bar in enumerate(ax.patches):
        if bar.get_height() == 0:
            continue

        bar.set_edgecolor(hue_palette[int(i / uniq_dataset_cnt)])
        bar.set_hatch(hatches[int(i / uniq_dataset_cnt)])

        bar.set_facecolor("white")

    plt.tight_layout()

    if show_legend:
        legend = ax.legend(
            loc="upper right",
            ncols=uniq_models_cnt,
            fontsize=TICKS_FONT_SIZE - 1,
            bbox_to_anchor=legend_bbox_to_anchor,
        )
        handles = legend.legend_handles
        for i, handle in enumerate(handles):
            handle.set_hatch(hatches[i])
            handle.set_edgecolor(hue_palette[i])  # set_edgecolors
            handle.set_facecolor("white")
    else:
        ax.legend().remove()

    if show_xlabel:
        ax.set_xlabel("Dataset", fontsize=font_size)
        # ax.set_xticks(rotation=0, ha='center', fontsize = 14)
    else:
        ax.xaxis.set_ticks([])
        ax.set_xlabel("")

    ax.set_ylabel(ylabel, fontsize=font_size)
    # ax.set_yticks(rotation=0, ha='right', fontsize = font_size_ticks)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=0)
    ax.tick_params(axis="both", which="major", labelsize=font_size_ticks)
    ax.tick_params(axis="both", which="minor", labelsize=font_size_ticks)


# as plot
plot_ml(ml_f1_results, show_legend=True)

figname = "benchmarks_barplot_realworld_ml"
plt.savefig(DIAGRAMS / f"{figname}.pdf", bbox_inches="tight")
plt.savefig(DIAGRAMS / f"{figname}.png", bbox_inches="tight")
plt.show()
