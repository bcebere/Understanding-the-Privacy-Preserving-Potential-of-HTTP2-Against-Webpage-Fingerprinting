#!/usr/bin/env python3
"""Server-defense resilience table: best-attacker F1, Top-5, and K*.

    python3 generate_table_serverdefs_perf.py

Prints one pivot per metric and the LaTeX body for
tab:server_defenses_realworld_f1_score.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from helpers import ml_models, security_results

workspace_name = Path.cwd().parent.name
WORKSPACE = f"/http2/experiments/{workspace_name}/"


DATASETS = ["1_amazon", "2_bbc", "3_reddit", "4_udemy", "5_wiki"]
PRETTY_DATASET = {
    "1_amazon": "Amazon",
    "2_bbc": "BBC",
    "3_reddit": "Reddit",
    "4_udemy": "Udemy",
    "5_wiki": "Wikipedia",
}
SHORT = {
    "Amazon": "Amz.",
    "BBC": "BBC",
    "Reddit": "Reddit",
    "Udemy": "Udemy",
    "Wikipedia": "Wiki",
}

# table columns, in order; groups are the deployments of one defense
COLS = [
    "srvdef_alpaca_html",
    "srvdef_alpaca_cdn1",
    "srvdef_alpaca_all",
    "srvdef_tamaraw_html",
    "srvdef_tamaraw_cdn1",
    "srvdef_tamaraw_all",
]
GROUPS = [(0, 1, 2), (3, 4, 5)]
DEFENSES = COLS + ["srvdef_h2ps"]

# (label, security_results key prefix, higher is more leakage)
METRICS = [
    ("F1", "f1_score_macro", False),
    ("Top-5", "acc_top5", False),
    (r"$\mathcal{K}^{*}$", None, True),  # K* comes from anon_cons
]

res = security_results(WORKSPACE)
res = res[res["dataset"].isin(DATASETS) & res["defense"].isin(DEFENSES)]
res = res.replace(PRETTY_DATASET)


def best_attacker(prefix):
    """-> DataFrame [Dataset, Defense, Model, Value, Error] for the strongest
    attacker per cell, by `prefix`."""
    rows = []
    for _, r in res.iterrows():
        for model in ml_models:
            mean, std = f"{prefix}_{model}_mean", f"{prefix}_{model}_std"
            if mean in r and not pd.isna(r[mean]):
                rows.append(
                    [r["dataset"], r["defense"], model, r[mean], r.get(std, np.nan)]
                )
    df = pd.DataFrame(
        rows, columns=["Dataset", "Defense", "Model", "Value", "Error"]
    ).dropna(subset=["Value"])
    if df.empty:
        return df
    return (
        df.loc[df.groupby(["Dataset", "Defense"])["Value"].idxmax()]
        .sort_values(["Dataset", "Defense"])
        .reset_index(drop=True)
    )


def pivot(df, values):
    return (
        df.pivot(index="Dataset", columns="Defense", values=values)
        .reset_index()
        .rename_axis(None, axis=1)
    )


def show(title, table, cols):
    print(f"\n{title}")
    present = ["Dataset"] + [c for c in cols if c in table.columns]
    print(table[present].to_string(index=False))


# ---- metric tables -------------------------------------------------------
numeric = {}

for label, prefix, _ in METRICS:
    if prefix is None:
        continue
    df = best_attacker(prefix)
    if df.empty:
        print(f"\n{label}: no scored cells")
        continue
    df["Value ± Error"] = df.apply(
        lambda r: (
            f"${r['Value']:.2f} \\pm {r['Error']:.2f}$"
            if not pd.isna(r["Error"])
            else f"${r['Value']:.2f}$"
        ),
        axis=1,
    )
    show(label, pivot(df, "Value ± Error"), DEFENSES)
    numeric[label] = pivot(df, "Value")

anon = res[["dataset", "defense", "anon_cons"]].dropna(subset=["anon_cons"])
anon = anon.rename(
    columns={"dataset": "Dataset", "defense": "Defense", "anon_cons": "Value"}
)
if not anon.empty:
    numeric[METRICS[-1][0]] = pivot(anon, "Value")
    show("K*", pivot(anon, "Value").round(2), DEFENSES)


# ---- latex ---------------------------------------------------------------
def marks(vals, higher_is_better):
    """-> (underline, bold) per column.

    One mark of each per row: underline the best single-server deployment
    across both defenses, bold the best of all deployments.
    """
    n = len(COLS)
    single, overall = [False] * n, [False] * n
    pick = max if higher_is_better else min

    valid = {i: v for i, v in enumerate(vals) if not np.isnan(v)}
    if not valid:
        return single, overall

    singles = {
        i: v for i, v in valid.items() if i in [c for g in GROUPS for c in g[:2]]
    }
    if singles:
        single[pick(singles, key=singles.get)] = True
    overall[pick(valid, key=valid.get)] = True

    return [s and not o for s, o in zip(single, overall)], overall


def cell(v, underline, bold):
    if np.isnan(v):
        return ""
    s = f"{v:.2f}"
    return (
        f"$\\mathbf{{{s}}}$"
        if bold
        else f"$\\underline{{{s}}}$"
        if underline
        else f"${s}$"
    )


def values_for(table, dataset):
    row = table[table["Dataset"] == dataset]
    return [
        (
            float(row[c].values[0])
            if not row.empty and c in row.columns and not pd.isna(row[c].values[0])
            else np.nan
        )
        for c in COLS
    ]


print("\n" + "=" * 70)
print("LATEX body")
print("=" * 70)
for i, key in enumerate(DATASETS):
    dataset = PRETTY_DATASET[key]
    print(f"    \\multirow{{3}}{{*}}{{{SHORT[dataset]}}}")
    for label, _, higher in METRICS:
        table = numeric.get(label)
        if table is None:
            continue
        vals = values_for(table, dataset)
        under, bold = marks(vals, higher)
        cells = " & ".join(cell(vals[j], under[j], bold[j]) for j in range(len(COLS)))
        print(f"{'':38s}& {label:<17} & {cells} \\\\")
    if i < len(DATASETS) - 1:
        print(r"    \midrule")
