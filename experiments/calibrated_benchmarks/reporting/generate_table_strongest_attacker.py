#!/usr/bin/env python3
"""Which attacker is strongest, per dataset and per defense.

    python3 generate_table_strongest_attacker.py
    python3 generate_table_strongest_attacker.py --metric acc_top5

The security tables report max-over-attackers F1 (BBQ 1), which hides which
attacker supplied it.  This counts how often each one wins, which is what
justifies keeping the full ensemble rather than a single model.
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from helpers import ml_models, security_results

METRIC = "f1_score_macro"
if "--metric" in sys.argv:
    METRIC = sys.argv[sys.argv.index("--metric") + 1]

_here = Path(__file__).parent  # dir of the (possibly symlinked) script
WORKSPACE = (
    Path(sys.argv[1])
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
    else _here / "workspace"
)


PRETTY_DATASET = {
    "1_amazon": "Amazon",
    "2_bbc": "BBC",
    "3_reddit": "Reddit",
    "4_udemy": "Udemy",
    "5_wiki": "Wikipedia",
}
PRETTY_MODEL = {
    "xgboost": "XGBoost",
    "kfp": "k-FP",
    "kfpv2": "k-FP",
    "varcnn": "VarCNN",
    "df": "DF",
    "holmes": "Holmes",
    "robustfp": "RobustFP",
}
PRETTY_DEFENSE = {
    "undefended": "Baseline",
    "cldef_httpos": "HTTPOS",
    "cldef_llama": "LLaMA",
    "cldef_front": "FRONT",
    "cldef_tamaraw": "CL-Tamaraw",
    "cldef_h2pc": "H2PC",
}
CLIENT_ONLY = [d for d in PRETTY_DEFENSE]

res = security_results(WORKSPACE)

rows = []
for _, r in res.iterrows():
    scores = {}
    for m in ml_models:
        key = f"{METRIC}_{m}_mean"
        if key in r and not pd.isna(r[key]):
            scores[m] = float(r[key])
    if not scores:
        continue
    winner = max(scores, key=scores.get)
    second = sorted(scores.values(), reverse=True)
    rows.append(
        dict(
            dataset=r["dataset"],
            defense=r["defense"],
            winner=winner,
            best=scores[winner],
            margin=(second[0] - second[1]) if len(second) > 1 else np.nan,
            n_models=len(scores),
            **{m: scores.get(m, np.nan) for m in ml_models},
        )
    )

df = pd.DataFrame(rows)
if df.empty:
    sys.exit("no scored cells found")

client = df[df["defense"].isin(CLIENT_ONLY)].copy()


def counts(sub, index):
    tab = sub.groupby([index, "winner"]).size().unstack(fill_value=0)
    tab = tab.reindex(columns=[m for m in ml_models if m in tab.columns])
    tab.columns = [PRETTY_MODEL.get(c, c) for c in tab.columns]
    tab["n"] = tab.sum(axis=1)
    return tab


print("=" * 70)
print(f"Strongest attacker by {METRIC} -- wins per dataset (client defenses)")
print("=" * 70)
per_ds = counts(client, "dataset")
per_ds.index = [PRETTY_DATASET.get(i, i) for i in per_ds.index]
print(per_ds.to_string())

print("\n" + "=" * 70)
print("Wins per defense")
print("=" * 70)
per_def = counts(client, "defense")
per_def.index = [PRETTY_DEFENSE.get(i, i) for i in per_def.index]
print(per_def.to_string())

print("\n" + "=" * 70)
print("Global (client defenses)")
print("=" * 70)
tally = Counter(client["winner"])
total = sum(tally.values())
for m, n in tally.most_common():
    print(f"  {PRETTY_MODEL.get(m, m):16s} {n:3d} / {total}  ({100*n/total:.0f}%)")

print(
    "\n  mean margin over runner-up: "
    f"{client['margin'].mean():.3f}  (median {client['margin'].median():.3f})"
)
never = [
    PRETTY_MODEL.get(m, m)
    for m in ml_models
    if m not in tally and any(not pd.isna(client[m]).all() for _ in [0])
]
if never:
    print(f"  never strongest: {', '.join(never)}")

print("\n" + "=" * 70)
print("Per cell: winner and its margin")
print("=" * 70)
show = client[["dataset", "defense", "winner", "best", "margin", "n_models"]].copy()
show["dataset"] = show["dataset"].map(lambda x: PRETTY_DATASET.get(x, x))
show["defense"] = show["defense"].map(lambda x: PRETTY_DEFENSE.get(x, x))
show["winner"] = show["winner"].map(lambda x: PRETTY_MODEL.get(x, x))
print(show.sort_values(["dataset", "defense"]).to_string(index=False))

# ---- latex ---------------------------------------------------------------
print("\n" + "=" * 70)
print("LATEX (wins per dataset)")
print("=" * 70)
cols = [c for c in per_ds.columns if c != "n"]
print(r"\begin{tabular}{l" + "r" * len(cols) + "}")
print(r"\toprule")
print("\\textbf{Dataset} & " + " & ".join(cols) + r" \\")
print(r"\midrule")
for idx, row in per_ds.iterrows():
    print(f"{idx} & " + " & ".join(str(int(row[c])) for c in cols) + r" \\")
print(r"\midrule")
tot = per_ds[cols].sum()
print(
    r"\textbf{Total} & "
    + " & ".join(f"\\textbf{{{int(tot[c])}}}" for c in cols)
    + r" \\"
)
print(r"\bottomrule")
print(r"\end{tabular}")
