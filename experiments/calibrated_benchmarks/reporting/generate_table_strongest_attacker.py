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
    "kfp": "k-FP",
    "varcnn": "VarCNN",
    "df": "DF",
    "holmes": "Holmes",
    "robustfp": "RobustFP",
}
PRETTY_DEFENSE = {
    "cldef_httpos": "HTTPOS",
    "cldef_llama": "LLaMA",
    "cldef_front": "FRONT",
    "cldef_tamaraw": "CL-Tamaraw",
    "cldef_h2pc": "H2PC",
    "srvdef_alpaca_html": "ALPaCA (1st)",
    "srvdef_alpaca_cdn1": "ALPaCA (CDN)",
    "srvdef_alpaca_cdn2": "ALPaCA (CDN2)",
    "srvdef_alpaca_all": "ALPaCA (all)",
    "srvdef_tamaraw_html": "SRV-TAM (1st)",
    "srvdef_tamaraw_cdn1": "SRV-TAM (CDN)",
    "srvdef_tamaraw_cdn2": "SRV-TAM (CDN2)",
    "srvdef_tamaraw_all": "SRV-TAM (all)",
    "srvdef_h2ps": "H2PS (1st)",
}
CLIENT_DEFENSES = [
    "cldef_httpos",
    "cldef_llama",
    "cldef_front",
    "cldef_tamaraw",
    "cldef_h2pc",
]
SERVER_DEFENSES = [d for d in PRETTY_DEFENSE if d.startswith("srvdef_")]


def side_of(defense):
    if defense in SERVER_DEFENSES:
        return "server"
    if defense in CLIENT_DEFENSES:
        return "client"
    return "other"


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
            side=side_of(r["defense"]),
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

df = df[df["side"] != "other"].copy()
client = df[df["side"] == "client"].copy()
server = df[df["side"] == "server"].copy()


def counts(sub, index):
    tab = sub.groupby([index, "winner"]).size().unstack(fill_value=0)
    tab = tab.reindex(columns=[m for m in ml_models if m in tab.columns])
    tab.columns = [PRETTY_MODEL.get(c, c) for c in tab.columns]
    tab["n"] = tab.sum(axis=1)
    return tab


for label, sub in (
    ("client defenses", client),
    ("server defenses", server),
    ("all defenses", df),
):
    if sub.empty:
        continue
    print("\n" + "=" * 70)
    print(f"Strongest attacker by {METRIC} -- wins per dataset ({label})")
    print("=" * 70)
    t = counts(sub, "dataset")
    t.index = [PRETTY_DATASET.get(i, i) for i in t.index]
    print(t.to_string())
    if label == "client defenses":
        per_ds = t

for label, sub in (("client defenses", client), ("server defenses", server)):
    if sub.empty:
        continue
    print("\n" + "=" * 70)
    print(f"Wins per defense ({label})")
    print("=" * 70)
    t = counts(sub, "defense")
    t.index = [PRETTY_DEFENSE.get(i, i) for i in t.index]
    print(t.to_string())

for label, sub in (
    ("client defenses", client),
    ("server defenses", server),
    ("all defenses", df),
):
    if sub.empty:
        continue
    print("\n" + "=" * 70)
    print(f"Global ({label})")
    print("=" * 70)
    tally = Counter(sub["winner"])
    total = sum(tally.values())
    for m, n in tally.most_common():
        print(f"  {PRETTY_MODEL.get(m, m):16s} {n:3d} / {total}  ({100*n/total:.0f}%)")
    print(
        "\n  mean margin over runner-up: "
        f"{sub['margin'].mean():.3f}  (median {sub['margin'].median():.3f})"
    )
    never = [
        PRETTY_MODEL.get(m, m)
        for m in ml_models
        if m not in tally and m in sub.columns and not sub[m].isna().all()
    ]
    if never:
        print(f"  never strongest: {', '.join(never)}")

print("\n" + "=" * 70)
print("Per cell: winner and its margin")
print("=" * 70)
show = df[["side", "dataset", "defense", "winner", "best", "margin", "n_models"]].copy()
show["dataset"] = show["dataset"].map(lambda x: PRETTY_DATASET.get(x, x))
show["defense"] = show["defense"].map(lambda x: PRETTY_DEFENSE.get(x, x))
show["winner"] = show["winner"].map(lambda x: PRETTY_MODEL.get(x, x))
print(show.sort_values(["side", "dataset", "defense"]).to_string(index=False))

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
