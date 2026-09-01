#!/usr/bin/env python3
"""Overhead tables at the selected calibration level.

    python3 generate_table_overheads.py

Emits two tables:
  * aggregated -- per-page overheads pooled across datasets, median (Q1--Q3),
    which is the main-paper format
  * per dataset -- the same statistic per dataset, for the appendix

Each dataset contributes the cell main_table_config selected for it, so the
pooled distribution mixes levels by design: it answers "what does this defense
cost as deployed", not "what does level L cost".  Overheads are recomputed
per page from ovh_<defense>_<level>.csv against that dataset's own baseline,
since the summary only carries pre-aggregated quantiles.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path.cwd().parent / "mocks"))
from main_table_config import CLIENT, SERVER  # noqa: E402

BASE = Path("/http2/experiments/sweep_calibration")
DATASETS = ["1_amazon", "2_bbc", "3_reddit", "4_udemy", "5_wiki"]

PRETTY_DATASET = {
    "1_amazon": "Amazon",
    "2_bbc": "BBC",
    "3_reddit": "Reddit",
    "4_udemy": "Udemy",
    "5_wiki": "Wikipedia",
}
PRETTY_DEFENSE = {
    "httpos": "HTTPOS",
    "llama": "LLaMA",
    "front": "FRONT",
    "tamaraw": "CL-TAM",
    "h2pc": "H2PC",
    "srvalpaca": "ALPaCA",
    "srvtamaraw": "SRV-TAM",
    "srvh2ps1p": "H2PS",
    "srvalpaca_1st": "ALPaCA (1st)",
    "srvalpaca_3rd_1": "ALPaCA (CDN)",
    "srvalpaca_all": "ALPaCA (all)",
    "srvtamaraw_1st": "SRV-TAM (1st)",
    "srvtamaraw_3rd_1": "SRV-TAM (CDN)",
    "srvtamaraw_all": "SRV-TAM (all)",
    "srvh2ps_1st": "H2PS (1st)",
}
CLIENT_ORDER = ["httpos", "llama", "front", "tamaraw", "h2pc"]
SERVER_ORDER = [
    "srvalpaca_1st",
    "srvalpaca_3rd_1",
    "srvalpaca_all",
    "srvtamaraw_1st",
    "srvtamaraw_3rd_1",
    "srvtamaraw_all",
    "srvh2ps_1st",
]
SERVER_KEY = {"srvalpaca": "alpaca", "srvtamaraw": "tamaraw", "srvh2ps": "h2ps"}


def selected_level(dataset, defense):
    for prefix, short in SERVER_KEY.items():
        if defense.startswith(prefix):
            return SERVER.get(dataset, {}).get(short)
    return CLIENT.get(dataset, {}).get(defense)


def load_csv(path):
    df = pd.read_csv(path, on_bad_lines="skip")
    df = df[df["testcase"] != "testcase"]
    for col in ("bytes_rx_at_real_end", "bytes_tx_at_real_end", "latency"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def per_page_overhead(baseline, defended):
    """-> per-page relative increase, joined on testcase."""
    cols = ["testcase", "latency", "bytes_rx_at_real_end", "bytes_tx_at_real_end"]
    base = baseline[cols].groupby("testcase").median().add_suffix("_base")
    deff = defended[cols].groupby("testcase").median().add_suffix("_def")
    m = base.join(deff, how="inner").reset_index()
    out = pd.DataFrame({"testcase": m["testcase"]})
    for name, col in (
        ("delta_up", "bytes_tx_at_real_end"),
        ("delta_down", "bytes_rx_at_real_end"),
        ("delta_T", "latency"),
    ):
        out[name] = (m[f"{col}_def"] - m[f"{col}_base"]) / m[f"{col}_base"]
    return out


def stat(series):
    s = series.dropna()
    if s.empty:
        return None
    return s.median(), s.quantile(0.25), s.quantile(0.75)


def fmt(t):
    if t is None:
        return "--"
    return f"${t[0]:.2f}\\ ({t[1]:.2f} - {t[2]:.2f})$"


# ---- collect -------------------------------------------------------------
pooled = {d: [] for d in CLIENT_ORDER + SERVER_ORDER}
per_ds = {}
baselines = []

for dataset in DATASETS:
    ovh = BASE / dataset / "overhead"
    bpath = ovh / "ovh_baseline.csv"
    if not bpath.exists():
        print(f"WARNING: no baseline for {dataset}")
        continue
    baseline = load_csv(bpath)
    baselines.append(baseline)

    for defense in CLIENT_ORDER + SERVER_ORDER:
        level = selected_level(dataset, defense)
        if level is None:
            continue
        path = ovh / f"ovh_{defense}_{level}.csv"
        if not path.exists():
            print(f"WARNING: missing {path.name} for {dataset}")
            continue
        o = per_page_overhead(baseline, load_csv(path))
        pooled[defense].append(o)
        per_ds[(dataset, defense)] = (level, o)

# ---- aggregated (main paper) --------------------------------------------
print("\n" + "=" * 66)
print("AGGREGATED across datasets, at each dataset's selected level")
print("=" * 66)
print(f"{'Defense':10s} {'n':>5s}  {'dUp':>22s} {'dDown':>22s} {'dT':>22s}")
agg = {}
for defense in CLIENT_ORDER + SERVER_ORDER:
    if not pooled[defense]:
        continue
    df = pd.concat(pooled[defense], ignore_index=True)
    agg[defense] = {
        k: stat(df[k].clip(lower=0)) for k in ("delta_up", "delta_down", "delta_T")
    }
    print(
        f"{PRETTY_DEFENSE[defense]:10s} {len(df):5d}  "
        f"{fmt(agg[defense]['delta_up']):>22s} "
        f"{fmt(agg[defense]['delta_down']):>22s} "
        f"{fmt(agg[defense]['delta_T']):>22s}"
    )

if baselines:
    b = pd.concat(baselines, ignore_index=True)
    up_kb = b["bytes_tx_at_real_end"].mean() / 1024
    down_kb = b["bytes_rx_at_real_end"].mean() / 1024
    lat = b["latency"].mean()
    print(f"\nBaseline avg.  Up={up_kb:.2f} KB  Down={down_kb:.2f} KB  T={lat:.2f} s")

# ---- per dataset (appendix) ---------------------------------------------
print("\n" + "=" * 66)
print("PER DATASET")
print("=" * 66)
for dataset in DATASETS:
    rows = [
        (d, *per_ds[(dataset, d)])
        for d in CLIENT_ORDER + SERVER_ORDER
        if (dataset, d) in per_ds
    ]
    if not rows:
        continue
    print(f"\n{PRETTY_DATASET[dataset]}")
    for defense, level, o in rows:
        s = {k: stat(o[k].clip(lower=0)) for k in ("delta_up", "delta_down", "delta_T")}
        print(
            f"  {PRETTY_DEFENSE[defense]:10s} {level:6s} "
            f"{fmt(s['delta_up']):>22s} {fmt(s['delta_down']):>22s} "
            f"{fmt(s['delta_T']):>22s}"
        )

# ---- latex, aggregated ---------------------------------------------------
print("\n" + "=" * 66)
print("LATEX (aggregated)")
print("=" * 66)
print(
    r"""\begin{tabular}{@{}lrrr@{}}
\toprule
\textbf{Defense} &
\multicolumn{1}{c}{$\Delta\text{Up}$} &
\multicolumn{1}{c}{$\Delta\text{Down}$} &
\multicolumn{1}{c}{$\Delta T$} \\
\midrule"""
)
for defense in CLIENT_ORDER + SERVER_ORDER:
    if defense not in agg:
        continue
    if defense == SERVER_ORDER[0]:
        print(r"\midrule")
    a = agg[defense]
    print(
        f"{PRETTY_DEFENSE[defense]:12s} & {fmt(a['delta_up'])} & "
        f"{fmt(a['delta_down'])} & {fmt(a['delta_T'])} \\\\"
    )
if baselines:
    print(r"\midrule")
    print(
        f"Baseline avg. & \\multicolumn{{1}}{{c}}{{${up_kb:.2f}$ KB}} & "
        f"\\multicolumn{{1}}{{c}}{{${down_kb:.1f}$ KB}} & "
        f"\\multicolumn{{1}}{{c}}{{${lat:.2f}$ s}} \\\\"
    )
print(r"\bottomrule")
print(r"\end{tabular}")
