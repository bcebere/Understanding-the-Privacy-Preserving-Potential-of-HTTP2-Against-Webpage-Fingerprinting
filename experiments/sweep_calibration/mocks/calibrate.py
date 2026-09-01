#!/usr/bin/env python3
"""
Turns the runner's ovh_*.csv files into the overhead table (BBQ 4 format) and
the y-values for the trade-off figure.

    python calibrate.py --workspace workspace --baseline nop

Bandwidth overhead is self-normalising: send_requests() already reports both
the real-stream byte total and the all-stream total snapshotted at the moment
the last real stream ended, so dUp / dDown need no separate undefended run.
Latency does need one -- dT is computed against the baseline scenario, joined
per testcase.

Expects scenarios named "<defense>_<level>", which is what
client_defenses.levels.cells() prints.
"""

import argparse
import json
import os
from glob import glob
from pathlib import Path

import pandas as pd

LEVEL_ORDER = {
    "vlow": 0,
    "low": 1,
    "lomid": 2,
    "mid1": 3,
    "mid2": 4,
    "high": 5,
    "vhigh": 6,
    "vvhigh": 7,
}


def _ratio(num, den):
    """Relative increase ratio, paper's dM = (M_d - M_b) / M_b."""
    out = num / den - 1.0
    return out.where(den > 0)


def load(workspace: str) -> pd.DataFrame:
    frames = []
    for path in sorted(glob(os.path.join(workspace, "ovh_*.csv"))):
        df = pd.read_csv(path)
        if df.empty:
            print(f"  skip (empty): {path}")
            continue
        df["source"] = os.path.basename(path)
        frames.append(df)
    if not frames:
        raise SystemExit(f"no ovh_*.csv found in {workspace}")
    return pd.concat(frames, ignore_index=True)


def split_scenario(df: pd.DataFrame) -> pd.DataFrame:
    """Prefer the explicit defense/level columns the runner now writes.
    Fall back to splitting "<defense>_<level>" per row for older CSVs;
    scenarios without a recognised level suffix (e.g. "nop") are mid1."""
    df = df.copy()
    if "defense" in df and "level" in df and df["defense"].notna().all():
        return df

    def one(scenario):
        head, sep, tail = str(scenario).rpartition("_")
        if sep and tail in LEVEL_ORDER:
            return head, tail
        return str(scenario), "mid1"

    df[["defense", "level"]] = df["scenario"].map(one).apply(pd.Series)
    return df


def compute(df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    df = split_scenario(df)

    df["dUp"] = _ratio(df["bytes_tx_at_real_end"], df["bytes_tx_real"])
    df["dDown"] = _ratio(df["bytes_rx_at_real_end"], df["bytes_rx_real"])

    base = df[df["defense"] == baseline]
    if base.empty:
        print(f"  ! no '{baseline}' scenario -- dT and dDownB will be blank")
        df["dT"] = pd.NA
        df["dDownB"] = pd.NA
        df["dUpB"] = pd.NA
    else:
        ref = base.groupby("testcase").agg(
            latency_base=("latency", "median"),
            rx_base=("bytes_rx_at_real_end", "median"),
            tx_base=("bytes_tx_at_real_end", "median"),
        )
        df = df.merge(ref, on="testcase", how="left")
        df["dT"] = _ratio(df["latency"], df["latency_base"])
        # wire bytes vs undefended wire bytes; sees defenses whose extra bytes
        # ride on real streams (HTTPOS range overlap, header padding)
        df["dDownB"] = _ratio(df["bytes_rx_at_real_end"], df["rx_base"])
        df["dUpB"] = _ratio(df["bytes_tx_at_real_end"], df["tx_base"])

    # total bytes: the axis Cai et al. use ("transmission size")
    df["bytes_total_real"] = df["bytes_rx_real"] + df["bytes_tx_real"]
    df["bytes_total_at_end"] = df["bytes_rx_at_real_end"] + df["bytes_tx_at_real_end"]
    df["dTotal"] = _ratio(df["bytes_total_at_end"], df["bytes_total_real"])
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (defense, level), g in df.groupby(["defense", "level"]):
        row = {"defense": defense, "level": level, "n_pages": len(g)}
        for metric in ("dUp", "dDown", "dUpB", "dDownB", "dTotal", "dT"):
            s = pd.to_numeric(g[metric], errors="coerce").dropna()
            if s.empty:
                row[metric] = None
                row[f"{metric}_q1"] = row[f"{metric}_q3"] = None
                continue
            row[metric] = round(float(s.median()), 3)
            row[f"{metric}_q1"] = round(float(s.quantile(0.25)), 3)
            row[f"{metric}_q3"] = round(float(s.quantile(0.75)), 3)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["_o"] = out["level"].map(LEVEL_ORDER).fillna(99)
    return out.sort_values(["defense", "_o"]).drop(columns="_o").reset_index(drop=True)


def latex_table(s: pd.DataFrame) -> str:
    """BBQ 4 style: median (Q1 - Q3)."""

    def cell(r, m):
        if r[m] is None:
            return "--"
        return f"{r[m]:.2f} ({r[m + '_q1']:.2f} - {r[m + '_q3']:.2f})"

    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Defense & Level & $\Delta$Up & $\Delta$Down & $\Delta T$ \\",
        r"\midrule",
    ]
    for _, r in s.iterrows():
        lines.append(
            f"{r['defense']} & {r['level']} & {cell(r,'dUp')} & "
            f"{cell(r,'dDown')} & {cell(r,'dT')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default="workspace")
    ap.add_argument("--baseline", default="nop")
    ap.add_argument("--out", default="overhead_summary.csv")
    args = ap.parse_args()

    raw = load(args.workspace)
    df = compute(raw, args.baseline)
    s = summarise(df)

    cols = ["defense", "level", "n_pages", "dUp", "dDown", "dUpB", "dDownB", "dT"]
    print("\n" + s[cols].to_string(index=False))

    s.to_csv(Path(args.workspace) / args.out, index=False)
    print(f"\nwrote {args.out}")

    # y-values for the trade-off figure, keyed by (defense, level)
    yvals = {
        f"{r['defense']}:{r['level']}": {
            "dDown": r["dDown"],
            "dDownB": r["dDownB"],
            "dUpB": r["dUpB"],
            "dT": r["dT"],
        }
        for _, r in s.iterrows()
    }
    ypath = os.path.join(args.workspace, "tradeoff_yvalues.json")
    with open(ypath, "w") as fh:
        json.dump(yvals, fh, indent=2)
    print(f"wrote {ypath}  (paste into the figure's DATA dict)")

    print("\n" + latex_table(s))


if __name__ == "__main__":
    main()
