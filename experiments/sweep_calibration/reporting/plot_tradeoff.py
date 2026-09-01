#!/usr/bin/env python3
"""One figure per dataset: download overhead against attacker F1, per defense.

    python3 plot_tradeoff.py <csv> [<csv> ...] --out fig

Writes one file per dataset, inserting the dataset name before the extension.
Cai et al. CCS'14 Fig. 1 orientation: lower-left is a better defence.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

TICKS_FONT_SIZE = 16
FONT_SIZE = 18


LEVEL_ORDER = ["vlow", "low", "lomid", "mid1", "mid2", "high"]

PRETTY_DATASET = {
    "1_amazon": "Amazon",
    "2_bbc": "BBC",
    "3_reddit": "Reddit",
    "4_udemy": "Udemy",
    "5_wiki": "Wikipedia",
    "6_wiki": "Wikipedia",
}
PRETTY_DEFENSE = {
    "front": "FRONT",
    "h2pc": "H2PC",
    "tamaraw": "CL-Tamaraw",
    "httpos": "HTTPOS",
    "llama": "LLaMA",
    "srvalpaca": "ALPaCA",
    "srvtamaraw": "SRV-Tamaraw",
    "srvh2ps1p": "H2PS",
}
DEFENSE_ORDER = [
    "front",
    "h2pc",
    "tamaraw",
    "httpos",
    "llama",
    "srvalpaca",
    "srvtamaraw",
    "srvh2ps1p",
]

sns.set(style="whitegrid")

hue_palette = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange-yellow
    "#56B4E9",  # sky blue
    "#000000",  # black
]
linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), "-", "--"]
markers = ["o", "s", "^", "v", "D", "P", "X"]
BANDS = [
    (0.80, 1.00, "#d62728", "strong attacker"),
    (0.50, 0.80, "#ff7f0e", "moderate"),
    (0.10, 0.50, "#2ca02c", "weak attacker"),
]


RANDOM_GUESS = 0.01
YTICKS = [1, 2, 3, 4, 5, 10]

# shared cost range across datasets; grows if a defense exceeds it
Y_MIN, Y_MAX = 0.95, 10.0


def dataset_name(path):
    stem = Path(path).stem
    for suffix in ("_client", "_server"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def plot_one(path, out):
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"  skip {path}: {exc}")
        return False
    if df.empty or "defense" not in df:
        return False

    name = dataset_name(path)
    order = {lv: n for n, lv in enumerate(LEVEL_ORDER)}
    present = [d for d in DEFENSE_ORDER if d in set(df["defense"])]
    present += [
        d for d in sorted(set(df["defense"])) if d not in present and d != "nop"
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = 0

    for i, defense in enumerate(present):
        sub = df[df["defense"] == defense].copy()
        sub["_o"] = sub["level"].map(order)
        sub = sub.dropna(subset=["_o"]).sort_values("_o")
        sub["x"] = pd.to_numeric(sub["f1"], errors="coerce")
        sub["y"] = 1.0 + pd.to_numeric(sub["dDownB"], errors="coerce")
        sub["ci"] = pd.to_numeric(sub.get("f1_ci"), errors="coerce").fillna(0.0)
        sub = sub.dropna(subset=["x", "y"])
        if sub.empty:
            continue

        color = hue_palette[i % len(hue_palette)]
        ls = linestyles[i % len(linestyles)]
        mk = markers[i % len(markers)]

        ax.plot(
            sub["x"],
            sub["y"],
            marker=mk,
            linewidth=3,
            color=color,
            linestyle=ls,
            label=PRETTY_DEFENSE.get(defense, defense),
        )
        ax.fill_betweenx(
            sub["y"],
            sub["x"] - sub["ci"],
            sub["x"] + sub["ci"],
            color=color,
            alpha=0.15,
        )
        plotted += 1

    # undefended reference
    nop = df[df["defense"] == "nop"]
    if not nop.empty:
        f1 = pd.to_numeric(nop["f1"], errors="coerce").iloc[0]
        ax.plot(
            [f1],
            [1.0],
            marker="*",
            markersize=15,
            color="black",
            linestyle="none",
            label="undefended",
            zorder=5,
        )

    if not plotted:
        plt.close(fig)
        return False

    for lo, hi, color, label in BANDS:
        ax.axvspan(lo, hi, color=color, alpha=0.04, zorder=0)
        # ax.axvline(lo, color=color, alpha=0.5, linewidth=1, zorder=1)
        ax.axvline(lo, color="#999999", alpha=0.6, linewidth=1, zorder=1)
        # ax.text((lo + hi) / 2, 0.99, label, transform=ax.get_xaxis_transform(),
        #        ha="center", va="top", fontsize=9, color=color, alpha=0.8)
    ax.axvline(RANDOM_GUESS, color="grey", linestyle=":", linewidth=1, zorder=1)

    ax.set_yscale("log")
    ax.set_ylim(Y_MIN, max(Y_MAX, ax.get_ylim()[1]))
    lo, hi = ax.get_ylim()
    ticks = [t for t in YTICKS if lo <= t <= hi] or YTICKS[:4]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}x" for t in ticks])
    ax.minorticks_off()

    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Best Macro-F1  (lower = better defence)", fontsize=FONT_SIZE)
    ax.set_ylabel("Bandwidth Overhead Ratio", fontsize=FONT_SIZE)
    ax.set_title(PRETTY_DATASET.get(name, name), fontsize=FONT_SIZE)

    ax.tick_params(axis="both", which="major", labelsize=TICKS_FONT_SIZE)
    ax.tick_params(axis="both", which="minor", labelsize=TICKS_FONT_SIZE)

    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(loc="best", fontsize=TICKS_FONT_SIZE, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(f"{out}.png", bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"  wrote {out}  ({plotted} defenses)")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csvs", nargs="+")
    parser.add_argument(
        "--out",
        required=True,
        help="output path; the dataset name is inserted before "
        "the extension when several CSVs are given",
    )
    args = parser.parse_args()

    out = Path(args.out)
    for path in sorted(args.csvs):
        name = dataset_name(path)
        target = (
            out
            if len(args.csvs) == 1
            else out.with_name(
                f"{out.stem}_{PRETTY_DATASET.get(name, name)}{out.suffix}"
            )
        )
        plot_one(path, str(target))


if __name__ == "__main__":
    main()
