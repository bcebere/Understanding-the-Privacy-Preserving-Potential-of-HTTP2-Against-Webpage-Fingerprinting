#!/usr/bin/env python3
"""Per-defense calibration figures: attacker F1 against cost, per dataset.
Writes two figures per defense -- bandwidth and latency -- each as .png+.pdf.

    python3 plot_defense.py <defense> <csv> [<csv> ...] --out fig
        -> fig_bandwidth.png/.pdf   x = bandwidth overhead ratio
        -> fig_latency.png/.pdf     x = latency overhead ratio

CSVs come from track_calibration_results.py --csv; the dataset name is taken
from the filename.  Points are ordered by defense intensity, not by cost, so a
level that costs more and defends worse shows as a visible backtrack.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

TICKS_FONT_SIZE = 16

X_MIN, X_MAX = -0.05, 4.0
FONT_SIZE = 18


LEVEL_ORDER = ["vlow", "low", "lomid", "mid1", "mid2", "high", "vhigh"]

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

RANDOM_GUESS = 0.01

# (column, filename suffix, axis label, ticks)
# HTTPOS is a request-side defense: range splitting and header padding both
# cost upload, and its download overhead is ~0 on every dataset.  Plotting
# dDownB for it would show a flat line at zero and hide the actual cost.
UPLOAD_DEFENSES = {"httpos"}

COSTS = [
    (
        "dDownB",
        "bandwidth",
        "Download Overhead compared to undefended traces",
        [0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25],
    ),
    (
        "dT",
        "latency",
        "Latency Overhead compared to undefended traces",
        [0, 0.5, 1, 2, 3, 4, 5, 7, 10],
    ),
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


def dataset_name(path):
    stem = Path(path).stem
    for suffix in ("_client", "_server"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def style_axes(ax, xlabel, ylabel, title):
    for lo, hi, color, label in BANDS:
        ax.axhspan(lo, hi, color=color, alpha=0.03, zorder=0)
        # ax.axhline(lo, color=color, alpha=0.3, linewidth=1, zorder=1)
        ax.axhline(lo, color="#999999", alpha=0.6, linewidth=1, zorder=1)

        # ax.text(
        #    0.995,
        #    (lo + hi) / 2,
        #    label,
        #    transform=ax.get_yaxis_transform(),
        #    ha="right",
        #    va="center",
        #    fontsize=TICKS_FONT_SIZE,
        #    color=color,
        #    alpha=0.8,
        # )
    ax.axhline(RANDOM_GUESS, color="grey", linestyle=":", linewidth=1, zorder=1)
    # ax.text(
    #    0.995,
    #    RANDOM_GUESS + 0.015,
    #    "random guess",
    #    transform=ax.get_yaxis_transform(),
    #    ha="right",
    #    va="bottom",
    #    fontsize=9,
    #    color="grey",
    # )
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.set_title(f"Calibration plot for the {title} defense", fontsize=FONT_SIZE)
    ax.set_ylim(0, 1.1)

    ax.tick_params(axis="both", which="major", labelsize=TICKS_FONT_SIZE)
    ax.tick_params(axis="both", which="minor", labelsize=TICKS_FONT_SIZE)

    ax.grid(alpha=0.25, linewidth=0.6)


def cost_for(defense, cost_col):
    """Which column and label to use, per defense."""
    if cost_col == "dDownB" and defense in UPLOAD_DEFENSES:
        return "dUpB", "Upload Overhead compared to undefended traces"
    return cost_col, None


def draw_cost(ax, args, frames, cost_col, xticks):
    """Draw one cost axis; returns how many datasets contributed a line."""
    cost_col, _ = cost_for(args.defense, cost_col)
    order = {lv: n for n, lv in enumerate(LEVEL_ORDER)}
    plotted = 0

    for i, (name, df) in enumerate(frames):
        sub = df[df["defense"] == args.defense].copy()
        if sub.empty or cost_col not in sub:
            continue

        sub["_o"] = sub["level"].map(order)
        sub = sub.dropna(subset=["_o"]).sort_values("_o")
        # raw overhead, not 1 + overhead: 0 is a meaningful point on the axis
        sub["x"] = pd.to_numeric(sub[cost_col], errors="coerce")
        sub["y"] = pd.to_numeric(sub[args.metric], errors="coerce")
        sub["ci"] = pd.to_numeric(sub.get("f1_ci"), errors="coerce").fillna(0.0)
        # a defense can load faster than baseline, so dT may be negative;
        # clip to 0 rather than dropping the level entirely
        sub = sub.dropna(subset=["x", "y"])
        sub["x"] = sub["x"].clip(lower=0.0)
        if sub.empty:
            continue

        color = hue_palette[i % len(hue_palette)]
        ls = linestyles[i % len(linestyles)]
        mk = markers[i % len(markers)]

        ax.plot(
            sub["x"],
            sub["y"],
            marker=mk,
            linewidth=1.5,
            color=color,
            linestyle=ls,
            label=PRETTY_DATASET.get(name, name),
        )
        ax.fill_between(
            sub["x"],
            sub["y"] - sub["ci"],
            sub["y"] + sub["ci"],
            color=color,
            alpha=0.15,
        )
        plotted += 1

    if not plotted:
        return 0

    # fixed span so defenses are comparable at a glance, even when one never
    # reaches the upper end
    ax.set_xlim(X_MIN, max(X_MAX, ax.get_xlim()[1]))
    lo, hi = ax.get_xlim()
    ticks = [t for t in xticks if lo <= t <= hi] or xticks[:4]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}x" for t in ticks])
    ax.minorticks_off()
    return plotted


def plot_cost(args, frames, cost_col, suffix, xlabel, xticks):
    """One figure per cost axis -- the individual plots."""
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = draw_cost(ax, args, frames, cost_col, xticks)
    if not plotted:
        print(f"  nothing to plot for {args.defense} / {suffix}")
        plt.close(fig)
        return

    _, alt = cost_for(args.defense, cost_col)
    style_axes(
        ax,
        alt or xlabel,
        "Macro-F1 (strongest attack)",
        PRETTY_DEFENSE.get(args.defense, args.defense),
    )
    ax.legend(loc="best", fontsize=TICKS_FONT_SIZE - 2, ncols=1, framealpha=0.9)

    fig.tight_layout()
    out = f"{args.out}_{suffix}"
    fig.savefig(f"{out}.png", bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}  ({plotted} datasets)")


def plot_stacked(args, frames):
    """Both cost axes in one figure: bandwidth on top, latency below.

    The y axis is shared -- it is the same attacker F1 in both panels -- so a
    reader can trace a level down and read off what it costs in each currency.
    """
    fig, axes = plt.subplots(len(COSTS), 1, figsize=(10, 9), sharey=True)
    drawn = 0
    for ax, (cost_col, _suffix, xlabel, xticks) in zip(axes, COSTS):
        n = draw_cost(ax, args, frames, cost_col, xticks)
        drawn += n
        if not n:
            ax.set_visible(False)
            continue
        _, alt = cost_for(args.defense, cost_col)
        style_axes(ax, alt or xlabel, "Macro-F1 (strongest attack)", "")
        ax.set_title("")

    if not drawn:
        print(f"  nothing to plot for {args.defense} / stacked")
        plt.close(fig)
        return

    visible = [a for a in axes if a.get_visible()]
    visible[0].set_title(
        f"Calibration plot for the "
        f"{PRETTY_DEFENSE.get(args.defense, args.defense)} defense",
        fontsize=FONT_SIZE,
    )
    visible[0].legend(loc="best", fontsize=TICKS_FONT_SIZE - 2, ncols=1, framealpha=0.9)

    fig.tight_layout()
    out = f"{args.out}_stacked"
    fig.savefig(f"{out}.png", bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}  ({drawn} lines)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("defense")
    parser.add_argument("csvs", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--metric", default="f1", help="f1 (default) or top5")
    args = parser.parse_args()

    frames = []
    for path in sorted(args.csvs):
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"  skip {path}: {exc}")
            continue
        if not df.empty and "defense" in df:
            frames.append((dataset_name(path), df))

    for cost_col, suffix, xlabel, xticks in COSTS:
        plot_cost(args, frames, cost_col, suffix, xlabel, xticks)
    plot_stacked(args, frames)


if __name__ == "__main__":
    main()
