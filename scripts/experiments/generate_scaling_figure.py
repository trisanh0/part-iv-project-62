#!/usr/bin/env python3
"""Generate publication-grade Figure 3 (Extractor Scaling Curves & Memory Dynamics).

Reads empirical summary data from:
  benchmark_results/TM26a19__followups/TM26a19__06_scaling_curves/scaling_medians_summary.csv
Generates high-contrast, decoupled-legend two-panel figure and saves to:
  1. benchmark_results/TM26a19__followups/TM26a19__06_scaling_curves/scaling_curves_call1_call2.png
  2. report/figures/fig_extractor_scaling_curves.png
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd


def generate_figure() -> None:
    csv_path = Path("benchmark_results/TM26a19__followups/TM26a19__06_scaling_curves/scaling_medians_summary.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Required summary data not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Publication scale and typography matching 12pt Times New Roman in LaTeX report
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.labelsize": 11.0,
        "axes.titlesize": 11.5,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 9.2,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.85,
    })

    extractors = [
        "polars_statistics",
        "numpy_statistical",
        "numba_efficient",
        "tsfel",
        "tsfresh_efficient",
    ]

    cfg = {
        "polars_statistics": {
            "name": r"$\mathtt{polars\_statistics}$",
            "color": "#0072B2",  # Deep Blue
            "marker": "o",
        },
        "numpy_statistical": {
            "name": r"$\mathtt{numpy\_statistical}$",
            "color": "#009E73",  # Green
            "marker": "s",
        },
        "numba_efficient": {
            "name": r"$\mathtt{numba\_efficient}$",
            "color": "#E69F00",  # Amber / Orange
            "marker": "^",
        },
        "tsfel": {
            "name": r"$\mathtt{tsfel}$",
            "color": "#CC79A7",  # Muted Purple
            "marker": "v",
        },
        "tsfresh_efficient": {
            "name": r"$\mathtt{tsfresh\_efficient}$",
            "color": "#D55E00",  # Vermilion / Red
            "marker": "D",
        },
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 3.25), dpi=300)

    for ext in extractors:
        sub = df[df["extractor"] == ext].sort_values("n_windows")
        if sub.empty:
            continue
        c = cfg[ext]["color"]
        m = cfg[ext]["marker"]

        # Call 1 (Cold): dashed line, hollow marker
        ax1.plot(
            sub["n_windows"],
            sub["median_call1_s"],
            linestyle="--",
            linewidth=1.4,
            color=c,
            marker=m,
            markersize=5.2,
            markerfacecolor="none",
            markeredgewidth=1.2,
            markeredgecolor=c,
            alpha=0.85,
            zorder=3,
        )

        # Call 2 (Warm): solid line, filled marker
        ax1.plot(
            sub["n_windows"],
            sub["median_call2_s"],
            linestyle="-",
            linewidth=1.8,
            color=c,
            marker=m,
            markersize=5.2,
            markerfacecolor=c,
            markeredgewidth=0.4,
            markeredgecolor="#222222",
            alpha=0.95,
            zorder=4,
        )

        # Right panel: Net Resident Memory (Log-Log)
        ax2.plot(
            sub["n_windows"],
            sub["median_net_rss_mb"],
            linestyle="-",
            linewidth=1.8,
            color=c,
            marker=m,
            markersize=5.2,
            markerfacecolor=c,
            markeredgewidth=0.4,
            markeredgecolor="#222222",
            alpha=0.95,
            zorder=4,
        )

    # Panel (a) Runtime Scaling (Log-Log)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xticks([50, 100, 200, 400, 800, 1600])
    ax1.set_xticklabels(["50", "100", "200", "400", "800", "1600"])
    ax1.set_xlabel(r"Number of Windows $N$ ($W=100, p=3$)", fontweight="bold")
    ax1.set_ylabel(r"Execution Time (seconds, log scale)", fontweight="bold")
    ax1.set_title(r"(a) Runtime Scaling: Cold vs Warm", fontweight="bold", pad=6)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5, color="#e0e0e0", alpha=0.85)

    # Panel (b) Net Resident Memory Scaling (Log-Log)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xticks([50, 100, 200, 400, 800, 1600])
    ax2.set_xticklabels(["50", "100", "200", "400", "800", "1600"])
    ax2.set_ylim(0.7, 3000.0)
    ax2.set_yticks([1, 10, 100, 1000])
    ax2.set_yticklabels(["1", "10", "100", "1000"])
    ax2.set_xlabel(r"Number of Windows $N$ ($W=100, p=3$)", fontweight="bold")
    ax2.set_ylabel(r"Net Resident Memory (MB, log scale)", fontweight="bold")
    ax2.set_title(r"(b) Net Resident Memory Scaling", fontweight="bold", pad=6)
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5, color="#e0e0e0", alpha=0.85)

    # Top unified decoupled legend
    ext_handles = [
        mlines.Line2D(
            [], [],
            color=cfg[ext]["color"],
            marker=cfg[ext]["marker"],
            linestyle="-",
            linewidth=1.6,
            markersize=5.2,
            markerfacecolor=cfg[ext]["color"],
            markeredgecolor="#222222",
            markeredgewidth=0.4,
            label=cfg[ext]["name"]
        ) for ext in extractors
    ]
    regime_handles = [
        mlines.Line2D(
            [], [],
            color="#444444",
            linestyle="--",
            linewidth=1.4,
            marker="o",
            markersize=5.0,
            markerfacecolor="none",
            markeredgewidth=1.2,
            label="Call 1 (Cold)"
        ),
        mlines.Line2D(
            [], [],
            color="#444444",
            linestyle="-",
            linewidth=1.8,
            marker="o",
            markersize=5.0,
            markerfacecolor="#444444",
            label="Call 2 (Warm)"
        ),
    ]

    leg1 = fig.legend(
        handles=ext_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=5,
        frameon=False,
        fontsize=9.2,
        handletextpad=0.32,
        columnspacing=0.6,
    )
    leg2 = fig.legend(
        handles=regime_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.962),
        ncol=2,
        frameon=False,
        fontsize=9.2,
        handletextpad=0.45,
        columnspacing=2.0,
    )
    fig.add_artist(leg1)

    plt.tight_layout(rect=[0, 0, 1, 0.89], w_pad=2.2)

    out1 = Path("benchmark_results/TM26a19__followups/TM26a19__06_scaling_curves/scaling_curves_call1_call2.png")
    out2 = Path("report/figures/fig_extractor_scaling_curves.png")

    fig.savefig(out1, dpi=300)
    fig.savefig(out2, dpi=300)
    plt.close(fig)
    print(f"Successfully generated scaling figure:\n  -> {out1}\n  -> {out2}")


if __name__ == "__main__":
    generate_figure()
