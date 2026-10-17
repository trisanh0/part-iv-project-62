#!/usr/bin/env python3
"""Generate publication-grade Figure 2 (Data Leakage Gap vs Temporal Overlap).

Reads empirical summary data from:
  benchmark_results/TM26a19__followups/TM26a19__04_leakage_vs_overlap/leakage_summary_table.csv
Generates high-contrast, publication-styled figure matching the TEMPO design language and saves to:
  1. benchmark_results/TM26a19__followups/TM26a19__04_leakage_vs_overlap/gap_balanced_acc_vs_overlap.png
  2. report/figures/fig_leakage_vs_overlap.png
"""

import ast
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def generate_figure() -> None:
    csv_path = Path("benchmark_results/TM26a19__followups/TM26a19__04_leakage_vs_overlap/leakage_summary_table.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Required summary data not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # 1:1 publication scale matching 12pt Times New Roman in LaTeX report
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11.0,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 8.5,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.85,
    })

    overlaps = df["overlap_pct"].tolist()
    bacc_gaps = df["bacc_gap_mean"].tolist()
    acc_gaps = df["acc_gap_mean"].tolist()

    # Parse 95% Student-t confidence intervals
    bacc_ci = [ast.literal_eval(x) if isinstance(x, str) else x for x in df["bacc_gap_ci95"]]
    acc_ci = [ast.literal_eval(x) if isinstance(x, str) else x for x in df["acc_gap_ci95"]]

    bacc_err_low = [m - ci[0] for m, ci in zip(bacc_gaps, bacc_ci)]
    bacc_err_high = [ci[1] - m for m, ci in zip(bacc_gaps, bacc_ci)]

    acc_err_low = [m - ci[0] for m, ci in zip(acc_gaps, acc_ci)]
    acc_err_high = [ci[1] - m for m, ci in zip(acc_gaps, acc_ci)]

    # Figure dimensions tailored to 0.70\\linewidth (~4.41 in) in LaTeX for true 1:1 body scale
    fig, ax = plt.subplots(figsize=(4.8, 3.2), dpi=300)

    # Okabe-Ito colourblind-safe palette:
    # Vermilion for Balanced Accuracy Gap (primary effect)
    # Deep Blue for Standard Accuracy Gap (reference baseline)
    c_bacc = "#D55E00"  # Vermilion
    c_acc = "#0072B2"   # Deep Blue

    # Baseline zero leakage line
    ax.axhline(
        0.0,
        color="#777777",
        linestyle=":",
        linewidth=1.0,
        alpha=0.85,
        label=r"Zero leakage baseline ($\Delta = 0$)",
        zorder=1,
    )

    # Balanced Accuracy Gap with 95% CIs
    ax.errorbar(
        overlaps,
        bacc_gaps,
        yerr=[bacc_err_low, bacc_err_high],
        fmt="o-",
        color=c_bacc,
        linewidth=1.6,
        markersize=5.2,
        markerfacecolor=c_bacc,
        markeredgecolor="#222222",
        markeredgewidth=0.4,
        capsize=3.5,
        capthick=1.1,
        label=r"Balanced Accuracy Gap ($\Delta_{\mathrm{BACC}}$)",
        zorder=3,
    )

    # Standard Accuracy Gap with 95% CIs
    ax.errorbar(
        overlaps,
        acc_gaps,
        yerr=[acc_err_low, acc_err_high],
        fmt="s--",
        color=c_acc,
        linewidth=1.4,
        markersize=4.6,
        markerfacecolor="none",
        markeredgecolor=c_acc,
        markeredgewidth=1.2,
        capsize=3.0,
        capthick=1.0,
        label=r"Standard Accuracy Gap ($\Delta_{\mathrm{Acc}}$)",
        zorder=2,
    )

    # Explicit discrete evaluation ticks
    ax.set_xticks([0, 25, 50, 75, 90])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "90%"])
    ax.set_xlim(-4, 94)

    # Y-axis ticks and limits
    ax.set_ylim(-0.02, 0.28)
    ax.set_yticks([0.00, 0.05, 0.10, 0.15, 0.20, 0.25])
    ax.set_yticklabels(["0.00", "0.05", "0.10", "0.15", "0.20", "0.25"])

    # Clean mathematical axis labels
    ax.set_xlabel(r"Window Overlap Ratio (%) [$W=100$, Stride $S \in \{100, 75, 50, 25, 10\}$]", fontweight="bold")
    ax.set_ylabel(r"Cross-Validation Leakage Gap ($\Delta$)", fontweight="bold")

    # Clean subtle grid
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, color="#e0e0e0", alpha=0.85)

    # Floating borderless legend in the clear upper-left region
    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.88,
        edgecolor="none",
        borderpad=0.35,
        labelspacing=0.25,
    )

    plt.tight_layout(pad=0.6)

    out_benchmark = Path("benchmark_results/TM26a19__followups/TM26a19__04_leakage_vs_overlap/gap_balanced_acc_vs_overlap.png")
    out_report = Path("report/figures/fig_leakage_vs_overlap.png")

    out_benchmark.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_benchmark, dpi=300)
    fig.savefig(out_report, dpi=300)
    plt.close(fig)
    print(f"Saved publication Figure 2 to:\n  - {out_benchmark}\n  - {out_report}")


if __name__ == "__main__":
    generate_figure()
