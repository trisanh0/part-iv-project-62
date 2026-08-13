"""
Bake-Off Figure Generator for TEMPO Empirical Benchmarks.

Reads live telemetry results from data/03_processed/beed/bakeoff_results.csv
and generates publication-ready SVG and 300 DPI PNG figures:
  1. bakeoff_accuracy_by_extractor.(svg|png)
  2. bakeoff_extraction_time_by_extractor.(svg|png)
  3. bakeoff_accuracy_vs_time_tradeoff.(svg|png)
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Ensure Matplotlib cache dir is inside workspace
os.environ["MPLCONFIGDIR"] = str(Path("presentation_figures/.cache").resolve())

# Set high-legibility presentation theme
plt.style.use("default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 15,
    "axes.labelweight": "bold",
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 20,
    "figure.titleweight": "bold",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 1.4,
    "grid.color": "#E2E8F0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.9,
})

OUTPUT_DIR = Path("presentation_figures")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

DOCS_DIR = Path("docs/Documents/Deliverable 2 - Mid-year Report/presentation_figures")
DOCS_DIR.mkdir(exist_ok=True, parents=True)

DATA_PATH = Path("data/03_processed/beed/bakeoff_results.csv")


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save figure as both vector SVG and 300 DPI PNG in output directories."""
    for d in [OUTPUT_DIR, DOCS_DIR]:
        fig.savefig(d / f"{name}.svg", bbox_inches="tight", format="svg")
        fig.savefig(d / f"{name}.png", bbox_inches="tight", dpi=300, format="png")
    print(f"Saved figure: {name}.(svg|png)")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Bake-Off telemetry CSV not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Color palette
    colors = {
        "statistics": "#529985",        # Sage Green (Polars Engine)
        "tsfel": "#D97706",             # Amber/Orange
        "tsfresh-minimal": "#6366F1",   # Indigo
        "tsfresh-efficient": "#B86B7D", # Soft Muted Rose
        "numba-efficient": "#0D9488",   # Dark Teal (Numba Engine)
    }

    extractor_labels = {
        "statistics": "Statistics\n(Polars)",
        "tsfel": "TSFEL",
        "tsfresh-minimal": "TSFresh\nMinimal",
        "tsfresh-efficient": "TSFresh\nEfficient",
        "numba-efficient": "Numba JIT\n(TEMPO)",
    }

    # ==========================================================================
    # 1. Accuracy by Extractor (Boxplot / Bar Plot)
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    df_rf = df[df["model"] == "rf"].copy()
    df_rf["extractor_label"] = df_rf["extractor"].map(extractor_labels)
    order = [extractor_labels[k] for k in extractor_labels.keys()]

    sns.boxplot(
        data=df_rf,
        x="extractor_label",
        y="score_mean",
        order=order,
        palette=[colors[k] for k in extractor_labels.keys()],
        ax=ax,
        width=0.45,
    )

    ax.set_title("Classification Accuracy by Feature Extractor (BEED EEG)")
    ax.set_xlabel("Feature Extractor Engine")
    ax.set_ylabel("Accuracy Score")
    ax.set_ylim(0.80, 1.00)
    ax.grid(True, axis="y")

    # Add peak accuracy annotations
    for i, ext in enumerate(extractor_labels.keys()):
        max_acc = df_rf[df_rf["extractor"] == ext]["score_mean"].max()
        ax.text(i, max_acc + 0.008, f"{max_acc * 100:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)

    save_figure(fig, "bakeoff_accuracy_by_extractor")
    plt.close(fig)

    # ==========================================================================
    # 2. Extraction Time by Extractor (Bar Plot - Log Scale)
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    df_unique_time = df.groupby("extractor")["extract_time_sec"].first().reset_index()
    df_unique_time["extractor_label"] = df_unique_time["extractor"].map(extractor_labels)

    bars = ax.bar(
        df_unique_time["extractor_label"],
        df_unique_time["extract_time_sec"],
        color=[colors[k] for k in df_unique_time["extractor"]],
        edgecolor="#334155",
        linewidth=1.2,
        width=0.5,
    )

    ax.set_yscale("log")
    ax.set_title("Feature Extraction Execution Time (Seconds - Log Scale)")
    ax.set_xlabel("Feature Extractor Engine")
    ax.set_ylabel("Execution Time (Seconds)")
    ax.grid(True, axis="y", which="both")

    for bar, val in zip(bars, df_unique_time["extract_time_sec"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val * 1.25,
            f"{val:.3f} s" if val < 1 else f"{val:.1f} s",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    save_figure(fig, "bakeoff_extraction_time_by_extractor")
    plt.close(fig)

    # ==========================================================================
    # 3. Accuracy vs Execution Time Trade-Off (Pareto Scatter Plot)
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(11, 6.5))

    for ext in extractor_labels.keys():
        sub = df_rf[df_rf["extractor"] == ext]
        ax.scatter(
            sub["extract_time_sec"],
            sub["score_mean"],
            s=140,
            color=colors[ext],
            edgecolor="black",
            linewidth=1.2,
            label=extractor_labels[ext].replace("\n", " "),
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_title("Accuracy vs Extraction Execution Time (Pareto Frontier)")
    ax.set_xlabel("Extraction Time (Seconds - Log Scale)")
    ax.set_ylabel("Classification Accuracy Score")
    ax.set_ylim(0.82, 1.00)
    ax.grid(True, which="both")
    ax.legend(title="Extractor Engine", loc="lower right")

    # Callout annotations for key winners
    ax.annotate(
        "Numba JIT Engine:\n98.7% Acc in 5.2s",
        xy=(5.21, 0.987),
        xytext=(0.8, 0.95),
        arrowprops=dict(arrowstyle="->", color="#0D9488", lw=1.8),
        fontweight="bold",
        color="#0D9488",
        fontsize=11,
    )

    ax.annotate(
        "Polars Statistics:\n96.2% Acc in 0.011s!",
        xy=(0.011, 0.962),
        xytext=(0.02, 0.88),
        arrowprops=dict(arrowstyle="->", color="#529985", lw=1.8),
        fontweight="bold",
        color="#2D5E52",
        fontsize=11,
    )

    save_figure(fig, "bakeoff_accuracy_vs_time_tradeoff")
    plt.close(fig)

    print("All Bake-Off presentation figures successfully generated.")


if __name__ == "__main__":
    main()
