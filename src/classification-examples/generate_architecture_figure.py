"""
Standardised Data Architecture Visual Generator for Slide 11.

Generates presentation-ready figures (SVG & 300 DPI PNG):
  1. Standardised Data Pipeline Architecture & Decoupled Parquet Schema (Flow Diagram)
  2. Storage Reduction & Read Speedup Benchmark (Raw CSV vs Standardised Parquet)

Outputs saved to:
  - presentation_figures/standardised_data_architecture_flow.(svg|png)
  - presentation_figures/standardised_data_storage_benchmark.(svg|png)
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns

# Ensure Matplotlib cache dir is inside workspace
os.environ["MPLCONFIGDIR"] = str(Path("presentation_figures/.cache").resolve())

# Semi-Pastel Terracotta Sand vs. Slate Periwinkle Palette for Data Architecture
COLOR_CSV = "#D49A6A"        # Soft Terracotta Sand (Raw CSV)
COLOR_PARQUET = "#4C4C7A"    # Deep Slate Periwinkle (Parquet)
COLOR_ACCENT = "#3B3B66"     # Dark Periwinkle Text Accent

# Set high-legibility presentation theme with Arial font & enlarged labels
plt.style.use("default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 14,
    "axes.titlesize": 20,
    "axes.titleweight": "bold",
    "axes.labelsize": 16,
    "axes.labelweight": "bold",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "legend.title_fontsize": 15,
    "figure.titlesize": 22,
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

def save_fig(fig, filename_stem):
    """Saves figure in both SVG and PNG format across output directories."""
    for d in [OUTPUT_DIR, DOCS_DIR]:
        svg_path = d / f"{filename_stem}.svg"
        png_path = d / f"{filename_stem}.png"
        fig.savefig(svg_path, bbox_inches="tight", format="svg")
        fig.savefig(png_path, bbox_inches="tight", dpi=300, format="png")
    print(f"Saved: {filename_stem}.(svg|png)")

# ==============================================================================
# FIGURE 1: Architectural Flow Diagram (Raw Ingestion -> Parquet -> Extractors)
# ==============================================================================
def plot_architecture_flow():
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axis("off")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    
    # Title - Simplified & Clean
    ax.text(5.5, 4.6, "Standardised Ingestion Architecture", ha="center", va="center", fontsize=20, fontweight="bold", color=COLOR_ACCENT)

    # Box 1: Raw Sources
    rect1 = patches.FancyBboxPatch((0.4, 1.2), 2.4, 2.6, boxstyle="round,pad=0.2,rounding_size=0.15", facecolor="#FAF3EE", edgecolor="#D49A6A", linewidth=1.8)
    ax.add_patch(rect1)
    ax.text(1.6, 3.4, "Raw Sources", ha="center", va="center", fontsize=15, fontweight="bold", color="#5C3B1E")
    ax.text(1.6, 2.7, "• BEED EEG\n• Pred-Maintenance\n• Synthetic Signals", ha="center", va="center", fontsize=12, color="#7A5230")
    ax.text(1.6, 1.6, "[Raw CSV Format]", ha="center", va="center", fontsize=11, fontweight="bold", color="#A66E43")

    # Arrow 1 -> 2
    ax.annotate("", xy=(3.4, 2.5), xytext=(2.9, 2.5), arrowprops=dict(arrowstyle="->", color=COLOR_PARQUET, lw=2.5, mutation_scale=18))

    # Box 2: Polars Ingestion Pipeline
    rect2 = patches.FancyBboxPatch((3.5, 1.2), 2.6, 2.6, boxstyle="round,pad=0.2,rounding_size=0.15", facecolor="#EBEBF4", edgecolor=COLOR_PARQUET, linewidth=2.0)
    ax.add_patch(rect2)
    ax.text(4.8, 3.4, "Polars Pipeline", ha="center", va="center", fontsize=15, fontweight="bold", color=COLOR_ACCENT)
    ax.text(4.8, 2.7, "• Zero-copy Casting\n• Null Imputation\n• Schema Validation", ha="center", va="center", fontsize=12, color="#333352")
    ax.text(4.8, 1.6, "[convert_datasets.py]", ha="center", va="center", fontsize=11, fontweight="bold", color=COLOR_PARQUET)

    # Arrow 2 -> 3
    ax.annotate("", xy=(6.7, 2.5), xytext=(6.2, 2.5), arrowprops=dict(arrowstyle="->", color=COLOR_PARQUET, lw=2.5, mutation_scale=18))

    # Box 3: Standardised Parquet Storage
    rect3 = patches.FancyBboxPatch((6.8, 1.2), 2.5, 2.6, boxstyle="round,pad=0.2,rounding_size=0.15", facecolor="#E2E2EF", edgecolor=COLOR_PARQUET, linewidth=2.2)
    ax.add_patch(rect3)
    ax.text(8.05, 3.4, "Parquet Storage", ha="center", va="center", fontsize=15, fontweight="bold", color=COLOR_ACCENT)
    ax.text(8.05, 2.7, "time_series.parquet\n(signals)\n\ntargets.parquet\n(labels)", ha="center", va="center", fontsize=12, color="#222238", fontweight="bold")
    ax.text(8.05, 1.6, "[Binary Schema]", ha="center", va="center", fontsize=11, fontweight="bold", color=COLOR_PARQUET)

    # Arrow 3 -> 4
    ax.annotate("", xy=(9.8, 2.5), xytext=(9.4, 2.5), arrowprops=dict(arrowstyle="->", color=COLOR_PARQUET, lw=2.5, mutation_scale=18))

    # Box 4: Extractors
    rect4 = patches.FancyBboxPatch((9.9, 1.2), 1.0, 2.6, boxstyle="round,pad=0.2,rounding_size=0.15", facecolor="#F3F3F8", edgecolor="#9B9BC2", linewidth=1.8)
    ax.add_patch(rect4)
    ax.text(10.4, 2.5, "TSFresh\nTSFEL\nTEMPO", ha="center", va="center", fontsize=13, fontweight="bold", color=COLOR_ACCENT)

    save_fig(fig, "standardised_data_architecture_flow")
    plt.close(fig)

# ==============================================================================
# FIGURE 2: Storage & IO Benchmark (Raw CSV vs Parquet)
# ==============================================================================
def plot_storage_benchmark():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    
    datasets = ["BEED EEG", "Pred-Maintenance"]
    x = np.arange(len(datasets))
    width = 0.42
    
    csv_sizes = [0.39, 0.50]      # MB
    parquet_sizes = [0.14, 0.10]  # MB
    
    csv_times = [9.69, 19.24]    # ms
    parquet_times = [2.85, 14.62] # ms
    
    # Subplot 1: File Size by Dataset - Font size 9.0pt / 8.5pt
    rects1 = ax1.bar(x - width/2, csv_sizes, width, label="Raw CSV", color=COLOR_CSV, hatch="//", edgecolor="#6E4221", linewidth=1.2, zorder=3)
    rects2 = ax1.bar(x + width/2, parquet_sizes, width, label="Parquet", color=COLOR_PARQUET, hatch="..", edgecolor="#222238", linewidth=1.2, zorder=3)
    
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.2f} MB", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9.0, fontweight="bold", color="#0F172A")
    for rect, pct in zip(rects2, ["-64%", "-80%"]):
        h = rect.get_height()
        ax1.annotate(f"{h:.2f} MB\n({pct})", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=COLOR_ACCENT)
        
    ax1.set_title("File Size by Dataset")
    ax1.set_ylabel("File Size (MB)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.set_xlim(-0.6, len(datasets) - 0.4)
    ax1.set_ylim(0, 0.70)
    ax1.grid(axis="y", zorder=0)
    # Legend in top left
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CBD5E1")
    sns.despine(ax=ax1, top=True, right=True)

    # Subplot 2: Read Latency by Dataset - Font size 9.0pt / 8.5pt
    rects3 = ax2.bar(x - width/2, csv_times, width, label="Raw CSV", color=COLOR_CSV, hatch="//", edgecolor="#6E4221", linewidth=1.2, zorder=3)
    rects4 = ax2.bar(x + width/2, parquet_times, width, label="Parquet", color=COLOR_PARQUET, hatch="..", edgecolor="#222238", linewidth=1.2, zorder=3)
    
    for rect in rects3:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f} ms", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9.0, fontweight="bold", color="#0F172A")
    for rect, pct in zip(rects4, ["-71%", "-24%"]):
        h = rect.get_height()
        ax2.annotate(f"{h:.1f} ms\n({pct})", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=COLOR_ACCENT)
        
    ax2.set_title("Read Latency by Dataset")
    ax2.set_ylabel("Read Latency (ms)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.set_xlim(-0.6, len(datasets) - 0.4)
    ax2.set_ylim(0, 27)
    ax2.grid(axis="y", zorder=0)
    # Legend in top left
    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CBD5E1")
    sns.despine(ax=ax2, top=True, right=True)

    save_fig(fig, "standardised_data_storage_benchmark")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating updated presentation-ready figures in Terracotta/Periwinkle palette...")
    plot_architecture_flow()
    plot_storage_benchmark()
    print("Updated figures successfully generated!")
