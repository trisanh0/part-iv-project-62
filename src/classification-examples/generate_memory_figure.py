"""
Memory Optimisation Figure Generator for Slide 10.

Generates presentation-ready figures (SVG & 300 DPI PNG) comparing:
  - Pandas Baseline vs NumPy Engine Peak Memory Consumption (MB) across scale N=20, 50, 100
  - Execution time trade-off across scale N=20, 50, 100

Outputs saved to:
  - presentation_figures/memory_optimisation_peak_memory.(svg|png)
  - presentation_figures/memory_optimisation_tradeoff.(svg|png)
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Ensure Matplotlib cache dir is inside workspace
os.environ["MPLCONFIGDIR"] = str(Path("presentation_figures/.cache").resolve())

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

# BEED Benchmark Telemetry Data from Table 1 of Mid-year Report
beed_data = [
    {"Scale": 20, "Engine": "Pandas Baseline", "PeakMemory_MB": 59.72, "ExecutionTime_s": 6.48},
    {"Scale": 20, "Engine": "NumPy Engine", "PeakMemory_MB": 6.71, "ExecutionTime_s": 24.11},
    {"Scale": 50, "Engine": "Pandas Baseline", "PeakMemory_MB": 151.66, "ExecutionTime_s": 9.29},
    {"Scale": 50, "Engine": "NumPy Engine", "PeakMemory_MB": 14.30, "ExecutionTime_s": 61.46},
    {"Scale": 100, "Engine": "Pandas Baseline", "PeakMemory_MB": 301.29, "ExecutionTime_s": 14.64},
    {"Scale": 100, "Engine": "NumPy Engine", "PeakMemory_MB": 26.93, "ExecutionTime_s": 125.08},
]

df_beed = pd.DataFrame(beed_data)

COLOR_PANDAS = "#E11D48"  # Rose Red (High Memory Spike)
COLOR_NUMPY = "#059669"   # Emerald Green (Low Memory Optimized)

def save_fig(fig, filename_stem):
    """Saves figure in both SVG and PNG format across output directories."""
    for d in [OUTPUT_DIR, DOCS_DIR]:
        svg_path = d / f"{filename_stem}.svg"
        png_path = d / f"{filename_stem}.png"
        fig.savefig(svg_path, bbox_inches="tight", format="svg")
        fig.savefig(png_path, bbox_inches="tight", dpi=300, format="png")
    print(f"Saved: {filename_stem}.(svg|png)")

# ==============================================================================
# FIGURE 1: Peak Memory Consumption (Grouped Bar Chart)
# ==============================================================================
def plot_memory_optimisation_peak_memory():
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    
    scales = [20, 50, 100]
    x = np.arange(len(scales))
    width = 0.36
    
    pandas_mem = [df_beed[(df_beed["Scale"] == s) & (df_beed["Engine"] == "Pandas Baseline")]["PeakMemory_MB"].values[0] for s in scales]
    numpy_mem = [df_beed[(df_beed["Scale"] == s) & (df_beed["Engine"] == "NumPy Engine")]["PeakMemory_MB"].values[0] for s in scales]
    
    rects1 = ax.bar(x - width/2, pandas_mem, width, label="Pandas Baseline", color=COLOR_PANDAS, zorder=3)
    rects2 = ax.bar(x + width/2, numpy_mem, width, label="NumPy Engine", color=COLOR_NUMPY, zorder=3)
    
    # Value annotations on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(
            f"{h:.1f} MB",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=12, fontweight="bold", color="#0F172A"
        )
        
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(
            f"{h:.1f} MB",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=12, fontweight="bold", color="#0F172A"
        )
        
    # Highlight 11x memory reduction callout at N=100
    ax.annotate(
        "11× Memory\nReduction",
        xy=(2 + width/2, 26.93),
        xytext=(2 + width/2 + 0.15, 120),
        arrowprops=dict(facecolor="#059669", edgecolor="#059669", width=2.5, headwidth=9, shrink=0.08),
        fontsize=13, fontweight="bold", color="#059669",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ECFDF5", edgecolor="#A7F3D0", lw=1.2)
    )
    
    ax.set_title("Peak Memory Consumption (BEED Dataset)")
    ax.set_xlabel("Sample Scale (N Rows)")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"N = {s}" for s in scales])
    ax.set_ylim(0, 360)
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CBD5E1")
    sns.despine(top=True, right=True)
    
    save_fig(fig, "memory_optimisation_peak_memory")
    plt.close(fig)

# ==============================================================================
# FIGURE 2: Memory vs Execution Speed Trade-off (2 Subplots side-by-side)
# ==============================================================================
def plot_memory_optimisation_tradeoff():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    
    scales = [20, 50, 100]
    x = np.arange(len(scales))
    width = 0.36
    
    pandas_mem = [df_beed[(df_beed["Scale"] == s) & (df_beed["Engine"] == "Pandas Baseline")]["PeakMemory_MB"].values[0] for s in scales]
    numpy_mem = [df_beed[(df_beed["Scale"] == s) & (df_beed["Engine"] == "NumPy Engine")]["PeakMemory_MB"].values[0] for s in scales]
    
    pandas_time = [df_beed[(df_beed["Scale"] == s) & (df_beed["Engine"] == "Pandas Baseline")]["ExecutionTime_s"].values[0] for s in scales]
    numpy_time = [df_beed[(df_beed["Scale"] == s) & (df_beed["Engine"] == "NumPy Engine")]["ExecutionTime_s"].values[0] for s in scales]
    
    # Subplot 1: Peak Memory
    rects1 = ax1.bar(x - width/2, pandas_mem, width, label="Pandas Baseline", color=COLOR_PANDAS, zorder=3)
    rects2 = ax1.bar(x + width/2, numpy_mem, width, label="NumPy Engine", color=COLOR_NUMPY, zorder=3)
    
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f} MB", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=11, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f} MB", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=11, fontweight="bold")
        
    ax1.set_title("Peak Memory Usage")
    ax1.set_xlabel("Sample Scale (Rows)")
    ax1.set_ylabel("Peak Memory (MB)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"N = {s}" for s in scales])
    ax1.set_ylim(0, 360)
    ax1.grid(axis="y", zorder=0)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CBD5E1")
    sns.despine(ax=ax1, top=True, right=True)
    
    # Subplot 2: Execution Time
    rects3 = ax2.bar(x - width/2, pandas_time, width, label="Pandas Baseline", color=COLOR_PANDAS, zorder=3)
    rects4 = ax2.bar(x + width/2, numpy_time, width, label="NumPy Engine", color=COLOR_NUMPY, zorder=3)
    
    for rect in rects3:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}s", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=11, fontweight="bold")
    for rect in rects4:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}s", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=11, fontweight="bold")
        
    ax2.set_title("Execution Time")
    ax2.set_xlabel("Sample Scale (Rows)")
    ax2.set_ylabel("Execution Time (s)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"N = {s}" for s in scales])
    ax2.set_ylim(0, 150)
    ax2.grid(axis="y", zorder=0)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CBD5E1")
    sns.despine(ax=ax2, top=True, right=True)
    
    save_fig(fig, "memory_optimisation_tradeoff")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating memory optimisation figures (SVG & PNG)...")
    plot_memory_optimisation_peak_memory()
    plot_memory_optimisation_tradeoff()
    print("Memory optimisation figures successfully generated!")
