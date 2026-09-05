"""
Presentation-Ready Figure Generator for TEMPO Mid-Year Report & Slides.

Reads directly from CSV data sources (no hardcoded metrics):
  - src/classification-examples/data_sources/cross_package_summary_statistics.csv
  - src/classification-examples/data_sources/fourier_summary_statistics.csv

Outputs:
  - Both .svg (vector for PowerPoint) and .png (300 DPI)
  - Saved to presentation_figures/ directory

Plots generated:
  1. prediction_accuracy_by_combination.(svg|png)
  2. extraction_time_by_tsfresh_extractor.(svg|png)
  3. prediction_accuracy_by_tsfresh_extractor.(svg|png)
  4. extraction_time_by_extractor.(svg|png)
  5. prediction_accuracy_by_extractor.(svg|png)
  6. prediction_accuracy_by_selector.(svg|png)
"""

import os
from pathlib import Path

# Ensure Matplotlib cache dir is inside workspace
os.environ["MPLCONFIGDIR"] = str(Path("presentation_figures/.cache").resolve())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

DATA_DIR = Path("src/classification-examples/data_sources")

# ==============================================================================
# LOAD CSV DATA SOURCES
# ==============================================================================
raw_cross_csv = DATA_DIR / "cross_package_summary_statistics.csv"
raw_fourier_csv = DATA_DIR / "fourier_summary_statistics.csv"

# Load multi-header CSVs
df_cross_raw = pd.read_csv(raw_cross_csv, header=[0, 1], index_col=[0, 1])
df_fourier_raw = pd.read_csv(raw_fourier_csv, header=[0, 1], index_col=[0, 1])

# Reset index to convert MultiIndex to columns for easier querying
df_cross = df_cross_raw.reset_index()
# Flatten column names for single-level access
df_cross.columns = [
    f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df_cross.columns
]
# Fix column names after flattening
df_cross.rename(columns={"Extractor_": "Extractor", "Selector_": "Selector"}, inplace=True)
df_cross["Selector"] = df_cross["Selector"].fillna("None").astype(str)
df_cross["Extractor"] = df_cross["Extractor"].fillna("None").astype(str)

df_fourier = df_fourier_raw.reset_index()
df_fourier.columns = [
    f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df_fourier.columns
]
df_fourier.rename(columns={"Extractor_": "Extractor", "Selector_": "Selector"}, inplace=True)
df_fourier["Selector"] = df_fourier["Selector"].fillna("None").astype(str)
df_fourier["Extractor"] = df_fourier["Extractor"].fillna("None").astype(str)

# Parse Fourier coefficient cutoff 'n' and preset name from Extractor string
def parse_fourier_extractor(ext_str):
    parts = ext_str.split("-")
    if len(parts) == 2 and parts[1] == "Minimal":
        return "Minimal", 0
    elif len(parts) == 3:
        preset = parts[1]
        n_val = int(parts[2].replace("FFT", ""))
        return preset, n_val
    return ext_str, 0

df_fourier[["Preset", "n"]] = pd.DataFrame(
    df_fourier["Extractor"].apply(parse_fourier_extractor).tolist(), index=df_fourier.index
)
df_fourier = df_fourier.sort_values(by=["Preset", "n"])

# ==============================================================================
# SEMI-PASTEL SLATE/PERIWINKLE HARMONY PALETTES & ACCESSIBILITY HATCHING
# ==============================================================================
COLOR_EXTRACTORS = {
    "Statistics": "#B86B7D",                # Soft Muted Rose/Mauve
    "TSFEL": "#529985",                     # Soft Sage Green/Teal
    "TSFresh-Minimal": "#D49A6A",           # Soft Terracotta/Sand
    "TSFresh Minimal": "#D49A6A",           # Soft Terracotta/Sand
    "TSFresh-Efficient-25FFT": "#4C4C7A",   # Deep Slate Periwinkle
    "TSFresh-Efficient-50FFT": "#4C4C7A",   # Deep Slate Periwinkle
    "TSFresh Efficient": "#4C4C7A",         # Deep Slate Periwinkle
}

HATCH_EXTRACTORS = {
    "Statistics": "",
    "TSFEL": "//",
    "TSFresh-Minimal": "\\\\",
    "TSFresh Minimal": "\\\\",
    "TSFresh-Efficient-25FFT": "..",
    "TSFresh-Efficient-50FFT": "..",
    "TSFresh Efficient": "..",
}

COLOR_SELECTORS = {
    "None": "#78789A",              # Muted Slate Lavender (Baseline)
    "SelectKBest": "#529985",       # Soft Sage Green/Teal
    "Boruta": "#7C6E99",            # Soft Muted Violet
    "TSFresh": "#4C4C7A",           # Deep Slate Periwinkle
}

HATCH_SELECTORS = {
    "None": "//",
    "SelectKBest": "",
    "Boruta": "\\\\",
    "TSFresh": "..",
}

COLOR_FOURIER = {
    "Minimal": "#D49A6A",           # Soft Terracotta/Sand (Consistent with TSFresh Minimal)
    "Efficient": "#529985",         # Soft Sage Green/Teal
    "Comprehensive": "#4C4C7A",     # Deep Slate Periwinkle
}

def save_fig(fig, filename_stem):
    """Saves figure in both SVG and PNG format across output directories."""
    for d in [OUTPUT_DIR, DOCS_DIR]:
        svg_path = d / f"{filename_stem}.svg"
        png_path = d / f"{filename_stem}.png"
        fig.savefig(svg_path, bbox_inches="tight", format="svg")
        fig.savefig(png_path, bbox_inches="tight", dpi=300, format="png")
    print(f"Saved: {filename_stem}.(svg|png)")

# ==============================================================================
# FIGURE 1: Extraction Time by Extractor (Cross-Package)
# ==============================================================================
def plot_extraction_time_by_extractor():
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    
    order = ["Statistics", "TSFEL", "TSFresh-Minimal", "TSFresh-Efficient-50FFT"]
    display_names = ["Statistics", "TSFEL", "TSFresh\nMinimal", "TSFresh\nEfficient"]
    
    means = [df_cross[df_cross["Extractor"] == e]["Extraction Time_mean"].mean() for e in order]
    stds = [df_cross[df_cross["Extractor"] == e]["Extraction Time_std"].mean() for e in order]
    colors = [COLOR_EXTRACTORS[e] for e in order]
    hatches = [HATCH_EXTRACTORS[e] for e in order]
    
    bars = ax.bar(
        display_names,
        means,
        yerr=stds,
        capsize=6,
        color=colors,
        width=0.55,
        edgecolor="#333352",
        linewidth=1.2,
        zorder=3,
        error_kw={"ecolor": "#334155", "lw": 1.8}
    )
    for bar, h_pat in zip(bars, hatches):
        bar.set_hatch(h_pat)
    
    for bar, std_val in zip(bars, stds):
        h = bar.get_height()
        val_text = f"{h:.2f}s" if h >= 0.1 else f"{h:.4f}s"
        ax.annotate(
            val_text,
            xy=(bar.get_x() + bar.get_width() / 2, h + (std_val if not np.isnan(std_val) else 0)),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=13, fontweight="bold", color="#0F172A"
        )
        
    ax.set_title("Extraction Time by Extractor")
    ax.set_ylabel("Extraction Time (s)")
    ax.set_ylim(0, max(means) * 1.25)
    ax.grid(axis="y", zorder=0)
    sns.despine(top=True, right=True)
    plt.xticks(rotation=0, ha="center")
    
    save_fig(fig, "extraction_time_by_extractor")
    plt.close(fig)

# ==============================================================================
# FIGURE 2: Prediction Accuracy by Extractor (Cross-Package)
# ==============================================================================
def plot_prediction_accuracy_by_extractor():
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    
    order = ["Statistics", "TSFEL", "TSFresh-Minimal", "TSFresh-Efficient-50FFT"]
    display_names = ["Statistics", "TSFEL", "TSFresh\nMinimal", "TSFresh\nEfficient"]
    
    means = [df_cross[df_cross["Extractor"] == e]["Prediction Accuracy_mean"].mean() * 100 for e in order]
    stds = [df_cross[df_cross["Extractor"] == e]["Prediction Accuracy_std"].mean() * 100 for e in order]
    colors = [COLOR_EXTRACTORS[e] for e in order]
    hatches = [HATCH_EXTRACTORS[e] for e in order]
    
    bars = ax.bar(
        display_names,
        means,
        yerr=stds,
        capsize=6,
        color=colors,
        width=0.55,
        edgecolor="#333352",
        linewidth=1.2,
        zorder=3,
        error_kw={"ecolor": "#334155", "lw": 1.8}
    )
    for bar, h_pat in zip(bars, hatches):
        bar.set_hatch(h_pat)
    
    for bar, std_val in zip(bars, stds):
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h + (std_val if not np.isnan(std_val) else 0)),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=13, fontweight="bold", color="#0F172A"
        )
        
    ax.set_title("Prediction Accuracy by Extractor")
    ax.set_ylabel("Classification Accuracy (%)")
    ax.set_ylim(90, 100)
    ax.grid(axis="y", zorder=0)
    sns.despine(top=True, right=True)
    plt.xticks(rotation=0, ha="center")
    
    save_fig(fig, "prediction_accuracy_by_extractor")
    plt.close(fig)

# ==============================================================================
# FIGURE 3: Prediction Accuracy by Selector (Cross-Package)
# ==============================================================================
def plot_prediction_accuracy_by_selector():
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    
    order = ["None", "SelectKBest", "Boruta", "TSFresh"]
    means = [df_cross[df_cross["Selector"] == s]["Prediction Accuracy_mean"].mean() * 100 for s in order]
    stds = [df_cross[df_cross["Selector"] == s]["Prediction Accuracy_std"].mean() * 100 for s in order]
    colors = [COLOR_SELECTORS[s] for s in order]
    hatches = [HATCH_SELECTORS[s] for s in order]
    
    bars = ax.bar(
        order,
        means,
        yerr=stds,
        capsize=6,
        color=colors,
        width=0.55,
        edgecolor="#333352",
        linewidth=1.2,
        zorder=3,
        error_kw={"ecolor": "#334155", "lw": 1.8}
    )
    for bar, h_pat in zip(bars, hatches):
        bar.set_hatch(h_pat)
    
    for bar, std_val in zip(bars, stds):
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h + (std_val if not np.isnan(std_val) else 0)),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=13, fontweight="bold", color="#0F172A"
        )
        
    ax.set_title("Prediction Accuracy by Selector")
    ax.set_ylabel("Classification Accuracy (%)")
    ax.set_ylim(90, 100)
    ax.grid(axis="y", zorder=0)
    sns.despine(top=True, right=True)
    plt.xticks(rotation=0)
    
    save_fig(fig, "prediction_accuracy_by_selector")
    plt.close(fig)

# ==============================================================================
# FIGURE 4: Prediction Accuracy by Combination (All 16 Pipelines)
# ==============================================================================
def plot_prediction_accuracy_by_combination():
    fig, ax = plt.subplots(figsize=(13.5, 5.5))
    
    extractor_order = ["Statistics", "TSFEL", "TSFresh-Minimal", "TSFresh-Efficient-50FFT"]
    display_ext_map = {
        "Statistics": "Statistics",
        "TSFEL": "TSFEL",
        "TSFresh-Minimal": "TSFresh Minimal",
        "TSFresh-Efficient-50FFT": "TSFresh Efficient"
    }
    selector_order = ["None", "SelectKBest", "Boruta", "TSFresh"]
    
    df_sorted = []
    for e in extractor_order:
        for s in selector_order:
            sub = df_cross[(df_cross["Extractor"] == e) & (df_cross["Selector"] == s)]
            if not sub.empty:
                row = sub.iloc[0].to_dict()
                row["ComboLabel"] = f"{display_ext_map[e]}\n+ {s}"
                row["DisplayExt"] = display_ext_map[e]
                df_sorted.append(row)
                
    df_combo = pd.DataFrame(df_sorted)
    
    colors = [COLOR_EXTRACTORS[row["DisplayExt"]] for _, row in df_combo.iterrows()]
    hatches = [HATCH_EXTRACTORS[row["DisplayExt"]] for _, row in df_combo.iterrows()]
    means = df_combo["Prediction Accuracy_mean"] * 100
    stds = df_combo["Prediction Accuracy_std"] * 100
    
    bars = ax.bar(
        df_combo["ComboLabel"],
        means,
        yerr=stds,
        capsize=5,
        color=colors,
        width=0.68,
        edgecolor="#333352",
        linewidth=1.2,
        zorder=3,
        error_kw={"ecolor": "#334155", "lw": 1.5}
    )
    for bar, h_pat in zip(bars, hatches):
        bar.set_hatch(h_pat)
    
    for bar, std_val in zip(bars, stds):
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h + (std_val if not np.isnan(std_val) else 0)),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=9.0, fontweight="bold", color="#0F172A"
        )
        
    ax.set_title("Prediction Accuracy by Combination")
    ax.set_ylabel("Classification Accuracy (%)")
    ax.set_ylim(90, 100)
    ax.grid(axis="y", zorder=0)
    sns.despine(top=True, right=True)
    plt.xticks(rotation=45, ha="right", fontsize=11)
    
    save_fig(fig, "prediction_accuracy_by_combination")
    plt.close(fig)

# ==============================================================================
# FIGURE 5: Extraction Time by TSFresh Extractor (Fourier Truncation)
# ==============================================================================
def plot_extraction_time_by_tsfresh_extractor():
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    
    for preset in ["Minimal", "Efficient", "Comprehensive"]:
        sub = df_fourier[df_fourier["Preset"] == preset]
        ax.errorbar(
            sub["n"],
            sub["Extraction Time_mean"],
            yerr=sub["Extraction Time_std"],
            marker="o" if preset != "Minimal" else "s",
            linewidth=2.8,
            markersize=8,
            capsize=5,
            label=f"TSFresh {preset}",
            color=COLOR_FOURIER[preset]
        )
        
    ax.set_title("Extraction Time by TSFresh Extractor")
    ax.set_xlabel("Fourier Coefficients (n)")
    ax.set_ylabel("Extraction Time (s)")
    ax.set_ylim(0, 32)
    ax.set_xticks([0, 5, 10, 25, 50, 75, 100])
    ax.grid(True, zorder=0)
    # Position legend in bottom right as requested
    ax.legend(title="TSFresh Preset", loc="lower right", frameon=True, facecolor="white", edgecolor="#CBD5E1")
    sns.despine(top=True, right=True)
    
    save_fig(fig, "extraction_time_by_tsfresh_extractor")
    plt.close(fig)

# ==============================================================================
# FIGURE 6: Prediction Accuracy by TSFresh Extractor (Fourier Truncation)
# ==============================================================================
def plot_prediction_accuracy_by_tsfresh_extractor():
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    
    for preset in ["Minimal", "Efficient", "Comprehensive"]:
        sub = df_fourier[df_fourier["Preset"] == preset]
        ax.errorbar(
            sub["n"],
            sub["Prediction Accuracy_mean"] * 100,
            yerr=sub["Prediction Accuracy_std"] * 100,
            marker="o" if preset != "Minimal" else "s",
            linewidth=2.8,
            markersize=8,
            capsize=5,
            label=f"TSFresh {preset}",
            color=COLOR_FOURIER[preset]
        )
        
    ax.set_title("Prediction Accuracy by TSFresh Extractor")
    ax.set_xlabel("Fourier Coefficients (n)")
    ax.set_ylabel("Classification Accuracy (%)")
    ax.set_ylim(70, 100)
    ax.set_xticks([0, 5, 10, 25, 50, 75, 100])
    ax.grid(True, zorder=0)
    ax.legend(title="TSFresh Preset", loc="lower right", frameon=True, facecolor="white", edgecolor="#CBD5E1")
    sns.despine(top=True, right=True)
    
    save_fig(fig, "prediction_accuracy_by_tsfresh_extractor")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating presentation-ready figures in semi-pastel palette with matching slate/periwinkle themes...")
    plot_extraction_time_by_extractor()
    plot_prediction_accuracy_by_extractor()
    plot_prediction_accuracy_by_selector()
    plot_prediction_accuracy_by_combination()
    plot_extraction_time_by_tsfresh_extractor()
    plot_prediction_accuracy_by_tsfresh_extractor()
    print("All 6 presentation figures successfully re-generated!")
