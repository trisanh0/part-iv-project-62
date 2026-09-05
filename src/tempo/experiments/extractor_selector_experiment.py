"""
Comprehensive Feature Extractor & Selector Comparison Experiment (TM268Xa).

Executes end-to-end cross-validated evaluation comparing 5 feature extraction paradigms
and 6 feature selection paradigms across 6 multi-domain datasets (Classification & Regression).
Generates telemetry, cross-fold Jaccard selection stability, paired Welch t-tests,
and publication-ready 300 DPI figures.
"""

import argparse
import datetime
import logging
import os
from pathlib import Path
import sys
import time

# Ensure matplotlib runs headless and cache doesn't block
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tempo.benchmark import BakeoffRunner, PipelineConfig
from tempo.analysis import run_statistical_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tempo.experiments.extractor_selector")


def generate_publication_plots(df_all: pd.DataFrame, out_dir: Path) -> None:
    """Generate high-impact publication plots comparing extractors and selectors."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
    })

    df_cls = df_all[df_all["Task"] == "classification"].copy()

    # --------------------------------------------------------------------------
    # Plot 1: 2D Extractor x Selector Accuracy Heatmap
    # --------------------------------------------------------------------------
    if not df_cls.empty and "Accuracy" in df_cls.columns:
        pivot_acc = df_cls.groupby(["Extractor", "Selector"])["Accuracy"].mean().unstack()
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        im = ax.imshow(pivot_acc.values, cmap="viridis", aspect="auto")
        
        ax.set_title("Mean Classification Accuracy: Extractor vs. Selector Paradigm", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Feature Selector", fontsize=11, fontweight="bold")
        ax.set_ylabel("Feature Extractor", fontsize=11, fontweight="bold")
        
        ax.set_xticks(range(len(pivot_acc.columns)))
        ax.set_xticklabels(pivot_acc.columns, rotation=30, ha="right", fontsize=10)
        ax.set_yticks(range(len(pivot_acc.index)))
        ax.set_yticklabels(pivot_acc.index, fontsize=10)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Test Accuracy", fontsize=10, fontweight="bold")
        
        for i in range(len(pivot_acc.index)):
            for j in range(len(pivot_acc.columns)):
                val = pivot_acc.values[i, j]
                if not np.isnan(val):
                    text_color = "white" if val < (pivot_acc.values[~np.isnan(pivot_acc.values)].mean()) else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontweight="bold", fontsize=9)
        
        plt.tight_layout()
        fig.savefig(out_dir / "extractor_vs_selector_accuracy_heatmap.png", dpi=300)
        fig.savefig(out_dir / "extractor_vs_selector_accuracy_heatmap.svg")
        plt.close(fig)
        logger.info("Saved Plot 1: extractor_vs_selector_accuracy_heatmap")

    # --------------------------------------------------------------------------
    # Plot 2: Selection Latency vs Downstream Accuracy Pareto Frontier
    # --------------------------------------------------------------------------
    if not df_cls.empty and "Selection Time (s)" in df_cls.columns:
        selector_agg = df_cls.groupby("Selector").agg({
            "Selection Time (s)": "mean",
            "Accuracy": "mean",
            "Feature Reduction (%)": "mean",
            "Selection Stability (Jaccard)": "mean",
        }).reset_index()

        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
        colors = plt.cm.tab10(np.linspace(0, 1, len(selector_agg)))
        
        for idx, row in selector_agg.iterrows():
            ax.scatter(
                max(1e-4, row["Selection Time (s)"]),
                row["Accuracy"],
                s=180,
                color=colors[idx],
                edgecolors="black",
                linewidth=1.2,
                label=row["Selector"],
                zorder=5,
            )
            ax.annotate(
                row["Selector"],
                (max(1e-4, row["Selection Time (s)"]), row["Accuracy"]),
                textcoords="offset points",
                xytext=(8, 4),
                fontweight="bold",
                fontsize=9,
            )

        ax.set_xscale("log")
        ax.set_title("Pareto Frontier: Selection Latency vs. Classification Accuracy", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Mean Selection Latency per Fold in Seconds (Log Scale)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean Cross-Validated Accuracy", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        fig.savefig(out_dir / "selector_pareto_latency_vs_accuracy.png", dpi=300)
        fig.savefig(out_dir / "selector_pareto_latency_vs_accuracy.svg")
        plt.close(fig)
        logger.info("Saved Plot 2: selector_pareto_latency_vs_accuracy")

    # --------------------------------------------------------------------------
    # Plot 3: Selector Stability (Jaccard Index) vs Feature Reduction Ratio
    # --------------------------------------------------------------------------
    if "Selection Stability (Jaccard)" in df_all.columns and "Feature Reduction (%)" in df_all.columns:
        sel_stab = df_all.groupby("Selector").agg({
            "Selection Stability (Jaccard)": "mean",
            "Feature Reduction (%)": "mean",
        }).reset_index()

        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        bars = ax.bar(
            sel_stab["Selector"],
            sel_stab["Selection Stability (Jaccard)"],
            color="#2b5c8f",
            edgecolor="black",
            width=0.55,
            zorder=3,
        )
        
        ax.set_ylim(0, 1.1)
        ax.set_title("Feature Selection Stability Across CV Folds (Mean Jaccard Index)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Feature Selector", fontsize=11, fontweight="bold")
        ax.set_ylabel("Jaccard Stability Index $J(S_i, S_j)$", fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
        ax.set_xticks(range(len(sel_stab)))
        ax.set_xticklabels(sel_stab["Selector"], rotation=25, ha="right", fontsize=10)


        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.02, f"{h:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=9)

        plt.tight_layout()
        fig.savefig(out_dir / "selector_stability_jaccard_comparison.png", dpi=300)
        fig.savefig(out_dir / "selector_stability_jaccard_comparison.svg")
        plt.close(fig)
        logger.info("Saved Plot 3: selector_stability_jaccard_comparison")


def main():
    parser = argparse.ArgumentParser(description="TEMPO Extractor & Selector Benchmark Experiment")
    parser.add_argument("--cls-config", type=str, default="configs/TM268Xa__extractor_selector_classification.yaml")
    parser.add_argument("--reg-config", type=str, default="configs/TM268Xa__extractor_selector_regression.yaml")
    parser.add_argument("--output-dir", type=str, default="benchmark_results/TM268Xa__extractor_selector_matrix")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0_master = time.perf_counter()
    logger.info("=================================================================")
    logger.info("STARTING GRAND EXTRACTOR & SELECTOR COMPARISON BENCHMARK (TM268Xa)")
    logger.info("=================================================================")

    # 1. Classification Bakeoff Run
    logger.info("--> Loading Classification Pipeline Config: %s", args.cls_config)
    cfg_cls = PipelineConfig.from_yaml(args.cls_config)
    runner_cls = BakeoffRunner(cfg_cls)
    df_cls = runner_cls.run()
    logger.info("--> Classification Bakeoff Finished. Completed %d records.", len(df_cls))

    # 2. Regression Bakeoff Run
    logger.info("--> Loading Extrinsic Regression Pipeline Config: %s", args.reg_config)
    cfg_reg = PipelineConfig.from_yaml(args.reg_config)
    runner_reg = BakeoffRunner(cfg_reg)
    df_reg = runner_reg.run()
    logger.info("--> Extrinsic Regression Bakeoff Finished. Completed %d records.", len(df_reg))

    # 3. Consolidate Master Results Table
    df_combined = pd.concat([df_cls, df_reg], ignore_index=True)
    summary_csv = out_dir / "extractor_selector_matrix_summary.csv"
    df_combined.to_csv(summary_csv, index=False)
    logger.info("Saved Consolidated Matrix Summary to: %s", summary_csv)

    # 4. Generate Multi-Hypothesis Paired t-Tests & Effect Sizes
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    if not df_cls.empty:
        run_statistical_analysis(
            csv_path=str(summary_csv),
            output_dir=str(analysis_dir / "classification"),
            task_type="classification",
            enable_ttests=True,
            enable_plots=True,
        )
    if not df_reg.empty:
        run_statistical_analysis(
            csv_path=str(summary_csv),
            output_dir=str(analysis_dir / "regression"),
            task_type="regression",
            enable_ttests=True,
            enable_plots=True,
        )

    # 5. Generate Publication Plots
    plots_dir = out_dir / "plots"
    generate_publication_plots(df_combined, plots_dir)

    elapsed = time.perf_counter() - t0_master
    logger.info("=================================================================")
    logger.info("GRAND EXTRACTOR & SELECTOR BENCHMARK COMPLETED in %.2f seconds (%.2f hours)", elapsed, elapsed / 3600.0)
    logger.info("All tables, statistics, and plots ready in: %s", out_dir)
    logger.info("=================================================================")


if __name__ == "__main__":
    main()
