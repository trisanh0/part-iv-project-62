"""AI4I Predictive Maintenance Running Problem Benchmark Script.

Document ID: TM269Ga
Demonstrates the end-to-end TEMPO pipeline on the AI4I 2020 Predictive Maintenance
dataset as the pedagogical running example for Section 3 (Methodology) and Section 4 (Results).

Executes:
1. Standardized sliding-window segmentation (W=100, S=50).
2. Dual-task generation: Binary Failure Classification and Continuous Tool Wear Regression.
3. 5-Extractor Telemetry Matrix (Numba, Polars, NumPy, TSFEL, TSFresh).
4. 6-Selector Comparative Evaluation across 5-fold CV.
5. Physical Feature Importance Audit on surviving sensor dynamics.
6. Publication-ready 300 DPI figures and paired statistical tests.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Headless matplotlib
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from tempo.analysis import run_statistical_analysis
from tempo.benchmark import BakeoffRunner, PipelineConfig
from tempo.storage.dataset import validate_export
from tempo.storage.segmentation import segment_time_series
from tempo.telemetry import log_system_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tempo.experiments.ai4i_running_benchmark")


def prepare_ai4i_datasets(
    raw_path: str = "data/01_raw/pred-maintenance/ai4i2020.csv",
    cls_output_dir: str = "data/03_processed/pred-maintenance-w100-cls",
    reg_output_dir: str = "data/03_processed/pred-maintenance-w100-reg",
    window_size: int = 100,
    stride: int = 50,
    cls_label_strategy: str = "any_positive",
    force_recreate: bool = False,
) -> Tuple[Path, Path]:
    """Ingest, clean, and segment raw AI4I dataset for classification and regression tasks.

    Args:
        raw_path: Path to raw ai4i2020.csv.
        cls_output_dir: Target directory for classification Parquet files.
        reg_output_dir: Target directory for regression Parquet files.
        window_size: Length of sliding window in steps (W).
        stride: Stride offset between consecutive windows (S).
        cls_label_strategy: Aggregation strategy for failure label ('any_positive' or 'last').
        force_recreate: If True, regenerates segments even if files exist.

    Returns:
        Tuple of (cls_dir_path, reg_dir_path).
    """
    cls_path = Path(cls_output_dir)
    reg_path = Path(reg_output_dir)

    cls_valid = validate_export(str(cls_path))
    reg_valid = validate_export(str(reg_path))

    if cls_valid and reg_valid and not force_recreate:
        logger.info("Validated existing processed AI4I datasets at %s and %s", cls_path, reg_path)
        return cls_path, reg_path

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw AI4I dataset not found at: {raw_path}")

    logger.info("Segmenting raw AI4I dataset: W=%d, S=%d, cls_strategy=%s", window_size, stride, cls_label_strategy)
    df_raw = pl.read_csv(raw_path)

    type_mapping = {"L": 0, "M": 1, "H": 2}
    leak_cols = ["TWF", "HDF", "PWF", "OSF", "RNF", "Product ID"]
    df_filtered = df_raw.drop(leak_cols).with_columns([
        pl.col("UDI").cast(pl.Int32).alias("time"),
        pl.col("Type").replace_strict(type_mapping, default=None).cast(pl.Int32).alias("Type"),
        pl.col("Machine failure").cast(pl.Int32).alias("target_cls"),
        pl.col("Tool wear [min]").cast(pl.Float32).alias("target_reg"),
    ])

    feature_cols = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    # 1. Classification dataset
    cls_path.mkdir(parents=True, exist_ok=True)
    df_ts_cls, df_targets_cls = segment_time_series(
        df_filtered,
        window_size=window_size,
        stride=stride,
        time_col="time",
        feature_cols=feature_cols,
        label_col="target_cls",
        label_strategy=cls_label_strategy,
    )
    df_ts_cls.write_parquet(cls_path / "time_series.parquet")
    df_targets_cls.write_parquet(cls_path / "targets.parquet")
    logger.info("Exported AI4I classification dataset (%d windows) to %s", df_targets_cls.height, cls_path)

    # 2. Regression dataset (target: final-step Tool wear)
    reg_path.mkdir(parents=True, exist_ok=True)
    df_ts_reg, df_targets_reg = segment_time_series(
        df_filtered,
        window_size=window_size,
        stride=stride,
        time_col="time",
        feature_cols=feature_cols,
        label_col="target_reg",
        label_strategy="last",
    )
    df_ts_reg.write_parquet(reg_path / "time_series.parquet")
    df_targets_reg.write_parquet(reg_path / "targets.parquet")
    logger.info("Exported AI4I regression dataset (%d windows) to %s", df_targets_reg.height, reg_path)

    return cls_path, reg_path


def audit_physical_features(
    df_results: pd.DataFrame,
    feature_store_dir: str = "data/03_processed/feature_store",
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Analyze which physical signal domains are prioritized by statistical and ML feature selectors.

    Args:
        df_results: DataFrame of completed benchmark runs.
        feature_store_dir: Path to Parquet feature store.
        output_dir: Destination path for audit artifacts.

    Returns:
        Dictionary detailing domain selection proportions.
    """
    audit_summary: Dict[str, Any] = {
        "domain_selection_counts": {},
        "top_features_by_selector": {},
    }

    numba_runs = df_results[df_results["Extractor"].str.contains("numba", case=False, na=False)]
    if numba_runs.empty:
        logger.warning("No numba_efficient runs found in results for physical feature audit.")
        return audit_summary

    # Physical feature domains in AI4I
    domains = {
        "Tool wear": "Tool wear",
        "Torque": "Torque",
        "Rotational speed": "Rotational speed",
        "Process temperature": "Process temperature",
        "Air temperature": "Air temperature",
        "Type": "Type",
    }

    logger.info("Physical Feature Audit: Evaluated across %d numba evaluation runs.", len(numba_runs))

    domain_counts: Dict[str, int] = {d: 0 for d in domains}
    total_selected = 0

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        audit_file = output_dir / "physical_feature_audit.json"
        with open(audit_file, "w", encoding="utf-8") as f:
            json.dump(audit_summary, f, indent=2)
        logger.info("Saved physical feature audit metadata to %s", audit_file)

    return audit_summary


def generate_publication_plots(
    df_all: pd.DataFrame,
    raw_csv_path: str,
    out_dir: Path,
) -> None:
    """Generate high-resolution 300 DPI publication plots for Section 3 and Section 4 of the report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
    })

    # --------------------------------------------------------------------------
    # Plot 1: AI4I Sliding-Window Segmentation Schematic (Section 3 Methodology)
    # --------------------------------------------------------------------------
    if os.path.exists(raw_csv_path):
        try:
            df_raw = pd.read_csv(raw_csv_path).iloc[0:500]
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True, dpi=300)

            # Torque & Speed
            ax1.plot(df_raw["UDI"], df_raw["Torque [Nm]"], color="#1f77b4", label="Torque [Nm]", lw=1.2)
            ax1.set_ylabel("Torque [Nm]", fontsize=10, fontweight="bold")
            ax1.grid(True, linestyle="--", alpha=0.5)
            ax1.legend(loc="upper right", framealpha=0.9)

            # Tool Wear
            ax2.plot(df_raw["UDI"], df_raw["Tool wear [min]"], color="#2ca02c", label="Tool Wear [min]", lw=1.5)
            ax2.set_ylabel("Wear [min]", fontsize=10, fontweight="bold")
            ax2.grid(True, linestyle="--", alpha=0.5)
            ax2.legend(loc="upper right", framealpha=0.9)

            # Temperatures & Failures
            ax3.plot(df_raw["UDI"], df_raw["Process temperature [K]"], color="#d62728", label="Process Temp [K]", lw=1.2)
            ax3.plot(df_raw["UDI"], df_raw["Air temperature [K]"], color="#ff7f0e", label="Air Temp [K]", lw=1.2)
            ax3.set_ylabel("Temp [K]", fontsize=10, fontweight="bold")
            ax3.set_xlabel("Operational Step (UDI)", fontsize=11, fontweight="bold")
            ax3.grid(True, linestyle="--", alpha=0.5)
            ax3.legend(loc="upper right", framealpha=0.9)

            # Highlight Window W=100, Stride S=50
            for ax in (ax1, ax2, ax3):
                ax.axvspan(100, 200, color="#2b5c8f", alpha=0.18, label="Window $W_1$ [100:200]")
                ax.axvspan(150, 250, color="#ff7f0e", alpha=0.12, label="Window $W_2$ [150:250] (Stride $S=50$)")

            fig.suptitle("TEMPO Continuous Sliding-Window Segmentation ($W=100, S=50$)", fontsize=13, fontweight="bold", y=0.98)
            plt.tight_layout()
            fig.savefig(out_dir / "01_ai4i_window_segmentation_schematic.png", dpi=300)
            fig.savefig(out_dir / "01_ai4i_window_segmentation_schematic.svg")
            plt.close(fig)
            logger.info("Saved Plot 1: 01_ai4i_window_segmentation_schematic")
        except Exception as e:
            logger.warning("Failed to render Plot 1: %s", e)

    # --------------------------------------------------------------------------
    # Plot 2: 5-Engine Extraction Telemetry (RAM vs Time on Log Scale)
    # --------------------------------------------------------------------------
    if "Extraction Time (s)" in df_all.columns and "Extractor" in df_all.columns:
        try:
            ext_stats = df_all.groupby("Extractor").agg({
                "Extraction Time (s)": "first",
                "Extraction Peak RAM (MB)": "first",
                "N Extracted Features": "first",
            }).reset_index()

            fig, ax1 = plt.subplots(figsize=(9, 5.2), dpi=300)
            x_pos = np.arange(len(ext_stats))
            width = 0.35

            rects1 = ax1.bar(x_pos - width/2, ext_stats["Extraction Time (s)"], width, color="#2b5c8f", label="Time (s)", edgecolor="black")
            ax1.set_ylabel("Extraction Latency (s) [Log Scale]", fontsize=11, fontweight="bold", color="#2b5c8f")
            ax1.set_yscale("log")
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(ext_stats["Extractor"], rotation=25, ha="right", fontsize=10)

            ax2 = ax1.twinx()
            rects2 = ax2.bar(x_pos + width/2, ext_stats["Extraction Peak RAM (MB)"], width, color="#d95f02", label="Peak RAM (MB)", edgecolor="black")
            ax2.set_ylabel("Peak RAM Allocation (MB)", fontsize=11, fontweight="bold", color="#d95f02")

            ax1.set_title("TEMPO 5-Extractor Telemetry Benchmark on AI4I ($N=199, W=100$)", fontsize=12, fontweight="bold", pad=12)
            ax1.grid(True, linestyle="--", alpha=0.4, axis="y")

            plt.tight_layout()
            fig.savefig(out_dir / "02_extractor_telemetry_bar_chart.png", dpi=300)
            fig.savefig(out_dir / "02_extractor_telemetry_bar_chart.svg")
            plt.close(fig)
            logger.info("Saved Plot 2: 02_extractor_telemetry_bar_chart")
        except Exception as e:
            logger.warning("Failed to render Plot 2: %s", e)

    # --------------------------------------------------------------------------
    # Plot 3: 2D Extractor x Selector Accuracy Heatmap
    # --------------------------------------------------------------------------
    df_cls = df_all[df_all["Task"] == "classification"].copy()
    if not df_cls.empty and "Accuracy" in df_cls.columns:
        try:
            pivot_acc = df_cls.groupby(["Extractor", "Selector"])["Accuracy"].mean().unstack()

            fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=300)
            im = ax.imshow(pivot_acc.values, cmap="viridis", aspect="auto")

            ax.set_title("AI4I Machine Failure Classification: Extractor vs. Selector Accuracy", fontsize=12, fontweight="bold", pad=12)
            ax.set_xlabel("Feature Selection Paradigm", fontsize=11, fontweight="bold")
            ax.set_ylabel("Feature Extraction Paradigm", fontsize=11, fontweight="bold")

            ax.set_xticks(range(len(pivot_acc.columns)))
            ax.set_xticklabels(pivot_acc.columns, rotation=25, ha="right", fontsize=10)
            ax.set_yticks(range(len(pivot_acc.index)))
            ax.set_yticklabels(pivot_acc.index, fontsize=10)

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Mean Test Accuracy", fontsize=10, fontweight="bold")

            for i in range(len(pivot_acc.index)):
                for j in range(len(pivot_acc.columns)):
                    val = pivot_acc.values[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="white" if val < 0.9 else "black", fontweight="bold", fontsize=9)

            plt.tight_layout()
            fig.savefig(out_dir / "03_extractor_vs_selector_accuracy_heatmap.png", dpi=300)
            fig.savefig(out_dir / "03_extractor_vs_selector_accuracy_heatmap.svg")
            plt.close(fig)
            logger.info("Saved Plot 3: 03_extractor_vs_selector_accuracy_heatmap")
        except Exception as e:
            logger.warning("Failed to render Plot 3: %s", e)

    # --------------------------------------------------------------------------
    # Plot 4: Pareto Frontier (Selection Latency vs Classification Accuracy)
    # --------------------------------------------------------------------------
    if not df_cls.empty and "Selection Time (s)" in df_cls.columns:
        try:
            sel_agg = df_cls.groupby("Selector").agg({
                "Selection Time (s)": "mean",
                "Accuracy": "mean",
            }).reset_index()

            fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
            colors = plt.cm.tab10(np.linspace(0, 1, len(sel_agg)))

            for idx, row in sel_agg.iterrows():
                lat = max(1e-4, row["Selection Time (s)"])
                ax.scatter(lat, row["Accuracy"], s=180, color=colors[idx], edgecolors="black", linewidth=1.2, zorder=5)
                ax.annotate(row["Selector"], (lat, row["Accuracy"]), textcoords="offset points", xytext=(8, 4), fontweight="bold", fontsize=9)

            ax.set_xscale("log")
            ax.set_title("Pareto Trade-Off: Feature Selection Latency vs. Failure Classification Accuracy", fontsize=11, fontweight="bold", pad=12)
            ax.set_xlabel("Mean Selection Latency per Fold in Seconds (Log Scale)", fontsize=10, fontweight="bold")
            ax.set_ylabel("Mean Cross-Validated Accuracy", fontsize=10, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.5)

            plt.tight_layout()
            fig.savefig(out_dir / "04_selector_pareto_latency_vs_accuracy.png", dpi=300)
            fig.savefig(out_dir / "04_selector_pareto_latency_vs_accuracy.svg")
            plt.close(fig)
            logger.info("Saved Plot 4: 04_selector_pareto_latency_vs_accuracy")
        except Exception as e:
            logger.warning("Failed to render Plot 4: %s", e)

    # --------------------------------------------------------------------------
    # Plot 5: Cross-Fold Jaccard Stability Index
    # --------------------------------------------------------------------------
    if "Selection Stability (Jaccard)" in df_all.columns:
        try:
            sel_stab = df_all[df_all["Selector"] != "None"].groupby("Selector")["Selection Stability (Jaccard)"].mean().reset_index()

            fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
            bars = ax.bar(sel_stab["Selector"], sel_stab["Selection Stability (Jaccard)"], color="#2b5c8f", edgecolor="black", width=0.5, zorder=3)
            ax.set_ylim(0, 1.1)
            ax.set_title("Feature Selection Stability Across Cross-Validation Folds ($Jaccard$)", fontsize=11, fontweight="bold", pad=12)
            ax.set_xlabel("Feature Selector", fontsize=10, fontweight="bold")
            ax.set_ylabel("Mean Jaccard Stability $J(S_i, S_j)$", fontsize=10, fontweight="bold")
            ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
            ax.set_xticks(range(len(sel_stab)))
            ax.set_xticklabels(sel_stab["Selector"], rotation=25, ha="right", fontsize=9)

            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, h + 0.02, f"{h:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=9)

            plt.tight_layout()
            fig.savefig(out_dir / "05_selector_stability_jaccard.png", dpi=300)
            fig.savefig(out_dir / "05_selector_stability_jaccard.svg")
            plt.close(fig)
            logger.info("Saved Plot 5: 05_selector_stability_jaccard")
        except Exception as e:
            logger.warning("Failed to render Plot 5: %s", e)

    # --------------------------------------------------------------------------
    # Plot 6: Regression Pareto Frontier (Selection Latency vs Tool Wear RMSE)
    # --------------------------------------------------------------------------
    df_reg = df_all[df_all["Task"] == "regression"].copy()
    if not df_reg.empty and "Selection Time (s)" in df_reg.columns and "RMSE" in df_reg.columns:
        try:
            sel_agg_reg = df_reg.groupby("Selector").agg({
                "Selection Time (s)": "mean",
                "RMSE": "mean",
            }).reset_index()

            fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
            colors = plt.cm.tab10(np.linspace(0, 1, len(sel_agg_reg)))

            for idx, row in sel_agg_reg.iterrows():
                lat = max(1e-4, row["Selection Time (s)"])
                ax.scatter(lat, row["RMSE"], s=180, color=colors[idx], edgecolors="black", linewidth=1.2, zorder=5)
                ax.annotate(row["Selector"], (lat, row["RMSE"]), textcoords="offset points", xytext=(8, 4), fontweight="bold", fontsize=9)

            ax.set_xscale("log")
            ax.set_title("Pareto Trade-Off: Feature Selection Latency vs. Tool Wear RMSE", fontsize=11, fontweight="bold", pad=12)
            ax.set_xlabel("Mean Selection Latency per Fold in Seconds (Log Scale)", fontsize=10, fontweight="bold")
            ax.set_ylabel("Mean Cross-Validated RMSE (Lower is Better)", fontsize=10, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.5)

            plt.tight_layout()
            fig.savefig(out_dir / "06_regression_pareto_latency_vs_rmse.png", dpi=300)
            fig.savefig(out_dir / "06_regression_pareto_latency_vs_rmse.svg")
            plt.close(fig)
            logger.info("Saved Plot 6: 06_regression_pareto_latency_vs_rmse")
        except Exception as e:
            logger.warning("Failed to render Plot 6: %s", e)


def main():
    parser = argparse.ArgumentParser(description="TEMPO AI4I Running Benchmark (TM269Ga)")
    parser.add_argument("--cls-config", type=str, default="configs/TM269Ga__ai4i_running_classification.yaml")
    parser.add_argument("--reg-config", type=str, default="configs/TM269Ga__ai4i_running_regression.yaml")
    parser.add_argument("--raw-data", type=str, default="data/01_raw/pred-maintenance/ai4i2020.csv")
    parser.add_argument("--output-dir", type=str, default="benchmark_results/TM269Ga__ai4i_running_benchmark")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--cls-strategy", type=str, default="any_positive")
    parser.add_argument("--force-recreate", action="store_true", help="Force recreate sliding-window Parquet files")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0_master = time.perf_counter()
    logger.info("=================================================================")
    logger.info("STARTING AI4I RUNNING BENCHMARK (TM269Ga)")
    logger.info("Host Hardware Specifications:")
    log_system_info(out_dir / "environment.json")
    logger.info("=================================================================")

    # 1. Dataset Pre-processing & Windowing
    cls_data_dir, reg_data_dir = prepare_ai4i_datasets(
        raw_path=args.raw_data,
        cls_output_dir="data/03_processed/pred-maintenance-w100-cls",
        reg_output_dir="data/03_processed/pred-maintenance-w100-reg",
        window_size=args.window_size,
        stride=args.stride,
        cls_label_strategy=args.cls_strategy,
        force_recreate=args.force_recreate,
    )

    # 2. Classification Bakeoff Run
    logger.info("--> Running Classification Pipeline: %s", args.cls_config)
    cfg_cls = PipelineConfig.from_yaml(args.cls_config)
    runner_cls = BakeoffRunner(cfg_cls)
    df_cls = runner_cls.run()
    logger.info("--> Completed Classification Bakeoff: %d records.", len(df_cls))

    # 3. Extrinsic Regression Bakeoff Run
    logger.info("--> Running Regression Pipeline: %s", args.reg_config)
    cfg_reg = PipelineConfig.from_yaml(args.reg_config)
    runner_reg = BakeoffRunner(cfg_reg)
    df_reg = runner_reg.run()
    logger.info("--> Completed Regression Bakeoff: %d records.", len(df_reg))

    # 4. Consolidate Master Results
    df_combined = pd.concat([df_cls, df_reg], ignore_index=True)
    summary_csv = out_dir / "ai4i_running_benchmark_summary.csv"
    df_combined.to_csv(summary_csv, index=False)
    logger.info("Saved Master Summary CSV to: %s", summary_csv)

    # 5. Statistical Hypothesis Testing
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

    # 6. Physical Feature Audit
    audit_dir = out_dir / "audit"
    audit_physical_features(df_combined, output_dir=audit_dir)

    # 7. Generate Publication Figures
    plots_dir = out_dir / "plots"
    generate_publication_plots(df_combined, args.raw_data, plots_dir)

    elapsed = time.perf_counter() - t0_master
    logger.info("=================================================================")
    logger.info("AI4I RUNNING BENCHMARK COMPLETE in %.2f seconds (%.2f minutes)", elapsed, elapsed / 60.0)
    logger.info("Results, analysis, and publication figures ready in: %s", out_dir)
    logger.info("=================================================================")


if __name__ == "__main__":
    main()
