"""
Statistical Analysis and Visualization Engine for the TEMPO Framework.

Performs multi-hypothesis paired t-tests with Benjamini-Hochberg False Discovery Rate (FDR)
adjustments, Cohen's d effect size calculation, summary statistics, and publication-ready
linear/log-scale boxplots across feature extraction and selection benchmarks.
"""

from itertools import combinations
import logging
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)

# Set clean aesthetic for publication figures
plt.style.use("default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
})


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Cohen's d effect size for paired samples."""
    diff = x - y
    std_diff = np.std(diff, ddof=1)
    if std_diff == 0:
        return 0.0
    return float(np.mean(diff) / std_diff)


def metric_direction(metric: str) -> Literal["higher", "lower"]:
    """Determine whether higher or lower values indicate superior performance."""
    lower_is_better = {
        "Extraction Time",
        "Selection Time",
        "Prediction Time",
        "Total Time",
        "Extraction Time (s)",
        "Selection Time (s)",
        "Prediction Time (s)",
        "Total Time (s)",
        "Extraction Peak RAM (MB)",
        "Extraction Peak RAM Increase (MB)",
        "Selection Peak RAM (MB)",
        "RMSE",
        "MAE",
        "N Extracted Features",
        "N Selected Features",
    }
    return "lower" if metric in lower_is_better else "higher"


def compute_summary_statistics(
    df: pd.DataFrame,
    group_cols: List[str],
    metrics: List[str],
) -> pd.DataFrame:
    """Compute mean, std, median, min, max across grouped dimensions."""
    valid_metrics = [m for m in metrics if m in df.columns]
    summary = df.groupby(group_cols)[valid_metrics].agg(["mean", "std", "median", "min", "max"])
    return summary


def pairwise_ttests(
    df: pd.DataFrame,
    group_col: str,
    metric: str,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Perform pairwise paired Student's t-tests with Benjamini-Hochberg FDR correction.
    
    Args:
        df: Telemetry results DataFrame.
        group_col: Column to group by ('Extractor', 'Selector', or 'Combination').
        metric: Target evaluation metric column name.
        alpha: Significance threshold (default 0.05).
        
    Returns:
        DataFrame containing pairwise comparisons, p-values, corrected p-values, and effect sizes.
    """
    if metric not in df.columns or group_col not in df.columns:
        return pd.DataFrame()

    groups = [g for g in df[group_col].unique() if pd.notna(g)]
    if len(groups) < 2:
        return pd.DataFrame()

    results = []
    direction = metric_direction(metric)

    # Required pairing key to ensure matched pairs across seeds/folds
    pair_keys = [c for c in ["Dataset", "Seed", "Fold"] if c in df.columns]
    if not pair_keys:
        pair_keys = ["Dataset"] if "Dataset" in df.columns else []

    for g1, g2 in combinations(groups, 2):
        df_g1 = df[df[group_col] == g1]
        df_g2 = df[df[group_col] == g2]

        if pair_keys:
            merged = pd.merge(df_g1, df_g2, on=pair_keys, suffixes=("_1", "_2"))
            x = merged[f"{metric}_1"].to_numpy()
            y = merged[f"{metric}_2"].to_numpy()
        else:
            min_len = min(len(df_g1), len(df_g2))
            x = df_g1[metric].iloc[:min_len].to_numpy()
            y = df_g2[metric].iloc[:min_len].to_numpy()

        if len(x) < 2:
            continue

        mean_1 = float(np.mean(x))
        mean_2 = float(np.mean(y))
        mean_diff = mean_1 - mean_2

        try:
            stat, pval = ttest_rel(x, y)
        except Exception:
            stat, pval = np.nan, np.nan

        d_val = cohens_d(x, y)

        if not np.isnan(pval):
            if direction == "higher":
                winner = g1 if mean_diff > 0 else g2
            else:
                winner = g1 if mean_diff < 0 else g2
        else:
            winner = "N/A"

        results.append({
            "Group 1": g1,
            "Group 2": g2,
            "Metric": metric,
            "N Pairs": len(x),
            "Mean 1": round(mean_1, 4),
            "Mean 2": round(mean_2, 4),
            "Mean Difference": round(mean_diff, 4),
            "Cohen's d": round(d_val, 4),
            "t-Statistic": round(stat, 4) if not np.isnan(stat) else np.nan,
            "p-Value": pval,
            "Winner (Pre-Correction)": winner,
        })

    if not results:
        return pd.DataFrame()

    res_df = pd.DataFrame(results)
    pvals = res_df["p-Value"].fillna(1.0).to_numpy()
    
    # Benjamini-Hochberg FDR correction
    reject, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    res_df["p-Value (BH-Corrected)"] = pvals_corrected
    res_df["Significant (FDR q<0.05)"] = reject
    res_df["Final Winner"] = np.where(
        reject,
        res_df["Winner (Pre-Correction)"],
        "No Statistically Significant Difference"
    )

    return res_df


def plot_metric_boxplot(
    df: pd.DataFrame,
    metric: str,
    group_col: str,
    output_path: Union[str, Path],
    log_scale: bool = False,
    title: Optional[str] = None,
) -> None:
    """Generate high-resolution boxplot for metric distribution by group."""
    if metric not in df.columns or group_col not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    groups = sorted(df[group_col].dropna().unique())
    data = [df[df[group_col] == g][metric].dropna().to_numpy() for g in groups]

    bp = ax.boxplot(
        data,
        labels=groups,
        patch_artist=True,
        medianprops={"color": "#111827", "linewidth": 2},
        boxprops={"facecolor": "#E0E7FF", "edgecolor": "#4338CA", "linewidth": 1.5},
        whiskerprops={"color": "#4338CA", "linewidth": 1.2},
        capprops={"color": "#4338CA", "linewidth": 1.2},
    )

    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel(f"{metric} (Log Scale)", fontsize=12, fontweight="bold")
    else:
        ax.set_ylabel(metric, fontsize=12, fontweight="bold")

    ax.set_xlabel(group_col, fontsize=12, fontweight="bold")
    plot_title = title or f"{metric} by {group_col}"
    ax.set_title(plot_title, fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=300)
    plt.close(fig)


def run_statistical_analysis(
    csv_path: Union[str, Path],
    output_dir: Union[str, Path] = "benchmark_results/analysis",
    task_type: Literal["classification", "regression"] = "classification",
    enable_ttests: bool = True,
    enable_plots: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Execute complete statistical test suite and plot generation on benchmark CSV."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "Extractor" in df.columns and "Selector" in df.columns:
        df["Combination"] = df["Extractor"].astype(str) + " + " + df["Selector"].astype(str)

    # Standard metric sets
    common_metrics = [
        "Extraction Time (s)",
        "Selection Time (s)",
        "Prediction Time (s)",
        "Total Time (s)",
        "Extraction Peak RAM (MB)",
        "N Extracted Features",
        "N Selected Features",
    ]
    perf_metrics = ["Accuracy"] if task_type == "classification" else ["RMSE", "MAE", "R2"]
    all_metrics = [m for m in common_metrics + perf_metrics if m in df.columns]

    outputs: Dict[str, pd.DataFrame] = {}

    # 1. Summary Statistics
    summary_df = compute_summary_statistics(df, group_cols=["Extractor", "Selector"], metrics=all_metrics)
    summary_df.to_csv(out_path / "summary_statistics.csv")
    outputs["summary"] = summary_df

    # 2. Pairwise t-tests
    if enable_ttests:
        for group in ["Extractor", "Selector", "Combination"]:
            if group not in df.columns:
                continue
            for metric in all_metrics:
                ttest_res = pairwise_ttests(df, group_col=group, metric=metric)
                if not ttest_res.empty:
                    clean_metric = metric.lower().replace(" ", "_").replace("(", "").replace(")", "")
                    fname = f"{group.lower()}_{clean_metric}_ttests.csv"
                    ttest_res.to_csv(out_path / fname, index=False)
                    outputs[f"ttest_{group}_{clean_metric}"] = ttest_res

    # 3. Boxplots
    if enable_plots:
        for group in ["Extractor", "Selector"]:
            if group not in df.columns:
                continue
            for metric in all_metrics:
                clean_metric = metric.lower().replace(" ", "_").replace("(", "").replace(")", "")
                plot_metric_boxplot(
                    df,
                    metric=metric,
                    group_col=group,
                    output_path=out_path / f"{clean_metric}_by_{group.lower()}.png",
                    log_scale=False,
                )
                if "RAM" in metric or "Time" in metric:
                    plot_metric_boxplot(
                        df,
                        metric=metric,
                        group_col=group,
                        output_path=out_path / f"{clean_metric}_by_{group.lower()}_log.png",
                        log_scale=True,
                        title=f"{metric} (Log Scale) by {group}",
                    )

    logger.info("Statistical analysis complete. Artifacts saved to: %s", out_path)
    return outputs
