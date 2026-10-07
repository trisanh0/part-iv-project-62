"""
Statistical Analysis and Visualization Engine for the TEMPO Framework.

Performs multi-hypothesis paired t-tests with Benjamini-Hochberg False Discovery Rate (FDR)
adjustments, Cohen's d effect size calculation, summary statistics, and publication-ready
linear/log-scale boxplots across feature extraction and selection benchmarks.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

from itertools import combinations
import json
import logging
from pathlib import Path
import re
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

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
        "Forecast RMSE",
        "Forecast MAE",
        "Mean Interval Width",
        "N Extracted Features",
        "N Selected Features",
        "fit_time_seconds",
        "inference_latency_ms",
        "Fit Time (s)",
        "Inference Latency (ms)",
        "Tau RMSE",
        "Tau MAE",
    }
    return "lower" if metric in lower_is_better else "higher"


def compute_summary_statistics(
    df: pd.DataFrame,
    group_cols: List[str],
    metrics: List[str],
) -> pd.DataFrame:
    """Compute mean, std, median, min, max across grouped dimensions."""
    valid_metrics = [m for m in metrics if m in df.columns]
    if not valid_metrics:
        return pd.DataFrame()
    df_grouped = df.copy()
    for col in group_cols:
        if col in df_grouped.columns:
            df_grouped[col] = df_grouped[col].fillna("None").astype(str)
    summary = df_grouped.groupby(group_cols, dropna=False)[valid_metrics].agg(["mean", "std", "median", "min", "max"])
    return summary


def pairwise_ttests(
    df: pd.DataFrame,
    group_col: str,
    metric: str,
    alpha: float = 0.05,
    pair_keys: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Perform pairwise paired Student's t-tests with Benjamini-Hochberg FDR correction.
    
    Args:
        df: Telemetry results DataFrame.
        group_col: Column to group by ('Extractor', 'Selector', or 'Combination').
        metric: Target evaluation metric column name.
        alpha: Significance threshold (default 0.05).
        pair_keys: Optional explicit pairing key columns. If None, dynamically includes
            ["Dataset", "Seed", "Fold", "Task", "Selector", "Model", "Extractor"]
            (excluding group_col) to eliminate Cartesian pseudo-replication.
        
    Returns:
        DataFrame containing pairwise comparisons, p-values, corrected p-values, and effect sizes.
    """
    if metric not in df.columns or group_col not in df.columns:
        return pd.DataFrame()

    df_eval = df.copy()
    for col in ["Extractor", "Selector", "Model"]:
        if col in df_eval.columns:
            df_eval[col] = df_eval[col].fillna("None").astype(str)

    groups = [g for g in df_eval[group_col].unique() if pd.notna(g)]
    if len(groups) < 2:
        return pd.DataFrame()

    results = []
    direction = metric_direction(metric)

    # Exclude cached loads if analyzing extraction duration, RAM, or Total Time
    if "is_cached" in df_eval.columns and ("Extraction" in metric or "Total Time" in metric):
        df_eval = df_eval[df_eval["is_cached"] != True]

    # Required pairing key to ensure matched pairs across conditions without Cartesian pseudo-replication
    if pair_keys is None:
        candidate_keys = ["Dataset", "Seed", "Fold", "Task", "Selector", "Model", "Extractor"]
        exclude_cols = {group_col}
        if group_col == "Extractor":
            exclude_cols.add("Combination")
        elif group_col == "Selector":
            exclude_cols.add("Combination")
        elif group_col == "Combination":
            exclude_cols.update({"Extractor", "Selector", "Combination"})
        elif group_col == "Model":
            exclude_cols.add("Combination")

        pair_keys = [c for c in candidate_keys if c in df_eval.columns and c not in exclude_cols]
        if not pair_keys:
            pair_keys = [c for c in ["Dataset", "Seed", "Fold"] if c in df_eval.columns and c not in exclude_cols]

    # Deduplicate on pair_keys + group_col to guarantee strict 1:1 pair matching
    eval_cols = [group_col] + pair_keys
    if all(c in df_eval.columns for c in eval_cols):
        df_eval = df_eval.drop_duplicates(subset=eval_cols, keep="last")

    for g1, g2 in combinations(groups, 2):
        df_g1 = df_eval[df_eval[group_col] == g1].dropna(subset=[metric])
        df_g2 = df_eval[df_eval[group_col] == g2].dropna(subset=[metric])

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

    boxplot_kwargs = {
        "patch_artist": True,
        "medianprops": {"color": "#111827", "linewidth": 2},
        "boxprops": {"facecolor": "#E0E7FF", "edgecolor": "#4338CA", "linewidth": 1.5},
        "whiskerprops": {"color": "#4338CA", "linewidth": 1.2},
        "capprops": {"color": "#4338CA", "linewidth": 1.2},
    }
    try:
        bp = ax.boxplot(data, tick_labels=groups, **boxplot_kwargs)
    except TypeError:
        bp = ax.boxplot(data, labels=groups, **boxplot_kwargs)

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


def plot_forecast_fan_chart(
    history: np.ndarray,
    true_future: np.ndarray,
    predictions: Dict[str, np.ndarray],
    lower_bounds: Optional[Dict[str, np.ndarray]] = None,
    upper_bounds: Optional[Dict[str, np.ndarray]] = None,
    title: str = "Forecast vs Actual Trajectory",
    output_path: Optional[Union[str, Path]] = None,
    history_len: Optional[int] = None,
    forecast_horizon: Optional[int] = None,
) -> plt.Figure:
    """Generate publication-quality time-series forecast fan chart.

    Renders historical observations up to the forecast origin (t=0) followed by
    predicted multi-step trajectories with shaded uncertainty intervals against
    the ground-truth trajectory.

    Args:
        history: 1D array of observed historical series values leading to origin.
        true_future: 1D array of actual future target values.
        predictions: Mapping from model/pipeline name to 1D predicted array.
        lower_bounds: Optional mapping from model/pipeline name to 1D lower interval array.
        upper_bounds: Optional mapping from model/pipeline name to 1D upper interval array.
        title: Title for the figure.
        output_path: Optional file path to export the plot.
        history_len: Optional explicit history length (defaults to len(history)).
        forecast_horizon: Optional explicit horizon length (defaults to len(true_future)).

    Returns:
        Matplotlib Figure object.
    """
    hist_arr = np.asarray(history, dtype=float).ravel()
    true_arr = np.asarray(true_future, dtype=float).ravel()

    h_len = history_len if history_len is not None else len(hist_arr)
    horizon = forecast_horizon if forecast_horizon is not None else len(true_arr)

    if len(hist_arr) > h_len:
        hist_arr = hist_arr[-h_len:]
    elif len(hist_arr) < h_len:
        h_len = len(hist_arr)

    if len(true_arr) > horizon:
        true_arr = true_arr[:horizon]
    elif len(true_arr) < horizon:
        horizon = len(true_arr)

    x_past = np.arange(-h_len + 1, 1)
    x_future = np.arange(1, horizon + 1)

    fig, ax = plt.subplots(figsize=(12.5, 6.2), dpi=120)

    # Shaded future region and origin divider
    ax.axvspan(0, horizon + 0.5, color="#F8FAFC", alpha=0.8, zorder=0)
    ax.axvline(0, color="#64748B", linestyle="--", linewidth=1.5, alpha=0.75, zorder=2)

    # Observed history trajectory
    ax.plot(x_past, hist_arr, color="#334155", linewidth=2.4, label="Observed History", zorder=4)
    if len(x_past) > 0:
        ax.scatter(0, hist_arr[-1], color="#334155", s=40, zorder=5)

    # Ground truth future trajectory
    ax.plot(
        x_future, true_arr, color="#0F172A", linewidth=2.6, marker="o", markersize=4.5,
        label="Actual Future", zorder=6
    )

    # Plot predictions and prediction intervals
    palette = ["#2563EB", "#059669", "#D97706", "#7C3AED", "#DC2626", "#0891B2"]
    for idx, (name, pred) in enumerate(predictions.items()):
        color = palette[idx % len(palette)]
        p_arr = np.asarray(pred, dtype=float).ravel()[:horizon]
        ax.plot(x_future, p_arr, color=color, linewidth=2.0, marker="s", markersize=3.8, label=name, zorder=5)

        if lower_bounds and name in lower_bounds and upper_bounds and name in upper_bounds:
            low_arr = np.asarray(lower_bounds[name], dtype=float).ravel()[:horizon]
            upp_arr = np.asarray(upper_bounds[name], dtype=float).ravel()[:horizon]
            ax.fill_between(x_future, low_arr, upp_arr, color=color, alpha=0.18, linewidth=0, zorder=1)

    ax.set_title(title, pad=14, fontweight="bold")
    ax.set_xlabel("Time Relative to Forecast Origin")
    ax.set_ylabel("Series Value")

    # Clean ticks
    tick_pos = []
    tick_labels = []
    if h_len > 1:
        tick_pos.extend([x_past[0], -max(1, h_len // 2)])
        tick_labels.extend([f"-{h_len - 1}", f"-{h_len // 2}"])
    elif h_len == 1:
        tick_pos.append(x_past[0])
        tick_labels.append("0")

    if 0 not in tick_pos:
        tick_pos.append(0)
        tick_labels.append("Origin (t=0)")

    if horizon > 1:
        mid_h = horizon // 2
        if mid_h > 0 and mid_h < horizon:
            tick_pos.extend([mid_h, horizon])
            tick_labels.extend([f"+{mid_h}", f"+{horizon}"])
        else:
            tick_pos.append(horizon)
            tick_labels.append(f"+{horizon}")
    elif horizon == 1:
        tick_pos.append(1)
        tick_labels.append("+1")

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels)

    ax.margins(x=0.02)
    ax.grid(axis="y", alpha=0.25, linestyle="-")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9)

    fig.tight_layout()

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_horizon_metric_profiles(
    df: pd.DataFrame,
    output_dir: Union[str, Path],
    method_col: str = "Combination",
) -> Dict[str, plt.Figure]:
    """Plot performance metric trajectories across forecast horizons h in [1, H].

    Flexibly matches standard underscore column names (e.g., 'Interval_Width_H1', 'RMSE_H1')
    and space-separated variants (e.g., 'Interval Width H1', 'Interval Width 1', 'RMSE H1').
    Generates comparative multi-combination plots and individual combination profiles with
    ±1 standard deviation error bands.

    Args:
        df: Benchmark results DataFrame containing horizon-wise metric columns.
        output_dir: Directory path to save generated horizon plots.
        method_col: Column identifying pipeline combination.

    Returns:
        Dictionary mapping metric names and combinations to Matplotlib Figure objects.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures: Dict[str, plt.Figure] = {}

    metric_prefixes = ["RMSE", "MAE", "Coverage", "Interval_Width"]
    for prefix in metric_prefixes:
        name_pat = re.escape(prefix).replace("_", r"[\s_]+")
        pattern = rf"^{name_pat}[\s_]*(?:\(?\s*H[\s_]*)?(\d+)\)?$"

        cols_by_step: Dict[int, str] = {}
        for c in df.columns:
            m = re.fullmatch(pattern, str(c).strip(), re.IGNORECASE)
            if m:
                step = int(m.group(1))
                if step not in cols_by_step:
                    cols_by_step[step] = c
                else:
                    existing_c = cols_by_step[step]
                    if df[existing_c].isna().sum() > df[c].isna().sum():
                        cols_by_step[step] = c

        if not cols_by_step:
            continue

        sorted_steps = sorted(cols_by_step.keys())
        cols = [cols_by_step[s] for s in sorted_steps]
        horizons = np.array(sorted_steps)
        clean_title = prefix.replace("_", " ")

        if method_col in df.columns and len(df[method_col].dropna().unique()) > 0:
            methods = sorted(df[method_col].dropna().unique(), key=lambda x: str(x))
        else:
            methods = ["All"]

        # 1. Comparative multi-combination horizon plot
        fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=120)
        for method in methods:
            if method_col in df.columns and method != "All":
                sub = df[df[method_col] == method]
            else:
                sub = df
            vals = sub[cols].apply(pd.to_numeric, errors="coerce")
            mean_vals = vals.mean(axis=0).to_numpy()
            std_vals = vals.std(axis=0).fillna(0.0).to_numpy()

            ax.plot(horizons, mean_vals, marker="o", markersize=4, linewidth=1.8, label=str(method))
            if len(sub) > 1:
                ax.fill_between(horizons, mean_vals - std_vals, mean_vals + std_vals, alpha=0.15)

        ax.set_title(f"{clean_title} Across Forecast Horizon", pad=12, fontweight="bold")
        ax.set_xlabel("Forecast Horizon Step (h)")
        ax.set_ylabel(f"Mean {clean_title}")
        if len(horizons) <= 15:
            ax.set_xticks(horizons)
        else:
            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
        ax.grid(axis="y", alpha=0.25)
        ax.grid(axis="x", alpha=0.15)
        if len(methods) > 1 or methods[0] != "All":
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8.5)
        fig.tight_layout()

        fig_path = out_dir / f"horizon_{prefix.lower()}_profile.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        if prefix.lower() == "interval_width":
            fig.savefig(out_dir / "interval_width_across_horizon_all_combinations.png", dpi=300, bbox_inches="tight")
        figures[prefix] = fig
        plt.close(fig)

        # 2. Individual combination profile plots with ±1 std fill bands
        for method in methods:
            if method_col in df.columns and method != "All":
                sub = df[df[method_col] == method]
            else:
                sub = df
            vals = sub[cols].apply(pd.to_numeric, errors="coerce")
            mean_vals = vals.mean(axis=0).to_numpy()
            std_vals = vals.std(axis=0).fillna(0.0).to_numpy()

            fig_ind, ax_ind = plt.subplots(figsize=(9.5, 5.0), dpi=120)
            ax_ind.plot(horizons, mean_vals, marker="o", markersize=4.5, linewidth=2.0, color="#1f77b4", label=f"Mean {clean_title}")
            ax_ind.fill_between(
                horizons,
                mean_vals - std_vals,
                mean_vals + std_vals,
                alpha=0.20,
                color="#1f77b4",
                label="$\pm 1$ std",
            )
            title_suffix = f"\n{method}" if method != "All" else ""
            ax_ind.set_title(f"{clean_title} Across Forecast Horizon{title_suffix}", pad=12, fontweight="bold")
            ax_ind.set_xlabel("Forecast Horizon Step (h)")
            ax_ind.set_ylabel(f"Mean {clean_title}")
            if len(horizons) <= 15:
                ax_ind.set_xticks(horizons)
            else:
                from matplotlib.ticker import MaxNLocator
                ax_ind.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
            ax_ind.grid(axis="y", alpha=0.25)
            ax_ind.grid(axis="x", alpha=0.15)
            ax_ind.legend(loc="best", frameon=False, fontsize=9.0)
            fig_ind.tight_layout()

            safe_method = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(method)).strip("_")
            ind_path = out_dir / f"horizon_{prefix.lower()}_{safe_method}.png"
            fig_ind.savefig(ind_path, dpi=300, bbox_inches="tight")
            if prefix.lower() == "interval_width":
                fig_ind.savefig(out_dir / f"interval_width_horizon_{safe_method}.png", dpi=300, bbox_inches="tight")

            figures[f"{prefix}_{safe_method}"] = fig_ind
            plt.close(fig_ind)

    return figures


def plot_interval_width_boxplot(
    df: pd.DataFrame,
    output_dir: Union[str, Path],
    metric: str = "Mean Interval Width",
    group_col: str = "Combination",
) -> Optional[plt.Figure]:
    """Generate comparison of uncertainty widths across pipeline combinations."""
    if metric not in df.columns or group_col not in df.columns:
        return None

    p = df[[group_col, metric]].copy()
    p[metric] = pd.to_numeric(p[metric], errors="coerce")
    p = p.dropna()
    if p.empty:
        return None

    order = p.groupby(group_col)[metric].median().sort_values().index.tolist()
    values = [p.loc[p[group_col] == name, metric].to_numpy() for name in order]

    fig_height = max(5.0, 0.45 * len(order) + 2.0)
    fig, ax = plt.subplots(figsize=(11.5, fig_height), dpi=300)
    boxplot_kwargs = {
        "patch_artist": True,
        "widths": 0.58,
        "showfliers": False,
        "medianprops": {"color": "#111827", "linewidth": 2.2},
        "boxprops": {"facecolor": "#DCFCE7", "edgecolor": "#059669", "linewidth": 1.2},
        "whiskerprops": {"color": "#64748B", "linewidth": 1.2},
        "capprops": {"color": "#64748B", "linewidth": 1.2},
    }
    try:
        ax.boxplot(values, orientation="horizontal", tick_labels=order, **boxplot_kwargs)
    except TypeError:
        try:
            ax.boxplot(values, vert=False, tick_labels=order, **boxplot_kwargs)
        except TypeError:
            ax.boxplot(values, vert=False, labels=order, **boxplot_kwargs)

    ax.set_title("Uncertainty Interval Width — All Combinations", pad=14, fontweight="bold")
    ax.set_xlabel("Mean Interval Width")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.20)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8.5)
    fig.tight_layout()

    out_path = Path(output_dir) / "uncertainty_interval_width_all_combinations.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return fig


def plot_interval_width_mean_bar(
    df: pd.DataFrame,
    output_dir: Union[str, Path],
    metric: str = "Mean Interval Width",
    group_col: str = "Combination",
) -> Optional[plt.Figure]:
    """Generate horizontal bar chart comparing mean uncertainty interval widths."""
    if metric not in df.columns or group_col not in df.columns:
        return None

    p = df[[group_col, metric]].copy()
    p[metric] = pd.to_numeric(p[metric], errors="coerce")
    p = p.dropna()
    if p.empty:
        return None

    g = p.groupby(group_col)[metric].agg(["mean", "std"]).sort_values("mean")
    if g.empty:
        return None

    fig_height = max(5.0, 0.45 * len(g) + 2.0)
    fig, ax = plt.subplots(figsize=(11.5, fig_height), dpi=300)
    y = np.arange(len(g))
    ax.barh(
        y,
        g["mean"].to_numpy(),
        xerr=g["std"].fillna(0.0).to_numpy(),
        capsize=3,
        color="#86EFAC",
        edgecolor="#059669",
        linewidth=0.8,
    )
    ax.set_title("Mean Uncertainty Interval Width", pad=14, fontweight="bold")
    ax.set_xlabel("Mean Interval Width ± 1 SD")
    ax.set_ylabel("")
    ax.set_yticks(y)
    ax.set_yticklabels(g.index, fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.20)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()

    out_path = Path(output_dir) / "uncertainty_interval_width_mean_all_combinations.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return fig


def generate_forecast_visualizations(
    df: pd.DataFrame,
    output_dir: Union[str, Path],
    sample_index: int = 0,
) -> List[str]:
    """Parse JSON-encoded forecast benchmark records and export fan charts and horizon profiles.

    Args:
        df: Benchmark results DataFrame.
        output_dir: Root output directory path.
        sample_index: Index of the test sample window to visualize (default 0).

    Returns:
        List of generated image file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_files: List[str] = []

    # 1. Horizon Profiles
    plot_horizon_metric_profiles(df, output_dir=out_dir)
    for f in out_dir.glob("horizon_*.png"):
        f_str = str(f)
        if f_str not in generated_files:
            generated_files.append(f_str)
    for leg_name in ["interval_width_across_horizon_all_combinations.png"]:
        leg_p = out_dir / leg_name
        if leg_p.exists() and str(leg_p) not in generated_files:
            generated_files.append(str(leg_p))

    # 2. Uncertainty Interval Width plots (Scott benchmark compatibility)
    if "Mean Interval Width" in df.columns and "Combination" in df.columns:
        fig_box = plot_interval_width_boxplot(df, output_dir=out_dir)
        if fig_box is not None:
            box_p = str(out_dir / "uncertainty_interval_width_all_combinations.png")
            if box_p not in generated_files:
                generated_files.append(box_p)
        fig_bar = plot_interval_width_mean_bar(df, output_dir=out_dir)
        if fig_bar is not None:
            bar_p = str(out_dir / "uncertainty_interval_width_mean_all_combinations.png")
            if bar_p not in generated_files:
                generated_files.append(bar_p)

    # 3. Fan Charts per Dataset and Seed
    req_cols = ["History Values", "True Values", "Predictions"]
    if not all(c in df.columns for c in req_cols):
        logger.info("Forecast fan charts skipped; missing sequence columns: %s", [c for c in req_cols if c not in df.columns])
        return generated_files

    summary_rows = []

    # 3. Fan Charts per Dataset and Seed
    for (ds, seed), group_df in df.groupby(["Dataset", "Seed"]):
        clean_ds = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(ds)).strip("_")
        clean_seed = str(seed)
        seed_dir = out_dir / "forecast_plots" / clean_ds / f"seed_{clean_seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        all_preds: Dict[str, np.ndarray] = {}
        all_lowers: Dict[str, np.ndarray] = {}
        all_uppers: Dict[str, np.ndarray] = {}
        first_hist: Optional[np.ndarray] = None
        first_true: Optional[np.ndarray] = None

        for _, row in group_df.iterrows():
            model_name = str(row.get("Model", ""))
            base_combo = str(row.get("Combination", f"{row.get('Extractor', 'none')}_{row.get('Selector', 'none')}"))
            if model_name and model_name not in ("None", "model", "") and "Model" in group_df.columns and group_df["Model"].nunique() > 1:
                combo = f"{base_combo} ({model_name})"
            else:
                combo = base_combo

            try:
                hist_mat = np.asarray(json.loads(row["History Values"]), dtype=float)
                true_mat = np.asarray(json.loads(row["True Values"]), dtype=float)
                pred_mat = np.asarray(json.loads(row["Predictions"]), dtype=float)
                low_mat = np.asarray(json.loads(row["Prediction Lower"]), dtype=float) if "Prediction Lower" in row and pd.notna(row["Prediction Lower"]) else None
                upp_mat = np.asarray(json.loads(row["Prediction Upper"]), dtype=float) if "Prediction Upper" in row and pd.notna(row["Prediction Upper"]) else None
            except Exception as e:
                logger.warning("Failed to decode JSON forecast arrays for combo '%s': %s", combo, e)
                continue

            n_samples = hist_mat.shape[0] if hist_mat.ndim > 1 else 1
            s_idx = min(max(int(sample_index), 0), n_samples - 1)

            hist_vec = hist_mat[s_idx] if hist_mat.ndim > 1 else hist_mat
            true_vec = true_mat[s_idx] if true_mat.ndim > 1 else true_mat
            pred_vec = pred_mat[s_idx] if pred_mat.ndim > 1 else pred_mat
            low_vec = low_mat[s_idx] if low_mat is not None and low_mat.ndim > 1 else low_mat
            upp_vec = upp_mat[s_idx] if upp_mat is not None and upp_mat.ndim > 1 else upp_mat

            if first_hist is None:
                first_hist = hist_vec
                first_true = true_vec

            all_preds[combo] = pred_vec
            if low_vec is not None:
                all_lowers[combo] = low_vec
            if upp_vec is not None:
                all_uppers[combo] = upp_vec

            # Horizon-wise summary record accumulation
            n_horizons = pred_mat.shape[1] if pred_mat.ndim > 1 else (pred_mat.shape[0] if pred_mat.ndim == 1 else 1)
            for h in range(n_horizons):
                summary_rows.append({
                    "Dataset": str(ds),
                    "Seed": seed,
                    "Extractor": str(row.get("Extractor", "")),
                    "Selector": str(row.get("Selector", "")),
                    "Combination": combo,
                    "Horizon": h + 1,
                    "True Mean": float(np.mean(true_mat[:, h])) if true_mat.ndim > 1 else float(true_mat[h] if h < len(true_mat) else np.nan),
                    "Prediction Mean": float(np.mean(pred_mat[:, h])) if pred_mat.ndim > 1 else float(pred_mat[h] if h < len(pred_mat) else np.nan),
                    "Prediction Lower Mean": float(np.mean(low_mat[:, h])) if low_mat is not None and low_mat.ndim > 1 else (float(low_mat[h]) if low_mat is not None and h < len(low_mat) else np.nan),
                    "Prediction Upper Mean": float(np.mean(upp_mat[:, h])) if upp_mat is not None and upp_mat.ndim > 1 else (float(upp_mat[h]) if upp_mat is not None and h < len(upp_mat) else np.nan),
                })

            # Generate individual combination fan chart
            clean_combo = re.sub(r"[^A-Za-z0-9_.-]+", "_", combo).strip("_")
            ind_path = seed_dir / f"forecast_fan_{clean_combo}.png"
            plot_forecast_fan_chart(
                history=hist_vec,
                true_future=true_vec,
                predictions={combo: pred_vec},
                lower_bounds={combo: low_vec} if low_vec is not None else None,
                upper_bounds={combo: upp_vec} if upp_vec is not None else None,
                title=f"Forecast Fan Chart\n{ds} | Seed {seed} | {combo}",
                output_path=ind_path,
            )
            generated_files.append(str(ind_path))

        # Generate comparative fan chart across all combinations
        if first_hist is not None and first_true is not None and all_preds:
            comp_path = seed_dir / "forecast_fan_comparison_all_combinations.png"
            plot_forecast_fan_chart(
                history=first_hist,
                true_future=first_true,
                predictions=all_preds,
                lower_bounds=all_lowers,
                upper_bounds=all_uppers,
                title=f"Comparative Forecast Fan Chart\n{ds} | Seed {seed}",
                output_path=comp_path,
            )
            generated_files.append(str(comp_path))

    if summary_rows:
        summary_csv = out_dir / "forecast_prediction_visualisation_summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    return generated_files


def plot_critical_difference_diagram(
    df: pd.DataFrame,
    metric: str = "Accuracy",
    group_col: str = "Extractor",
    dataset_col: str = "Dataset",
    alpha: float = 0.05,
    title: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Optional[plt.Figure]:
    """Generate publication-ready Critical Difference (CD) diagram using pure Matplotlib.

    Computes average ranks across evaluation datasets, determines pairwise statistical
    significance using the Wilcoxon signed-rank test (or paired t-test), identifies
    non-significant cliques, and renders a clean Demšar-style horizontal rank diagram.

    Args:
        df: DataFrame containing benchmark telemetry results.
        metric: Performance metric column to evaluate (e.g. 'Accuracy', 'R2', 'RMSE').
        group_col: Grouping column representing evaluated methods ('Extractor', 'Selector').
        dataset_col: Column identifying evaluation datasets or benchmark splits.
        alpha: Statistical significance threshold (default: 0.05).
        title: Optional custom figure title.
        output_path: Optional file path to save figure artifact.

    Returns:
        Matplotlib Figure instance, or None if insufficient methods/data.
    """
    if df.empty or metric not in df.columns or group_col not in df.columns or dataset_col not in df.columns:
        return None

    pivoted = df.groupby([dataset_col, group_col])[metric].mean().unstack(group_col)
    pivoted = pivoted.dropna(axis=1, how="all").dropna(axis=0, how="any")

    if pivoted.empty or pivoted.shape[0] < 1 or pivoted.shape[1] < 2:
        logger.warning(
            "CD diagram requires at least 2 valid methods and at least 1 common evaluation dataset; found %d datasets and %d methods.",
            pivoted.shape[0] if not pivoted.empty else 0,
            pivoted.shape[1] if not pivoted.empty else 0,
        )
        return None

    direction = metric_direction(metric)
    ascending = True if direction == "lower" else False
    ranks = pivoted.rank(axis=1, ascending=ascending, method="average")
    avg_ranks = ranks.mean(axis=0).sort_values()

    methods = list(avg_ranks.index)
    k = len(methods)
    n_datasets = len(ranks)

    from scipy.stats import ttest_rel, wilcoxon

    sig_diff = np.zeros((k, k), dtype=bool)
    for i in range(k):
        for j in range(i + 1, k):
            m1, m2 = methods[i], methods[j]
            v1, v2 = pivoted[m1].to_numpy(), pivoted[m2].to_numpy()
            diff = v1 - v2
            if np.all(diff == 0):
                is_sig = False
            elif n_datasets >= 5:
                try:
                    _, p_val = wilcoxon(v1, v2)
                    is_sig = bool(p_val < alpha)
                except Exception:
                    _, p_val = ttest_rel(v1, v2)
                    is_sig = bool(p_val < alpha)
            else:
                _, p_val = ttest_rel(v1, v2)
                is_sig = bool(p_val < alpha)
            sig_diff[i, j] = is_sig
            sig_diff[j, i] = is_sig

    cliques = []
    for i in range(k):
        for j in range(i + 1, k):
            span = list(range(i, j + 1))
            all_non_sig = True
            for a in span:
                for b in span:
                    if a != b and sig_diff[a, b]:
                        all_non_sig = False
                        break
                if not all_non_sig:
                    break
            if all_non_sig:
                cliques.append((i, j))

    filtered_cliques = []
    for c in cliques:
        is_sub = False
        for other in cliques:
            if other != c and other[0] <= c[0] and other[1] >= c[1]:
                is_sub = True
                break
        if not is_sub and c not in filtered_cliques:
            filtered_cliques.append(c)

    fig, ax = plt.subplots(figsize=(max(8.0, k * 1.3), 3.5), dpi=100)

    rank_min, rank_max = 1.0, float(k)
    pad = 0.5
    ax.plot([rank_min - pad * 0.5, rank_max + pad * 0.5], [0, 0], color="black", lw=1.5)

    ticks = np.arange(1, k + 1)
    for t in ticks:
        ax.plot([t, t], [-0.05, 0.05], color="black", lw=1.2)
        ax.text(t, -0.15, str(t), ha="center", va="top", fontsize=9, fontweight="bold")

    split = int(np.ceil(k / 2))
    left_methods = methods[:split]
    right_methods = methods[split:]

    y_text_top = 0.4
    y_step = 0.22

    for idx, m in enumerate(left_methods):
        r = avg_ranks[m]
        y_pos = y_text_top + (len(left_methods) - 1 - idx) * y_step
        ax.plot(r, 0, marker="o", color="#1f77b4", markersize=6)
        x_text = rank_min - 0.2
        ax.plot([r, r, x_text], [0, y_pos, y_pos], color="#888888", lw=0.9, linestyle="--")
        label = f"{m} ({r:.2f})"
        ax.text(x_text - 0.05, y_pos, label, ha="right", va="center", fontsize=9, fontweight="semibold")

    for idx, m in enumerate(right_methods):
        r = avg_ranks[m]
        y_pos = y_text_top + idx * y_step
        ax.plot(r, 0, marker="o", color="#1f77b4", markersize=6)
        x_text = rank_max + 0.2
        ax.plot([r, r, x_text], [0, y_pos, y_pos], color="#888888", lw=0.9, linestyle="--")
        label = f"{m} ({r:.2f})"
        ax.text(x_text + 0.05, y_pos, label, ha="left", va="center", fontsize=9, fontweight="semibold")

    base_clique_y = -0.38
    for c_idx, (start_idx, end_idx) in enumerate(filtered_cliques):
        c_y = base_clique_y - c_idx * 0.12
        r_start = avg_ranks[methods[start_idx]]
        r_end = avg_ranks[methods[end_idx]]
        ax.plot([r_start, r_end], [c_y, c_y], color="#333333", lw=3.5, solid_capstyle="round")

    max_label_len = max(len(str(m)) for m in methods) if methods else 10
    x_margin = max(1.8, max_label_len * 0.08)
    ax.set_ylim(base_clique_y - len(filtered_cliques) * 0.12 - 0.2, y_text_top + max(len(left_methods), len(right_methods), 1) * y_step + 0.1)
    ax.set_xlim(rank_min - x_margin, rank_max + x_margin)
    ax.axis("off")

    plot_title = title or f"Critical Difference Diagram ({group_col} by {metric})"
    ax.set_title(plot_title, fontsize=11, fontweight="bold", pad=12)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info("Saved Critical Difference diagram to: %s", output_path)

    return fig


def plot_tau_estimation_scatter(
    y_true: Union[np.ndarray, pd.Series, List[float]],
    y_pred: Union[np.ndarray, pd.Series, List[float]],
    title: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Optional[plt.Figure]:
    """Generate true vs predicted tau scatter plot with regression diagnostics.

    Evaluates R^2, RMSE, and MAE for extrinsic parameter estimation on drift-bifurcation
    dynamics and plots empirical predictions against the theoretical identity line (y = x).

    Args:
        y_true: Ground truth continuous parameter tau values.
        y_pred: Estimated/predicted continuous parameter tau values.
        title: Optional custom figure title.
        output_path: Optional file path to save figure artifact.

    Returns:
        Matplotlib Figure instance, or None if inputs are empty or invalid.
    """
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()

    if len(y_t) != len(y_p):
        logger.warning("y_true and y_pred must have identical lengths for tau scatter plot (got %d vs %d).", len(y_t), len(y_p))
        return None

    mask = np.isfinite(y_t) & np.isfinite(y_p)
    if not np.any(mask):
        logger.warning("No valid finite values found for tau estimation scatter plot.")
        return None

    y_t = y_t[mask]
    y_p = y_p[mask]

    n_samples = len(y_t)
    diff = y_p - y_t
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))

    ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
    ss_res = float(np.sum(diff ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    fig, ax = plt.subplots(figsize=(6.0, 5.5), dpi=100)

    ax.scatter(
        y_t,
        y_p,
        alpha=0.65,
        color="#1f77b4",
        edgecolors="#0d3d63",
        linewidths=0.6,
        s=40,
        label="Predictions",
    )

    min_val = min(float(np.min(y_t)), float(np.min(y_p)))
    max_val = max(float(np.max(y_t)), float(np.max(y_p)))
    span = max_val - min_val
    margin = 0.08 * span if span > 0 else 0.2
    lims = [min_val - margin, max_val + margin]

    ax.plot(lims, lims, color="#d62728", linestyle="--", lw=1.8, label="Identity line ($y = x$)")

    metrics_str = f"$R^2 = {r2:.4f}$\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}\n$N = {n_samples}$"
    ax.text(
        0.05,
        0.95,
        metrics_str,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.9),
        fontsize=9.5,
    )

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True Bifurcation Parameter $\\tau$", fontsize=10.5, fontweight="semibold")
    ax.set_ylabel("Predicted Bifurcation Parameter $\\hat{\\tau}$", fontsize=10.5, fontweight="semibold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, fontsize=9.5)

    plot_title = title or "Extrinsic Parameter Estimation: True vs. Predicted $\\tau$"
    ax.set_title(plot_title, fontsize=11, fontweight="bold", pad=10)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info("Saved tau estimation scatter plot to: %s", output_path)

    return fig


def plot_tau_recovery_points(
    tau_values: Union[np.ndarray, pd.Series, Sequence[float]],
    predicted_by_tau: Union[np.ndarray, pd.Series, Sequence[float]],
    title: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Optional[plt.Figure]:
    """Plot discrete parameter recovery diagnostics (true vs predicted tau).

    Visualizes discrete condition parameter recovery (e.g. 5-point drift-bifurcation
    benchmark seeds) comparing true parameter values (red markers) against predicted
    values (blue markers) across conditions.

    Args:
        tau_values: Sequence of ground-truth tau parameter values.
        predicted_by_tau: Sequence of predicted tau values corresponding to tau_values.
        title: Optional custom figure title.
        output_path: Optional file path to save figure artifact.

    Returns:
        Matplotlib Figure instance, or None if inputs are empty or invalid.
    """
    if tau_values is None or predicted_by_tau is None:
        logger.warning("tau_values and predicted_by_tau must not be None.")
        return None

    try:
        tau_arr = np.asarray(tau_values, dtype=float).ravel()
        pred_arr = np.asarray(predicted_by_tau, dtype=float).ravel()
    except (ValueError, TypeError) as e:
        logger.warning("Could not convert tau inputs to numeric arrays: %s", e)
        return None

    if len(tau_arr) == 0 or len(pred_arr) == 0:
        logger.warning("Empty array provided for tau recovery points.")
        return None

    if len(tau_arr) != len(pred_arr):
        logger.warning(
            "tau_values and predicted_by_tau must have identical lengths (got %d vs %d).",
            len(tau_arr),
            len(pred_arr),
        )
        return None

    if not np.any(np.isfinite(tau_arr)) and not np.any(np.isfinite(pred_arr)):
        logger.warning("No valid finite values found for tau recovery plot.")
        return None

    n_points = len(tau_arr)
    x = np.arange(1, n_points + 1)

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=120)

    for xi, yt, yp in zip(x, tau_arr, pred_arr):
        if np.isfinite(yt) and np.isfinite(yp):
            ax.plot([xi, xi], [yt, yp], color="#94A3B8", linestyle="--", linewidth=1.2, zorder=3)

    ax.scatter(
        x,
        tau_arr,
        s=85,
        color="#DC2626",
        edgecolor="#991B1B",
        linewidth=0.8,
        label="True $\\tau$",
        zorder=4,
    )
    ax.scatter(
        x,
        pred_arr,
        s=85,
        color="#2563EB",
        edgecolor="#1D4ED8",
        linewidth=0.8,
        label="Predicted $\\hat{\\tau}$",
        zorder=5,
    )

    plot_title = title or "Discrete $\\tau$ Parameter Recovery"
    ax.set_title(plot_title, pad=14, fontweight="bold")
    ax.set_xlabel("Condition / $\\tau$ Index", fontweight="semibold")
    ax.set_ylabel("Parameter Value ($\\tau$)", fontweight="semibold")
    ax.set_xticks(x)
    ax.set_xlim(0.5, n_points + 0.5)
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", alpha=0.15)
    ax.legend(loc="best", frameon=True, fontsize=9.5)
    fig.tight_layout()

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=300, bbox_inches="tight")
        logger.info("Saved tau recovery points plot to: %s", out_p)
        plt.close(fig)

    return fig


def run_statistical_analysis(
    csv_path: Union[str, Path],
    output_dir: Union[str, Path] = "benchmark_results/analysis",
    task_type: Literal["classification", "regression", "forecasting"] = "classification",
    enable_ttests: bool = True,
    enable_plots: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Execute complete statistical test suite and plot generation on benchmark CSV."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    for col in ["Extractor", "Selector", "Model"]:
        if col in df.columns:
            df[col] = df[col].fillna("None").astype(str)

    if "Extractor" in df.columns and "Selector" in df.columns:
        df["Combination"] = df["Extractor"].astype(str) + " + " + df["Selector"].astype(str)

    # Standard metric sets
    common_metrics = [
        "Extraction Time (s)",
        "Selection Time (s)",
        "fit_time_seconds",
        "inference_latency_ms",
        "Fit Time (s)",
        "Inference Latency (ms)",
        "Prediction Time (s)",
        "Total Time (s)",
        "Extraction Peak RAM (MB)",
        "N Extracted Features",
        "N Selected Features",
        "Feature Reduction (%)",
        "Selection Stability (Jaccard)",
    ]

    outputs: Dict[str, pd.DataFrame] = {}

    # Separate classification, regression, and forecasting summaries
    tasks = [t for t in df["Task"].dropna().unique()] if "Task" in df.columns else [task_type]
    if not tasks:
        tasks = [task_type]

    for tsk in tasks:
        df_task = df[df["Task"] == tsk] if "Task" in df.columns else df.copy()
        if tsk == "classification":
            perf_metrics = ["Accuracy"]
        elif tsk == "forecasting":
            perf_metrics = ["Forecast RMSE", "Forecast MAE", "Interval Coverage", "Mean Interval Width", "RMSE", "MAE"]
        else:
            perf_metrics = ["RMSE", "MAE", "R2"]
            tau_cols = [c for c in ["Tau RMSE", "Tau MAE", "Tau R2"] if c in df_task.columns]
            if tau_cols:
                perf_metrics.extend(tau_cols)

        task_metrics = [m for m in common_metrics + perf_metrics if m in df_task.columns]

        # Exclude cached loads from extraction runtime, hardware, and total time calculations
        df_task_clean = df_task.copy()
        if "is_cached" in df_task_clean.columns:
            ext_cols = [
                "Extraction Time (s)",
                "Extraction Peak RAM (MB)",
                "Extraction Peak RAM Increase (MB)",
                "Extraction Avg CPU (%)",
                "Extraction Peak CPU (%)",
                "Extraction Peak GPU (%)",
                "Extraction Peak GPU RAM (MB)",
                "Total Time (s)",
            ]
            for ec in ext_cols:
                if ec in df_task_clean.columns:
                    df_task_clean.loc[df_task_clean["is_cached"] == True, ec] = np.nan

        # 1. Summary Statistics
        summary_task = compute_summary_statistics(
            df_task_clean, group_cols=["Extractor", "Selector"], metrics=task_metrics
        )
        task_suffix = f"_{tsk}" if len(tasks) > 1 or "Task" in df.columns else ""
        summary_task.to_csv(out_path / f"summary_statistics{task_suffix}.csv")
        outputs[f"summary{task_suffix}"] = summary_task

        # 2. Pairwise t-tests
        if enable_ttests:
            for group in ["Extractor", "Selector", "Combination"]:
                if group not in df_task_clean.columns:
                    continue
                for metric in task_metrics:
                    ttest_res = pairwise_ttests(df_task_clean, group_col=group, metric=metric)
                    if not ttest_res.empty:
                        clean_metric = metric.lower().replace(" ", "_").replace("(", "").replace(")", "")
                        fname = f"{group.lower()}_{clean_metric}{task_suffix}_ttests.csv"
                        ttest_res.to_csv(out_path / fname, index=False)
                        outputs[f"ttest_{group}_{clean_metric}{task_suffix}"] = ttest_res

        # 3. Boxplots
        if enable_plots:
            for group in ["Extractor", "Selector"]:
                if group not in df_task_clean.columns:
                    continue
                for metric in task_metrics:
                    clean_metric = metric.lower().replace(" ", "_").replace("(", "").replace(")", "")
                    plot_metric_boxplot(
                        df_task_clean,
                        metric=metric,
                        group_col=group,
                        output_path=out_path / f"{clean_metric}{task_suffix}_by_{group.lower()}.png",
                        log_scale=False,
                    )
                    if "RAM" in metric or "Time" in metric:
                        plot_metric_boxplot(
                            df_task_clean,
                            metric=metric,
                            group_col=group,
                            output_path=out_path / f"{clean_metric}{task_suffix}_by_{group.lower()}_log.png",
                            log_scale=True,
                            title=f"{metric} (Log Scale) by {group}",
                        )

            # 4. Fan Charts & Horizon Profiles for forecasting
            if tsk == "forecasting" or any(c in df_task.columns for c in ["History Values", "Predictions"]):
                generate_forecast_visualizations(df_task, output_dir=out_path)

            # 5. Critical Difference (CD) Diagrams
            primary_metric = perf_metrics[0] if perf_metrics else "Accuracy"
            if primary_metric in df_task_clean.columns and "Dataset" in df_task_clean.columns:
                for group in ["Extractor", "Selector"]:
                    if group in df_task_clean.columns:
                        cd_path = out_path / f"cd_diagram_{group.lower()}_{primary_metric.lower().replace(' ', '_')}{task_suffix}.png"
                        fig_cd = plot_critical_difference_diagram(
                            df_task_clean,
                            metric=primary_metric,
                            group_col=group,
                            dataset_col="Dataset",
                            output_path=cd_path,
                        )
                        if fig_cd is not None:
                            outputs[f"cd_{group.lower()}{task_suffix}"] = fig_cd
                            plt.close(fig_cd)

            # 6. Drift-bifurcation tau estimation diagnostics
            is_tau_task = False
            if "Dataset" in df_task.columns:
                is_tau_task = any(df_task["Dataset"].astype(str).str.contains("driftbif|bifurcation|tau").tolist())
            if not is_tau_task and any(c in df_task.columns for c in ["Tau RMSE", "Tau MAE", "Tau R2"]):
                is_tau_task = True

            if is_tau_task and "True Values" in df_task.columns and "Predictions" in df_task.columns:
                for _, r in df_task.iterrows():
                    try:
                        tv_str = r.get("True Values")
                        pv_str = r.get("Predictions")
                        if tv_str and pv_str and not pd.isna(tv_str) and not pd.isna(pv_str):
                            t_vals = json.loads(tv_str)
                            p_vals = json.loads(pv_str)
                            scatter_path = out_path / f"tau_estimation_scatter_{r.get('Extractor', 'ext')}_{r.get('Selector', 'sel')}.png"
                            fig_tau = plot_tau_estimation_scatter(
                                t_vals,
                                p_vals,
                                title=f"Tau Parameter Estimation ({r.get('Extractor')} | {r.get('Selector')})",
                                output_path=scatter_path,
                            )
                            if fig_tau is not None:
                                outputs[f"tau_scatter_{r.get('Extractor', 'ext')}_{r.get('Selector', 'sel')}{task_suffix}"] = fig_tau
                    except Exception as e:
                        logger.warning("Could not generate tau estimation scatter plot (%s)", e)

            # 7. Discrete 5-point tau recovery diagnostics (Scott benchmark compatibility)
            has_discrete_tau_cols = any(c in df_task.columns for c in ["True Tau Values", "Test True Tau Values", "Predicted Tau Values"])
            if has_discrete_tau_cols:
                def _parse_seq(val):
                    if val is None:
                        return None
                    if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
                        try:
                            arr = np.asarray(val, dtype=float).ravel()
                            return arr if arr.size > 0 else None
                        except Exception:
                            return None
                    if isinstance(val, str):
                        s = val.strip()
                        if not s or s.lower() in ("nan", "none", "null"):
                            return None
                        try:
                            return np.asarray(json.loads(s), dtype=float).ravel()
                        except Exception:
                            try:
                                cleaned = s.strip("[]() ")
                                parts = [float(x) for x in re.split(r"[\s,]+", cleaned) if x]
                                return np.asarray(parts, dtype=float)
                            except Exception:
                                return None
                    return None

                for _, r in df_task.iterrows():
                    try:
                        tv_arr = _parse_seq(r.get("True Tau Values"))
                        tt_arr = _parse_seq(r.get("Test True Tau Values"))
                        pv_arr = _parse_seq(r.get("Predicted Tau Values"))
                        if tv_arr is not None and tt_arr is not None and pv_arr is not None:
                            unique_taus = np.sort(np.unique(tv_arr))
                            if len(unique_taus) > 0 and len(pv_arr) == len(tt_arr):
                                pred_by_tau = []
                                for tau in unique_taus:
                                    mask = np.isclose(tt_arr, tau, rtol=0.0, atol=1e-10)
                                    if np.any(mask):
                                        pred_by_tau.append(float(np.mean(pv_arr[mask])))
                                    else:
                                        pred_by_tau.append(np.nan)
                                combo = str(r.get("Combination", f"{r.get('Extractor', 'ext')}_{r.get('Selector', 'sel')}"))
                                safe_combo = re.sub(r"[^A-Za-z0-9_.-]+", "_", combo).strip("_")
                                tau_path = out_path / f"tau_recovery_{safe_combo}.png"
                                fig_rec = plot_tau_recovery_points(
                                    tau_values=unique_taus,
                                    predicted_by_tau=pred_by_tau,
                                    title=f"Inverse Regression Tau Recovery\n{combo}",
                                    output_path=tau_path,
                                )
                                if fig_rec is not None:
                                    plot_tau_recovery_points(
                                        tau_values=unique_taus,
                                        predicted_by_tau=pred_by_tau,
                                        title=f"Inverse Regression Tau Recovery\n{combo}",
                                        output_path=out_path / f"tau_prediction_{safe_combo}.png",
                                    )
                                    outputs[f"tau_recovery_{safe_combo}{task_suffix}"] = fig_rec
                    except Exception as e:
                        logger.warning("Could not generate discrete tau recovery points plot (%s)", e)

    if "summary" not in outputs:
        primary = outputs.get(f"summary_{task_type}", list(outputs.values())[0])
        primary.to_csv(out_path / "summary_statistics.csv")
        outputs["summary"] = primary

    logger.info("Statistical analysis complete. Artifacts saved to: %s", out_path)
    return outputs


__all__ = [
    "cohens_d",
    "metric_direction",
    "compute_summary_statistics",
    "pairwise_ttests",
    "plot_metric_boxplot",
    "plot_forecast_fan_chart",
    "plot_horizon_metric_profiles",
    "plot_interval_width_boxplot",
    "plot_interval_width_mean_bar",
    "plot_critical_difference_diagram",
    "plot_tau_estimation_scatter",
    "plot_tau_recovery_points",
    "generate_forecast_visualizations",
    "run_statistical_analysis",
]

