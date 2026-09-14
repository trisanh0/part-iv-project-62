"""
Statistical Analysis and Visualization Engine for the TEMPO Framework.

Performs multi-hypothesis paired t-tests with Benjamini-Hochberg False Discovery Rate (FDR)
adjustments, Cohen's d effect size calculation, summary statistics, and publication-ready
linear/log-scale boxplots across feature extraction and selection benchmarks.
"""

from itertools import combinations
import json
import logging
import os
from pathlib import Path
import re
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
        "Forecast RMSE",
        "Forecast MAE",
        "Mean Interval Width",
        "N Extracted Features",
        "N Selected Features",
        "fit_time_seconds",
        "inference_latency_ms",
        "Fit Time (s)",
        "Inference Latency (ms)",
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

    Args:
        df: Benchmark results DataFrame containing horizon-wise metric columns.
        output_dir: Directory path to save generated horizon plots.
        method_col: Column identifying pipeline combination.

    Returns:
        Dictionary mapping metric names to Matplotlib Figure objects.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures: Dict[str, plt.Figure] = {}

    metric_prefixes = ["RMSE", "MAE", "Coverage", "Interval_Width"]
    for prefix in metric_prefixes:
        cols = []
        for c in df.columns:
            m = re.fullmatch(rf"{prefix}_H(\d+)", c, re.IGNORECASE)
            if m:
                cols.append((int(m.group(1)), c))
        if not cols:
            continue

        cols = [c for _, c in sorted(cols)]
        horizons = np.arange(1, len(cols) + 1)

        fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=120)
        methods = sorted(df[method_col].dropna().unique())

        for method in methods:
            sub = df[df[method_col] == method]
            vals = sub[cols].apply(pd.to_numeric, errors="coerce")
            mean_vals = vals.mean(axis=0).to_numpy()
            std_vals = vals.std(axis=0).fillna(0.0).to_numpy()

            ax.plot(horizons, mean_vals, marker="o", markersize=4, linewidth=1.8, label=method)
            if len(sub) > 1:
                ax.fill_between(horizons, mean_vals - std_vals, mean_vals + std_vals, alpha=0.15)

        clean_title = prefix.replace("_", " ")
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
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8.5)
        fig.tight_layout()

        fig_path = out_dir / f"horizon_{prefix.lower()}_profile.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        figures[prefix] = fig
        plt.close(fig)

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

    req_cols = ["History Values", "True Values", "Predictions"]
    if not all(c in df.columns for c in req_cols):
        logger.warning("Forecast visualization skipped; missing columns: %s", [c for c in req_cols if c not in df.columns])
        return generated_files

    # 1. Horizon Profiles
    plot_horizon_metric_profiles(df, output_dir=out_dir)

    # 2. Uncertainty Interval Width plots (Scott benchmark compatibility)
    if "Mean Interval Width" in df.columns and "Combination" in df.columns:
        fig_box = plot_interval_width_boxplot(df, output_dir=out_dir)
        if fig_box is not None:
            generated_files.append(str(out_dir / "uncertainty_interval_width_all_combinations.png"))
        fig_bar = plot_interval_width_mean_bar(df, output_dir=out_dir)
        if fig_bar is not None:
            generated_files.append(str(out_dir / "uncertainty_interval_width_mean_all_combinations.png"))

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
    "generate_forecast_visualizations",
    "run_statistical_analysis",
]

