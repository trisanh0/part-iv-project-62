"""
Statistical analysis for benchmark_results.csv.
Supports classification, regression, and forecasting.

Forecasting additionally analyses uncertainty interval width.
"""

import os
import json
from itertools import combinations
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

INPUT_FILE = "benchmark_results12.csv"
OUTPUT_DIR = "analysis_forecasting"
DATA_TYPE = "forecasting"

os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(INPUT_FILE)

for c in ["Extractor", "Selector"]:
    df[c] = df[c].fillna("None").astype(str)
df["Combination"] = df["Extractor"] + " + " + df["Selector"]

COMMON_METRICS = [
    "Extraction Time", "Selection Time", "Prediction Time", "Total Time",
    "N Extracted Features", "N Selected Features"
]
CLASSIFICATION_METRICS = ["Prediction Accuracy"]
REGRESSION_METRICS = ["RMSE", "MAE", "R2"]
FORECASTING_METRICS = [
    "Forecast RMSE", "Forecast MAE",
    "Interval Coverage", "Mean Interval Width"
]
RESOURCE_METRICS = [
    "Extraction Peak RAM (MB)", "Extraction Avg CPU (%)",
    "Extraction Peak CPU (%)", "Extraction Peak GPU (%)",
    "Extraction Peak GPU RAM (MB)", "Selection Peak RAM (MB)",
    "Selection Avg CPU (%)", "Selection Peak CPU (%)",
    "Selection Peak GPU (%)", "Selection Peak GPU RAM (MB)"
]

if DATA_TYPE == "classification":
    requested_metrics = CLASSIFICATION_METRICS + COMMON_METRICS
elif DATA_TYPE == "regression":
    requested_metrics = REGRESSION_METRICS + COMMON_METRICS + RESOURCE_METRICS
elif DATA_TYPE == "forecasting":
    requested_metrics = FORECASTING_METRICS + COMMON_METRICS + RESOURCE_METRICS
else:
    raise ValueError("DATA_TYPE must be 'classification', 'regression', or 'forecasting'")

metrics = [m for m in requested_metrics if m in df.columns]
print("\nMetrics being analysed:")
for m in metrics:
    print("  -", m)

df.groupby(["Extractor", "Selector"])[metrics].agg(
    ["mean", "std", "median", "min", "max"]
).to_csv(os.path.join(OUTPUT_DIR, "summary_statistics.csv"))

def boxplot(metric, group, log_transform=False):
    plot_df = df.copy()
    if log_transform:
        values = pd.to_numeric(plot_df[metric], errors="coerce")
        if (values.dropna() <= 0).any():
            return
        plot_df[metric] = np.log10(values)

    width = max(16, len(df[group].unique()) * 0.6) if group == "Combination" else 12
    fig, ax = plt.subplots(figsize=(width, 6))
    plot_df.boxplot(column=metric, by=group, ax=ax)
    fig.suptitle("")
    ax.set_title(f"{metric} by {group}" + (" (log10 transformed)" if log_transform else ""))
    ax.set_xlabel("")
    ax.set_ylabel(f"log10({metric})" if log_transform else metric)
    ax.tick_params(axis="x", labelrotation=60, labelsize=9)
    fig.tight_layout()
    suffix = "_log_transformed" if log_transform else ""
    filename = f"{metric.replace(' ', '_').lower()}_by_{group.lower()}{suffix}.png"
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)

for group in ["Extractor", "Selector", "Combination"]:
    for metric in metrics:
        boxplot(metric, group, False)
        boxplot(metric, group, True)

# ------------------------------------------------------------
# Forecast uncertainty-width visualisations
# ------------------------------------------------------------

def interval_width_boxplot(group):
    metric = "Mean Interval Width"
    if metric not in df.columns:
        print("WARNING: Mean Interval Width is missing; skipping width plots.")
        return

    width = max(16, len(df[group].unique()) * 0.6) if group == "Combination" else 12
    fig, ax = plt.subplots(figsize=(width, 6))
    df.boxplot(column=metric, by=group, ax=ax)
    fig.suptitle("")
    ax.set_title(f"Uncertainty Interval Width by {group}")
    ax.set_xlabel("")
    ax.set_ylabel("Mean uncertainty interval width")
    ax.tick_params(axis="x", labelrotation=60, labelsize=9)
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"uncertainty_interval_width_by_{group.lower()}.png"
        ),
        dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

def interval_width_all():
    metric = "Mean Interval Width"
    if metric not in df.columns:
        return

    order = df.groupby("Combination")[metric].median().sort_values().index
    p = df.copy()
    p["Combination"] = pd.Categorical(
        p["Combination"], categories=order, ordered=True
    )
    fig, ax = plt.subplots(figsize=(max(16, len(order) * 0.65), 7))
    p.boxplot(column=metric, by="Combination", ax=ax)
    fig.suptitle("")
    ax.set_title("Uncertainty Interval Width — All Extractor/Selector Combinations")
    ax.set_xlabel("Extractor + Selector")
    ax.set_ylabel("Mean uncertainty interval width")
    ax.tick_params(axis="x", labelrotation=70, labelsize=8)
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            "uncertainty_interval_width_all_combinations.png"
        ),
        dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

def interval_width_mean_bar():
    metric = "Mean Interval Width"
    if metric not in df.columns:
        return

    g = df.groupby("Combination")[metric].agg(["mean", "std"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(max(14, len(g) * 0.65), 7))
    x = np.arange(len(g))
    ax.bar(x, g["mean"].to_numpy(), yerr=g["std"].fillna(0).to_numpy(), capsize=4)
    ax.set_title("Mean Uncertainty Interval Width — All Combinations")
    ax.set_xlabel("Extractor + Selector")
    ax.set_ylabel("Mean uncertainty interval width")
    ax.set_xticks(x)
    ax.set_xticklabels(g.index, rotation=70, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            "uncertainty_interval_width_mean_all_combinations.png"
        ),
        dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

if DATA_TYPE == "forecasting" and "Mean Interval Width" in df.columns:
    print("\nGenerating uncertainty interval-width plots...")
    for group in ["Extractor", "Selector", "Combination"]:
        interval_width_boxplot(group)
    interval_width_all()
    interval_width_mean_bar()

# Optional horizon-wise widths, if the CSV contains them.
def horizon_width_columns():
    found = []
    for c in df.columns:
        m = re.fullmatch(r"Interval Width H?(\d+)", str(c).strip(), re.I)
        if m:
            found.append((int(m.group(1)), c))
    return [c for _, c in sorted(found)]

def horizon_width_profiles():
    cols = horizon_width_columns()
    if not cols:
        print("No horizon-wise interval-width columns found; skipping horizon profiles.")
        return

    p = df.copy()
    p[cols] = p[cols].apply(pd.to_numeric, errors="coerce")
    h = np.arange(1, len(cols) + 1)
    methods = sorted(p["Combination"].dropna().unique())

    for method in methods:
        s = p[p["Combination"] == method]
        mean = s[cols].mean().to_numpy()
        std = s[cols].std().fillna(0).to_numpy()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(h, mean, marker="o", linewidth=2)
        ax.fill_between(h, mean - std, mean + std, alpha=0.2)
        ax.set_title(f"Uncertainty Interval Width Across Forecast Horizon\n{method}")
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel("Mean uncertainty interval width")
        ax.set_xticks(h)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", method)
        fig.savefig(
            os.path.join(OUTPUT_DIR, f"interval_width_horizon_{safe}.png"),
            dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    for method in methods:
        mean = p[p["Combination"] == method][cols].mean().to_numpy()
        ax.plot(h, mean, marker="o", linewidth=1.5, label=method)
    ax.set_title("Uncertainty Interval Width Across Forecast Horizon — All Combinations")
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Mean uncertainty interval width")
    ax.set_xticks(h)
    ax.grid(alpha=0.25)
    if len(methods) <= 25:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "interval_width_across_horizon_all_combinations.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

if DATA_TYPE == "forecasting":
    horizon_width_profiles()


# ------------------------------------------------------------
# Forecast prediction visualisations
# ------------------------------------------------------------

FORECAST_COLUMNS = [
    "True Values", "Predictions", "Prediction Lower", "Prediction Upper"
]


def _parse_forecast_cell(value):
    """Parse a JSON-encoded forecast array from the benchmark CSV."""
    if pd.isna(value):
        return None
    try:
        return np.asarray(json.loads(value), dtype=float)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _forecast_records():
    """Return benchmark rows containing usable forecast arrays."""
    missing = [c for c in FORECAST_COLUMNS if c not in df.columns]
    if missing:
        print(
            "WARNING: Forecast visualisations skipped; missing columns: "
            + ", ".join(missing)
        )
        return []

    records = []
    for idx, row in df.iterrows():
        arrays = {c: _parse_forecast_cell(row[c]) for c in FORECAST_COLUMNS}
        if any(v is None for v in arrays.values()):
            continue

        shapes = {v.shape for v in arrays.values()}
        if len(shapes) != 1:
            print(f"WARNING: Skipping malformed forecast row {idx}: shapes={shapes}")
            continue

        shape = next(iter(shapes))
        if len(shape) != 2:
            print(
                f"WARNING: Skipping forecast row {idx}: expected "
                f"(samples, horizon), got {shape}"
            )
            continue

        records.append({
            "Dataset": str(row["Dataset"]),
            "Seed": row["Seed"],
            "Extractor": str(row["Extractor"]),
            "Selector": str(row["Selector"]),
            "Combination": str(row["Combination"]),
            **arrays,
        })

    return records


def _safe_filename(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _seed_label(seed):
    try:
        return str(int(seed))
    except (TypeError, ValueError):
        return str(seed)


def _plot_forecast_summary(records, title, filename):
    """
    Plot the actual test targets against forecast predictions and uncertainty.

    A record contains the complete (samples, horizon) arrays for one benchmark
    run. To keep the figure readable, the plot shows the horizon-wise mean over
    the test samples belonging to that *single seed and combination*. This is
    important: true values are never averaged across different seeds.
    """
    if not records:
        return

    horizons = {r["Predictions"].shape[1] for r in records}
    if len(horizons) != 1:
        print(f"WARNING: Cannot plot records with different horizons: {horizons}")
        return

    horizon = next(iter(horizons))
    h = np.arange(1, horizon + 1)

    # Each call to this function is scoped to one dataset/seed/combination,
    # except for the explicit per-seed all-combinations plot below. Therefore
    # true values are always from the same seed as the predictions.
    true_stack = np.concatenate([r["True Values"] for r in records], axis=0)
    true_mean = np.mean(true_stack, axis=0)

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(
        h, true_mean, marker="o", linewidth=2.5,
        label="True values", zorder=5
    )

    for record in records:
        pred = record["Predictions"]
        lower = record["Prediction Lower"]
        upper = record["Prediction Upper"]

        pred_mean = np.mean(pred, axis=0)
        lower_mean = np.mean(lower, axis=0)
        upper_mean = np.mean(upper, axis=0)

        label = record["Combination"]
        ax.plot(h, pred_mean, marker="o", linewidth=1.8, label=label)
        ax.fill_between(
            h, lower_mean, upper_mean,
            alpha=0.16, label=f"{label} interval"
        )

    ax.set_title(title)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Value")
    ax.set_xticks(h)
    ax.grid(alpha=0.25)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def forecast_prediction_visualisations():
    """
    Generate seed-specific forecast prediction visualisations.

    For every Dataset + Seed + Extractor + Selector combination, create a
    separate image containing that seed's true values, predictions, and
    prediction interval.

    Additionally, for every Dataset + Seed, create one combined image showing
    every extractor/selector prediction and interval for that seed. The true
    series is shared because all combinations for a seed use the same test
    target generated from that seed's dataset.
    """
    records = _forecast_records()
    if not records:
        print(
            "No usable stored forecast predictions found. "
            "Re-run the forecasting benchmark with the updated benchmark script."
        )
        return

    print("\nGenerating seed-specific forecast prediction visualisations...")

    output_root = os.path.join(OUTPUT_DIR, "forecast_predictions")
    os.makedirs(output_root, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Individual image for every Dataset + Seed + Extractor + Selector
    # ------------------------------------------------------------------
    by_run = {}
    for record in records:
        key = (
            record["Dataset"],
            _seed_label(record["Seed"]),
            record["Combination"],
        )
        by_run.setdefault(key, []).append(record)

    summary_rows = []

    for (dataset, seed, combo), run_records in sorted(by_run.items()):
        dataset_dir = os.path.join(output_root, _safe_filename(dataset))
        seed_dir = os.path.join(dataset_dir, f"seed_{_safe_filename(seed)}")
        os.makedirs(seed_dir, exist_ok=True)

        filename = os.path.join(
            seed_dir,
            f"forecast_predictions_{_safe_filename(combo)}.png",
        )

        _plot_forecast_summary(
            run_records,
            title=(
                "True Values vs Forecast and Prediction Interval\n"
                f"Dataset: {dataset} | Seed: {seed} | {combo}"
            ),
            filename=filename,
        )

        # Store exactly the values represented by the image.
        true = np.concatenate([r["True Values"] for r in run_records], axis=0)
        pred = np.concatenate([r["Predictions"] for r in run_records], axis=0)
        lower = np.concatenate([r["Prediction Lower"] for r in run_records], axis=0)
        upper = np.concatenate([r["Prediction Upper"] for r in run_records], axis=0)

        for horizon_idx in range(pred.shape[1]):
            summary_rows.append({
                "Dataset": dataset,
                "Seed": seed,
                "Combination": combo,
                "Horizon": horizon_idx + 1,
                "True Mean": np.mean(true[:, horizon_idx]),
                "Prediction Mean": np.mean(pred[:, horizon_idx]),
                "Prediction Lower Mean": np.mean(lower[:, horizon_idx]),
                "Prediction Upper Mean": np.mean(upper[:, horizon_idx]),
            })

    # ------------------------------------------------------------------
    # 2. One combined image per Dataset + Seed.
    # ------------------------------------------------------------------
    by_seed = {}
    for record in records:
        key = (record["Dataset"], _seed_label(record["Seed"]))
        by_seed.setdefault(key, []).append(record)

    for (dataset, seed), seed_records in sorted(by_seed.items()):
        seed_dir = os.path.join(
            output_root,
            _safe_filename(dataset),
            f"seed_{_safe_filename(seed)}",
        )
        os.makedirs(seed_dir, exist_ok=True)

        by_combo = {}
        for record in seed_records:
            by_combo.setdefault(record["Combination"], []).append(record)

        # Collapse duplicate rows for the same combination while preserving
        # the seed-specific true values. Normally there is one row per combo,
        # but concatenation also handles repeated benchmark rows safely.
        combined_records = []
        for combo, combo_records in sorted(by_combo.items()):
            combined_records.append({
                "Dataset": dataset,
                "Seed": seed,
                "Extractor": combo.split(" + ", 1)[0],
                "Selector": combo.split(" + ", 1)[1] if " + " in combo else "None",
                "Combination": combo,
                "True Values": np.concatenate(
                    [r["True Values"] for r in combo_records], axis=0
                ),
                "Predictions": np.concatenate(
                    [r["Predictions"] for r in combo_records], axis=0
                ),
                "Prediction Lower": np.concatenate(
                    [r["Prediction Lower"] for r in combo_records], axis=0
                ),
                "Prediction Upper": np.concatenate(
                    [r["Prediction Upper"] for r in combo_records], axis=0
                ),
            })

        filename = os.path.join(
            seed_dir,
            "forecast_predictions_all_combinations.png",
        )
        _plot_forecast_summary(
            combined_records,
            title=(
                "True Values vs Forecast and Prediction Interval — All Combinations\n"
                f"Dataset: {dataset} | Seed: {seed}"
            ),
            filename=filename,
        )

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "forecast_prediction_visualisation_summary.csv",
        ),
        index=False,
    )

    print(f"Forecast visualisations written to: {output_root}")


if DATA_TYPE == "forecasting":
    forecast_prediction_visualisations()


def cohens_d(x, y):
    d = x - y
    s = d.std(ddof=1)
    return np.nan if s == 0 else d.mean() / s

def metric_direction(metric):
    if metric in ["Prediction Accuracy", "R2", "Interval Coverage"]:
        return "higher"
    if metric in [
        "RMSE", "MAE", "Forecast RMSE", "Forecast MAE", "Mean Interval Width",
        "Extraction Time", "Selection Time", "Prediction Time", "Total Time",
        "Extraction Peak RAM (MB)", "Extraction Avg CPU (%)",
        "Extraction Peak CPU (%)", "Extraction Peak GPU (%)",
        "Extraction Peak GPU RAM (MB)", "Selection Peak RAM (MB)",
        "Selection Avg CPU (%)", "Selection Peak CPU (%)",
        "Selection Peak GPU (%)", "Selection Peak GPU RAM (MB)",
        "N Extracted Features", "N Selected Features"
    ]:
        return "lower"
    return "neutral"

def pairwise(group_col, metric):
    methods = sorted(df[group_col].dropna().unique())
    rows = []

    for a, b in combinations(methods, 2):
        a_df = df[df[group_col] == a][["Dataset", "Seed", metric]].rename(columns={metric: "Value A"})
        b_df = df[df[group_col] == b][["Dataset", "Seed", metric]].rename(columns={metric: "Value B"})
        paired = pd.merge(a_df, b_df, on=["Dataset", "Seed"], how="inner").dropna()
        if len(paired) < 2:
            rows.append({
                "Method 1": a, "Method 2": b, "N Pairs": len(paired),
                "Mean 1": np.nan, "Mean 2": np.nan, "Difference": np.nan,
                "t statistic": np.nan, "p value": np.nan, "Cohens d": np.nan,
                "Adjusted p": np.nan, "Significant": False,
                "Conclusion": "Insufficient paired observations"
            })
            continue

        x, y = paired["Value A"].to_numpy(float), paired["Value B"].to_numpy(float)
        t, p = ttest_rel(x, y)
        m1, m2 = x.mean(), y.mean()
        direction = metric_direction(metric)

        if p >= 0.05:
            conclusion = "No significant difference"
        elif direction == "higher":
            conclusion = f"{a if m1 > m2 else b} is significantly better"
        elif direction == "lower":
            conclusion = f"{a if m1 < m2 else b} is significantly better"
        else:
            conclusion = "Significant difference"

        rows.append({
            "Method 1": a, "Method 2": b, "N Pairs": len(paired),
            "Mean 1": m1, "Mean 2": m2, "Difference": m1 - m2,
            "t statistic": t, "p value": p, "Cohens d": cohens_d(x, y),
            "Adjusted p": np.nan, "Significant": False,
            "Conclusion": conclusion
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        valid = out["p value"].notna()
        if valid.sum():
            out.loc[valid, "Adjusted p"] = multipletests(
                out.loc[valid, "p value"], method="fdr_bh"
            )[1]
            out.loc[valid, "Significant"] = out.loc[valid, "Adjusted p"] < 0.05
    return out

groups = [
    ("Extractor", "extractor"),
    ("Selector", "selector"),
    ("Combination", "combination")
]

for group_col, group_name in groups:
    for metric in metrics:
        print(f"Testing {group_name} / {metric}")
        pairwise(group_col, metric).to_csv(
            os.path.join(
                OUTPUT_DIR,
                f"{group_name}_{metric.replace(' ', '_').lower()}_ttests.csv"
            ),
            index=False
        )

if DATA_TYPE in ["regression", "forecasting"]:
    available = [m for m in RESOURCE_METRICS if m in df.columns]
    if available:
        df.groupby(["Extractor", "Selector"])[available].agg(
            ["mean", "std", "median", "min", "max"]
        ).to_csv(os.path.join(OUTPUT_DIR, "resource_usage_summary.csv"))

print("\n==================================================")
print(f"Analysis complete ({DATA_TYPE})")
print(f"Results saved to: {OUTPUT_DIR}")
print("==================================================")
