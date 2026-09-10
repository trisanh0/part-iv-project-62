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

# Consistent, presentation-ready plotting defaults.
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "lines.linewidth": 2.2,
})

# Use the newest numbered benchmark result by default, so the analysis does
# not silently keep reading an older CSV after the benchmark is re-run.
RESULT_FILES = list(__import__("glob").glob("benchmark_results*.csv"))
if not RESULT_FILES:
    raise FileNotFoundError(
        "No benchmark_results*.csv file found. Run the forecasting benchmark first."
    )

def _result_number(path):
    match = re.search(r"benchmark_results(\d+)\.csv$", os.path.basename(path))
    return int(match.group(1)) if match else -1

INPUT_FILE = max(RESULT_FILES, key=_result_number)
OUTPUT_DIR = "analysis_forecasting_again"
DATA_TYPE = "forecasting"

os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(INPUT_FILE)
print(f"Analysing benchmark file: {INPUT_FILE}")

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
    """Create a clean horizontal boxplot with individual observations."""
    plot_df = df[[group, metric]].copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna()
    if plot_df.empty:
        return

    if log_transform:
        if (plot_df[metric] <= 0).any():
            return
        plot_df[metric] = np.log10(plot_df[metric])

    order = (
        plot_df.groupby(group)[metric]
        .median()
        .sort_values()
        .index
        .tolist()
    )
    values = [plot_df.loc[plot_df[group] == name, metric].to_numpy() for name in order]

    fig_height = max(4.8, 0.55 * len(order) + 1.8)
    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    bp = ax.boxplot(
        values,
        vert=False,
        labels=order,
        patch_artist=True,
        widths=0.58,
        showfliers=False,
        medianprops={"color": "#111827", "linewidth": 2.2},
        boxprops={"facecolor": "#DBEAFE", "edgecolor": "#2563EB", "linewidth": 1.2},
        whiskerprops={"color": "#64748B", "linewidth": 1.2},
        capprops={"color": "#64748B", "linewidth": 1.2},
    )

    # Light jittered observations make small benchmark sample sizes visible.
    rng = np.random.default_rng(0)
    for y_pos, vals in enumerate(values, start=1):
        jitter = rng.uniform(-0.075, 0.075, size=len(vals))
        ax.scatter(
            vals,
            y_pos + jitter,
            s=24,
            alpha=0.48,
            color="#475569",
            edgecolors="none",
            zorder=3,
        )

    ax.set_title(
        f"{metric} by {group}" + (" — log10 scale" if log_transform else ""),
        pad=14,
    )
    ax.set_xlabel(f"log10({metric})" if log_transform else metric)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.20)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()

    suffix = "_log_transformed" if log_transform else ""
    filename = f"{metric.replace(' ', '_').lower()}_by_{group.lower()}{suffix}.png"
    fig.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches="tight")
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
    boxplot(metric, group, False)


def interval_width_all():
    """Presentation-ready comparison of uncertainty widths across combinations."""
    metric = "Mean Interval Width"
    if metric not in df.columns:
        return

    p = df[["Combination", metric]].copy()
    p[metric] = pd.to_numeric(p[metric], errors="coerce")
    p = p.dropna()
    if p.empty:
        return

    order = p.groupby("Combination")[metric].median().sort_values().index.tolist()
    values = [p.loc[p["Combination"] == name, metric].to_numpy() for name in order]

    fig, ax = plt.subplots(figsize=(12, max(5.5, 0.48 * len(order) + 2)))
    ax.boxplot(
        values,
        vert=False,
        labels=order,
        patch_artist=True,
        widths=0.58,
        showfliers=False,
        medianprops={"color": "#111827", "linewidth": 2.2},
        boxprops={"facecolor": "#DCFCE7", "edgecolor": "#059669", "linewidth": 1.2},
        whiskerprops={"color": "#64748B", "linewidth": 1.2},
        capprops={"color": "#64748B", "linewidth": 1.2},
    )
    ax.set_title("Uncertainty interval width — all extractor/selector combinations", pad=14)
    ax.set_xlabel("Mean interval width")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.20)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8.5)
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "uncertainty_interval_width_all_combinations.png"),
        bbox_inches="tight",
    )
    plt.close(fig)


def interval_width_mean_bar():
    metric = "Mean Interval Width"
    if metric not in df.columns:
        return

    g = (
        df.groupby("Combination")[metric]
        .agg(["mean", "std"])
        .sort_values("mean")
    )
    if g.empty:
        return

    fig, ax = plt.subplots(figsize=(11.5, max(5.5, 0.48 * len(g) + 2)))
    y = np.arange(len(g))
    ax.barh(
        y,
        g["mean"].to_numpy(),
        xerr=g["std"].fillna(0).to_numpy(),
        capsize=3,
        color="#86EFAC",
        edgecolor="#059669",
        linewidth=0.8,
    )
    ax.set_title("Mean uncertainty interval width", pad=14)
    ax.set_xlabel("Mean interval width ± 1 SD")
    ax.set_ylabel("")
    ax.set_yticks(y)
    ax.set_yticklabels(g.index, fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.20)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "uncertainty_interval_width_mean_all_combinations.png"),
        bbox_inches="tight",
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

# The benchmark stores the complete test histories as well as the future
# targets/predictions. This lets the analysis show a genuine forecasting plot:
# observed past -> forecast origin -> actual future vs prediction.
FORECAST_COLUMNS = [
    "History Values", "True Values", "Predictions",
    "Prediction Lower", "Prediction Upper",
]


def _parse_forecast_cell(value):
    """Parse a JSON-encoded forecast/history array from the benchmark CSV."""
    if pd.isna(value):
        return None
    try:
        return np.asarray(json.loads(value), dtype=float)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _forecast_records():
    """Return benchmark rows containing usable forecasting arrays."""
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

        history_shape = arrays["History Values"].shape
        forecast_shapes = {
            arrays[c].shape
            for c in [
                "True Values", "Predictions",
                "Prediction Lower", "Prediction Upper"
            ]
        }

        # History has a different second dimension from the forecast horizon,
        # so it must not be compared directly with the forecast shapes.
        if len(history_shape) != 2 or len(forecast_shapes) != 1:
            print(
                f"WARNING: Skipping malformed forecast row {idx}: "
                f"history={history_shape}, forecasts={forecast_shapes}"
            )
            continue

        forecast_shape = next(iter(forecast_shapes))
        if forecast_shape[0] != history_shape[0]:
            print(
                f"WARNING: Skipping forecast row {idx}: history and forecast "
                f"sample counts differ ({history_shape[0]} vs {forecast_shape[0]})"
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


def _plot_forecast_summary(records, title, filename, sample_index=0):
    """Plot observed history, actual future, forecasts and intervals."""
    if not records:
        return

    history = np.asarray(records[0]["History Values"], dtype=float)
    true = np.asarray(records[0]["True Values"], dtype=float)
    if history.ndim != 2 or true.ndim != 2:
        raise ValueError("Forecast arrays must have shape (samples, timepoints).")
    if history.shape[0] != true.shape[0]:
        raise ValueError("History and future arrays must have the same number of samples.")

    for record in records:
        for key in ["Predictions", "Prediction Lower", "Prediction Upper"]:
            arr = np.asarray(record[key])
            if arr.shape != true.shape:
                raise ValueError(
                    f"{key} shape {arr.shape} does not match True Values shape {true.shape}."
                )

    sample_index = min(max(int(sample_index), 0), history.shape[0] - 1)
    history_len = history.shape[1]
    horizon = true.shape[1]
    past = history[sample_index]
    actual_future = true[sample_index]
    x_past = np.arange(-history_len + 1, 1)
    x_future = np.arange(1, horizon + 1)

    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    ax.axvspan(0, horizon + 1, alpha=0.055, zorder=0)
    ax.axvline(0, linestyle="--", linewidth=1.5, alpha=0.65, zorder=2)

    ax.plot(x_past, past, linewidth=2.6, label="Observed history", zorder=4)
    ax.scatter(x_past[-1], past[-1], s=45, zorder=5)

    # This is the actual held-out future against which the forecast is judged.
    ax.plot(
        x_future, actual_future, linewidth=2.8, marker="o", markersize=4.5,
        label="Actual future", zorder=6
    )

    for record in records:
        pred = np.asarray(record["Predictions"], dtype=float)[sample_index]
        lower = np.asarray(record["Prediction Lower"], dtype=float)[sample_index]
        upper = np.asarray(record["Prediction Upper"], dtype=float)[sample_index]
        label = record["Combination"]
        ax.plot(x_future, pred, linewidth=2.2, marker="o", markersize=4,
                label=label, zorder=5)
        ax.fill_between(x_future, lower, upper, alpha=0.16, linewidth=0, zorder=1)

    ax.set_title(title, pad=16)
    ax.set_xlabel("Time relative to forecast origin")
    ax.set_ylabel("Series value")
    ax.set_xticks([x_past[0], -history_len // 2, 0, horizon // 2, horizon])
    ax.set_xticklabels([
        f"-{history_len - 1}", f"-{history_len // 2}",
        "Forecast\norigin", f"+{horizon // 2}", f"+{horizon}"
    ])
    ax.margins(x=0.02)
    ax.grid(axis="y", alpha=0.20)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def forecast_prediction_visualisations():
    """Create a forecast-vs-actual plot for every Dataset/Seed/Combination.

    Output:
        analysis_forecasting/seed_plots/<dataset>/seed_<seed>/
            forecast_<extractor>_<selector>.png
            forecast_comparison_all_combinations.png

    The individual plot uses test sample 0. In the benchmark, test sample 0 is
    the first rolling window whose forecast starts at the chronological
    train/test boundary, so it is the cleanest single visual diagnostic.
    """
    records = _forecast_records()
    if not records:
        print(
            "No usable stored forecast predictions found. "
            "Re-run the forecasting benchmark with the updated benchmark script."
        )
        return

    output_root = os.path.join(OUTPUT_DIR, "seed_plots")
    os.makedirs(output_root, exist_ok=True)
    summary_rows = []

    # One row in benchmark_results.csv represents one extractor/selector run.
    by_run = {}
    for record in records:
        by_run[(record["Dataset"], _seed_label(record["Seed"]),
                record["Combination"])] = record

    # ------------------------------------------------------------
    # Individual plots: one file for every extractor/selector combo.
    # ------------------------------------------------------------
    for (dataset, seed, combo), record in sorted(by_run.items()):
        seed_dir = os.path.join(
            output_root, _safe_filename(dataset), f"seed_{_safe_filename(seed)}"
        )
        os.makedirs(seed_dir, exist_ok=True)

        _plot_forecast_summary(
            [record],
            title=f"Forecast vs actual future\n{dataset} | Seed {seed} | {combo}",
            filename=os.path.join(seed_dir, f"forecast_{_safe_filename(combo)}.png"),
            sample_index=0,
        )

        true = record["True Values"]
        pred = record["Predictions"]
        lower = record["Prediction Lower"]
        upper = record["Prediction Upper"]
        for h in range(pred.shape[1]):
            summary_rows.append({
                "Dataset": dataset, "Seed": seed,
                "Extractor": record["Extractor"], "Selector": record["Selector"],
                "Combination": combo, "Horizon": h + 1,
                "True Mean": np.mean(true[:, h]),
                "Prediction Mean": np.mean(pred[:, h]),
                "Prediction Lower Mean": np.mean(lower[:, h]),
                "Prediction Upper Mean": np.mean(upper[:, h]),
            })

    # ------------------------------------------------------------
    # Comparison plot: all combinations for the same seed/origin.
    # ------------------------------------------------------------
    by_seed = {}
    for record in by_run.values():
        by_seed.setdefault(
            (record["Dataset"], _seed_label(record["Seed"])), []
        ).append(record)

    for (dataset, seed), seed_records in sorted(by_seed.items()):
        seed_dir = os.path.join(
            output_root, _safe_filename(dataset), f"seed_{_safe_filename(seed)}"
        )
        os.makedirs(seed_dir, exist_ok=True)
        _plot_forecast_summary(
            sorted(seed_records, key=lambda r: r["Combination"]),
            title=f"Forecast comparison vs actual future\n{dataset} | Seed {seed}",
            filename=os.path.join(seed_dir, "forecast_comparison_all_combinations.png"),
            sample_index=0,
        )

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(OUTPUT_DIR, "forecast_prediction_visualisation_summary.csv"),
        index=False,
    )
    print(f"Forecast prediction plots written to: {output_root}")


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
