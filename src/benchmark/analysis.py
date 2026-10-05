"""
Statistical analysis for benchmark_results.csv.
Supports classification, regression, inverse regression, and forecasting.

Forecasting additionally analyses uncertainty interval width.
"""

import os
import json
from itertools import combinations
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, friedmanchisquare, studentized_range, wilcoxon

try:
    import scikit_posthocs as sp
except ImportError:
    sp = None
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
OUTPUT_DIR = "analysis_inv"
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
INVERSE_REGRESSION_METRICS = [
    "Tau RMSE", "Tau MAE", "Tau Bias", "Tau Max Absolute Error",
    "Tau Relative MAE (%)", "Tau R2"
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
elif DATA_TYPE == "inverse_regression":
    requested_metrics = INVERSE_REGRESSION_METRICS + COMMON_METRICS + RESOURCE_METRICS
elif DATA_TYPE == "forecasting":
    requested_metrics = FORECASTING_METRICS + COMMON_METRICS + RESOURCE_METRICS
else:
    raise ValueError("DATA_TYPE must be 'classification', 'regression', 'inverse_regression', or 'forecasting'")

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


# ------------------------------------------------------------
# Inverse-regression prediction visualisations
# ------------------------------------------------------------

def _parse_array_cell(value):
    if pd.isna(value):
        return None
    try:
        return np.asarray(json.loads(value), dtype=float)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def inverse_regression_prediction_visualisations():
    """Create simple five-point tau recovery plots, one set per seed.

    The benchmark creates five tau values for each seed and reuses those exact
    values for every extractor/selector/predictor combination on that seed.

    Each plot has:
      x = 1, 2, 3, 4, 5 (the five shared tau cases)
      red dot = true tau
      blue dot = mean predicted tau for that true-tau case

    The test split contains multiple simulated series for each tau.  Predictions
    are therefore averaged within each of the five tau groups so the plot has
    exactly one predicted point per tau value.
    """
    required = ["True Tau Values", "Test True Tau Values", "Predicted Tau Values"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(
            "WARNING: inverse-regression plots skipped; missing columns: "
            + ", ".join(missing)
        )
        return

    root = os.path.join(OUTPUT_DIR, "seed_plots")
    os.makedirs(root, exist_ok=True)
    summary_rows = []

    for _, row in df.iterrows():
        predictions = _parse_array_cell(row["Predicted Tau Values"])
        true_test = _parse_array_cell(row["Test True Tau Values"])
        tau_values = _parse_array_cell(row["True Tau Values"])

        if (
            predictions is None
            or true_test is None
            or tau_values is None
            or predictions.size == 0
            or true_test.size != predictions.size
        ):
            continue

        predictions = predictions.ravel()
        true_test = true_test.ravel()
        tau_values = np.sort(np.unique(tau_values.ravel()))

        # This benchmark is explicitly defined around five shared tau values.
        if len(tau_values) != 5:
            print(
                f"WARNING: skipping inverse-regression row for "
                f"{row.get('Dataset', 'unknown')} seed {row.get('Seed', 'unknown')}: "
                f"expected 5 tau values, found {len(tau_values)}."
            )
            continue

        dataset = str(row["Dataset"])
        seed = _seed_label(row["Seed"])
        combo = str(row["Combination"])

        seed_dir = os.path.join(
            root,
            _safe_filename(dataset),
            f"seed_{_safe_filename(seed)}",
        )
        os.makedirs(seed_dir, exist_ok=True)

        # Collapse the repeated test observations down to one prediction per
        # shared tau value.  The true value is exactly the tau on the x-axis.
        predicted_by_tau = []
        for tau in tau_values:
            mask = np.isclose(true_test, tau, rtol=0.0, atol=1e-10)
            if not np.any(mask):
                predicted_by_tau.append(np.nan)
            else:
                predicted_by_tau.append(float(np.mean(predictions[mask])))

        predicted_by_tau = np.asarray(predicted_by_tau, dtype=float)
        x = np.arange(1, 6)

        fig, ax = plt.subplots(figsize=(9.5, 5.8))

        # Exactly two point colours: true tau and predicted tau.
        ax.scatter(
            x,
            tau_values,
            s=75,
            color="#DC2626",
            label="True tau",
            zorder=4,
        )
        ax.scatter(
            x,
            predicted_by_tau,
            s=75,
            color="#2563EB",
            label="Predicted tau",
            zorder=5,
        )

        ax.set_title(
            f"Inverse regression tau recovery\n"
            f"{dataset} | Seed {seed} | {combo}",
            pad=14,
        )
        ax.set_xlabel("Tau value")
        ax.set_ylabel("Tau")
        ax.set_xticks(x)
        ax.set_xlim(0.5, 5.5)
        ax.grid(axis="y", alpha=0.20)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="best")

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                seed_dir,
                f"tau_prediction_{_safe_filename(combo)}.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        for i, (tau, prediction) in enumerate(
            zip(tau_values, predicted_by_tau),
            start=1,
        ):
            error = prediction - tau if np.isfinite(prediction) else np.nan
            summary_rows.append({
                "Dataset": dataset,
                "Seed": seed,
                "Extractor": row["Extractor"],
                "Selector": row["Selector"],
                "Combination": combo,
                "Tau Index": i,
                "True Tau": tau,
                "Predicted Tau": prediction,
                "Error": error,
                "Absolute Error": abs(error) if np.isfinite(error) else np.nan,
            })

    if summary_rows:
        pred_df = pd.DataFrame(summary_rows)
        pred_df.to_csv(
            os.path.join(OUTPUT_DIR, "inverse_regression_predictions.csv"),
            index=False,
        )

        combo_seed = df[
            [
                "Dataset",
                "Seed",
                "Combination",
                "True Tau Values",
                "Tau RMSE",
                "Tau MAE",
                "Tau Bias",
            ]
        ].copy()
        combo_seed.to_csv(
            os.path.join(OUTPUT_DIR, "inverse_regression_seed_summary.csv"),
            index=False,
        )

    print(f"Inverse-regression prediction plots written to: {root}")


if DATA_TYPE == "inverse_regression":
    inverse_regression_prediction_visualisations()


def seed_metric_visualisations():
    """Create seed-specific metric plots for classification/regression.

    Aggregate comparison plots remain in OUTPUT_DIR.  These plots are
    specifically tied to one Dataset/Seed pair, so they live under:

        seed_plots/<dataset>/seed_<seed>/

    One image is created per metric.  No seed-specific image is written to the
    base analysis directory.
    """
    if DATA_TYPE not in {"classification", "regression"}:
        return

    root = os.path.join(OUTPUT_DIR, "seed_plots")
    os.makedirs(root, exist_ok=True)

    seed_columns = ["Dataset", "Seed"]
    available_metrics = [m for m in metrics if m in df.columns]

    for (dataset, seed), group in df.groupby(seed_columns, dropna=False):
        dataset_label = str(dataset)
        seed_label = _seed_label(seed)
        seed_dir = os.path.join(
            root,
            _safe_filename(dataset_label),
            f"seed_{_safe_filename(seed_label)}",
        )
        os.makedirs(seed_dir, exist_ok=True)

        for metric in available_metrics:
            plot_df = group[["Combination", metric]].copy()
            plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
            plot_df = plot_df.dropna()

            if plot_df.empty:
                continue

            # Keep the benchmark's combination ordering stable rather than
            # making the seed-specific figure reorder methods by performance.
            plot_df = (
                plot_df.groupby("Combination", as_index=False)[metric]
                .mean()
            )

            fig_width = max(8.5, 0.65 * len(plot_df) + 3.5)
            fig, ax = plt.subplots(figsize=(fig_width, 5.8))

            x = np.arange(len(plot_df))
            ax.bar(
                x,
                plot_df[metric].to_numpy(),
                width=0.72,
            )

            ax.set_title(
                f"{metric} | {dataset_label} | Seed {seed_label}",
                pad=14,
            )
            ax.set_xlabel("Extractor + Selector")
            ax.set_ylabel(metric)
            ax.set_xticks(x)
            ax.set_xticklabels(
                plot_df["Combination"],
                rotation=45,
                ha="right",
                fontsize=8.5,
            )
            ax.grid(axis="y", alpha=0.20)
            ax.grid(axis="x", visible=False)

            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    seed_dir,
                    f"metric_{_safe_filename(metric)}.png",
                ),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)


seed_metric_visualisations()


def cohens_d(x, y):
    d = x - y
    s = d.std(ddof=1)
    return np.nan if s == 0 else d.mean() / s

def metric_direction(metric):
    if metric in ["Prediction Accuracy", "R2", "Interval Coverage"]:
        return "higher"
    if metric in [
        "RMSE", "MAE", "Forecast RMSE", "Forecast MAE",
        "Tau RMSE", "Tau MAE", "Tau Max Absolute Error",
        "Tau Relative MAE (%)", "Mean Interval Width",
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

# ------------------------------------------------------------
# Critical difference diagrams
# ------------------------------------------------------------
#
# CD diagrams compare methods across repeated benchmark blocks. Each
# Dataset/Seed pair is treated as one block, so the methods are ranked within
# the same benchmark replicate before average ranks are calculated.
#
# scikit-posthocs provides the Nemenyi post-hoc p-value matrix and the
# critical_difference_diagram() renderer. This is the standard blocked/rank
# comparison used for classifier/method comparisons across multiple datasets.
# ------------------------------------------------------------

CD_ALPHA = 0.05


def critical_difference_diagram(group_col, metric, alpha=CD_ALPHA):
    """
    Create and save a critical difference diagram for one metric/group.

    Rows are blocked by Dataset + Seed. Only blocks containing every method
    are used, which prevents missing method runs from changing the ranks.

    Statistical handling:
        - 2 methods:
            Friedman/Nemenyi cannot be used. A paired Wilcoxon signed-rank
            test is used instead, and a two-method CD-style plot is produced.
        - 3+ methods:
            Friedman omnibus test + Nemenyi post-hoc test + CD diagram.

    Returns a dictionary containing the statistical results, or None when
    there is insufficient data.
    """

    if sp is None:
        print(
            "WARNING: scikit-posthocs is not installed; skipping critical "
            "difference diagrams. Install it with: pip install scikit-posthocs"
        )
        return None

    if metric not in df.columns:
        return None

    work = df[["Dataset", "Seed", group_col, metric]].copy()

    work[metric] = pd.to_numeric(
        work[metric],
        errors="coerce"
    )

    work = work.dropna(
        subset=[group_col, metric]
    )

    # If a method occurs more than once in a block, average those rows first.
    matrix = work.pivot_table(
        index=["Dataset", "Seed"],
        columns=group_col,
        values=metric,
        aggfunc="mean",
    )

    # Only use complete Dataset/Seed blocks.
    matrix = matrix.dropna(
        axis=0,
        how="any"
    )

    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        print(
            f"Skipping CD diagram for {group_col} / {metric}: "
            f"need at least 2 complete blocks and 2 methods; "
            f"found {matrix.shape[0]} blocks and "
            f"{matrix.shape[1]} methods."
        )
        return None

    direction = metric_direction(metric)

    if direction == "neutral":
        print(
            f"Skipping CD diagram for {group_col} / {metric}: "
            "metric has no defined higher/lower direction."
        )
        return None

    ##########################################################################
    # RANK METHODS
    ##########################################################################

    # Rank 1 = best.
    ranks = matrix.rank(
        axis=1,
        method="average",
        ascending=(direction == "lower"),
    )

    avg_rank = ranks.mean(
        axis=0
    ).sort_values()

    n_blocks = matrix.shape[0]
    n_methods = matrix.shape[1]

    ##########################################################################
    # OUTPUT DIRECTORY
    ##########################################################################

    output_dir = os.path.join(
        OUTPUT_DIR,
        "critical_difference_diagrams",
        group_col.lower(),
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    safe_metric = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        metric
    ).strip("_").lower()

    ##########################################################################
    # SAVE DATA USED FOR THE ANALYSIS
    ##########################################################################

    matrix.to_csv(
        os.path.join(
            output_dir,
            f"{safe_metric}_blocked_values.csv"
        )
    )

    ranks.to_csv(
        os.path.join(
            output_dir,
            f"{safe_metric}_ranks_by_block.csv"
        )
    )

    avg_rank.rename(
        "Average Rank"
    ).to_csv(
        os.path.join(
            output_dir,
            f"{safe_metric}_average_ranks.csv"
        )
    )

    ##########################################################################
    # TWO-METHOD CASE
    ##########################################################################

    if n_methods == 2:

        method_1 = matrix.columns[0]
        method_2 = matrix.columns[1]

        values_1 = matrix[method_1].to_numpy()
        values_2 = matrix[method_2].to_numpy()

        ######################################################################
        # Paired Wilcoxon signed-rank test
        ######################################################################

        try:

            wilcoxon_stat, wilcoxon_p = wilcoxon(
                values_1,
                values_2,
                alternative="two-sided"
            )

        except ValueError:

            # Can occur when all paired differences are zero.
            wilcoxon_stat = 0.0
            wilcoxon_p = 1.0

        ######################################################################
        # Rank-biserial effect size
        ######################################################################

        differences = values_1 - values_2

        non_zero = differences[differences != 0]

        if len(non_zero) > 0:

            positive = np.sum(non_zero > 0)
            negative = np.sum(non_zero < 0)

            effect_size = (
                positive - negative
            ) / len(non_zero)

        else:

            effect_size = 0.0

        ######################################################################
        # Save pairwise p-value matrix
        ######################################################################

        p_values = pd.DataFrame(
            1.0,
            index=matrix.columns,
            columns=matrix.columns,
        )

        p_values.loc[
            method_1,
            method_2
        ] = wilcoxon_p

        p_values.loc[
            method_2,
            method_1
        ] = wilcoxon_p

        p_values.to_csv(
            os.path.join(
                output_dir,
                f"{safe_metric}_pairwise_p_values.csv"
            )
        )

        ######################################################################
        # Two-method CD-style plot
        #
        # There is no Friedman/Nemenyi critical difference for only two
        # methods. Instead, display the average ranks and connect methods
        # when their paired comparison is not statistically significant.
        ######################################################################

        fig_width = 10
        fig, ax = plt.subplots(
            figsize=(fig_width, 4.8)
        )

        # Draw horizontal rank axis.
        ax.set_xlim(
            0.5,
            2.5
        )

        ax.set_ylim(
            0,
            1
        )

        ax.set_yticks([])

        ax.set_xlabel(
            "Average Rank (lower rank = better)"
        )

        ax.set_title(
            f"Critical Difference Diagram — {metric}\n"
            f"{group_col} | {n_blocks} blocks | "
            f"Wilcoxon p = {wilcoxon_p:.4g}"
        )

        ######################################################################
        # Plot the two methods
        ######################################################################

        y = 0.55

        for rank, method in enumerate(
            avg_rank.index,
            start=1
        ):

            ax.plot(
                avg_rank[method],
                y,
                "o",
                markersize=9
            )

            if rank == 1:

                ax.text(
                    avg_rank[method] - 0.04,
                    y + 0.08,
                    f"{method} ({avg_rank[method]:.2f})",
                    ha="right",
                    va="bottom"
                )

            else:

                ax.text(
                    avg_rank[method] + 0.04,
                    y + 0.08,
                    f"{method} ({avg_rank[method]:.2f})",
                    ha="left",
                    va="bottom"
                )

        ######################################################################
        # If not statistically significant, connect the methods.
        ######################################################################

        if wilcoxon_p >= alpha:

            x1 = avg_rank.iloc[0]
            x2 = avg_rank.iloc[1]

            ax.plot(
                [x1, x2],
                [y - 0.10, y - 0.10],
                linewidth=6,
                solid_capstyle="round"
            )

            ax.text(
                (x1 + x2) / 2,
                y - 0.20,
                f"Not significant (p = {wilcoxon_p:.4g})",
                ha="center",
                va="top"
            )

        else:

            ax.text(
                1.5,
                y - 0.20,
                f"Significant difference (p = {wilcoxon_p:.4g})",
                ha="center",
                va="top"
            )

        ######################################################################
        # Save image
        ######################################################################

        fig.tight_layout()

        filename = os.path.join(
            output_dir,
            f"critical_difference_{safe_metric}.png"
        )

        fig.savefig(
            filename,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        ######################################################################
        # Save statistical summary
        ######################################################################

        summary = pd.DataFrame({
            "Method": avg_rank.index,
            "Average Rank": avg_rank.to_numpy(),
        })

        summary["N Blocks"] = n_blocks
        summary["N Methods"] = n_methods
        summary["Test"] = "Wilcoxon signed-rank"
        summary["Test Statistic"] = wilcoxon_stat
        summary["p-value"] = wilcoxon_p
        summary["Effect Size"] = effect_size
        summary["Alpha"] = alpha

        summary.to_csv(
            os.path.join(
                output_dir,
                f"{safe_metric}_cd_summary.csv"
            ),
            index=False
        )

        print(
            f"Two-method CD diagram written: {filename} "
            f"({n_blocks} blocks, Wilcoxon p={wilcoxon_p:.4g})"
        )

        return {
            "matrix": matrix,
            "ranks": ranks,
            "average_ranks": avg_rank,
            "p_values": p_values,
            "test": "Wilcoxon signed-rank",
            "test_statistic": wilcoxon_stat,
            "p_value": wilcoxon_p,
            "effect_size": effect_size,
            "filename": filename,
        }

    ##########################################################################
    # THREE OR MORE METHODS
    ##########################################################################

    # Nemenyi post-hoc test.
    p_values = sp.posthoc_nemenyi_friedman(
        matrix
    )

    p_values = p_values.loc[
        avg_rank.index,
        avg_rank.index
    ]

    ##########################################################################
    # Friedman omnibus test
    ##########################################################################

    friedman_stat, friedman_p = friedmanchisquare(
        *[
            matrix[col].to_numpy()
            for col in avg_rank.index
        ]
    )

    ##########################################################################
    # Critical difference
    ##########################################################################

    q_alpha = (
        studentized_range.ppf(
            1 - alpha,
            n_methods,
            np.inf
        )
        / np.sqrt(2)
    )

    cd = q_alpha * np.sqrt(
        n_methods * (n_methods + 1)
        / (6.0 * n_blocks)
    )

    ##########################################################################
    # Save Nemenyi p-values
    ##########################################################################

    p_values.to_csv(
        os.path.join(
            output_dir,
            f"{safe_metric}_nemenyi_p_values.csv"
        )
    )

    ##########################################################################
    # Create CD diagram
    ##########################################################################

    fig_width = max(
        10,
        1.2 * n_methods + 5
    )

    fig, ax = plt.subplots(
        figsize=(fig_width, 4.8)
    )

    sp.critical_difference_diagram(
        ranks=avg_rank,
        sig_matrix=p_values,
        cd=cd,
        alpha=alpha,
        ax=ax,
        label_fmt_left="{label} ({rank:.2f})",
        label_fmt_right="({rank:.2f}) {label}",
        text_h_margin=0.04,
    )

    ax.set_title(
        f"Critical Difference Diagram — {metric}\n"
        f"{group_col} | {n_blocks} blocks | "
        f"Friedman p = {friedman_p:.4g} | CD = {cd:.3f}",
        pad=18,
    )

    fig.tight_layout()

    filename = os.path.join(
        output_dir,
        f"critical_difference_{safe_metric}.png"
    )

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    ##########################################################################
    # Save summary
    ##########################################################################

    summary = pd.DataFrame({
        "Method": avg_rank.index,
        "Average Rank": avg_rank.to_numpy(),
    })

    summary["N Blocks"] = n_blocks
    summary["N Methods"] = n_methods
    summary["Friedman Statistic"] = friedman_stat
    summary["Friedman p"] = friedman_p
    summary["Critical Difference"] = cd
    summary["Alpha"] = alpha

    summary.to_csv(
        os.path.join(
            output_dir,
            f"{safe_metric}_cd_summary.csv"
        ),
        index=False,
    )

    print(
        f"CD diagram written: {filename} "
        f"({n_blocks} blocks, Friedman p={friedman_p:.4g}, CD={cd:.3f})"
    )

    return {
        "matrix": matrix,
        "ranks": ranks,
        "average_ranks": avg_rank,
        "p_values": p_values,
        "friedman_statistic": friedman_stat,
        "friedman_p": friedman_p,
        "cd": cd,
        "filename": filename,
    }


def generate_critical_difference_diagrams():
    """Generate CD diagrams for every analysed metric and comparison level."""
    if sp is None:
        print(
            "WARNING: CD diagrams skipped because scikit-posthocs is not "
            "installed. Install with: pip install scikit-posthocs"
        )
        return

    print("\nGenerating critical difference diagrams...")
    for group_col, group_name in groups:
        for metric in metrics:
            print(f"CD analysis: {group_name} / {metric}")
            critical_difference_diagram(group_col, metric)



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

generate_critical_difference_diagrams()

if DATA_TYPE in ["regression", "inverse_regression", "forecasting"]:
    available = [m for m in RESOURCE_METRICS if m in df.columns]
    if available:
        df.groupby(["Extractor", "Selector"])[available].agg(
            ["mean", "std", "median", "min", "max"]
        ).to_csv(os.path.join(OUTPUT_DIR, "resource_usage_summary.csv"))

print("\n==================================================")
print(f"Analysis complete ({DATA_TYPE})")
print(f"Results saved to: {OUTPUT_DIR}")
print("==================================================")
