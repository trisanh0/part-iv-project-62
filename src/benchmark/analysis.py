"""
Statistical analysis for benchmark_results.csv.

Switch between classification and regression using DATA_TYPE.
"""

import os
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "benchmark_results9.csv"

OUTPUT_DIR = "analysis_classmemtest"

# DATA_TYPE = "regression"
DATA_TYPE = "classification"


# ============================================================
# Setup
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

df["Extractor"] = (
    df["Extractor"]
    .fillna("None")
    .astype(str)
)

df["Selector"] = (
    df["Selector"]
    .fillna("None")
    .astype(str)
)

df["Combination"] = (
    df["Extractor"]
    + " + "
    + df["Selector"]
)


# ============================================================
# Metrics
# ============================================================

COMMON_METRICS = [

    "Extraction Time",
    "Selection Time",
    "Prediction Time",
    "Total Time",

    "N Extracted Features",
    "N Selected Features",
]


CLASSIFICATION_METRICS = [

    "Prediction Accuracy",

]


REGRESSION_METRICS = [

    "RMSE",
    "MAE",
    "R2",

]


# Resource metrics are only currently returned for regression
# according to the benchmark result structure you provided.

RESOURCE_METRICS = [

    # Extraction
    "Extraction Peak RAM (MB)",
    "Extraction Avg CPU (%)",
    "Extraction Peak CPU (%)",
    "Extraction Peak GPU (%)",
    "Extraction Peak GPU RAM (MB)",

    # Selection
    "Selection Peak RAM (MB)",
    "Selection Avg CPU (%)",
    "Selection Peak CPU (%)",
    "Selection Peak GPU (%)",
    "Selection Peak GPU RAM (MB)",

]


if DATA_TYPE == "classification":

    metrics = (
        CLASSIFICATION_METRICS
        + COMMON_METRICS
    )

elif DATA_TYPE == "regression":

    metrics = (
        REGRESSION_METRICS
        + COMMON_METRICS
        + RESOURCE_METRICS
    )

else:

    raise ValueError(
        "DATA_TYPE must be either "
        "'classification' or 'regression'"
    )


# Only analyse columns that actually exist in the CSV.
metrics = [
    metric
    for metric in metrics
    if metric in df.columns
]

print("\nMetrics being analysed:")

for metric in metrics:
    print(f"  - {metric}")


# ============================================================
# Summary statistics
# ============================================================

summary = (
    df
    .groupby(["Extractor", "Selector"])[metrics]
    .agg([
        "mean",
        "std",
        "median",
        "min",
        "max"
    ])
)

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "summary_statistics.csv"
    )
)


# ============================================================
# Boxplots
# ============================================================

def boxplot(metric, group, log_transform=False):

    width = 12

    if group == "Combination":

        width = max(
            16,
            len(df[group].unique()) * 0.6
        )

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    plot_df = df.copy()

    if log_transform:

        # Convert the metric to numeric
        values = pd.to_numeric(
            plot_df[metric],
            errors="coerce"
        )

        # Log10 requires strictly positive values
        if (values.dropna() <= 0).any():

            print(
                f"Skipping log-transformed plot for "
                f"'{metric}' because it contains "
                f"zero or negative values."
            )

            return

        # Actually transform the DATA
        plot_df[metric] = np.log10(values)

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(width, 6)
    )

    # --------------------------------------------------------
    # Draw boxplot using either original or transformed data
    # --------------------------------------------------------

    plot_df.boxplot(
        column=metric,
        by=group,
        ax=ax
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    if log_transform:

        ax.set_title(
            f"{metric} by {group} "
            f"(log10 transformed)"
        )

    else:

        ax.set_title(
            f"{metric} by {group}"
        )

    # Remove pandas' automatic super-title
    fig.suptitle("")

    # --------------------------------------------------------
    # Axis formatting
    # --------------------------------------------------------

    ax.tick_params(
        axis="x",
        labelrotation=60,
        labelsize=9
    )

    ax.set_xlabel("")

    if log_transform:

        ax.set_ylabel(
            f"log10({metric})"
        )

    else:

        ax.set_ylabel(metric)

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    fig.tight_layout()

    # --------------------------------------------------------
    # Filename
    # --------------------------------------------------------

    base_filename = (
        f"{metric.replace(' ', '_').lower()}"
        f"_by_{group.lower()}"
    )

    if log_transform:

        filename = (
            f"{base_filename}_log_transformed.png"
        )

    else:

        filename = (
            f"{base_filename}.png"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            filename
        ),
        dpi=300
    )

    plt.close(fig)


# ============================================================
# Generate normal AND log-transformed boxplots
# ============================================================

for group in [
    "Extractor",
    "Selector",
    "Combination"
]:

    for metric in metrics:

        # Original boxplot
        boxplot(
            metric,
            group,
            log_transform=False
        )

        # Log10-transformed boxplot
        boxplot(
            metric,
            group,
            log_transform=True
        )


# ============================================================
# Effect size
# ============================================================

def cohens_d(x, y):

    """
    Cohen's d for paired observations.

    d = mean(x-y) / std(x-y)
    """

    d = x - y

    s = d.std(ddof=1)

    if s == 0:

        return np.nan

    return d.mean() / s


# ============================================================
# Determine what "better" means
# ============================================================

def metric_direction(metric):

    """
    Return the preferred direction for a metric.

    'higher' means larger values are better.

    'lower' means smaller values are better.

    'neutral' means no performance direction is assumed.
    """

    # Higher is better
    if metric in [
        "Prediction Accuracy",
        "R2",
    ]:

        return "higher"

    # Lower is better
    if metric in [

        "RMSE",
        "MAE",

        "Extraction Time",
        "Selection Time",
        "Prediction Time",
        "Total Time",

        "Extraction Peak RAM (MB)",
        "Extraction Avg CPU (%)",
        "Extraction Peak CPU (%)",
        "Extraction Peak GPU (%)",
        "Extraction Peak GPU RAM (MB)",

        "Selection Peak RAM (MB)",
        "Selection Avg CPU (%)",
        "Selection Peak CPU (%)",
        "Selection Peak GPU (%)",
        "Selection Peak GPU RAM (MB)",

        "N Extracted Features",
        "N Selected Features",

    ]:

        return "lower"

    return "neutral"


# ============================================================
# Pairwise statistical tests
# ============================================================

def pairwise(group_col, metric):

    methods = sorted(
        df[group_col]
        .dropna()
        .unique()
    )

    rows = []

    for a, b in combinations(methods, 2):

        # ----------------------------------------------------
        # Get observations for each method
        #
        # Pair using Dataset + Seed.
        # ----------------------------------------------------

        a_df = (
            df[df[group_col] == a]
            [["Dataset", "Seed", metric]]
            .rename(
                columns={
                    metric: "Value A"
                }
            )
        )

        b_df = (
            df[df[group_col] == b]
            [["Dataset", "Seed", metric]]
            .rename(
                columns={
                    metric: "Value B"
                }
            )
        )

        paired = pd.merge(
            a_df,
            b_df,
            on=[
                "Dataset",
                "Seed"
            ],
            how="inner"
        )

        paired = paired.dropna(
            subset=[
                "Value A",
                "Value B"
            ]
        )

        # ----------------------------------------------------
        # Need at least 2 paired observations
        # ----------------------------------------------------

        if len(paired) < 2:

            rows.append({

                "Method 1": a,
                "Method 2": b,

                "N Pairs": len(paired),

                "Mean 1": np.nan,
                "Mean 2": np.nan,

                "Difference": np.nan,

                "t statistic": np.nan,
                "p value": np.nan,

                "Cohens d": np.nan,

                "Adjusted p": np.nan,

                "Significant": False,

                "Conclusion":
                    "Insufficient paired observations"

            })

            continue

        x = paired["Value A"].to_numpy(
            dtype=float
        )

        y = paired["Value B"].to_numpy(
            dtype=float
        )

        # ----------------------------------------------------
        # Paired t-test
        # ----------------------------------------------------

        t, p = ttest_rel(
            x,
            y
        )

        # ----------------------------------------------------
        # Effect size
        # ----------------------------------------------------

        d = cohens_d(
            x,
            y
        )

        m1 = x.mean()
        m2 = y.mean()

        difference = m1 - m2

        # ----------------------------------------------------
        # Determine conclusion
        # ----------------------------------------------------

        direction = metric_direction(
            metric
        )

        if p >= 0.05:

            conclusion = (
                "No significant difference"
            )

        else:

            if direction == "higher":

                better = (
                    a
                    if m1 > m2
                    else b
                )

                conclusion = (
                    f"{better} is significantly "
                    f"better"
                )

            elif direction == "lower":

                better = (
                    a
                    if m1 < m2
                    else b
                )

                conclusion = (
                    f"{better} is significantly "
                    f"better"
                )

            else:

                conclusion = (
                    "Significant difference"
                )

        rows.append({

            "Method 1": a,
            "Method 2": b,

            "N Pairs": len(paired),

            "Mean 1": m1,
            "Mean 2": m2,

            "Difference": difference,

            "t statistic": t,
            "p value": p,

            "Cohens d": d,

            "Adjusted p": np.nan,

            "Significant": False,

            "Conclusion": conclusion

        })

    out = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Multiple-comparison correction
    # --------------------------------------------------------

    if not out.empty:

        valid = out["p value"].notna()

        if valid.sum() > 0:

            adjusted = multipletests(
                out.loc[
                    valid,
                    "p value"
                ],
                method="fdr_bh"
            )[1]

            out.loc[
                valid,
                "Adjusted p"
            ] = adjusted

            out.loc[
                valid,
                "Significant"
            ] = (
                out.loc[
                    valid,
                    "Adjusted p"
                ] < 0.05
            )

    return out


# ============================================================
# Pairwise tests
# ============================================================

groups = [

    (
        "Extractor",
        "extractor"
    ),

    (
        "Selector",
        "selector"
    ),

    (
        "Combination",
        "combination"
    ),

]


for group_col, group_name in groups:

    for metric in metrics:

        print(
            f"Testing "
            f"{group_name} / "
            f"{metric}"
        )

        result = pairwise(
            group_col,
            metric
        )

        result.to_csv(

            os.path.join(
                OUTPUT_DIR,
                f"{group_name}_"
                f"{metric.replace(' ', '_').lower()}"
                f"_ttests.csv"
            ),

            index=False
        )


# ============================================================
# Additional resource summaries
# ============================================================

if DATA_TYPE == "regression":

    resource_summary = (
        df
        .groupby(
            ["Extractor", "Selector"]
        )[RESOURCE_METRICS]
        .agg([
            "mean",
            "std",
            "median",
            "min",
            "max"
        ])
    )

    resource_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "resource_usage_summary.csv"
        )
    )


# ============================================================
# Print completion
# ============================================================

print()

print(
    "=================================================="
)

print(
    f"Analysis complete "
    f"({DATA_TYPE})"
)

print(
    f"Results saved to: {OUTPUT_DIR}"
)

print(
    "=================================================="
)