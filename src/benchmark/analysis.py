"""
analysis.py

Statistical analysis for benchmark_results.csv.

Switch between classification and regression using:

    DATA_TYPE = "classification"

or:

    DATA_TYPE = "regression"
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

INPUT_FILE = "benchmark_results5.csv"
OUTPUT_DIR = "analysis_regression"

DATA_TYPE = "regression"
# DATA_TYPE = "classification"


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
]

FEATURE_METRICS = [
    "N Extracted Features",
    "N Selected Features",
]


if DATA_TYPE == "classification":

    PERFORMANCE_METRICS = [
        "Prediction Accuracy",
    ]

elif DATA_TYPE == "regression":

    PERFORMANCE_METRICS = [
        "RMSE",
        "MAE",
        "R2",
    ]

else:

    raise ValueError(
        "DATA_TYPE must be either "
        "'classification' or 'regression'"
    )


metrics = (
    PERFORMANCE_METRICS
    + COMMON_METRICS
    + FEATURE_METRICS
)


# Only use metrics that actually exist in the CSV.
# This makes the script slightly more robust to older
# benchmark result files.
metrics = [
    m for m in metrics
    if m in df.columns
]


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

def boxplot(metric, group):

    width = 12

    if group == "Combination":

        width = max(
            16,
            len(df[group].unique()) * 0.6
        )

    plt.figure(
        figsize=(width, 6)
    )

    df.boxplot(
        column=metric,
        by=group
    )

    plt.title(
        f"{metric} by {group}"
    )

    plt.suptitle("")

    plt.xticks(
        rotation=60,
        ha="right",
        fontsize=9
    )

    plt.tight_layout()

    filename = (
        f"{metric.replace(' ', '_').lower()}"
        f"_by_{group.lower()}.png"
    )

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            filename
        ),
        dpi=300
    )

    plt.close()


for group in [
    "Extractor",
    "Selector",
    "Combination"
]:

    for metric in metrics:

        boxplot(
            metric,
            group
        )


# ============================================================
# Effect size
# ============================================================

def cohens_d(x, y):

    d = x - y

    s = d.std(
        ddof=1
    )

    if s == 0:
        return np.nan

    return d.mean() / s


# ============================================================
# Metric direction
# ============================================================

def higher_is_better(metric):

    if metric == "Prediction Accuracy":
        return True

    if metric == "R2":
        return True

    # Everything else is lower-is-better:
    #
    # RMSE
    # MAE
    # Extraction Time
    # Selection Time
    # Prediction Time
    # Total Time
    #
    # Feature counts are also lower-is-better if
    # compared, although these are not normally
    # performance metrics.

    return False


# ============================================================
# Conclusion
# ============================================================

def make_conclusion(
    metric,
    method_a,
    method_b,
    mean_a,
    mean_b,
    p_value
):

    if p_value >= 0.05:

        return (
            "No significant difference"
        )

    if higher_is_better(metric):

        if mean_a > mean_b:

            return (
                f"{method_a} is significantly better "
                f"(higher {metric})"
            )

        elif mean_b > mean_a:

            return (
                f"{method_b} is significantly better "
                f"(higher {metric})"
            )

        else:

            return (
                "No significant difference"
            )

    else:

        if mean_a < mean_b:

            return (
                f"{method_a} is significantly better "
                f"(lower {metric})"
            )

        elif mean_b < mean_a:

            return (
                f"{method_b} is significantly better "
                f"(lower {metric})"
            )

        else:

            return (
                "No significant difference"
            )


# ============================================================
# Pairwise tests
# ============================================================

def pairwise(
    group_col,
    metric
):

    methods = sorted(
        df[group_col].unique()
    )

    rows = []

    for a, b in combinations(
        methods,
        2
    ):

        # ----------------------------------------------------
        # Get paired observations
        # ----------------------------------------------------

        a_df = (
            df[df[group_col] == a]
            .sort_values(
                ["Dataset", "Seed"]
            )
        )

        b_df = (
            df[df[group_col] == b]
            .sort_values(
                ["Dataset", "Seed"]
            )
        )

        # ----------------------------------------------------
        # Align observations
        #
        # This is preferable to simply truncating the arrays,
        # because paired t-tests require corresponding
        # observations.
        # ----------------------------------------------------

        keys = [
            "Dataset",
            "Seed"
        ]

        paired = pd.merge(
            a_df[keys + [metric]],
            b_df[keys + [metric]],
            on=keys,
            suffixes=("_a", "_b")
        )

        if len(paired) < 2:

            continue

        x = paired[
            f"{metric}_a"
        ].to_numpy()

        y = paired[
            f"{metric}_b"
        ].to_numpy()

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

        # ----------------------------------------------------
        # Means
        # ----------------------------------------------------

        m1 = x.mean()
        m2 = y.mean()

        # ----------------------------------------------------
        # Conclusion
        # ----------------------------------------------------

        conclusion = make_conclusion(
            metric,
            a,
            b,
            m1,
            m2,
            p
        )

        rows.append({

            "Method 1": a,

            "Method 2": b,

            "Mean 1": m1,

            "Mean 2": m2,

            "Difference": m1 - m2,

            "t statistic": t,

            "p value": p,

            "Cohens d": d,

            "Conclusion": conclusion
        })

    out = pd.DataFrame(
        rows
    )

    if not out.empty:

        out["Adjusted p"] = (
            multipletests(
                out["p value"],
                method="fdr_bh"
            )[1]
        )

        out["Significant"] = (
            out["Adjusted p"] < 0.05
        )

    return out


# ============================================================
# Statistical tests
# ============================================================

for group, name in [
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
    )
]:

    for metric in (
        PERFORMANCE_METRICS
        + COMMON_METRICS
    ):

        if metric not in df.columns:
            continue

        results = pairwise(
            group,
            metric
        )

        results.to_csv(
            os.path.join(
                OUTPUT_DIR,
                f"{name}_{metric.replace(' ', '_').lower()}_ttests.csv"
            ),
            index=False
        )


# ============================================================
# Feature-count comparisons
#
# These are written separately because they describe the
# feature representation rather than predictive performance.
# ============================================================

for group, name in [
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
    )
]:

    for metric in FEATURE_METRICS:

        if metric not in df.columns:
            continue

        results = pairwise(
            group,
            metric
        )

        results.to_csv(
            os.path.join(
                OUTPUT_DIR,
                f"{name}_{metric.replace(' ', '_').lower()}_ttests.csv"
            ),
            index=False
        )


# ============================================================
# Feature reduction summary
# ============================================================

if (
    "N Extracted Features" in df.columns
    and "N Selected Features" in df.columns
):

    df["Feature Reduction"] = (
        df["N Extracted Features"]
        - df["N Selected Features"]
    )

    df["Feature Reduction %"] = np.where(
        df["N Extracted Features"] > 0,
        (
            df["Feature Reduction"]
            / df["N Extracted Features"]
            * 100
        ),
        0
    )

    feature_summary = (
        df
        .groupby([
            "Extractor",
            "Selector"
        ])[
            [
                "N Extracted Features",
                "N Selected Features",
                "Feature Reduction",
                "Feature Reduction %"
            ]
        ]
        .agg([
            "mean",
            "std",
            "median",
            "min",
            "max"
        ])
    )

    feature_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "feature_reduction_summary.csv"
        )
    )


# ============================================================
# Finish
# ============================================================

print(
    f"Analysis complete "
    f"({DATA_TYPE})."
)

print(
    f"Results saved to: {OUTPUT_DIR}"
)
