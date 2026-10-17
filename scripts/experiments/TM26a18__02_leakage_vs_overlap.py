#!/usr/bin/env python3
"""
Test 2: Leakage vs Overlap on AI4I Predictive Maintenance.

Repeats the Shuffled vs Blocked cross-validation comparison across sliding window
strides S in {100, 75, 50, 25, 10} corresponding to temporal overlaps from 0.0 to 0.90.
Uses polars_statistical_extractor with Random Forest and Logistic Regression.

Demonstrates that the optimistic performance gap Delta = Score_shuffled - Score_blocked
originates from adjacent window observation leakage, establishing the exact mechanism
for Section 4.1 of the report.

Output:
  benchmark_results/TM26a18__empirical_mechanisms/02_leakage_vs_overlap/leakage_vs_overlap_results.csv
  benchmark_results/TM26a18__empirical_mechanisms/02_leakage_vs_overlap/leakage_curve_summary.csv
"""

import logging
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Ensure src is on python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tempo.extraction import polars_statistical_extractor
from tempo.storage.dataset import segment_time_series

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tempo.leakage_test")


def load_and_preprocess_raw_ai4i(raw_path: str = "data/01_raw/pred-maintenance/ai4i2020.csv") -> pl.DataFrame:
    """Load and prepare continuous AI4I data."""
    df = pl.read_csv(raw_path)
    leak_cols = ["TWF", "HDF", "PWF", "OSF", "RNF", "Product ID"]
    df_filtered = df.drop([c for c in leak_cols if c in df.columns])

    type_mapping = {"L": 0, "M": 1, "H": 2}
    df_proc = df_filtered.with_columns([
        pl.col("UDI").cast(pl.Int32).alias("time"),
        pl.col("Type").replace_strict(type_mapping, default=0).cast(pl.Int32).alias("Type"),
        pl.col("Machine failure").cast(pl.Int32).alias("target"),
    ])
    return df_proc


def run_leakage_sweep(output_dir: Path) -> None:
    """Execute leakage vs overlap stride sweep."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_df = load_and_preprocess_raw_ai4i()

    window_size = 100
    strides = [100, 75, 50, 25, 10]
    n_splits = 5
    n_seeds = 5

    feature_cols = [
        "Type", "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
    ]

    records = []

    for stride in strides:
        overlap_pct = round((1.0 - (stride / window_size)) * 100.0, 1)
        logger.info("=" * 70)
        logger.info("Processing Stride S=%d (Overlap: %.1f%%, W=%d)", stride, overlap_pct, window_size)
        logger.info("=" * 70)

        t0 = time.perf_counter()
        df_ts, df_targets = segment_time_series(
            raw_df,
            window_size=window_size,
            stride=stride,
            time_col="time",
            feature_cols=feature_cols,
            label_col="target",
            label_strategy="any_positive",
        )
        seg_time = time.perf_counter() - t0

        n_windows = df_targets.height
        n_positives = int(df_targets["target"].sum())
        logger.info("Generated %d windows (%d failures) in %.2fs", n_windows, n_positives, seg_time)

        # Feature Extraction
        t0 = time.perf_counter()
        feat_pl = polars_statistical_extractor(df_ts)
        non_feat = [c for c in feat_pl.columns if c in ("sequence_id", "id", "step", "time")]
        X_df = feat_pl.drop(non_feat).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y_arr = df_targets["target"].to_numpy()
        ext_time = time.perf_counter() - t0
        logger.info("Extracted %d features in %.2fs", X_df.shape[1], ext_time)

        # Check if enough failures for 5 splits
        if n_positives < n_splits:
            logger.warning("Insufficient positive samples (%d) for %d splits; skipping", n_positives, n_splits)
            continue

        for seed in range(n_seeds):
            # 1. Shuffled Stratified K-Fold (Standard leaky ML practice)
            cv_shuffled = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

            # 2. Stratified Time-Blocked K-Fold (Leak-free temporal evaluation)
            pos_indices = np.where(y_arr == 1)[0]
            neg_indices = np.where(y_arr == 0)[0]
            pos_splits = np.array_split(pos_indices, n_splits)
            neg_splits = np.array_split(neg_indices, n_splits)
            blocked_splits = []
            for fold_idx in range(n_splits):
                test_idx = np.sort(np.concatenate([pos_splits[fold_idx], neg_splits[fold_idx]]))
                train_idx = np.setdiff1d(np.arange(n_windows), test_idx)
                blocked_splits.append((train_idx, test_idx))

            for model_name in ["random_forest", "logistic_regression"]:
                # Evaluate Shuffled
                accs_shuffled = []
                for tr_idx, te_idx in cv_shuffled.split(X_df, y_arr):
                    if model_name == "random_forest":
                        clf = RandomForestClassifier(n_estimators=50, random_state=seed)
                    else:
                        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
                    clf.fit(X_df.iloc[tr_idx], y_arr[tr_idx])
                    preds = clf.predict(X_df.iloc[te_idx])
                    accs_shuffled.append(accuracy_score(y_arr[te_idx], preds))
                mean_shuffled = float(np.mean(accs_shuffled))

                # Evaluate Blocked
                accs_blocked = []
                for tr_idx, te_idx in blocked_splits:
                    if model_name == "random_forest":
                        clf = RandomForestClassifier(n_estimators=50, random_state=seed)
                    else:
                        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
                    clf.fit(X_df.iloc[tr_idx], y_arr[tr_idx])
                    preds = clf.predict(X_df.iloc[te_idx])
                    accs_blocked.append(accuracy_score(y_arr[te_idx], preds))
                mean_blocked = float(np.mean(accs_blocked))

                gap = mean_shuffled - mean_blocked

                records.append({
                    "stride": stride,
                    "overlap_ratio": round(1.0 - (stride / window_size), 2),
                    "overlap_pct": overlap_pct,
                    "n_windows": n_windows,
                    "n_positives": n_positives,
                    "seed": seed,
                    "model": model_name,
                    "accuracy_shuffled": mean_shuffled,
                    "accuracy_blocked": mean_blocked,
                    "leakage_gap": gap,
                })

    df_results = pd.DataFrame(records)
    csv_path = output_dir / "leakage_vs_overlap_results.csv"
    df_results.to_csv(csv_path, index=False)

    # Summary table
    summary = df_results.groupby(["model", "stride", "overlap_pct", "n_windows"]).agg(
        acc_shuffled_mean=("accuracy_shuffled", "mean"),
        acc_blocked_mean=("accuracy_blocked", "mean"),
        leakage_gap_mean=("leakage_gap", "mean"),
        leakage_gap_std=("leakage_gap", "std"),
    ).reset_index()

    summary_csv = output_dir / "leakage_curve_summary.csv"
    summary.to_csv(summary_csv, index=False)

    logger.info("=== Leakage vs Overlap Curve Summary ===")
    logger.info("\n%s", summary.to_string(index=False))
    logger.info("Leakage curve evaluation complete. Outputs saved to %s", output_dir)


if __name__ == "__main__":
    out = Path("benchmark_results/TM26a18__empirical_mechanisms/02_leakage_vs_overlap")
    run_leakage_sweep(out)
