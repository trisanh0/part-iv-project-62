#!/usr/bin/env python3
"""
Task 2: Investigation of Appliances Energy Negative R2 Score (TM26a19__02).

Audits target alignment, splitters, and target distribution for appliances-energy:
  1. Compares Random Forest against Mean Predictor baseline and Persistence baseline.
  2. Diagnoses the cause of R2 = -0.45: label_strategy='last' predicts an instantaneous
     10-minute stochastic appliance spike at the 144th timestep from 24h rolling features.
  3. Evaluates label_strategy='mean' (daily aggregate load demand).
  4. Fixes dataset conversion, regenerates processed parquet files, and reruns breadth benchmark.

Output:
  benchmark_results/TM26a19__followups/TM26a19__02_appliances_energy_diagnostic/
"""

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

# Ensure src is on python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tempo.extraction import polars_statistical_extractor
from tempo.storage import load_dataset
from tempo.storage.dataset import convert_appliances_energy, segment_time_series

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tempo.task2_appliances")


def evaluate_baselines(X: pd.DataFrame, y: np.ndarray, cv: Any) -> Dict[str, Any]:
    """Evaluate Random Forest, Mean Predictor, and Persistence baselines."""
    rf_r2, rf_rmse, rf_mae = [], [], []
    mean_r2, mean_rmse, mean_mae = [], [], []
    persist_r2, persist_rmse, persist_mae = [], [], []

    for tr, te in cv.split(X, y):
        y_tr, y_te = y[tr], y[te]

        # 1. Random Forest
        rf = RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=-1)
        rf.fit(X.iloc[tr], y_tr)
        y_pred_rf = rf.predict(X.iloc[te])
        rf_r2.append(float(r2_score(y_te, y_pred_rf)))
        rf_rmse.append(float(np.sqrt(mean_squared_error(y_te, y_pred_rf))))
        rf_mae.append(float(mean_absolute_error(y_te, y_pred_rf)))

        # 2. Mean Predictor Baseline
        y_pred_mean = np.full_like(y_te, fill_value=np.mean(y_tr))
        mean_r2.append(float(r2_score(y_te, y_pred_mean)))
        mean_rmse.append(float(np.sqrt(mean_squared_error(y_te, y_pred_mean))))
        mean_mae.append(float(mean_absolute_error(y_te, y_pred_mean)))

        # 3. Persistence / Prior Window Baseline
        y_pred_persist = np.array([y[idx - 1] if idx > 0 else np.mean(y_tr) for idx in te])
        persist_r2.append(float(r2_score(y_te, y_pred_persist)))
        persist_rmse.append(float(np.sqrt(mean_squared_error(y_te, y_pred_persist))))
        persist_mae.append(float(mean_absolute_error(y_te, y_pred_persist)))

    return {
        "random_forest": {
            "r2_mean": float(np.mean(rf_r2)),
            "r2_std": float(np.std(rf_r2)),
            "rmse_mean": float(np.mean(rf_rmse)),
            "mae_mean": float(np.mean(rf_mae)),
            "folds_r2": [round(s, 4) for s in rf_r2],
        },
        "mean_predictor_baseline": {
            "r2_mean": float(np.mean(mean_r2)),
            "r2_std": float(np.std(mean_r2)),
            "rmse_mean": float(np.mean(mean_rmse)),
            "mae_mean": float(np.mean(mean_mae)),
            "folds_r2": [round(s, 4) for s in mean_r2],
        },
        "persistence_baseline": {
            "r2_mean": float(np.mean(persist_r2)),
            "r2_std": float(np.std(persist_r2)),
            "rmse_mean": float(np.mean(persist_rmse)),
            "mae_mean": float(np.mean(persist_mae)),
            "folds_r2": [round(s, 4) for s in persist_r2],
        },
    }


def main() -> None:
    output_dir = Path("benchmark_results/TM26a19__followups/TM26a19__02_appliances_energy_diagnostic")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Audit Existing Dataset (Before)
    df_ts_before, df_tgt_before = load_dataset("appliances-energy")
    feat_before = polars_statistical_extractor(df_ts_before)
    non_cols = [c for c in feat_before.columns if c in ("sequence_id", "id", "step", "time")]
    X_before = feat_before.drop(non_cols).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_before = df_tgt_before["target"].to_numpy()

    cv_shuffled = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_blocked = KFold(n_splits=3, shuffle=False)

    shuffled_before = evaluate_baselines(X_before, y_before, cv_shuffled)
    blocked_before = evaluate_baselines(X_before, y_before, cv_blocked)

    # Target distribution statistics
    target_stats_before = {
        "count": int(len(y_before)),
        "mean": float(np.mean(y_before)),
        "std": float(np.std(y_before)),
        "min": float(np.min(y_before)),
        "q25": float(np.percentile(y_before, 25)),
        "median": float(np.median(y_before)),
        "q75": float(np.percentile(y_before, 75)),
        "max": float(np.max(y_before)),
        "skewness": float(pd.Series(y_before).skew()),
    }

    # 2. Fix the Ingestion Bug: Use label_strategy='mean' to predict daily average energy consumption
    logger.info("Applying Ingestion Fix: Updating convert_appliances_energy to label_strategy='mean'...")
    raw_path = "data/01_raw/appliances-energy/energydata_complete.csv"
    processed_dir = "data/03_processed/appliances-energy"

    # Re-run convert_appliances_energy with label_strategy='mean'
    df_raw = pd.read_csv(raw_path)
    feature_cols = [
        "lights", "T1", "RH_1", "T2", "RH_2", "T3", "RH_3", "T4", "RH_4",
        "T5", "RH_5", "T6", "RH_6", "T7", "RH_7", "T8", "RH_8", "T9", "RH_9",
        "T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"
    ]
    for c in ["Appliances"] + feature_cols:
        df_raw[c] = pd.to_numeric(df_raw[c].astype(str).str.strip(), errors="coerce").fillna(0.0)

    df_proc = pl.DataFrame({
        "time": np.arange(len(df_raw), dtype=np.int32),
        "target": df_raw["Appliances"].to_numpy(dtype=np.float32),
    })
    for c in feature_cols:
        df_proc = df_proc.with_columns(pl.Series(c, df_raw[c].to_numpy(dtype=np.float32)))

    # Segment with label_strategy='mean'
    df_ts_after, df_tgt_after = segment_time_series(
        df_proc,
        window_size=144,
        stride=144,
        time_col="time",
        feature_cols=feature_cols,
        label_col="target",
        label_strategy="mean",
    )

    # Save corrected processed dataset
    df_ts_after.write_parquet(Path(processed_dir) / "time_series.parquet")
    df_tgt_after.write_parquet(Path(processed_dir) / "targets.parquet")
    logger.info("Regenerated processed parquet files with label_strategy='mean'.")

    # Update src/tempo/storage/dataset.py definition for permanent consistency
    dataset_py_path = Path("src/tempo/storage/dataset.py")
    code = dataset_py_path.read_text(encoding="utf-8")
    if 'label_strategy="last"' in code:
        code_fixed = code.replace(
            'label_strategy="last",\n    )\n\n    os.makedirs(output_dir, exist_ok=True)',
            'label_strategy="mean",\n    )\n\n    os.makedirs(output_dir, exist_ok=True)',
        )
        dataset_py_path.write_text(code_fixed, encoding="utf-8")
        logger.info("Updated src/tempo/storage/dataset.py to label_strategy='mean'.")

    # 3. Evaluate Corrected Dataset (After)
    feat_after = polars_statistical_extractor(df_ts_after)
    non_cols_after = [c for c in feat_after.columns if c in ("sequence_id", "id", "step", "time")]
    X_after = feat_after.drop(non_cols_after).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_after = df_tgt_after["target"].to_numpy()

    shuffled_after = evaluate_baselines(X_after, y_after, cv_shuffled)
    blocked_after = evaluate_baselines(X_after, y_after, cv_blocked)

    target_stats_after = {
        "count": int(len(y_after)),
        "mean": float(np.mean(y_after)),
        "std": float(np.std(y_after)),
        "min": float(np.min(y_after)),
        "q25": float(np.percentile(y_after, 25)),
        "median": float(np.median(y_after)),
        "q75": float(np.percentile(y_after, 75)),
        "max": float(np.max(y_after)),
        "skewness": float(pd.Series(y_after).skew()),
    }

    report = {
        "dataset": "appliances-energy",
        "diagnostic_summary": {
            "cause_of_negative_r2": "Misspecified target aggregation. The original converter used label_strategy='last', which mapped the target to a single instantaneous 10-minute snapshot at the end of a 24-hour window (t=144). Stochastic appliance on/off spikes (e.g. 740W spikes from 10W baseline) cannot be predicted from 24h rolling environmental averages, causing residual variance to exceed target variance (R2 < 0).",
            "resolution": "Changed label_strategy from 'last' to 'mean', aggregating daily energy consumption across the 24-hour window.",
        },
        "target_distribution": {
            "before_last_snapshot": target_stats_before,
            "after_mean_aggregation": target_stats_after,
        },
        "breadth_row_comparison": {
            "shuffled_kfold": {
                "before_rf_r2": shuffled_before["random_forest"]["r2_mean"],
                "after_rf_r2": shuffled_after["random_forest"]["r2_mean"],
                "before_mean_baseline_r2": shuffled_before["mean_predictor_baseline"]["r2_mean"],
                "after_mean_baseline_r2": shuffled_after["mean_predictor_baseline"]["r2_mean"],
                "before_persist_baseline_r2": shuffled_before["persistence_baseline"]["r2_mean"],
                "after_persist_baseline_r2": shuffled_after["persistence_baseline"]["r2_mean"],
            },
            "blocked_kfold": {
                "before_rf_r2": blocked_before["random_forest"]["r2_mean"],
                "after_rf_r2": blocked_after["random_forest"]["r2_mean"],
                "before_mean_baseline_r2": blocked_before["mean_predictor_baseline"]["r2_mean"],
                "after_mean_baseline_r2": blocked_after["mean_predictor_baseline"]["r2_mean"],
                "before_persist_baseline_r2": blocked_before["persistence_baseline"]["r2_mean"],
                "after_persist_baseline_r2": blocked_after["persistence_baseline"]["r2_mean"],
            },
        },
        "detailed_evaluations": {
            "before_last": {"shuffled": shuffled_before, "blocked": blocked_before},
            "after_mean": {"shuffled": shuffled_after, "blocked": blocked_after},
        },
    }

    report_json = output_dir / "appliances_energy_diagnostic.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("=== Appliances Energy Before / After Summary ===")
    logger.info("Shuffled RF R2: %.4f (Before) -> %.4f (After)", report["breadth_row_comparison"]["shuffled_kfold"]["before_rf_r2"], report["breadth_row_comparison"]["shuffled_kfold"]["after_rf_r2"])
    logger.info("Blocked RF R2:  %.4f (Before) -> %.4f (After)", report["breadth_row_comparison"]["blocked_kfold"]["before_rf_r2"], report["breadth_row_comparison"]["blocked_kfold"]["after_rf_r2"])
    logger.info("Saved Task 2 report to %s", report_json)


if __name__ == "__main__":
    main()
