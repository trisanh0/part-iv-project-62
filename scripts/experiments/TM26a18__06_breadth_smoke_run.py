#!/usr/bin/env python3
"""
Test 6: Breadth Smoke Run across All Seven Datasets.

Evaluates TEMPO's scale-agnostic unified data ingestion and feature extraction
architecture across all seven core physical datasets without dataset-specific code:
  1. BEED (Bangalore EEG Epilepsy Dataset)
  2. AI4I Predictive Maintenance (pred-maintenance)
  3. UCI Human Activity Recognition (har)
  4. Gas Sensor Array Drift (gas-sensor-drift-sub)
  5. Appliances Energy Prediction (appliances-energy)
  6. Beijing PM2.5 Meteorology (beijing-pm25)
  7. Langevin Bifurcation Dynamics (drift-bifurcation-cls)

Records N (samples), L (steps), D (channels), P (features), runtimes, and scores,
proving framework portability across heterogeneous physical domains.

Output:
  benchmark_results/TM26a18__empirical_mechanisms/06_breadth_smoke_run/breadth_smoke_benchmarks.csv
  benchmark_results/TM26a18__empirical_mechanisms/06_breadth_smoke_run/breadth_smoke_table.md
"""

import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold

# Ensure src is on python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tempo.extraction import polars_statistical_extractor
from tempo.storage import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tempo.breadth_test")


DATASETS_SPEC = [
    {
        "name": "beed",
        "domain": "Biomedical Telemetry (EEG)",
        "task": "classification",
        "metric_name": "Accuracy",
    },
    {
        "name": "pred-maintenance",
        "domain": "Industrial IoT & Failure",
        "task": "classification",
        "metric_name": "Accuracy",
    },
    {
        "name": "har",
        "domain": "Inertial Kinetics (IMU)",
        "task": "classification",
        "metric_name": "Accuracy",
    },
    {
        "name": "gas-sensor-drift-sub",
        "domain": "Chemical Sensing Array",
        "task": "classification",
        "metric_name": "Accuracy",
    },
    {
        "name": "appliances-energy",
        "domain": "Building Power & Thermodynamics",
        "task": "regression",
        "metric_name": "R2",
    },
    {
        "name": "beijing-pm25",
        "domain": "Environmental Meteorology",
        "task": "regression",
        "metric_name": "R2",
    },
    {
        "name": "drift-bifurcation-cls",
        "domain": "Stochastic Physics (Langevin)",
        "task": "classification",
        "metric_name": "Accuracy",
    },
]


def run_breadth_smoke_run(output_dir: Path) -> None:
    """Execute breadth smoke test across all seven datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []

    logger.info("=" * 80)
    logger.info("STARTING BREADTH SMOKE RUN ACROSS ALL 7 DATASETS")
    logger.info("=" * 80)

    for spec in DATASETS_SPEC:
        ds_name = spec["name"]
        domain = spec["domain"]
        task = spec["task"]
        metric_name = spec["metric_name"]

        logger.info("--- Evaluating Dataset: %s (%s) ---", ds_name, domain)
        t_start = time.perf_counter()

        try:
            # 1. Ingestion
            t0 = time.perf_counter()
            df_ts, df_tgt = load_dataset(ds_name)
            ingest_time = time.perf_counter() - t0

            # Characterize shapes
            id_col = "sequence_id" if "sequence_id" in df_ts.columns else "id"
            step_col = "step" if "step" in df_ts.columns else "time"
            exclude_cols = {id_col, step_col}
            raw_channel_cols = [c for c in df_ts.columns if c not in exclude_cols]

            N_instances = df_tgt.height if df_tgt is not None else 0
            D_channels = len(raw_channel_cols)
            L_steps = df_ts.height // max(1, N_instances)

            # 2. Fast Feature Extraction via Polars
            t0 = time.perf_counter()
            feat_pl = polars_statistical_extractor(df_ts)
            non_feat = [c for c in feat_pl.columns if c in (id_col, step_col, "id", "time")]
            X_df = feat_pl.drop(non_feat).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0.0)
            extract_time = time.perf_counter() - t0
            P_features = X_df.shape[1]

            # Targets
            y_arr = df_tgt["target"].to_numpy()

            # 3. Model Training & 3-Fold Cross Validation
            t0 = time.perf_counter()
            if task == "classification":
                n_classes = len(np.unique(y_arr))
                # Subsample HAR if N > 2000 for smoke evaluation speed
                if N_instances > 2000:
                    sub_idx = np.random.default_rng(42).choice(N_instances, size=1500, replace=False)
                    X_eval = X_df.iloc[sub_idx].reset_index(drop=True)
                    y_eval = y_arr[sub_idx]
                else:
                    X_eval = X_df
                    y_eval = y_arr

                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                fold_scores = []
                for tr, te in cv.split(X_eval, y_eval):
                    clf = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)
                    clf.fit(X_eval.iloc[tr], y_eval[tr])
                    fold_scores.append(accuracy_score(y_eval[te], clf.predict(X_eval.iloc[te])))
                score_val = float(np.mean(fold_scores))
            else:
                if N_instances > 2000:
                    sub_idx = np.random.default_rng(42).choice(N_instances, size=1500, replace=False)
                    X_eval = X_df.iloc[sub_idx].reset_index(drop=True)
                    y_eval = y_arr[sub_idx]
                else:
                    X_eval = X_df
                    y_eval = y_arr

                cv = KFold(n_splits=3, shuffle=True, random_state=42)
                fold_scores = []
                for tr, te in cv.split(X_eval, y_eval):
                    reg = RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=-1)
                    reg.fit(X_eval.iloc[tr], y_eval[tr])
                    fold_scores.append(r2_score(y_eval[te], reg.predict(X_eval.iloc[te])))
                score_val = float(np.mean(fold_scores))

            model_time = time.perf_counter() - t0
            total_time = time.perf_counter() - t_start

            record = {
                "dataset": ds_name,
                "domain": domain,
                "task": task,
                "N_samples": N_instances,
                "L_steps": L_steps,
                "D_channels": D_channels,
                "P_extracted_features": P_features,
                "metric_name": metric_name,
                "cv_score": round(score_val, 4),
                "ingest_time_s": round(ingest_time, 4),
                "extract_time_s": round(extract_time, 4),
                "model_time_s": round(model_time, 4),
                "total_time_s": round(total_time, 4),
                "status": "SUCCESS",
                "error": None,
            }
            logger.info(
                "SUCCESS: %s | N=%d, L=%d, D=%d, P=%d | %s=%.4f in %.2fs",
                ds_name, N_instances, L_steps, D_channels, P_features,
                metric_name, score_val, total_time,
            )
        except Exception as e:
            logger.error("FAILED on dataset %s: %s", ds_name, e)
            record = {
                "dataset": ds_name,
                "domain": domain,
                "task": task,
                "N_samples": 0,
                "L_steps": 0,
                "D_channels": 0,
                "P_extracted_features": 0,
                "metric_name": metric_name,
                "cv_score": np.nan,
                "ingest_time_s": 0.0,
                "extract_time_s": 0.0,
                "model_time_s": 0.0,
                "total_time_s": round(time.perf_counter() - t_start, 4),
                "status": "FAILED",
                "error": str(e),
            }

        records.append(record)

    df_breadth = pd.DataFrame(records)
    csv_path = output_dir / "breadth_smoke_benchmarks.csv"
    df_breadth.to_csv(csv_path, index=False)

    # Generate Markdown Table for Report
    table_lines = [
        "# TEMPO Universal Dataset Coverage (Breadth Smoke Benchmark)",
        "",
        "| Dataset Identifier | Physical Domain | Task | $N$ (Instances) | $L$ (Steps) | $D$ (Channels) | $P$ (Features) | Metric | CV Score | Total Runtime | Status |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in records:
        table_lines.append(
            f"| `{r['dataset']}` | {r['domain']} | {r['task']} | {r['N_samples']:,} | {r['L_steps']} | {r['D_channels']} | {r['P_extracted_features']} | {r['metric_name']} | **{r['cv_score']}** | {r['total_time_s']:.2f}s | `{r['status']}` |"
        )

    md_path = output_dir / "breadth_smoke_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(table_lines) + "\n")

    logger.info("=== Breadth Smoke Benchmark Summary Table ===")
    logger.info("\n%s", df_breadth[["dataset", "domain", "N_samples", "L_steps", "D_channels", "P_extracted_features", "cv_score", "total_time_s", "status"]].to_string(index=False))
    logger.info("Saved breadth benchmarks to %s and %s", csv_path, md_path)


if __name__ == "__main__":
    out = Path("benchmark_results/TM26a18__empirical_mechanisms/06_breadth_smoke_run")
    run_breadth_smoke_run(out)
