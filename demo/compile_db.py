"""TEMPO Benchmark Results Compiler and Ingestion Engine.

This module consolidates, cleans, and standardises all canonical experimental
benchmark runs from `benchmark_results/` into a unified SQLite relational database
(`demo/data/demo_benchmarks.db`) and a pre-compiled JSON runtime catalogue
(`demo/data/demo_catalog.json`) for Display Day demonstration.
"""

from __future__ import annotations

import datetime
import glob
import json
import logging
import math
import os
import sqlite3
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("demo.compiler")

DEFAULT_DB_PATH = "demo/data/demo_benchmarks.db"
DEFAULT_JSON_PATH = "demo/data/demo_catalog.json"

TRACK_METADATA: dict[str, dict[str, Any]] = {
    "beed": {
        "display_name": "BEED Bioacoustics Grand Prix",
        "domain": "Bioacoustics",
        "task_type": "classification",
        "primary_metric": "accuracy",
        "is_curated_flagship": True,
        "description": "Multi-class acoustic bird species identification across continuous environmental soundscapes.",
        "baseline_extractor": "tsfresh_efficient",
        "baseline_model": "random_forest",
    },
    "pred-maintenance-w100-cls": {
        "display_name": "AI4I Turbofan Reliability Circuit",
        "domain": "Industrial IoT",
        "task_type": "classification",
        "primary_metric": "accuracy",
        "is_curated_flagship": True,
        "description": "Predictive machine failure classification under high-speed milling and variable thermal load.",
        "baseline_extractor": "tsfresh_efficient",
        "baseline_model": "random_forest",
    },
    "gas-sensor-drift-sub": {
        "display_name": "Gas Sensor Drift Speedway",
        "domain": "Chemical Sensing",
        "task_type": "classification",
        "primary_metric": "accuracy",
        "is_curated_flagship": True,
        "description": "Multi-sensor drift classification across 6 chemical compounds with multi-month temporal degradation.",
        "baseline_extractor": "tsfresh_efficient",
        "baseline_model": "random_forest",
    },
    "drift-bifurcation-reg-1k": {
        "display_name": "Langevin Bifurcation Autostrada",
        "domain": "Nonlinear Dynamics",
        "task_type": "regression",
        "primary_metric": "tau_r2",
        "is_curated_flagship": True,
        "description": "Supercritical pitchfork bifurcation parameter recovery (tau >= tau_c) across 1,000 stochastic trajectories.",
        "baseline_extractor": "tsfresh_efficient",
        "baseline_model": "random_forest",
    },
    "drift-bifurcation-cls": {
        "display_name": "Bifurcation Regime Switchway",
        "domain": "Nonlinear Dynamics",
        "task_type": "classification",
        "primary_metric": "accuracy",
        "is_curated_flagship": False,
        "description": "Binary classification of supercritical pitchfork bifurcation regimes across fluctuating potential wells.",
        "baseline_extractor": "tsfresh_efficient",
        "baseline_model": "random_forest",
    },
    "pred-maintenance-w100-reg": {
        "display_name": "Milling Tool Wear Oval",
        "domain": "Industrial IoT",
        "task_type": "regression",
        "primary_metric": "r2",
        "is_curated_flagship": False,
        "description": "Continuous tool wear degradation estimation across multivariate telemetry streams.",
        "baseline_extractor": "tsfresh_efficient",
        "baseline_model": "random_forest",
    },
    "appliances-energy": {
        "display_name": "Smart Home Energy Circuit",
        "domain": "Building Physics",
        "task_type": "regression",
        "primary_metric": "r2",
        "is_curated_flagship": False,
        "description": "Low-energy house electrical appliance power demand regression from weather and indoor ambient conditions.",
        "baseline_extractor": "tsfresh_efficient",
        "baseline_model": "random_forest",
    },
    "beijing-pm25": {
        "display_name": "Urban Atmospheric Driftway",
        "domain": "Environmental Science",
        "task_type": "regression",
        "primary_metric": "r2",
        "is_curated_flagship": False,
        "description": "Multivariate particulate matter (PM2.5) air pollution regression under atmospheric stagnation.",
        "baseline_extractor": "tsfresh_efficient",
        "baseline_model": "random_forest",
    },
    "har": {
        "display_name": "Human Kinematics Super-Sprint",
        "domain": "Biomechanics",
        "task_type": "classification",
        "primary_metric": "accuracy",
        "is_curated_flagship": False,
        "description": "Tri-axial smartphone accelerometer and gyroscope human activity recognition benchmark.",
        "baseline_extractor": "tsfresh_efficient",
        "baseline_model": "random_forest",
    },
}

SELECTOR_NORMALIZATION_MAP = {
    "select_k_best": "select_k_best_anova",
    "subsampled": "subsampled_fdr",
}


def normalize_selector(value: Any) -> str:
    """Normalize null, None, or empty selector strings to 'None' and standardize aliases."""
    if value is None or pd.isna(value):
        return "None"
    s = str(value).strip()
    if s == "" or s.lower() in ("none", "null", "nan"):
        return "None"
    return SELECTOR_NORMALIZATION_MAP.get(s, s)


def load_canonical_csvs(repo_root: str = ".") -> pd.DataFrame:
    """Scan and ingest all non-archived canonical benchmark_results.csv files."""
    pattern = os.path.join(repo_root, "benchmark_results/**/benchmark_results.csv")
    csv_paths = sorted(glob.glob(pattern, recursive=True))
    dataframes: list[pd.DataFrame] = []

    for path in csv_paths:
        if "archive" in path:
            continue
        try:
            df = pd.read_csv(path)
            rel_parts = os.path.relpath(path, repo_root).split(os.sep)
            run_id = rel_parts[1] if len(rel_parts) > 1 else "unknown"
            df["source_run_id"] = run_id
            df["source_csv_path"] = path
            dataframes.append(df)
            logger.info("Loaded %d rows from %s", len(df), path)
        except Exception as err:
            logger.warning("Failed to load %s: %s", path, err)

    if not dataframes:
        raise RuntimeError("No benchmark CSVs found in repository.")

    master_df = pd.concat(dataframes, ignore_index=True)
    logger.info("Total raw rows loaded: %d", len(master_df))
    return master_df


def clean_and_harmonize_master(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names, types, and metric aliases."""
    clean = df.copy()

    # Normalize selector
    clean["Selector"] = clean["Selector"].apply(normalize_selector)

    # Standardize fit time
    if "Fit Time (s)" not in clean.columns and "fit_time_seconds" in clean.columns:
        clean["Fit Time (s)"] = clean["fit_time_seconds"]
    elif "Fit Time (s)" in clean.columns and "fit_time_seconds" in clean.columns:
        clean["Fit Time (s)"] = clean["Fit Time (s)"].fillna(clean["fit_time_seconds"])

    # Standardize inference latency
    if "Inference Latency (ms)" not in clean.columns and "inference_latency_ms" in clean.columns:
        clean["Inference Latency (ms)"] = clean["inference_latency_ms"]
    elif "Inference Latency (ms)" in clean.columns and "inference_latency_ms" in clean.columns:
        clean["Inference Latency (ms)"] = clean["Inference Latency (ms)"].fillna(clean["inference_latency_ms"])

    # Compute total latency (Extraction + Selection + Fit)
    ext_time = clean["Extraction Time (s)"].fillna(0.0)
    sel_time = clean["Selection Time (s)"].fillna(0.0) if "Selection Time (s)" in clean.columns else 0.0
    fit_time = clean["Fit Time (s)"].fillna(0.0)
    clean["Total Latency (s)"] = ext_time + sel_time + fit_time

    # Standardize numeric columns
    numeric_cols = [
        "Extraction Time (s)",
        "Extraction Peak RAM (MB)",
        "Selection Time (s)",
        "Fit Time (s)",
        "Inference Latency (ms)",
        "Total Latency (s)",
        "N Extracted Features",
        "N Selected Features",
        "Feature Reduction (%)",
        "Accuracy",
        "R2",
        "RMSE",
        "MAE",
        "Tau R2",
        "Tau RMSE",
        "Tau MAE",
        "Forecast RMSE",
        "Forecast MAE",
    ]
    for col in numeric_cols:
        if col in clean.columns:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")
        else:
            clean[col] = np.nan

    return clean


def calculate_pareto_optimality(group: pd.DataFrame, score_col: str, lat_col: str, ram_col: str) -> pd.Series:
    """Compute Pareto frontier membership across performance, latency, and peak RAM."""
    n = len(group)
    is_pareto = np.ones(n, dtype=bool)

    scores = group[score_col].to_numpy()
    lats = group[lat_col].to_numpy()
    rams = group[ram_col].to_numpy()

    for i in range(n):
        if np.isnan(scores[i]) or np.isnan(lats[i]) or np.isnan(rams[i]):
            is_pareto[i] = False
            continue
        for j in range(n):
            if i == j:
                continue
            if np.isnan(scores[j]) or np.isnan(lats[j]) or np.isnan(rams[j]):
                continue
            # j dominates i if j has >= score, <= latency, <= ram, and at least one strictly better
            if (scores[j] >= scores[i]) and (lats[j] <= lats[i]) and (rams[j] <= rams[i]):
                if (scores[j] > scores[i]) or (lats[j] < lats[i]) or (rams[j] < rams[i]):
                    is_pareto[i] = False
                    break

    return pd.Series(is_pareto, index=group.index)


def aggregate_configurations(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Group by (Dataset, Extractor, Selector, Model) and calculate mean, std, min, max."""
    group_cols = ["Dataset", "Task", "Extractor", "Selector", "Model"]

    records: list[dict[str, Any]] = []

    for (dataset, task, extractor, selector, model), sub in clean_df.groupby(group_cols):
        n_samples = len(sub)

        # Extraction time stats
        ext_series = sub["Extraction Time (s)"].dropna()
        ext_mean = float(ext_series.mean()) if len(ext_series) > 0 else 0.0
        ext_std = float(ext_series.std(ddof=0)) if len(ext_series) > 1 else 0.0
        ext_min = float(ext_series.min()) if len(ext_series) > 0 else 0.0
        ext_max = float(ext_series.max()) if len(ext_series) > 0 else 0.0

        # RAM stats
        ram_series = sub["Extraction Peak RAM (MB)"].dropna()
        ram_mean = float(ram_series.mean()) if len(ram_series) > 0 else 0.0
        ram_std = float(ram_series.std(ddof=0)) if len(ram_series) > 1 else 0.0
        ram_min = float(ram_series.min()) if len(ram_series) > 0 else 0.0
        ram_max = float(ram_series.max()) if len(ram_series) > 0 else 0.0

        # Latencies
        sel_time_mean = float(sub["Selection Time (s)"].dropna().mean()) if len(sub["Selection Time (s)"].dropna()) > 0 else 0.0
        fit_time_mean = float(sub["Fit Time (s)"].dropna().mean()) if len(sub["Fit Time (s)"].dropna()) > 0 else 0.0
        infer_mean = float(sub["Inference Latency (ms)"].dropna().mean()) if len(sub["Inference Latency (ms)"].dropna()) > 0 else 0.0
        tot_lat_mean = ext_mean + sel_time_mean + fit_time_mean

        # Features
        n_ext = int(sub["N Extracted Features"].dropna().median()) if len(sub["N Extracted Features"].dropna()) > 0 else 0
        n_sel = int(sub["N Selected Features"].dropna().median()) if len(sub["N Selected Features"].dropna()) > 0 else n_ext

        # Metric score depending on task
        if task == "classification":
            score_series = sub["Accuracy"].dropna()
        elif dataset in ("drift-bifurcation-reg", "drift-bifurcation-reg-1k"):
            score_series = sub["Tau R2"].dropna()
            if len(score_series) == 0:
                score_series = sub["R2"].dropna()
        elif task == "forecasting":
            # For forecasting, use 1.0 - normalized RMSE as score
            rmse_series = sub["Forecast RMSE"].dropna()
            score_series = -rmse_series if len(rmse_series) > 0 else pd.Series(dtype=float)
        else:
            score_series = sub["R2"].dropna()

        score_mean = float(score_series.mean()) if len(score_series) > 0 else 0.0
        score_std = float(score_series.std(ddof=0)) if len(score_series) > 1 else 0.0
        score_min = float(score_series.min()) if len(score_series) > 0 else 0.0
        score_max = float(score_series.max()) if len(score_series) > 0 else 0.0

        records.append({
            "track_id": dataset,
            "task_type": task,
            "extractor": extractor,
            "selector": selector,
            "model": model,
            "sample_count": n_samples,
            "extraction_time_mean": ext_mean,
            "extraction_time_std": ext_std,
            "extraction_time_min": ext_min,
            "extraction_time_max": ext_max,
            "extraction_ram_mean": ram_mean,
            "extraction_ram_std": ram_std,
            "extraction_ram_min": ram_min,
            "extraction_ram_max": ram_max,
            "selection_time_mean": sel_time_mean,
            "fit_time_mean": fit_time_mean,
            "inference_latency_mean": infer_mean,
            "total_latency_mean": tot_lat_mean,
            "n_extracted_mean": n_ext,
            "n_selected_mean": n_sel,
            "score_mean": score_mean,
            "score_std": score_std,
            "score_min": score_min,
            "score_max": score_max,
        })

    agg_df = pd.DataFrame(records)

    # Compute relative speedups and RAM reductions vs tsfresh baseline within each track
    speedup_col = []
    ram_reduc_col = []
    pareto_score_col = []

    for track_id, sub_track in agg_df.groupby("track_id"):
        # Find baseline tsfresh_efficient or tsfresh_minimal with None selector
        baseline_match = sub_track[
            (sub_track["extractor"] == "tsfresh_efficient") &
            (sub_track["selector"] == "None")
        ]
        if baseline_match.empty:
            baseline_match = sub_track[sub_track["extractor"].str.contains("tsfresh")]

        base_ext_time = float(baseline_match["extraction_time_mean"].median()) if not baseline_match.empty else 1.0
        base_ram = float(baseline_match["extraction_ram_mean"].median()) if not baseline_match.empty else 1000.0

        for _, row in sub_track.iterrows():
            cur_time = max(row["extraction_time_mean"], 1e-4)
            speedup = base_ext_time / cur_time if cur_time > 0 else 1.0

            cur_ram = row["extraction_ram_mean"]
            ram_reduc = ((base_ram - cur_ram) / base_ram) * 100.0 if base_ram > 0 else 0.0

            # Composite Pareto Score:
            # Score = 100 * ScoreNorm - 15 * log10(max(tot_lat, 0.001)) - 5 * (RAM / 1000)
            norm_score = max(0.0, min(1.0, row["score_mean"]))
            tot_lat = max(row["total_latency_mean"], 1e-3)
            pareto_val = (100.0 * norm_score) - (15.0 * math.log10(tot_lat)) - (5.0 * (cur_ram / 1000.0))

            speedup_col.append((row.name, speedup))
            ram_reduc_col.append((row.name, ram_reduc))
            pareto_score_col.append((row.name, pareto_val))

    for idx, val in speedup_col:
        agg_df.loc[idx, "speedup_vs_tsfresh"] = round(val, 2)
    for idx, val in ram_reduc_col:
        agg_df.loc[idx, "ram_reduction_pct"] = round(val, 2)
    for idx, val in pareto_score_col:
        agg_df.loc[idx, "pareto_composite_score"] = round(val, 2)

    # Compute Pareto optimality per track
    agg_df["is_pareto_optimal"] = False
    for track_id, group in agg_df.groupby("track_id"):
        pareto_mask = calculate_pareto_optimality(group, "score_mean", "total_latency_mean", "extraction_ram_mean")
        agg_df.loc[group.index, "is_pareto_optimal"] = pareto_mask

    return agg_df


def build_sqlite_database(clean_df: pd.DataFrame, agg_df: pd.DataFrame, db_path: str = DEFAULT_DB_PATH) -> None:
    """Write raw runs, metadata, and aggregated tables to SQLite."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # 1. Raw runs table
    cur.execute("""
    CREATE TABLE raw_benchmark_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_run_id TEXT NOT NULL,
        dataset TEXT NOT NULL,
        task_type TEXT NOT NULL,
        seed INTEGER,
        extractor TEXT NOT NULL,
        selector TEXT NOT NULL,
        model TEXT NOT NULL,
        is_cached INTEGER,
        fallback_triggered INTEGER,
        extraction_time_s REAL,
        extraction_ram_mb REAL,
        selection_time_s REAL,
        fit_time_s REAL,
        inference_latency_ms REAL,
        total_latency_s REAL,
        n_extracted_features INTEGER,
        n_selected_features INTEGER,
        feature_reduction_pct REAL,
        accuracy REAL,
        r2 REAL,
        rmse REAL,
        mae REAL,
        tau_r2 REAL,
        tau_rmse REAL,
        tau_mae REAL,
        forecast_rmse REAL,
        forecast_mae REAL
    )
    """)

    raw_insert_rows = []
    for _, r in clean_df.iterrows():
        raw_insert_rows.append((
            str(r.get("source_run_id", "")),
            str(r.get("Dataset", "")),
            str(r.get("Task", "")),
            int(r.get("Seed", 42)) if pd.notna(r.get("Seed")) else 42,
            str(r.get("Extractor", "")),
            str(r.get("Selector", "None")),
            str(r.get("Model", "")),
            1 if r.get("is_cached") is True else 0,
            1 if r.get("fallback_triggered") is True else 0,
            float(r.get("Extraction Time (s)")) if pd.notna(r.get("Extraction Time (s)")) else None,
            float(r.get("Extraction Peak RAM (MB)")) if pd.notna(r.get("Extraction Peak RAM (MB)")) else None,
            float(r.get("Selection Time (s)")) if pd.notna(r.get("Selection Time (s)")) else None,
            float(r.get("Fit Time (s)")) if pd.notna(r.get("Fit Time (s)")) else None,
            float(r.get("Inference Latency (ms)")) if pd.notna(r.get("Inference Latency (ms)")) else None,
            float(r.get("Total Latency (s)")) if pd.notna(r.get("Total Latency (s)")) else None,
            int(r.get("N Extracted Features")) if pd.notna(r.get("N Extracted Features")) else None,
            int(r.get("N Selected Features")) if pd.notna(r.get("N Selected Features")) else None,
            float(r.get("Feature Reduction (%)")) if pd.notna(r.get("Feature Reduction (%)")) else None,
            float(r.get("Accuracy")) if pd.notna(r.get("Accuracy")) else None,
            float(r.get("R2")) if pd.notna(r.get("R2")) else None,
            float(r.get("RMSE")) if pd.notna(r.get("RMSE")) else None,
            float(r.get("MAE")) if pd.notna(r.get("MAE")) else None,
            float(r.get("Tau R2")) if pd.notna(r.get("Tau R2")) else None,
            float(r.get("Tau RMSE")) if pd.notna(r.get("Tau RMSE")) else None,
            float(r.get("Tau MAE")) if pd.notna(r.get("Tau MAE")) else None,
            float(r.get("Forecast RMSE")) if pd.notna(r.get("Forecast RMSE")) else None,
            float(r.get("Forecast MAE")) if pd.notna(r.get("Forecast MAE")) else None,
        ))

    cur.executemany("""
    INSERT INTO raw_benchmark_runs (
        source_run_id, dataset, task_type, seed, extractor, selector, model,
        is_cached, fallback_triggered, extraction_time_s, extraction_ram_mb,
        selection_time_s, fit_time_s, inference_latency_ms, total_latency_s,
        n_extracted_features, n_selected_features, feature_reduction_pct,
        accuracy, r2, rmse, mae, tau_r2, tau_rmse, tau_mae, forecast_rmse, forecast_mae
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, raw_insert_rows)

    # 2. Tracks table
    cur.execute("""
    CREATE TABLE tracks (
        track_id TEXT PRIMARY KEY,
        display_name TEXT NOT NULL,
        domain TEXT NOT NULL,
        task_type TEXT NOT NULL,
        primary_metric TEXT NOT NULL,
        is_curated_flagship INTEGER NOT NULL,
        description TEXT NOT NULL,
        baseline_extractor TEXT NOT NULL,
        baseline_model TEXT NOT NULL
    )
    """)

    track_rows = []
    unique_tracks = sorted(clean_df["Dataset"].unique())
    for t_id in unique_tracks:
        meta = TRACK_METADATA.get(t_id, {
            "display_name": t_id.replace("-", " ").title(),
            "domain": "General Benchmark",
            "task_type": "classification",
            "primary_metric": "accuracy",
            "is_curated_flagship": False,
            "description": f"Benchmark evaluation dataset for {t_id}.",
            "baseline_extractor": "tsfresh_efficient",
            "baseline_model": "random_forest",
        })
        track_rows.append((
            t_id,
            meta["display_name"],
            meta["domain"],
            meta["task_type"],
            meta["primary_metric"],
            1 if meta.get("is_curated_flagship", False) else 0,
            meta["description"],
            meta["baseline_extractor"],
            meta["baseline_model"],
        ))

    cur.executemany("INSERT INTO tracks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", track_rows)

    # 3. Aggregated configurations table
    cur.execute("""
    CREATE TABLE aggregated_configs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        track_id TEXT NOT NULL,
        task_type TEXT NOT NULL,
        extractor TEXT NOT NULL,
        selector TEXT NOT NULL,
        model TEXT NOT NULL,
        sample_count INTEGER NOT NULL,
        extraction_time_mean REAL,
        extraction_time_std REAL,
        extraction_time_min REAL,
        extraction_time_max REAL,
        extraction_ram_mean REAL,
        extraction_ram_std REAL,
        extraction_ram_min REAL,
        extraction_ram_max REAL,
        selection_time_mean REAL,
        fit_time_mean REAL,
        inference_latency_mean REAL,
        total_latency_mean REAL,
        n_extracted_mean INTEGER,
        n_selected_mean INTEGER,
        score_mean REAL,
        score_std REAL,
        score_min REAL,
        score_max REAL,
        speedup_vs_tsfresh REAL,
        ram_reduction_pct REAL,
        is_pareto_optimal INTEGER,
        pareto_composite_score REAL,
        FOREIGN KEY (track_id) REFERENCES tracks(track_id)
    )
    """)

    agg_insert_rows = []
    for _, r in agg_df.iterrows():
        agg_insert_rows.append((
            str(r["track_id"]),
            str(r["task_type"]),
            str(r["extractor"]),
            str(r["selector"]),
            str(r["model"]),
            int(r["sample_count"]),
            float(r["extraction_time_mean"]),
            float(r["extraction_time_std"]),
            float(r["extraction_time_min"]),
            float(r["extraction_time_max"]),
            float(r["extraction_ram_mean"]),
            float(r["extraction_ram_std"]),
            float(r["extraction_ram_min"]),
            float(r["extraction_ram_max"]),
            float(r["selection_time_mean"]),
            float(r["fit_time_mean"]),
            float(r["inference_latency_mean"]),
            float(r["total_latency_mean"]),
            int(r["n_extracted_mean"]),
            int(r["n_selected_mean"]),
            float(r["score_mean"]),
            float(r["score_std"]),
            float(r["score_min"]),
            float(r["score_max"]),
            float(r["speedup_vs_tsfresh"]),
            float(r["ram_reduction_pct"]),
            1 if r["is_pareto_optimal"] else 0,
            float(r["pareto_composite_score"]),
        ))

    cur.executemany("""
    INSERT INTO aggregated_configs (
        track_id, task_type, extractor, selector, model, sample_count,
        extraction_time_mean, extraction_time_std, extraction_time_min, extraction_time_max,
        extraction_ram_mean, extraction_ram_std, extraction_ram_min, extraction_ram_max,
        selection_time_mean, fit_time_mean, inference_latency_mean, total_latency_mean,
        n_extracted_mean, n_selected_mean, score_mean, score_std, score_min, score_max,
        speedup_vs_tsfresh, ram_reduction_pct, is_pareto_optimal, pareto_composite_score
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, agg_insert_rows)

    # 4. Indices
    cur.execute("CREATE INDEX idx_raw_track ON raw_benchmark_runs(dataset, extractor, selector, model)")
    cur.execute("CREATE INDEX idx_agg_lookup ON aggregated_configs(track_id, extractor, selector, model)")
    cur.execute("CREATE INDEX idx_agg_pareto ON aggregated_configs(track_id, is_pareto_optimal)")

    con.commit()
    con.close()
    logger.info("Successfully built SQLite database at %s (%d raw, %d aggregated)", db_path, len(raw_insert_rows), len(agg_insert_rows))


def export_runtime_json(agg_df: pd.DataFrame, json_path: str = DEFAULT_JSON_PATH) -> None:
    """Export aggregated configurations into hierarchical JSON runtime catalogue."""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    tracks_catalog: dict[str, Any] = {}
    flagship_tracks: list[str] = []

    for track_id, group in agg_df.groupby("track_id"):
        meta = TRACK_METADATA.get(track_id, {
            "display_name": track_id.replace("-", " ").title(),
            "domain": "General Benchmark",
            "task_type": group["task_type"].iloc[0],
            "primary_metric": "accuracy",
            "is_curated_flagship": False,
            "description": f"Benchmark evaluation dataset for {track_id}.",
            "baseline_extractor": "tsfresh_efficient",
            "baseline_model": "random_forest",
        })

        is_flagship = meta.get("is_curated_flagship", False)
        if is_flagship:
            flagship_tracks.append(track_id)

        avail_extractors = sorted(group["extractor"].unique().tolist())
        avail_selectors = sorted(group["selector"].unique().tolist())
        avail_models = sorted(group["model"].unique().tolist())

        configs_dict: dict[str, Any] = {}
        pareto_keys: list[str] = []

        for _, row in group.iterrows():
            key = f"{row['extractor']}|{row['selector']}|{row['model']}"
            is_pareto = bool(row["is_pareto_optimal"])
            if is_pareto:
                pareto_keys.append(key)

            configs_dict[key] = {
                "extractor": row["extractor"],
                "selector": row["selector"],
                "model": row["model"],
                "sample_count": int(row["sample_count"]),
                "extraction_time_s": round(float(row["extraction_time_mean"]), 4),
                "extraction_time_bounds": [
                    round(float(row["extraction_time_min"]), 4),
                    round(float(row["extraction_time_max"]), 4),
                ],
                "extraction_ram_mb": round(float(row["extraction_ram_mean"]), 2),
                "extraction_ram_bounds": [
                    round(float(row["extraction_ram_min"]), 2),
                    round(float(row["extraction_ram_max"]), 2),
                ],
                "selection_time_s": round(float(row["selection_time_mean"]), 4),
                "fit_time_s": round(float(row["fit_time_mean"]), 4),
                "inference_latency_ms": round(float(row["inference_latency_mean"]), 4),
                "total_latency_s": round(float(row["total_latency_mean"]), 4),
                "n_extracted": int(row["n_extracted_mean"]),
                "n_selected": int(row["n_selected_mean"]),
                "score": round(float(row["score_mean"]), 4),
                "score_std": round(float(row["score_std"]), 4),
                "score_bounds": [
                    round(float(row["score_min"]), 4),
                    round(float(row["score_max"]), 4),
                ],
                "speedup_vs_tsfresh": round(float(row["speedup_vs_tsfresh"]), 2),
                "ram_reduction_pct": round(float(row["ram_reduction_pct"]), 2),
                "is_pareto_optimal": is_pareto,
                "pareto_composite_score": round(float(row["pareto_composite_score"]), 2),
            }

        tracks_catalog[track_id] = {
            "track_id": track_id,
            "display_name": meta["display_name"],
            "domain": meta["domain"],
            "task_type": meta["task_type"],
            "primary_metric": meta["primary_metric"],
            "is_curated_flagship": is_flagship,
            "description": meta["description"],
            "available_extractors": avail_extractors,
            "available_selectors": avail_selectors,
            "available_models": avail_models,
            "baseline": {
                "extractor": meta["baseline_extractor"],
                "selector": "None",
                "model": meta["baseline_model"],
            },
            "pareto_frontier": pareto_keys,
            "configs": configs_dict,
        }

    output_payload = {
        "version": "1.0.0",
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_canonical_runs": int(agg_df["sample_count"].sum()),
        "total_unique_configs": len(agg_df),
        "flagship_tracks": flagship_tracks,
        "tracks": tracks_catalog,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    logger.info("Successfully exported JSON runtime catalogue to %s (size: %.1f KB)", json_path, os.path.getsize(json_path) / 1024.0)


def compile_all(repo_root: str = ".", db_path: str = DEFAULT_DB_PATH, json_path: str = DEFAULT_JSON_PATH) -> None:
    """Execute complete compilation pipeline: Ingest -> Harmonize -> Aggregate -> SQLite -> JSON."""
    logger.info("Starting TEMPO benchmark results compilation...")
    clean_df = clean_and_harmonize_master(load_canonical_csvs(repo_root))
    agg_df = aggregate_configurations(clean_df)
    build_sqlite_database(clean_df, agg_df, db_path=db_path)
    export_runtime_json(agg_df, json_path=json_path)
    logger.info("Compilation complete!")


if __name__ == "__main__":
    compile_all()
