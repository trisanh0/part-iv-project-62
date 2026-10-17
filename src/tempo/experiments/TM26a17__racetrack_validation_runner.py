#!/usr/bin/env python3
"""
TEMPO Racetrack Validation & Statistical Governance Runner (TM26a17).

Executes the three-module overnight validation suite designed to verify the
benchmarking harness, eliminate statistical pseudoreplication, evaluate feature
selection FDR/power against known ground truth, and measure isolated hardware telemetry.

Modules:
  Module A: Framework Integrity & Leakage Verification (A/A Testing & Blocked CV)
            - 100 paired iterations of Random Forest with dual random_state seeds on identical folds.
            - Null hypothesis p-value calibration and empirical false positive rate (target alpha = 0.05).
            - Blocked CV evaluation verifying prevention of single-class fold collapse.
            - Output: raw_fold_scores_run_a.csv / .parquet

  Module B: Controlled Ground-Truth Benchmark (FDR / Power on Planted Signals)
            - Prior declaration of expected_rankings.json before execution.
            - Clean tabular matrix with P_true = 20 planted signals and P_noise = 180 pure noise features.
            - Monte Carlo evaluation of empirical False Discovery Proportion (FDP) and Statistical Power
              across FDR, Subsampled FDR, ANOVA, and Boruta.
            - N = 10 synthetic dynamic regimes evaluated across 5 extractors with selector fixed to None.
              Seeds within regimes are averaged to provide N = 10 independent points for CD analysis.
            - Output: raw_fold_scores_run_b.csv / .parquet, fdr_power_results.csv

  Module C: Subprocess Telemetry & Memory Isolation Benchmark
            - Isolated child subprocess execution per extractor to reset monotonic ru_maxrss.
            - Platform-aware byte-to-MB normalization (Darwin bytes vs Linux KB).
            - Allocator-clean wall-clock timing runs strictly separated from tracemalloc.
            - Telemetry monitor overhead measurement across repeated trials.
            - Output: telemetry_subprocess_benchmarks.csv, monitor_overhead_benchmarks.csv

Conforms to Contexere RAG standard: TM26a17__racetrack_validation.
"""

import argparse
import datetime
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

# Ensure src is on python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tempo.extraction import (
    numba_efficient_extractor,
    numpy_statistical_extractor,
    polars_statistical_extractor,
    tsfel_extractor,
    tsfresh_extractor,
)
from tempo.selection import (
    SubsampledFeatureSelector,
    boruta_selector,
    tsfresh_selector,
)
from tempo.storage import load_dataset, to_numpy_tensor
from tempo.telemetry import ResourceTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tempo.racetrack_validation")


# ==============================================================================
# Helper Utilities: System Metadata & Ground Truth Registration
# ==============================================================================

def capture_environment_metadata(output_dir: Path) -> Path:
    """Capture runtime environment and git provenance in Contexere format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    env_file = output_dir / "TM26a17__environment.json"

    # Git commit hash
    git_commit = "unknown"
    try:
        git_proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if git_proc.returncode == 0:
            git_commit = git_proc.stdout.strip()
    except Exception as e:
        logger.warning("Could not read git commit: %s", e)

    meta = {
        "timestamp_iso": datetime.datetime.now().isoformat(),
        "git_commit": git_commit,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python_version": sys.version,
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "polars": pl.__version__,
        },
    }

    with open(env_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Recorded environment metadata to %s (git commit: %s)", env_file, git_commit[:8])
    return env_file


def register_expected_rankings(output_dir: Path) -> Path:
    """Register and timestamp a priori hypothesised rankings before Run B execution."""
    rankings_file = output_dir / "expected_rankings.json"
    hypotheses = {
        "declaration_timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "description": "A priori declared hypotheses for Run B synthetic planted-signal benchmark",
        "hypotheses": {
            "feature_selection": {
                "fdr_control": "tsfresh_selector (Benjamini-Hochberg FDR) and SubsampledFeatureSelector will maintain empirical FDP <= 0.05.",
                "anova_inflation": "Standard ANOVA uncorrected p-value thresholding will exhibit severe FDP inflation (> 0.20) under noise columns.",
                "boruta_shadow": "Boruta shadow feature comparison will achieve high power (> 0.85) while controlling FDP <= 0.05.",
            },
            "extractor_scaling": {
                "efficiency_ranking": [
                    "polars_statistics",
                    "numba_efficient",
                    "numpy_statistical",
                    "tsfel",
                    "tsfresh_efficient",
                ],
                "explanation": "Polars and Numba execute vectorized/compiled kernels without DataFrame copying; tsfresh incurs high per-feature extraction overhead.",
            },
        },
        "planted_signal_specification": {
            "p_true": 20,
            "p_noise": 180,
            "total_features": 200,
            "true_feature_indices": list(range(20)),
        },
    }

    with open(rankings_file, "w", encoding="utf-8") as f:
        json.dump(hypotheses, f, indent=2)

    logger.info("A priori expected rankings committed to %s", rankings_file)
    return rankings_file


def extract_features_unified(df_ts: pl.DataFrame, ext_name: str) -> pd.DataFrame:
    """Convert input time series to required representation and extract features."""
    exclude_cols = {"sequence_id", "step", "id", "time"}
    feature_cols = [c for c in df_ts.columns if c not in exclude_cols]

    if ext_name == "polars_statistics":
        df_feat_pl = polars_statistical_extractor(df_ts)
        feat_cols = [c for c in df_feat_pl.columns if c not in exclude_cols]
        features = pd.DataFrame(df_feat_pl.select(feat_cols).to_dict(as_series=False))

    elif ext_name == "numpy_statistical":
        tensor = to_numpy_tensor(df_ts, feature_cols=feature_cols)
        features = numpy_statistical_extractor(tensor, feature_names=feature_cols)

    elif ext_name == "numba_efficient":
        tensor = to_numpy_tensor(df_ts, feature_cols=feature_cols)
        if tensor.ndim == 2:
            tensor = tensor[:, :, np.newaxis]
        features = numba_efficient_extractor(tensor, raw_feature_names=feature_cols, n_fft_coeffs=25)

    elif ext_name == "tsfel":
        tensor = to_numpy_tensor(df_ts, feature_cols=feature_cols)
        if tensor.ndim == 2:
            features = tsfel_extractor(tensor)
        elif tensor.ndim == 3:
            n_channels = tensor.shape[2]
            channel_dfs = []
            for c in range(n_channels):
                df_ch = tsfel_extractor(tensor[:, :, c])
                df_ch.columns = [f"ch{c}_{col}" for col in df_ch.columns]
                channel_dfs.append(df_ch)
            features = pd.concat(channel_dfs, axis=1)
        else:
            raise ValueError(f"Unexpected tensor shape {tensor.shape} for tsfel")

    elif ext_name.startswith("tsfresh"):
        df_pandas = pd.DataFrame(df_ts.to_dict(as_series=False))
        param_set = "minimal" if "minimal" in ext_name else "efficient"
        features = tsfresh_extractor(df_pandas, parameter_set=param_set, fft_coefficients=None)

    else:
        raise ValueError(f"Unknown extractor: {ext_name}")

    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0).reset_index(drop=True)


# ==============================================================================
# Module A: Framework Integrity & Leakage Verification (A/A Testing & Blocked CV)
# ==============================================================================

def run_module_a(
    output_dir: Path,
    n_trials: int = 100,
    n_splits: int = 5,
) -> None:
    """Execute Module A: RF dual-seed A/A null test and Blocked CV class check."""
    logger.info("=" * 75)
    logger.info("STARTING MODULE A: Framework Integrity & Leakage Verification")
    logger.info("Iterations: %d | Folds: %d", n_trials, n_splits)
    logger.info("=" * 75)

    # 1. Load AI4I Predictive Maintenance and generate features
    df_ts, df_tgt = load_dataset("pred-maintenance")
    feat_df = polars_statistical_extractor(df_ts)
    # Align features with targets
    X = feat_df.select([c for c in feat_df.columns if c not in ("sequence_id", "id", "time")]).to_pandas()
    y = df_tgt["target"].to_numpy()

    # Clean infinite/NaN values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    raw_scores_records: List[Dict[str, Any]] = []
    p_values: List[float] = []
    score_diffs: List[float] = []

    for trial in range(n_trials):
        seed_a = trial * 2
        seed_b = trial * 2 + 1
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=trial)

        scores_a: List[float] = []
        scores_b: List[float] = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            clf_a = RandomForestClassifier(n_estimators=50, random_state=seed_a)
            clf_b = RandomForestClassifier(n_estimators=50, random_state=seed_b)

            t0 = time.perf_counter()
            clf_a.fit(X_tr, y_tr)
            fit_time_a = time.perf_counter() - t0

            t0 = time.perf_counter()
            clf_b.fit(X_tr, y_tr)
            fit_time_b = time.perf_counter() - t0

            sc_a = float(clf_a.score(X_te, y_te))
            sc_b = float(clf_b.score(X_te, y_te))

            scores_a.append(sc_a)
            scores_b.append(sc_b)

            # Record raw fold data
            raw_scores_records.append({
                "experiment": "module_a_aa_test",
                "dataset": "pred-maintenance",
                "trial": trial,
                "fold_idx": fold_idx,
                "model_a_seed": seed_a,
                "model_b_seed": seed_b,
                "score_a": sc_a,
                "score_b": sc_b,
                "score_diff": sc_a - sc_b,
                "fit_time_a_s": fit_time_a,
                "fit_time_b_s": fit_time_b,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "test_positives": int(np.sum(y_te)),
            })

        mean_diff = float(np.mean(scores_a) - np.mean(scores_b))
        score_diffs.append(mean_diff)

        # 5-fold paired t-test
        t_stat, p_val = stats.ttest_rel(scores_a, scores_b)
        p_values.append(1.0 if np.isnan(p_val) else float(p_val))

    # Evaluate empirical false positive rate
    p_arr = np.array(p_values)
    empirical_fpr_05 = float(np.mean(p_arr < 0.05))
    empirical_fpr_01 = float(np.mean(p_arr < 0.01))

    logger.info("--- Module A A/A Test Results (N=%d trials) ---", n_trials)
    logger.info("Mean Score Diff: %.5f (std: %.5f)", np.mean(score_diffs), np.std(score_diffs))
    logger.info("Empirical FPR at alpha=0.05: %.2f%% (nominal: 5.0%%)", empirical_fpr_05 * 100.0)
    logger.info("Empirical FPR at alpha=0.01: %.2f%% (nominal: 1.0%%)", empirical_fpr_01 * 100.0)

    # 2. Blocked CV Class Balance Audit
    # Compare naive contiguous blocking vs Stratified Time-Blocked K-Fold on pred-maintenance-w100-cls
    try:
        _, df_tgt_w100 = load_dataset("pred-maintenance-w100-cls")
        y_w100 = df_tgt_w100["target"].to_numpy()
        n_samples = len(y_w100)

        # Naive contiguous chunking
        contiguous_chunks = np.array_split(np.arange(n_samples), n_splits)
        for fold_idx, chunk in enumerate(contiguous_chunks):
            pos_count = int(np.sum(y_w100[chunk]))
            neg_count = len(chunk) - pos_count
            raw_scores_records.append({
                "experiment": "module_a_blocked_cv_audit",
                "dataset": "pred-maintenance-w100-cls",
                "cv_strategy": "naive_contiguous",
                "fold_idx": fold_idx,
                "total_samples": len(chunk),
                "positives": pos_count,
                "negatives": neg_count,
                "single_class_collapse": (pos_count == 0 or neg_count == 0),
            })

        # Stratified time-blocked (sorting by time within class, preserving temporal order)
        pos_indices = np.where(y_w100 == 1)[0]
        neg_indices = np.where(y_w100 == 0)[0]
        pos_splits = np.array_split(pos_indices, n_splits)
        neg_splits = np.array_split(neg_indices, n_splits)

        for fold_idx in range(n_splits):
            test_chunk = np.sort(np.concatenate([pos_splits[fold_idx], neg_splits[fold_idx]]))
            pos_count = int(np.sum(y_w100[test_chunk]))
            neg_count = len(test_chunk) - pos_count
            raw_scores_records.append({
                "experiment": "module_a_blocked_cv_audit",
                "dataset": "pred-maintenance-w100-cls",
                "cv_strategy": "stratified_time_blocked",
                "fold_idx": fold_idx,
                "total_samples": len(test_chunk),
                "positives": pos_count,
                "negatives": neg_count,
                "single_class_collapse": (pos_count == 0 or neg_count == 0),
            })
    except Exception as e:
        logger.warning("Could not execute blocked CV audit on pred-maintenance-w100-cls: %s", e)

    # Save Module A raw results
    df_raw_a = pd.DataFrame(raw_scores_records)
    csv_path = output_dir / "raw_fold_scores_run_a.csv"
    parquet_path = output_dir / "raw_fold_scores_run_a.parquet"
    df_raw_a.to_csv(csv_path, index=False)
    df_raw_a.to_parquet(parquet_path, index=False)
    logger.info("Module A complete. Saved %d raw score records to %s", len(df_raw_a), csv_path)


# ==============================================================================
# Module B: Controlled Ground-Truth Benchmark (FDR / Power on Planted Signals)
# ==============================================================================

def run_module_b_fdr_power(
    output_dir: Path,
    n_monte_carlo: int = 50,
    p_true: int = 20,
    p_noise: int = 180,
    n_samples: int = 250,
) -> None:
    """Evaluate empirical False Discovery Proportion (FDP) and Statistical Power."""
    logger.info("=" * 75)
    logger.info("MODULE B PART 1: Tabular Ground-Truth Feature Selection Evaluation")
    logger.info("Monte Carlo Trials: %d | P_true: %d | P_noise: %d", n_monte_carlo, p_true, p_noise)
    logger.info("=" * 75)

    fdr_power_records: List[Dict[str, Any]] = []
    p_total = p_true + p_noise
    true_feature_set = set([f"feat_{i}" for i in range(p_true)])

    for trial in range(n_monte_carlo):
        rng = np.random.default_rng(1000 + trial)

        # Generate X: true informative features + noise features
        X_mat = rng.normal(0.0, 1.0, size=(n_samples, p_total))
        # Plant non-linear & linear relationships in first p_true columns
        weights = rng.uniform(0.5, 1.5, size=p_true)
        latent_signal = np.dot(X_mat[:, :p_true], weights) + rng.normal(0.0, 0.5, size=n_samples)
        # Binary target via median split
        y_vec = (latent_signal > np.median(latent_signal)).astype(int)

        col_names = [f"feat_{i}" for i in range(p_total)]
        X_df = pd.DataFrame(X_mat, columns=col_names)
        y_ser = pd.Series(y_vec)

        # 1. Method: Benjamini-Hochberg FDR (tsfresh_selector)
        t0 = time.perf_counter()
        try:
            X_sel_fdr = tsfresh_selector(X_df, y_ser, fdr_level=0.05)
            fdr_time = time.perf_counter() - t0
            selected_fdr = set(X_sel_fdr.columns)
        except Exception as e:
            logger.warning("FDR selector failed in trial %d: %s", trial, e)
            selected_fdr = set()
            fdr_time = 0.0

        # 2. Method: Subsampled Feature Selector (10% ratio)
        t0 = time.perf_counter()
        try:
            sub_sel = SubsampledFeatureSelector(
                base_selector="tsfresh",
                sample_ratio=0.10,
                fdr_level=0.05,
                random_state=trial,
            )
            X_sel_sub = sub_sel.fit_transform(X_df, y_ser)
            sub_time = time.perf_counter() - t0
            selected_sub = set(sub_sel.selected_feature_names_ or list(X_sel_sub.columns))
        except Exception as e:
            logger.warning("Subsampled selector failed in trial %d: %s", trial, e)
            selected_sub = set()
            sub_time = 0.0

        # 3. Method: Standard ANOVA F-test without FDR (SelectKBest p < 0.05)
        t0 = time.perf_counter()
        try:
            f_vals, p_vals = f_classif(X_mat, y_vec)
            anova_selected_indices = np.where(p_vals < 0.05)[0]
            selected_anova = set([col_names[i] for i in anova_selected_indices])
            anova_time = time.perf_counter() - t0
        except Exception as e:
            selected_anova = set()
            anova_time = 0.0

        # 4. Method: Boruta Shadow Feature Selection
        t0 = time.perf_counter()
        try:
            X_sel_boruta = boruta_selector(X_df, y_ser, max_iter=20, random_state=trial)
            boruta_time = time.perf_counter() - t0
            selected_boruta = set(X_sel_boruta.columns)
        except Exception as e:
            logger.warning("Boruta failed in trial %d: %s", trial, e)
            selected_boruta = set()
            boruta_time = 0.0

        # Compute metrics for each selector
        selectors_eval = [
            ("tsfresh_bh_fdr", selected_fdr, fdr_time),
            ("subsampled_fdr_0.10", selected_sub, sub_time),
            ("anova_uncorrected", selected_anova, anova_time),
            ("boruta_shadow", selected_boruta, boruta_time),
        ]

        for sel_name, sel_set, sel_duration in selectors_eval:
            n_sel = len(sel_set)
            tp = len(sel_set.intersection(true_feature_set))
            fp = n_sel - tp
            fn = p_true - tp

            fdp = (fp / float(n_sel)) if n_sel > 0 else 0.0
            power = tp / float(p_true)

            fdr_power_records.append({
                "trial": trial,
                "selector": sel_name,
                "n_selected": n_sel,
                "true_positives": tp,
                "false_positives": fp,
                "false_discovery_proportion": fdp,
                "statistical_power": power,
                "duration_seconds": sel_duration,
            })

    df_fdr_power = pd.DataFrame(fdr_power_records)
    out_csv = output_dir / "fdr_power_results.csv"
    df_fdr_power.to_csv(out_csv, index=False)

    # Log summary
    summary = df_fdr_power.groupby("selector").agg({
        "false_discovery_proportion": ["mean", "std"],
        "statistical_power": ["mean", "std"],
        "duration_seconds": "mean",
    })
    logger.info("--- Feature Selector Ground-Truth FDR & Power Summary ---")
    logger.info("\n%s", summary)


def run_module_b_extractor_cd(
    output_dir: Path,
    n_regimes: int = 10,
    seeds_per_regime: int = 5,
    series_len: int = 200,
    n_series: int = 100,
) -> None:
    """Evaluate 5 extractors across N=10 distinct dynamic regimes with selector=None."""
    logger.info("=" * 75)
    logger.info("MODULE B PART 2: Extractor Critical Difference Evaluation")
    logger.info("Independent Regimes: %d | Seeds per Regime: %d (Aggregated)", n_regimes, seeds_per_regime)
    logger.info("=" * 75)

    extractor_names = [
        "polars_statistics",
        "numba_efficient",
        "numpy_statistical",
        "tsfel",
        "tsfresh_efficient",
    ]

    records: List[Dict[str, Any]] = []

    # Define 10 distinct physical regimes (varying frequency, AR drift, noise, and spikes)
    regime_configs = [
        {"freq": 0.05, "noise": 0.10, "spike_prob": 0.01, "desc": "LowFreq_Clean"},
        {"freq": 0.20, "noise": 0.10, "spike_prob": 0.01, "desc": "HighFreq_Clean"},
        {"freq": 0.05, "noise": 0.50, "spike_prob": 0.01, "desc": "LowFreq_Noisy"},
        {"freq": 0.20, "noise": 0.50, "spike_prob": 0.01, "desc": "HighFreq_Noisy"},
        {"freq": 0.10, "noise": 0.20, "spike_prob": 0.05, "desc": "Transient_Spikes"},
        {"freq": 0.02, "noise": 0.20, "spike_prob": 0.00, "desc": "Slow_Drift"},
        {"freq": 0.15, "noise": 0.30, "spike_prob": 0.02, "desc": "MidFreq_Mixed"},
        {"freq": 0.30, "noise": 0.15, "spike_prob": 0.01, "desc": "UltraHighFreq"},
        {"freq": 0.08, "noise": 0.40, "spike_prob": 0.03, "desc": "Perturbed_Harmonic"},
        {"freq": 0.12, "noise": 0.25, "spike_prob": 0.02, "desc": "Standard_Benchmark"},
    ][:n_regimes]

    for regime_idx, cfg in enumerate(regime_configs):
        regime_desc = cfg["desc"]
        logger.info("Running Regime %d/%d: %s", regime_idx + 1, len(regime_configs), regime_desc)

        for s_idx in range(seeds_per_regime):
            seed = regime_idx * 100 + s_idx
            rng = np.random.default_rng(seed)

            # Generate synthetic series
            entities, times, values, targets = [], [], [], []
            for i in range(n_series):
                t = np.arange(series_len)
                amp = rng.uniform(0.5, 2.0)
                phase = rng.uniform(0.0, 2.0 * np.pi)
                signal = amp * np.sin(2.0 * np.pi * cfg["freq"] * t + phase)
                noise = rng.normal(0.0, cfg["noise"], size=series_len)
                spikes = (rng.uniform(0.0, 1.0, size=series_len) < cfg["spike_prob"]) * rng.normal(3.0, 1.0, size=series_len)
                x = signal + noise + spikes

                entities.append(np.full(series_len, i, dtype=np.int32))
                times.append(t.astype(np.int32))
                values.append(x.astype(np.float32))
                targets.append(1 if amp > 1.25 else 0)

            df_ts = pl.DataFrame({
                "sequence_id": np.concatenate(entities),
                "step": np.concatenate(times),
                "value": np.concatenate(values),
            })
            y_arr = np.array(targets, dtype=int)

            for ext_name in extractor_names:
                t0 = time.perf_counter()
                try:
                    X_df = extract_features_unified(df_ts, ext_name)
                    extract_time = time.perf_counter() - t0

                    # 3-fold evaluation
                    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
                    fold_accs = []
                    for tr_idx, te_idx in cv.split(X_df, y_arr):
                        clf = RandomForestClassifier(n_estimators=30, random_state=seed)
                        clf.fit(X_df.iloc[tr_idx], y_arr[tr_idx])
                        fold_accs.append(float(clf.score(X_df.iloc[te_idx], y_arr[te_idx])))

                    mean_acc = float(np.mean(fold_accs))
                except Exception as e:
                    logger.warning("Extractor %s failed in regime %s seed %d: %s", ext_name, regime_desc, seed, e)
                    mean_acc = np.nan
                    extract_time = np.nan
                    X_df = pd.DataFrame()

                records.append({
                    "regime_idx": regime_idx,
                    "regime_desc": regime_desc,
                    "seed": seed,
                    "extractor": ext_name,
                    "selector": "None",
                    "accuracy": mean_acc,
                    "extraction_time_s": extract_time,
                    "n_features": X_df.shape[1] if not X_df.empty else 0,
                })

    df_b = pd.DataFrame(records)
    csv_path = output_dir / "raw_fold_scores_run_b.csv"
    parquet_path = output_dir / "raw_fold_scores_run_b.parquet"
    df_b.to_csv(csv_path, index=False)
    df_b.to_parquet(parquet_path, index=False)

    # Aggregate seeds within each regime (N=10 independent regimes)
    df_regimes = df_b.groupby(["regime_idx", "regime_desc", "extractor"])["accuracy"].mean().reset_index()
    regimes_csv = output_dir / "synthetic_regimes_cd_matrix.csv"
    df_regimes.pivot(index=["regime_idx", "regime_desc"], columns="extractor", values="accuracy").to_csv(regimes_csv)
    logger.info("Module B Part 2 complete. Saved N=%d regime CD matrix to %s", n_regimes, regimes_csv)


# ==============================================================================
# Module C: Subprocess Telemetry & Memory Isolation Benchmark
# ==============================================================================

def _worker_extractor_benchmark(extractor_name: str, n_series: int, series_len: int, pipe: Any) -> None:
    """Isolated subprocess worker measuring clean ru_maxrss per extractor."""
    try:
        import platform
        import resource
        import time
        import numpy as np
        import polars as pl

        # Generate standard synthetic workload
        rng = np.random.default_rng(42)
        entities = np.repeat(np.arange(n_series, dtype=np.int32), series_len)
        times = np.tile(np.arange(series_len, dtype=np.int32), n_series)
        vals = rng.normal(size=n_series * series_len).astype(np.float32)

        df_ts = pl.DataFrame({
            "sequence_id": entities,
            "step": times,
            "value": vals,
        })

        t0 = time.perf_counter()
        res_df = extract_features_unified(df_ts, extractor_name)
        duration = time.perf_counter() - t0

        usage = resource.getrusage(resource.RUSAGE_SELF)
        raw_rss = usage.ru_maxrss

        # Platform-aware memory conversion: Darwin bytes vs Linux KB
        if platform.system() == "Darwin":
            peak_rss_mb = raw_rss / (1024.0 * 1024.0)
        else:
            peak_rss_mb = raw_rss / 1024.0

        n_features = res_df.shape[1]
        pipe.send({
            "status": "success",
            "extractor": extractor_name,
            "duration_s": duration,
            "peak_rss_mb": peak_rss_mb,
            "raw_maxrss": raw_rss,
            "n_features": n_features,
        })
    except Exception as e:
        pipe.send({"status": "error", "error": str(e), "extractor": extractor_name})


def run_module_c(
    output_dir: Path,
    n_repeats: int = 5,
    n_series: int = 500,
    series_len: int = 250,
) -> None:
    """Execute Module C: Subprocess-isolated memory benchmarking and telemetry overhead."""
    logger.info("=" * 75)
    logger.info("STARTING MODULE C: Subprocess Telemetry & Memory Isolation Benchmark")
    logger.info("Subprocess Isolation Repeats: %d | Workload: N=%d, L=%d", n_repeats, n_series, series_len)
    logger.info("=" * 75)

    extractors = [
        "polars_statistics",
        "numba_efficient",
        "numpy_statistical",
        "tsfel",
        "tsfresh_efficient",
    ]

    telemetry_records: List[Dict[str, Any]] = []

    for ext_name in extractors:
        for repeat in range(n_repeats):
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(
                target=_worker_extractor_benchmark,
                args=(ext_name, n_series, series_len, child_conn),
            )
            p.start()
            res = parent_conn.recv()
            p.join()

            if res.get("status") == "success":
                telemetry_records.append({
                    "extractor": ext_name,
                    "repeat": repeat,
                    "duration_s": round(res["duration_s"], 4),
                    "peak_rss_mb": round(res["peak_rss_mb"], 2),
                    "raw_maxrss": res["raw_maxrss"],
                    "n_features": res["n_features"],
                })
                logger.info(
                    "Extractor %s (rep %d): Duration=%.3fs | Isolated Peak RSS=%.2f MB",
                    ext_name, repeat, res["duration_s"], res["peak_rss_mb"],
                )
            else:
                logger.error("Extractor %s failed: %s", ext_name, res.get("error"))

    df_telemetry = pd.DataFrame(telemetry_records)
    out_csv = output_dir / "telemetry_subprocess_benchmarks.csv"
    df_telemetry.to_csv(out_csv, index=False)

    # Telemetry Monitor Overhead Evaluation
    logger.info("Evaluating ResourceTracker monitor overhead (with vs without)...")
    overhead_records: List[Dict[str, Any]] = []

    rng = np.random.default_rng(42)
    entities = np.repeat(np.arange(200, dtype=np.int32), 200)
    times = np.tile(np.arange(200, dtype=np.int32), 200)
    vals = rng.normal(size=200 * 200).astype(np.float32)
    df_sample = pl.DataFrame({"sequence_id": entities, "step": times, "value": vals})

    for trial in range(10):
        # 1. Without monitor
        t0 = time.perf_counter()
        _ = polars_statistical_extractor(df_sample)
        dur_unmonitored = time.perf_counter() - t0

        # 2. With ResourceTracker
        tracker = ResourceTracker(interval=0.01)
        with tracker:
            t0 = time.perf_counter()
            _ = polars_statistical_extractor(df_sample)
            dur_monitored = time.perf_counter() - t0

        overhead_pct = ((dur_monitored - dur_unmonitored) / dur_unmonitored) * 100.0
        overhead_records.append({
            "trial": trial,
            "duration_unmonitored_s": dur_unmonitored,
            "duration_monitored_s": dur_monitored,
            "overhead_pct": overhead_pct,
        })

    df_overhead = pd.DataFrame(overhead_records)
    overhead_csv = output_dir / "monitor_overhead_benchmarks.csv"
    df_overhead.to_csv(overhead_csv, index=False)
    logger.info(
        "Monitor Overhead Evaluation Complete: Mean Overhead = %.2f%% (+/- %.2f%%)",
        df_overhead["overhead_pct"].mean(), df_overhead["overhead_pct"].std(),
    )


# ==============================================================================
# Master Orchestrator & CLI
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="TEMPO Racetrack Validation Runner (TM26a17)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results/TM26a17__racetrack_validation",
        help="Directory where validation artifacts and raw scores are saved",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Execute minimal pre-flight smoke test across all modules (<30s runtime)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = datetime.datetime.now()
    logger.info("=" * 75)
    logger.info("TEMPO TM26a17 Racetrack Validation Suite initiated at %s", t_start.isoformat())
    logger.info("Output Directory: %s", out_dir)
    logger.info("Smoke Test Mode: %s", args.smoke_test)
    logger.info("=" * 75)

    # 1. Capture Environment Metadata
    capture_environment_metadata(out_dir)

    # 2. Register Expected Rankings prior to Run B
    register_expected_rankings(out_dir)

    if args.smoke_test:
        logger.info("Running Smoke Test Configuration...")
        run_module_a(out_dir, n_trials=3, n_splits=3)
        run_module_b_fdr_power(out_dir, n_monte_carlo=2, p_true=5, p_noise=15, n_samples=50)
        run_module_b_extractor_cd(out_dir, n_regimes=2, seeds_per_regime=1, series_len=50, n_series=20)
        run_module_c(out_dir, n_repeats=1, n_series=50, series_len=50)
    else:
        logger.info("Running Full Overnight Validation Suite...")
        run_module_a(out_dir, n_trials=100, n_splits=5)
        run_module_b_fdr_power(out_dir, n_monte_carlo=50, p_true=20, p_noise=180, n_samples=250)
        run_module_b_extractor_cd(out_dir, n_regimes=10, seeds_per_regime=5, series_len=200, n_series=100)
        run_module_c(out_dir, n_repeats=5, n_series=500, series_len=250)

    t_elapsed = (datetime.datetime.now() - t_start).total_seconds()
    logger.info("=" * 75)
    logger.info("RACETRACK VALIDATION RUNNER COMPLETED IN %.1f seconds (%.2f minutes)", t_elapsed, t_elapsed / 60.0)
    logger.info("=" * 75)


if __name__ == "__main__":
    main()
