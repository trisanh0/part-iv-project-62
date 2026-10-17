#!/usr/bin/env python3
"""
Test 3: Determinism and Reproducibility Benchmark.

Evaluates:
  1. Fresh Recompute vs Fresh Recompute: Whether feature matrices and fold scores
     are bit-identical across independent executions, quantifying floating-point
     order-of-reduction perturbations from Numba parallel=True.
  2. Cache Hit vs Fresh Recompute: Whether binary serialization (Parquet) preserves
     exact bit-level representation without degradation.

Output:
  benchmark_results/TM26a18__empirical_mechanisms/03_determinism_reproducibility/reproducibility_report.json
  benchmark_results/TM26a18__empirical_mechanisms/03_determinism_reproducibility/reproducibility_report.csv
"""

import json
import logging
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

# Ensure src is on python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tempo.extraction import numba_efficient_extractor
from tempo.storage import load_dataset, to_numpy_tensor
from tempo.storage.feature_store import FeatureStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tempo.reproducibility_test")


def run_determinism_audit(output_dir: Path) -> None:
    """Execute determinism and caching audit."""
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_cache_dir = output_dir / "test_feature_store"
    if temp_cache_dir.exists():
        shutil.rmtree(temp_cache_dir)
    temp_cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading AI4I dataset...")
    df_ts, df_tgt = load_dataset("pred-maintenance")
    y_arr = df_tgt["target"].to_numpy()

    exclude_cols = {"sequence_id", "id", "step", "time"}
    feature_cols = [c for c in df_ts.columns if c not in exclude_cols]
    tensor = to_numpy_tensor(df_ts, feature_cols=feature_cols)
    if tensor.ndim == 2:
        tensor = tensor[:, :, np.newaxis]

    # Initialize isolated feature store
    fstore = FeatureStore(storage_dir=str(temp_cache_dir), backend="parquet")
    params = {"fft_coefficients": 25}

    logger.info("=== Run 1: Fresh Extraction (Cold Cache) ===")
    t0 = time.perf_counter()
    X1 = numba_efficient_extractor(tensor, raw_feature_names=feature_cols, n_fft_coeffs=25)
    X1 = X1.replace([np.inf, -np.inf], np.nan).fillna(0.0).reset_index(drop=True)
    dur_fresh1 = time.perf_counter() - t0

    # Train 5-fold RF
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores1 = []
    for tr, te in cv.split(X1, y_arr):
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X1.iloc[tr], y_arr[tr])
        scores1.append(float(clf.score(X1.iloc[te], y_arr[te])))

    # Save to feature store
    fstore.save(
        features=X1,
        dataset_name="pred-maintenance",
        extractor_name="numba_efficient",
        params=params,
    )
    logger.info("Run 1 complete. Mean Score: %.5f in %.2fs", np.mean(scores1), dur_fresh1)

    logger.info("=== Run 2: Fresh Recompute (Independent Memory) ===")
    t0 = time.perf_counter()
    X2 = numba_efficient_extractor(tensor, raw_feature_names=feature_cols, n_fft_coeffs=25)
    X2 = X2.replace([np.inf, -np.inf], np.nan).fillna(0.0).reset_index(drop=True)
    dur_fresh2 = time.perf_counter() - t0

    scores2 = []
    for tr, te in cv.split(X2, y_arr):
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X2.iloc[tr], y_arr[tr])
        scores2.append(float(clf.score(X2.iloc[te], y_arr[te])))
    logger.info("Run 2 complete. Mean Score: %.5f in %.2fs", np.mean(scores2), dur_fresh2)

    logger.info("=== Run 3: Cache Hit (Deserialized from Disk) ===")
    t0 = time.perf_counter()
    X_cached = fstore.load(
        dataset_name="pred-maintenance",
        extractor_name="numba_efficient",
        params=params,
    )
    X_cached = X_cached.replace([np.inf, -np.inf], np.nan).fillna(0.0).reset_index(drop=True)
    dur_cache = time.perf_counter() - t0

    scores_cached = []
    for tr, te in cv.split(X_cached, y_arr):
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_cached.iloc[tr], y_arr[tr])
        scores_cached.append(float(clf.score(X_cached.iloc[te], y_arr[te])))
    logger.info("Run 3 complete. Mean Score: %.5f in %.4fs (%.1fx faster)", np.mean(scores_cached), dur_cache, dur_fresh1 / max(1e-4, dur_cache))

    # Comparisons
    # 1. Fresh1 vs Fresh2
    mat1 = X1.to_numpy()
    mat2 = X2.to_numpy()
    mat_cached = X_cached.to_numpy()

    fresh_diff = np.abs(mat1 - mat2)
    fresh_max_diff = float(np.max(fresh_diff))
    fresh_mean_diff = float(np.mean(fresh_diff))
    fresh_bit_identical = bool(np.array_equal(mat1, mat2))
    fresh_allclose_1e12 = bool(np.allclose(mat1, mat2, rtol=1e-12, atol=1e-12))
    fresh_allclose_1e6 = bool(np.allclose(mat1, mat2, rtol=1e-6, atol=1e-6))

    # 2. Fresh1 vs Cache
    cache_diff = np.abs(mat1 - mat_cached)
    cache_max_diff = float(np.max(cache_diff))
    cache_bit_identical = bool(np.array_equal(mat1, mat_cached))

    # 3. Fold Score comparisons
    scores_diff_fresh = [float(s1 - s2) for s1, s2 in zip(scores1, scores2)]
    scores_diff_cache = [float(s1 - sc) for s1, sc in zip(scores1, scores_cached)]
    scores_identical_fresh = bool(scores1 == scores2)
    scores_identical_cache = bool(scores1 == scores_cached)

    report = {
        "dataset": "pred-maintenance",
        "extractor": "numba_efficient",
        "n_sequences": tensor.shape[0],
        "n_timesteps": tensor.shape[1],
        "n_features": X1.shape[1],
        "recompute_audit": {
            "bit_identical": fresh_bit_identical,
            "allclose_1e12": fresh_allclose_1e12,
            "allclose_1e6": fresh_allclose_1e6,
            "max_absolute_difference": fresh_max_diff,
            "mean_absolute_difference": fresh_mean_diff,
            "fold_scores_identical": scores_identical_fresh,
            "fold_scores_diff": scores_diff_fresh,
        },
        "cache_audit": {
            "bit_identical": cache_bit_identical,
            "max_absolute_difference": cache_max_diff,
            "fold_scores_identical": scores_identical_cache,
            "fold_scores_diff": scores_diff_cache,
            "load_speedup": round(dur_fresh1 / max(1e-4, dur_cache), 1),
        },
        "timings_s": {
            "fresh_run_1": round(dur_fresh1, 4),
            "fresh_run_2": round(dur_fresh2, 4),
            "cache_load": round(dur_cache, 4),
        },
        "fold_scores": {
            "run_1": scores1,
            "run_2": scores2,
            "cached": scores_cached,
        },
    }

    report_json_path = output_dir / "reproducibility_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    df_rep = pd.DataFrame([{
        "metric": "Fresh Recompute Bit-Identical",
        "value": str(fresh_bit_identical),
        "detail": f"Max Abs Diff: {fresh_max_diff:.2e}",
    }, {
        "metric": "Fresh Recompute Allclose (1e-12)",
        "value": str(fresh_allclose_1e12),
        "detail": "Numba parallel=True reduction order invariance",
    }, {
        "metric": "Cache Hit Bit-Identical",
        "value": str(cache_bit_identical),
        "detail": f"Max Abs Diff: {cache_max_diff:.2e}",
    }, {
        "metric": "Fold Scores Identical (Fresh)",
        "value": str(scores_identical_fresh),
        "detail": f"Score diff: {scores_diff_fresh}",
    }, {
        "metric": "Fold Scores Identical (Cache)",
        "value": str(scores_identical_cache),
        "detail": f"Score diff: {scores_diff_cache}",
    }, {
        "metric": "Cache Load Speedup",
        "value": f"{report['cache_audit']['load_speedup']}x",
        "detail": f"{dur_fresh1:.2f}s fresh vs {dur_cache:.4f}s cache",
    }])

    csv_path = output_dir / "reproducibility_report.csv"
    df_rep.to_csv(csv_path, index=False)

    logger.info("=== Reproducibility & Determinism Audit Summary ===")
    logger.info("\n%s", df_rep.to_string(index=False))
    logger.info("Saved report to %s", report_json_path)


if __name__ == "__main__":
    out = Path("benchmark_results/TM26a18__empirical_mechanisms/03_determinism_reproducibility")
    run_determinism_audit(out)
