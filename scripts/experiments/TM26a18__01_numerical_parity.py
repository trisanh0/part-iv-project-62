#!/usr/bin/env python3
"""
Test 1: Numerical Parity (Numba vs tsfresh) on AI4I Windows.

Computes shared features between numba_efficient_extractor and tsfresh_extractor
on identical AI4I Predictive Maintenance sequence windows, evaluates numerical
parity against floating-point tolerances (np.allclose), reports max absolute
differences per feature family, and catalogs mathematical definition nuances.

Output:
  benchmark_results/TM26a18__empirical_mechanisms/01_numerical_parity/parity_results.csv
  benchmark_results/TM26a18__empirical_mechanisms/01_numerical_parity/parity_family_summary.csv
"""

import logging
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import polars as pl

# Ensure src is on python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tempo.extraction import numba_efficient_extractor, tsfresh_extractor
from tempo.storage import load_dataset, to_numpy_tensor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tempo.parity_test")


def assign_feature_family(col: str) -> str:
    """Classify feature column name into mathematical family."""
    col_lower = col.lower()
    if any(k in col_lower for k in ("quantile", "median")):
        return "Quantiles"
    elif "autocorrelation" in col_lower:
        return "Autocorrelation"
    elif any(k in col_lower for k in ("linear_trend", "trend")):
        return "Linear Trend"
    elif any(k in col_lower for k in ("fft", "fourier")):
        return "FFT Coefficients"
    elif any(k in col_lower for k in ("skewness", "kurtosis")):
        return "Higher Order Moments"
    elif any(k in col_lower for k in ("variance", "standard_deviation")):
        return "Dispersion (Var/Std)"
    elif any(k in col_lower for k in ("mean", "minimum", "maximum", "root_mean_square", "abs_energy", "absolute_maximum")):
        return "Location & Energy"
    elif any(k in col_lower for k in ("change", "strike", "count_above", "count_below", "ratio")):
        return "Dynamics & Extremes"
    else:
        return "Other"


def run_parity_evaluation(output_dir: Path, n_sequences: int = 50) -> None:
    """Run numerical parity evaluation on AI4I windows."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loading AI4I dataset (pred-maintenance)...")
    df_ts, _ = load_dataset("pred-maintenance")

    # Select first n_sequences to ensure rapid, rigorous evaluation
    id_col = "sequence_id" if "sequence_id" in df_ts.columns else "id"
    unique_ids = df_ts[id_col].unique().head(n_sequences).to_list()
    df_ts_subset = df_ts.filter(pl.col(id_col).is_in(unique_ids))

    exclude_cols = {id_col, "step", "time"}
    feature_cols = [c for c in df_ts_subset.columns if c not in exclude_cols]
    # For clarity and speed, test on first 3 physical sensor channels
    eval_channels = feature_cols[:3]
    logger.info("Evaluating channels: %s across %d sequences", eval_channels, len(unique_ids))

    df_eval = df_ts_subset.select([id_col, "step" if "step" in df_ts_subset.columns else "time"] + eval_channels)

    # 1. Compute via tsfresh_extractor
    logger.info("Extracting features using tsfresh_extractor...")
    df_pandas = pd.DataFrame(df_eval.to_dict(as_series=False))
    t0 = time.perf_counter()
    tsfresh_feats = tsfresh_extractor(df_pandas, parameter_set="efficient", fft_coefficients=25)
    tsfresh_time = time.perf_counter() - t0
    logger.info("tsfresh extracted %d features in %.2fs", tsfresh_feats.shape[1], tsfresh_time)

    # 2. Compute via numba_efficient_extractor
    logger.info("Extracting features using numba_efficient_extractor...")
    tensor = to_numpy_tensor(df_eval, feature_cols=eval_channels)
    if tensor.ndim == 2:
        tensor = tensor[:, :, np.newaxis]
    t0 = time.perf_counter()
    numba_feats = numba_efficient_extractor(tensor, raw_feature_names=eval_channels, n_fft_coeffs=25)
    numba_time = time.perf_counter() - t0
    logger.info("numba extracted %d features in %.2fs (%.1fx speedup)", numba_feats.shape[1], numba_time, tsfresh_time / max(1e-4, numba_time))

    # Match common columns
    common_cols = sorted(list(set(tsfresh_feats.columns).intersection(set(numba_feats.columns))))
    logger.info("Identified %d exactly matching column names", len(common_cols))

    records = []
    for col in common_cols:
        val_tsfresh = tsfresh_feats[col].to_numpy().astype(np.float64)
        val_numba = numba_feats[col].to_numpy().astype(np.float64)

        # Handle NaNs / Infs consistently
        valid_mask = np.isfinite(val_tsfresh) & np.isfinite(val_numba)
        if not np.any(valid_mask):
            continue

        diff = np.abs(val_tsfresh[valid_mask] - val_numba[valid_mask])
        max_abs_diff = float(np.max(diff))
        mean_abs_diff = float(np.mean(diff))

        # Check tolerances
        is_close_1e5 = bool(np.allclose(val_numba[valid_mask], val_tsfresh[valid_mask], rtol=1e-5, atol=1e-5))
        is_close_1e6 = bool(np.allclose(val_numba[valid_mask], val_tsfresh[valid_mask], rtol=1e-6, atol=1e-6))
        is_exact = bool(np.all(val_numba[valid_mask] == val_tsfresh[valid_mask]))

        family = assign_feature_family(col)
        records.append({
            "feature": col,
            "family": family,
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "is_close_1e5": is_close_1e5,
            "is_close_1e6": is_close_1e6,
            "is_exact": is_exact,
            "sample_tsfresh": float(val_tsfresh[valid_mask][0]),
            "sample_numba": float(val_numba[valid_mask][0]),
        })

    df_res = pd.DataFrame(records)
    csv_path = output_dir / "parity_results.csv"
    df_res.to_csv(csv_path, index=False)

    # Summarize per feature family
    family_summary = df_res.groupby("family").agg(
        n_features=("feature", "count"),
        exact_match_count=("is_exact", "sum"),
        close_1e5_count=("is_close_1e5", "sum"),
        close_1e6_count=("is_close_1e6", "sum"),
        max_abs_diff_max=("max_abs_diff", "max"),
        max_abs_diff_mean=("max_abs_diff", "mean"),
    ).reset_index()

    summary_path = output_dir / "parity_family_summary.csv"
    family_summary.to_csv(summary_path, index=False)

    logger.info("=== Numerical Parity Summary per Family ===")
    logger.info("\n%s", family_summary.to_string(index=False))
    logger.info("Parity evaluation complete. Outputs saved to %s", output_dir)


if __name__ == "__main__":
    out = Path("benchmark_results/TM26a18__empirical_mechanisms/01_numerical_parity")
    run_parity_evaluation(out, n_sequences=50)
