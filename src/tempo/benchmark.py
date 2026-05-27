"""
TEMPO Benchmarking Framework: tsfresh vs. NumPy Swap-Out.

This framework systematically benchmarks the feature extraction phase of time-series
pipelines. It evaluates the standard tsfresh implementation (relying on Pandas DataFrames
and roll operations) against alternative Pandas-free approaches utilizing NumPy.

The benchmarking measures execution wall time and peak memory footprint using in-process
tracemalloc to guarantee clean, platform-independent, and sandbox-compatible tracking.
"""

import csv
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.feature_extraction.feature_calculators import (
    absolute_maximum,
    length,
    maximum,
    mean,
    median,
    minimum,
    root_mean_square,
    standard_deviation,
    sum_values,
    variance,
)
import tsfresh.utilities.dataframe_functions as df_funcs


# ==============================================================================
# 0. Pandas 3.0 Compatibility Monkey Patch for tsfresh
# ==============================================================================

def patched_roll_out_time_series(
    timeshift,
    grouped_data,
    rolling_direction,
    max_timeshift,
    min_timeshift,
    column_sort,
    column_id,
):
    """Patched version of _roll_out_time_series supporting Pandas 3.0.
    
    Pandas 3.0 drops grouping columns from the group DataFrames during apply.
    This patch detects the missing ID column and reconstructs it from the group's
    name property.
    """
    def _f(x):
        if rolling_direction > 0:
            shift_until = timeshift
            shift_from = max(shift_until - max_timeshift - 1, 0)
            df_temp = x.iloc[shift_from:shift_until] if shift_until <= len(x) else None
        else:
            shift_from = max(timeshift - 1, 0)
            shift_until = shift_from + max_timeshift + 1
            df_temp = x.iloc[shift_from:shift_until]

        if df_temp is None or len(df_temp) < min_timeshift + 1:
            return

        df_temp = df_temp.copy()

        if column_sort and rolling_direction > 0:
            timeshift_value = df_temp[column_sort].iloc[-1]
        elif column_sort and rolling_direction < 0:
            timeshift_value = df_temp[column_sort].iloc[0]
        else:
            timeshift_value = timeshift - 1

        # Reconstruct missing column_id from group name for Pandas 3.0+ compatibility
        if column_id not in df_temp.columns:
            group_key = x.name
            if isinstance(group_key, tuple):
                val = group_key[-1]
            else:
                val = group_key
            df_temp[column_id] = val

        df_temp["id"] = df_temp[column_id].apply(lambda row: (row, timeshift_value))
        return df_temp

    return [grouped_data.apply(_f)]


# Apply the monkey patch immediately
df_funcs._roll_out_time_series = patched_roll_out_time_series


# ==============================================================================
# 1. Permissive Data Loaders (Pandas-free)
# ==============================================================================

def load_beed(file_path: Path) -> Tuple[np.ndarray, List[str]]:
    """Loads the BEED sensor dataset, bypassing Pandas entirely.
    
    Args:
        file_path: Path to the BEED csv data file.
        
    Returns:
        A tuple of (data_matrix, feature_names), where data_matrix is a 2D float64
        array of shape (N, n_features).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        
        target_col = "y"
        target_idx = header.index(target_col)
        
        # Features are all columns except the target
        feature_indices = [i for i, name in enumerate(header) if i != target_idx]
        feature_names = [header[i] for i in feature_indices]
        
        rows = []
        for row in reader:
            if not row:
                continue
            rows.append([float(row[i]) for i in feature_indices])
            
    return np.array(rows, dtype=np.float64), feature_names


def load_pred_maintenance(file_path: Path) -> Tuple[np.ndarray, List[str]]:
    """Loads the AI4I 2020 Predictive Maintenance dataset, bypassing Pandas entirely.
    
    Args:
        file_path: Path to the ai4i2020 csv data file.
        
    Returns:
        A tuple of (data_matrix, feature_names), where data_matrix is a 2D float64
        array of shape (N, n_features) and type column mapped to float values.
    """
    type_map = {"L": 0.0, "M": 1.0, "H": 2.0}
    exclude_cols = {"Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF", "UDI", "Product ID"}
    
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        
        feature_indices = []
        feature_names = []
        for i, name in enumerate(header):
            if name not in exclude_cols:
                feature_indices.append(i)
                feature_names.append(name)
                
        type_idx = header.index("Type")
        
        rows = []
        for row in reader:
            if not row:
                continue
            row_data = []
            for idx in feature_indices:
                val = row[idx]
                if idx == type_idx:
                    row_data.append(type_map.get(val, 0.0))
                else:
                    row_data.append(float(val))
            rows.append(row_data)
            
    return np.array(rows, dtype=np.float64), feature_names


def load_dataset(name: str, base_dir: Path) -> Tuple[np.ndarray, List[str]]:
    """Loads target dataset by name."""
    if name == "beed":
        return load_beed(base_dir / "data" / "01_raw" / "beed" / "BEED_Data.csv")
    elif name == "pred-maintenance":
        return load_pred_maintenance(base_dir / "data" / "01_raw" / "pred-maintenance" / "ai4i2020.csv")
    else:
        raise ValueError(f"Unknown dataset name: {name}")


# ==============================================================================
# 2. Feature Extraction Methodologies
# ==============================================================================

def benchmark_pandas_tsfresh(data: np.ndarray, feature_names: List[str], max_timeshift: int) -> np.ndarray:
    """Baseline: extract features using the standard Pandas-based tsfresh pipeline."""
    df = pd.DataFrame(data, columns=feature_names)
    df["id"] = 1
    df["time"] = np.arange(len(df))
    
    df_ts = df[["id", "time"] + feature_names]
    
    # Roll using tsfresh built-in utility
    df_rolled = df_funcs.roll_time_series(
        df_ts, column_id="id", column_sort="time",
        max_timeshift=max_timeshift, min_timeshift=0,
        disable_progressbar=True
    )
    
    # Extract features using minimal parameters
    extraction_settings = MinimalFCParameters()
    X_features = extract_features(
        df_rolled, column_id="id", column_sort="time",
        default_fc_parameters=extraction_settings,
        disable_progressbar=True
    )
    
    # Realign index to temporal sequence steps (t) and sort to ensure ordered mapping
    X_features.index = [idx[1] for idx in X_features.index]
    X_features.sort_index(inplace=True)
    
    return X_features.to_numpy(dtype=np.float64)


def benchmark_numpy_calculators(data: np.ndarray, max_timeshift: int) -> np.ndarray:
    """Alternative 1: window iteration, calling tsfresh calculators directly on 1D NumPy slices."""
    N, n_features = data.shape
    n_calculators = 10
    out = np.empty((N, n_features * n_calculators), dtype=np.float64)
    
    # The exact order inside MinimalFCParameters
    calculators = [
        sum_values, median, mean, length, standard_deviation,
        variance, root_mean_square, maximum, absolute_maximum, minimum
    ]
    
    for t in range(N):
        start = max(0, t - max_timeshift)
        end = t + 1
        window = data[start:end, :]
        
        row_features = []
        for j in range(n_features):
            x = window[:, j]
            for calc in calculators:
                row_features.append(calc(x))
                
        out[t, :] = row_features
        
    return out


def benchmark_numpy_vectorized(data: np.ndarray, max_timeshift: int) -> np.ndarray:
    """Alternative 2: loop-free C-level vectorized sliding window reductions using NumPy views."""
    N, n_features = data.shape
    W = max_timeshift + 1
    
    if N < W:
        # Fall back to loop-based version if length is smaller than full window size
        return benchmark_numpy_calculators(data, max_timeshift)
        
    out = np.empty((N, n_features * 10), dtype=np.float64)
    
    # 1. Compute boundary elements (t < W - 1) using standard calculators
    calculators = [
        sum_values, median, mean, length, standard_deviation,
        variance, root_mean_square, maximum, absolute_maximum, minimum
    ]
    for t in range(W - 1):
        window = data[0:t+1, :]
        row_features = []
        for j in range(n_features):
            x = window[:, j]
            for calc in calculators:
                row_features.append(calc(x))
        out[t, :] = row_features
        
    # 2. Compute full-window steps (t >= W - 1) using vectorized axis reductions
    from numpy.lib.stride_tricks import sliding_window_view
    
    # Construct sliding window view of shape (N - W + 1, n_features, W)
    wins = sliding_window_view(data, window_shape=W, axis=0)
    
    out_vec = np.empty((N - W + 1, n_features, 10), dtype=np.float64)
    
    out_vec[:, :, 0] = np.sum(wins, axis=2)
    out_vec[:, :, 1] = np.median(wins, axis=2)
    out_vec[:, :, 2] = np.mean(wins, axis=2)
    out_vec[:, :, 3] = float(W)
    out_vec[:, :, 4] = np.std(wins, axis=2, ddof=0)
    out_vec[:, :, 5] = np.var(wins, axis=2, ddof=0)
    out_vec[:, :, 6] = np.sqrt(np.mean(np.square(wins), axis=2))
    out_vec[:, :, 7] = np.max(wins, axis=2)
    out_vec[:, :, 8] = np.max(np.abs(wins), axis=2)
    out_vec[:, :, 9] = np.min(wins, axis=2)
    
    # Flatten features dimensions: (N - W + 1, n_features, 10) -> (N - W + 1, n_features * 10)
    out[W - 1:, :] = out_vec.reshape(N - W + 1, n_features * 10)
    
    return out


# ==============================================================================
# 3. Telemetry and Measurement
# ==============================================================================

def run_in_process(func_name: str, data: np.ndarray, feature_names: List[str], max_timeshift: int) -> Tuple[np.ndarray, float, float]:
    """Runs target function in-process, using high-precision telemetry."""
    # Force clean garbage collection to establish a reliable baseline
    import gc
    gc.collect()
    
    # Start tracing allocations
    tracemalloc.start()
    
    if func_name == "pandas":
        func = benchmark_pandas_tsfresh
        args = (data, feature_names, max_timeshift)
    elif func_name == "numpy_calc":
        func = benchmark_numpy_calculators
        args = (data, max_timeshift)
    elif func_name == "numpy_vec":
        func = benchmark_numpy_vectorized
        args = (data, max_timeshift)
    else:
        raise ValueError(f"Unknown method name: {func_name}")
        
    start_time = time.perf_counter()
    res_arr = func(*args)
    elapsed_time = time.perf_counter() - start_time
    
    _, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Convert peak memory allocated to Megabytes
    peak_mem_mb = peak_mem_bytes / (1024.0 * 1024.0)
    
    return res_arr, elapsed_time, peak_mem_mb


# ==============================================================================
# 4. Benchmarking Suite
# ==============================================================================

def run_benchmarks():
    """Executes full mathematical validation and scales benchmarking runs."""
    base_dir = Path.cwd()
    max_timeshift = 4
    
    datasets = ["beed", "pred-maintenance"]
    
    print("=" * 80)
    print("TEMPO HIGH-PERFORMANCE TIME-SERIES FEATURE EXTRACTION BENCHMARKS")
    print("=" * 80)
    
    for ds_name in datasets:
        print(f"\nEvaluating Dataset: {ds_name.upper()}")
        print("-" * 50)
        
        # Load dataset
        data, feature_names = load_dataset(ds_name, base_dir)
        print(f"Loaded {data.shape[0]} rows with {data.shape[1]} raw variables.")
        
        # 1. Perform strict mathematical equivalence verification on small slice
        print("\nChecking mathematical equivalence of feature matrices...")
        test_slice_len = 50
        slice_data = data[:test_slice_len]
        
        res_pandas, _, _ = run_in_process("pandas", slice_data, feature_names, max_timeshift)
        res_calc, _, _ = run_in_process("numpy_calc", slice_data, feature_names, max_timeshift)
        res_vec, _, _ = run_in_process("numpy_vec", slice_data, feature_names, max_timeshift)
        
        # Compare dimensions
        print(f" * Dimensions: Pandas={res_pandas.shape}, NumPy Calc={res_calc.shape}, NumPy Vec={res_vec.shape}")
        
        # Validate mathematical parity
        match_calc = np.allclose(res_pandas, res_calc, rtol=1e-5, atol=1e-5, equal_nan=True)
        match_vec = np.allclose(res_pandas, res_vec, rtol=1e-5, atol=1e-5, equal_nan=True)
        
        if match_calc and match_vec:
            print(" * VERIFICATION SUCCESSFUL: NumPy methods match Pandas outputs exactly.")
        else:
            print(" * WARNING: Parity verification mismatch!")
            print(f"   Pandas vs. NumPy Loop Matches: {match_calc}")
            print(f"   Pandas vs. NumPy Vectorized Matches: {match_vec}")
            # Output absolute differences to diagnose if necessary
            diff_calc = np.nanmax(np.abs(res_pandas - res_calc))
            diff_vec = np.nanmax(np.abs(res_pandas - res_vec))
            print(f"   Max absolute delta: Loop={diff_calc:.6f}, Vectorized={diff_vec:.6f}")
            
        # 2. Run scaling benchmarks
        scale_sizes = [100, 1000, 5000, len(data)]
        
        print("\nRunning Scaling Benchmarks...")
        print("| Scale (Rows) | Method | Time (s) | Peak Memory (MB) | Speedup |")
        print("|---|---|---|---|---|")
        
        for size in scale_sizes:
            if size > len(data):
                continue
                
            sub_data = data[:size]
            
            # Run Pandas Baseline
            _, t_pandas, m_pandas = run_in_process("pandas", sub_data, feature_names, max_timeshift)
            print(f"| {size} | Pandas Baseline | {t_pandas:.4f}s | {m_pandas:.2f} MB | 1.0x (Ref) |")
            
            # Run NumPy Loop
            _, t_calc, m_calc = run_in_process("numpy_calc", sub_data, feature_names, max_timeshift)
            speedup_calc = t_pandas / t_calc if t_calc > 0 else 0.0
            print(f"| {size} | NumPy Loop | {t_calc:.4f}s | {m_calc:.2f} MB | {speedup_calc:.1f}x |")
            
            # Run NumPy Vectorized
            _, t_vec, m_vec = run_in_process("numpy_vec", sub_data, feature_names, max_timeshift)
            speedup_vec = t_pandas / t_vec if t_vec > 0 else 0.0
            print(f"| {size} | NumPy Vectorized | {t_vec:.4f}s | {m_vec:.2f} MB | {speedup_vec:.1f}x |")
            print("|---|---|---|---|---|")
            
    print("\n" + "=" * 80)
    print("BENCHMARK EXECUTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmarks()
