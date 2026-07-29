"""TEMPO Telemetry Benchmark Suite: Pandas TSFresh vs CPython NumPy vs Numba JIT.

Compares feature extraction execution wall-clock time and peak memory footprint across
scaling row counts (N = 20, 50, 100, 200) using in-process tracemalloc tracking.
"""

import gc
import sys
import time
import tracemalloc
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from tempo.benchmark_efficient import (
    load_dataset,
    benchmark_pandas_tsfresh_efficient,
    benchmark_numpy_efficient,
)
from tempo.extraction.numba_engine import numba_feature_extractor

warnings.filterwarnings("ignore")


def benchmark_numba_engine(
    data: np.ndarray,
    raw_feature_names: List[str],
    n_fft_coeffs: int = 25,
) -> Tuple[np.ndarray, List[str]]:
    """Alternative: Numba JIT compiled extractor."""
    df_feat = numba_feature_extractor(data, raw_feature_names=raw_feature_names, n_fft_coeffs=n_fft_coeffs)
    return df_feat.to_numpy(dtype=np.float64), list(df_feat.columns)


def run_in_process(
    method_name: str,
    data: np.ndarray,
    feature_names: List[str],
    max_timeshift: int = 4,
) -> Tuple[np.ndarray, float, float]:
    """Execute target extraction method in-process with tracemalloc telemetry."""
    gc.collect()
    tracemalloc.start()

    start_time = time.perf_counter()

    if method_name == "pandas":
        res_arr, _ = benchmark_pandas_tsfresh_efficient(data, feature_names, max_timeshift)
    elif method_name == "numpy_loop":
        res_arr, _ = benchmark_numpy_efficient(data, max_timeshift, feature_names)
    elif method_name == "numba_jit":
        res_arr, _ = benchmark_numba_engine(data, feature_names, n_fft_coeffs=25)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    elapsed_time = time.perf_counter() - start_time
    _, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mem_mb = peak_mem_bytes / (1024.0 * 1024.0)
    return res_arr, elapsed_time, peak_mem_mb


def run_numba_benchmarks():
    """Execute telemetry benchmark suite comparing Pandas, CPython NumPy loop, and Numba JIT."""
    base_dir = Path.cwd()
    datasets = ["beed", "pred-maintenance"]
    scale_sizes = [20, 50, 100, 200]

    # Warmup Numba JIT compilation so compilation overhead isn't counted in benchmark runs
    dummy = np.random.randn(10, 5)
    _ = numba_feature_extractor(dummy, n_fft_coeffs=5)

    print("=" * 90)
    print("TEMPO FEATURE EXTRACTION TELEMETRY: PANDAS vs CPYTHON LOOP vs NUMBA JIT")
    print("=" * 90)

    for ds_name in datasets:
        print(f"\nTarget Dataset: {ds_name.upper()}")
        print("-" * 90)

        data, feature_names = load_dataset(ds_name, base_dir)
        print(f"Loaded dataset matrix shape: {data.shape[0]} rows x {data.shape[1]} raw variables")

        print("\n| Scale (N) | Method | Time (s) | Peak Memory (MB) | Speedup vs Ref | Memory Drop |")
        print("|:---:|:---:|:---:|:---:|:---:|:---:|")

        for size in scale_sizes:
            if size > len(data):
                continue

            sub_data = data[:size]

            # 1. Pandas Baseline
            _, t_pandas, m_pandas = run_in_process("pandas", sub_data, feature_names)
            print(f"| {size} | Pandas Baseline | {t_pandas:8.4f} s | {m_pandas:7.2f} MB | 1.0x (Ref) | 1.0x (Ref) |")

            # 2. CPython NumPy Loop
            _, t_numpy, m_numpy = run_in_process("numpy_loop", sub_data, feature_names)
            sp_numpy = t_pandas / t_numpy if t_numpy > 0 else 0.0
            mem_numpy = m_pandas / m_numpy if m_numpy > 0 else 0.0
            print(f"| {size} | CPython Loop | {t_numpy:8.4f} s | {m_numpy:7.2f} MB | {sp_numpy:7.2f}x | {mem_numpy:7.2f}x |")

            # 3. Numba JIT Engine
            _, t_numba, m_numba = run_in_process("numba_jit", sub_data, feature_names)
            sp_numba = t_pandas / t_numba if t_numba > 0 else 0.0
            mem_numba = m_pandas / m_numba if m_numba > 0 else 0.0
            print(f"| {size} | Numba JIT Engine | {t_numba:8.4f} s | {m_numba:7.2f} MB | {sp_numba:7.2f}x | {mem_numba:7.2f}x |")
            print("|:---:|:---:|:---:|:---:|:---:|:---:|")

    print("\n" + "=" * 90)
    print("TELEMETRY BENCHMARK COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    run_numba_benchmarks()
