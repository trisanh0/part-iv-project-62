"""
Scale Threshold Sweep Experiment Module for the TEMPO Framework.

Evaluates execution latency, throughput (samples/sec), and peak memory scaling
across a 2D grid of time-series length (N) and sequence count (M) to determine
exact operational crossover thresholds between CPython (tsfresh) and Numba JIT.
"""

import argparse
import datetime
import gc
import logging
import os
from pathlib import Path
import time
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tempo.extraction import numba_efficient_extractor, tsfresh_extractor
from tempo.telemetry import ResourceStats, ResourceTracker

logger = logging.getLogger("tempo.experiments.threshold_sweep")


def generate_synthetic_data(
    n_sequences: int,
    seq_length: int,
    n_channels: int = 1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Generate synthetic time series for benchmarking.
    
    Returns:
        Tuple of (Pandas DataFrame for tsfresh, 3D NumPy array for Numba).
    """
    np.random.seed(seed)
    
    # 3D Tensor: (n_sequences, seq_length, n_channels)
    tensor = np.random.randn(n_sequences, seq_length, n_channels).astype(np.float64)

    # Convert to long-format DataFrame for tsfresh
    records = []
    for seq_id in range(n_sequences):
        for step in range(seq_length):
            row = {"id": seq_id, "time": step}
            for ch in range(n_channels):
                row[f"val_{ch}"] = tensor[seq_id, step, ch]
            records.append(row)

    df_long = pd.DataFrame(records)
    return df_long, tensor


def run_threshold_grid_sweep(
    lengths: Sequence[int] = (50, 100, 250, 500, 1000),
    counts: Sequence[int] = (10, 25, 50, 100),
    fft_coefficients: int = 25,
    output_dir: str = "benchmark_results/threshold_sweep",
) -> pd.DataFrame:
    """Execute 2D parameter grid sweep across series length (N) and sequence count (M)."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []

    print(f"Starting Scale Threshold Grid Sweep ({len(lengths)} lengths x {len(counts)} counts)...")

    for N in lengths:
        for M in counts:
            total_points = M * N
            print(f"--> Evaluating Grid Point: N={N} (length), M={M} (sequences) | Total Points: {total_points:,}")

            df_tsfresh, tensor_numba = generate_synthetic_data(n_sequences=M, seq_length=N)
            feature_names = [f"val_{i}" for i in range(tensor_numba.shape[2])]

            # 1. Benchmark Numba JIT Engine
            gc.collect()
            with ResourceTracker(interval=0.01) as numba_tracker:
                t0 = time.perf_counter()
                _ = numba_efficient_extractor(
                    tensor_numba,
                    raw_feature_names=feature_names,
                    n_fft_coeffs=fft_coefficients,
                )
                numba_time = time.perf_counter() - t0

            numba_stats = numba_tracker.stats

            # 2. Benchmark TSFresh Efficient Engine
            gc.collect()
            with ResourceTracker(interval=0.01) as tsfresh_tracker:
                t0 = time.perf_counter()
                _ = tsfresh_extractor(
                    df_tsfresh,
                    parameter_set="efficient",
                    fft_coefficients=fft_coefficients,
                )
                tsfresh_time = time.perf_counter() - t0

            tsfresh_stats = tsfresh_tracker.stats

            speedup = tsfresh_time / max(1e-6, numba_time)
            mem_ratio = (tsfresh_stats.peak_ram_mb if tsfresh_stats else 1.0) / max(
                1e-3, (numba_stats.peak_ram_mb if numba_stats else 1.0)
            )

            records.append({
                "Seq Length (N)": N,
                "Seq Count (M)": M,
                "Total Data Points": total_points,
                "Numba Time (s)": round(numba_time, 4),
                "TSFresh Time (s)": round(tsfresh_time, 4),
                "Speedup (x)": round(speedup, 2),
                "Numba Peak RAM (MB)": numba_stats.peak_ram_mb if numba_stats else 0.0,
                "TSFresh Peak RAM (MB)": tsfresh_stats.peak_ram_mb if tsfresh_stats else 0.0,
                "Memory Reduction Ratio": round(mem_ratio, 2),
                "Numba Throughput (seq/s)": round(M / max(1e-6, numba_time), 2),
                "TSFresh Throughput (seq/s)": round(M / max(1e-6, tsfresh_time), 2),
            })

    df_sweep = pd.DataFrame(records)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = out_path / f"threshold_sweep_{timestamp}.csv"
    df_sweep.to_csv(csv_file, index=False)
    print(f"\nSaved threshold grid results to: {csv_file}")

    # Plot Speedup Surface / Heatmap
    _plot_crossover_results(df_sweep, out_path)
    return df_sweep


def _plot_crossover_results(df: pd.DataFrame, out_path: Path) -> None:
    """Generate publication-ready 2D heatmap plots of speedup and memory reduction."""
    plt.style.use("default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    pivot_speedup = df.pivot(index="Seq Length (N)", columns="Seq Count (M)", values="Speedup (x)")
    pivot_mem = df.pivot(index="Seq Length (N)", columns="Seq Count (M)", values="Memory Reduction Ratio")

    im1 = ax1.imshow(pivot_speedup.values, cmap="YlGnBu", aspect="auto", origin="lower")
    ax1.set_title("Numba JIT Relative Speedup vs TSFresh (x)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Sequence Count (M)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Time-Series Length (N)", fontsize=11, fontweight="bold")
    ax1.set_xticks(range(len(pivot_speedup.columns)))
    ax1.set_xticklabels(pivot_speedup.columns)
    ax1.set_yticks(range(len(pivot_speedup.index)))
    ax1.set_yticklabels(pivot_speedup.index)
    plt.colorbar(im1, ax=ax1)

    for i in range(len(pivot_speedup.index)):
        for j in range(len(pivot_speedup.columns)):
            val = pivot_speedup.values[i, j]
            ax1.text(j, i, f"{val:.1f}x", ha="center", va="center", color="black", fontweight="bold")

    im2 = ax2.imshow(pivot_mem.values, cmap="OrRd", aspect="auto", origin="lower")
    ax2.set_title("Peak RAM Reduction Ratio (TSFresh / Numba)", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Sequence Count (M)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Time-Series Length (N)", fontsize=11, fontweight="bold")
    ax2.set_xticks(range(len(pivot_mem.columns)))
    ax2.set_xticklabels(pivot_mem.columns)
    ax2.set_yticks(range(len(pivot_mem.index)))
    ax2.set_yticklabels(pivot_mem.index)
    plt.colorbar(im2, ax=ax2)

    for i in range(len(pivot_mem.index)):
        for j in range(len(pivot_mem.columns)):
            val = pivot_mem.values[i, j]
            ax2.text(j, i, f"{val:.1f}x", ha="center", va="center", color="black", fontweight="bold")

    plt.tight_layout()
    plot_file = out_path / "numba_vs_tsfresh_scale_crossover.png"
    plt.savefig(plot_file, dpi=300)
    plt.close(fig)
    print(f"Saved crossover plots to: {plot_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale Threshold Sweep: Numba JIT vs CPython TSFresh")
    parser.add_argument("--lengths", nargs="+", type=int, default=[50, 100, 250, 500], help="List of sequence lengths N")
    parser.add_argument("--counts", nargs="+", type=int, default=[10, 25, 50], help="List of sequence counts M")
    parser.add_argument("--output", type=str, default="benchmark_results/threshold_sweep", help="Output directory")

    args = parser.parse_args()
    run_threshold_grid_sweep(lengths=args.lengths, counts=args.counts, output_dir=args.output)
