#!/usr/bin/env python3
"""
Test 4: Scaling Curves Benchmark (Log-Log Execution Time & Peak RSS vs Window Count).

Measures extraction runtime and isolated OS peak RSS across window counts
N in {50, 100, 200, 400, 800, 1600} for polars_statistics, numba_efficient,
numpy_statistical, tsfel, and tsfresh_efficient (capped at N <= 400).

Each configuration runs in a fresh, isolated child subprocess to guarantee
pristine ru_maxrss measurements without high-water mark inheritance.
Generates log-log scaling curves and estimates empirical asymptotic complexity.

Output:
  benchmark_results/TM26a18__empirical_mechanisms/04_scaling_curves/scaling_benchmarks.csv
  benchmark_results/TM26a18__empirical_mechanisms/04_scaling_curves/scaling_curves.png
"""

import logging
import multiprocessing as mp
import os
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

# Ensure src is on python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tempo.extraction import (
    numba_efficient_extractor,
    numpy_statistical_extractor,
    polars_statistical_extractor,
    tsfel_extractor,
    tsfresh_extractor,
)
from tempo.storage import to_numpy_tensor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tempo.scaling_test")


def _worker_benchmark(extractor_name: str, n_windows: int, series_len: int, pipe: Any) -> None:
    """Subprocess worker executing one extraction run with isolated RSS tracking."""
    try:
        # Measure baseline process RSS before data generation
        base_usage = resource.getrusage(resource.RUSAGE_SELF)
        base_raw = base_usage.ru_maxrss
        base_mb = (base_raw / (1024.0 * 1024.0)) if platform.system() == "Darwin" else (base_raw / 1024.0)

        # Generate synthetic test sequence
        rng = np.random.default_rng(42)
        entities = np.repeat(np.arange(n_windows, dtype=np.int32), series_len)
        times = np.tile(np.arange(series_len, dtype=np.int32), n_windows)
        vals = rng.normal(size=n_windows * series_len).astype(np.float32)
        df_ts = pl.DataFrame({"sequence_id": entities, "step": times, "value": vals})

        feature_cols = ["value"]
        t0 = time.perf_counter()

        if extractor_name == "polars_statistics":
            _ = polars_statistical_extractor(df_ts)
        elif extractor_name == "numpy_statistical":
            tensor = to_numpy_tensor(df_ts, feature_cols=feature_cols)
            _ = numpy_statistical_extractor(tensor, feature_names=feature_cols)
        elif extractor_name == "numba_efficient":
            tensor = to_numpy_tensor(df_ts, feature_cols=feature_cols)
            if tensor.ndim == 2:
                tensor = tensor[:, :, np.newaxis]
            _ = numba_efficient_extractor(tensor, raw_feature_names=feature_cols, n_fft_coeffs=25)
        elif extractor_name == "tsfel":
            tensor = to_numpy_tensor(df_ts, feature_cols=feature_cols)
            _ = tsfel_extractor(tensor)
        elif extractor_name == "tsfresh_efficient":
            df_pandas = pd.DataFrame(df_ts.to_dict(as_series=False))
            _ = tsfresh_extractor(df_pandas, parameter_set="efficient", fft_coefficients=None)
        else:
            raise ValueError(f"Unknown extractor: {extractor_name}")

        duration = time.perf_counter() - t0

        final_usage = resource.getrusage(resource.RUSAGE_SELF)
        final_raw = final_usage.ru_maxrss
        final_mb = (final_raw / (1024.0 * 1024.0)) if platform.system() == "Darwin" else (final_raw / 1024.0)
        net_rss_mb = max(0.0, final_mb - base_mb)

        pipe.send({
            "status": "success",
            "extractor": extractor_name,
            "n_windows": n_windows,
            "duration_s": duration,
            "peak_rss_mb": final_mb,
            "net_rss_mb": net_rss_mb,
        })
    except Exception as e:
        pipe.send({"status": "error", "error": str(e), "extractor": extractor_name, "n_windows": n_windows})


def run_scaling_benchmark(output_dir: Path) -> None:
    """Execute scaling benchmark across window counts N."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n_values = [50, 100, 200, 400, 800, 1600]
    series_len = 200

    extractors = [
        "polars_statistics",
        "numpy_statistical",
        "numba_efficient",
        "tsfel",
        "tsfresh_efficient",
    ]

    records = []

    for n in n_values:
        for ext in extractors:
            # Cap tsfresh at N <= 400
            if ext == "tsfresh_efficient" and n > 400:
                logger.info("Skipping tsfresh_efficient for N=%d (capped at N=400 to prevent overnight bottleneck)", n)
                continue

            logger.info("Evaluating %s at N=%d windows (L=%d)...", ext, n, series_len)
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(target=_worker_benchmark, args=(ext, n, series_len, child_conn))
            p.start()
            res = parent_conn.recv()
            p.join()

            if res.get("status") == "success":
                records.append({
                    "extractor": ext,
                    "n_windows": n,
                    "series_len": series_len,
                    "duration_s": round(res["duration_s"], 4),
                    "peak_rss_mb": round(res["peak_rss_mb"], 2),
                    "net_rss_mb": round(res["net_rss_mb"], 2),
                })
                logger.info(
                    "  %s (N=%d): %.4fs | Peak RSS: %.2f MB | Net RSS: %.2f MB",
                    ext, n, res["duration_s"], res["peak_rss_mb"], res["net_rss_mb"],
                )
            else:
                logger.error("  %s failed at N=%d: %s", ext, n, res.get("error"))

    df_scale = pd.DataFrame(records)
    csv_path = output_dir / "scaling_benchmarks.csv"
    df_scale.to_csv(csv_path, index=False)

    # Generate log-log scaling plot
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        markers = {"polars_statistics": "o", "numpy_statistical": "s", "numba_efficient": "^", "tsfel": "v", "tsfresh_efficient": "D"}
        colors = {"polars_statistics": "#1f77b4", "numpy_statistical": "#2ca02c", "numba_efficient": "#ff7f0e", "tsfel": "#9467bd", "tsfresh_efficient": "#d62728"}

        for ext in extractors:
            sub = df_scale[df_scale["extractor"] == ext]
            if sub.empty:
                continue
            ax1.plot(sub["n_windows"], sub["duration_s"], marker=markers.get(ext, "o"), color=colors.get(ext, "black"), label=ext, linewidth=2)
            ax2.plot(sub["n_windows"], sub["peak_rss_mb"], marker=markers.get(ext, "o"), color=colors.get(ext, "black"), label=ext, linewidth=2)

        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlabel("Number of Windows (N)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Execution Time (seconds)", fontsize=11, fontweight="bold")
        ax1.set_title("Runtime Scaling (Log-Log)", fontsize=12, fontweight="bold")
        ax1.grid(True, which="both", ls="--", alpha=0.5)
        ax1.legend(fontsize=9)

        ax2.set_xscale("log")
        ax2.set_xlabel("Number of Windows (N)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Peak Resident Memory (MB)", fontsize=11, fontweight="bold")
        ax2.set_title("Peak Memory Scaling (Log-Linear)", fontsize=12, fontweight="bold")
        ax2.grid(True, which="both", ls="--", alpha=0.5)
        ax2.legend(fontsize=9)

        plt.tight_layout()
        plot_path = output_dir / "scaling_curves.png"
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        logger.info("Saved scaling figure to %s", plot_path)
    except Exception as e:
        logger.warning("Could not generate scaling plot: %s", e)

    logger.info("Scaling benchmark complete. Output saved to %s", csv_path)


if __name__ == "__main__":
    out = Path("benchmark_results/TM26a18__empirical_mechanisms/04_scaling_curves")
    run_scaling_benchmark(out)
