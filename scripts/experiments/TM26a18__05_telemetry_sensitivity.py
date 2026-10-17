#!/usr/bin/env python3
"""
Test 5: Telemetry Sampling-Rate Sensitivity Benchmark.

Evaluates ResourceTracker sampling frequencies at 10 Hz, 20 Hz, 100 Hz, and 200 Hz
against the true OS kernel peak resident memory (ru_maxrss) for both polars_statistics
and numba_efficient on AI4I sequence data.

Quantifies the empirical sampling attenuation: how low polling rates (10-20 Hz)
systematically miss short transient memory allocation spikes compared to high-frequency
polling (100-200 Hz) and the OS kernel high-water mark.

Output:
  benchmark_results/TM26a18__empirical_mechanisms/05_telemetry_sensitivity/telemetry_sensitivity.csv
  benchmark_results/TM26a18__empirical_mechanisms/05_telemetry_sensitivity/sensitivity_summary.csv
"""

import logging
import multiprocessing as mp
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import polars as pl

# Ensure src is on python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tempo.extraction import numba_efficient_extractor, polars_statistical_extractor
from tempo.storage import load_dataset, to_numpy_tensor
from tempo.telemetry import ResourceTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tempo.telemetry_sensitivity")


def _worker_sensitivity(ext_name: str, freq_hz: int, interval_s: float, repeat: int, pipe: Any) -> None:
    """Isolated subprocess evaluating monitor sampling vs OS kernel ru_maxrss."""
    try:
        df_ts, _ = load_dataset("pred-maintenance")
        exclude_cols = {"sequence_id", "id", "step", "time"}
        feature_cols = [c for c in df_ts.columns if c not in exclude_cols]

        tracker = ResourceTracker(interval=interval_s)

        if ext_name == "polars_statistics":
            with tracker:
                _ = polars_statistical_extractor(df_ts)
        elif ext_name == "numba_efficient":
            tensor = to_numpy_tensor(df_ts, feature_cols=feature_cols)
            if tensor.ndim == 2:
                tensor = tensor[:, :, np.newaxis]
            with tracker:
                _ = numba_efficient_extractor(tensor, raw_feature_names=feature_cols, n_fft_coeffs=25)
        else:
            raise ValueError(f"Unknown extractor: {ext_name}")

        stats = tracker.stats
        sampled_peak_mb = stats.peak_ram_mb if stats else 0.0
        duration_s = stats.duration_seconds if stats else 0.0

        usage = resource.getrusage(resource.RUSAGE_SELF)
        raw_rss = usage.ru_maxrss
        os_peak_mb = (raw_rss / (1024.0 * 1024.0)) if platform.system() == "Darwin" else (raw_rss / 1024.0)

        missed_peak_mb = max(0.0, os_peak_mb - sampled_peak_mb)
        attenuation_pct = (missed_peak_mb / os_peak_mb) * 100.0 if os_peak_mb > 0 else 0.0

        pipe.send({
            "status": "success",
            "extractor": ext_name,
            "frequency_hz": freq_hz,
            "interval_s": interval_s,
            "repeat": repeat,
            "duration_s": duration_s,
            "sampled_peak_mb": round(sampled_peak_mb, 2),
            "os_kernel_peak_mb": round(os_peak_mb, 2),
            "missed_peak_mb": round(missed_peak_mb, 2),
            "attenuation_pct": round(attenuation_pct, 2),
        })
    except Exception as e:
        pipe.send({"status": "error", "error": str(e), "extractor": ext_name, "frequency_hz": freq_hz})


def run_sensitivity_sweep(output_dir: Path, n_repeats: int = 5) -> None:
    """Execute sensitivity sweep across sampling frequencies."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frequencies = [
        (10, 0.100),
        (20, 0.050),
        (100, 0.010),
        (200, 0.005),
    ]

    extractors = ["polars_statistics", "numba_efficient"]
    records = []

    for ext in extractors:
        for freq_hz, interval_s in frequencies:
            for rep in range(n_repeats):
                parent_conn, child_conn = mp.Pipe()
                p = mp.Process(
                    target=_worker_sensitivity,
                    args=(ext, freq_hz, interval_s, rep, child_conn),
                )
                p.start()
                res = parent_conn.recv()
                p.join()

                if res.get("status") == "success":
                    records.append(res)
                    logger.info(
                        "[%s | %d Hz (rep %d)] Sampled Peak=%.2f MB | OS True Peak=%.2f MB | Missed=%.2f MB (%.1f%%)",
                        ext, freq_hz, rep, res["sampled_peak_mb"], res["os_kernel_peak_mb"],
                        res["missed_peak_mb"], res["attenuation_pct"],
                    )
                else:
                    logger.error("Failed on %s at %d Hz: %s", ext, freq_hz, res.get("error"))

    df_sens = pd.DataFrame(records)
    csv_path = output_dir / "telemetry_sensitivity.csv"
    df_sens.to_csv(csv_path, index=False)

    summary = df_sens.groupby(["extractor", "frequency_hz"]).agg(
        sampled_peak_mean=("sampled_peak_mb", "mean"),
        os_peak_mean=("os_kernel_peak_mb", "mean"),
        missed_peak_mean=("missed_peak_mb", "mean"),
        attenuation_pct_mean=("attenuation_pct", "mean"),
        duration_mean_s=("duration_s", "mean"),
    ).reset_index()

    summary_csv = output_dir / "sensitivity_summary.csv"
    summary.to_csv(summary_csv, index=False)

    logger.info("=== Telemetry Sampling Sensitivity Summary ===")
    logger.info("\n%s", summary.to_string(index=False))
    logger.info("Sensitivity evaluation complete. Saved to %s", output_dir)


if __name__ == "__main__":
    out = Path("benchmark_results/TM26a18__empirical_mechanisms/05_telemetry_sensitivity")
    run_sensitivity_sweep(out, n_repeats=5)
