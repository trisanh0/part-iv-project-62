#!/usr/bin/env python3
"""
TM26a19 Task 7: Telemetry Sampling-Rate Sensitivity & Overhead Benchmark.

1. Workload with a large, short-lived allocation spike:
   Uses numba_efficient_extractor at N = 1000 windows (W = 100, 3 physical channels).
   This workload executes rapidly (~0.5s warm) while generating transient spectral
   matrices and feature blocks, creating a brief allocation spike.

2. Peak Memory Comparison:
   Compares sampled peak RAM from ResourceTracker at 10, 20, 100, and 200 Hz against
   the true OS kernel high-water mark (ru_maxrss) across 5 independent repeats each.
   Runs each repeat in a fresh child subprocess to prevent high-water mark contamination.

3. Monitor Overhead Evaluation (Excluding JIT):
   Separately evaluates execution wall-clock time with vs without the monitor at each
   sampling frequency (10, 20, 100, 200 Hz) across 10 repeats.
   Excludes JIT compilation by warming up the kernel prior to timed execution.
   Reports median, IQR, and percentage runtime overhead.

Output Directory:
  benchmark_results/TM26a19__followups/TM26a19__07_telemetry_sensitivity/
"""

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
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import polars as pl

# Ensure src is on python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tempo.extraction import numba_efficient_extractor
from tempo.telemetry import ResourceTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tempo.tm26a19_telemetry")


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "log", "-n", "1", "--format=%H"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


def _worker_sampling_peak(
    freq_hz: int,
    interval_s: float,
    n_windows: int,
    window_len: int,
    n_channels: int,
    pipe: Any,
) -> None:
    """Subprocess worker: measures sampled peak RAM vs OS ru_maxrss under transient workload."""
    try:
        rng = np.random.default_rng(42)
        tensor = rng.normal(size=(n_windows, window_len, n_channels)).astype(np.float64)
        raw_names = [f"channel_{c}" for c in range(n_channels)]

        # Pre-warm JIT kernel inside process so JIT compilation is excluded from peak memory
        dummy = tensor[:2]
        _ = numba_efficient_extractor(dummy, raw_feature_names=raw_names, n_fft_coeffs=25)

        # Baseline ru_maxrss after warmup
        base_usage = resource.getrusage(resource.RUSAGE_SELF)
        base_raw = base_usage.ru_maxrss
        base_mb = (base_raw / (1024.0 * 1024.0)) if platform.system() == "Darwin" else (base_raw / 1024.0)

        # Profile with ResourceTracker
        tracker = ResourceTracker(interval=interval_s)
        t0 = time.perf_counter()
        with tracker:
            _ = numba_efficient_extractor(tensor, raw_feature_names=raw_names, n_fft_coeffs=25)
        t1 = time.perf_counter()
        duration_s = t1 - t0

        stats = tracker.stats
        sampled_peak_mb = stats.peak_ram_mb if stats else 0.0

        # OS kernel true peak
        final_usage = resource.getrusage(resource.RUSAGE_SELF)
        final_raw = final_usage.ru_maxrss
        os_peak_mb = (final_raw / (1024.0 * 1024.0)) if platform.system() == "Darwin" else (final_raw / 1024.0)

        missed_peak_mb = max(0.0, os_peak_mb - sampled_peak_mb)
        attenuation_pct = (missed_peak_mb / os_peak_mb) * 100.0 if os_peak_mb > 0 else 0.0

        pipe.send({
            "status": "success",
            "frequency_hz": freq_hz,
            "interval_s": interval_s,
            "duration_s": duration_s,
            "base_rss_mb": base_mb,
            "sampled_peak_mb": sampled_peak_mb,
            "os_kernel_peak_mb": os_peak_mb,
            "missed_peak_mb": missed_peak_mb,
            "attenuation_pct": attenuation_pct,
        })
    except Exception as e:
        pipe.send({
            "status": "error",
            "error": str(e),
            "frequency_hz": freq_hz,
        })


def _worker_overhead_run(
    with_monitor: bool,
    interval_s: float,
    n_windows: int,
    window_len: int,
    n_channels: int,
    pipe: Any,
) -> None:
    """Subprocess worker: measures wall-clock execution time with or without monitor."""
    try:
        rng = np.random.default_rng(999)
        tensor = rng.normal(size=(n_windows, window_len, n_channels)).astype(np.float64)
        raw_names = [f"channel_{c}" for c in range(n_channels)]

        # Warmup JIT
        dummy = tensor[:2]
        _ = numba_efficient_extractor(dummy, raw_feature_names=raw_names, n_fft_coeffs=25)

        # Timed execution
        if with_monitor:
            tracker = ResourceTracker(interval=interval_s)
            t0 = time.perf_counter()
            with tracker:
                _ = numba_efficient_extractor(tensor, raw_feature_names=raw_names, n_fft_coeffs=25)
            t1 = time.perf_counter()
        else:
            t0 = time.perf_counter()
            _ = numba_efficient_extractor(tensor, raw_feature_names=raw_names, n_fft_coeffs=25)
            t1 = time.perf_counter()

        pipe.send({
            "status": "success",
            "duration_s": t1 - t0,
        })
    except Exception as e:
        pipe.send({
            "status": "error",
            "error": str(e),
        })


def run_telemetry_sensitivity(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    git_hash = get_git_commit()

    env_info = {
        "git_commit": git_hash,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "workload": "numba_efficient_extractor (N=1000, W=100, P=3)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(output_dir / "environment.json", "w") as f:
        json.dump(env_info, f, indent=2)

    n_windows = 1000
    window_len = 100
    n_channels = 3
    frequencies = [
        (10, 0.100),
        (20, 0.050),
        (100, 0.010),
        (200, 0.005),
    ]

    # Part 1: Sampling Peak vs ru_maxrss (5 repeats per frequency)
    logger.info("Part 1: Evaluating Sampling Peak vs ru_maxrss (5 repeats each)...")
    sampling_records = []
    for freq_hz, interval_s in frequencies:
        for rep in range(5):
            logger.info("Sampling check: %d Hz (interval=%.3fs) repeat %d/5...", freq_hz, interval_s, rep + 1)
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(
                target=_worker_sampling_peak,
                args=(freq_hz, interval_s, n_windows, window_len, n_channels, child_conn),
            )
            p.start()
            res = parent_conn.recv()
            p.join()

            if res.get("status") == "success":
                sampling_records.append({
                    "frequency_hz": freq_hz,
                    "interval_s": interval_s,
                    "repeat": rep + 1,
                    "duration_s": round(res["duration_s"], 4),
                    "base_rss_mb": round(res["base_rss_mb"], 2),
                    "sampled_peak_mb": round(res["sampled_peak_mb"], 2),
                    "os_kernel_peak_mb": round(res["os_kernel_peak_mb"], 2),
                    "missed_peak_mb": round(res["missed_peak_mb"], 2),
                    "attenuation_pct": round(res["attenuation_pct"], 2),
                })
            else:
                logger.error("Error at %d Hz rep %d: %s", freq_hz, rep + 1, res.get("error"))

    df_sampling_raw = pd.DataFrame(sampling_records)
    df_sampling_raw.to_csv(output_dir / "telemetry_sampling_raw_repeats.csv", index=False)

    df_sampling_summary = df_sampling_raw.groupby(["frequency_hz", "interval_s"]).agg(
        n_repeats=("repeat", "count"),
        sampled_peak_median_mb=("sampled_peak_mb", "median"),
        os_peak_median_mb=("os_kernel_peak_mb", "median"),
        missed_peak_median_mb=("missed_peak_mb", "median"),
        attenuation_pct_median=("attenuation_pct", "median"),
        duration_median_s=("duration_s", "median"),
    ).reset_index()
    df_sampling_summary.to_csv(output_dir / "telemetry_sampling_summary.csv", index=False)

    logger.info("=== Part 1: Telemetry Sampling Summary ===")
    logger.info("\n%s", df_sampling_summary.to_string(index=False))

    # Part 2: Wall-Clock Monitor Overhead (10 repeats each, excluding JIT)
    logger.info("Part 2: Measuring Wall-Clock Monitor Overhead (10 repeats each)...")

    # Baseline: without monitor
    baseline_durations = []
    for rep in range(10):
        parent_conn, child_conn = mp.Pipe()
        p = mp.Process(
            target=_worker_overhead_run,
            args=(False, 0.050, n_windows, window_len, n_channels, child_conn),
        )
        p.start()
        res = parent_conn.recv()
        p.join()
        if res.get("status") == "success":
            baseline_durations.append(res["duration_s"])

    baseline_median = float(np.median(baseline_durations))
    baseline_iqr = float(np.percentile(baseline_durations, 75) - np.percentile(baseline_durations, 25))
    logger.info("Baseline (no monitor): median=%.4fs, IQR=%.4fs across 10 repeats", baseline_median, baseline_iqr)

    overhead_rows = []
    overhead_raw_records = []

    for d in baseline_durations:
        overhead_raw_records.append({
            "mode": "without_monitor",
            "frequency_hz": 0,
            "interval_s": 0.0,
            "duration_s": d,
        })

    for freq_hz, interval_s in frequencies:
        durations = []
        for rep in range(10):
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(
                target=_worker_overhead_run,
                args=(True, interval_s, n_windows, window_len, n_channels, child_conn),
            )
            p.start()
            res = parent_conn.recv()
            p.join()
            if res.get("status") == "success":
                durations.append(res["duration_s"])
                overhead_raw_records.append({
                    "mode": f"with_monitor_{freq_hz}hz",
                    "frequency_hz": freq_hz,
                    "interval_s": interval_s,
                    "duration_s": res["duration_s"],
                })

        med = float(np.median(durations))
        iqr = float(np.percentile(durations, 75) - np.percentile(durations, 25))
        delta_s = med - baseline_median
        overhead_pct = (delta_s / baseline_median) * 100.0 if baseline_median > 0 else 0.0

        overhead_rows.append({
            "frequency_hz": freq_hz,
            "interval_s": interval_s,
            "n_repeats": len(durations),
            "median_duration_s": round(med, 4),
            "iqr_duration_s": round(iqr, 4),
            "baseline_median_s": round(baseline_median, 4),
            "baseline_iqr_s": round(baseline_iqr, 4),
            "net_overhead_s": round(delta_s, 4),
            "overhead_pct": round(overhead_pct, 2),
        })

    df_overhead_raw = pd.DataFrame(overhead_raw_records)
    df_overhead_raw.to_csv(output_dir / "telemetry_overhead_raw_repeats.csv", index=False)

    df_overhead = pd.DataFrame(overhead_rows)
    df_overhead.to_csv(output_dir / "telemetry_overhead_summary.csv", index=False)

    logger.info("=== Part 2: Wall-Clock Monitor Overhead Summary ===")
    logger.info("\n%s", df_overhead.to_string(index=False))
    logger.info("Task 7 completed successfully. Outputs saved to %s", output_dir)


if __name__ == "__main__":
    out_dir = Path("benchmark_results/TM26a19__followups/TM26a19__07_telemetry_sensitivity")
    run_telemetry_sensitivity(out_dir)
