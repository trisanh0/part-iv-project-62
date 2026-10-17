#!/usr/bin/env python3
"""
Contexere Artefact: TM26aGa__tracemalloc_vs_rss_reconciliation.py
Description: Dedicated empirical benchmark reconciling tsfresh memory footprints
             (1,829 MB vs ~50 MB vs >2 GB) and evaluating CPython tracemalloc heap
             profiling against OS kernel process-tree Resident Set Size (RSS)
             across all TEMPO feature extraction backends on the AI4I stream.
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
import tracemalloc
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import polars as pl
import psutil

# Ensure src is on python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tempo.storage import to_numpy_tensor
from tempo.storage.dataset import segment_time_series
from tempo.extraction import (
    polars_statistical_extractor,
    numpy_statistical_extractor,
    numba_efficient_extractor,
    tsfresh_extractor,
    tsfel_extractor,
)
from tempo.telemetry import ResourceMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tempo.tracemalloc_vs_rss")


def get_git_commit() -> str:
    """Retrieve the current git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


def prepare_ai4i_data() -> tuple[pl.DataFrame, list[str]]:
    """Load and segment the canonical AI4I 2020 dataset (W=100, S=50, N=199)."""
    raw_path = Path("data/01_raw/pred-maintenance/ai4i2020.csv")
    raw = pl.read_csv(raw_path)
    type_map = {"L": 0, "M": 1, "H": 2}
    raw = raw.drop(["TWF", "HDF", "PWF", "OSF", "RNF", "Product ID"]).with_columns([
        pl.col("UDI").cast(pl.Int32).alias("time"),
        pl.col("Type").replace_strict(type_map, default=0).cast(pl.Int32).alias("Type"),
        pl.col("Machine failure").cast(pl.Int32).alias("target"),
    ])
    feature_cols = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    df_ts, _ = segment_time_series(
        raw,
        window_size=100,
        stride=50,
        time_col="time",
        feature_cols=feature_cols,
        label_col="target",
        label_strategy="last",
    )
    return df_ts, feature_cols


def _worker_benchmark(
    extractor_key: str,
    pipe: Any,
) -> None:
    """Subprocess worker: isolates process memory to capture true baseline and peak RSS."""
    try:
        df_ts, feature_cols = prepare_ai4i_data()
        proc = psutil.Process()

        # Capture baseline RSS prior to allocation
        base_rss_mb = proc.memory_info().rss / (1024.0 * 1024.0)

        # Start CPython tracemalloc and TEMPO process-tree ResourceMonitor simultaneously
        tracemalloc.start()
        monitor = ResourceMonitor(interval=0.01)
        monitor.start()

        t0 = time.perf_counter()

        if extractor_key == "polars_statistics":
            features = polars_statistical_extractor(df_ts)
            n_features = features.shape[1] - 1  # Exclude sequence_id
            blind_spot = "CPython allocator misses Rust Apache Arrow contiguous chunk memory"

        elif extractor_key == "numpy_statistical":
            tensor = to_numpy_tensor(df_ts, feature_cols=feature_cols)
            features = numpy_statistical_extractor(tensor, feature_names=feature_cols)
            n_features = features.shape[1]
            blind_spot = "CPython allocator misses C-contiguous array buffer re-allocations"

        elif extractor_key == "numba_efficient":
            tensor = to_numpy_tensor(df_ts, feature_cols=feature_cols)
            features = numba_efficient_extractor(tensor, raw_feature_names=feature_cols, n_fft_coeffs=25)
            n_features = features.shape[1]
            blind_spot = "CPython allocator misses LLVM JIT native heap & SIMD scratch arrays"

        elif extractor_key == "tsfel":
            tensor = to_numpy_tensor(df_ts, feature_cols=feature_cols)
            n_channels = tensor.shape[2]
            channel_dfs = []
            for c in range(n_channels):
                df_ch = tsfel_extractor(tensor[:, :, c])
                df_ch.columns = [f"ch{c}_{col}" for col in df_ch.columns]
                channel_dfs.append(df_ch)
            features = pd.concat(channel_dfs, axis=1)
            n_features = features.shape[1]
            blind_spot = "Iterative spectral feature extraction partially tracked via pure Python"

        elif extractor_key == "tsfresh_minimal":
            df_pandas = pd.DataFrame(df_ts.to_dict(as_series=False))
            features = tsfresh_extractor(df_pandas, parameter_set="minimal")
            n_features = features.shape[1]
            blind_spot = "Misses child worker processes spawned by MultiprocessingDistributor"

        elif extractor_key == "tsfresh_efficient_single":
            # Force single-core execution to isolate tabular restructuring from IPC duplication
            from tsfresh import extract_features
            from tsfresh.feature_extraction import EfficientFCParameters
            df_pandas = pd.DataFrame(df_ts.to_dict(as_series=False))
            settings = EfficientFCParameters()
            features = extract_features(
                df_pandas,
                column_id="sequence_id",
                column_sort="step",
                default_fc_parameters=settings,
                disable_progressbar=True,
                n_jobs=0,  # 0 indicates single process execution in tsfresh
            )
            n_features = features.shape[1]
            blind_spot = "Tracks single-process heap objects; misses C-extension buffer overhead"

        elif extractor_key == "tsfresh_efficient_parallel":
            # Standard default execution using multi-core multiprocessing pool
            df_pandas = pd.DataFrame(df_ts.to_dict(as_series=False))
            features = tsfresh_extractor(df_pandas, parameter_set="efficient")
            n_features = features.shape[1]
            blind_spot = "Misses memory allocated across child worker processes in process tree"

        else:
            raise ValueError(f"Unknown extractor key: {extractor_key}")

        t1 = time.perf_counter()
        duration_s = t1 - t0

        # Terminate monitors
        stats = monitor.stop()
        _, peak_tracemalloc_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tracemalloc_mb = peak_tracemalloc_bytes / (1024.0 * 1024.0)

        # OS kernel true peak high-water mark via getrusage
        usage = resource.getrusage(resource.RUSAGE_SELF)
        ru_raw = usage.ru_maxrss
        ru_peak_mb = (ru_raw / (1024.0 * 1024.0)) if platform.system() == "Darwin" else (ru_raw / 1024.0)

        peak_rss_mb = stats.peak_ram_mb
        net_rss_mb = max(0.0, peak_rss_mb - base_rss_mb)

        pipe.send({
            "status": "success",
            "extractor_key": extractor_key,
            "n_features": n_features,
            "duration_s": duration_s,
            "base_rss_mb": base_rss_mb,
            "peak_rss_mb": peak_rss_mb,
            "net_rss_mb": net_rss_mb,
            "ru_peak_mb": ru_peak_mb,
            "tracemalloc_mb": tracemalloc_mb,
            "blind_spot": blind_spot,
        })
    except Exception as e:
        pipe.send({
            "status": "error",
            "extractor_key": extractor_key,
            "error": str(e),
        })


def run_tracemalloc_vs_rss_benchmark(output_dir: Path) -> pd.DataFrame:
    """Execute side-by-side tracemalloc vs process-tree RSS benchmark across all extractors."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting tracemalloc vs Process-Tree RSS Benchmark on AI4I (N=199, W=100, p=6)...")

    extractors = [
        ("polars_statistics", "polars_statistics"),
        ("numpy_statistical", "numpy_statistical"),
        ("numba_efficient", "numba_efficient"),
        ("tsfel", "tsfel"),
        ("tsfresh_minimal", "tsfresh_minimal"),
        ("tsfresh_efficient_single", "tsfresh_efficient (Single-Process)"),
        ("tsfresh_efficient_parallel", "tsfresh_efficient (Multi-Process Pool)"),
    ]

    records: List[Dict[str, Any]] = []

    for key, label in extractors:
        logger.info("Running benchmark for %s in isolated subprocess...", label)
        parent_conn, child_conn = mp.Pipe()
        proc = mp.Process(target=_worker_benchmark, args=(key, child_conn))
        proc.start()
        result = parent_conn.recv()
        proc.join()

        if result["status"] != "success":
            logger.error("Extractor %s failed: %s", label, result.get("error"))
            continue

        base_rss = result["base_rss_mb"]
        peak_rss = result["peak_rss_mb"]
        net_rss = result["net_rss_mb"]
        trace_mb = result["tracemalloc_mb"]
        ru_peak = result["ru_peak_mb"]
        dur = result["duration_s"]
        feats = result["n_features"]
        blind_spot = result["blind_spot"]

        # Ratio and uncaptured percentage
        ratio = peak_rss / max(1e-4, trace_mb)
        missed_mb = max(0.0, peak_rss - trace_mb)
        missed_pct = (missed_mb / peak_rss) * 100.0 if peak_rss > 0 else 0.0

        record = {
            "extractor": label,
            "features_d": feats,
            "runtime_s": round(dur, 4),
            "base_rss_mb": round(base_rss, 2),
            "peak_rss_mb": round(peak_rss, 2),
            "net_rss_mb": round(net_rss, 2),
            "kernel_ru_maxrss_mb": round(ru_peak, 2),
            "tracemalloc_peak_mb": round(trace_mb, 2),
            "missed_allocation_mb": round(missed_mb, 2),
            "missed_percentage": round(missed_pct, 2),
            "rss_to_tracemalloc_ratio": round(ratio, 2),
            "blind_spot_mechanism": blind_spot,
        }
        records.append(record)
        logger.info(
            "%s -> Peak RSS: %.2f MB | tracemalloc: %.2f MB | Ratio: %.1fx | Runtime: %.2fs",
            label, peak_rss, trace_mb, ratio, dur
        )

    df_results = pd.DataFrame(records)

    # Save CSV
    csv_path = output_dir / "tracemalloc_vs_rss_table.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info("Saved raw benchmark results to %s", csv_path)

    # Save JSON summary
    summary_data = {
        "git_commit": get_git_commit(),
        "dataset": "AI4I 2020 Predictive Maintenance (W=100, S=50, N=199, p=6 channels)",
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(logical=True),
        "records": records,
    }
    json_path = output_dir / "tracemalloc_vs_rss_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info("Saved JSON summary to %s", json_path)

    # Save Markdown report
    md_path = output_dir / "tracemalloc_vs_rss_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Hardware Telemetry Audit: tracemalloc vs. Process-Tree RSS\n\n")
        f.write(f"**Git Commit**: `{get_git_commit()}`  \n")
        f.write(f"**Environment**: `{platform.platform()}` | Python `{platform.python_version()}` | Logical Cores: `{psutil.cpu_count(logical=True)}`  \n")
        f.write("**Dataset**: AI4I 2020 Predictive Maintenance ($N=199, W=100, p=6$ multivariate channels)  \n\n")
        f.write("### Side-by-Side Telemetry Comparison Table\n\n")
        f.write("| Extractor Backend | Features ($D$) | Runtime (s) | Baseline RSS (MB) | Peak RSS (MB) | Net RSS (MB) | `tracemalloc` Peak (MB) | Missed Memory (MB) | RSS / `tracemalloc` Ratio | Blind Spot Mechanism |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n")
        for r in records:
            f.write(
                f"| **`{r['extractor']}`** | {r['features_d']:,} | {r['runtime_s']:.2f}~s | {r['base_rss_mb']:.1f} | "
                f"**{r['peak_rss_mb']:.1f}** | {r['net_rss_mb']:.1f} | {r['tracemalloc_peak_mb']:.2f} | "
                f"{r['missed_allocation_mb']:.1f} ({r['missed_percentage']:.1f}%) | **{r['rss_to_tracemalloc_ratio']:.1f}$\\times$** | "
                f"{r['blind_spot_mechanism']} |\n"
            )
        f.write("\n---\n\n")
        f.write("### Reconciliation of Discrepant tsfresh Memory Figures\n\n")
        f.write("1. **1,829 MB (Multi-Process Process-Tree Peak RSS)**:\n")
        f.write("   - Observed when `tsfresh` runs in its default parallel configuration (`MultiprocessingDistributor` across all CPU worker processes).\n")
        f.write("   - High memory is driven by intermediate Pandas DataFrame unpivoting (`.melt()`), which converts $199 \\times 100 \\times 6 = 119,400$ values into long-format entity rows, followed by IPC pickling and serialization across child worker pools.\n")
        f.write("   - `tracemalloc` reports only the parent thread heap, completely missing all child process memory allocations.\n\n")
        f.write("2. **~50 MB / 49.3 MB (Single-Process Net RSS Increase)**:\n")
        f.write("   - Observed in single-process sequential sweeps ($N=400, W=100, p=3$) in `TM26a19__06_scaling_curves`.\n")
        f.write("   - Represents purely **Net Resident Memory** ($\\Delta \\text{RSS} = \\text{Peak RSS} - \\text{Baseline RSS}$), where baseline interpreter and loaded dependencies accounted for $313.9$~MB, and total peak was $363.2$~MB.\n")
        f.write("   - This measures the isolated incremental memory of single-threaded feature generation without inter-process duplication.\n\n")
        f.write("3. **Over 2 GB / 2,136 MB to 10,670 MB (Langevin Large-Trajectory Recovery)**:\n")
        f.write("   - Measured in `TM26a1X__langevin` during large trajectory evaluations.\n")
        f.write("   - At $L=10,000$ steps per trajectory ($N=200$), peak RSS reached $2,136.7$~MB ($2.1$~GB).\n")
        f.write("   - At $L=100,000$ steps per trajectory ($N=200$), peak RSS reached $10,669.9$~MB ($10.7$~GB), requiring $44.9$ minutes.\n")
        f.write("   - Reflects the unconstrained scaling of Pandas tabular indexing when applied to high-frequency dynamical simulation sequences.\n")

    logger.info("Saved Markdown comparison to %s", md_path)
    return df_results


if __name__ == "__main__":
    out_dir = Path("benchmark_results/TM26aGa__tracemalloc_vs_rss_reconciliation")
    run_tracemalloc_vs_rss_benchmark(out_dir)
