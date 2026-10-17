#!/usr/bin/env python3
"""
TM26a19 Task 6: Scaling Curves Benchmark (JIT & Cold Startup Separated).

Measures extraction runtime and isolated OS resident memory scaling across window counts
N in {50, 100, 200, 400, 800, 1600} at window size W = 100 for:
  - polars_statistics
  - numpy_statistical
  - numba_efficient
  - tsfel
  - tsfresh_efficient (capped at N <= 400)

Key methodological controls:
1. Each configuration and repeat runs in a completely fresh, isolated child subprocess
   to prevent memory high-water mark inheritance.
2. In each subprocess, calls the extractor twice on identical input:
   - duration_call1_s: First call (includes JIT compilation / engine startup)
   - duration_call2_s: Second call (warm / steady-state runtime)
3. For tsfresh, measures fixed overhead on a trivially small input (N=2, L=10).
4. Window length is strictly W = 100.
5. 3 repeats per point; reports median duration and net memory (peak - baseline).
6. Fits log-log OLS slopes over upper half of N range (N in [200, 1600] for fast engines;
   N in [200, 400] for tsfresh) and reports slopes with R^2.

Output Directory:
  benchmark_results/TM26a19__followups/TM26a19__06_scaling_curves/
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
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

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
from tempo.telemetry import ResourceMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tempo.tm26a19_scaling")


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "log", "-n", "1", "--format=%H"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


def _worker_scaling_benchmark(
    extractor_name: str,
    n_windows: int,
    window_len: int,
    pipe: Any,
) -> None:
    """Isolated subprocess worker: executes call 1 (cold/JIT) and call 2 (warm)."""
    try:
        # Generate synthetic test sequence at W = 100
        rng = np.random.default_rng(12345 + n_windows)
        entities = np.repeat(np.arange(n_windows, dtype=np.int32), window_len)
        times = np.tile(np.arange(window_len, dtype=np.int32), n_windows)
        vals = rng.normal(size=n_windows * window_len).astype(np.float64)
        df_ts = pl.DataFrame({"sequence_id": entities, "step": times, "value": vals})
        feature_cols = ["value"]

        # Pre-format inputs so timing measures strictly feature extraction
        if extractor_name == "polars_statistics":
            input_data = df_ts
        elif extractor_name in ("numpy_statistical", "tsfel"):
            tensor = to_numpy_tensor(df_ts, feature_cols=feature_cols)
            input_data = tensor
        elif extractor_name == "numba_efficient":
            tensor = to_numpy_tensor(df_ts, feature_cols=feature_cols)
            if tensor.ndim == 2:
                tensor = tensor[:, :, np.newaxis]
            input_data = tensor
        elif extractor_name == "tsfresh_efficient":
            df_pandas = pd.DataFrame(df_ts.to_dict(as_series=False))
            input_data = df_pandas
        else:
            raise ValueError(f"Unknown extractor: {extractor_name}")

        def execute_extractor() -> Any:
            if extractor_name == "polars_statistics":
                return polars_statistical_extractor(input_data)
            elif extractor_name == "numpy_statistical":
                return numpy_statistical_extractor(input_data, feature_names=feature_cols)
            elif extractor_name == "numba_efficient":
                return numba_efficient_extractor(input_data, raw_feature_names=feature_cols, n_fft_coeffs=25)
            elif extractor_name == "tsfel":
                return tsfel_extractor(input_data)
            elif extractor_name == "tsfresh_efficient":
                return tsfresh_extractor(input_data, parameter_set="efficient", fft_coefficients=25)

        # Monitor process-tree memory across extraction calls using TEMPO telemetry
        monitor = ResourceMonitor(interval=0.005)
        monitor.start()

        # Call 1: Cold / includes JIT compilation & thread pool startup
        t0 = time.perf_counter()
        _ = execute_extractor()
        t1 = time.perf_counter()
        duration_call1 = t1 - t0

        # Call 2: Warm / steady state on identical input
        t0 = time.perf_counter()
        _ = execute_extractor()
        t1 = time.perf_counter()
        duration_call2 = t1 - t0

        stats = monitor.stop()
        base_mb = stats.start_ram_mb
        final_mb = stats.peak_ram_mb
        net_rss_mb = stats.peak_ram_increase_mb

        pipe.send({
            "status": "success",
            "extractor": extractor_name,
            "n_windows": n_windows,
            "window_len": window_len,
            "duration_call1_s": duration_call1,
            "duration_call2_s": duration_call2,
            "base_rss_mb": base_mb,
            "peak_rss_mb": final_mb,
            "net_rss_mb": net_rss_mb,
        })
    except Exception as e:
        pipe.send({
            "status": "error",
            "error": str(e),
            "extractor": extractor_name,
            "n_windows": n_windows,
        })


def measure_tsfresh_fixed_overhead() -> Dict[str, Any]:
    """Measure fixed startup overhead of tsfresh using a trivially small input (N=2, L=10)."""
    logger.info("Measuring tsfresh fixed startup overhead (N=2, L=10)...")
    records = []
    for rep in range(3):
        parent_conn, child_conn = mp.Pipe()
        p = mp.Process(target=_worker_scaling_benchmark, args=("tsfresh_efficient", 2, 10, child_conn))
        p.start()
        res = parent_conn.recv()
        p.join()
        if res.get("status") == "success":
            records.append({
                "repeat": rep,
                "duration_call1_s": res["duration_call1_s"],
                "duration_call2_s": res["duration_call2_s"],
                "net_rss_mb": res["net_rss_mb"],
            })

    call1_times = [r["duration_call1_s"] for r in records]
    call2_times = [r["duration_call2_s"] for r in records]
    summary = {
        "n_windows": 2,
        "window_len": 10,
        "repeats": records,
        "median_call1_s": float(np.median(call1_times)),
        "median_call2_s": float(np.median(call2_times)),
        "explanation": "Call 1 reflects multiprocessing pool spin-up, worker serialization, and configuration dict parsing. Call 2 reflects warm pool invocation."
    }
    logger.info("tsfresh fixed overhead: call 1 median = %.4fs, call 2 median = %.4fs",
                summary["median_call1_s"], summary["median_call2_s"])
    return summary


def run_scaling_benchmark(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    git_hash = get_git_commit()

    env_info = {
        "git_commit": git_hash,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "polars_version": pl.__version__,
        "window_length_W": 100,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(output_dir / "environment.json", "w") as f:
        json.dump(env_info, f, indent=2)

    # 1. Measure fixed overhead of tsfresh
    tsfresh_overhead = measure_tsfresh_fixed_overhead()
    with open(output_dir / "tsfresh_fixed_overhead.json", "w") as f:
        json.dump(tsfresh_overhead, f, indent=2)

    n_values = [50, 100, 200, 400, 800, 1600]
    window_len = 100  # Strictly W = 100 as specified
    n_repeats = 3

    extractors = [
        "polars_statistics",
        "numpy_statistical",
        "numba_efficient",
        "tsfel",
        "tsfresh_efficient",
    ]

    raw_records = []

    for n in n_values:
        for ext in extractors:
            # Cap tsfresh at N <= 400
            if ext == "tsfresh_efficient" and n > 400:
                logger.info("Skipping tsfresh_efficient for N=%d (capped at N=400 to prevent overnight bottleneck)", n)
                continue

            for rep in range(n_repeats):
                logger.info("Run: %s | N=%d | W=%d | Rep=%d/3...", ext, n, window_len, rep + 1)
                parent_conn, child_conn = mp.Pipe()
                p = mp.Process(target=_worker_scaling_benchmark, args=(ext, n, window_len, child_conn))
                p.start()
                res = parent_conn.recv()
                p.join()

                if res.get("status") == "success":
                    raw_records.append({
                        "extractor": ext,
                        "n_windows": n,
                        "window_len": window_len,
                        "repeat": rep + 1,
                        "duration_call1_s": res["duration_call1_s"],
                        "duration_call2_s": res["duration_call2_s"],
                        "base_rss_mb": res["base_rss_mb"],
                        "peak_rss_mb": res["peak_rss_mb"],
                        "net_rss_mb": res["net_rss_mb"],
                    })
                else:
                    logger.error("  %s failed at N=%d rep=%d: %s", ext, n, rep + 1, res.get("error"))

    df_raw = pd.DataFrame(raw_records)
    df_raw.to_csv(output_dir / "scaling_raw_repeats.csv", index=False)

    # Compute medians per configuration
    summary_rows = []
    for (ext, n), group in df_raw.groupby(["extractor", "n_windows"]):
        summary_rows.append({
            "extractor": ext,
            "n_windows": n,
            "window_len": window_len,
            "n_repeats": len(group),
            "median_call1_s": float(group["duration_call1_s"].median()),
            "median_call2_s": float(group["duration_call2_s"].median()),
            "iqr_call1_s": float(group["duration_call1_s"].quantile(0.75) - group["duration_call1_s"].quantile(0.25)),
            "iqr_call2_s": float(group["duration_call2_s"].quantile(0.75) - group["duration_call2_s"].quantile(0.25)),
            "median_peak_rss_mb": float(group["peak_rss_mb"].median()),
            "median_net_rss_mb": float(group["net_rss_mb"].median()),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.sort_values(by=["extractor", "n_windows"], inplace=True)
    df_summary.to_csv(output_dir / "scaling_medians_summary.csv", index=False)

    # Fit log-log OLS slopes over upper half of N range per extractor
    # For extractors up to 1600: upper half is N in {200, 400, 800, 1600}
    # For tsfresh up to 400: upper half is N in {200, 400}
    slope_rows = []
    for ext in extractors:
        sub = df_summary[df_summary["extractor"] == ext]
        if ext == "tsfresh_efficient":
            upper_sub = sub[sub["n_windows"] >= 200]
        else:
            upper_sub = sub[sub["n_windows"] >= 200]

        if len(upper_sub) >= 2:
            x_log = np.log(upper_sub["n_windows"].to_numpy())

            # Call 1 (Cold / JIT)
            y_log1 = np.log(upper_sub["median_call1_s"].to_numpy())
            slope1, intercept1, r1, p1, stderr1 = stats.linregress(x_log, y_log1)

            # Call 2 (Warm)
            y_log2 = np.log(upper_sub["median_call2_s"].to_numpy())
            slope2, intercept2, r2, p2, stderr2 = stats.linregress(x_log, y_log2)

            # Net memory slope
            y_mem = upper_sub["median_net_rss_mb"].to_numpy()
            # If all net memory positive
            if np.all(y_mem > 0):
                y_mem_log = np.log(y_mem)
                slope_mem, _, r_mem, _, _ = stats.linregress(x_log, y_mem_log)
                r2_mem = r_mem ** 2
            else:
                slope_mem = np.nan
                r2_mem = np.nan

            slope_rows.append({
                "extractor": ext,
                "n_range_upper": f"{upper_sub['n_windows'].min()}-{upper_sub['n_windows'].max()}",
                "slope_call1_cold": round(float(slope1), 4),
                "r2_call1_cold": round(float(r1 ** 2), 4),
                "slope_call2_warm": round(float(slope2), 4),
                "r2_call2_warm": round(float(r2 ** 2), 4),
                "slope_net_rss": round(float(slope_mem), 4) if not np.isnan(slope_mem) else np.nan,
                "r2_net_rss": round(float(r2_mem), 4) if not np.isnan(r2_mem) else np.nan,
                "growth_regime_warm": "Sub-linear (O(N^b, b<1))" if slope2 < 0.95 else ("Linear (O(N))" if slope2 <= 1.05 else f"Super-linear (O(N^{slope2:.2f}))")
            })

    df_slopes = pd.DataFrame(slope_rows)
    df_slopes.to_csv(output_dir / "scaling_slopes_upper_half.csv", index=False)

    logger.info("=== Scaling Slopes (Upper Half of N) ===")
    logger.info("\n%s", df_slopes.to_string(index=False))

    # Plot scaling curves comparing Call 1 (dashed) and Call 2 (solid) alongside Net Memory
    try:
        import matplotlib.lines as mlines

        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.0,
            "grid.color": "#e0e0e0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.85,
        })

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.4), dpi=300)

        cfg = {
            "polars_statistics": {
                "name": r"$\mathtt{polars\_statistics}$",
                "color": "#0072B2",
                "marker": "o",
            },
            "numpy_statistical": {
                "name": r"$\mathtt{numpy\_statistical}$",
                "color": "#009E73",
                "marker": "s",
            },
            "numba_efficient": {
                "name": r"$\mathtt{numba\_efficient}$",
                "color": "#E69F00",
                "marker": "^",
            },
            "tsfel": {
                "name": r"$\mathtt{tsfel}$",
                "color": "#CC79A7",
                "marker": "v",
            },
            "tsfresh_efficient": {
                "name": r"$\mathtt{tsfresh\_efficient}$",
                "color": "#D55E00",
                "marker": "D",
            },
        }

        for ext in extractors:
            sub = df_summary[df_summary["extractor"] == ext].sort_values("n_windows")
            if sub.empty:
                continue
            c = cfg.get(ext, {}).get("color", "black")
            m = cfg.get(ext, {}).get("marker", "o")

            # Plot Call 1 (cold/JIT, dashed) and Call 2 (warm, solid)
            ax1.plot(
                sub["n_windows"],
                sub["median_call1_s"],
                marker=m,
                color=c,
                linestyle="--",
                linewidth=1.2,
                markersize=4.5,
                markerfacecolor="none",
                markeredgewidth=1.1,
                markeredgecolor=c,
                alpha=0.85,
                zorder=3,
            )
            ax1.plot(
                sub["n_windows"],
                sub["median_call2_s"],
                marker=m,
                color=c,
                linestyle="-",
                linewidth=1.6,
                markersize=4.5,
                markerfacecolor=c,
                markeredgewidth=0.4,
                markeredgecolor="#222222",
                alpha=0.95,
                zorder=4,
            )

            # Plot Net Memory (Log-Log)
            ax2.plot(
                sub["n_windows"],
                sub["median_net_rss_mb"],
                marker=m,
                color=c,
                linestyle="-",
                linewidth=1.6,
                markersize=4.5,
                markerfacecolor=c,
                markeredgewidth=0.4,
                markeredgecolor="#222222",
                alpha=0.95,
                zorder=4,
            )

        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xticks([50, 100, 200, 400, 800, 1600])
        ax1.set_xticklabels(["50", "100", "200", "400", "800", "1600"])
        ax1.set_xlabel(r"Number of Windows $N$ ($W=100, p=3$)", fontweight="bold")
        ax1.set_ylabel(r"Execution Time (seconds, log scale)", fontweight="bold")
        ax1.set_title(r"(a) Runtime Scaling: Cold vs Warm (Log-Log)", fontweight="bold", pad=6)
        ax1.grid(True, which="both", linestyle="--", linewidth=0.5, color="#e0e0e0", alpha=0.85)

        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xticks([50, 100, 200, 400, 800, 1600])
        ax2.set_xticklabels(["50", "100", "200", "400", "800", "1600"])
        ax2.set_ylim(0.7, 3000.0)
        ax2.set_yticks([1, 10, 100, 1000])
        ax2.set_yticklabels(["1", "10", "100", "1000"])
        ax2.set_xlabel(r"Number of Windows $N$ ($W=100, p=3$)", fontweight="bold")
        ax2.set_ylabel(r"Net Resident Memory (MB, log scale)", fontweight="bold")
        ax2.set_title(r"(b) Net Resident Memory Scaling (Log-Log)", fontweight="bold", pad=6)
        ax2.grid(True, which="both", linestyle="--", linewidth=0.5, color="#e0e0e0", alpha=0.85)

        # Top unified decoupled legend
        ext_handles = [
            mlines.Line2D(
                [], [],
                color=cfg[ext]["color"],
                marker=cfg[ext]["marker"],
                linestyle="-",
                linewidth=1.5,
                markersize=4.5,
                markerfacecolor=cfg[ext]["color"],
                markeredgecolor="#222222",
                markeredgewidth=0.4,
                label=cfg[ext]["name"],
            ) for ext in extractors if ext in cfg
        ]
        regime_handles = [
            mlines.Line2D(
                [], [],
                color="#444444",
                linestyle="--",
                linewidth=1.3,
                marker="o",
                markersize=4.2,
                markerfacecolor="none",
                markeredgewidth=1.1,
                label="Call 1 (Cold / JIT / Initial)",
            ),
            mlines.Line2D(
                [], [],
                color="#444444",
                linestyle="-",
                linewidth=1.6,
                marker="o",
                markersize=4.2,
                markerfacecolor="#444444",
                label="Call 2 (Warm / Steady-State)",
            ),
        ]

        leg1 = fig.legend(
            handles=ext_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=5,
            frameon=False,
            fontsize=7.8,
            handletextpad=0.35,
            columnspacing=0.8,
        )
        leg2 = fig.legend(
            handles=regime_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=2,
            frameon=False,
            fontsize=7.8,
            handletextpad=0.45,
            columnspacing=2.0,
        )
        fig.add_artist(leg1)

        plt.tight_layout(rect=[0, 0, 1, 0.90])
        plot_path = output_dir / "scaling_curves_call1_call2.png"
        fig.savefig(plot_path, dpi=300)
        report_fig = Path("report/figures/fig_extractor_scaling_curves.png")
        if report_fig.parent.exists():
            fig.savefig(report_fig, dpi=300)
        plt.close(fig)
        logger.info("Saved scaling figure to %s and %s", plot_path, report_fig)
    except Exception as e:
        logger.warning("Could not generate scaling plot: %s", e)

    logger.info("Task 6 complete. Outputs saved to %s", output_dir)


if __name__ == "__main__":
    out = Path("benchmark_results/TM26a19__followups/TM26a19__06_scaling_curves")
    run_scaling_benchmark(out)
