"""Langevin Drift-Parameter Recovery Benchmark Runner (TM26a1X).

Evaluates recovery of the physical bifurcation control parameter tau across
dissipative soliton Langevin trajectories across scaling trajectory lengths:
L in {1,000; 10,000; 100,000; 1,000,000}.

Compares four feature extraction paradigms:
1. polars_statistics (baseline summary statistics)
2. numba_efficient (vectorized JIT EfficientFCParameters replication)
3. tsfresh_efficient (standard tsfresh engine processed in entity chunks)
4. langevin_features (friedrich_coefficients + max_langevin_fixed_point)

Evaluates feature selection (None vs Benjamini-Hochberg FDR) and regression
estimators (Ridge and 100-tree RandomForest) under strict 5-fold cross-validation.
"""

import argparse
import gc
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tsfresh import select_features
from tsfresh.examples.driftbif_simulation import sample_tau, velocity
from tsfresh.feature_extraction import EfficientFCParameters, extract_features
from tsfresh.feature_extraction.feature_calculators import (
    friedrich_coefficients,
    max_langevin_fixed_point,
)

from tempo.extraction import numba_efficient_extractor, polars_statistical_extractor
from tempo.telemetry import ResourceTracker, log_system_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tempo.experiments.langevin_recovery")


def simulate_langevin_dataset(
    tau_arr: np.ndarray,
    length: int,
    sim_noise_seed: int = 42,
    kappa_3: float = 0.3,
    q_param: float = 1950.0,
    r_noise: float = 3e-4,
    delta_t: float = 0.05,
) -> Tuple[pl.DataFrame, pl.DataFrame, np.ndarray, np.ndarray]:
    """Simulate independent 1D Langevin velocity trajectories for a given length.

    Uses float64 throughout to preserve high numerical fidelity on small velocity
    increments (~10^-3) required for Friedrich polynomial coefficient estimation.

    Args:
        tau_arr: Pre-sampled 1D array of bifurcation parameters tau (float64).
        length: Number of discrete temporal steps per trajectory.
        sim_noise_seed: Random seed for trajectory thermal stochastic realization.
        kappa_3: Inverse bifurcation parameter (tau_c = 1.0 / kappa_3).
        q_param: Soliton shape parameter Q.
        r_noise: Thermal Langevin noise standard deviation.
        delta_t: Temporal discretization increment dt.

    Returns:
        Tuple of (df_ts_polars, df_targets_polars, X_array, y_array).
    """
    n_trajectories = len(tau_arr)
    rng = np.random.default_rng(sim_noise_seed)

    x_mat = np.empty((n_trajectories, length), dtype=np.float64)
    entities = []
    times = []
    velocities = []

    t_steps = np.arange(length, dtype=np.int32)

    for i in range(n_trajectories):
        sim_seed = int(rng.integers(0, 2**31 - 1))
        soliton = velocity(
            tau=float(tau_arr[i]),
            kappa_3=kappa_3,
            Q=q_param,
            R=r_noise,
            delta_t=delta_t,
            seed=sim_seed,
        )
        # 1D velocity simulation matching friedrich_coefficients notebook in float64
        v = soliton.simulate(length, v0=np.zeros(1)).flatten().astype(np.float64)
        x_mat[i, :] = v

        entities.append(np.full(length, i, dtype=np.int32))
        times.append(t_steps)
        velocities.append(v)

    df_ts = pl.DataFrame({
        "sequence_id": np.concatenate(entities),
        "step": np.concatenate(times),
        "velocity": np.concatenate(velocities),
    })

    df_targets = pl.DataFrame({
        "sequence_id": np.arange(n_trajectories, dtype=np.int32),
        "target": tau_arr,
    })

    return df_ts, df_targets, x_mat, tau_arr


def extract_polars_features(df_ts: pl.DataFrame) -> Tuple[pd.DataFrame, float, float]:
    """Extract baseline statistical features using Polars."""
    gc.collect()
    with ResourceTracker(interval=0.01) as tracker:
        t0 = time.perf_counter()
        df_feat_pl = polars_statistical_extractor(
            df_ts, id_col="sequence_id", value_cols=["velocity"]
        )
        elapsed = time.perf_counter() - t0

    df_feat = df_feat_pl.to_pandas().set_index("sequence_id")
    peak_ram = tracker.stats.peak_ram_mb if tracker.stats else 0.0
    return df_feat, elapsed, peak_ram


def extract_numba_features(x_mat: np.ndarray) -> Tuple[pd.DataFrame, float, float]:
    """Extract EfficientFCParameters replication features using Numba."""
    # Warmup JIT using float64 to ensure compilation occurs outside timed window
    _ = numba_efficient_extractor(np.random.randn(2, 50).astype(np.float64))
    gc.collect()
    with ResourceTracker(interval=0.01) as tracker:
        t0 = time.perf_counter()
        df_feat = numba_efficient_extractor(x_mat)
        elapsed = time.perf_counter() - t0

    df_feat.index = np.arange(len(x_mat), dtype=np.int32)
    peak_ram = tracker.stats.peak_ram_mb if tracker.stats else 0.0
    return df_feat, elapsed, peak_ram


def extract_tsfresh_chunked(
    df_ts: pl.DataFrame,
    chunk_size: int = 20,
    n_jobs: int = 4,
    time_limit_sec: float = 5400.0,
) -> Tuple[Optional[pd.DataFrame], float, float, bool]:
    """Extract EfficientFCParameters using tsfresh in trajectory chunks.

    Returns:
        Tuple of (df_features_or_None, elapsed_sec, peak_ram_mb, was_skipped).
    """
    df_pandas = df_ts.to_pandas()
    unique_ids = df_pandas["sequence_id"].unique()
    num_chunks = int(np.ceil(len(unique_ids) / chunk_size))
    chunk_id_groups = np.array_split(unique_ids, num_chunks)

    settings = EfficientFCParameters()
    results = []
    elapsed_total = 0.0
    peak_ram = 0.0

    gc.collect()
    t_start_master = time.perf_counter()

    for idx, c_ids in enumerate(chunk_id_groups):
        sub_df = df_pandas[df_pandas["sequence_id"].isin(c_ids)]
        with ResourceTracker(interval=0.02) as tracker:
            t0 = time.perf_counter()
            chunk_feat = extract_features(
                sub_df,
                column_id="sequence_id",
                column_sort="step",
                default_fc_parameters=settings,
                n_jobs=n_jobs,
                disable_progressbar=True,
            )
            t_chunk = time.perf_counter() - t0
            elapsed_total += t_chunk

        if tracker.stats and tracker.stats.peak_ram_mb > peak_ram:
            peak_ram = tracker.stats.peak_ram_mb

        results.append(chunk_feat)

        # Extrapolate total time after first chunk
        if idx == 0 and num_chunks > 1:
            projected_total = t_chunk * num_chunks
            logger.info(
                "Chunk 1/%d took %.1fs. Projected tsfresh total: %.1fs (%.1f min).",
                num_chunks,
                t_chunk,
                projected_total,
                projected_total / 60.0,
            )
            if projected_total > time_limit_sec:
                logger.warning(
                    "Projected tsfresh runtime (%.1f min) exceeds 90-minute limit. Skipping.",
                    projected_total / 60.0,
                )
                return None, projected_total, peak_ram, True

        if (time.perf_counter() - t_start_master) > time_limit_sec:
            logger.warning("tsfresh chunk extraction exceeded 90 minutes. Aborting remaining chunks.")
            return None, time.perf_counter() - t_start_master, peak_ram, True

    df_full = pd.concat(results, axis=0).sort_index()
    return df_full, elapsed_total, peak_ram, False


def extract_langevin_features(x_mat: np.ndarray) -> Tuple[pd.DataFrame, float, float]:
    """Extract physical Langevin parameters (friedrich_coefficients and max_langevin_fixed_point)."""
    param = [{"coeff": i, "m": 3, "r": 30} for i in range(4)]
    n_samples = len(x_mat)
    rows = []

    gc.collect()
    with ResourceTracker(interval=0.01) as tracker:
        t0 = time.perf_counter()
        for i in range(n_samples):
            v_seq = x_mat[i]
            fc_list = friedrich_coefficients(v_seq, param)
            fp_val = max_langevin_fixed_point(v_seq, m=3, r=30)
            row_dict = {f"friedrich_coeff__{k}": v for k, v in fc_list}
            row_dict["max_langevin_fixed_point__m_3__r_30"] = fp_val
            rows.append(row_dict)
        elapsed = time.perf_counter() - t0

    df_feat = pd.DataFrame(rows, index=np.arange(n_samples, dtype=np.int32))
    peak_ram = tracker.stats.peak_ram_mb if tracker.stats else 0.0
    return df_feat, elapsed, peak_ram


def evaluate_models_cross_validation(
    x_df: pd.DataFrame,
    y_arr: np.ndarray,
    feature_set_name: str,
    length: int,
    use_fdr: bool = False,
    fdr_level: float = 0.05,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray], Dict[str, int]]:
    """Perform 5-fold cross-validation for Ridge, RandomForest, and Mean Baseline.

    Drops only truly constant columns (std == 0 or relative std below 1e-10) to
    preserve physical parameters (e.g. Friedrich linear terms with variations ~10^-4).
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    models = {
        "mean_baseline": lambda: DummyRegressor(strategy="mean"),
        "ridge": lambda: Ridge(alpha=10.0),
        "random_forest": lambda: RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
    }

    metrics_tracker: Dict[str, Dict[str, List[float]]] = {
        m_name: {"r2": [], "rmse": [], "mae": []} for m_name in models
    }
    all_oof_preds: Dict[str, np.ndarray] = {
        m_name: np.zeros(len(y_arr), dtype=np.float64) for m_name in models
    }
    feature_selection_counts: Dict[str, int] = {}

    for train_idx, test_idx in kf.split(x_df, y_arr):
        x_train_raw = x_df.iloc[train_idx].copy()
        x_test_raw = x_df.iloc[test_idx].copy()
        y_train = y_arr[train_idx]
        y_test = y_arr[test_idx]

        # Median imputation on training set for NaN values
        col_medians = x_train_raw.median()
        x_train = x_train_raw.fillna(col_medians).fillna(0.0)
        x_test = x_test_raw.fillna(col_medians).fillna(0.0)

        # Drop ONLY truly constant columns (std == 0.0 or relative variation below 1e-10)
        train_std = x_train.std()
        train_mean_abs = x_train.mean().abs()
        is_truly_constant = (
            train_std.isna()
            | (train_std == 0.0)
            | (train_std / (train_mean_abs + 1e-12) < 1e-10)
        )
        valid_cols = train_std[~is_truly_constant].index
        if len(valid_cols) > 0:
            x_train = x_train[valid_cols]
            x_test = x_test[valid_cols]
        else:
            x_train = x_train.iloc[:, :1]
            x_test = x_test.iloc[:, :1]

        # FDR Selection if requested
        if use_fdr and x_train.shape[1] > 1:
            try:
                y_train_series = pd.Series(y_train, index=x_train.index)
                x_train_fdr = select_features(
                    x_train,
                    y_train_series,
                    fdr_level=fdr_level,
                    ml_task="regression",
                    show_warnings=False,
                )
                selected_cols = list(x_train_fdr.columns)
            except Exception as e:
                logger.warning("FDR selection encountered exception: %s. Falling back to all features.", e)
                selected_cols = list(x_train.columns)

            if len(selected_cols) == 0:
                # Honest fallback to unselected if FDR rejects all hypotheses
                selected_cols = list(x_train.columns)

            for c in selected_cols:
                feature_selection_counts[c] = feature_selection_counts.get(c, 0) + 1

            x_train = x_train[selected_cols]
            x_test = x_test[selected_cols]

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        # Protect against numerical overflow on rare non-finite values
        x_train_scaled = np.nan_to_num(x_train_scaled, nan=0.0, posinf=100.0, neginf=-100.0)
        x_test_scaled = np.nan_to_num(x_test_scaled, nan=0.0, posinf=100.0, neginf=-100.0)

        for m_name, m_factory in models.items():
            model = m_factory()
            if m_name == "ridge":
                model.fit(x_train_scaled, y_train)
                preds = model.predict(x_test_scaled)
            elif m_name == "random_forest":
                # Tree models do not require scaling
                model.fit(x_train, y_train)
                preds = model.predict(x_test)
            else:
                model.fit(x_train, y_train)
                preds = model.predict(x_test)

            all_oof_preds[m_name][test_idx] = preds

            r2_val = float(r2_score(y_test, preds))
            rmse_val = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae_val = float(mean_absolute_error(y_test, preds))

            metrics_tracker[m_name]["r2"].append(r2_val)
            metrics_tracker[m_name]["rmse"].append(rmse_val)
            metrics_tracker[m_name]["mae"].append(mae_val)

    rows = []
    sel_name = "fdr" if use_fdr else "none"
    for m_name in models:
        r2_list = metrics_tracker[m_name]["r2"]
        rmse_list = metrics_tracker[m_name]["rmse"]
        mae_list = metrics_tracker[m_name]["mae"]

        rows.append({
            "length": length,
            "feature_set": feature_set_name,
            "selection": sel_name,
            "model": m_name,
            "r2_mean": float(np.mean(r2_list)),
            "r2_std": float(np.std(r2_list)),
            "rmse_mean": float(np.mean(rmse_list)),
            "rmse_std": float(np.std(rmse_list)),
            "mae_mean": float(np.mean(mae_list)),
            "mae_std": float(np.std(mae_list)),
        })

    return rows, all_oof_preds, feature_selection_counts


def plot_summary_figures(
    results_df: pd.DataFrame,
    predictions_map: Dict[str, Dict[str, np.ndarray]],
    y_true_map: Dict[int, np.ndarray],
    longest_completed_length: int,
    output_path: Path,
) -> None:
    """Generate two-panel summary figure:

    (a) Best R^2 vs Trajectory Length per Feature Set (log x-axis), excluding mean_baseline.
    (b) Predicted vs True tau for the best configuration at longest length vs polars_statistics.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Best R^2 vs Length (Excluding mean_baseline to isolate real model learning curves)
    ax_a = axes[0]
    feature_sets = results_df["feature_set"].unique()
    markers = {"polars_statistics": "o", "numba_efficient": "s", "tsfresh_efficient": "^", "langevin_features": "D"}
    colors = {"polars_statistics": "#1f77b4", "numba_efficient": "#ff7f0e", "tsfresh_efficient": "#2ca02c", "langevin_features": "#d62728"}

    real_models_df = results_df[results_df["model"] != "mean_baseline"]

    for fs in feature_sets:
        fs_df = real_models_df[real_models_df["feature_set"] == fs]
        if fs_df.empty:
            continue
        # For each length, find max R^2 across genuine models and selection
        best_r2_per_len = fs_df.groupby("length")["r2_mean"].max().reset_index()
        best_r2_per_len = best_r2_per_len.sort_values("length")

        ax_a.plot(
            best_r2_per_len["length"],
            best_r2_per_len["r2_mean"],
            marker=markers.get(fs, "o"),
            color=colors.get(fs, "black"),
            label=fs,
            linewidth=2,
            markersize=7,
        )

    ax_a.set_xscale("log")
    ax_a.set_xlabel("Trajectory Length L (Time Steps)", fontsize=11, fontweight="bold")
    ax_a.set_ylabel("Best Validation $R^2$ (Models Only)", fontsize=11, fontweight="bold")
    ax_a.set_title("(a) Recovery Fidelity ($R^2$) vs Trajectory Length", fontsize=12, fontweight="bold")
    ax_a.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_a.legend(loc="lower right", frameon=True)

    # Panel B: Predicted vs True tau at longest completed length
    ax_b = axes[1]
    y_true = y_true_map.get(longest_completed_length, np.array([]))

    # Find best overall configuration at longest length
    sub_longest = results_df[
        (results_df["length"] == longest_completed_length) & (results_df["model"] != "mean_baseline")
    ]
    if not sub_longest.empty and len(y_true) > 0:
        best_row = sub_longest.sort_values("r2_mean", ascending=False).iloc[0]
        best_key = f"{best_row['feature_set']}__{best_row['selection']}__{best_row['model']}"
        best_preds = predictions_map[longest_completed_length].get(best_key, y_true)

        # Baseline polars_statistics ridge/rf
        sub_polars = sub_longest[sub_longest["feature_set"] == "polars_statistics"]
        if not sub_polars.empty:
            polars_row = sub_polars.sort_values("r2_mean", ascending=False).iloc[0]
            polars_key = f"{polars_row['feature_set']}__{polars_row['selection']}__{polars_row['model']}"
            polars_preds = predictions_map[longest_completed_length].get(polars_key, y_true)
            ax_b.scatter(
                y_true,
                polars_preds,
                color="#1f77b4",
                alpha=0.6,
                edgecolors="none",
                s=35,
                label=f"polars_statistics ($R^2$={polars_row['r2_mean']:.3f})",
            )

        ax_b.scatter(
            y_true,
            best_preds,
            color="#d62728",
            alpha=0.75,
            edgecolors="k",
            linewidth=0.5,
            s=45,
            label=f"Best: {best_row['feature_set']} ({best_row['model']}, $R^2$={best_row['r2_mean']:.3f})",
        )

        min_val = min(y_true.min(), best_preds.min())
        max_val = max(y_true.max(), best_preds.max())
        ax_b.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.7, label="Ideal Calibration")

    ax_b.set_xlabel("True Bifurcation Parameter $\\tau$", fontsize=11, fontweight="bold")
    ax_b.set_ylabel("Predicted $\\hat{\\tau}$", fontsize=11, fontweight="bold")
    ax_b.set_title(f"(b) Predicted vs True $\\tau$ (L = {longest_completed_length:,})", fontsize=12, fontweight="bold")
    ax_b.grid(True, linestyle="--", alpha=0.5)
    ax_b.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved two-panel summary figure to %s", output_path)


def run_langevin_experiment(
    output_dir: Path,
    time_budget_hours: float = 6.0,
    trajectory_lengths: Optional[List[int]] = None,
    trajectories_per_length: Optional[Dict[int, int]] = None,
    random_seed: int = 42,
    dry_run: bool = False,
) -> None:
    """Master workflow orchestrating Steps 1 to 4 of the Langevin recovery test."""
    t_master_start = time.perf_counter()
    time_budget_sec = time_budget_hours * 3600.0

    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    feat_dir = output_dir / "features"
    fig_dir = output_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "langevin_recovery.log"
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(file_handler)

    # 1. Capture environment metadata and git hash
    env_info = log_system_info(output_dir / "environment.json")
    git_hash = env_info.get("git", {}).get("commit", "unknown")
    with open(output_dir / "git_hash.txt", "w", encoding="utf-8") as f:
        f.write(f"{git_hash}\n")

    if dry_run:
        lengths = [500, 1000]
        n_traj_map = {500: 20, 1000: 20}
        skip_tsfresh_threshold = 1000000
    else:
        lengths = trajectory_lengths or [1000, 10000, 100000, 1000000]
        n_traj_map = trajectories_per_length or {1000: 200, 10000: 200, 100000: 200, 1000000: 100}
        skip_tsfresh_threshold = 5400.0  # 90 minutes

    # Seed master tau array ONCE so the same physical control parameters are reused across all lengths
    max_trajectories = max(n_traj_map.values())
    np_random_state = np.random.get_state()
    np.random.seed(random_seed)
    try:
        master_tau_list = sample_tau(max_trajectories, kappa_3=0.3, ratio=0.5, rel_increase=0.15)
    finally:
        np.random.set_state(np_random_state)
    master_tau_arr = np.array(master_tau_list, dtype=np.float64)

    all_results = []
    extraction_telemetry = []
    oof_predictions_by_length = {}
    y_true_by_length = {}
    fdr_counts_by_length_and_set: Dict[int, Dict[str, Dict[str, int]]] = {}
    longest_completed_length = lengths[0]

    logger.info("Starting Langevin recovery benchmark across lengths: %s", lengths)

    for length in lengths:
        elapsed_master = time.perf_counter() - t_master_start
        if elapsed_master >= time_budget_sec:
            logger.warning(
                "Elapsed time (%.2f hours) has reached the total budget (%.2f hours). Stopping.",
                elapsed_master / 3600.0,
                time_budget_hours,
            )
            break

        n_traj = n_traj_map.get(length, 200)
        tau_arr_len = master_tau_arr[:n_traj]
        fdr_counts_by_length_and_set[length] = {}

        logger.info(
            "=================================================================\n"
            "STAGE: Trajectory Length L = %d (N = %d trajectories)\n"
            "=================================================================",
            length,
            n_traj,
        )

        # Step 1: Simulate or load from Parquet cache
        ts_cache_path = data_dir / f"simulated_trajectories_L{length}.parquet"
        targets_cache_path = data_dir / f"targets_L{length}.parquet"

        if ts_cache_path.exists() and targets_cache_path.exists():
            logger.info("Loading cached trajectories from %s", ts_cache_path)
            df_ts = pl.read_parquet(ts_cache_path)
            df_targets = pl.read_parquet(targets_cache_path)
            tau_arr = df_targets["target"].to_numpy().astype(np.float64)
            x_mat = df_ts["velocity"].to_numpy().reshape(n_traj, length).astype(np.float64)
        else:
            logger.info("Simulating %d Langevin trajectories of length %d in float64...", n_traj, length)
            df_ts, df_targets, x_mat, tau_arr = simulate_langevin_dataset(
                tau_arr=tau_arr_len,
                length=length,
                sim_noise_seed=random_seed + length,
            )
            df_ts.write_parquet(ts_cache_path)
            df_targets.write_parquet(targets_cache_path)
            logger.info("Cached simulation trajectories to %s", ts_cache_path)

        y_true_by_length[length] = tau_arr
        oof_predictions_by_length[length] = {}

        # Step 2: Extract Four Feature Sets
        feature_sets_extracted: Dict[str, pd.DataFrame] = {}
        extract_times: Dict[str, float] = {}

        # 1. polars_statistics
        feat_p_path = feat_dir / f"features_polars_L{length}.parquet"
        if feat_p_path.exists():
            feat_polars = pd.read_parquet(feat_p_path)
            t_polars, ram_polars = 0.0, 0.0
        else:
            feat_polars, t_polars, ram_polars = extract_polars_features(df_ts)
            feat_polars.to_parquet(feat_p_path)
        feature_sets_extracted["polars_statistics"] = feat_polars
        extract_times["polars_statistics"] = t_polars
        extraction_telemetry.append({
            "length": length,
            "feature_set": "polars_statistics",
            "time_sec": t_polars,
            "peak_ram_mb": ram_polars,
            "num_features": feat_polars.shape[1],
            "status": "completed",
        })

        # 2. numba_efficient
        feat_n_path = feat_dir / f"features_numba_L{length}.parquet"
        if feat_n_path.exists():
            feat_numba = pd.read_parquet(feat_n_path)
            t_numba, ram_numba = 0.0, 0.0
        else:
            feat_numba, t_numba, ram_numba = extract_numba_features(x_mat)
            feat_numba.to_parquet(feat_n_path)
        feature_sets_extracted["numba_efficient"] = feat_numba
        extract_times["numba_efficient"] = t_numba
        extraction_telemetry.append({
            "length": length,
            "feature_set": "numba_efficient",
            "time_sec": t_numba,
            "peak_ram_mb": ram_numba,
            "num_features": feat_numba.shape[1],
            "status": "completed",
        })

        # 3. langevin_features
        feat_l_path = feat_dir / f"features_langevin_L{length}.parquet"
        if feat_l_path.exists():
            feat_langevin = pd.read_parquet(feat_l_path)
            t_lang, ram_lang = 0.0, 0.0
        else:
            feat_langevin, t_lang, ram_lang = extract_langevin_features(x_mat)
            feat_langevin.to_parquet(feat_l_path)
        feature_sets_extracted["langevin_features"] = feat_langevin
        extract_times["langevin_features"] = t_lang
        extraction_telemetry.append({
            "length": length,
            "feature_set": "langevin_features",
            "time_sec": t_lang,
            "peak_ram_mb": ram_lang,
            "num_features": feat_langevin.shape[1],
            "status": "completed",
        })

        # 4. tsfresh_efficient (with chunking and >90min skip check)
        feat_ts_path = feat_dir / f"features_tsfresh_L{length}.parquet"
        if feat_ts_path.exists():
            feat_tsfresh = pd.read_parquet(feat_ts_path)
            feature_sets_extracted["tsfresh_efficient"] = feat_tsfresh
            extract_times["tsfresh_efficient"] = 0.0
            extraction_telemetry.append({
                "length": length,
                "feature_set": "tsfresh_efficient",
                "time_sec": 0.0,
                "peak_ram_mb": 0.0,
                "num_features": feat_tsfresh.shape[1],
                "status": "completed_cached",
            })
        elif length >= 1000000:
            logger.info("Skipping tsfresh_efficient at length L=%d (exceeds 90-minute limit).", length)
            extraction_telemetry.append({
                "length": length,
                "feature_set": "tsfresh_efficient",
                "time_sec": 0.0,
                "peak_ram_mb": 0.0,
                "num_features": 0,
                "status": "skipped (>90 min limit)",
            })
        else:
            feat_tsfresh, t_tsfresh, ram_tsfresh, was_skipped = extract_tsfresh_chunked(
                df_ts, chunk_size=20, n_jobs=4, time_limit_sec=skip_tsfresh_threshold
            )
            if was_skipped or feat_tsfresh is None:
                extraction_telemetry.append({
                    "length": length,
                    "feature_set": "tsfresh_efficient",
                    "time_sec": t_tsfresh,
                    "peak_ram_mb": ram_tsfresh,
                    "num_features": 0,
                    "status": "skipped (>90 min limit)",
                })
            else:
                feat_tsfresh.to_parquet(feat_ts_path)
                feature_sets_extracted["tsfresh_efficient"] = feat_tsfresh
                extract_times["tsfresh_efficient"] = t_tsfresh
                extraction_telemetry.append({
                    "length": length,
                    "feature_set": "tsfresh_efficient",
                    "time_sec": t_tsfresh,
                    "peak_ram_mb": ram_tsfresh,
                    "num_features": feat_tsfresh.shape[1],
                    "status": "completed",
                })

        # Step 3: Model Evaluation (5-fold KFold)
        for fs_name, x_feat in feature_sets_extracted.items():
            t_ext = extract_times.get(fs_name, 0.0)

            # A. Selection: None
            res_none, preds_none, _ = evaluate_models_cross_validation(
                x_feat, tau_arr, fs_name, length, use_fdr=False, random_state=random_seed
            )
            for r in res_none:
                r["extract_time_sec"] = t_ext
                all_results.append(r)
                pred_key = f"{fs_name}__none__{r['model']}"
                oof_predictions_by_length[length][pred_key] = preds_none[r["model"]]

            # B. Selection: FDR (Skip for set 4: langevin_features)
            if fs_name != "langevin_features":
                res_fdr, preds_fdr, fdr_counts = evaluate_models_cross_validation(
                    x_feat, tau_arr, fs_name, length, use_fdr=True, random_state=random_seed
                )
                for r in res_fdr:
                    r["extract_time_sec"] = t_ext
                    all_results.append(r)
                    pred_key = f"{fs_name}__fdr__{r['model']}"
                    oof_predictions_by_length[length][pred_key] = preds_fdr[r["model"]]

                # Record FDR feature counts for every feature set
                fdr_counts_by_length_and_set[length][fs_name] = fdr_counts

        longest_completed_length = length

    # Save benchmark tabular results and telemetry
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(output_dir / "benchmark_results.csv", index=False)
    df_telemetry = pd.DataFrame(extraction_telemetry)
    df_telemetry.to_csv(output_dir / "extraction_telemetry.csv", index=False)

    # Save detailed FDR selection frequencies across all feature sets and lengths
    fdr_rows = []
    for l_val, sets_dict in fdr_counts_by_length_and_set.items():
        for fs_val, counts_dict in sets_dict.items():
            for f_name, count in counts_dict.items():
                fdr_rows.append({
                    "length": l_val,
                    "feature_set": fs_val,
                    "feature_name": f_name,
                    "folds_selected": count,
                    "total_folds": 5,
                })
    if fdr_rows:
        df_fdr = pd.DataFrame(fdr_rows)
        df_fdr.to_csv(output_dir / "fdr_selected_features.csv", index=False)

    logger.info("Saved tabular benchmark results to %s", output_dir / "benchmark_results.csv")

    # Step 4: Figures and Deliverable Report
    plot_summary_figures(
        df_results,
        oof_predictions_by_length,
        y_true_by_length,
        longest_completed_length,
        fig_dir / "langevin_recovery_summary.png",
    )

    # Extract top FDR features at longest length for all feature sets
    top_fdr_features_by_set = {}
    if longest_completed_length in fdr_counts_by_length_and_set:
        for fs_name, counts in fdr_counts_by_length_and_set[longest_completed_length].items():
            sorted_feats = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
            top_fdr_features_by_set[fs_name] = sorted_feats

    # Generate Markdown Summary Deliverable
    generate_summary_markdown(
        output_file=Path("TM26a1X__langevin_summary.md"),
        results_df=df_results,
        telemetry_df=df_telemetry,
        top_fdr_features_by_set=top_fdr_features_by_set,
        longest_length=longest_completed_length,
        fig_path=fig_dir / "langevin_recovery_summary.png",
        git_hash=git_hash,
    )


def generate_summary_markdown(
    output_file: Path,
    results_df: pd.DataFrame,
    telemetry_df: pd.DataFrame,
    top_fdr_features_by_set: Dict[str, List[Tuple[str, int]]],
    longest_length: int,
    fig_path: Path,
    git_hash: str,
) -> None:
    """Write TM26a1X__langevin_summary.md deliverable report."""
    md_lines = [
        "# Langevin Drift-Parameter Recovery Experiment Summary (TM26a1X)",
        "",
        f"**Git Commit Hash**: `{git_hash}`  ",
        f"**Longest Completed Trajectory Length**: `L = {longest_length:,}`  ",
        f"**Summary Figure**: `benchmark_results/TM26a1X__langevin/figures/{fig_path.name}`  ",
        "",
        "---",
        "",
        "## Step 0: Initial Codebase and Literature Diagnostics",
        "",
        "1. **Simulator and Default Parameters**:",
        "   - Simulator: `tsfresh.examples.driftbif_simulation.velocity` wrapped in `load_driftbif`.",
        "   - Control parameter $\\tau$ range: Drawn from `sample_tau(n, kappa_3=0.3, ratio=0.5, rel_increase=0.15)`.",
        "     With critical point $\\tau_c = 1 / \\kappa_3 = 3.3333$, the sampled range is $\\tau \\in [3.0833, 3.5833]$ (symmetric window around $\\tau_c$).",
        "   - Default parameters: $\\Delta t = 0.05$, $Q = 1950.0$, $R = 3 \\times 10^{-4}$, $\\kappa_3 = 0.3$.",
        "",
        "2. **Friedrich and Langevin Docstrings & Notebook Parameters**:",
        "   - `friedrich_coefficients`: Fits polynomial order $m$ to conditional average velocity drifts across $r$ quantiles.",
        "   - `max_langevin_fixed_point`: Estimates the largest positive fixed point $\\mathrm{argmax}_x \\{h(x) = 0\\}$.",
        "   - Notebook parameters: $m = 3$ (cubic polynomial) and $r = 30$ (30 quantiles).",
        "   - Docstrings specifically caution: *'For short time-series this method is highly dependent on the parameters.'*",
        "",
        "3. **Inclusion in `EfficientFCParameters`**:",
        "   - **Yes**. Both `friedrich_coefficients` (coefficients 0, 1, 2, 3 with $m=3, r=30$) and `max_langevin_fixed_point` ($m=3, r=30$) are included in `EfficientFCParameters`.",
        "",
        "4. **Previous TEMPO Langevin Benchmark Diagnostics**:",
        "   - Simulator: `velocity` (via `generate_drift_bifurcation_dataset`).",
        "   - Samples per trajectory: $L = 500$ time steps.",
        "   - Number of trajectories: $N = 1,000$ (in `TM26a9a` / `TM26a15a`) and $N = 200$ (in `TM26a6a`).",
        "   - $\\tau$ range: $\\tau \\in [3.4, 4.2]$ in `TM26a9a` (restricted supercritical regime) and $\\tau \\in [3.5, 4.5]$ in `TM26a6a`.",
        "   - Time step: $\\Delta t = 0.05$.",
        "",
        "---",
        "",
        "## Feature Extraction Telemetry",
        "",
        "| Length $L$ | Feature Set | Extraction Time (s) | Peak Memory (MB) | Feature Count | Status |",
        "| :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for _, row in telemetry_df.iterrows():
        md_lines.append(
            f"| {row['length']:,} | `{row['feature_set']}` | {row['time_sec']:.2f} | {row['peak_ram_mb']:.1f} | {row['num_features']} | {row['status']} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## Benchmark Results Table",
        "",
        "| Length $L$ | Feature Set | Selection | Model | Validation $R^2$ | RMSE | Extraction Time (s) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for _, row in results_df.iterrows():
        md_lines.append(
            f"| {row['length']:,} | `{row['feature_set']}` | {row['selection']} | {row['model']} | {row['r2_mean']:.4f} ± {row['r2_std']:.4f} | {row['rmse_mean']:.4f} ± {row['rmse_std']:.4f} | {row.get('extract_time_sec', 0.0):.2f} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        f"## Most Frequently Selected Features under FDR by Feature Set (L = {longest_length:,})",
        "",
    ])

    if top_fdr_features_by_set:
        for fs_name, top_feats in top_fdr_features_by_set.items():
            md_lines.extend([
                f"### Feature Set: `{fs_name}`",
                "",
                "| Rank | Feature Identifier | Selection Frequency (out of 5 Folds) |",
                "| :---: | :--- | :---: |",
            ])
            for idx, (f_name, count) in enumerate(top_feats, start=1):
                md_lines.append(f"| {idx} | `{f_name}` | {count}/5 |")
            md_lines.append("")
    else:
        md_lines.append("No features selected under FDR or FDR was not evaluated.")

    md_lines.extend([
        "",
        "---",
        "",
        "## Artifact and Figure Locations",
        "",
        f"- Two-Panel Figure: `benchmark_results/TM26a1X__langevin/figures/{fig_path.name}`",
        "- Raw Benchmark Results: `benchmark_results/TM26a1X__langevin/benchmark_results.csv`",
        "- Extraction Telemetry: `benchmark_results/TM26a1X__langevin/extraction_telemetry.csv`",
        "- FDR Feature Selection Log: `benchmark_results/TM26a1X__langevin/fdr_selected_features.csv`",
        "- System Environment Metadata: `benchmark_results/TM26a1X__langevin/environment.json`",
        "- Git Commit Hash: `benchmark_results/TM26a1X__langevin/git_hash.txt`",
    ])

    output_file.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Wrote summary markdown deliverable to %s", output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Langevin Recovery Benchmark Runner (TM26a1X)")
    parser.add_argument("--output-dir", type=str, default="benchmark_results/TM26a1X__langevin")
    parser.add_argument("--budget-hours", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Quick verification test")
    args = parser.parse_args()

    run_langevin_experiment(
        output_dir=Path(args.output_dir),
        time_budget_hours=args.budget_hours,
        random_seed=args.seed,
        dry_run=args.dry_run,
    )
