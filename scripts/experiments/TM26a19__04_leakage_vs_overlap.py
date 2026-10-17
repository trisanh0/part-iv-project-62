#!/usr/bin/env python3
"""
Task 4: Leakage vs Overlap with Purged Blocked Folds (TM26a19__04).

Repeats the AI4I shuffled-vs-blocked comparison across strides S in {100, 75, 50, 25, 10} (W = 100):
  1. 20 random seeds for shuffled partitions.
  2. Blocked folds strictly enforce purge of ceil(W/S) - 1 adjacent windows on each side of test block.
  3. Reports N, positive-class rate, majority-class accuracy, mean and 95% CI of accuracy
     and balanced accuracy for both splitters, and the gap with its CI.
  4. Logs per-fold class counts and flags any fold missing a class.
  5. Produces gap (balanced accuracy) vs overlap figure with 95% CIs.

Output:
  benchmark_results/TM26a19__followups/TM26a19__04_leakage_vs_overlap/
"""

import json
import logging
import math
from pathlib import Path
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

# Ensure src is on python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tempo.extraction import polars_statistical_extractor
from tempo.storage.dataset import segment_time_series

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tempo.task4_leakage")


def load_raw_ai4i(raw_path: str = "data/01_raw/pred-maintenance/ai4i2020.csv") -> pl.DataFrame:
    """Load and prepare continuous AI4I data."""
    df = pl.read_csv(raw_path)
    leak_cols = ["TWF", "HDF", "PWF", "OSF", "RNF", "Product ID"]
    df_filtered = df.drop([c for c in leak_cols if c in df.columns])

    type_mapping = {"L": 0, "M": 1, "H": 2}
    df_proc = df_filtered.with_columns([
        pl.col("UDI").cast(pl.Int32).alias("time"),
        pl.col("Type").replace_strict(type_mapping, default=0).cast(pl.Int32).alias("Type"),
        pl.col("Machine failure").cast(pl.Int32).alias("target"),
    ])
    return df_proc


def create_purged_blocked_splits(n_samples: int, n_splits: int, purge_len: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate purged chronological splits where train windows overlapping test block are removed."""
    block_size = n_samples // n_splits
    splits = []
    indices = np.arange(n_samples)

    for k in range(n_splits):
        t_start = k * block_size
        t_end = n_samples if k == n_splits - 1 else (k + 1) * block_size
        test_idx = np.arange(t_start, t_end)

        # Purge boundary: remove purge_len windows before t_start and after t_end
        purge_start = max(0, t_start - purge_len)
        purge_end = min(n_samples, t_end + purge_len)
        excluded_range = np.arange(purge_start, purge_end)

        train_idx = np.setdiff1d(indices, excluded_range)
        splits.append((train_idx, test_idx))

    return splits


def compute_ci(data: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """Compute mean and symmetric Student-t confidence interval half-width."""
    arr = np.array(data)
    n = len(arr)
    mean_val = float(np.mean(arr))
    if n <= 1:
        return mean_val, 0.0, 0.0
    sem = stats.sem(arr)
    t_crit = stats.t.ppf((1 + confidence) / 2.0, df=n - 1)
    ci_half = float(t_crit * sem)
    return mean_val, mean_val - ci_half, mean_val + ci_half


def main() -> None:
    output_dir = Path("benchmark_results/TM26a19__followups/TM26a19__04_leakage_vs_overlap")
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_ai4i()
    window_size = 100
    strides = [100, 75, 50, 25, 10]
    n_splits = 5
    n_seeds = 20

    feature_cols = [
        "Type", "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
    ]

    pipeline_spec = {
        "extractor": "polars_statistical_extractor",
        "selector": "None",
        "model": "RandomForestClassifier(n_estimators=50)",
        "window_size_W": window_size,
        "n_shuffled_seeds": n_seeds,
        "n_cv_splits": n_splits,
    }

    raw_eval_records = []
    stride_summary_records = []
    blocked_fold_audit = []

    for stride in strides:
        overlap_pct = round((1.0 - (stride / window_size)) * 100.0, 1)
        purge_windows = math.ceil(window_size / stride) - 1

        logger.info("=" * 75)
        logger.info("Stride S=%d | Overlap: %.1f%% | Purge Window Each Side: %d", stride, overlap_pct, purge_windows)
        logger.info("=" * 75)

        # 1. Segment Time Series
        df_ts, df_targets = segment_time_series(
            raw_df,
            window_size=window_size,
            stride=stride,
            time_col="time",
            feature_cols=feature_cols,
            label_col="target",
            label_strategy="any_positive",
        )

        n_windows = df_targets.height
        y = df_targets["target"].to_numpy()
        n_pos = int(np.sum(y))
        n_neg = n_windows - n_pos
        pos_rate = float(n_pos / n_windows)
        maj_class_rate = float(max(n_pos, n_neg) / n_windows)

        # 2. Extract Features
        feat_pl = polars_statistical_extractor(df_ts)
        non_feat = [c for c in feat_pl.columns if c in ("sequence_id", "id", "step", "time")]
        X = feat_pl.drop(non_feat).to_pandas().replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 3. Create Purged Blocked Splits
        blocked_splits = create_purged_blocked_splits(n_windows, n_splits, purge_windows)

        # Audit blocked fold class balance
        for fold_idx, (tr_idx, te_idx) in enumerate(blocked_splits):
            y_te = y[te_idx]
            y_tr = y[tr_idx]
            pos_te = int(np.sum(y_te))
            neg_te = len(y_te) - pos_te
            pos_tr = int(np.sum(y_tr))
            neg_tr = len(y_tr) - pos_tr
            missing_class = bool(pos_te == 0 or neg_te == 0)

            blocked_fold_audit.append({
                "stride": stride,
                "overlap_pct": overlap_pct,
                "fold_idx": fold_idx,
                "n_test": len(te_idx),
                "n_train_purged": len(tr_idx),
                "n_purged_windows": n_windows - len(te_idx) - len(tr_idx),
                "test_positives": pos_te,
                "test_negatives": neg_te,
                "train_positives": pos_tr,
                "train_negatives": neg_tr,
                "missing_class_flag": missing_class,
            })

        # 4. Evaluate Blocked CV across seeds with pooled out-of-fold predictions
        blocked_pooled_accs = []
        blocked_pooled_baccs = []
        blocked_fold_baccs = []
        for seed in range(n_seeds):
            preds_pooled = np.zeros(n_windows, dtype=int)
            fold_bacc = []
            for tr_idx, te_idx in blocked_splits:
                clf = RandomForestClassifier(n_estimators=50, random_state=seed)
                clf.fit(X.iloc[tr_idx], y[tr_idx])
                te_pred = clf.predict(X.iloc[te_idx])
                preds_pooled[te_idx] = te_pred
                fold_bacc.append(balanced_accuracy_score(y[te_idx], te_pred))
            blocked_pooled_accs.append(float(accuracy_score(y, preds_pooled)))
            blocked_pooled_baccs.append(float(balanced_accuracy_score(y, preds_pooled)))
            blocked_fold_baccs.append(float(np.mean(fold_bacc)))

        # 5. Evaluate Shuffled CV across seeds with pooled out-of-fold predictions
        shuffled_pooled_accs = []
        shuffled_pooled_baccs = []
        shuffled_fold_baccs = []
        for seed in range(n_seeds):
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            preds_pooled = np.zeros(n_windows, dtype=int)
            fold_bacc = []
            for tr_idx, te_idx in cv.split(X, y):
                clf = RandomForestClassifier(n_estimators=50, random_state=seed)
                clf.fit(X.iloc[tr_idx], y[tr_idx])
                te_pred = clf.predict(X.iloc[te_idx])
                preds_pooled[te_idx] = te_pred
                fold_bacc.append(balanced_accuracy_score(y[te_idx], te_pred))
            shuffled_pooled_accs.append(float(accuracy_score(y, preds_pooled)))
            shuffled_pooled_baccs.append(float(balanced_accuracy_score(y, preds_pooled)))
            shuffled_fold_baccs.append(float(np.mean(fold_bacc)))

            # Record per-seed paired evaluations
            gap_acc = shuffled_pooled_accs[-1] - blocked_pooled_accs[-1]
            gap_bacc = shuffled_pooled_baccs[-1] - blocked_pooled_baccs[-1]
            gap_fold_bacc = shuffled_fold_baccs[-1] - blocked_fold_baccs[-1]
            raw_eval_records.append({
                "stride": stride,
                "overlap_pct": overlap_pct,
                "seed": seed,
                "shuffled_pooled_accuracy": shuffled_pooled_accs[-1],
                "blocked_pooled_accuracy": blocked_pooled_accs[-1],
                "accuracy_gap": gap_acc,
                "shuffled_pooled_balanced_accuracy": shuffled_pooled_baccs[-1],
                "blocked_pooled_balanced_accuracy": blocked_pooled_baccs[-1],
                "balanced_accuracy_gap": gap_bacc,
                "shuffled_fold_balanced_accuracy": shuffled_fold_baccs[-1],
                "blocked_fold_balanced_accuracy": blocked_fold_baccs[-1],
                "fold_balanced_accuracy_gap": gap_fold_bacc,
            })

        # Calculate means and 95% CIs on pooled out-of-fold metrics
        shuf_acc_m, shuf_acc_low, shuf_acc_high = compute_ci(shuffled_pooled_accs)
        shuf_bacc_m, shuf_bacc_low, shuf_bacc_high = compute_ci(shuffled_pooled_baccs)
        blk_acc_m, blk_acc_low, blk_acc_high = compute_ci(blocked_pooled_accs)
        blk_bacc_m, blk_bacc_low, blk_bacc_high = compute_ci(blocked_pooled_baccs)

        gaps_acc = [s - b for s, b in zip(shuffled_pooled_accs, blocked_pooled_accs)]
        gaps_bacc = [s - b for s, b in zip(shuffled_pooled_baccs, blocked_pooled_baccs)]
        gap_acc_m, gap_acc_low, gap_acc_high = compute_ci(gaps_acc)
        gap_bacc_m, gap_bacc_low, gap_bacc_high = compute_ci(gaps_bacc)

        # Also compute fold-averaged balanced accuracy metrics for diagnostic audit
        shuf_fbacc_m, _, _ = compute_ci(shuffled_fold_baccs)
        blk_fbacc_m, _, _ = compute_ci(blocked_fold_baccs)
        fold_gaps_bacc = [s - b for s, b in zip(shuffled_fold_baccs, blocked_fold_baccs)]
        gap_fbacc_m, _, _ = compute_ci(fold_gaps_bacc)

        stride_summary_records.append({
            "stride": stride,
            "overlap_pct": overlap_pct,
            "purge_windows": purge_windows,
            "n_windows": n_windows,
            "positive_class_rate": round(pos_rate, 4),
            "majority_class_accuracy": round(maj_class_rate, 4),
            "shuffled_acc_mean": round(shuf_acc_m, 4),
            "shuffled_acc_ci95": [round(shuf_acc_low, 4), round(shuf_acc_high, 4)],
            "blocked_acc_mean": round(blk_acc_m, 4),
            "blocked_acc_ci95": [round(blk_acc_low, 4), round(blk_acc_high, 4)],
            "acc_gap_mean": round(gap_acc_m, 4),
            "acc_gap_ci95": [round(gap_acc_low, 4), round(gap_acc_high, 4)],
            "shuffled_bacc_mean": round(shuf_bacc_m, 4),
            "shuffled_bacc_ci95": [round(shuf_bacc_low, 4), round(shuf_bacc_high, 4)],
            "blocked_bacc_mean": round(blk_bacc_m, 4),
            "blocked_bacc_ci95": [round(blk_bacc_low, 4), round(blk_bacc_high, 4)],
            "bacc_gap_mean": round(gap_bacc_m, 4),
            "bacc_gap_ci95": [round(gap_bacc_low, 4), round(gap_bacc_high, 4)],
            "shuffled_fold_bacc_mean": round(shuf_fbacc_m, 4),
            "blocked_fold_bacc_mean": round(blk_fbacc_m, 4),
            "fold_bacc_gap_mean": round(gap_fbacc_m, 4),
        })

    # Save outputs
    df_raw = pd.DataFrame(raw_eval_records)
    df_raw.to_csv(output_dir / "leakage_raw_seed_evaluations.csv", index=False)

    df_audit = pd.DataFrame(blocked_fold_audit)
    df_audit.to_csv(output_dir / "blocked_fold_class_audit.csv", index=False)

    df_sum = pd.DataFrame(stride_summary_records)
    df_sum.to_csv(output_dir / "leakage_summary_table.csv", index=False)

    with open(output_dir / "leakage_metadata.json", "w", encoding="utf-8") as f:
        json.dump({"pipeline": pipeline_spec, "summary": stride_summary_records}, f, indent=2)

    # Plot figure: Balanced Accuracy Gap vs Overlap with 95% CIs
    try:
        overlaps = [r["overlap_pct"] for r in stride_summary_records]
        bacc_gaps = [r["bacc_gap_mean"] for r in stride_summary_records]
        bacc_err_low = [r["bacc_gap_mean"] - r["bacc_gap_ci95"][0] for r in stride_summary_records]
        bacc_err_high = [r["bacc_gap_ci95"][1] - r["bacc_gap_mean"] for r in stride_summary_records]

        acc_gaps = [r["acc_gap_mean"] for r in stride_summary_records]
        acc_err_low = [r["acc_gap_mean"] - r["acc_gap_ci95"][0] for r in stride_summary_records]
        acc_err_high = [r["acc_gap_ci95"][1] - r["acc_gap_mean"] for r in stride_summary_records]

        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.0,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "grid.color": "#e0e0e0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.85,
        })

        fig, ax = plt.subplots(figsize=(4.8, 3.2), dpi=300)

        c_bacc = "#D55E00"  # Vermilion
        c_acc = "#0072B2"   # Deep Blue

        ax.axhline(
            0.0,
            color="#777777",
            linestyle=":",
            linewidth=1.0,
            alpha=0.85,
            label=r"Zero leakage baseline ($\Delta = 0$)",
            zorder=1,
        )

        ax.errorbar(
            overlaps,
            bacc_gaps,
            yerr=[bacc_err_low, bacc_err_high],
            fmt="o-",
            color=c_bacc,
            linewidth=1.6,
            markersize=5.2,
            markerfacecolor=c_bacc,
            markeredgecolor="#222222",
            markeredgewidth=0.4,
            capsize=3.5,
            capthick=1.1,
            label=r"Balanced Accuracy Gap ($\Delta_{\mathrm{BACC}}$)",
            zorder=3,
        )

        ax.errorbar(
            overlaps,
            acc_gaps,
            yerr=[acc_err_low, acc_err_high],
            fmt="s--",
            color=c_acc,
            linewidth=1.4,
            markersize=4.6,
            markerfacecolor="none",
            markeredgecolor=c_acc,
            markeredgewidth=1.2,
            capsize=3.0,
            capthick=1.0,
            label=r"Standard Accuracy Gap ($\Delta_{\mathrm{Acc}}$)",
            zorder=2,
        )

        ax.set_xticks([0, 25, 50, 75, 90])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "90%"])
        ax.set_xlim(-4, 94)

        ax.set_ylim(-0.02, 0.28)
        ax.set_yticks([0.00, 0.05, 0.10, 0.15, 0.20, 0.25])
        ax.set_yticklabels(["0.00", "0.05", "0.10", "0.15", "0.20", "0.25"])

        ax.set_xlabel(r"Window Overlap Ratio (%) [$W=100$, Stride $S \in \{100, 75, 50, 25, 10\}$]", fontweight="bold")
        ax.set_ylabel(r"Cross-Validation Leakage Gap ($\Delta$)", fontweight="bold")

        ax.grid(True, which="major", linestyle="--", linewidth=0.5, color="#e0e0e0", alpha=0.85)

        ax.legend(
            loc="upper left",
            frameon=True,
            framealpha=0.88,
            edgecolor="none",
            borderpad=0.35,
            labelspacing=0.25,
        )

        plt.tight_layout(pad=0.6)
        plot_path = output_dir / "gap_balanced_acc_vs_overlap.png"
        fig.savefig(plot_path, dpi=300)
        report_fig_path = Path("report/figures/fig_leakage_vs_overlap.png")
        if report_fig_path.parent.exists():
            fig.savefig(report_fig_path, dpi=300)
            logger.info("Copied figure to report path: %s", report_fig_path)
        plt.close(fig)
        logger.info("Saved figure to %s", plot_path)
    except Exception as e:
        logger.warning("Could not generate leakage plot: %s", e)

    logger.info("=== Task 4 Leakage Summary ===")
    logger.info("\n%s", df_sum[["stride", "overlap_pct", "purge_windows", "n_windows", "acc_gap_mean", "bacc_gap_mean", "blocked_bacc_mean", "shuffled_bacc_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
