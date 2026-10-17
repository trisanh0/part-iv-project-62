import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

def generate_figure():
    base_dir = Path("benchmark_results/TM26a1X__langevin")
    df_results = pd.read_csv(base_dir / "benchmark_results.csv")

    # True 1:1 publication scale matching 12pt Times New Roman in LaTeX
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.labelsize": 10.0,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 7.5,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.85,
    })

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 3.1), dpi=300)

    # High-contrast colourblind-safe palette:
    # polars = Deep Blue (#0072B2) vs tsfresh = Vermilion/Red (#D55E00) for maximum perceptual distance
    # numba = Green (#009E73), langevin = Warm Amber (#E69F00)
    c_polars = "#0072B2"
    c_tsfresh = "#D55E00"
    c_numba = "#009E73"
    c_langevin = "#E69F00"

    palette = {
        "polars_statistics": {
            "color": c_polars,
            "ls": "-",
            "marker": "o",
            "ms": 5.0,
            "mfc": "none",
            "mew": 1.2,
            "lw": 1.5,
            "label": r"$\mathtt{polars\_statistics}$ ($D = 5$)",
        },
        "tsfresh_efficient": {
            "color": c_tsfresh,
            "ls": "-.",
            "marker": "^",
            "ms": 4.5,
            "mfc": c_tsfresh,
            "mew": 0.4,
            "lw": 1.5,
            "label": r"$\mathtt{tsfresh\_efficient}$ ($D = 777$)",
        },
        "numba_efficient": {
            "color": c_numba,
            "ls": "--",
            "marker": "s",
            "ms": 4.0,
            "mfc": c_numba,
            "mew": 0.4,
            "lw": 1.5,
            "label": r"$\mathtt{numba\_efficient}$ ($D = 447$)",
        },
        "langevin_features": {
            "color": c_langevin,
            "ls": ":",
            "marker": "D",
            "ms": 4.0,
            "mfc": c_langevin,
            "mew": 0.4,
            "lw": 1.5,
            "label": r"Langevin features ($D = 5$)",
        },
    }

    # --- Panel A: Learning Curves ---
    real_models = df_results[df_results["model"] != "mean_baseline"]
    lengths = [1000, 10000, 100000, 1000000]

    for fs in ["polars_statistics", "numba_efficient", "tsfresh_efficient", "langevin_features"]:
        sub = real_models[real_models["feature_set"] == fs]
        if sub.empty:
            continue
        best_r2 = sub.groupby("length")["r2_mean"].max().reset_index().sort_values("length")
        cfg = palette[fs]

        ax_a.plot(
            best_r2["length"],
            best_r2["r2_mean"],
            linestyle=cfg["ls"],
            linewidth=cfg["lw"],
            color=cfg["color"],
            marker=cfg["marker"],
            markersize=cfg["ms"],
            markerfacecolor=cfg["mfc"],
            markeredgecolor=cfg["color"] if cfg["mfc"] == "none" else "#222222",
            markeredgewidth=cfg["mew"],
            label=cfg["label"],
            alpha=0.95,
            zorder=3,
        )

    ax_a.axhline(0.0, color="#777777", linestyle=":", linewidth=0.9, alpha=0.8, label=r"Mean baseline ($R^2 \approx 0$)", zorder=2)

    ax_a.set_xscale("log")
    ax_a.set_xticks(lengths)
    ax_a.set_xticklabels([r"$10^3$", r"$10^4$", r"$10^5$", r"$10^6$"])
    ax_a.minorticks_off()

    ax_a.set_ylim(-0.12, 1.05)
    ax_a.set_xlabel(r"Trajectory Length $L$ (Time Steps)", fontweight="bold")
    ax_a.set_ylabel(r"Cross-Validated $R^2$", fontweight="bold")
    ax_a.set_title(r"(a) Validation $R^2$ vs Length $L$", fontweight="bold", pad=7)

    ax_a.grid(True, which="major", linestyle="--", linewidth=0.5, color="#e0e0e0", alpha=0.85)
    ax_a.legend(loc="lower right", frameon=True, framealpha=0.85, edgecolor="none", borderpad=0.3, labelspacing=0.25)

    # --- Panel B: Predicted vs True tau at L = 10^5 ---
    y_true = pd.read_parquet(base_dir / "data" / "targets_L100000.parquet")["target"].to_numpy()
    feat_polars = pd.read_parquet(base_dir / "features" / "features_polars_L100000.parquet")
    feat_tsfresh = pd.read_parquet(base_dir / "features" / "features_tsfresh_L100000.parquet")

    def get_oof(X_df, y):
        stds = X_df.std(axis=0)
        valid_cols = stds[stds > 0].index
        X = X_df[valid_cols].copy()
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros(len(y))
        for train_idx, test_idx in cv.split(X, y):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr = y[train_idx]
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=4)
            model.fit(X_tr, y_tr)
            oof[test_idx] = model.predict(X_te)
        return oof

    pred_polars = get_oof(feat_polars, y_true)
    pred_tsfresh = get_oof(feat_tsfresh, y_true)

    val_min, val_max = 3.06, 3.60
    ax_b.plot([val_min, val_max], [val_min, val_max], color="#444444", linestyle="--", linewidth=1.0, alpha=0.8, label=r"Ideal ($y = x$)", zorder=1)
    ax_b.axvline(1.0 / 0.3, color="#777777", linestyle=":", linewidth=1.0, alpha=0.8, label=r"$\tau_c = 1/\kappa_3 \approx 3.333$", zorder=1)

    # Scatter: open Deep Blue rings for polars vs solid Vermilion/Red dots for tsfresh
    ax_b.scatter(
        y_true,
        pred_polars,
        facecolors="none",
        edgecolors=c_polars,
        linewidths=0.9,
        s=22,
        alpha=0.75,
        label=r"$\mathtt{polars\_statistics}$ ($R^2 = 0.931$)",
        zorder=2,
    )
    ax_b.scatter(
        y_true,
        pred_tsfresh,
        facecolors=c_tsfresh,
        edgecolors="none",
        s=12,
        alpha=0.80,
        label=r"$\mathtt{tsfresh\_efficient}$ ($R^2 = 0.927$)",
        zorder=3,
    )

    ax_b.set_xlim(val_min, val_max)
    ax_b.set_ylim(val_min, val_max)
    ax_b.set_xlabel(r"True Bifurcation Parameter $\tau$", fontweight="bold")
    ax_b.set_ylabel(r"Predicted Parameter $\hat{\tau}$", fontweight="bold")
    ax_b.set_title(r"(b) Predicted vs True $\tau$ ($L = 10^5$)", fontweight="bold", pad=7)

    ax_b.grid(True, which="major", linestyle="--", linewidth=0.5, color="#e0e0e0", alpha=0.85)
    ax_b.legend(loc="upper left", frameon=True, framealpha=0.85, edgecolor="none", borderpad=0.3, labelspacing=0.25)

    plt.tight_layout(w_pad=2.0)

    out1 = base_dir / "figures" / "langevin_recovery_summary.png"
    out2 = Path("report/figures/langevin_recovery_summary.png")
    fig.savefig(out1, dpi=300)
    fig.savefig(out2, dpi=300)
    print("Successfully generated high-contrast figure.")

if __name__ == "__main__":
    generate_figure()
