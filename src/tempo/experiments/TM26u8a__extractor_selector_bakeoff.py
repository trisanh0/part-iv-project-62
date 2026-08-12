"""TEMPO Extractor + Selector Bake-Off Benchmark Suite.

Executes comparative telemetry across feature extractors (Polars Statistics, TSFEL,
TSFresh Minimal/Efficient/Comprehensive, Numba JIT) and selectors (SelectKBest, FDR)
evaluating execution time, feature reduction ratios, peak memory footprint, and
predictive performance (R^2 / Accuracy) using 5-fold cross-validation and Scott's in-memory caching.
"""

import gc
import os
import time
import tracemalloc
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import StratifiedKFold, KFold

from tempo.storage import load_dataset, to_numpy_tensor
from tempo.extraction import (
    polars_statistical_extractor,
    tsfresh_extractor,
    numba_efficient_extractor,
    tsfel_extractor,
)
from tempo.selection import select_k_best, tsfresh_selector


class ExtractorCache:
    """In-memory feature extraction cache matching Scott's caching pattern."""

    def __init__(self, name: str, extractor_fn, representation: str = "df", kwargs: Optional[dict] = None):
        self.name = name
        self.extractor_fn = extractor_fn
        self.representation = representation
        self.kwargs = kwargs or {}
        self.cached_seed: Optional[int] = None
        self.cached_features: Optional[pd.DataFrame] = None
        self.cached_extract_time: float = 0.0
        self.cached_peak_mem_mb: float = 0.0

    def extract(self, df_ts: pl.DataFrame, seed: int, n_fft_coeffs: int = 25) -> Tuple[pd.DataFrame, float, float]:
        """Extract features with in-memory caching per seed."""
        if self.cached_seed == seed and self.cached_features is not None:
            return self.cached_features, self.cached_extract_time, self.cached_peak_mem_mb

        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()

        if self.name == "statistics":
            df_feat_pl = polars_statistical_extractor(df_ts)
            id_col = "sequence_id" if "sequence_id" in df_feat_pl.columns else "id"
            df_features = df_feat_pl.to_pandas().set_index(id_col)

        elif self.name == "tsfel":
            df_pandas = df_ts.to_pandas()
            id_col = "sequence_id" if "sequence_id" in df_pandas.columns else "id"
            df_features = tsfel_extractor(df_pandas, column_id=id_col)

        elif self.name.startswith("tsfresh"):
            df_pandas = df_ts.to_pandas()
            param_set = self.name.split("-")[1]
            fft_limit = n_fft_coeffs if param_set != "minimal" else None
            df_features = tsfresh_extractor(df_pandas, parameter_set=param_set, fft_coefficients=fft_limit)

        elif self.name == "numba-efficient":
            exclude_cols = {"sequence_id", "step", "id", "time"}
            feature_cols = [c for c in df_ts.columns if c not in exclude_cols]
            tensor = to_numpy_tensor(df_ts, feature_cols=feature_cols)
            if tensor.ndim == 2:
                tensor = tensor[:, :, np.newaxis]
            df_features = numba_efficient_extractor(tensor, raw_feature_names=feature_cols, n_fft_coeffs=n_fft_coeffs)

        else:
            raise ValueError(f"Unknown extractor name: {self.name}")

        extract_time = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mem_mb = peak_bytes / (1024.0 * 1024.0)

        # Update in-memory cache
        self.cached_seed = seed
        self.cached_features = df_features
        self.cached_extract_time = extract_time
        self.cached_peak_mem_mb = peak_mem_mb

        return df_features, extract_time, peak_mem_mb


def run_bakeoff_experiment(
    dataset_name: str = "beed",
    extractors: Optional[List[str]] = None,
    selectors: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    n_seeds: int = 5,
    n_splits: int = 5,
    n_fft_coeffs: int = 25,
    k_best: int = 20,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Execute 5-fold cross-validated Extractor + Selector Bake-Off across seeds.

    Args:
        dataset_name: Identifier of plug-and-play TEMPO dataset ('beed', 'pred-maintenance', 'simulated').
        extractors: List of extractor names to benchmark.
        selectors: List of selector names to benchmark.
        models: List of model families ('rf', 'linear').
        n_seeds: Number of independent random seeds to evaluate.
        n_splits: Number of cross-validation folds per seed.
        n_fft_coeffs: Maximum Fourier coefficients for truncated spectral extractors.
        k_best: Number of features to retain for K-Best feature selection.
        output_dir: Directory path to save resulting Parquet and CSV telemetry summaries.

    Returns:
        pandas DataFrame containing full 5-fold cross-validated telemetry results.
    """
    if extractors is None:
        extractors = [
            "statistics",
            "tsfel",
            "tsfresh-minimal",
            "tsfresh-efficient",
            "numba-efficient",
        ]
    if selectors is None:
        selectors = ["none", "kbest", "tsfresh-fdr"]
    if models is None:
        models = ["rf", "linear"]

    # Load plug-and-play standardized TEMPO dataset
    df_ts, df_targets = load_dataset(dataset_name)

    # Initialize in-memory extractor caches matching Scott's pattern
    caches = {ext_name: ExtractorCache(ext_name, None) for ext_name in extractors}

    results = []

    for seed in range(n_seeds):
        print(f"\n--- Seed {seed + 1}/{n_seeds} on Dataset '{dataset_name.upper()}' ---")

        for ext_name in extractors:
            cache = caches[ext_name]

            # 1. Extract (or pull from in-memory cache)
            try:
                df_features, extract_time, peak_mem_mb = cache.extract(df_ts, seed=seed, n_fft_coeffs=n_fft_coeffs)
            except Exception as e:
                print(f"Extraction failed for {ext_name} on seed {seed}: {e}")
                continue

            n_features_extracted = df_features.shape[1]

            # Align targets
            target_id_col = "sequence_id" if "sequence_id" in df_targets.columns else "id"
            df_targets_pd = df_targets.to_pandas().set_index(target_id_col)
            common_idx = df_features.index.intersection(df_targets_pd.index)

            X = df_features.loc[common_idx].fillna(0.0)
            y = df_targets_pd.loc[common_idx, "target"].values

            is_classification = np.issubdtype(y.dtype, np.integer) or len(np.unique(y)) <= 10

            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed) if is_classification else KFold(n_splits=n_splits, shuffle=True, random_state=seed)

            for sel_name in selectors:
                for model_name in models:
                    fold_scores = []
                    fold_select_times = []
                    fold_predict_times = []
                    n_selected_list = []

                    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]

                        # 2. Feature Selection
                        t0_sel = time.perf_counter()
                        if sel_name == "none":
                            X_tr_sel, X_te_sel = X_train, X_test
                        elif sel_name == "kbest":
                            X_tr_sel = select_k_best(X_train, y_train, k=k_best)
                            X_te_sel = X_test.reindex(columns=X_tr_sel.columns, fill_value=0.0)
                        elif sel_name == "tsfresh-fdr":
                            X_tr_sel = tsfresh_selector(X_train, y_train)
                            if X_tr_sel.shape[1] == 0:
                                X_tr_sel = X_train
                            X_te_sel = X_test.reindex(columns=X_tr_sel.columns, fill_value=0.0)
                        else:
                            raise ValueError(f"Unknown selector: {sel_name}")

                        sel_time = time.perf_counter() - t0_sel
                        fold_select_times.append(sel_time)
                        n_selected_list.append(X_tr_sel.shape[1])

                        # 3. Model Fit & Predict
                        t0_pred = time.perf_counter()
                        if model_name == "rf":
                            clf = RandomForestClassifier(n_estimators=100, random_state=seed) if is_classification else RandomForestRegressor(n_estimators=100, random_state=seed)
                        elif model_name == "linear":
                            clf = LogisticRegression(max_iter=1000, random_state=seed) if is_classification else Ridge(random_state=seed)
                        else:
                            raise ValueError(f"Unknown model: {model_name}")

                        clf.fit(X_tr_sel, y_train)
                        y_pred = clf.predict(X_te_sel)
                        pred_time = time.perf_counter() - t0_pred
                        fold_predict_times.append(pred_time)

                        score = float(accuracy_score(y_test, y_pred)) if is_classification else float(r2_score(y_test, y_pred))
                        fold_scores.append(score)

                    results.append({
                        "dataset": dataset_name,
                        "seed": seed,
                        "extractor": ext_name,
                        "selector": sel_name,
                        "model": model_name,
                        "n_features_extracted": n_features_extracted,
                        "n_features_selected": int(np.mean(n_selected_list)),
                        "extract_time_sec": extract_time,
                        "peak_mem_mb": peak_mem_mb,
                        "mean_select_time_sec": float(np.mean(fold_select_times)),
                        "mean_predict_time_sec": float(np.mean(fold_predict_times)),
                        "total_time_sec": extract_time + float(np.sum(fold_select_times)) + float(np.sum(fold_predict_times)),
                        "metric_type": "accuracy" if is_classification else "r2_score",
                        "score_mean": float(np.mean(fold_scores)),
                        "score_std": float(np.std(fold_scores)),
                    })

    df_results = pd.DataFrame(results)

    if output_dir is None:
        output_dir = os.path.join("data", "03_processed", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    parquet_path = os.path.join(output_dir, "bakeoff_results.parquet")
    csv_path = os.path.join(output_dir, "bakeoff_results.csv")

    df_results.to_parquet(parquet_path, index=False)
    df_results.to_csv(csv_path, index=False)

    print(f"\nBake-Off Telemetry saved to {parquet_path} and {csv_path}")
    return df_results


if __name__ == "__main__":
    print("TEMPO Extractor + Selector Bake-Off module ready.")
    print("To run bakeoff, execute: run_bakeoff_experiment('beed')")
