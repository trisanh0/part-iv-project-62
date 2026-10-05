"""
TEMPO Unified Benchmarking Harness and 5-Stage Pipeline Engine.

Provides an end-to-end, multi-stage benchmarking pipeline comparing time-series
feature extractors, feature selectors, and predictive models across classification
and extrinsic regression tasks with rigorous hardware telemetry and statistical evaluation.
"""

from dataclasses import asdict, dataclass, field
import datetime
import gc
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold


from tempo.extraction import (
    numba_efficient_extractor,
    numpy_statistical_extractor,
    polars_statistical_extractor,
    tsfel_extractor,
    tsfresh_extractor,
)
from tempo.selection import (
    SubsampledFeatureSelector,
    boruta_selector,
    l1_selector,
    mutual_info_selector,
    select_k_best,
    tree_importance_selector,
    tsfresh_selector,
    variance_threshold_selector,
)
from tempo.storage import load_dataset, to_numpy_tensor
from tempo.storage.feature_store import BackendType, FeatureStore
from tempo.telemetry import ResourceStats, ResourceTracker, log_system_info

logger = logging.getLogger("tempo.benchmark")



# ==============================================================================
# 1. Pipeline Configuration Schema
# ==============================================================================

@dataclass
class ExtractorConfig:
    """Configuration for a feature extractor."""
    name: str
    representation: Literal["pandas", "polars", "numpy", "auto"] = "auto"
    fft_coefficients: Optional[int] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectorConfig:
    """Configuration for a feature selector."""
    name: str
    sample_ratio: float = 0.10
    fdr_level: float = 0.05
    k: int = 20
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Master configuration for TEMPO 5-Stage Benchmarking Pipeline."""
    dataset_paths: List[str] = field(default_factory=list)
    task_type: Literal["classification", "regression", "forecasting"] = "classification"
    extractors: List[Union[str, Dict[str, Any], ExtractorConfig]] = field(
        default_factory=lambda: ["numba_efficient", "tsfresh_minimal", "polars_statistics"]
    )
    selectors: List[Union[str, Dict[str, Any], SelectorConfig, None]] = field(
        default_factory=lambda: [None, "fdr", "select_k_best"]
    )
    models: List[Union[str, Dict[str, Any]]] = field(
        default_factory=lambda: ["random_forest"]
    )
    n_splits: int = 5
    seeds: List[int] = field(default_factory=lambda: [42])
    cache_backend: BackendType = "memory"
    cache_dir: str = "data/03_processed/feature_store"
    output_dir: str = "benchmark_results"

    # Forecasting Specific Parameters
    forecast_horizon: int = 20
    history_len: int = 100
    forecast_lower: float = 0.05
    forecast_upper: float = 0.95
    test_size: float = 0.2
    
    # Telemetry and Execution Flags
    enable_telemetry: bool = True
    telemetry_interval: float = 0.05
    enable_ttests: bool = True
    enable_plots: bool = True
    enable_logging: bool = True
    log_file: Optional[str] = "tempo_benchmark.log"
    resume: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        d = asdict(self)
        return d

    def to_json(self, file_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Construct configuration from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, file_path: Union[str, Path]) -> "PipelineConfig":
        """Load configuration from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_yaml(cls, file_path: Union[str, Path]) -> "PipelineConfig":
        """Load configuration from YAML file (if pyyaml available)."""
        try:
            import yaml
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return cls.from_dict(data)
        except ImportError:
            raise ImportError("PyYAML is required to load YAML configs. Install via: pip install pyyaml")


# ==============================================================================
# 2. Benchmarking Engine
# ==============================================================================

class BakeoffRunner:
    """Execution engine for multi-stage cross-validated time-series ML bakeoffs."""

    def __init__(self, config: PipelineConfig):
        """Initialise runner with pipeline configuration."""
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_store = FeatureStore(
            storage_dir=config.cache_dir,
            backend=config.cache_backend,
        )
        self._setup_logging()
        self._log_environment()

    def _setup_logging(self) -> None:
        """Configure structured execution logging."""
        if not self.config.enable_logging:
            return

        log_path = self.output_dir / (self.config.log_file or "tempo_benchmark.log")
        handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.info("TEMPO Benchmark Runner initialised. Task: %s", self.config.task_type)

    def _log_environment(self) -> None:
        """Capture and record environment and system metadata to environment.json."""
        try:
            env_file = self.output_dir / "environment.json"
            log_system_info(env_file)
            logger.info("Recorded runtime environment metadata to %s", env_file)
        except Exception as e:
            logger.warning("Failed to record environment metadata: %s", e)

    def _extract_features(
        self,
        df_ts: pl.DataFrame,
        extractor: Union[str, Dict[str, Any], ExtractorConfig],
        dataset_name: str,
    ) -> Tuple[pd.DataFrame, ResourceStats, bool]:
        """Extract or retrieve cached features for a given dataset and extractor.

        Returns:
            Tuple of (features_df, resource_stats, is_cached).
        """
        if isinstance(extractor, ExtractorConfig):
            ext_name = extractor.name
            fft_limit = extractor.fft_coefficients
            kwargs = extractor.kwargs
        elif isinstance(extractor, dict):
            ext_name = extractor.get("name", "unknown")
            fft_limit = extractor.get("fft_coefficients")
            kwargs = extractor.get("kwargs", {})
        else:
            ext_name = str(extractor)
            fft_limit = None
            kwargs = {}

        params = {"fft_coefficients": fft_limit, **kwargs}

        # Check feature store cache first
        if self.feature_store.exists(dataset_name, ext_name, params):
            logger.info("Cache hit for extractor '%s' on dataset '%s'", ext_name, dataset_name)
            t0 = time.perf_counter()
            features = self.feature_store.load(dataset_name, ext_name, params)
            features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0).reset_index(drop=True)
            load_time = time.perf_counter() - t0
            dummy_stats = ResourceStats(
                start_ram_mb=0.0,
                peak_ram_mb=0.0,
                peak_ram_increase_mb=0.0,
                avg_cpu_percent=0.0,
                peak_cpu_percent=0.0,
                peak_gpu_percent=0.0,
                peak_gpu_memory_mb=0.0,
                duration_seconds=round(load_time, 4),
            )
            return features, dummy_stats, True

        # Feature Extraction Execution with Telemetry
        gc.collect()
        tracker = ResourceTracker(interval=self.config.telemetry_interval)

        with tracker:
            if ext_name in ("numpy_statistical", "numpy"):
                exclude_cols = {"sequence_id", "step", "id", "time"}
                feature_cols = [c for c in df_ts.columns if c not in exclude_cols]
                tensor = to_numpy_tensor(df_ts, feature_cols=feature_cols)
                features = numpy_statistical_extractor(tensor, feature_names=feature_cols)

            elif ext_name == "polars_statistics" or ext_name == "statistics":
                df_feat_pl = polars_statistical_extractor(df_ts)
                exclude_cols = {"sequence_id", "id", "step", "time"}
                feat_cols = [c for c in df_feat_pl.columns if c not in exclude_cols]
                features = pd.DataFrame(df_feat_pl.select(feat_cols).to_dict(as_series=False))

            elif ext_name == "tsfel":
                exclude_cols = {"sequence_id", "step", "id", "time"}
                feature_cols = [c for c in df_ts.columns if c not in exclude_cols]
                tensor = to_numpy_tensor(df_ts, feature_cols=feature_cols)
                if tensor.ndim == 2:
                    features = tsfel_extractor(tensor)
                elif tensor.ndim == 3:
                    n_channels = tensor.shape[2]
                    channel_dfs = []
                    for c in range(n_channels):
                        df_ch = tsfel_extractor(tensor[:, :, c])
                        df_ch.columns = [f"ch{c}_{col}" for col in df_ch.columns]
                        channel_dfs.append(df_ch)
                    features = pd.concat(channel_dfs, axis=1)
                else:
                    raise ValueError(f"Expected 2D or 3D tensor for TSFEL, got shape {tensor.shape}")

            elif ext_name == "numba_efficient" or ext_name == "numba":
                exclude_cols = {"sequence_id", "step", "id", "time"}
                feature_cols = [c for c in df_ts.columns if c not in exclude_cols]
                tensor = to_numpy_tensor(df_ts, feature_cols=feature_cols)
                if tensor.ndim == 2:
                    tensor = tensor[:, :, np.newaxis]
                n_fft = fft_limit if fft_limit is not None else 25
                features = numba_efficient_extractor(
                    tensor, raw_feature_names=feature_cols, n_fft_coeffs=n_fft
                )

            elif ext_name.startswith("tsfresh"):
                df_pandas = pd.DataFrame(df_ts.to_dict(as_series=False))
                parts = ext_name.split("_")
                param_set = parts[1] if len(parts) > 1 else "efficient"
                if param_set not in ("minimal", "efficient", "comprehensive"):
                    param_set = "efficient"
                fft_val = fft_limit if param_set != "minimal" else None
                features = tsfresh_extractor(
                    df_pandas, parameter_set=param_set, fft_coefficients=fft_val
                )
                if hasattr(features, "sort_index"):
                    features = features.sort_index()

            else:
                raise ValueError(f"Unsupported extractor identifier: {ext_name}")

        stats = tracker.stats or ResourceStats(0, 0, 0, 0, 0, 0, 0, 0)
        
        # Robust NaN and Inf handling across all feature spaces
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0).reset_index(drop=True)

        # Save to persistent feature store
        self.feature_store.save(
            features=features,
            dataset_name=dataset_name,
            extractor_name=ext_name,
            params=params,
        )
        return features, stats, False


    def _select_features(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        selector: Optional[Union[str, Dict[str, Any], SelectorConfig]],
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, ResourceStats, int, int, List[str], bool]:
        """Apply feature selector to train/test splits with hardware telemetry."""
        n_initial = X_train.shape[1]
        all_cols = list(X_train.columns)

        if selector is None or selector == "none" or selector == "None":
            dummy_stats = ResourceStats(0, 0, 0, 0, 0, 0, 0, 0)
            return X_train, X_test, dummy_stats, n_initial, n_initial, all_cols, False

        sel_name = selector.name if isinstance(selector, SelectorConfig) else (
            selector.get("name") if isinstance(selector, dict) else str(selector)
        )

        # For multi-horizon forecasting targets (2D), derive a 1D continuous target
        # signal (e.g. horizon mean) and configure task_type='regression' so standard
        # statistical filters execute mathematically without dimension mismatch.
        sel_task_type = "regression" if self.config.task_type in ("regression", "forecasting") else "classification"
        if isinstance(y_train, (pd.DataFrame, np.ndarray)) and getattr(y_train, "ndim", 1) == 2:
            sel_y = pd.Series(np.asarray(y_train).mean(axis=1))
        elif hasattr(y_train, "iloc") and len(y_train) > 0 and isinstance(y_train.iloc[0], (list, np.ndarray)):
            sel_y = pd.Series([float(np.mean(row)) for row in y_train])
        else:
            sel_y = pd.Series(np.asarray(y_train).ravel())

        gc.collect()
        tracker = ResourceTracker(interval=self.config.telemetry_interval)

        with tracker:
            if sel_name in ("tsfresh", "fdr"):
                fdr_val = 0.05
                if isinstance(selector, (dict, SelectorConfig)):
                    fdr_val = getattr(selector, "fdr_level", selector.get("fdr_level", 0.05) if isinstance(selector, dict) else 0.05)
                X_train_sel = tsfresh_selector(X_train, sel_y, fdr_level=fdr_val, task_type=sel_task_type)
                selected_cols = list(X_train_sel.columns)
                X_test_sel = X_test[selected_cols] if selected_cols else X_test.iloc[:, 0:0]

            elif sel_name in ("select_k_best", "select_k_best_anova", "anova"):
                k_val = 20
                if isinstance(selector, (dict, SelectorConfig)):
                    k_val = getattr(selector, "k", selector.get("k", 20) if isinstance(selector, dict) else 20)
                X_train_sel = select_k_best(X_train, sel_y, k=k_val, task_type=sel_task_type)
                selected_cols = list(X_train_sel.columns)
                X_test_sel = X_test[selected_cols] if selected_cols else X_test.iloc[:, 0:0]

            elif sel_name in ("mutual_info", "select_k_best_mutual_info", "mi"):
                k_val = 25
                if isinstance(selector, (dict, SelectorConfig)):
                    k_val = getattr(selector, "k", selector.get("k", 25) if isinstance(selector, dict) else 25)
                X_train_sel = mutual_info_selector(X_train, sel_y, k=k_val, task_type=sel_task_type)
                selected_cols = list(X_train_sel.columns)
                X_test_sel = X_test[selected_cols] if selected_cols else X_test.iloc[:, 0:0]

            elif sel_name == "boruta":
                n_est = 50
                m_iter = 20
                if isinstance(selector, (dict, SelectorConfig)):
                    n_est = getattr(selector, "n_estimators", selector.get("n_estimators", 50) if isinstance(selector, dict) else 50)
                    m_iter = getattr(selector, "max_iter", selector.get("max_iter", 20) if isinstance(selector, dict) else 20)
                X_train_sel = boruta_selector(
                    X_train,
                    sel_y,
                    n_estimators=n_est,
                    max_iter=m_iter,
                    random_state=seed,
                    task_type=sel_task_type,
                )
                selected_cols = list(X_train_sel.columns)
                X_test_sel = X_test[selected_cols] if selected_cols else X_test.iloc[:, 0:0]

            elif sel_name in ("variance_threshold", "variance", "zero_variance"):
                thresh_val = 0.0
                if isinstance(selector, (dict, SelectorConfig)):
                    thresh_val = getattr(selector, "threshold", selector.get("threshold", 0.0) if isinstance(selector, dict) else 0.0)
                X_train_sel = variance_threshold_selector(X_train, threshold=thresh_val)
                selected_cols = list(X_train_sel.columns)
                X_test_sel = X_test[selected_cols] if selected_cols else X_test.iloc[:, 0:0]

            elif sel_name in ("l1", "lasso"):
                c_val = 1.0
                alpha_val = 0.01
                if isinstance(selector, (dict, SelectorConfig)):
                    c_val = getattr(selector, "C", selector.get("C", 1.0) if isinstance(selector, dict) else 1.0)
                    alpha_val = getattr(selector, "alpha", selector.get("alpha", 0.01) if isinstance(selector, dict) else 0.01)
                X_train_sel = l1_selector(
                    X_train,
                    sel_y,
                    C=c_val,
                    alpha=alpha_val,
                    task_type=sel_task_type,
                    random_state=seed,
                )
                selected_cols = list(X_train_sel.columns)
                X_test_sel = X_test[selected_cols] if selected_cols else X_test.iloc[:, 0:0]

            elif sel_name in ("extra_trees", "extratrees"):
                n_est = 50
                thresh_val = "median"
                if isinstance(selector, (dict, SelectorConfig)):
                    n_est = getattr(selector, "n_estimators", selector.get("n_estimators", 50) if isinstance(selector, dict) else 50)
                    thresh_val = getattr(selector, "threshold", selector.get("threshold", "median") if isinstance(selector, dict) else "median")
                X_train_sel = tree_importance_selector(
                    X_train,
                    sel_y,
                    model_type="extra_trees",
                    n_estimators=n_est,
                    threshold=thresh_val,
                    task_type=sel_task_type,
                    random_state=seed,
                )
                selected_cols = list(X_train_sel.columns)
                X_test_sel = X_test[selected_cols] if selected_cols else X_test.iloc[:, 0:0]

            elif sel_name in ("random_forest", "rf_importance", "randomforest"):
                n_est = 50
                thresh_val = "median"
                if isinstance(selector, (dict, SelectorConfig)):
                    n_est = getattr(selector, "n_estimators", selector.get("n_estimators", 50) if isinstance(selector, dict) else 50)
                    thresh_val = getattr(selector, "threshold", selector.get("threshold", "median") if isinstance(selector, dict) else "median")
                X_train_sel = tree_importance_selector(
                    X_train,
                    sel_y,
                    model_type="random_forest",
                    n_estimators=n_est,
                    threshold=thresh_val,
                    task_type=sel_task_type,
                    random_state=seed,
                )
                selected_cols = list(X_train_sel.columns)
                X_test_sel = X_test[selected_cols] if selected_cols else X_test.iloc[:, 0:0]

            elif sel_name in ("subsampled", "subsampled_fdr"):
                ratio = 0.10
                base_sel = "tsfresh"
                if isinstance(selector, (dict, SelectorConfig)):
                    ratio = getattr(selector, "sample_ratio", selector.get("sample_ratio", 0.10) if isinstance(selector, dict) else 0.10)
                    base_sel = getattr(selector, "base_selector", selector.get("base_selector", "tsfresh") if isinstance(selector, dict) else "tsfresh")
                
                sub_sel = SubsampledFeatureSelector(
                    base_selector=base_sel,
                    sample_ratio=ratio,
                    random_state=seed,
                    task_type=sel_task_type,
                )
                X_train_sel = sub_sel.fit_transform(X_train, sel_y)
                if sub_sel.fallback_triggered_:
                    selected_cols = []
                else:
                    selected_cols = sub_sel.selected_feature_names_ or list(X_train_sel.columns)
                X_test_sel = sub_sel.transform(X_test)
            else:
                logger.warning("Unrecognised selector '%s'; falling back to unselected features.", sel_name)
                X_train_sel = X_train
                X_test_sel = X_test
                selected_cols = all_cols

        stats = tracker.stats or ResourceStats(0, 0, 0, 0, 0, 0, 0, 0)
        
        # Check if zero features survived selection
        survived_cols = list(selected_cols)
        fallback_triggered = False
        if len(survived_cols) == 0:
            fallback_triggered = True
            logger.warning(
                "No features survived selector '%s'; fallback triggered, retaining full feature space to avoid model fit failure.",
                sel_name,
            )
            X_train_sel = X_train
            X_test_sel = X_test
            n_selected = 0
        else:
            n_selected = len(survived_cols)

        # Ensure clean finite outputs
        X_train_sel = X_train_sel.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_test_sel = X_test_sel.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        return X_train_sel, X_test_sel, stats, n_initial, n_selected, survived_cols, fallback_triggered


    def _get_model(self, model_spec: Union[str, Dict[str, Any]], seed: int = 42) -> Any:
        """Instantiate scikit-learn compatible estimator based on task type."""
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer

        if isinstance(model_spec, dict):
            model_name = model_spec.get("name", "random_forest")
            seed_val = model_spec.get("random_state", seed)
        else:
            model_name = model_spec
            seed_val = seed
        
        if self.config.task_type == "classification":
            if model_name == "random_forest":
                return RandomForestClassifier(n_estimators=100, random_state=seed_val, n_jobs=-1)
            elif model_name == "logistic_regression":
                return make_pipeline(
                    SimpleImputer(strategy="constant", fill_value=0.0),
                    StandardScaler(with_mean=False),
                    LogisticRegression(max_iter=2000, random_state=seed_val),
                )
            else:
                return RandomForestClassifier(n_estimators=100, random_state=seed_val, n_jobs=-1)
        else:
            if model_name == "random_forest":
                return RandomForestRegressor(n_estimators=100, random_state=seed_val, n_jobs=-1)
            elif model_name == "ridge":
                return make_pipeline(
                    SimpleImputer(strategy="constant", fill_value=0.0),
                    StandardScaler(with_mean=False),
                    Ridge(random_state=seed_val),
                )
            else:
                return RandomForestRegressor(n_estimators=100, random_state=seed_val, n_jobs=-1)

    def run(self, resume: Optional[bool] = None) -> pd.DataFrame:
        """Execute the full 5-stage benchmark suite across datasets, extractors, and selectors."""
        results: List[Dict[str, Any]] = []

        if not self.config.dataset_paths:
            logger.warning("No datasets configured in PipelineConfig.")
            return pd.DataFrame()

        should_resume = self.config.resume if resume is None else resume
        checkpoint_path = self.output_dir / "benchmark_results_checkpoint.csv"
        completed_combos = set()
        if should_resume and checkpoint_path.exists():
            try:
                df_existing = pd.read_csv(checkpoint_path)
                results = df_existing.to_dict(orient="records")
                for r in results:
                    key = (
                        str(r.get("Dataset")),
                        str(r.get("Extractor")),
                        str(r.get("Selector")),
                        str(r.get("Model")),
                        int(r.get("Seed", 42)),
                    )
                    completed_combos.add(key)
                logger.info(
                    "Resuming from existing checkpoint at %s (%d completed combinations loaded).",
                    checkpoint_path,
                    len(completed_combos),
                )
            except Exception as e:
                logger.warning("Could not read existing checkpoint file (%s); starting fresh.", e)

        for ds_path in self.config.dataset_paths:
            ds_name = Path(ds_path).name
            logger.info("Loading dataset: %s", ds_name)
            df_ts, df_targets = load_dataset(ds_path)

            # Ensure target instances are aligned strictly with sequence ordering
            id_col_name = "sequence_id" if "sequence_id" in df_targets.columns else ("id" if "id" in df_targets.columns else None)
            if id_col_name is not None and id_col_name in df_targets.columns:
                df_targets = df_targets.sort(id_col_name)

            if self.config.task_type == "forecasting":
                horizon_cols = [c for c in df_targets.columns if c.startswith("target_") and c[7:].isdigit()]
                if horizon_cols:
                    horizon_cols = sorted(horizon_cols, key=lambda c: int(c.split("_")[1]))
                    y_all = df_targets.select(horizon_cols).to_numpy().astype(np.float32)
                elif "target" in df_targets.columns:
                    tgt_col = df_targets["target"]
                    if tgt_col.dtype == pl.List or (tgt_col.len() > 0 and isinstance(tgt_col[0], (list, np.ndarray))):
                        y_all = np.array(tgt_col.to_list(), dtype=np.float32)
                    else:
                        y_all = tgt_col.to_numpy().astype(np.float32).reshape(-1, 1)
                else:
                    raise ValueError("No valid target columns found in df_targets for forecasting.")
            else:
                y_all = pd.Series(df_targets["target"].to_numpy())

            # Detect entity / subject grouping column for Group CV
            group_col = None
            for cand in ["subject_id", "group", "entity_id", "subject", "patient_id"]:
                if cand in df_targets.columns:
                    group_col = cand
                    break

            is_tau_dataset = (
                "tau_value" in df_targets.columns
                or "tau" in df_targets.columns
                or any(k in ds_name.lower() for k in ("driftbif", "bifurcation", "tau"))
            )

            for seed in self.config.seeds:
                # Chronological / Group / Purged Cross-Validation Generator
                use_group_cv = False
                if self.config.task_type == "forecasting":
                    if "split" in df_targets.columns and {"train", "test"}.issubset(set(df_targets["split"].unique().to_list())):
                        train_idx = np.where((df_targets["split"] == "train").to_numpy())[0]
                        test_idx = np.where((df_targets["split"] == "test").to_numpy())[0]
                        splits = [(train_idx, test_idx)]
                    else:
                        n_samples = len(df_targets)
                        n_train = int(n_samples * (1.0 - self.config.test_size))
                        train_idx = np.arange(0, n_train)
                        test_idx = np.arange(n_train, n_samples)
                        splits = [(train_idx, test_idx)]
                elif group_col is not None:
                    groups = df_targets[group_col].to_numpy()
                    unique_groups = np.unique(groups)
                    n_groups = len(unique_groups)
                    n_splits = min(self.config.n_splits, n_groups)
                    if n_splits >= 2:
                        use_group_cv = True
                        logger.info(
                            "Applying Group CV on column '%s' with %d unique groups (%d splits) on dataset '%s'",
                            group_col, n_groups, n_splits, ds_name,
                        )
                        splits = None
                        if self.config.task_type == "classification" and y_all.nunique() > 1:
                            try:
                                cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                                splits = list(cv.split(df_targets, y_all, groups=groups))
                            except Exception as e:
                                logger.warning(
                                    "StratifiedGroupKFold failed (%s); falling back to seeded GroupKFold.", e
                                )

                        if splits is None:
                            rng = np.random.default_rng(seed)
                            perm = rng.permutation(unique_groups)
                            group_map = dict(zip(unique_groups, perm))
                            seeded_groups = np.array([group_map[g] for g in groups])
                            cv = GroupKFold(n_splits=n_splits)
                            splits = list(cv.split(df_targets, y_all, groups=seeded_groups))
                    else:
                        logger.warning(
                            "Group column '%s' has only %d unique group(s); falling back to non-group CV.",
                            group_col, n_groups,
                        )

                if self.config.task_type != "forecasting" and not use_group_cv:
                    if self.config.task_type == "classification" and y_all.nunique() > 1:
                        cv = StratifiedKFold(n_splits=self.config.n_splits, shuffle=True, random_state=seed)
                        splits = list(cv.split(df_targets, y_all))
                    else:
                        cv = KFold(n_splits=self.config.n_splits, shuffle=True, random_state=seed)
                        splits = list(cv.split(df_targets))

                for extractor in self.config.extractors:
                    ext_name = extractor.name if isinstance(extractor, ExtractorConfig) else (
                        extractor.get("name") if isinstance(extractor, dict) else str(extractor)
                    )

                    # Extract / Load full feature matrix
                    X_all, ext_stats, is_cached = self._extract_features(df_ts, extractor, ds_name)

                    for selector in self.config.selectors:
                        sel_name = "None" if selector is None else (
                            selector.name if isinstance(selector, SelectorConfig) else (
                                selector.get("name") if isinstance(selector, dict) else str(selector)
                            )
                        )

                        for model_spec in self.config.models:
                            model_name = model_spec if isinstance(model_spec, str) else model_spec.get("name", "model")
                            combo_key = (ds_name, ext_name, sel_name, model_name, int(seed))
                            if combo_key in completed_combos:
                                logger.info(
                                    "Resuming: skipping completed [%s | %s | %s | %s | seed %s]",
                                    *combo_key,
                                )
                                continue

                            fold_accs, fold_rmses, fold_maes, fold_r2s = [], [], [], []
                            fold_coverages, fold_widths = [], []
                            fold_h_rmses, fold_h_maes, fold_h_covs, fold_h_wids = [], [], [], []
                            last_preds, last_lower, last_upper, last_true, last_history = None, None, None, None, None
                            fold_sel_times = []
                            fold_fit_times, fold_pred_times, fold_infer_latencies_ms = [], [], []
                            fold_n_sel = []
                            fold_selected_sets = []
                            fold_fallback_flags = []
                            all_test_true_list = []
                            all_test_preds_list = []

                            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                                X_train_raw = X_all.iloc[train_idx].reset_index(drop=True)
                                if isinstance(y_all, np.ndarray):
                                    y_train_fold = y_all[train_idx]
                                    y_test_fold = y_all[test_idx]
                                else:
                                    y_train_fold = y_all.iloc[train_idx].reset_index(drop=True)
                                    y_test_fold = y_all.iloc[test_idx].reset_index(drop=True)
                                X_test_raw = X_all.iloc[test_idx].reset_index(drop=True)

                                # Stage 3: Feature Selection
                                X_tr_sel, X_te_sel, sel_stats, n_init, n_sel, sel_cols, fallback_flag = self._select_features(
                                    X_train_raw, y_train_fold, X_test_raw, selector, seed=seed
                                )
                                fold_selected_sets.append(set(sel_cols))
                                fold_fallback_flags.append(fallback_flag)

                                # Stage 4: Model Training & Inference (Disaggregated)
                                model = self._get_model(model_spec, seed=seed)
                                t_fit_start = time.perf_counter()
                                if isinstance(y_train_fold, np.ndarray) and y_train_fold.ndim == 2 and y_train_fold.shape[1] == 1:
                                    y_fit = y_train_fold.ravel()
                                elif hasattr(y_train_fold, "iloc") and getattr(y_train_fold, "ndim", 1) == 2 and y_train_fold.shape[1] == 1:
                                    y_fit = y_train_fold.iloc[:, 0].to_numpy()
                                else:
                                    y_fit = y_train_fold
                                model.fit(X_tr_sel, y_fit)
                                fit_time_sec = time.perf_counter() - t_fit_start

                                t_pred_start = time.perf_counter()
                                preds = model.predict(X_te_sel)
                                pred_time_sec = time.perf_counter() - t_pred_start
                                n_test = max(1, len(X_te_sel))
                                infer_latency_ms = (pred_time_sec / n_test) * 1000.0

                                # Evaluation metrics
                                if self.config.task_type == "classification":
                                    acc = accuracy_score(y_test_fold, preds)
                                    fold_accs.append(acc)
                                elif self.config.task_type == "regression":
                                    preds_arr = np.asarray(preds, dtype=np.float32).ravel()
                                    y_te_arr = np.asarray(y_test_fold, dtype=np.float32).ravel()
                                    fold_rmses.append(np.sqrt(mean_squared_error(y_te_arr, preds_arr)))
                                    fold_maes.append(mean_absolute_error(y_te_arr, preds_arr))
                                    fold_r2s.append(r2_score(y_te_arr, preds_arr))
                                    all_test_true_list.append(y_te_arr)
                                    all_test_preds_list.append(preds_arr)
                                elif self.config.task_type == "forecasting":
                                    preds_arr = np.asarray(preds, dtype=np.float32)
                                    y_test_arr = np.asarray(y_test_fold, dtype=np.float32)
                                    if preds_arr.ndim == 1:
                                        preds_arr = preds_arr.reshape(-1, 1)
                                    if y_test_arr.ndim == 1:
                                        y_test_arr = y_test_arr.reshape(-1, 1)

                                    rf_model = None
                                    if hasattr(model, "estimators_"):
                                        rf_model = model
                                        X_te_mat = X_te_sel.to_numpy() if hasattr(X_te_sel, "to_numpy") else np.asarray(X_te_sel)
                                    elif hasattr(model, "steps") and hasattr(model.steps[-1][1], "estimators_"):
                                        rf_model = model.steps[-1][1]
                                        X_te_trans = model[:-1].transform(X_te_sel)
                                        X_te_mat = X_te_trans.to_numpy() if hasattr(X_te_trans, "to_numpy") else np.asarray(X_te_trans)

                                    if rf_model is not None:
                                        tree_preds = [np.asarray(tree.predict(X_te_mat), dtype=np.float32) for tree in rf_model.estimators_]
                                        tree_forecasts = np.stack(tree_preds, axis=0)
                                        lower = np.quantile(tree_forecasts, self.config.forecast_lower, axis=0)
                                        upper = np.quantile(tree_forecasts, self.config.forecast_upper, axis=0)
                                    else:
                                        train_preds = np.asarray(model.predict(X_tr_sel), dtype=np.float32)
                                        if train_preds.ndim == 1:
                                            train_preds = train_preds.reshape(-1, 1)
                                        y_tr_mat = np.asarray(y_train_fold, dtype=np.float32)
                                        if y_tr_mat.ndim == 1:
                                            y_tr_mat = y_tr_mat.reshape(-1, 1)
                                        res = y_tr_mat - train_preds
                                        res_std = np.std(res, axis=0, ddof=1) if len(res) > 1 else np.zeros(preds_arr.shape[1])
                                        try:
                                            from scipy.stats import norm
                                            z = float(norm.ppf(self.config.forecast_upper))
                                        except Exception:
                                            z = 1.644853
                                        lower = preds_arr - z * res_std
                                        upper = preds_arr + z * res_std

                                    if lower.ndim == 1:
                                        lower = lower.reshape(-1, 1)
                                    if upper.ndim == 1:
                                        upper = upper.reshape(-1, 1)

                                    errors = preds_arr - y_test_arr
                                    fold_rmses.append(float(np.sqrt(np.mean(errors ** 2))))
                                    fold_maes.append(float(np.mean(np.abs(errors))))
                                    covered = (y_test_arr >= lower) & (y_test_arr <= upper)
                                    fold_coverages.append(float(np.mean(covered)))
                                    fold_widths.append(float(np.mean(upper - lower)))

                                    # Horizon-wise calculations (shape: [H])
                                    h_rmse = np.atleast_1d(np.sqrt(np.mean(errors ** 2, axis=0)))
                                    h_mae = np.atleast_1d(np.mean(np.abs(errors), axis=0))
                                    h_cov = np.atleast_1d(np.mean(covered, axis=0))
                                    h_wid = np.atleast_1d(np.mean(upper - lower, axis=0))
                                    fold_h_rmses.append(h_rmse)
                                    fold_h_maes.append(h_mae)
                                    fold_h_covs.append(h_cov)
                                    fold_h_wids.append(h_wid)

                                    last_preds = preds_arr
                                    last_lower = lower
                                    last_upper = upper
                                    last_true = y_test_arr

                                    # Extract historical input series for test sequences
                                    id_col_name = "sequence_id" if "sequence_id" in df_targets.columns else "id"
                                    test_seq_ids = df_targets[id_col_name].to_numpy()[test_idx]
                                    df_ts_test = df_ts.filter(pl.col(id_col_name).is_in(test_seq_ids))
                                    step_col_name = "step" if "step" in df_ts.columns else "time"
                                    non_id_cols = [c for c in df_ts.columns if c not in (id_col_name, step_col_name)]
                                    if "target" in df_ts.columns:
                                        feat_col_name = "target"
                                    elif "value" in df_ts.columns:
                                        feat_col_name = "value"
                                    elif non_id_cols:
                                        feat_col_name = non_id_cols[0]
                                    else:
                                        feat_col_name = df_ts.columns[-1]

                                    hist_pddf = (
                                        df_ts_test.sort([id_col_name, step_col_name])
                                        .select([id_col_name, feat_col_name])
                                        .to_pandas()
                                        .groupby(id_col_name)[feat_col_name]
                                        .apply(list)
                                    )
                                    last_history = np.asarray([hist_pddf.get(sid, []) for sid in test_seq_ids], dtype=np.float32)

                                fold_sel_times.append(sel_stats.duration_seconds)
                                fold_fit_times.append(fit_time_sec)
                                fold_pred_times.append(pred_time_sec)
                                fold_infer_latencies_ms.append(infer_latency_ms)
                                fold_n_sel.append(n_sel)

                            # Calculate Jaccard Selection Stability across folds
                            if selector is None or sel_name in ("none", "None"):
                                jaccard_stability = 1.0
                            elif any(fold_fallback_flags) and all(len(s) == 0 for s in fold_selected_sets):
                                logger.warning(
                                    "Zero features survived selector '%s' across all folds; setting Jaccard stability to 0.0.",
                                    sel_name,
                                )
                                jaccard_stability = 0.0
                            elif len(fold_selected_sets) > 1:
                                jaccard_list = []
                                for i in range(len(fold_selected_sets)):
                                    for j in range(i + 1, len(fold_selected_sets)):
                                        u = len(fold_selected_sets[i].union(fold_selected_sets[j]))
                                        if u > 0:
                                            jaccard_list.append(len(fold_selected_sets[i].intersection(fold_selected_sets[j])) / u)
                                        else:
                                            jaccard_list.append(0.0)
                                jaccard_stability = round(float(np.mean(jaccard_list)), 4) if jaccard_list else 0.0
                            else:
                                jaccard_stability = 0.0 if any(fold_fallback_flags) else 1.0

                            # Exclusion of cached loads from extraction runtime and RAM benchmarking
                            ext_time = np.nan if is_cached else ext_stats.duration_seconds
                            ext_ram = np.nan if is_cached else ext_stats.peak_ram_mb
                            ext_ram_inc = np.nan if is_cached else ext_stats.peak_ram_increase_mb
                            ext_cpu_avg = np.nan if is_cached else ext_stats.avg_cpu_percent
                            ext_cpu_peak = np.nan if is_cached else ext_stats.peak_cpu_percent
                            ext_gpu_peak = np.nan if is_cached else ext_stats.peak_gpu_percent
                            ext_gpu_ram = np.nan if is_cached else ext_stats.peak_gpu_memory_mb

                            total_time = (
                                (0.0 if is_cached else ext_stats.duration_seconds)
                                + float(np.sum(fold_sel_times))
                                + float(np.sum(fold_fit_times))
                                + float(np.sum(fold_pred_times))
                            )
                            
                            mean_fit_sec = round(float(np.mean(fold_fit_times)), 4)
                            mean_pred_sec = round(float(np.mean(fold_pred_times)), 4)
                            mean_infer_ms = round(float(np.mean(fold_infer_latencies_ms)), 4)
                            mean_n_sel = float(np.mean(fold_n_sel))
                            
                            if any(fold_fallback_flags) and mean_n_sel == 0:
                                feat_reduction_pct = 100.0
                            else:
                                feat_reduction_pct = round(
                                    (1.0 - (mean_n_sel / float(max(1, n_init)))) * 100.0, 2
                                )

                            record = {
                                "Dataset": ds_name,
                                "Task": self.config.task_type,
                                "Seed": seed,
                                "Extractor": ext_name,
                                "Selector": sel_name,
                                "Model": model_name,
                                "is_cached": is_cached,
                                "fallback_triggered": any(fold_fallback_flags),
                                
                                # Extraction Telemetry (NaN when cached)
                                "Extraction Time (s)": ext_time,
                                "Extraction Peak RAM (MB)": ext_ram,
                                "Extraction Peak RAM Increase (MB)": ext_ram_inc,
                                "Extraction Avg CPU (%)": ext_cpu_avg,
                                "Extraction Peak CPU (%)": ext_cpu_peak,
                                "Extraction Peak GPU (%)": ext_gpu_peak,
                                "Extraction Peak GPU RAM (MB)": ext_gpu_ram,

                                # Selection and Latency Telemetry
                                "Selection Time (s)": round(float(np.mean(fold_sel_times)), 4),
                                "fit_time_seconds": mean_fit_sec,
                                "inference_latency_ms": mean_infer_ms,
                                "Fit Time (s)": mean_fit_sec,
                                "Inference Latency (ms)": mean_infer_ms,
                                "Prediction Time (s)": mean_pred_sec,
                                "N Extracted Features": n_init,
                                "N Selected Features": round(mean_n_sel, 1),
                                "Feature Reduction (%)": feat_reduction_pct,
                                "Selection Stability (Jaccard)": jaccard_stability,
                                "Total Time (s)": np.nan if is_cached else round(total_time, 4),
                            }

                            if self.config.task_type == "classification":
                                record["Accuracy"] = round(float(np.mean(fold_accs)), 4)
                            elif self.config.task_type == "regression":
                                record["RMSE"] = round(float(np.mean(fold_rmses)), 4)
                                record["MAE"] = round(float(np.mean(fold_maes)), 4)
                                record["R2"] = round(float(np.mean(fold_r2s)), 4)

                                if all_test_true_list and all_test_preds_list:
                                    y_true_all = np.concatenate(all_test_true_list)
                                    y_pred_all = np.concatenate(all_test_preds_list)
                                    record["True Values"] = json.dumps(y_true_all.tolist())
                                    record["Predictions"] = json.dumps(y_pred_all.tolist())

                                    if is_tau_dataset:
                                        record["Tau RMSE"] = round(float(np.sqrt(mean_squared_error(y_true_all, y_pred_all))), 4)
                                        record["Tau MAE"] = round(float(mean_absolute_error(y_true_all, y_pred_all)), 4)
                                        record["Tau R2"] = round(float(r2_score(y_true_all, y_pred_all)), 4)
                            elif self.config.task_type == "forecasting":
                                mean_rmse = float(np.mean(fold_rmses))
                                mean_mae = float(np.mean(fold_maes))
                                mean_cov = float(np.mean(fold_coverages))
                                mean_wid = float(np.mean(fold_widths))

                                record["Forecast RMSE"] = round(mean_rmse, 4)
                                record["Forecast MAE"] = round(mean_mae, 4)
                                record["RMSE"] = round(mean_rmse, 4)
                                record["MAE"] = round(mean_mae, 4)
                                record["Interval Coverage"] = round(mean_cov, 4)
                                record["Mean Interval Width"] = round(mean_wid, 4)

                                avg_h_rmse = np.atleast_1d(np.mean(fold_h_rmses, axis=0)) if fold_h_rmses else np.array([])
                                avg_h_mae = np.atleast_1d(np.mean(fold_h_maes, axis=0)) if fold_h_maes else np.array([])
                                avg_h_cov = np.atleast_1d(np.mean(fold_h_covs, axis=0)) if fold_h_covs else np.array([])
                                avg_h_wid = np.atleast_1d(np.mean(fold_h_wids, axis=0)) if fold_h_wids else np.array([])

                                for h_idx in range(len(avg_h_rmse)):
                                    h_num = h_idx + 1
                                    record[f"RMSE_H{h_num}"] = round(float(avg_h_rmse[h_idx]), 4)
                                    record[f"MAE_H{h_num}"] = round(float(avg_h_mae[h_idx]), 4)
                                    record[f"Coverage_H{h_num}"] = round(float(avg_h_cov[h_idx]), 4)
                                    record[f"Interval_Width_H{h_num}"] = round(float(avg_h_wid[h_idx]), 4)

                                if last_history is not None:
                                    record["History Values"] = json.dumps(last_history.tolist())
                                if last_true is not None:
                                    record["True Values"] = json.dumps(last_true.tolist())
                                if last_preds is not None:
                                    record["Predictions"] = json.dumps(last_preds.tolist())
                                if last_lower is not None:
                                    record["Prediction Lower"] = json.dumps(last_lower.tolist())
                                if last_upper is not None:
                                    record["Prediction Upper"] = json.dumps(last_upper.tolist())

                            results.append(record)
                            completed_combos.add(combo_key)
                            try:
                                pd.DataFrame(results).to_csv(checkpoint_path, index=False)
                            except Exception as e:
                                logger.warning("Failed to update checkpoint CSV: %s", e)

                            logger.info(
                                "Evaluated: [%s | %s | %s | %s] -> Score: %s (Fit: %.3fs, Infer: %.2fms, Total: %.2fs)",
                                ds_name, ext_name, sel_name, model_name,
                                record.get("Accuracy") if self.config.task_type == "classification" else (
                                    record.get("Forecast RMSE") if self.config.task_type == "forecasting" else record.get("R2")
                                ),
                                mean_fit_sec, mean_infer_ms, total_time,
                            )

        df_results = pd.DataFrame(results)
        
        # Save output CSV and canonical file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = self.output_dir / f"benchmark_results_{timestamp}.csv"
        df_results.to_csv(out_csv, index=False)
        canonical_csv = self.output_dir / "benchmark_results.csv"
        df_results.to_csv(canonical_csv, index=False)
        logger.info("Saved benchmark telemetry to: %s and %s", out_csv, canonical_csv)

        # Trigger analysis if enabled
        if self.config.enable_ttests or self.config.enable_plots:
            self._trigger_analysis(canonical_csv)

        return df_results

    def _trigger_analysis(self, results_csv: Path) -> None:
        """Invoke statistical analysis and figure generation on benchmark results."""
        try:
            from tempo.analysis import run_statistical_analysis
            run_statistical_analysis(
                csv_path=str(results_csv),
                output_dir=str(self.output_dir / "analysis"),
                task_type=self.config.task_type,
                enable_ttests=self.config.enable_ttests,
                enable_plots=self.config.enable_plots,
            )
        except Exception as e:
            logger.warning("Automated analysis hook skipped or failed: %s", e)


# ==============================================================================
# 3. Command-Line Entrypoint
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TEMPO Automated Time-Series ML Benchmarking Engine")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON or YAML PipelineConfig file")
    parser.add_argument("--dataset", type=str, default="data/03_processed/beed", help="Dataset directory path")
    parser.add_argument("--task", type=str, choices=["classification", "regression", "forecasting"], default="classification")
    parser.add_argument("--cache", type=str, choices=["memory", "parquet", "hdf5", "none"], default="memory")
    parser.add_argument("--dry-run", action="store_true", help="Run quick dry-run test with minimal extractors")

    args = parser.parse_args()

    if args.config:
        if args.config.endswith((".yaml", ".yml")):
            cfg = PipelineConfig.from_yaml(args.config)
        else:
            cfg = PipelineConfig.from_json(args.config)
    else:
        cfg = PipelineConfig(
            dataset_paths=[args.dataset] if os.path.exists(args.dataset) else [],
            task_type=args.task,
            cache_backend=args.cache,
            extractors=["polars_statistics"] if args.dry_run else ["numba_efficient", "tsfresh_minimal", "polars_statistics"],
            selectors=[None, "select_k_best"] if args.dry_run else [None, "fdr", "select_k_best"],
            n_splits=2 if args.dry_run else 5,
        )

    runner = BakeoffRunner(cfg)
    print("Starting TEMPO Benchmarking Run...")
    df_res = runner.run()
    print("\nBenchmark Run Complete. Summary Results:\n")
    print(df_res)
