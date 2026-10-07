"""
Subsampled Two-Stage Feature Selection Module for the TEMPO Framework.

Implements a high-throughput, two-stage feature selection workflow designed for
large-scale and streaming sensor datasets. Subsamples the raw data matrix,
fits statistical significance filters (FDR hypothesis tests / Boruta / SelectKBest),
and applies the surviving feature mask across the entire dataset.
"""

import inspect
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from tsfresh import select_features

from tempo.selection.statistical import select_k_best, mutual_info_selector, variance_threshold_selector, l1_selector
from tempo.selection.wrappers import boruta_selector, tree_importance_selector

logger = logging.getLogger(__name__)


class SubsampledFeatureSelector(BaseEstimator, TransformerMixin):
    """Two-stage subsampled feature selector.
    
    Subsamples a fraction of data rows, runs feature selection to identify
    statistically significant features, and transforms the full matrix by
    filtering to the selected feature subspace.
    """

    def __init__(
        self,
        base_selector: Optional[Union[str, Callable, BaseEstimator]] = "tsfresh",
        sample_ratio: float = 0.10,
        fdr_level: float = 0.05,
        random_state: Optional[int] = 42,
        stratify: bool = True,
        min_samples: int = 20,
        task_type: str = "classification",
        k: int = 20,
        threshold: Optional[Union[float, str]] = None,
        C: float = 1.0,
        alpha: float = 0.01,
        n_estimators: int = 50,
        max_iter: int = 20,
        base_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Initialise the subsampled feature selector.
        
        Args:
            base_selector: Selection strategy ('tsfresh', 'select_k_best', or an sklearn selector instance).
            sample_ratio: Fraction of rows to sample for feature evaluation (0 < sample_ratio <= 1.0).
            fdr_level: False Discovery Rate threshold for tsfresh hypothesis testing.
            random_state: Seed for reproducible subsampling.
            stratify: Whether to preserve class label proportions in subsampling.
            min_samples: Minimum absolute number of rows required in subsample.
            task_type: Problem type ('classification' or 'regression').
            k: Top-k features for SelectKBest / Mutual Info selectors.
            threshold: Variance or importance cutoff threshold.
            C: Inverse regularization strength for L1 logistic regression.
            alpha: Regularization strength for Lasso / L1 regression.
            n_estimators: Number of trees for tree-based or Boruta selectors.
            max_iter: Maximum iterations for Boruta selector.
            base_params: Explicit parameter dictionary forwarded to base selector.
            **kwargs: Additional parameters forwarded to base selector.
        """
        self.base_selector = base_selector
        self.sample_ratio = sample_ratio
        self.fdr_level = fdr_level
        self.random_state = random_state
        self.stratify = stratify
        self.min_samples = min_samples
        self.task_type = task_type
        self.k = k
        self.threshold = threshold
        self.C = C
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.base_params = base_params or {}
        self.kwargs = kwargs

        # Fitted attributes
        self.selected_feature_names_: Optional[List[str]] = None
        self.survived_features_: List[str] = []
        self.support_mask_: Optional[np.ndarray] = None
        self.n_features_in_: int = 0
        self.n_features_out_: int = 0
        self.fallback_triggered_: bool = False
        self.telemetry_: Dict[str, Any] = {}

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, Sequence],
    ) -> "SubsampledFeatureSelector":
        """Fit the selector on a subsample of X and y.
        
        Args:
            X: Input feature matrix of shape (n_samples, n_features).
            y: Target values of shape (n_samples,).
            
        Returns:
            Fitted SubsampledFeatureSelector instance.
        """
        t0 = time.perf_counter()

        # Standardise inputs to pandas
        if isinstance(X, np.ndarray):
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = X.copy()
            feature_names = [str(c) for c in X_df.columns]
            X_df.columns = feature_names

        X_df = X_df.reset_index(drop=True)
        if isinstance(y, (pd.Series, np.ndarray)):
            y_series = pd.Series(np.asarray(y), index=X_df.index)
        else:
            y_series = pd.Series(y, index=X_df.index)

        n_samples, n_features = X_df.shape
        self.n_features_in_ = n_features

        # Determine subsample size
        if self.sample_ratio >= 1.0 or n_samples <= self.min_samples:
            subsample_idx = np.arange(n_samples)
            X_sub = X_df
            y_sub = y_series
        else:
            target_n = max(self.min_samples, int(np.ceil(n_samples * self.sample_ratio)))

            # Stratification check (only applicable for classification with multi-class representation)
            strat_labels = None
            if self.stratify and self.task_type == "classification" and y_series.nunique() > 1:
                val_counts = y_series.value_counts()
                if (val_counts >= 2).all():
                    strat_labels = y_series
                    min_class_count = int(val_counts.min())
                    # Ensure subsample retains at least 2 samples per class
                    target_n = max(target_n, int(np.ceil(2.0 * n_samples / float(min_class_count))))

            target_n = min(target_n, n_samples)

            try:
                subsample_idx, _ = train_test_split(
                    np.arange(n_samples),
                    train_size=target_n,
                    random_state=self.random_state,
                    stratify=strat_labels,
                )
            except Exception:
                subsample_idx, _ = train_test_split(
                    np.arange(n_samples),
                    train_size=target_n,
                    random_state=self.random_state,
                    stratify=None,
                )

            X_sub = X_df.iloc[subsample_idx].reset_index(drop=True)
            y_sub = y_series.iloc[subsample_idx].reset_index(drop=True)

        # Merge hyperparameters: self defaults < self.base_params < self.kwargs
        resolved_params: Dict[str, Any] = {
            "k": self.k,
            "threshold": self.threshold,
            "C": self.C,
            "alpha": self.alpha,
            "n_estimators": self.n_estimators,
            "max_iter": self.max_iter,
            "fdr_level": self.fdr_level,
            "random_state": self.random_state,
        }
        resolved_params.update(self.base_params)
        resolved_params.update(self.kwargs)

        # Execute base selection on subsample
        if self.base_selector == "tsfresh" or self.base_selector is None:
            X_sub_clean = X_sub.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            nunique = X_sub_clean.nunique()
            non_const = nunique[nunique > 1].index
            if len(non_const) > 0:
                X_sub_clean = X_sub_clean[non_const]
            try:
                ml_task = "regression" if self.task_type == "regression" else "auto"
                fdr = float(resolved_params.get("fdr_level", self.fdr_level))
                X_filtered = select_features(X_sub_clean, y_sub, fdr_level=fdr, ml_task=ml_task)
                survived = list(X_filtered.columns)
            except Exception:
                survived = []

        elif self.base_selector in ("select_k_best", "select_k_best_anova", "anova"):
            k_val = int(resolved_params.get("k", self.k))
            score_func = resolved_params.get("score_func", None)
            df_sel = select_k_best(X_sub, y_sub, k=k_val, score_func=score_func, task_type=self.task_type)
            survived = list(df_sel.columns)

        elif self.base_selector in ("mutual_info", "mi"):
            k_val = int(resolved_params.get("k", self.k))
            rs = resolved_params.get("random_state", self.random_state)
            df_sel = mutual_info_selector(X_sub, y_sub, k=k_val, task_type=self.task_type, random_state=rs)
            survived = list(df_sel.columns)

        elif self.base_selector == "boruta":
            n_est = int(resolved_params.get("n_estimators", self.n_estimators))
            m_iter = int(resolved_params.get("max_iter", self.max_iter))
            m_samp = resolved_params.get("max_samples", 0.7)
            rs = resolved_params.get("random_state", self.random_state or 42)
            df_sel = boruta_selector(
                X_sub,
                y_sub,
                n_estimators=n_est,
                max_iter=m_iter,
                max_samples=m_samp,
                task_type=self.task_type,
                random_state=rs,
            )
            survived = list(df_sel.columns)

        elif self.base_selector in ("variance_threshold", "variance", "zero_variance"):
            thresh_param = resolved_params.get("threshold")
            thresh = 0.0 if thresh_param is None else float(thresh_param)
            df_sel = variance_threshold_selector(X_sub, threshold=thresh)
            survived = list(df_sel.columns)

        elif self.base_selector in ("l1", "lasso"):
            c_val = float(resolved_params.get("C", self.C))
            a_val = float(resolved_params.get("alpha", self.alpha))
            rs = resolved_params.get("random_state", self.random_state or 42)
            df_sel = l1_selector(
                X_sub,
                y_sub,
                C=c_val,
                alpha=a_val,
                task_type=self.task_type,
                random_state=rs,
            )
            survived = list(df_sel.columns)

        elif self.base_selector in ("extra_trees", "extratrees"):
            n_est = int(resolved_params.get("n_estimators", self.n_estimators))
            thresh = resolved_params.get("threshold")
            if thresh is None:
                thresh = "median"
            rs = resolved_params.get("random_state", self.random_state or 42)
            df_sel = tree_importance_selector(
                X_sub,
                y_sub,
                model_type="extra_trees",
                n_estimators=n_est,
                threshold=thresh,
                task_type=self.task_type,
                random_state=rs,
            )
            survived = list(df_sel.columns)

        elif self.base_selector in ("random_forest", "rf_importance", "randomforest"):
            n_est = int(resolved_params.get("n_estimators", self.n_estimators))
            thresh = resolved_params.get("threshold")
            if thresh is None:
                thresh = "median"
            rs = resolved_params.get("random_state", self.random_state or 42)
            df_sel = tree_importance_selector(
                X_sub,
                y_sub,
                model_type="random_forest",
                n_estimators=n_est,
                threshold=thresh,
                task_type=self.task_type,
                random_state=rs,
            )
            survived = list(df_sel.columns)

        elif hasattr(self.base_selector, "fit") and hasattr(self.base_selector, "get_support"):
            self.base_selector.fit(X_sub, y_sub)
            supp = self.base_selector.get_support()
            survived = [f for f, s in zip(feature_names, supp) if s]

        elif callable(self.base_selector):
            sig = inspect.signature(self.base_selector)
            call_kwargs = {}
            for param_name, param_val in resolved_params.items():
                if param_name in sig.parameters:
                    call_kwargs[param_name] = param_val
            if "task_type" in sig.parameters and "task_type" not in call_kwargs:
                call_kwargs["task_type"] = self.task_type

            res = self.base_selector(X_sub, y_sub, **call_kwargs)
            if isinstance(res, pd.DataFrame):
                survived = list(res.columns)
            elif isinstance(res, (list, np.ndarray)):
                survived = [str(f) for f in res]
            else:
                survived = feature_names
        else:
            survived = feature_names

        # Track surviving features honestly before fallback
        raw_survived = len(survived)
        self.fallback_triggered_ = (raw_survived == 0)

        if self.fallback_triggered_:
            logger.warning(
                "No features survived selection; fallback triggered, retaining full feature space (%d features).",
                n_features,
            )
            # Retain all features in transform to prevent downstream crash
            self.survived_features_ = []
            self.selected_feature_names_ = list(feature_names)
            self.n_features_out_ = n_features
            reduction_pct = 0.0
        else:
            self.survived_features_ = list(survived)
            self.selected_feature_names_ = list(survived)
            self.n_features_out_ = raw_survived
            reduction_pct = round(
                (1.0 - (self.n_features_out_ / float(n_features))) * 100.0, 2
            )

        self.support_mask_ = np.array([f in self.selected_feature_names_ for f in feature_names])

        fit_time = time.perf_counter() - t0
        self.telemetry_ = {
            "fit_time_sec": round(fit_time, 4),
            "subsample_size": len(X_sub),
            "total_size": n_samples,
            "sample_ratio_actual": round(len(X_sub) / float(n_samples), 4),
            "n_initial_features": n_features,
            "n_survived_features": raw_survived,
            "n_selected_features": self.n_features_out_,
            "fallback_triggered": self.fallback_triggered_,
            "feature_reduction_pct": reduction_pct,
        }
        logger.info(
            "Subsampled selection complete: %d -> %d features (fallback=%s) in %.4fs (subsample: %d/%d rows)",
            n_features,
            self.n_features_out_,
            self.fallback_triggered_,
            fit_time,
            len(X_sub),
            n_samples,
        )
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Filter X to only the selected feature subspace.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features).
            
        Returns:
            Reduced feature matrix of shape (n_samples, n_features_out).
        """
        if self.selected_feature_names_ is None or self.support_mask_ is None:
            raise RuntimeError("SubsampledFeatureSelector must be fitted before calling transform().")

        if isinstance(X, pd.DataFrame):
            available = [c for c in self.selected_feature_names_ if c in X.columns]
            return X[available]
        elif isinstance(X, np.ndarray):
            return X[:, self.support_mask_]
        else:
            raise TypeError(f"Unsupported feature matrix type: {type(X)}")

    def get_support(self, indices: bool = False) -> Union[np.ndarray, List[int]]:
        """Get mask or integer indices of selected features."""
        if self.support_mask_ is None:
            raise RuntimeError("SubsampledFeatureSelector must be fitted first.")
        if indices:
            return np.where(self.support_mask_)[0].tolist()
        return self.support_mask_


def evaluate_subsampling_sweep(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    sample_ratios: Sequence[float] = (0.05, 0.10, 0.20, 0.50, 1.0),
    base_selector: Optional[Union[str, Callable, BaseEstimator]] = "tsfresh",
    fdr_level: float = 0.05,
    random_state: int = 42,
    task_type: str = "classification",
    **kwargs: Any,
) -> pd.DataFrame:
    """Benchmark feature selection latency and feature reduction across sample ratios.
    
    Args:
        X: Feature matrix.
        y: Target series.
        sample_ratios: Sequence of sample ratios to evaluate.
        base_selector: Base selector strategy or estimator.
        fdr_level: FDR alpha level.
        random_state: Seed for reproducibility.
        task_type: Target problem type ('classification' or 'regression').
        **kwargs: Additional parameters forwarded to SubsampledFeatureSelector.
        
    Returns:
        DataFrame containing telemetry summary for each sample ratio.
    """
    records: List[Dict[str, Any]] = []

    for ratio in sample_ratios:
        selector = SubsampledFeatureSelector(
            base_selector=base_selector,
            sample_ratio=ratio,
            fdr_level=fdr_level,
            random_state=random_state,
            task_type=task_type,
            **kwargs,
        )
        selector.fit(X, y)
        telemetry = selector.telemetry_
        records.append({
            "Sample Ratio Config": ratio,
            "Sample Ratio Actual": telemetry["sample_ratio_actual"],
            "Subsample Rows": telemetry["subsample_size"],
            "Total Rows": telemetry["total_size"],
            "Fit Time (s)": telemetry["fit_time_sec"],
            "Initial Features": telemetry["n_initial_features"],
            "Survived Features": telemetry.get("n_survived_features", telemetry["n_selected_features"]),
            "Selected Features": telemetry["n_selected_features"],
            "Fallback Triggered": telemetry["fallback_triggered"],
            "Feature Reduction (%)": telemetry["feature_reduction_pct"],
        })

    return pd.DataFrame(records)

