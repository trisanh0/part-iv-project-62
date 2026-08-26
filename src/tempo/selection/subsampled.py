"""
Subsampled Two-Stage Feature Selection Module for the TEMPO Framework.

Implements a high-throughput, two-stage feature selection workflow designed for
large-scale and streaming sensor datasets. Subsamples the raw data matrix,
fits statistical significance filters (FDR hypothesis tests / Boruta / SelectKBest),
and applies the surviving feature mask across the entire dataset.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from tsfresh import select_features

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
    ):
        """Initialise the subsampled feature selector.
        
        Args:
            base_selector: Selection strategy ('tsfresh', 'select_k_best', or an sklearn selector instance).
            sample_ratio: Fraction of rows to sample for feature evaluation (0 < sample_ratio <= 1.0).
            fdr_level: False Discovery Rate threshold for tsfresh hypothesis testing.
            random_state: Seed for reproducible subsampling.
            stratify: Whether to preserve class label proportions in subsampling.
            min_samples: Minimum absolute number of rows required in subsample.
        """
        self.base_selector = base_selector
        self.sample_ratio = sample_ratio
        self.fdr_level = fdr_level
        self.random_state = random_state
        self.stratify = stratify
        self.min_samples = min_samples

        # Fitted attributes
        self.selected_feature_names_: Optional[List[str]] = None
        self.support_mask_: Optional[np.ndarray] = None
        self.n_features_in_: int = 0
        self.n_features_out_: int = 0
        self.telemetry_: Dict[str, float] = {}

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

        if not isinstance(y, pd.Series):
            y_series = pd.Series(y, index=X_df.index)
        else:
            y_series = y.copy()

        n_samples, n_features = X_df.shape
        self.n_features_in_ = n_features

        # Determine subsample size
        if self.sample_ratio >= 1.0 or n_samples <= self.min_samples:
            subsample_idx = X_df.index
            X_sub = X_df
            y_sub = y_series
        else:
            target_n = max(self.min_samples, int(np.ceil(n_samples * self.sample_ratio)))
            target_n = min(target_n, n_samples)
            test_fraction = 1.0 - (target_n / float(n_samples))

            # Stratification check
            strat_labels = None
            if self.stratify and y_series.nunique() > 1:
                # Only stratify if all classes have at least 2 samples
                val_counts = y_series.value_counts()
                if (val_counts >= 2).all():
                    strat_labels = y_series

            try:
                subsample_idx, _ = train_test_split(
                    X_df.index,
                    train_size=target_n,
                    random_state=self.random_state,
                    stratify=strat_labels,
                )
            except Exception:
                subsample_idx, _ = train_test_split(
                    X_df.index,
                    train_size=target_n,
                    random_state=self.random_state,
                    stratify=None,
                )

            X_sub = X_df.loc[subsample_idx]
            y_sub = y_series.loc[subsample_idx]

        # Execute base selection on subsample
        if self.base_selector == "tsfresh" or self.base_selector is None:
            X_filtered = select_features(X_sub, y_sub, fdr_level=self.fdr_level)
            self.selected_feature_names_ = list(X_filtered.columns)

        elif hasattr(self.base_selector, "fit") and hasattr(self.base_selector, "get_support"):
            self.base_selector.fit(X_sub, y_sub)
            supp = self.base_selector.get_support()
            self.selected_feature_names_ = [f for f, s in zip(feature_names, supp) if s]

        elif callable(self.base_selector):
            res = self.base_selector(X_sub, y_sub)
            if isinstance(res, pd.DataFrame):
                self.selected_feature_names_ = list(res.columns)
            elif isinstance(res, (list, np.ndarray)):
                self.selected_feature_names_ = [str(f) for f in res]
            else:
                self.selected_feature_names_ = feature_names
        else:
            self.selected_feature_names_ = feature_names

        # Fallback if no features selected (retain all to prevent downstream crash)
        if not self.selected_feature_names_:
            logger.warning("No features survived selection; falling back to full feature space.")
            self.selected_feature_names_ = feature_names

        self.support_mask_ = np.array([f in self.selected_feature_names_ for f in feature_names])
        self.n_features_out_ = len(self.selected_feature_names_)

        fit_time = time.perf_counter() - t0
        self.telemetry_ = {
            "fit_time_sec": round(fit_time, 4),
            "subsample_size": len(X_sub),
            "total_size": n_samples,
            "sample_ratio_actual": round(len(X_sub) / float(n_samples), 4),
            "n_initial_features": n_features,
            "n_selected_features": self.n_features_out_,
            "feature_reduction_pct": round(
                (1.0 - (self.n_features_out_ / float(n_features))) * 100.0, 2
            ),
        }
        logger.info(
            "Subsampled selection complete: %d -> %d features in %.4fs (subsample: %d/%d rows)",
            n_features,
            self.n_features_out_,
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
    fdr_level: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Benchmark feature selection latency and feature reduction across sample ratios.
    
    Args:
        X: Feature matrix.
        y: Target series.
        sample_ratios: Sequence of sample ratios to evaluate.
        fdr_level: FDR alpha level.
        random_state: Seed for reproducibility.
        
    Returns:
        DataFrame containing telemetry summary for each sample ratio.
    """
    records: List[Dict[str, Any]] = []

    for ratio in sample_ratios:
        selector = SubsampledFeatureSelector(
            base_selector="tsfresh",
            sample_ratio=ratio,
            fdr_level=fdr_level,
            random_state=random_state,
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
            "Selected Features": telemetry["n_selected_features"],
            "Feature Reduction (%)": telemetry["feature_reduction_pct"],
        })

    return pd.DataFrame(records)
