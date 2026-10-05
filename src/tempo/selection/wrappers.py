"""Wrapper-based feature selection methods."""

import logging
from typing import Literal, Optional, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

logger = logging.getLogger(__name__)

try:
    from boruta import BorutaPy
    _HAS_BORUTA = True
except ImportError:
    BorutaPy = None
    _HAS_BORUTA = False


def boruta_selector(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    n_estimators: int = 50,
    max_iter: int = 20,
    max_samples: Optional[int] = 1000,
    random_state: int = 42,
    task_type: Literal["classification", "regression"] = "classification",
) -> pd.DataFrame:
    """Select features using Boruta wrapper with Random Forest importance.

    Args:
        X: Feature matrix as pandas DataFrame.
        y: Target classification labels or continuous regression targets.
        n_estimators: Number of decision trees.
        max_iter: Maximum number of Boruta iterations (default: 20).
        max_samples: Maximum number of samples to use during shadow ranking (default: 1000).
        random_state: Seed for reproducibility.
        task_type: Target problem type ('classification' or 'regression').

    Returns:
        Filtered pandas DataFrame containing selected features.
    """
    if not _HAS_BORUTA:
        raise ImportError("boruta package is required for boruta_selector. Install with pip install boruta.")

    X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if X_clean.shape[1] == 0:
        return X_clean.iloc[:, 0:0]

    y_arr = np.asarray(y)

    # Subsample rows if dataset is large to maintain tractable runtime
    if max_samples is not None and len(X_clean) > max_samples:
        rng = np.random.RandomState(random_state)
        indices = rng.choice(len(X_clean), size=max_samples, replace=False)
        X_fit = X_clean.iloc[indices].values
        y_fit = y_arr[indices]
    else:
        X_fit = X_clean.values
        y_fit = y_arr

    if task_type == "regression":
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )

    selector = BorutaPy(
        estimator=rf,
        n_estimators=n_estimators,
        max_iter=max_iter,
        random_state=random_state,
    )

    try:
        selector.fit(X_fit, y_fit)
        selected_columns = X_clean.columns[selector.support_]

        if len(selected_columns) == 0:
            return X_clean.iloc[:, 0:0]

        return X_clean[selected_columns]
    except Exception as e:
        logger.warning(
            "Boruta feature selection encountered an error (%s); returning empty feature set to trigger honest fallback.",
            e,
        )
        return X_clean.iloc[:, 0:0]


def tree_importance_selector(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    model_type: Literal["extra_trees", "random_forest"] = "extra_trees",
    n_estimators: int = 50,
    threshold: Union[str, float] = "median",
    task_type: Literal["classification", "regression"] = "classification",
    random_state: int = 42,
) -> pd.DataFrame:
    """Filter features using tree-based importance metrics via SelectFromModel.

    Supports ExtraTrees and RandomForest ensembles for both classification and regression.

    Args:
        X: Feature matrix as pandas DataFrame.
        y: Target classification labels or continuous regression targets.
        model_type: Tree ensemble type ('extra_trees' or 'random_forest').
        n_estimators: Number of decision trees in ensemble.
        threshold: Threshold criterion for SelectFromModel (e.g. 'median', 'mean', or float).
        task_type: Target problem type ('classification' or 'regression').
        random_state: Random state seed for reproducibility.

    Returns:
        Filtered pandas DataFrame containing selected features.
    """
    X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if X_clean.shape[1] == 0 or X_clean.shape[0] == 0 or y is None:
        return X_clean.iloc[:, 0:0]

    y_arr = np.asarray(y)
    if y_arr.size == 0:
        return X_clean.iloc[:, 0:0]
    if y_arr.ndim > 1:
        y_arr = y_arr.ravel()

    if task_type == "classification" and len(np.unique(y_arr)) < 2:
        logger.warning("Tree importance classification requires at least 2 distinct classes; returning empty feature set.")
        return X_clean.iloc[:, 0:0]

    try:
        from sklearn.ensemble import (
            ExtraTreesClassifier,
            ExtraTreesRegressor,
            RandomForestClassifier,
            RandomForestRegressor,
        )
        from sklearn.feature_selection import SelectFromModel

        m_type = str(model_type).lower().replace("-", "_")
        if m_type in ("extra_trees", "extratrees"):
            if task_type == "regression":
                estimator = ExtraTreesRegressor(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=-1,
                )
            else:
                estimator = ExtraTreesClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=-1,
                )
        elif m_type in ("random_forest", "randomforest"):
            if task_type == "regression":
                estimator = RandomForestRegressor(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=-1,
                )
            else:
                estimator = RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=-1,
                )
        else:
            raise ValueError(f"Unsupported model_type '{model_type}'. Must be 'extra_trees' or 'random_forest'.")

        selector = SelectFromModel(estimator=estimator, threshold=threshold)
        selector.fit(X_clean, y_arr)

        return X_clean.iloc[:, selector.get_support()]
    except Exception as e:
        logger.warning(
            "Tree importance selection encountered an error (%s); returning empty feature set to trigger honest fallback.",
            e,
        )
        return X_clean.iloc[:, 0:0]



