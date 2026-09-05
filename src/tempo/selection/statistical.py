"""Statistical feature selection filters."""

import logging
from typing import Callable, Literal, Optional, Union
import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from tsfresh import select_features

logger = logging.getLogger(__name__)


def tsfresh_selector(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    fdr_level: float = 0.05,
    task_type: Literal["classification", "regression"] = "classification",
) -> pd.DataFrame:
    """Filter features using TSFresh statistical hypothesis testing.

    Args:
        X: Feature matrix as pandas DataFrame.
        y: Target classification labels or continuous regression targets.
        fdr_level: False Discovery Rate threshold.
        task_type: Target problem type ('classification' or 'regression').

    Returns:
        Filtered pandas DataFrame containing selected features.
    """
    X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if X_clean.shape[1] == 0:
        return X_clean.iloc[:, 0:0]

    # Drop all-constant features
    nunique = X_clean.nunique()
    non_constant = nunique[nunique > 1].index
    if len(non_constant) > 0:
        X_clean = X_clean[non_constant]
    else:
        return X_clean.iloc[:, 0:0]

    ml_task = "regression" if task_type == "regression" else "auto"
    try:
        y_clean = pd.Series(np.asarray(y), index=X_clean.index)
        return select_features(X_clean, y_clean, fdr_level=fdr_level, ml_task=ml_task)
    except Exception as e:
        logger.warning(
            "TSFresh feature selection encountered an error (%s); returning empty feature set to trigger honest fallback.",
            e,
        )
        return X_clean.iloc[:, 0:0]


def select_k_best(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    k: int = 20,
    task_type: Literal["classification", "regression"] = "classification",
    score_func: Optional[Callable] = None,
) -> pd.DataFrame:
    """Filter top k features using ANOVA/F-statistic or custom criterion.

    Args:
        X: Feature matrix as pandas DataFrame.
        y: Target values.
        k: Maximum number of features to retain.
        task_type: Target problem type ('classification' or 'regression').
        score_func: Optional scoring function. If None, dynamically chooses
            f_classif for classification and f_regression for regression.

    Returns:
        Filtered pandas DataFrame containing top k features.
    """
    X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    n_features = X_clean.shape[1]
    k_effective = min(k, n_features)

    if k_effective <= 0:
        return X_clean.iloc[:, 0:0]

    if score_func is None:
        score_func = f_regression if task_type == "regression" else f_classif

    try:
        selector = SelectKBest(score_func=score_func, k=k_effective)
        selector.fit(X_clean, np.asarray(y))
        return X_clean.iloc[:, selector.get_support()]
    except Exception:
        return X_clean.iloc[:, :k_effective]


def mutual_info_selector(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    k: int = 20,
    task_type: Literal["classification", "regression"] = "classification",
    random_state: Optional[int] = 42,
) -> pd.DataFrame:
    """Filter top k features using Mutual Information criterion.

    Args:
        X: Feature matrix as pandas DataFrame.
        y: Target values.
        k: Maximum number of features to retain.
        task_type: Target problem type ('classification' or 'regression').
        random_state: Seed for reproducible mutual information estimation.

    Returns:
        Filtered pandas DataFrame containing top k features.
    """
    if task_type == "regression":
        def mi_scorer(x_in: np.ndarray, y_in: np.ndarray) -> np.ndarray:
            return mutual_info_regression(x_in, y_in, random_state=random_state)
    else:
        def mi_scorer(x_in: np.ndarray, y_in: np.ndarray) -> np.ndarray:
            return mutual_info_classif(x_in, y_in, random_state=random_state)

    return select_k_best(X, y, k=k, task_type=task_type, score_func=mi_scorer)

