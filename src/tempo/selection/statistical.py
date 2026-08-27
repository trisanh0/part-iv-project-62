"""Statistical feature selection filters."""

import pandas as pd
import numpy as np
from tsfresh import select_features
from sklearn.feature_selection import SelectKBest, mutual_info_classif


def tsfresh_selector(X: pd.DataFrame, y: pd.Series | np.ndarray) -> pd.DataFrame:
    """Filter features using TSFresh statistical hypothesis testing.

    Args:
        X: Feature matrix as pandas DataFrame.
        y: Classification target values.

    Returns:
        Filtered pandas DataFrame containing selected features.
    """
    X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Drop all-constant features
    nunique = X_clean.nunique()
    non_constant = nunique[nunique > 1].index
    if len(non_constant) > 0:
        X_clean = X_clean[non_constant]
    try:
        return select_features(X_clean, y)
    except Exception:
        return X_clean


def select_k_best(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    k: int = 20,
) -> pd.DataFrame:
    """Filter top k features using ANOVA/Mutual Information criterion.

    Args:
        X: Feature matrix as pandas DataFrame.
        y: Target values.
        k: Maximum number of features to retain.

    Returns:
        Filtered pandas DataFrame containing top k features.
    """
    X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    n_features = X_clean.shape[1]
    k_effective = min(k, n_features)

    if k_effective == 0:
        return X_clean

    try:
        selector = SelectKBest(score_func=mutual_info_classif, k=k_effective)
        selector.fit(X_clean, y)
        return X_clean.iloc[:, selector.get_support()]
    except Exception:
        return X_clean.iloc[:, :k_effective]
