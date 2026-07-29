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
    return select_features(X, y)


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
    n_features = X.shape[1]
    k_effective = min(k, n_features)

    selector = SelectKBest(score_func=mutual_info_classif, k=k_effective)
    selector.fit(X, y)

    return X.iloc[:, selector.get_support()]
