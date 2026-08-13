"""Wrapper-based feature selection methods."""

from typing import Union
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

try:
    from boruta import BorutaPy
    _HAS_BORUTA = True
except ImportError:
    BorutaPy = None
    _HAS_BORUTA = False


def boruta_selector(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    n_estimators: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """Select features using Boruta wrapper with Random Forest importance.

    Args:
        X: Feature matrix as pandas DataFrame.
        y: Target classification labels.
        n_estimators: Number of decision trees.
        random_state: Seed for reproducibility.

    Returns:
        Filtered pandas DataFrame containing selected features.
    """
    if not _HAS_BORUTA:
        raise ImportError("boruta package is required for boruta_selector. Install with pip install boruta.")

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )

    selector = BorutaPy(
        estimator=rf,
        n_estimators="auto",
        random_state=random_state,
    )

    selector.fit(X.values, y)
    selected_columns = X.columns[selector.support_]

    return X[selected_columns]
