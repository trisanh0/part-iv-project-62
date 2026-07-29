"""Wrapper-based feature selection methods."""

import pandas as pd
import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier


def boruta_selector(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
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
