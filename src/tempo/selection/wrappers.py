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


