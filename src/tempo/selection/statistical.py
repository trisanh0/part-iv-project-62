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


def variance_threshold_selector(
    X: pd.DataFrame,
    threshold: float = 0.0,
    y: Optional[Union[pd.Series, np.ndarray]] = None,
) -> pd.DataFrame:
    """Filter low-variance features using VarianceThreshold.

    Args:
        X: Feature matrix as pandas DataFrame.
        threshold: Variance threshold below which features are discarded.
            Default 0.0 filters zero-variance (constant) features.
        y: Optional target array (ignored, provided for selector signature compatibility).

    Returns:
        Filtered pandas DataFrame containing selected features.
    """
    X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if X_clean.shape[1] == 0 or X_clean.shape[0] == 0:
        return X_clean.iloc[:, 0:0]

    try:
        from sklearn.feature_selection import VarianceThreshold

        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X_clean)
        return X_clean.iloc[:, selector.get_support()]
    except ValueError as e:
        # Sklearn raises ValueError when 0 features meet the specified variance threshold
        if "No feature in X meets the variance threshold" in str(e):
            return X_clean.iloc[:, 0:0]
        logger.warning("VarianceThreshold feature selection error: %s", e)
        return X_clean.iloc[:, 0:0]
    except Exception as e:
        logger.warning(
            "VarianceThreshold feature selection encountered an error (%s); returning empty feature set.",
            e,
        )
        return X_clean.iloc[:, 0:0]


def l1_selector(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    C: float = 1.0,
    alpha: float = 0.01,
    task_type: Literal["classification", "regression"] = "classification",
    random_state: int = 42,
) -> pd.DataFrame:
    """Filter features using L1-penalized linear models via SelectFromModel.

    Standardizes feature inputs prior to optimization to ensure scale-agnostic
    regularization. Uses L1 Logistic Regression (liblinear) for classification tasks and
    Lasso for continuous regression tasks.

    Args:
        X: Feature matrix as pandas DataFrame.
        y: Target classification labels or continuous regression targets.
        C: Inverse regularization strength for L1 Logistic Regression.
        alpha: Regularization penalty for Lasso regression.
        task_type: Target problem type ('classification' or 'regression').
        random_state: Random state seed for reproducibility.

    Returns:
        Filtered pandas DataFrame containing selected non-zero features.
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
        logger.warning("L1 classification requires at least 2 distinct classes; returning empty feature set.")
        return X_clean.iloc[:, 0:0]

    try:
        import warnings
        from sklearn.feature_selection import SelectFromModel
        from sklearn.linear_model import Lasso, LogisticRegression

        # Standardize features numerically to avoid scale bias in L1 penalisation
        vals = X_clean.to_numpy(dtype=np.float64)
        means = np.mean(vals, axis=0)
        stds = np.std(vals, axis=0)
        scales = np.where(stds > 0, stds, 1.0)
        X_norm = (vals - means) / scales

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if task_type == "classification":
                n_classes = len(np.unique(y_arr))
                solver_name = "saga" if n_classes > 2 else "liblinear"
                estimator = LogisticRegression(
                    penalty="l1",
                    solver=solver_name,
                    C=C,
                    max_iter=500,
                    random_state=random_state,
                )
            else:
                estimator = Lasso(
                    alpha=alpha,
                    random_state=random_state,
                )

            selector = SelectFromModel(estimator=estimator)
            selector.fit(X_norm, y_arr)

        return X_clean.iloc[:, selector.get_support()]
    except Exception as e:
        logger.warning(
            "L1 feature selection encountered an error (%s); returning empty feature set to trigger honest fallback.",
            e,
        )
        return X_clean.iloc[:, 0:0]


