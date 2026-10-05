from tsfresh import select_features

from boruta import BorutaPy

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)

from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    SelectFromModel,
)

from sklearn.linear_model import LogisticRegression, Lasso


# ============================================================
# TSFresh
# ============================================================

def tsfresh_selector(X, y, **kwargs):

    return select_features(X, y)


# ============================================================
# SelectKBest + Mutual Information
# ============================================================

def select_k_best(X, y, data_type):

    if data_type == "classification":
        info = mutual_info_classif

    elif data_type in ("regression", "inverse_regression"):
        info = mutual_info_regression

    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

    selector = SelectKBest(
        info,
        k=min(20, X.shape[1])
    )

    selector.fit(X, y)

    return X.iloc[:, selector.get_support()]


# ============================================================
# 1. F-TEST
# ============================================================

def f_test_selector(X, y, data_type):

    if data_type == "classification":
        score_func = f_classif

    elif data_type in ("regression", "inverse_regression"):
        score_func = f_regression

    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

    selector = SelectKBest(
        score_func,
        k=min(20, X.shape[1])
    )

    selector.fit(X, y)

    return X.iloc[:, selector.get_support()]


# ============================================================
# 2. VARIANCE THRESHOLD
# ============================================================

def variance_threshold_selector(X, y, data_type):

    # Remove features with zero variance.
    #
    # This is deliberately unsupervised: y is not required.
    selector = VarianceThreshold(
        threshold=0.0
    )

    X_selected = selector.fit_transform(X)

    selected_columns = X.columns[
        selector.get_support()
    ]

    return X.loc[:, selected_columns]


# ============================================================
# 3. EXTRA TREES FEATURE IMPORTANCE
# ============================================================

def extra_trees_selector(X, y, data_type):

    if data_type == "classification":

        model = ExtraTreesClassifier(
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        )

    elif data_type in ("regression", "inverse_regression"):

        model = ExtraTreesRegressor(
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        )

    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

    selector = SelectFromModel(
        model,
        threshold="median"
    )

    selector.fit(X, y)

    return X.loc[:, selector.get_support()]


# ============================================================
# 4. L1 FEATURE SELECTION
# ============================================================

def l1_selector(X, y, data_type):

    if data_type == "classification":

        model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=1.0,
            random_state=42,
            max_iter=1000
        )

    elif data_type in ("regression", "inverse_regression"):

        model = Lasso(
            alpha=0.01,
            max_iter=5000,
            random_state=42
        )

    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

    selector = SelectFromModel(
        model,
        threshold="mean"
    )

    selector.fit(X, y)

    return X.loc[:, selector.get_support()]


# ============================================================
# 5. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

def random_forest_selector(X, y, data_type):

    if data_type == "classification":

        model = RandomForestClassifier(
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        )

    elif data_type in ("regression", "inverse_regression"):

        model = RandomForestRegressor(
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        )

    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

    selector = SelectFromModel(
        model,
        threshold="median"
    )

    selector.fit(X, y)

    return X.loc[:, selector.get_support()]


# ============================================================
# BORUTA
# ============================================================

def boruta_selector(X, y, data_type):

    if data_type == "classification":

        rf = RandomForestClassifier(
            n_estimators=50,
            random_state=42
        )

    elif data_type in ("regression", "inverse_regression"):

        rf = RandomForestRegressor(
            n_estimators=10,
            max_depth=5,
            random_state=0
        )

    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

    selector = BorutaPy(
        estimator=rf,
        n_estimators="auto",
        random_state=42,
        max_iter=10
    )

    selector.fit(
        X.values,
        y
    )

    selected_columns = X.columns[
        selector.support_
    ]

    return X[selected_columns]