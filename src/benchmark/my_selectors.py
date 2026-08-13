from tsfresh import select_features
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
    mutual_info_regression
)


def tsfresh_selector(X, y, **kwargs):

    return select_features(X, y)


def select_k_best(X, y, data_type):

    if data_type == "classification":
        info = mutual_info_classif
    elif data_type == "regression":
        info = mutual_info_regression

    selector = SelectKBest(
        info,
        k=min(20, X.shape[1])
    )

    selector.fit(X, y)

    return X.iloc[:, selector.get_support()]


def boruta_selector(X, y, data_type):

    """
    Boruta feature selection.

    Uses Random Forest feature importance to identify
    features that are statistically relevant.

    Input:
        X: pandas DataFrame
        y: numpy array

    Output:
        pandas DataFrame
    """
    if data_type == "classification":
        rf = RandomForestClassifier(
            n_estimators=50,
            random_state=42
        )
    elif data_type == "regression":
        rf = RandomForestRegressor(
            n_estimators=10,
            max_depth=5,
            random_state=0)
        
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