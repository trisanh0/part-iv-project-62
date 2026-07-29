from tsfresh import select_features

from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif
)


def tsfresh_selector(X, y):

    return select_features(X, y)


def select_k_best(X, y):

    selector = SelectKBest(
        mutual_info_classif,
        k=min(20, X.shape[1])
    )

    selector.fit(X, y)

    return X.iloc[:, selector.get_support()]