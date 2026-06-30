import pandas as pd
import polars as pl
import time
import numpy as np

from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters

from sklearn.feature_selection import SelectFdr, f_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def time_perf(name, fn):
    start = time.perf_counter()
    out = fn()
    end = time.perf_counter()
    print(f"{name}: {end - start:.2f} sec")
    return out


# =========================
# LOAD RAW
# =========================
df_raw = pd.read_csv("X_train.txt", sep=r"\s+", header=None)
df_raw = df_raw.iloc[0:200, :]
df_raw["id"] = df_raw.index

df_test_raw = pd.read_csv("X_test.txt", sep=r"\s+", header=None)
df_test_raw = df_test_raw.iloc[0:200, :]   # FIXED
df_test_raw["id"] = df_test_raw.index

y = pd.read_csv("y_train.txt", header=None).iloc[0:200, :].squeeze()
y_test = pd.read_csv("y_test.txt", header=None).iloc[0:200, :].squeeze()

# =========================
# SAMPLE
# =========================
df_raw, _, y, _ = train_test_split(
    df_raw,
    y,
    test_size=0.5,
    random_state=42,
    stratify=y
)

df_test_raw, _, y_test, _ = train_test_split(
    df_test_raw,
    y_test,
    test_size=0.5,
    random_state=42,
    stratify=y_test
)

# =========================
# MELT
# =========================
df = df_raw.melt(
    id_vars="id",
    var_name="time",
    value_name="value"
)

df_test = df_test_raw.melt(
    id_vars="id",
    var_name="time",
    value_name="value"
)

fc_params = EfficientFCParameters()


def run_pandas_tsfresh():

    X = extract_features(
        df,
        column_id="id",
        column_sort="time",
        default_fc_parameters=fc_params,
        n_jobs=0
    )

    X_test = extract_features(
        df_test,
        column_id="id",
        column_sort="time",
        default_fc_parameters=fc_params,
        n_jobs=0
    )

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.reindex(columns=X.columns).fillna(0)

    selector = SelectFdr(f_classif, alpha=0.5)

    X_sel = selector.fit_transform(X, y)
    X_test_sel = selector.transform(X_test)

    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3
    )

    model.fit(X_sel, y)
    preds = model.predict(X_test_sel)

    return accuracy_score(y_test, preds)


def run_polars_tsfresh():

    # convert to polars
    train_pl = pl.from_pandas(df)
    test_pl = pl.from_pandas(df_test)

    # NO UNPIVOT HERE
    X = extract_features(
        train_pl.to_pandas(),
        column_id="id",
        column_sort="time",
        default_fc_parameters=fc_params,
        n_jobs=0
    )

    X_test = extract_features(
        test_pl.to_pandas(),
        column_id="id",
        column_sort="time",
        default_fc_parameters=fc_params,
        n_jobs=0
    )

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.reindex(columns=X.columns).fillna(0)

    selector = SelectFdr(f_classif, alpha=0.5)

    X_sel = selector.fit_transform(X, y)
    X_test_sel = selector.transform(X_test)

    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3
    )

    model.fit(X_sel, y)
    preds = model.predict(X_test_sel)

    return accuracy_score(y_test, preds)


if __name__ == "__main__":

    acc1 = time_perf("Pandas + tsfresh", run_pandas_tsfresh)
    acc2 = time_perf("Polars + tsfresh", run_polars_tsfresh)

    print(acc1, acc2)