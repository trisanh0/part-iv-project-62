import numpy as np
import pandas as pd
import getml
import tsfel
from copy import deepcopy
from sktime.transformations.series.hurst import HurstExponentTransformer
from tsfeatures import tsfeatures
# import pycatch22

from tsfresh import extract_features
from tsfresh.feature_extraction import (
    MinimalFCParameters,
    EfficientFCParameters,
    ComprehensiveFCParameters,
)


def statistical(X):

    return pd.DataFrame({
        "mean": np.mean(X, axis=1),
        "std": np.std(X, axis=1),
        "min": np.min(X, axis=1),
        "max": np.max(X, axis=1),
        "energy": np.sum(X**2, axis=1),
    })



def fft_parameters(n):
    return [
        {"coeff": i, "attr": attr}
        for i in range(n)
        for attr in ["real", "imag", "abs", "angle"]
    ]


def tsfresh_extractor(
    X,
    parameter_set="minimal",
    fft_coefficients=None,
):
    """
    parameter_set:
        "minimal"
        "efficient"
        "comprehensive"

    fft_coefficients:
        None = use tsfresh defaults
        integer = override fft_coefficient calculation
    """

    if parameter_set == "minimal":
        settings = MinimalFCParameters()

    elif parameter_set == "efficient":
        settings = EfficientFCParameters()

    elif parameter_set == "comprehensive":
        settings = ComprehensiveFCParameters()

    else:
        raise ValueError(parameter_set)

    settings = deepcopy(settings)

    if (
        fft_coefficients is not None
        and parameter_set != "minimal"
    ):
        settings["fft_coefficient"] = fft_parameters(
            fft_coefficients
        )

    return extract_features(
        X,
        column_id="id",
        column_sort="time",
        default_fc_parameters=settings,
    )


def tsfel_extractor(X):

    cfg = tsfel.get_features_by_domain(
        ["statistical"]
    )

    # remove duplicate feature definitions
    seen = set()
    clean_cfg = {}

    for domain, features in cfg.items():
        clean_cfg[domain] = {}

        for name, params in features.items():
            if name not in seen:
                clean_cfg[domain][name] = params
                seen.add(name)

    features = []

    for series in X:

        df = tsfel.time_series_features_extractor(
            clean_cfg,
            series,
            fs=1,
            verbose=0
        )

        features.append(df.iloc[0])

    return pd.DataFrame(features)


def tsfeatures_extractor(X):
    X = np.asarray(X)

    n_series, n_points = X.shape

    df = pd.DataFrame({
        "unique_id": np.repeat(np.arange(n_series), n_points),
        "ds": np.tile(np.arange(n_points), n_series),
        "y": X.ravel()
    })

    return tsfeatures(df, freq=1)




