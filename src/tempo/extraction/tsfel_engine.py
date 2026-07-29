"""TSFEL extraction engine wrapper."""

import pandas as pd
import tsfel


def tsfel_extractor(
    sequences: list | pd.DataFrame,
    domain: str = "statistical",
    fs: float = 1.0,
) -> pd.DataFrame:
    """Extract features from time-series sequences using TSFEL.

    Args:
        sequences: List of 1D arrays/Series or a 2D matrix of time series.
        domain: Feature domain ("statistical", "spectral", "temporal").
        fs: Sampling frequency in Hz.

    Returns:
        pandas DataFrame of extracted TSFEL features.
    """
    cfg = tsfel.get_features_by_domain([domain])

    # Remove duplicate feature definitions
    seen = set()
    clean_cfg = {}
    for dom, features in cfg.items():
        clean_cfg[dom] = {}
        for name, params in features.items():
            if name not in seen:
                clean_cfg[dom][name] = params
                seen.add(name)

    extracted = []
    for series in sequences:
        df_feat = tsfel.time_series_features_extractor(
            clean_cfg,
            series,
            fs=fs,
            verbose=0,
        )
        extracted.append(df_feat.iloc[0])

    return pd.DataFrame(extracted)
