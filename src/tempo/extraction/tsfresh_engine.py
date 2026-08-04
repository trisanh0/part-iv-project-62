"""TSFresh extraction engine wrappers with Fourier coefficient truncation."""

from copy import deepcopy
from typing import Optional
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import (
    MinimalFCParameters,
    EfficientFCParameters,
    ComprehensiveFCParameters,
)


def fft_parameters(n_coeffs: int):
    """Generate TSFresh fft_coefficient parameter list for n_coeffs.

    Args:
        n_coeffs: Number of Fourier coefficients to retain per attribute.

    Returns:
        List of dicts specifying coefficient indices and attributes.
    """
    return [
        {"coeff": i, "attr": attr}
        for i in range(n_coeffs)
        for attr in ["real", "imag", "abs", "angle"]
    ]


def tsfresh_extractor(
    df: pd.DataFrame,
    parameter_set: str = "minimal",
    fft_coefficients: Optional[int] = None,
    column_id: Optional[str] = None,
    column_sort: Optional[str] = None,
) -> pd.DataFrame:
    """Extract features using TSFresh with optional Fourier coefficient limit.

    Args:
        df: Input pandas DataFrame in long format.
        parameter_set: Preset parameter set ("minimal", "efficient", "comprehensive").
        fft_coefficients: Optional maximum number of Fourier coefficients.
        column_id: Column name identifying time-series entities.
        column_sort: Column name identifying temporal order.

    Returns:
        pandas DataFrame of extracted features.
    """
    if column_id is None:
        if "sequence_id" in df.columns:
            column_id = "sequence_id"
        elif "id" in df.columns:
            column_id = "id"
        else:
            raise KeyError("Neither 'sequence_id' nor 'id' column found in DataFrame.")

    if column_sort is None:
        if "step" in df.columns:
            column_sort = "step"
        elif "time" in df.columns:
            column_sort = "time"
        else:
            column_sort = None

    if parameter_set == "minimal":
        settings = MinimalFCParameters()
    elif parameter_set == "efficient":
        settings = EfficientFCParameters()
    elif parameter_set == "comprehensive":
        settings = ComprehensiveFCParameters()
    else:
        raise ValueError(f"Unknown parameter_set: {parameter_set}")

    settings = deepcopy(settings)

    if fft_coefficients is not None and parameter_set != "minimal":
        settings["fft_coefficient"] = fft_parameters(fft_coefficients)

    return extract_features(
        df,
        column_id=column_id,
        column_sort=column_sort,
        default_fc_parameters=settings,
        disable_progressbar=True,
    )
