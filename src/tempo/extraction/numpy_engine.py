"""Direct NumPy-bound feature extraction engine.

Bypasses Pandas DataFrame unpivoting to minimise heap memory allocation.
"""

import numpy as np
import pandas as pd


def numpy_statistical_extractor(X: np.ndarray) -> pd.DataFrame:
    """Extract summary statistical features using direct 2D NumPy operations.

    Args:
        X: 2D NumPy array of shape (n_samples, sequence_length).

    Returns:
        pandas DataFrame containing summary features (mean, std, min, max, energy).
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X.shape}")

    return pd.DataFrame({
        "mean": np.mean(X, axis=1),
        "std": np.std(X, axis=1),
        "min": np.min(X, axis=1),
        "max": np.max(X, axis=1),
        "energy": np.sum(X**2, axis=1),
    })
