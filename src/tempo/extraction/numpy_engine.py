from typing import Optional, Sequence
import numpy as np
import pandas as pd


def numpy_statistical_extractor(
    X: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Extract summary statistical features using direct NumPy operations.

    Args:
        X: 2D or 3D NumPy array of shape (n_samples, sequence_length) or (n_samples, sequence_length, n_channels).
        feature_names: Optional channel names.

    Returns:
        pandas DataFrame containing summary features (mean, std, min, max, energy).
    """
    if X.ndim == 2:
        if feature_names is None:
            return pd.DataFrame({
                "mean": np.mean(X, axis=1),
                "std": np.std(X, axis=1),
                "min": np.min(X, axis=1),
                "max": np.max(X, axis=1),
                "energy": np.sum(X**2, axis=1),
            })
        X = X[:, :, np.newaxis]
    if X.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {X.shape}")

    n_samples, seq_len, n_channels = X.shape
    cols = {}
    for ch in range(n_channels):
        prefix = feature_names[ch] if feature_names and ch < len(feature_names) else f"ch_{ch}"
        ch_slice = X[:, :, ch]
        cols[f"{prefix}__mean"] = np.mean(ch_slice, axis=1)
        cols[f"{prefix}__std"] = np.std(ch_slice, axis=1)
        cols[f"{prefix}__min"] = np.min(ch_slice, axis=1)
        cols[f"{prefix}__max"] = np.max(ch_slice, axis=1)
        cols[f"{prefix}__energy"] = np.sum(ch_slice**2, axis=1)

    return pd.DataFrame(cols)

