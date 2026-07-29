"""Numba JIT compiled feature extraction engine.

Compiles feature extraction loops into native machine code using @numba.njit.
Passes raw C memory pointers to eliminate Python interpreter overhead and object
allocations while retaining high memory efficiency.
"""

from typing import Tuple
import numba
import numpy as np
import pandas as pd



@numba.njit(fastmath=True, parallel=False)
def _numba_statistical_kernel(X: np.ndarray) -> np.ndarray:
    """Numba-compiled single-pass statistical feature calculation over 2D matrix.

    Args:
        X: 2D array of shape (N, T) representing N series of length T.

    Returns:
        2D array of shape (N, 9) containing:
        [mean, std, variance, min, max, energy, abs_energy, skewness, kurtosis]
    """
    N, T = X.shape
    out = np.empty((N, 9), dtype=np.float64)

    for i in range(N):
        # Accumulate moments in a single C-speed pass over row i
        s = 0.0
        sq_s = 0.0
        abs_s = 0.0
        val_min = X[i, 0]
        val_max = X[i, 0]

        for t in range(T):
            v = X[i, t]
            s += v
            sq_s += v * v
            abs_s += abs(v)
            if v < val_min:
                val_min = v
            if v > val_max:
                val_max = v

        mean_val = s / T
        variance_val = max(0.0, (sq_s / T) - (mean_val * mean_val))
        std_val = np.sqrt(variance_val)

        # Higher-order moments: skewness and kurtosis
        m3 = 0.0
        m4 = 0.0
        for t in range(T):
            diff = X[i, t] - mean_val
            m3 += diff * diff * diff
            m4 += diff * diff * diff * diff

        m3 /= T
        m4 /= T

        if std_val > 1e-12:
            skew_val = m3 / (std_val ** 3)
            kurt_val = (m4 / (std_val ** 4)) - 3.0
        else:
            skew_val = 0.0
            kurt_val = 0.0

        out[i, 0] = mean_val
        out[i, 1] = std_val
        out[i, 2] = variance_val
        out[i, 3] = val_min
        out[i, 4] = val_max
        out[i, 5] = sq_s
        out[i, 6] = abs_s
        out[i, 7] = skew_val
        out[i, 8] = kurt_val

    return out


def _vectorized_fft_kernel(X: np.ndarray, n_coeffs: int = 25) -> Tuple[np.ndarray, list[str]]:
    """Vectorized 2D FFT calculation across all sequences simultaneously.

    Args:
        X: 2D array of shape (N, T).
        n_coeffs: Number of Fourier coefficients to retain per attribute.

    Returns:
        Tuple of (2D feature matrix of shape (N, 4 * n_coeffs), column names).
    """
    N, T = X.shape
    fft_vals = np.fft.rfft(X, axis=1)

    max_available = fft_vals.shape[1]
    n_k = min(n_coeffs, max_available)

    fft_sub = fft_vals[:, :n_k]

    real_part = np.real(fft_sub)
    imag_part = np.imag(fft_sub)
    abs_part = np.abs(fft_sub)
    angle_part = np.angle(fft_sub)

    matrix = np.hstack([real_part, imag_part, abs_part, angle_part])

    col_names = []
    for attr in ["real", "imag", "abs", "angle"]:
        for k in range(n_k):
            col_names.append(f"fft_coefficient__coeff_{k}__attr_{attr}")

    return matrix, col_names


def numba_feature_extractor(
    X: np.ndarray,
    raw_feature_names: list[str] | None = None,
    n_fft_coeffs: int = 25,
) -> pd.DataFrame:
    """Numba-compiled feature extraction engine.

    Args:
        X: 2D NumPy array of shape (N, T) or 3D array of shape (N, T, P).
        raw_feature_names: Optional names for raw input variables.
        n_fft_coeffs: Number of FFT coefficients to extract.

    Returns:
        Pandas DataFrame containing extracted features.
    """
    if X.ndim == 2:
        X_3d = X[:, :, np.newaxis]
    elif X.ndim == 3:
        X_3d = X
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {X.shape}")

    N, T, P = X_3d.shape

    if raw_feature_names is None:
        raw_feature_names = [f"var_{j}" for j in range(P)]

    stat_names = [
        "mean", "std", "variance", "min", "max", "energy",
        "abs_energy", "skewness", "kurtosis"
    ]

    all_matrices = []
    all_col_names = []

    for j in range(P):
        var_name = raw_feature_names[j]
        signal_2d = X_3d[:, :, j]

        # Pass 1: Numba JIT compiled single-pass statistical kernel
        stats_mat = _numba_statistical_kernel(signal_2d)
        for idx, s_name in enumerate(stat_names):
            all_col_names.append(f"{var_name}__{s_name}")
        all_matrices.append(stats_mat)

        # Pass 2: Vectorized 2D FFT kernel
        if n_fft_coeffs > 0:
            fft_mat, fft_cols = _vectorized_fft_kernel(signal_2d, n_coeffs=n_fft_coeffs)
            for f_col in fft_cols:
                all_col_names.append(f"{var_name}__{f_col}")
            all_matrices.append(fft_mat)

    combined_matrix = np.hstack(all_matrices)
    return pd.DataFrame(combined_matrix, columns=all_col_names)
