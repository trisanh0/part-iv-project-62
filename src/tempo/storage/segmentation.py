"""Sliding-window segmentation utilities for continuous time-series data."""

from typing import List, Optional, Tuple
import polars as pl
import numpy as np


def segment_time_series(
    df: pl.DataFrame,
    window_size: int,
    stride: int,
    time_col: str = "time",
    feature_cols: Optional[List[str]] = None,
    label_col: Optional[str] = None,
    label_strategy: str = "majority_vote",
) -> Tuple[pl.DataFrame, Optional[pl.DataFrame]]:
    """Segment continuous time-series DataFrame into discrete windowed sequences.

    Args:
        df: Input Polars DataFrame containing raw continuous temporal data.
        window_size: Number of temporal steps per window segment.
        stride: Step stride offset between consecutive sliding windows.
        time_col: Column name identifying temporal order.
        feature_cols: List of signal feature column names to segment.
        label_col: Optional column name containing target classification labels.
        label_strategy: Target label aggregation strategy ('majority_vote', 'last', 'any_positive').

    Returns:
        Tuple of (df_ts, df_targets), where df_ts is keyed by ('sequence_id', 'step').
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    if feature_cols is None:
        exclude_cols = {time_col}
        if label_col:
            exclude_cols.add(label_col)
        feature_cols = [c for c in df.columns if c not in exclude_cols]

    n_rows = df.height
    if n_rows < window_size:
        raise ValueError(f"Dataset length {n_rows} smaller than window_size {window_size}")

    n_windows = (n_rows - window_size) // stride + 1

    ts_frames = []
    target_records = []

    # Process signal values into structured sequence windows
    feature_matrices = {c: df[c].to_numpy() for c in feature_cols}
    if label_col and label_col in df.columns:
        label_array = df[label_col].to_numpy()
    else:
        label_array = None

    for i in range(n_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size

        window_dict = {
            "sequence_id": np.full(window_size, i, dtype=np.int32),
            "step": np.arange(window_size, dtype=np.int32),
        }
        for col in feature_cols:
            window_dict[col] = feature_matrices[col][start_idx:end_idx].astype(np.float32)

        ts_frames.append(pl.DataFrame(window_dict))

        if label_array is not None:
            window_labels = label_array[start_idx:end_idx]
            if label_strategy == "last":
                target_val = window_labels[-1]
            elif label_strategy == "any_positive":
                target_val = 1 if np.any(window_labels > 0) else 0
            elif label_strategy == "majority_vote":
                vals, counts = np.unique(window_labels, return_counts=True)
                target_val = vals[np.argmax(counts)]
            else:
                raise ValueError(f"Unknown label_strategy: {label_strategy}")

            target_records.append({"sequence_id": i, "target": int(target_val)})

    df_ts = pl.concat(ts_frames)

    if target_records:
        df_targets = pl.DataFrame(target_records).with_columns([
            pl.col("sequence_id").cast(pl.Int32),
            pl.col("target").cast(pl.Int32),
        ])
    else:
        df_targets = None

    return df_ts, df_targets
