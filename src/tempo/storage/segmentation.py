"""Sliding-window segmentation utilities for continuous time-series data."""

from typing import List, Optional, Tuple
import polars as pl
import numpy as np


def segment_time_series(
    df: pl.DataFrame,
    window_size: int,
    stride: Optional[int] = None,
    time_col: str = "time",
    feature_cols: Optional[List[str]] = None,
    label_col: Optional[str] = None,
    label_strategy: str = "majority_vote",
    group_col: Optional[str] = None,
    forecast_horizon: Optional[int] = None,
) -> Tuple[pl.DataFrame, Optional[pl.DataFrame]]:
    """Segment continuous time-series DataFrame into discrete windowed sequences.

    Args:
        df: Input Polars DataFrame containing raw continuous temporal data.
        window_size: Number of temporal steps per window segment (history window W).
        stride: Step stride offset between consecutive sliding windows. If None,
            defaults to window_size (strictly non-overlapping windows to eliminate data leakage).
        time_col: Column name identifying temporal order.
        feature_cols: List of signal feature column names to segment.
        label_col: Optional column name containing target classification labels or continuous values.
        label_strategy: Target label aggregation strategy ('majority_vote', 'last', 'mean', 'any_positive').
        group_col: Optional column name containing subject/entity grouping metadata.
        forecast_horizon: Optional number of future steps H to forecast. When specified, targets
            are structured as multi-step forecast vectors rather than single scalar values.

    Returns:
        Tuple of (df_ts, df_targets), where df_ts is keyed by ('sequence_id', 'step').
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if stride is None:
        stride = window_size
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if forecast_horizon is not None and forecast_horizon <= 0:
        raise ValueError(f"forecast_horizon must be positive, got {forecast_horizon}")

    if feature_cols is None:
        exclude_cols = {time_col}
        if label_col:
            exclude_cols.add(label_col)
        if group_col:
            exclude_cols.add(group_col)
        feature_cols = [c for c in df.columns if c not in exclude_cols]

    n_rows = df.height
    min_required = window_size + (forecast_horizon if forecast_horizon is not None else 0)
    if n_rows < min_required:
        raise ValueError(f"Dataset length {n_rows} smaller than required window length {min_required}")

    if forecast_horizon is not None:
        n_windows = (n_rows - window_size - forecast_horizon) // stride + 1
    else:
        n_windows = (n_rows - window_size) // stride + 1

    ts_frames = []
    target_records = []

    # Process signal values into structured sequence windows
    feature_matrices = {c: df[c].to_numpy() for c in feature_cols}
    if label_col and label_col in df.columns:
        label_series = df[label_col]
        is_continuous = label_series.dtype.is_float() or forecast_horizon is not None
        label_array = label_series.to_numpy()
    else:
        label_array = None
        is_continuous = False

    group_array = df[group_col].to_numpy() if group_col and group_col in df.columns else None

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
            if forecast_horizon is not None:
                future_targets = label_array[end_idx:end_idx + forecast_horizon]
                target_entry = {
                    "sequence_id": i,
                    "target": [float(v) for v in future_targets],
                }
                for h_idx, v in enumerate(future_targets):
                    target_entry[f"target_{h_idx}"] = float(v)
            else:
                window_labels = label_array[start_idx:end_idx]
                if label_strategy == "last":
                    target_val = window_labels[-1]
                elif label_strategy == "mean":
                    target_val = float(np.nanmean(window_labels))
                elif label_strategy == "any_positive":
                    target_val = 1 if np.any(window_labels > 0) else 0
                elif label_strategy == "majority_vote":
                    if is_continuous:
                        target_val = float(np.nanmean(window_labels))
                    else:
                        vals, counts = np.unique(window_labels, return_counts=True)
                        target_val = vals[np.argmax(counts)]
                else:
                    raise ValueError(f"Unknown label_strategy: {label_strategy}")

                target_entry = {
                    "sequence_id": i,
                    "target": float(target_val) if is_continuous else int(target_val),
                }

            if group_array is not None:
                target_entry["group"] = group_array[start_idx:end_idx][-1]
                if group_col and group_col != "group":
                    target_entry[group_col] = group_array[start_idx:end_idx][-1]

            target_records.append(target_entry)

    df_ts = pl.concat(ts_frames)

    if target_records:
        df_targets = pl.DataFrame(target_records)
        casts = [pl.col("sequence_id").cast(pl.Int32)]
        if forecast_horizon is not None:
            for h_idx in range(forecast_horizon):
                if f"target_{h_idx}" in df_targets.columns:
                    casts.append(pl.col(f"target_{h_idx}").cast(pl.Float32))
        else:
            target_cast = pl.Float64 if is_continuous else pl.Int32
            casts.append(pl.col("target").cast(target_cast))

        if "group" in df_targets.columns:
            casts.append(pl.col("group"))
        if group_col and group_col in df_targets.columns and group_col != "group":
            casts.append(pl.col(group_col))
        df_targets = df_targets.with_columns(casts)
    else:
        df_targets = None

    return df_ts, df_targets


def segment_forecasting_series(
    df: pl.DataFrame,
    history_len: int,
    forecast_horizon: int,
    stride: Optional[int] = 1,
    time_col: str = "time",
    feature_cols: Optional[List[str]] = None,
    target_col: Optional[str] = None,
    group_col: Optional[str] = None,
    test_size: Optional[float] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Segment continuous time series into strictly causal history windows and forecast horizons.

    Guarantees chronological temporal integrity:
    When test_size is provided, the timeline is chronologically partitioned into a historical training
    region and a held-out testing region. Training targets are strictly within the training region, and
    testing targets are strictly within the held-out future.

    Args:
        df: Polars DataFrame containing temporal sequences.
        history_len: History observation window length W.
        forecast_horizon: Forecast target horizon length H.
        stride: Step stride between consecutive sliding windows. Defaults to 1.
        time_col: Column name identifying temporal order.
        feature_cols: Signal column names to include as features.
        target_col: Target column name to forecast. If None, defaults to first feature column.
        group_col: Column name identifying distinct series/entities.
        test_size: Chronological test fraction (e.g. 0.2). If None, segments all available points.

    Returns:
        Tuple of (df_ts, df_targets) in standardized SDF format.
    """
    if history_len <= 0:
        raise ValueError(f"history_len must be positive, got {history_len}")
    if forecast_horizon <= 0:
        raise ValueError(f"forecast_horizon must be positive, got {forecast_horizon}")
    if stride is None:
        stride = 1
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if test_size is not None and not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be strictly between 0 and 1, got {test_size}")

    # Determine series partition
    id_col = None
    if group_col and group_col in df.columns:
        id_col = group_col
    elif "sequence_id" in df.columns:
        id_col = "sequence_id"
    elif "id" in df.columns:
        id_col = "id"

    if id_col is not None:
        series_ids = df[id_col].unique(maintain_order=True).to_list()
    else:
        series_ids = [0]

    # Resolve feature and target columns
    reserved_cols = {time_col}
    if id_col:
        reserved_cols.add(id_col)
    if group_col:
        reserved_cols.add(group_col)

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in reserved_cols]
    if not feature_cols:
        raise ValueError("No feature columns found to segment.")

    if target_col is None:
        target_col = "target" if "target" in df.columns else feature_cols[0]

    ts_frames = []
    target_records = []
    global_seq_id = 0

    for s_idx, sid in enumerate(series_ids):
        if id_col is not None:
            sub_df = df.filter(pl.col(id_col) == sid)
        else:
            sub_df = df

        if time_col in sub_df.columns:
            sub_df = sub_df.sort(time_col)

        n_pts = sub_df.height
        if n_pts < history_len + forecast_horizon:
            continue

        feat_mats = {c: sub_df[c].to_numpy() for c in feature_cols}
        tgt_vec = sub_df[target_col].to_numpy().astype(np.float32)

        if test_size is not None:
            split = int(n_pts * (1.0 - test_size))
            min_train_len = history_len + forecast_horizon
            if split < min_train_len:
                raise ValueError(
                    f"Series '{sid}' training length ({split}) is shorter than required history_len + forecast_horizon ({min_train_len})."
                )
            if n_pts - split < forecast_horizon:
                raise ValueError(
                    f"Series '{sid}' test length ({n_pts - split}) is shorter than forecast_horizon ({forecast_horizon})."
                )

            # Training windows: target must not cross into the held-out test region
            train_last_start = split - history_len - forecast_horizon
            for start in range(0, train_last_start + 1, stride):
                end_history = start + history_len
                end_forecast = end_history + forecast_horizon

                w_dict = {
                    "sequence_id": np.full(history_len, global_seq_id, dtype=np.int32),
                    "step": np.arange(history_len, dtype=np.int32),
                }
                for c in feature_cols:
                    w_dict[c] = feat_mats[c][start:end_history].astype(np.float32)
                ts_frames.append(pl.DataFrame(w_dict))

                horizon_vals = tgt_vec[end_history:end_forecast]
                t_entry = {
                    "sequence_id": global_seq_id,
                    "target": [float(v) for v in horizon_vals],
                    "split": "train",
                    "series_id": sid if id_col is not None else s_idx,
                }
                if id_col is not None:
                    t_entry["group"] = sid
                for h_i, v in enumerate(horizon_vals):
                    t_entry[f"target_{h_i}"] = float(v)
                target_records.append(t_entry)
                global_seq_id += 1

            # Testing windows: forecast target must be strictly in the held-out future
            test_first_start = max(0, split - history_len)
            test_last_start = n_pts - history_len - forecast_horizon
            for start in range(test_first_start, test_last_start + 1, stride):
                forecast_start = start + history_len
                if forecast_start < split:
                    continue
                end_forecast = forecast_start + forecast_horizon

                w_dict = {
                    "sequence_id": np.full(history_len, global_seq_id, dtype=np.int32),
                    "step": np.arange(history_len, dtype=np.int32),
                }
                for c in feature_cols:
                    w_dict[c] = feat_mats[c][start:forecast_start].astype(np.float32)
                ts_frames.append(pl.DataFrame(w_dict))

                horizon_vals = tgt_vec[forecast_start:end_forecast]
                t_entry = {
                    "sequence_id": global_seq_id,
                    "target": [float(v) for v in horizon_vals],
                    "split": "test",
                    "series_id": sid if id_col is not None else s_idx,
                }
                if id_col is not None:
                    t_entry["group"] = sid
                for h_i, v in enumerate(horizon_vals):
                    t_entry[f"target_{h_i}"] = float(v)
                target_records.append(t_entry)
                global_seq_id += 1
        else:
            # Non-partitioned sliding windows
            last_start = n_pts - history_len - forecast_horizon
            for start in range(0, last_start + 1, stride):
                end_history = start + history_len
                end_forecast = end_history + forecast_horizon

                w_dict = {
                    "sequence_id": np.full(history_len, global_seq_id, dtype=np.int32),
                    "step": np.arange(history_len, dtype=np.int32),
                }
                for c in feature_cols:
                    w_dict[c] = feat_mats[c][start:end_history].astype(np.float32)
                ts_frames.append(pl.DataFrame(w_dict))

                horizon_vals = tgt_vec[end_history:end_forecast]
                t_entry = {
                    "sequence_id": global_seq_id,
                    "target": [float(v) for v in horizon_vals],
                    "split": "all",
                    "series_id": sid if id_col is not None else s_idx,
                }
                if id_col is not None:
                    t_entry["group"] = sid
                for h_i, v in enumerate(horizon_vals):
                    t_entry[f"target_{h_i}"] = float(v)
                target_records.append(t_entry)
                global_seq_id += 1

    if not ts_frames:
        raise ValueError("No windows could be generated with the given parameters and data length.")

    df_ts = pl.concat(ts_frames)
    df_targets = pl.DataFrame(target_records)

    casts = [pl.col("sequence_id").cast(pl.Int32)]
    for h_i in range(forecast_horizon):
        if f"target_{h_i}" in df_targets.columns:
            casts.append(pl.col(f"target_{h_i}").cast(pl.Float32))
    df_targets = df_targets.with_columns(casts)

    return df_ts, df_targets


def to_forecasting_tensors(
    df_ts: pl.DataFrame,
    df_targets: pl.DataFrame,
    feature_cols: Optional[List[str]] = None,
    id_col: str = "sequence_id",
    step_col: str = "step",
    ensure_3d: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert windowed forecasting DataFrames into 3D feature tensor and 2D target matrix.

    Args:
        df_ts: Long-format Polars DataFrame containing historical sequences.
        df_targets: Polars DataFrame containing multi-step forecast targets.
        feature_cols: Optional list of signal feature column names.
        id_col: Column name identifying sequence instances.
        step_col: Column name identifying temporal steps within sequence.
        ensure_3d: If True, guarantees X has shape (n_samples, W, C) where C >= 1.

    Returns:
        Tuple of (X, y), where X is shaped (n_samples, W, C) and y is shaped (n_samples, H).
    """
    if id_col not in df_ts.columns:
        if "id" in df_ts.columns:
            id_col = "id"
        else:
            raise KeyError(f"Sequence ID column '{id_col}' not found in df_ts.")

    if step_col not in df_ts.columns:
        if "time" in df_ts.columns:
            step_col = "time"
        else:
            raise KeyError(f"Step column '{step_col}' not found in df_ts.")

    if feature_cols is None:
        feature_cols = [c for c in df_ts.columns if c not in (id_col, step_col)]

    df_sorted = df_ts.sort([id_col, step_col])
    sequence_ids = df_sorted[id_col].unique(maintain_order=True)
    n_sequences = sequence_ids.len()
    if n_sequences == 0:
        return np.empty((0, 0, len(feature_cols))), np.empty((0, 0))

    first_seq_len = df_sorted.filter(pl.col(id_col) == sequence_ids[0]).height
    n_channels = len(feature_cols)

    arr_flat = df_sorted.select(feature_cols).to_numpy()
    if n_channels == 1 and not ensure_3d:
        X = arr_flat.reshape(n_sequences, first_seq_len)
    else:
        X = arr_flat.reshape(n_sequences, first_seq_len, n_channels)

    # Extract multi-horizon target matrix y
    target_id_col = id_col if id_col in df_targets.columns else ("id" if "id" in df_targets.columns else None)
    df_tgt_sorted = df_targets.sort(target_id_col) if target_id_col is not None else df_targets

    horizon_cols = [c for c in df_tgt_sorted.columns if c.startswith("target_") and c[7:].isdigit()]
    if horizon_cols:
        horizon_cols = sorted(horizon_cols, key=lambda c: int(c.split("_")[1]))
        y = df_tgt_sorted.select(horizon_cols).to_numpy().astype(np.float32)
    elif "target" in df_tgt_sorted.columns:
        target_series = df_tgt_sorted["target"]
        if target_series.dtype == pl.List or (target_series.len() > 0 and isinstance(target_series[0], (list, np.ndarray))):
            y = np.array(target_series.to_list(), dtype=np.float32)
        else:
            y = target_series.to_numpy().astype(np.float32).reshape(-1, 1)
    else:
        raise KeyError("No valid target columns found in df_targets.")

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Sample count mismatch: X has {X.shape[0]} sequences but y has {y.shape[0]} targets."
        )

    return X, y

