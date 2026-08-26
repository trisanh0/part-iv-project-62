"""Dataset standardisation, zero-copy tensor conversion, and schema validation for TEMPO."""

import os
from typing import List, Optional, Tuple, Union
import polars as pl
import numpy as np

from tempo.export import save_dataframe
from tempo.storage.segmentation import segment_time_series


def generate_simulated_dataset(
    output_dir: str,
    n_series: int = 1000,
    series_len: int = 200,
    seed: int = 42,
    rag_prefix: Optional[str] = None,
) -> None:
    """Generate a synthetic time-series classification dataset.

    Args:
        output_dir: Directory path where output Parquet files are written.
        n_series: Number of independent time-series signals to generate.
        series_len: Temporal length of each sequence.
        seed: Random state seed.
        rag_prefix: Optional Contexere RAG prefix identifier (e.g. 'TM').
    """
    rng = np.random.default_rng(seed)

    entities = []
    times = []
    values = []
    targets = []

    for i in range(n_series):
        t = np.arange(series_len)
        freq = rng.uniform(0.05, 0.20)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        trend = rng.uniform(-0.01, 0.01) * t
        amplitude = rng.uniform(0.5, 2.0)

        signal = amplitude * np.sin(2.0 * np.pi * freq * t + phase)
        noise = rng.normal(0.0, 0.20, size=series_len)
        spikes = rng.choice([0.0, 1.0], size=series_len, p=[0.98, 0.02]) * rng.normal(3.0, 1.0, size=series_len)
        x = signal + trend + noise + spikes

        entities.append(np.full(series_len, i, dtype=np.int32))
        times.append(t.astype(np.int32))
        values.append(x.astype(np.float32))

        label = 1 if amplitude > 1.25 else 0
        targets.append(label)

    df_ts = pl.DataFrame({
        "sequence_id": np.concatenate(entities),
        "step": np.concatenate(times),
        "value": np.concatenate(values),
    })

    df_targets = pl.DataFrame({
        "sequence_id": np.arange(n_series, dtype=np.int32),
        "target": np.array(targets, dtype=np.int32),
    })

    os.makedirs(output_dir, exist_ok=True)
    if rag_prefix:
        save_dataframe(df_ts, prefix=rag_prefix, keyword="simulated_time_series", output_dir=output_dir)
        save_dataframe(df_targets, prefix=rag_prefix, keyword="simulated_targets", output_dir=output_dir)
    else:
        df_ts.write_parquet(os.path.join(output_dir, "time_series.parquet"))
        df_targets.write_parquet(os.path.join(output_dir, "targets.parquet"))


def convert_predictive_maintenance(
    raw_path: str,
    output_dir: str,
    window_size: int = 200,
    stride: int = 50,
    label_strategy: str = "last",
    rag_prefix: Optional[str] = None,
) -> None:
    """Standardize and segment AI4I 2020 Predictive Maintenance dataset.

    Args:
        raw_path: File path of raw source CSV.
        output_dir: Output directory path.
        window_size: Length of each sliding window segment.
        stride: Stride offset between consecutive windows.
        label_strategy: Target label aggregation strategy ('last', 'any_positive', 'majority_vote').
        rag_prefix: Optional Contexere RAG prefix identifier.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    df = pl.read_csv(raw_path)
    leak_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    drop_cols = leak_cols + ["Product ID"]
    df_filtered = df.drop(drop_cols)

    type_mapping = {"L": 0, "M": 1, "H": 2}
    df_processed = df_filtered.with_columns([
        pl.col("UDI").cast(pl.Int32).alias("time"),
        pl.col("Type").replace_strict(type_mapping, default=None).cast(pl.Int32).alias("Type"),
        pl.col("Machine failure").cast(pl.Int32).alias("target"),
    ])

    feature_cols = [
        "Type", "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
    ]

    df_segmented_ts, df_segmented_targets = segment_time_series(
        df_processed,
        window_size=window_size,
        stride=stride,
        time_col="time",
        feature_cols=feature_cols,
        label_col="target",
        label_strategy=label_strategy,
    )

    os.makedirs(output_dir, exist_ok=True)
    if rag_prefix:
        save_dataframe(df_segmented_ts, prefix=rag_prefix, keyword="pred_maintenance_time_series", output_dir=output_dir)
        save_dataframe(df_segmented_targets, prefix=rag_prefix, keyword="pred_maintenance_targets", output_dir=output_dir)
    else:
        df_segmented_ts.write_parquet(os.path.join(output_dir, "time_series.parquet"))
        df_segmented_targets.write_parquet(os.path.join(output_dir, "targets.parquet"))


def convert_beed(
    raw_path: str,
    output_dir: str,
    window_size: int = 200,
    stride: int = 50,
    label_strategy: str = "majority_vote",
    rag_prefix: Optional[str] = None,
) -> None:
    """Standardize and segment BEED EEG dataset.

    Args:
        raw_path: File path of raw source CSV.
        output_dir: Output directory path.
        window_size: Length of each sliding window segment.
        stride: Stride offset between consecutive windows.
        label_strategy: Target label aggregation strategy.
        rag_prefix: Optional Contexere RAG prefix identifier.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    df = pl.read_csv(raw_path)
    n_rows = df.height
    df_processed = df.with_columns([
        pl.int_range(0, n_rows, dtype=pl.Int32).alias("time"),
        pl.col("y").cast(pl.Int32).alias("target"),
    ])

    feature_cols = [f"X{i}" for i in range(1, 17)]
    df_processed = df_processed.with_columns([
        pl.col(c).cast(pl.Float32) for c in feature_cols
    ]).fill_nan(0.0).fill_null(0.0)

    df_segmented_ts, df_segmented_targets = segment_time_series(
        df_processed,
        window_size=window_size,
        stride=stride,
        time_col="time",
        feature_cols=feature_cols,
        label_col="target",
        label_strategy=label_strategy,
    )

    os.makedirs(output_dir, exist_ok=True)
    if rag_prefix:
        save_dataframe(df_segmented_ts, prefix=rag_prefix, keyword="beed_time_series", output_dir=output_dir)
        save_dataframe(df_segmented_targets, prefix=rag_prefix, keyword="beed_targets", output_dir=output_dir)
    else:
        df_segmented_ts.write_parquet(os.path.join(output_dir, "time_series.parquet"))
        df_segmented_targets.write_parquet(os.path.join(output_dir, "targets.parquet"))


def to_numpy_tensor(
    df_ts: pl.DataFrame,
    id_col: str = "sequence_id",
    step_col: str = "step",
    feature_cols: Optional[List[str]] = None,
) -> np.ndarray:
    """Convert long Polars DataFrame into dense 3D or 2D NumPy tensor.

    Args:
        df_ts: Long format Polars DataFrame.
        id_col: Column name identifying sequence instances.
        step_col: Column name identifying temporal steps within sequence.
        feature_cols: List of signal feature column names.

    Returns:
        3D NumPy array of shape (N_sequences, T_steps, P_channels), or 2D array (N, T) if P=1.
    """
    if id_col not in df_ts.columns:
        if "id" in df_ts.columns:
            id_col = "id"
        else:
            raise KeyError(f"Sequence ID column '{id_col}' not found in DataFrame.")

    if step_col not in df_ts.columns:
        if "time" in df_ts.columns:
            step_col = "time"
        else:
            raise KeyError(f"Step column '{step_col}' not found in DataFrame.")

    if feature_cols is None:
        feature_cols = [c for c in df_ts.columns if c not in (id_col, step_col)]

    df_sorted = df_ts.sort([id_col, step_col])

    sequence_ids = df_sorted[id_col].unique(maintain_order=True)
    n_sequences = sequence_ids.len()
    
    first_seq_len = df_sorted.filter(pl.col(id_col) == sequence_ids[0]).height
    n_channels = len(feature_cols)

    arr_flat = df_sorted.select(feature_cols).to_numpy()

    if n_channels == 1:
        tensor = arr_flat.reshape(n_sequences, first_seq_len)
    else:
        tensor = arr_flat.reshape(n_sequences, first_seq_len, n_channels)

    return tensor


def load_dataset(
    name: str,
    raw_dir: str = "data/01_raw",
    processed_dir: str = "data/03_processed",
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Load or auto-convert standardized TEMPO dataset.

    Args:
        name: Name identifier of dataset ('simulated', 'beed', 'pred-maintenance').
        raw_dir: Base directory path for raw datasets.
        processed_dir: Base directory path for standardized Parquet datasets.

    Returns:
        Tuple of (df_ts, df_targets) as Polars DataFrames.
    """
    # If name is already a valid directory with parquet files, use it directly
    if os.path.exists(os.path.join(name, "time_series.parquet")):
        ds_processed_dir = name
    else:
        ds_processed_dir = os.path.join(processed_dir, name)

    ts_path = os.path.join(ds_processed_dir, "time_series.parquet")
    target_path = os.path.join(ds_processed_dir, "targets.parquet")

    if not (os.path.exists(ts_path) and os.path.exists(target_path)):
        os.makedirs(ds_processed_dir, exist_ok=True)
        if name == "simulated":
            generate_simulated_dataset(output_dir=ds_processed_dir)
        elif name == "beed":
            raw_file = os.path.join(raw_dir, "beed", "BEED_Data.csv")
            convert_beed(raw_file, output_dir=ds_processed_dir)
        elif name in ("pred-maintenance", "predictive_maintenance"):
            raw_file = os.path.join(raw_dir, "pred-maintenance", "ai4i2020.csv")
            convert_predictive_maintenance(raw_file, output_dir=ds_processed_dir)
        else:
            raise ValueError(f"Unknown dataset name '{name}' and no processed files found at {ds_processed_dir}")

    df_ts = pl.read_parquet(ts_path)
    df_targets = pl.read_parquet(target_path)
    return df_ts, df_targets


def validate_export(export_dir: str) -> bool:
    """Validate exported files against TEMPO Parquet schema.

    Args:
        export_dir: Path to directory containing output Parquet files.

    Returns:
        True if schema checks pass.
    """
    ts_path = os.path.join(export_dir, "time_series.parquet")
    target_path = os.path.join(export_dir, "targets.parquet")

    if not os.path.exists(ts_path) or not os.path.exists(target_path):
        return False

    df_ts = pl.read_parquet(ts_path)
    df_target = pl.read_parquet(target_path)

    has_id = "sequence_id" in df_ts.columns or "id" in df_ts.columns
    has_step = "step" in df_ts.columns or "time" in df_ts.columns
    if not (has_id and has_step):
        return False

    has_target_id = "sequence_id" in df_target.columns or "id" in df_target.columns
    has_target_col = "target" in df_target.columns
    if not (has_target_id and has_target_col):
        return False

    if df_ts.null_count().sum().row(0)[0] > 0:
        return False

    return True
