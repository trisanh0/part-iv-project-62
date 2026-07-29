"""Dataset standardisation and schema validation for TEMPO framework."""

import os
import polars as pl
import numpy as np


def generate_simulated_dataset(
    output_dir: str,
    n_series: int = 1000,
    series_len: int = 200,
    seed: int = 42,
) -> None:
    """Generate a synthetic time-series classification dataset.

    Args:
        output_dir: Directory path where output Parquet files are written.
        n_series: Number of independent time-series signals to generate.
        series_len: Temporal length of each sequence.
        seed: Random state seed.
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
        "id": np.concatenate(entities),
        "time": np.concatenate(times),
        "value": np.concatenate(values),
    })

    df_targets = pl.DataFrame({
        "id": np.arange(n_series, dtype=np.int32),
        "target": np.array(targets, dtype=np.int32),
    })

    os.makedirs(output_dir, exist_ok=True)
    df_ts.write_parquet(os.path.join(output_dir, "time_series.parquet"))
    df_targets.write_parquet(os.path.join(output_dir, "targets.parquet"))


def convert_predictive_maintenance(
    raw_path: str,
    output_dir: str,
) -> None:
    """Standardize the AI4I 2020 Predictive Maintenance dataset.

    Args:
        raw_path: File path of raw source CSV.
        output_dir: Output directory path.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    df = pl.read_csv(raw_path)
    leak_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    drop_cols = leak_cols + ["Product ID"]
    df_filtered = df.drop(drop_cols)

    type_mapping = {"L": 0, "M": 1, "H": 2}
    df_processed = df_filtered.with_columns([
        pl.lit(1, dtype=pl.Int32).alias("id"),
        pl.col("UDI").cast(pl.Int32).alias("time"),
        pl.col("Type").replace_strict(type_mapping, default=None).cast(pl.Int32).alias("Type"),
    ])

    feature_cols = [
        "id", "time", "Type", "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
    ]

    df_ts = df_processed.select(feature_cols).fill_nan(0.0).fill_null(0.0)
    df_targets = df_processed.select([
        pl.lit(1, dtype=pl.Int32).alias("id"),
        pl.col("UDI").cast(pl.Int32).alias("time"),
        pl.col("Machine failure").cast(pl.Int32).alias("target"),
    ])

    os.makedirs(output_dir, exist_ok=True)
    df_ts.write_parquet(os.path.join(output_dir, "time_series.parquet"))
    df_targets.write_parquet(os.path.join(output_dir, "targets.parquet"))


def convert_beed(
    raw_path: str,
    output_dir: str,
) -> None:
    """Standardize the BEED EEG dataset.

    Args:
        raw_path: File path of raw source CSV.
        output_dir: Output directory path.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    df = pl.read_csv(raw_path)
    n_rows = df.height
    df_processed = df.with_columns([
        pl.lit(1, dtype=pl.Int32).alias("id"),
        pl.int_range(0, n_rows, dtype=pl.Int32).alias("time"),
    ])

    feature_cols = [f"X{i}" for i in range(1, 17)]
    select_cols = ["id", "time"] + feature_cols

    df_ts = df_processed.select(select_cols).cast({c: pl.Float32 for c in feature_cols})
    df_ts = df_ts.fill_nan(0.0).fill_null(0.0)

    df_targets = df_processed.select([
        pl.lit(1, dtype=pl.Int32).alias("id"),
        pl.col("time").alias("time"),
        pl.col("y").cast(pl.Int32).alias("target"),
    ])

    os.makedirs(output_dir, exist_ok=True)
    df_ts.write_parquet(os.path.join(output_dir, "time_series.parquet"))
    df_targets.write_parquet(os.path.join(output_dir, "targets.parquet"))


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

    if "id" not in df_ts.columns or "time" not in df_ts.columns:
        return False

    if "id" not in df_target.columns or "target" not in df_target.columns:
        return False

    if df_ts.null_count().sum().row(0)[0] > 0:
        return False

    return True
