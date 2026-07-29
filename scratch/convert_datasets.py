"""Dataset conversion pipeline for time-series classification benchmarks.

This script standardizes and converts raw data files into a consistent
representation of time-series signals and target binary labels, writing the 
resulting tables into single Parquet files.
"""

import os
import polars as pl
import numpy as np


def generate_simulated_dataset(
    output_dir: str,
    n_series: int = 1000,
    series_len: int = 200,
    seed: int = 42
) -> None:
    """Generate a synthetic time-series classification dataset.

    The generator builds signals containing a combination of trend components,
    periodic sinusoids, random noise, and transient anomalies (spikes).
    Classification labels are determined by signal amplitude variance thresholds.

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
        
        # Build composite signal
        signal = amplitude * np.sin(2.0 * np.pi * freq * t + phase)
        noise = rng.normal(0.0, 0.20, size=series_len)
        spikes = rng.choice([0.0, 1.0], size=series_len, p=[0.98, 0.02]) * rng.normal(3.0, 1.0, size=series_len)
        x = signal + trend + noise + spikes
        
        entities.append(np.full(series_len, i, dtype=np.int32))
        times.append(t.astype(np.int32))
        values.append(x.astype(np.float32))
        
        # Define binary label based on threshold of amplitude
        label = 1 if amplitude > 1.25 else 0
        targets.append(label)
        
    df_ts = pl.DataFrame({
        "id": np.concatenate(entities),
        "time": np.concatenate(times),
        "value": np.concatenate(values)
    })
    
    df_targets = pl.DataFrame({
        "id": np.arange(n_series, dtype=np.int32),
        "target": np.array(targets, dtype=np.int32)
    })
    
    os.makedirs(output_dir, exist_ok=True)
    df_ts.write_parquet(os.path.join(output_dir, "time_series.parquet"))
    df_targets.write_parquet(os.path.join(output_dir, "targets.parquet"))
    print(f"Exported simulated dataset to {output_dir}")


def convert_predictive_maintenance(
    raw_path: str,
    output_dir: str
) -> None:
    """Standardize the AI4I 2020 Predictive Maintenance dataset.

    Converts the continuous predictive maintenance log into a standard
    long-format time series. We use a single constant time-series ID (id=1)
    and map the chronological record index (UDI) as the time column.

    Args:
        raw_path: File path of the raw source CSV.
        output_dir: Output directory path.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw file not found: {raw_path}")
        
    # Read raw table using Polars
    df = pl.read_csv(raw_path)
    
    # Drop leak columns and uninformative metadata
    leak_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    drop_cols = leak_cols + ["Product ID"]
    df_filtered = df.drop(drop_cols)
    
    # Standardize column names and types
    # Map Type: L -> 0, M -> 1, H -> 2
    type_mapping = {"L": 0, "M": 1, "H": 2}
    
    df_processed = df_filtered.with_columns([
        pl.lit(1, dtype=pl.Int32).alias("id"),
        pl.col("UDI").cast(pl.Int32).alias("time"),
        pl.col("Type").replace_strict(type_mapping, default=None).cast(pl.Int32).alias("Type")
    ])
    
    feature_cols = [
        "id", "time", "Type", "Air temperature [K]", "Process temperature [K]", 
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
    ]
    
    # Validate missing elements or NaNs and fill with zero
    df_ts = df_processed.select(feature_cols).fill_nan(0.0).fill_null(0.0)
    
    df_targets = df_processed.select([
        pl.lit(1, dtype=pl.Int32).alias("id"),
        pl.col("UDI").cast(pl.Int32).alias("time"),
        pl.col("Machine failure").cast(pl.Int32).alias("target")
    ])
    
    os.makedirs(output_dir, exist_ok=True)
    df_ts.write_parquet(os.path.join(output_dir, "time_series.parquet"))
    df_targets.write_parquet(os.path.join(output_dir, "targets.parquet"))
    print(f"Exported predictive maintenance to {output_dir}")


def convert_beed(
    raw_path: str,
    output_dir: str
) -> None:
    """Standardize the BEED EEG dataset.

    Converts the tabular BEED raw sensor CSV into a long-format sequence
    structured for rolling feature extraction.

    Args:
        raw_path: File path of the raw source CSV.
        output_dir: Output directory path.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw file not found: {raw_path}")
        
    df = pl.read_csv(raw_path)
    
    # Setup identifier indices
    n_rows = df.height
    df_processed = df.with_columns([
        pl.lit(1, dtype=pl.Int32).alias("id"),
        pl.int_range(0, n_rows, dtype=pl.Int32).alias("time")
    ])
    
    # Extract electrode feature column names (X1 to X16)
    feature_cols = [f"X{i}" for i in range(1, 17)]
    select_cols = ["id", "time"] + feature_cols
    
    df_ts = df_processed.select(select_cols).cast({c: pl.Float32 for c in feature_cols})
    df_ts = df_ts.fill_nan(0.0).fill_null(0.0)
    
    df_targets = df_processed.select([
        pl.lit(1, dtype=pl.Int32).alias("id"),
        pl.col("time").alias("time"),
        pl.col("y").cast(pl.Int32).alias("target")
    ])
    
    os.makedirs(output_dir, exist_ok=True)
    df_ts.write_parquet(os.path.join(output_dir, "time_series.parquet"))
    df_targets.write_parquet(os.path.join(output_dir, "targets.parquet"))
    print(f"Exported BEED dataset to {output_dir}")


def validate_export(
    export_dir: str
) -> bool:
    """Validate that exported files comply with schema requirements.

    Args:
        export_dir: Path to output directory containing parquet files.

    Returns:
        True if compliance checks pass.
    """
    ts_path = os.path.join(export_dir, "time_series.parquet")
    target_path = os.path.join(export_dir, "targets.parquet")
    
    if not os.path.exists(ts_path) or not os.path.exists(target_path):
        print(f"Error: Missing files in {export_dir}")
        return False
        
    df_ts = pl.read_parquet(ts_path)
    df_target = pl.read_parquet(target_path)
    
    # Schema check
    if "id" not in df_ts.columns or "time" not in df_ts.columns:
        print(f"Error: Standard columns missing from time_series in {export_dir}")
        return False
        
    if "id" not in df_target.columns or "target" not in df_target.columns:
        print(f"Error: Standard columns missing from targets in {export_dir}")
        return False
        
    # Check for NaN / Null values
    if df_ts.null_count().sum().row(0)[0] > 0:
        print(f"Error: Nulls found in time_series features in {export_dir}")
        return False
        
    print(f"Validation successful for {export_dir}")
    print(f"  Time series rows: {df_ts.height}, target entries: {df_target.height}")
    return True


if __name__ == "__main__":
    # Define directory boundaries
    raw_base = "data/01_raw"
    interim_base = "data/02_interim"
    
    # 1. Simulated
    generate_simulated_dataset(
        output_dir=os.path.join(interim_base, "simulated")
    )
    
    # 2. Predictive Maintenance
    convert_predictive_maintenance(
        raw_path=os.path.join(raw_base, "pred-maintenance", "ai4i2020.csv"),
        output_dir=os.path.join(interim_base, "pred-maintenance")
    )
    
    # 3. BEED EEG
    convert_beed(
        raw_path=os.path.join(raw_base, "beed", "BEED_Data.csv"),
        output_dir=os.path.join(interim_base, "beed")
    )
    
    # Run validations
    print("\n--- Validating Standardized Datasets ---")
    valid_sim = validate_export(os.path.join(interim_base, "simulated"))
    valid_pm = validate_export(os.path.join(interim_base, "pred-maintenance"))
    valid_beed = validate_export(os.path.join(interim_base, "beed"))
    
    if valid_sim and valid_pm and valid_beed:
        print("\nAll datasets standardized and validated successfully.")
    else:
        print("\nError: One or more datasets failed schema validation.")
