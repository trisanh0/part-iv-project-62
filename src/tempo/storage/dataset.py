"""Dataset standardisation, zero-copy tensor conversion, and schema validation for TEMPO."""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
import polars as pl
import numpy as np

from tempo.export import save_dataframe
from tempo.storage.segmentation import segment_time_series, segment_forecasting_series


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


def generate_simulated_forecasting_dataset(
    output_dir: Optional[str] = None,
    n_series: int = 100,
    series_len: int = 200,
    history_len: int = 100,
    forecast_horizon: int = 20,
    test_size: float = 0.2,
    seed: int = 42,
    rag_prefix: Optional[str] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Generate synthetic time-series forecasting dataset with causal chronological integrity.

    Synthesizes independent non-stationary signals combining multi-harmonic seasonalities,
    linear trends, autoregressive AR(1) dynamics, low-frequency amplitude modulations,
    and localized shock disturbances. Sliding windows are generated strictly respecting
    causal temporal boundaries to eliminate future lookahead bias.

    Args:
        output_dir: Optional directory path where Parquet files will be written.
        n_series: Number of independent time-series signals to synthesize.
        series_len: Total temporal length of each continuous raw series.
        history_len: Length of the historical observation window W.
        forecast_horizon: Length of the future forecast horizon H.
        test_size: Fraction of each series held out for future evaluation (0 < test_size < 1).
        seed: Random state seed for reproducibility.
        rag_prefix: Optional Contexere RAG prefix identifier (e.g. 'TM').

    Returns:
        Tuple of (df_ts, df_targets) conforming to TEMPO standardized format.
    """
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    if series_len <= history_len + forecast_horizon:
        raise ValueError(
            f"series_len ({series_len}) must exceed history_len + forecast_horizon ({history_len + forecast_horizon})"
        )

    split = int(series_len * (1.0 - test_size))
    min_train_len = history_len + forecast_horizon
    if split < min_train_len:
        raise ValueError(
            f"Training period ({split}) must be at least history_len + forecast_horizon ({min_train_len})"
        )
    if series_len - split < forecast_horizon:
        raise ValueError(
            f"Testing period ({series_len - split}) must contain at least forecast_horizon ({forecast_horizon}) points"
        )

    rng = np.random.default_rng(seed)
    raw_df_parts = []

    for sid in range(n_series):
        t = np.arange(series_len, dtype=np.float32)

        period = rng.uniform(18.0, 45.0)
        amplitude = rng.uniform(0.5, 2.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        trend_slope = rng.uniform(-0.015, 0.015)
        ar_strength = rng.uniform(0.45, 0.90)
        noise_std = rng.uniform(0.05, 0.25)

        seasonal = amplitude * np.sin(2.0 * np.pi * t / period + phase)
        trend = trend_slope * t

        ar = np.zeros(series_len, dtype=np.float32)
        innovations = rng.normal(0.0, noise_std, size=series_len)
        for j in range(1, series_len):
            ar[j] = ar_strength * ar[j - 1] + innovations[j]

        modulation = 1.0 + 0.15 * np.sin(2.0 * np.pi * t / 80.0)

        shock = np.zeros(series_len, dtype=np.float32)
        possible_starts = np.arange(10, series_len - 3)
        n_shocks = rng.integers(0, min(3, len(possible_starts)) + 1)
        if len(possible_starts) > 0 and n_shocks > 0:
            for start_shock in rng.choice(possible_starts, size=n_shocks, replace=False):
                magnitude = rng.normal(0.0, 1.0)
                shock[start_shock:start_shock + 3] += magnitude * np.array([1.0, 0.6, 0.3], dtype=np.float32)

        series = (modulation * seasonal + trend + ar + shock).astype(np.float32)

        raw_df_parts.append(
            pl.DataFrame({
                "series_id": np.full(series_len, sid, dtype=np.int32),
                "time": np.arange(series_len, dtype=np.int32),
                "value": series,
            })
        )

    df_raw = pl.concat(raw_df_parts)

    df_ts, df_targets = segment_forecasting_series(
        df=df_raw,
        history_len=history_len,
        forecast_horizon=forecast_horizon,
        stride=1,
        time_col="time",
        feature_cols=["value"],
        target_col="value",
        group_col="series_id",
        test_size=test_size,
    )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        if rag_prefix:
            save_dataframe(df_ts, prefix=rag_prefix, keyword="simulated_forecasting_time_series", output_dir=output_dir)
            save_dataframe(df_targets, prefix=rag_prefix, keyword="simulated_forecasting_targets", output_dir=output_dir)
        else:
            df_ts.write_parquet(os.path.join(output_dir, "time_series.parquet"))
            df_targets.write_parquet(os.path.join(output_dir, "targets.parquet"))

    return df_ts, df_targets


def convert_predictive_maintenance(
    raw_path: str,
    output_dir: str,
    window_size: int = 200,
    stride: Optional[int] = None,
    label_strategy: str = "last",
    rag_prefix: Optional[str] = None,
) -> None:
    """Standardize and segment AI4I 2020 Predictive Maintenance dataset.

    Args:
        raw_path: File path of raw source CSV.
        output_dir: Output directory path.
        window_size: Length of each sliding window segment.
        stride: Stride offset between consecutive windows. Defaults to window_size (non-overlapping).
        label_strategy: Target label aggregation strategy ('last', 'any_positive', 'majority_vote').
        rag_prefix: Optional Contexere RAG prefix identifier.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    if stride is None:
        stride = window_size

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
    stride: Optional[int] = None,
    label_strategy: str = "majority_vote",
    rag_prefix: Optional[str] = None,
) -> None:
    """Standardize and segment BEED EEG dataset.

    Args:
        raw_path: File path of raw source CSV.
        output_dir: Output directory path.
        window_size: Length of each sliding window segment.
        stride: Stride offset between consecutive windows. Defaults to window_size (non-overlapping).
        label_strategy: Target label aggregation strategy.
        rag_prefix: Optional Contexere RAG prefix identifier.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    if stride is None:
        stride = window_size

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


def convert_uci_har(
    raw_dir: str,
    output_dir: str,
    rag_prefix: Optional[str] = None,
) -> None:
    """Standardize UCI Human Activity Recognition (HAR) inertial signals.

    Preserves subject/entity metadata in targets to prevent cross-subject leakage in CV.

    Args:
        raw_dir: Directory containing raw 'UCI HAR Dataset' folder or zip.
        output_dir: Output directory path.
        rag_prefix: Optional Contexere RAG prefix identifier.
    """
    base = Path(raw_dir)
    if (base / "UCI HAR Dataset").exists():
        base = base / "UCI HAR Dataset"
    elif (base / "har" / "UCI HAR Dataset").exists():
        base = base / "har" / "UCI HAR Dataset"
    elif (base / "UCI HAR Dataset.zip").exists():
        import zipfile
        with zipfile.ZipFile(base / "UCI HAR Dataset.zip", "r") as z:
            z.extractall(base)
        base = base / "UCI HAR Dataset"

    signal_names = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z"
    ]

    all_signals = []
    all_targets = []
    all_subjects = []

    for split in ["train", "test"]:
        split_dir = base / split
        y_path = split_dir / f"y_{split}.txt"
        y = np.loadtxt(y_path, dtype=int)
        all_targets.append(y)

        subject_path = split_dir / f"subject_{split}.txt"
        if subject_path.exists():
            subjects = np.loadtxt(subject_path, dtype=int)
            all_subjects.append(subjects)

        channels = []
        for sig in signal_names:
            sig_file = split_dir / "Inertial Signals" / f"{sig}_{split}.txt"
            arr = np.loadtxt(sig_file)
            channels.append(arr)

        tensor_split = np.stack(channels, axis=-1)
        all_signals.append(tensor_split)

    full_tensor = np.concatenate(all_signals, axis=0)
    full_y = np.concatenate(all_targets, axis=0)
    full_subjects = np.concatenate(all_subjects, axis=0) if all_subjects else None

    n_sequences, seq_len, n_channels = full_tensor.shape
    seq_ids = np.repeat(np.arange(n_sequences, dtype=np.int32), seq_len)
    steps = np.tile(np.arange(seq_len, dtype=np.int32), n_sequences)
    flat_signals = full_tensor.reshape(-1, n_channels)

    data_dict = {"sequence_id": seq_ids, "step": steps}
    for i, sig_name in enumerate(signal_names):
        data_dict[sig_name] = flat_signals[:, i].astype(np.float32)

    df_ts = pl.DataFrame(data_dict)
    
    target_dict = {
        "sequence_id": np.arange(n_sequences, dtype=np.int32),
        "target": full_y.astype(np.int32),
    }
    if full_subjects is not None:
        target_dict["subject_id"] = full_subjects.astype(np.int32)
        target_dict["group"] = full_subjects.astype(np.int32)

    df_targets = pl.DataFrame(target_dict)

    os.makedirs(output_dir, exist_ok=True)
    if rag_prefix:
        save_dataframe(df_ts, prefix=rag_prefix, keyword="har_time_series", output_dir=output_dir)
        save_dataframe(df_targets, prefix=rag_prefix, keyword="har_targets", output_dir=output_dir)
    else:
        df_ts.write_parquet(os.path.join(output_dir, "time_series.parquet"))
        df_targets.write_parquet(os.path.join(output_dir, "targets.parquet"))


convert_har = convert_uci_har


def convert_appliances_energy(
    raw_path: str,
    output_dir: str,
    window_size: int = 144,
    stride: Optional[int] = None,
    rag_prefix: Optional[str] = None,
) -> None:
    """Standardize and segment UCI Appliances Energy Prediction dataset."""
    import pandas as pd
    df_pd = pd.read_csv(raw_path)
    n_rows = len(df_pd)

    if stride is None:
        stride = window_size

    feature_cols = [
        "lights", "T1", "RH_1", "T2", "RH_2", "T3", "RH_3", "T4", "RH_4",
        "T5", "RH_5", "T6", "RH_6", "T7", "RH_7", "T8", "RH_8", "T9", "RH_9",
        "T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"
    ]

    for c in ["Appliances"] + feature_cols:
        df_pd[c] = pd.to_numeric(df_pd[c].astype(str).str.strip(), errors="coerce").fillna(0.0)

    df_proc = pl.DataFrame({
        "time": np.arange(n_rows, dtype=np.int32),
        "target": df_pd["Appliances"].to_numpy(dtype=np.float32),
    })
    for c in feature_cols:
        df_proc = df_proc.with_columns(pl.Series(c, df_pd[c].to_numpy(dtype=np.float32)))

    df_ts, df_targets = segment_time_series(
        df_proc,
        window_size=window_size,
        stride=stride,
        time_col="time",
        feature_cols=feature_cols,
        label_col="target",
        label_strategy="last",
    )

    os.makedirs(output_dir, exist_ok=True)
    if rag_prefix:
        save_dataframe(df_ts, prefix=rag_prefix, keyword="appliances_energy_time_series", output_dir=output_dir)
        save_dataframe(df_targets, prefix=rag_prefix, keyword="appliances_energy_targets", output_dir=output_dir)
    else:
        df_ts.write_parquet(os.path.join(output_dir, "time_series.parquet"))
        df_targets.write_parquet(os.path.join(output_dir, "targets.parquet"))


def convert_beijing_pm25(
    raw_path: str,
    output_dir: str,
    window_size: int = 24,
    stride: Optional[int] = None,
    rag_prefix: Optional[str] = None,
) -> None:
    """Standardize and segment UCI Beijing PM2.5 dataset."""
    import pandas as pd
    df_pd = pd.read_csv(raw_path)
    df_pd["pm2.5"] = df_pd["pm2.5"].ffill().bfill().fillna(0.0)

    if stride is None:
        stride = window_size

    cbwd_dummies = pd.get_dummies(df_pd["cbwd"], prefix="cbwd", dtype=float)
    df_pd = pd.concat([df_pd, cbwd_dummies], axis=1)

    feature_cols = ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"] + list(cbwd_dummies.columns)
    df_proc = pl.DataFrame({
        "time": np.arange(len(df_pd), dtype=np.int32),
        "target": df_pd["pm2.5"].to_numpy(dtype=np.float32),
    })
    for c in feature_cols:
        df_proc = df_proc.with_columns(pl.Series(c, df_pd[c].to_numpy(dtype=np.float32)))

    df_ts, df_targets = segment_time_series(
        df_proc,
        window_size=window_size,
        stride=stride,
        time_col="time",
        feature_cols=feature_cols,
        label_col="target",
        label_strategy="last",
    )

    os.makedirs(output_dir, exist_ok=True)
    if rag_prefix:
        save_dataframe(df_ts, prefix=rag_prefix, keyword="beijing_pm25_time_series", output_dir=output_dir)
        save_dataframe(df_targets, prefix=rag_prefix, keyword="beijing_pm25_targets", output_dir=output_dir)
    else:
        df_ts.write_parquet(os.path.join(output_dir, "time_series.parquet"))
        df_targets.write_parquet(os.path.join(output_dir, "targets.parquet"))



def convert_gas_sensor_drift(
    raw_dir: str,
    output_dir: str,
    rag_prefix: Optional[str] = None,
) -> None:
    """Standardize UCI Gas Sensor Array Drift dataset."""
    dataset_dir = Path(raw_dir)
    if (dataset_dir / "Dataset").exists():
        dataset_dir = dataset_dir / "Dataset"
    elif (dataset_dir / "gas-sensor-drift" / "Dataset").exists():
        dataset_dir = dataset_dir / "gas-sensor-drift" / "Dataset"

    all_features = []
    all_labels = []
    all_batches = []

    for batch_num in range(1, 11):
        batch_file = dataset_dir / f"batch{batch_num}.dat"
        if not batch_file.exists():
            continue
        with open(batch_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                label = int(parts[0].split(";")[0])
                feat_vec = np.zeros(128, dtype=np.float32)
                for item in parts[1:]:
                    if ":" in item:
                        idx_str, val_str = item.split(":", 1)
                        idx = int(idx_str) - 1
                        if 0 <= idx < 128:
                            feat_vec[idx] = float(val_str)
                all_features.append(feat_vec)
                all_labels.append(label)
                all_batches.append(batch_num)

    arr_features = np.array(all_features, dtype=np.float32)
    arr_labels = np.array(all_labels, dtype=np.int32)
    arr_batches = np.array(all_batches, dtype=np.int32)
    n_samples = arr_features.shape[0]
    seq_len = 8
    n_channels = 16
    tensor_gas = arr_features.reshape(n_samples, seq_len, n_channels)

    seq_ids = np.repeat(np.arange(n_samples, dtype=np.int32), seq_len)
    steps = np.tile(np.arange(seq_len, dtype=np.int32), n_samples)
    flat_data = tensor_gas.reshape(-1, n_channels)

    data_dict = {"sequence_id": seq_ids, "step": steps}
    for ch in range(n_channels):
        data_dict[f"sensor_{ch}"] = flat_data[:, ch]

    df_ts = pl.DataFrame(data_dict)
    df_targets = pl.DataFrame({
        "sequence_id": np.arange(n_samples, dtype=np.int32),
        "target": arr_labels,
        "batch": arr_batches,
        "group": arr_batches,
    })

    os.makedirs(output_dir, exist_ok=True)
    if rag_prefix:
        save_dataframe(df_ts, prefix=rag_prefix, keyword="gas_sensor_drift_time_series", output_dir=output_dir)
        save_dataframe(df_targets, prefix=rag_prefix, keyword="gas_sensor_drift_targets", output_dir=output_dir)
    else:
        df_ts.write_parquet(os.path.join(output_dir, "time_series.parquet"))
        df_targets.write_parquet(os.path.join(output_dir, "targets.parquet"))


def load_dataset(
    name: str,
    raw_dir: str = "data/01_raw",
    processed_dir: str = "data/03_processed",
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Load or auto-convert standardized TEMPO dataset.

    Args:
        name: Name identifier of dataset.
        raw_dir: Base directory path for raw datasets.
        processed_dir: Base directory path for standardized Parquet datasets.

    Returns:
        Tuple of (df_ts, df_targets) as Polars DataFrames.
    """
    path_obj = Path(name)
    if len(path_obj.parts) > 1 or os.path.exists(os.path.join(name, "time_series.parquet")):
        ds_processed_dir = name
        base_name = path_obj.name
    else:
        ds_processed_dir = os.path.join(processed_dir, name)
        base_name = name

    ts_path = os.path.join(ds_processed_dir, "time_series.parquet")
    target_path = os.path.join(ds_processed_dir, "targets.parquet")

    if not (os.path.exists(ts_path) and os.path.exists(target_path)):
        os.makedirs(ds_processed_dir, exist_ok=True)
        if base_name in ("simulated", "synthetic"):
            generate_simulated_dataset(output_dir=ds_processed_dir)
        elif base_name in ("simulated_forecasting", "synthetic_forecasting", "forecasting"):
            generate_simulated_forecasting_dataset(output_dir=ds_processed_dir)
        elif base_name == "beed":
            raw_file = os.path.join(raw_dir, "beed", "BEED_Data.csv")
            if not os.path.exists(raw_file):
                raise FileNotFoundError(
                    f"BEED raw file not found at '{raw_file}' and no processed files at '{ds_processed_dir}'. "
                    "To run TEMPO without local raw datasets, use dataset 'simulated' or download pre-processed Parquet files."
                )
            convert_beed(raw_file, output_dir=ds_processed_dir)
        elif base_name in ("pred-maintenance", "predictive_maintenance"):
            raw_file = os.path.join(raw_dir, "pred-maintenance", "ai4i2020.csv")
            if not os.path.exists(raw_file):
                raise FileNotFoundError(
                    f"Predictive Maintenance raw file not found at '{raw_file}' and no processed files at '{ds_processed_dir}'."
                )
            convert_predictive_maintenance(raw_file, output_dir=ds_processed_dir)
        elif base_name in ("har", "uci-har"):
            raw_path = os.path.join(raw_dir, "har")
            if not os.path.exists(raw_path):
                raise FileNotFoundError(
                    f"UCI HAR raw directory not found at '{raw_path}' and no processed files at '{ds_processed_dir}'."
                )
            convert_uci_har(raw_dir=raw_path, output_dir=ds_processed_dir)
        elif base_name in ("appliances-energy", "appliances_energy"):
            raw_file = os.path.join(raw_dir, "appliances-energy", "energydata_complete.csv")
            if not os.path.exists(raw_file):
                raise FileNotFoundError(
                    f"Appliances Energy raw file not found at '{raw_file}' and no processed files at '{ds_processed_dir}'."
                )
            convert_appliances_energy(raw_file, output_dir=ds_processed_dir)
        elif base_name in ("beijing-pm25", "beijing_pm25"):
            raw_file = os.path.join(raw_dir, "beijing-pm25", "PRSA_data_2010.1.1-2014.12.31.csv")
            if not os.path.exists(raw_file):
                raise FileNotFoundError(
                    f"Beijing PM2.5 raw file not found at '{raw_file}' and no processed files at '{ds_processed_dir}'."
                )
            convert_beijing_pm25(raw_file, output_dir=ds_processed_dir)
        elif base_name in ("gas-sensor-drift", "gas_sensor_drift"):
            raw_path = os.path.join(raw_dir, "gas-sensor-drift")
            if not os.path.exists(raw_path):
                raise FileNotFoundError(
                    f"Gas Sensor Drift raw directory not found at '{raw_path}' and no processed files at '{ds_processed_dir}'."
                )
            convert_gas_sensor_drift(raw_dir=raw_path, output_dir=ds_processed_dir)
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
    has_target_col = "target" in df_target.columns or any(
        c.startswith("target_") and c[7:].isdigit() for c in df_target.columns
    )
    if not (has_target_id and has_target_col):
        return False

    if df_ts.null_count().sum().row(0)[0] > 0:
        return False

    return True
