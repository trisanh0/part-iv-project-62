"""Data ingestion, dataset standardisation, and binary Parquet storage."""

from tempo.storage.dataset import (
    generate_simulated_dataset,
    generate_simulated_forecasting_dataset,
    generate_drift_bifurcation_dataset,
    convert_predictive_maintenance,
    convert_beed,
    convert_uci_har,
    convert_har,
    convert_appliances_energy,
    convert_beijing_pm25,
    convert_gas_sensor_drift,
    to_numpy_tensor,
    load_dataset,
    validate_export,
)
from tempo.storage.segmentation import (
    segment_time_series,
    segment_forecasting_series,
    to_forecasting_tensors,
)

from tempo.storage.feature_store import (
    FeatureStore,
    compute_dataset_fingerprint,
    compute_extractor_version,
)

__all__ = [
    "generate_simulated_dataset",
    "generate_simulated_forecasting_dataset",
    "generate_drift_bifurcation_dataset",
    "convert_predictive_maintenance",
    "convert_beed",
    "convert_uci_har",
    "convert_har",
    "convert_appliances_energy",
    "convert_beijing_pm25",
    "convert_gas_sensor_drift",
    "to_numpy_tensor",
    "load_dataset",
    "validate_export",
    "segment_time_series",
    "segment_forecasting_series",
    "to_forecasting_tensors",
    "FeatureStore",
    "compute_dataset_fingerprint",
    "compute_extractor_version",
]
