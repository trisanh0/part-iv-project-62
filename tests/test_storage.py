"""Unit tests for universal SDF storage, segmentation, tensor conversion, and loading."""

import unittest
import os
import tempfile
import polars as pl
import numpy as np
from tempo.storage import (
    generate_simulated_dataset,
    validate_export,
    segment_time_series,
    to_numpy_tensor,
    load_dataset,
)


class TestStorage(unittest.TestCase):
    def test_generate_and_validate_simulated_dataset(self):
        """Verify synthetic dataset generation and schema validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_simulated_dataset(output_dir=tmpdir, n_series=10, series_len=50)

            self.assertTrue(os.path.exists(os.path.join(tmpdir, "time_series.parquet")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "targets.parquet")))

            self.assertTrue(validate_export(tmpdir))

            df_ts = pl.read_parquet(os.path.join(tmpdir, "time_series.parquet"))
            df_target = pl.read_parquet(os.path.join(tmpdir, "targets.parquet"))

            self.assertEqual(df_ts.height, 500)
            self.assertEqual(df_target.height, 10)
            self.assertIn("sequence_id", df_ts.columns)
            self.assertIn("step", df_ts.columns)

    def test_segment_time_series(self):
        """Verify sliding-window segmentation on continuous temporal data."""
        df_raw = pl.DataFrame({
            "time": np.arange(1000, dtype=np.int32),
            "sensor1": np.random.randn(1000).astype(np.float32),
            "sensor2": np.random.randn(1000).astype(np.float32),
            "label": np.random.choice([0, 1], size=1000).astype(np.int32),
        })

        df_ts, df_targets = segment_time_series(
            df_raw,
            window_size=200,
            stride=50,
            time_col="time",
            feature_cols=["sensor1", "sensor2"],
            label_col="label",
            label_strategy="majority_vote",
        )

        n_expected_windows = (1000 - 200) // 50 + 1
        self.assertEqual(df_targets.height, n_expected_windows)
        self.assertEqual(df_ts.height, n_expected_windows * 200)
        self.assertIn("sequence_id", df_ts.columns)
        self.assertIn("step", df_ts.columns)

    def test_to_numpy_tensor_conversion(self):
        """Verify long Polars DataFrame to 3D/2D NumPy tensor conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_simulated_dataset(output_dir=tmpdir, n_series=10, series_len=50)
            df_ts = pl.read_parquet(os.path.join(tmpdir, "time_series.parquet"))

            tensor_2d = to_numpy_tensor(df_ts, feature_cols=["value"])
            self.assertEqual(tensor_2d.shape, (10, 50))

            df_ts_multi = df_ts.with_columns(
                (pl.col("value") * 2.0).alias("value2")
            )
            tensor_3d = to_numpy_tensor(df_ts_multi, feature_cols=["value", "value2"])
            self.assertEqual(tensor_3d.shape, (10, 50, 2))

    def test_load_dataset_simulated(self):
        """Verify auto-loader for simulated dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df_ts, df_targets = load_dataset("simulated", processed_dir=tmpdir)
            self.assertGreater(df_ts.height, 0)
            self.assertGreater(df_targets.height, 0)
            self.assertTrue(validate_export(os.path.join(tmpdir, "simulated")))

    def test_segment_time_series_default_stride(self):
        """Verify sliding-window segmentation defaults to non-overlapping stride=window_size."""
        df_raw = pl.DataFrame({
            "time": np.arange(1000, dtype=np.int32),
            "sensor1": np.random.randn(1000).astype(np.float32),
            "label": np.random.choice([0, 1], size=1000).astype(np.int32),
        })

        # stride=None should default to window_size=200
        df_ts, df_targets = segment_time_series(
            df_raw,
            window_size=200,
            stride=None,
            time_col="time",
            feature_cols=["sensor1"],
            label_col="label",
        )

        n_expected_windows = 1000 // 200  # 5 non-overlapping windows
        self.assertEqual(df_targets.height, n_expected_windows)
        self.assertEqual(df_ts.height, n_expected_windows * 200)

    def test_segment_time_series_continuous_target(self):
        """Verify continuous regression targets are preserved as Float64 without integer truncation."""
        np.random.seed(42)
        continuous_targets = np.random.uniform(10.5, 99.7, size=500).astype(np.float32)
        df_raw = pl.DataFrame({
            "time": np.arange(500, dtype=np.int32),
            "sensor": np.random.randn(500).astype(np.float32),
            "energy_target": continuous_targets,
        })

        df_ts, df_targets = segment_time_series(
            df_raw,
            window_size=100,
            stride=100,
            feature_cols=["sensor"],
            label_col="energy_target",
            label_strategy="mean",
        )

        self.assertEqual(df_targets.height, 5)
        self.assertEqual(df_targets["target"].dtype, pl.Float64)
        # Check that fractional precision is preserved (not truncated to integer)
        self.assertTrue(any(abs(val - round(val)) > 0.01 for val in df_targets["target"].to_list()))

    def test_segment_time_series_group_metadata(self):
        """Verify subject/entity group metadata is preserved in segmentation targets."""
        subjects = np.repeat([1, 2, 3, 4], 100).astype(np.int32)
        df_raw = pl.DataFrame({
            "time": np.arange(400, dtype=np.int32),
            "sensor": np.random.randn(400).astype(np.float32),
            "label": np.random.choice([0, 1], size=400).astype(np.int32),
            "subject_id": subjects,
        })

        df_ts, df_targets = segment_time_series(
            df_raw,
            window_size=100,
            stride=100,
            feature_cols=["sensor"],
            label_col="label",
            group_col="subject_id",
        )

        self.assertEqual(df_targets.height, 4)
        self.assertIn("group", df_targets.columns)
        self.assertIn("subject_id", df_targets.columns)
        self.assertEqual(df_targets["subject_id"].to_list(), [1, 2, 3, 4])

    def test_convert_har_alias_and_exports(self):
        """Verify convert_har alias is exported in tempo.storage and matches convert_uci_har."""
        from tempo.storage import (
            convert_har,
            convert_uci_har,
            convert_appliances_energy,
            convert_beijing_pm25,
            convert_gas_sensor_drift,
        )
        self.assertIs(convert_har, convert_uci_har)
        self.assertTrue(callable(convert_appliances_energy))
        self.assertTrue(callable(convert_beijing_pm25))
        self.assertTrue(callable(convert_gas_sensor_drift))

    def test_convert_gas_sensor_drift_batch_metadata(self):
        """Verify convert_gas_sensor_drift preserves batch and group metadata in targets."""
        from tempo.storage import convert_gas_sensor_drift
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = os.path.join(tmpdir, "raw")
            out_dir = os.path.join(tmpdir, "out")
            os.makedirs(raw_dir)
            # Create synthetic batch1 and batch2 files
            for b in [1, 2]:
                with open(os.path.join(raw_dir, f"batch{b}.dat"), "w") as f:
                    f.write(f"1;1.0 1:0.5 2:1.2\n")
                    f.write(f"2;2.0 1:0.8 2:1.5\n")

            convert_gas_sensor_drift(raw_dir=raw_dir, output_dir=out_dir)
            targets = pl.read_parquet(os.path.join(out_dir, "targets.parquet"))
            self.assertEqual(targets.height, 4)
            self.assertIn("batch", targets.columns)
            self.assertIn("group", targets.columns)
            self.assertEqual(targets["batch"].to_list(), [1, 1, 2, 2])
            self.assertEqual(targets["group"].to_list(), [1, 1, 2, 2])


if __name__ == "__main__":
    unittest.main()

