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


if __name__ == "__main__":
    unittest.main()
