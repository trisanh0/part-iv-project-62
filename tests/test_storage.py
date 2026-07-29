"""Unit tests for dataset conversion and Parquet schema validation."""

import unittest
import os
import tempfile
import polars as pl
from tempo.storage import generate_simulated_dataset, validate_export


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


if __name__ == "__main__":
    unittest.main()

