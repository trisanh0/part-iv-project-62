"""Unit tests for TEMPO feature extraction modules."""

import unittest
import numpy as np
import pandas as pd
import polars as pl

from tempo.extraction import (
    numpy_statistical_extractor,
    polars_statistical_extractor,
    tsfresh_extractor,
    numba_feature_extractor,
)


class TestExtraction(unittest.TestCase):
    def test_numpy_statistical_extractor(self):
        """Verify 2D NumPy array statistical extraction output shape and values."""
        X = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [10.0, 20.0, 30.0, 40.0, 50.0],
        ])
        df_feat = numpy_statistical_extractor(X)

        self.assertIsInstance(df_feat, pd.DataFrame)
        self.assertEqual(df_feat.shape, (2, 5))
        self.assertEqual(set(df_feat.columns), {"mean", "std", "min", "max", "energy"})
        self.assertAlmostEqual(df_feat.iloc[0]["mean"], 3.0)
        self.assertAlmostEqual(df_feat.iloc[0]["min"], 1.0)
        self.assertAlmostEqual(df_feat.iloc[0]["max"], 5.0)

    def test_polars_statistical_extractor(self):
        """Verify Polars statistical feature extraction preserves entity grouping."""
        df_raw = pl.DataFrame({
            "id": [0, 0, 0, 1, 1, 1],
            "time": [0, 1, 2, 0, 1, 2],
            "val": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        })
        df_feat = polars_statistical_extractor(df_raw, id_col="id", value_cols=["val"])

        self.assertIsInstance(df_feat, pl.DataFrame)
        self.assertEqual(df_feat.height, 2)
        self.assertIn("val__mean", df_feat.columns)
        mean_val = df_feat.filter(pl.col("id") == 0)["val__mean"][0]
        self.assertAlmostEqual(mean_val, 2.0)

    def test_tsfresh_extractor_minimal(self):
        """Verify TSFresh minimal extraction wrapper."""
        df_raw = pd.DataFrame({
            "id": [1, 1, 1, 2, 2, 2],
            "time": [0, 1, 2, 0, 1, 2],
            "val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        df_feat = tsfresh_extractor(df_raw, parameter_set="minimal")

        self.assertIsInstance(df_feat, pd.DataFrame)
        self.assertEqual(df_feat.shape[0], 2)
        self.assertGreater(df_feat.shape[1], 0)

    def test_numba_feature_extractor(self):
        """Verify Numba JIT feature extractor shape and mathematical accuracy."""
        X = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [10.0, 20.0, 30.0, 40.0, 50.0],
        ])
        df_feat = numba_feature_extractor(X, n_fft_coeffs=2)

        self.assertIsInstance(df_feat, pd.DataFrame)
        self.assertEqual(df_feat.shape[0], 2)
        self.assertIn("var_0__mean", df_feat.columns)
        self.assertAlmostEqual(df_feat.iloc[0]["var_0__mean"], 3.0)
        self.assertAlmostEqual(df_feat.iloc[0]["var_0__min"], 1.0)
        self.assertAlmostEqual(df_feat.iloc[0]["var_0__max"], 5.0)


if __name__ == "__main__":
    unittest.main()


