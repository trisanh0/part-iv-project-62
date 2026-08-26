"""Unit tests for Subsampled Two-Stage Feature Selection."""

import unittest
import numpy as np
import pandas as pd
from tempo.selection.subsampled import SubsampledFeatureSelector, evaluate_subsampling_sweep


class TestSubsampledSelector(unittest.TestCase):
    def test_subsampled_selector_fit_transform(self):
        """Verify fit and transform lifecycle on synthetic data."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(80, 20), columns=[f"feat_{i}" for i in range(20)])
        y = pd.Series((X["feat_0"] > 0).astype(int), name="target")

        selector = SubsampledFeatureSelector(sample_ratio=0.25, random_state=42)
        selector.fit(X, y)

        self.assertIsNotNone(selector.selected_feature_names_)
        self.assertGreater(len(selector.selected_feature_names_), 0)
        self.assertEqual(selector.n_features_in_, 20)
        self.assertLessEqual(selector.n_features_out_, 20)

        X_trans = selector.transform(X)
        self.assertEqual(X_trans.shape[0], 80)
        self.assertEqual(X_trans.shape[1], selector.n_features_out_)

    def test_subsampling_sweep_evaluator(self):
        """Verify evaluate_subsampling_sweep produces valid comparison table."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(60, 15), columns=[f"feat_{i}" for i in range(15)])
        y = pd.Series(np.random.randint(0, 2, size=60), name="target")

        df_sweep = evaluate_subsampling_sweep(X, y, sample_ratios=[0.30, 0.60, 1.0])
        self.assertEqual(len(df_sweep), 3)
        self.assertIn("Sample Ratio Config", df_sweep.columns)
        self.assertIn("Fit Time (s)", df_sweep.columns)


if __name__ == "__main__":
    unittest.main()
