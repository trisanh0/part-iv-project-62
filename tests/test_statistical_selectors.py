"""Unit tests for statistical and wrapper feature selectors."""

import unittest
import numpy as np
import pandas as pd
from tempo.selection import (
    select_k_best,
    mutual_info_selector,
    boruta_selector,
    tsfresh_selector,
    SubsampledFeatureSelector,
)


class TestStatisticalSelectors(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n_samples = 60
        n_features = 15
        self.X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )
        # Classification target (binary)
        self.y_cls = pd.Series((self.X["feature_0"] + self.X["feature_1"] > 0).astype(int), name="target")
        # Continuous regression target
        self.y_reg = pd.Series(
            2.5 * self.X["feature_0"] - 1.8 * self.X["feature_1"] + np.random.randn(n_samples) * 0.1,
            name="target",
        )

    def test_select_k_best_classification(self):
        """Verify select_k_best uses classification scorer (f_classif)."""
        res = select_k_best(self.X, self.y_cls, k=5, task_type="classification")
        self.assertEqual(res.shape[0], len(self.X))
        self.assertEqual(res.shape[1], 5)
        # Strongly correlated features should be included
        self.assertIn("feature_0", res.columns)

    def test_select_k_best_regression(self):
        """Verify select_k_best uses regression scorer (f_regression)."""
        res = select_k_best(self.X, self.y_reg, k=5, task_type="regression")
        self.assertEqual(res.shape[0], len(self.X))
        self.assertEqual(res.shape[1], 5)
        self.assertIn("feature_0", res.columns)
        self.assertIn("feature_1", res.columns)

    def test_mutual_info_selector_classification(self):
        """Verify mutual_info_selector works cleanly on classification tasks."""
        res = mutual_info_selector(self.X, self.y_cls, k=4, task_type="classification", random_state=42)
        self.assertEqual(res.shape[0], len(self.X))
        self.assertEqual(res.shape[1], 4)

    def test_mutual_info_selector_regression(self):
        """Verify mutual_info_selector works cleanly on regression tasks."""
        res = mutual_info_selector(self.X, self.y_reg, k=4, task_type="regression", random_state=42)
        self.assertEqual(res.shape[0], len(self.X))
        self.assertEqual(res.shape[1], 4)

    def test_select_k_best_boundary_zero(self):
        """Verify select_k_best handles k=0 boundary without error."""
        res = select_k_best(self.X, self.y_cls, k=0)
        self.assertEqual(res.shape[1], 0)

    def test_boruta_selector_classification_and_regression(self):
        """Verify boruta_selector supports classification and regression."""
        # Using a small feature set to keep execution fast
        X_small = self.X.iloc[:, :6]
        res_cls = boruta_selector(X_small, self.y_cls, n_estimators=10, random_state=42, task_type="classification")
        self.assertIsInstance(res_cls, pd.DataFrame)
        self.assertEqual(res_cls.shape[0], len(self.X))

        res_reg = boruta_selector(X_small, self.y_reg, n_estimators=10, random_state=42, task_type="regression")
        self.assertIsInstance(res_reg, pd.DataFrame)
        self.assertEqual(res_reg.shape[0], len(self.X))

    def test_subsampled_selector_null_selection_honesty(self):
        """Verify SubsampledFeatureSelector flags fallback and reports honestly when zero features survive."""
        # A base selector that always returns empty DataFrame
        def null_selector(X, y):
            return pd.DataFrame(index=X.index)

        selector = SubsampledFeatureSelector(
            base_selector=null_selector,
            sample_ratio=0.5,
            random_state=42,
        )
        selector.fit(self.X, self.y_cls)

        # Fallback must be flagged
        self.assertTrue(selector.fallback_triggered_)
        self.assertEqual(selector.n_features_out_, 0)
        self.assertEqual(selector.telemetry_["n_selected_features"], 0)
        self.assertEqual(selector.telemetry_["feature_reduction_pct"], 100.0)

        # Transform still returns full matrix to prevent downstream crash
        X_trans = selector.transform(self.X)
        self.assertEqual(X_trans.shape[1], self.X.shape[1])

    def test_boruta_selector_empty_feature_space(self):
        """Verify boruta_selector handles empty feature space without crashing."""
        X_empty = self.X.iloc[:, :0]
        res = boruta_selector(X_empty, self.y_cls, task_type="classification")
        self.assertEqual(res.shape, (len(self.X), 0))

    def test_subsampled_selector_string_name(self):
        """Verify SubsampledFeatureSelector properly recognizes string name 'select_k_best'."""
        selector = SubsampledFeatureSelector(
            base_selector="select_k_best",
            sample_ratio=1.0,
            task_type="classification",
        )
        selector.fit(self.X, self.y_cls)
        # Should NOT select all 15 features if select_k_best is properly executed (default k=20, min(20, 15)=15, but let's test with select_k_best callable with k=5)
        self.assertFalse(selector.fallback_triggered_)
        self.assertEqual(selector.n_features_in_, 15)

    def test_subsampled_selector_callable_regression(self):
        """Verify SubsampledFeatureSelector forwards task_type='regression' to callable selectors."""
        selector = SubsampledFeatureSelector(
            base_selector=select_k_best,
            sample_ratio=1.0,
            task_type="regression",
        )
        # Must execute cleanly on continuous target using f_regression without warnings
        selector.fit(self.X, self.y_reg)
        self.assertFalse(selector.fallback_triggered_)

    def test_tsfresh_selector_null_on_constant(self):
        """Verify tsfresh_selector returns empty DataFrame on all-constant feature matrix."""
        X_const = pd.DataFrame(np.ones((20, 5)), columns=[f"const_{i}" for i in range(5)])
        y = pd.Series(np.random.randint(0, 2, 20))
        res = tsfresh_selector(X_const, y, task_type="classification")
        self.assertEqual(res.shape, (20, 0))


if __name__ == "__main__":
    unittest.main()
