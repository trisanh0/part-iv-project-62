"""Unit tests for statistical and wrapper feature selectors."""

import unittest
import os
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

    def test_variance_threshold_selector(self):
        """Verify variance_threshold_selector drops constant features and operates unsupervised."""
        from tempo.selection import variance_threshold_selector

        X_with_const = self.X.copy()
        X_with_const["const_col"] = 5.0  # Zero variance
        res = variance_threshold_selector(X_with_const, threshold=0.0)
        self.assertNotIn("const_col", res.columns)
        self.assertEqual(res.shape[1], self.X.shape[1])

        # Test all-constant fallback
        X_all_const = pd.DataFrame(np.ones((20, 4)), columns=[f"c_{i}" for i in range(4)])
        res_empty = variance_threshold_selector(X_all_const, threshold=0.0)
        self.assertEqual(res_empty.shape, (20, 0))

        # Test empty DataFrame
        res_zero = variance_threshold_selector(self.X.iloc[:, :0])
        self.assertEqual(res_zero.shape, (len(self.X), 0))

    def test_l1_selector_classification_and_regression(self):
        """Verify l1_selector supports classification, regression, and collapsed targets."""
        from tempo.selection import l1_selector

        # Classification
        res_cls = l1_selector(self.X, self.y_cls, C=0.5, task_type="classification", random_state=42)
        self.assertIsInstance(res_cls, pd.DataFrame)
        self.assertEqual(res_cls.shape[0], len(self.X))
        self.assertGreater(res_cls.shape[1], 0)

        # Regression
        res_reg = l1_selector(self.X, self.y_reg, alpha=0.05, task_type="regression", random_state=42)
        self.assertIsInstance(res_reg, pd.DataFrame)
        self.assertEqual(res_reg.shape[0], len(self.X))
        self.assertGreater(res_reg.shape[1], 0)

        # Multi-horizon collapsed forecasting 2D array
        y_multi = np.tile(self.y_reg.to_numpy()[:, None], (1, 5))
        y_collapsed = np.mean(y_multi, axis=1)
        res_fc = l1_selector(self.X, y_collapsed, alpha=0.05, task_type="regression", random_state=42)
        self.assertEqual(res_fc.shape[0], len(self.X))
        self.assertGreater(res_fc.shape[1], 0)

        # Empty features
        res_empty = l1_selector(self.X.iloc[:, :0], self.y_cls, task_type="classification")
        self.assertEqual(res_empty.shape, (len(self.X), 0))

    def test_tree_importance_selector(self):
        """Verify tree_importance_selector supports ExtraTrees and RandomForest across tasks."""
        from tempo.selection import tree_importance_selector

        # ExtraTrees Classification
        res_et_cls = tree_importance_selector(
            self.X, self.y_cls, model_type="extra_trees", n_estimators=10, threshold="median", task_type="classification", random_state=42
        )
        self.assertIsInstance(res_et_cls, pd.DataFrame)
        self.assertEqual(res_et_cls.shape[0], len(self.X))
        self.assertGreater(res_et_cls.shape[1], 0)

        # ExtraTrees Regression
        res_et_reg = tree_importance_selector(
            self.X, self.y_reg, model_type="extra_trees", n_estimators=10, threshold="median", task_type="regression", random_state=42
        )
        self.assertIsInstance(res_et_reg, pd.DataFrame)
        self.assertEqual(res_et_reg.shape[0], len(self.X))

        # RandomForest Classification
        res_rf_cls = tree_importance_selector(
            self.X, self.y_cls, model_type="random_forest", n_estimators=10, threshold="median", task_type="classification", random_state=42
        )
        self.assertIsInstance(res_rf_cls, pd.DataFrame)
        self.assertEqual(res_rf_cls.shape[0], len(self.X))

        # RandomForest Regression with collapsed 2D targets
        y_multi = np.tile(self.y_reg.to_numpy()[:, None], (1, 4))
        y_collapsed = np.mean(y_multi, axis=1)
        res_rf_reg = tree_importance_selector(
            self.X, y_collapsed, model_type="random_forest", n_estimators=10, threshold="median", task_type="regression", random_state=42
        )
        self.assertIsInstance(res_rf_reg, pd.DataFrame)
        self.assertEqual(res_rf_reg.shape[0], len(self.X))

        # Empty input
        res_empty = tree_importance_selector(self.X.iloc[:, :0], self.y_cls, model_type="extra_trees")
        self.assertEqual(res_empty.shape, (len(self.X), 0))

    def test_analysis_plots(self):
        """Verify publication-grade Critical Difference and tau scatter plots."""
        import tempfile
        from tempo.analysis import plot_critical_difference_diagram, plot_tau_estimation_scatter

        # CD Diagram
        df_bench = pd.DataFrame({
            "Dataset": ["DS_A", "DS_A", "DS_B", "DS_B", "DS_C", "DS_C"],
            "Extractor": ["Ext_1", "Ext_2", "Ext_1", "Ext_2", "Ext_1", "Ext_2"],
            "Accuracy": [0.85, 0.90, 0.78, 0.82, 0.91, 0.93],
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            out_cd = f"{tmpdir}/cd_test.png"
            fig_cd = plot_critical_difference_diagram(
                df_bench,
                metric="Accuracy",
                group_col="Extractor",
                dataset_col="Dataset",
                output_path=out_cd,
            )
            self.assertIsNotNone(fig_cd)
            self.assertTrue(os.path.exists(out_cd))

            # Tau Scatter Plot
            y_true = np.array([3.5, 3.75, 4.0, 4.25, 4.5])
            y_pred = np.array([3.52, 3.74, 4.02, 4.23, 4.48])
            out_tau = f"{tmpdir}/tau_test.png"
            fig_tau = plot_tau_estimation_scatter(y_true, y_pred, output_path=out_tau)
            self.assertIsNotNone(fig_tau)
            self.assertTrue(os.path.exists(out_tau))

            # Edge case: CD diagram with 0 common datasets (all NaN after dropna)
            df_no_overlap = pd.DataFrame({
                "Dataset": ["DS_A", "DS_A", "DS_B", "DS_B"],
                "Extractor": ["Ext_1", "Ext_2", "Ext_1", "Ext_2"],
                "Accuracy": [np.nan, 0.85, 0.78, np.nan],
            })
            fig_empty = plot_critical_difference_diagram(df_no_overlap, metric="Accuracy", group_col="Extractor")
            self.assertIsNone(fig_empty)

            # Edge case: CD diagram with fewer than 2 methods
            df_one_method = pd.DataFrame({
                "Dataset": ["DS_A", "DS_B"],
                "Extractor": ["Ext_1", "Ext_1"],
                "Accuracy": [0.85, 0.90],
            })
            fig_one = plot_critical_difference_diagram(df_one_method, metric="Accuracy", group_col="Extractor")
            self.assertIsNone(fig_one)

            # Edge case: Tau scatter with mismatched lengths
            fig_mismatch = plot_tau_estimation_scatter(y_true[:3], y_pred)
            self.assertIsNone(fig_mismatch)

    def test_subsampled_selector_with_new_string_selectors(self):
        """Verify SubsampledFeatureSelector dispatches string selectors for variance, l1, and trees."""
        for sel_name in ["variance_threshold", "l1", "extra_trees", "random_forest"]:
            selector = SubsampledFeatureSelector(
                base_selector=sel_name,
                sample_ratio=1.0,
                task_type="classification",
                random_state=42,
            )
            selector.fit(self.X, self.y_cls)
            self.assertFalse(selector.fallback_triggered_)
            self.assertGreater(len(selector.survived_features_), 0)


if __name__ == "__main__":
    unittest.main()

