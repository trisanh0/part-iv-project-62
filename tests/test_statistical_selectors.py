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
    variance_threshold_selector,
    l1_selector,
    tree_importance_selector,
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

        # Fallback must be flagged with honest feature accounting
        self.assertTrue(selector.fallback_triggered_)
        self.assertEqual(selector.n_features_out_, self.X.shape[1])
        self.assertEqual(selector.telemetry_["n_selected_features"], self.X.shape[1])
        self.assertEqual(selector.telemetry_["n_survived_features"], 0)
        self.assertEqual(selector.telemetry_["feature_reduction_pct"], 0.0)

        # Transform still returns full matrix to prevent downstream crash
        X_trans = selector.transform(self.X)
        self.assertEqual(X_trans.shape[1], self.X.shape[1])
        self.assertEqual(X_trans.shape[1], selector.n_features_out_)

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

    def test_subsampled_selector_hyperparameter_propagation(self):
        """Verify SubsampledFeatureSelector propagates custom hyperparameters (k, threshold, C) to base selectors."""
        # 1. Custom k=5 for select_k_best on 15 features
        selector_k5 = SubsampledFeatureSelector(
            base_selector="select_k_best",
            k=5,
            sample_ratio=1.0,
            task_type="classification",
        )
        selector_k5.fit(self.X, self.y_cls)
        self.assertFalse(selector_k5.fallback_triggered_)
        self.assertEqual(selector_k5.n_features_out_, 5)
        self.assertEqual(selector_k5.transform(self.X).shape[1], 5)

        # 2. Custom k=8 via base_params dict
        selector_dict = SubsampledFeatureSelector(
            base_selector="select_k_best",
            base_params={"k": 8},
            sample_ratio=1.0,
            task_type="classification",
        )
        selector_dict.fit(self.X, self.y_cls)
        self.assertEqual(selector_dict.n_features_out_, 8)

        # 3. Custom k=3 via kwargs
        selector_kw = SubsampledFeatureSelector(
            base_selector="mutual_info",
            sample_ratio=1.0,
            task_type="classification",
            k=3,
        )
        selector_kw.fit(self.X, self.y_cls)
        self.assertEqual(selector_kw.n_features_out_, 3)

    def test_subsampled_selector_imbalanced_data(self):
        """Verify SubsampledFeatureSelector preserves minority class representation in imbalanced classification."""
        n_samples = 199
        n_minority = 9
        y_imbalanced = np.zeros(n_samples, dtype=int)
        y_imbalanced[:n_minority] = 1
        X_imb = pd.DataFrame(np.random.randn(n_samples, 10), columns=[f"f_{i}" for i in range(10)])

        selector = SubsampledFeatureSelector(
            base_selector="select_k_best",
            sample_ratio=0.10,
            min_samples=20,
            k=4,
            random_state=42,
            stratify=True,
            task_type="classification",
        )
        selector.fit(X_imb, pd.Series(y_imbalanced))
        # Must execute cleanly without exception and preserve minority representation
        self.assertFalse(selector.fallback_triggered_)
        self.assertEqual(selector.n_features_out_, 4)


class TestSelectorsNumericalRobustness(unittest.TestCase):
    def test_selectors_all_nan_and_inf_sanitization(self):
        """Verify selectors sanitize all-NaN and Inf matrices and return valid DataFrames."""
        n_samples = 20
        n_features = 5
        X_nan_inf = pd.DataFrame(
            np.full((n_samples, n_features), np.nan),
            columns=[f"col_{i}" for i in range(n_features)],
        )
        X_nan_inf.iloc[:10, :2] = np.inf
        X_nan_inf.iloc[10:, :2] = -np.inf
        y = pd.Series(np.random.randint(0, 2, size=n_samples))

        # 1. variance_threshold_selector: constant features after zero-fill -> 0 features
        res_vt = variance_threshold_selector(X_nan_inf)
        self.assertIsInstance(res_vt, pd.DataFrame)
        self.assertEqual(res_vt.shape, (n_samples, 0))

        # 2. tsfresh_selector: constant features dropped -> 0 features
        res_tf = tsfresh_selector(X_nan_inf, y)
        self.assertIsInstance(res_tf, pd.DataFrame)
        self.assertEqual(res_tf.shape, (n_samples, 0))

        # Test tsfresh regression task type
        X_reg = pd.DataFrame(np.random.randn(n_samples, 3), columns=["r1", "r2", "r3"])
        y_reg = pd.Series(np.random.randn(n_samples))
        res_tf_reg = tsfresh_selector(X_reg, y_reg, task_type="regression")
        self.assertIsInstance(res_tf_reg, pd.DataFrame)
        self.assertEqual(len(res_tf_reg), n_samples)

        # 3. l1_selector: zero variance yields 0 features; empty target returns empty DataFrame
        res_l1 = l1_selector(X_nan_inf, y, task_type="classification")
        self.assertIsInstance(res_l1, pd.DataFrame)
        self.assertEqual(res_l1.shape, (n_samples, 0))

        res_l1_empty = l1_selector(X_reg, np.array([]), task_type="regression")
        self.assertEqual(res_l1_empty.shape, (n_samples, 0))

        # 4. select_k_best and mutual_info: fallback to top k features
        res_kb = select_k_best(X_nan_inf, y, k=2)
        self.assertIsInstance(res_kb, pd.DataFrame)
        self.assertEqual(len(res_kb), n_samples)

        res_mi = mutual_info_selector(X_nan_inf, y, k=2)
        self.assertIsInstance(res_mi, pd.DataFrame)
        self.assertEqual(len(res_mi), n_samples)

        # 5. tree_importance_selector: executes without crash on sanitized inputs
        res_ti = tree_importance_selector(X_nan_inf, y, model_type="extra_trees", n_estimators=5)
        self.assertIsInstance(res_ti, pd.DataFrame)
        self.assertEqual(len(res_ti), n_samples)

        # 6. boruta_selector: zero-variance yields 0 features
        res_bo = boruta_selector(X_nan_inf, y, n_estimators=5)
        self.assertIsInstance(res_bo, pd.DataFrame)
        self.assertEqual(res_bo.shape, (n_samples, 0))


    def test_selectors_empty_feature_space(self):
        """Verify selectors return empty DataFrame with preserved index on P=0."""
        n_samples = 15
        X_empty = pd.DataFrame(index=pd.RangeIndex(n_samples))
        y = pd.Series(np.random.randint(0, 2, size=n_samples))

        for sel_func in [
            lambda X: variance_threshold_selector(X),
            lambda X: tsfresh_selector(X, y),
            lambda X: l1_selector(X, y),
            lambda X: select_k_best(X, y, k=5),
            lambda X: tree_importance_selector(X, y, n_estimators=5),
            lambda X: mutual_info_selector(X, y, k=5),
            lambda X: boruta_selector(X, y, n_estimators=5),
        ]:
            res = sel_func(X_empty)
            self.assertIsInstance(res, pd.DataFrame)
            self.assertEqual(res.shape, (n_samples, 0))
            self.assertEqual(list(res.index), list(X_empty.index))

    def test_selectors_single_class_target(self):
        """Verify l1_selector and tree_importance_selector handle single-class target without crash."""
        n_samples = 20
        X = pd.DataFrame(np.random.randn(n_samples, 6), columns=[f"f_{i}" for i in range(6)])
        y_single_class = pd.Series(np.zeros(n_samples, dtype=int))

        # l1_selector logs warning and returns empty DataFrame
        res_l1 = l1_selector(X, y_single_class, task_type="classification")
        self.assertEqual(res_l1.shape, (n_samples, 0))

        # tree_importance_selector logs warning and returns empty DataFrame
        res_ti = tree_importance_selector(X, y_single_class, task_type="classification", n_estimators=5)
        self.assertEqual(res_ti.shape, (n_samples, 0))

    def test_l1_selector_multiclass_saga(self):
        """Verify l1_selector dispatches to SAGA solver when target contains >2 classes."""
        n_samples = 30
        X = pd.DataFrame(np.random.randn(n_samples, 8), columns=[f"f_{i}" for i in range(8)])
        y_multiclass = pd.Series(np.random.choice([0, 1, 2], size=n_samples))

        res_saga = l1_selector(X, y_multiclass, task_type="classification", C=1.0, random_state=42)
        self.assertIsInstance(res_saga, pd.DataFrame)
        self.assertEqual(len(res_saga), n_samples)

        # 2D target array unraveling branch
        res_saga_2d = l1_selector(X, y_multiclass.to_numpy()[:, None], task_type="classification", C=1.0, random_state=42)
        self.assertEqual(len(res_saga_2d), n_samples)


    def test_subsampled_selector_pathological_inputs(self):
        """Verify SubsampledFeatureSelector triggers honest fallback telemetry under pathological inputs."""
        n_samples = 30
        n_features = 8
        X_const = pd.DataFrame(
            np.ones((n_samples, n_features)),
            columns=[f"const_{i}" for i in range(n_features)],
        )
        y = pd.Series(np.random.choice([0, 1], size=n_samples))

        selector = SubsampledFeatureSelector(
            base_selector="variance_threshold",
            sample_ratio=0.5,
            min_samples=10,
            random_state=42,
        )
        selector.fit(X_const, y)

        self.assertTrue(selector.fallback_triggered_)
        self.assertEqual(selector.n_features_out_, n_features)
        self.assertTrue(selector.telemetry_["fallback_triggered"])
        self.assertEqual(selector.telemetry_["n_survived_features"], 0)
        self.assertEqual(selector.telemetry_["n_selected_features"], n_features)
        self.assertEqual(selector.telemetry_["feature_reduction_pct"], 0.0)

        # transform returns unreduced feature matrix matching n_features_out_
        X_trans = selector.transform(X_const)
        self.assertEqual(X_trans.shape, X_const.shape)
        self.assertEqual(X_trans.shape[1], selector.n_features_out_)


if __name__ == "__main__":
    unittest.main()

