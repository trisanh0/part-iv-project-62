"""Unit tests for statistical analysis and evaluation engine."""

import json
import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from tempo.analysis import (
    compute_summary_statistics,
    pairwise_ttests,
    plot_horizon_metric_profiles,
    plot_tau_recovery_points,
    run_statistical_analysis,
)


class TestAnalysis(unittest.TestCase):
    def test_pairwise_ttests_no_cartesian_pseudo_replication(self):
        """Verify pairwise_ttests matches pairs on (Dataset, Seed, Fold, Selector, Model) without Cartesian explosion."""
        records = []
        extractors = ["ext_A", "ext_B"]
        selectors = ["sel_1", "sel_2"]
        models = ["rf", "ridge"]
        seeds = [42, 43]
        folds = [0, 1]

        # 2 extractors x 2 selectors x 2 models x 2 seeds x 2 folds = 32 records (16 per extractor)
        for ext in extractors:
            for sel in selectors:
                for model in models:
                    for seed in seeds:
                        for fold in folds:
                            score = 0.85 if ext == "ext_A" else 0.80
                            records.append({
                                "Dataset": "test_ds",
                                "Task": "classification",
                                "Seed": seed,
                                "Fold": fold,
                                "Extractor": ext,
                                "Selector": sel,
                                "Model": model,
                                "Accuracy": score + np.random.randn() * 0.01,
                            })

        df = pd.DataFrame(records)
        ttest_res = pairwise_ttests(df, group_col="Extractor", metric="Accuracy")

        self.assertFalse(ttest_res.empty)
        # N pairs MUST equal 16 (the exact number of matched conditions), NOT 16 * 4 = 64 (pseudo-replicated)
        self.assertEqual(ttest_res["N Pairs"].iloc[0], 16)

    def test_run_statistical_analysis_separate_tasks(self):
        """Verify separate classification and regression summaries are created without NaN metric averaging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "benchmark.csv"
            records = [
                # Classification record
                {
                    "Dataset": "ds_cls",
                    "Task": "classification",
                    "Seed": 42,
                    "Fold": 0,
                    "Extractor": "ext_A",
                    "Selector": "None",
                    "Model": "rf",
                    "Accuracy": 0.90,
                    "RMSE": np.nan,
                    "MAE": np.nan,
                    "R2": np.nan,
                    "Extraction Time (s)": 1.5,
                    "is_cached": False,
                },
                # Regression record
                {
                    "Dataset": "ds_reg",
                    "Task": "regression",
                    "Seed": 42,
                    "Fold": 0,
                    "Extractor": "ext_A",
                    "Selector": "None",
                    "Model": "ridge",
                    "Accuracy": np.nan,
                    "RMSE": 2.5,
                    "MAE": 1.8,
                    "R2": 0.75,
                    "Extraction Time (s)": 2.0,
                    "is_cached": False,
                },
            ]
            pd.DataFrame(records).to_csv(csv_path, index=False)

            out_dir = Path(tmpdir) / "analysis"
            outputs = run_statistical_analysis(
                csv_path=csv_path,
                output_dir=out_dir,
                enable_ttests=False,
                enable_plots=False,
            )

            # Both task-specific summaries should exist
            self.assertIn("summary_classification", outputs)
            self.assertIn("summary_regression", outputs)

            # Classification summary should have non-null Accuracy and no RMSE/MAE
            cls_sum = outputs["summary_classification"]
            self.assertIn(("Accuracy", "mean"), cls_sum.columns)
            self.assertFalse(np.isnan(cls_sum[("Accuracy", "mean")].iloc[0]))
            self.assertNotIn(("RMSE", "mean"), cls_sum.columns)

            # Regression summary should have non-null RMSE and no Accuracy
            reg_sum = outputs["summary_regression"]
            self.assertIn(("RMSE", "mean"), reg_sum.columns)
            self.assertFalse(np.isnan(reg_sum[("RMSE", "mean")].iloc[0]))
            self.assertNotIn(("Accuracy", "mean"), reg_sum.columns)

    def test_pairwise_ttests_combination_group_col(self):
        """Verify pairwise_ttests with group_col='Combination' correctly pairs across models and seeds."""
        records = []
        combos = ["ext_A + sel_1", "ext_B + sel_2"]
        seeds = [42, 43, 44]
        models = ["rf", "ridge"]

        for combo in combos:
            ext, sel = combo.split(" + ")
            base_acc = 0.85 if "ext_A" in ext else 0.75
            for seed in seeds:
                for model in models:
                    records.append({
                        "Dataset": "test_ds",
                        "Task": "classification",
                        "Seed": seed,
                        "Extractor": ext,
                        "Selector": sel,
                        "Combination": combo,
                        "Model": model,
                        "Accuracy": base_acc + np.random.randn() * 0.005,
                    })

        df = pd.DataFrame(records)
        ttest_res = pairwise_ttests(df, group_col="Combination", metric="Accuracy")

        self.assertFalse(ttest_res.empty, "Combination t-tests should not be empty")
        self.assertEqual(ttest_res["N Pairs"].iloc[0], 6)  # 3 seeds x 2 models = 6 matched pairs
        self.assertEqual(ttest_res["Group 1"].iloc[0], "ext_A + sel_1")
        self.assertEqual(ttest_res["Group 2"].iloc[0], "ext_B + sel_2")

    def test_metric_direction_latency_and_fit_time(self):
        """Verify metric_direction and winner logic correctly favor lower fit time and inference latency."""
        from tempo.analysis import metric_direction
        self.assertEqual(metric_direction("fit_time_seconds"), "lower")
        self.assertEqual(metric_direction("inference_latency_ms"), "lower")
        self.assertEqual(metric_direction("Fit Time (s)"), "lower")
        self.assertEqual(metric_direction("Inference Latency (ms)"), "lower")

        # Create test where ext_fast has lower fit time than ext_slow
        df = pd.DataFrame([
            {"Dataset": "ds", "Task": "cls", "Seed": 42, "Extractor": "ext_fast", "fit_time_seconds": 1.2, "inference_latency_ms": 0.5},
            {"Dataset": "ds", "Task": "cls", "Seed": 43, "Extractor": "ext_fast", "fit_time_seconds": 1.3, "inference_latency_ms": 0.6},
            {"Dataset": "ds", "Task": "cls", "Seed": 42, "Extractor": "ext_slow", "fit_time_seconds": 5.0, "inference_latency_ms": 2.0},
            {"Dataset": "ds", "Task": "cls", "Seed": 43, "Extractor": "ext_slow", "fit_time_seconds": 5.2, "inference_latency_ms": 2.1},
        ])
        res_fit = pairwise_ttests(df, group_col="Extractor", metric="fit_time_seconds")
        self.assertFalse(res_fit.empty)
        # ext_fast has lower time, so it should be the winner
        self.assertEqual(res_fit["Winner (Pre-Correction)"].iloc[0], "ext_fast")

        res_lat = pairwise_ttests(df, group_col="Extractor", metric="inference_latency_ms")
        self.assertFalse(res_lat.empty)
        self.assertEqual(res_lat["Winner (Pre-Correction)"].iloc[0], "ext_fast")

    def test_plot_horizon_metric_profiles_flexible_matching(self):
        """Verify plot_horizon_metric_profiles parses standard and Scott's variant column formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "plots"

            # Case 1: Standard underscore format (e.g. Interval_Width_H1, RMSE_H1)
            df_std = pd.DataFrame([
                {
                    "Combination": "Model_A",
                    "RMSE_H1": 0.20, "RMSE_H2": 0.25,
                    "Interval_Width_H1": 0.50, "Interval_Width_H2": 0.60,
                },
                {
                    "Combination": "Model_B",
                    "RMSE_H1": 0.30, "RMSE_H2": 0.35,
                    "Interval_Width_H1": 0.70, "Interval_Width_H2": 0.80,
                },
            ])
            figs_std = plot_horizon_metric_profiles(df_std, output_dir=out_dir)
            self.assertIn("RMSE", figs_std)
            self.assertIn("Interval_Width", figs_std)
            self.assertIn("RMSE_Model_A", figs_std)
            self.assertIn("Interval_Width_Model_A", figs_std)
            self.assertTrue((out_dir / "horizon_rmse_profile.png").exists())
            self.assertTrue((out_dir / "horizon_interval_width_profile.png").exists())
            self.assertTrue((out_dir / "horizon_interval_width_Model_A.png").exists())
            self.assertTrue((out_dir / "interval_width_horizon_Model_A.png").exists())
            self.assertTrue((out_dir / "interval_width_across_horizon_all_combinations.png").exists())

            # Case 2: Scott's space-delimited format with H (e.g. Interval Width H1, RMSE H1)
            out_dir_scott_h = Path(tmpdir) / "scott_h"
            df_scott_h = pd.DataFrame([
                {
                    "Combination": "Combo_X",
                    "RMSE H1": 0.15, "RMSE H2": 0.22,
                    "Interval Width H1": 0.40, "Interval Width H2": 0.55,
                },
                {
                    "Combination": "Combo_Y",
                    "RMSE H1": 0.25, "RMSE H2": 0.32,
                    "Interval Width H1": 0.60, "Interval Width H2": 0.75,
                },
            ])
            figs_scott_h = plot_horizon_metric_profiles(df_scott_h, output_dir=out_dir_scott_h)
            self.assertIn("RMSE", figs_scott_h)
            self.assertIn("Interval_Width", figs_scott_h)
            self.assertTrue((out_dir_scott_h / "horizon_rmse_profile.png").exists())
            self.assertTrue((out_dir_scott_h / "horizon_interval_width_Combo_X.png").exists())
            self.assertTrue((out_dir_scott_h / "interval_width_horizon_Combo_X.png").exists())

            # Case 3: Scott's space-delimited format without H (e.g. Interval Width 1, RMSE 1)
            out_dir_scott_num = Path(tmpdir) / "scott_num"
            df_scott_num = pd.DataFrame([
                {
                    "Combination": "Combo_Z",
                    "RMSE 1": 0.18, "RMSE 2": 0.24,
                    "Interval Width 1": 0.45, "Interval Width 2": 0.58,
                },
            ])
            figs_scott_num = plot_horizon_metric_profiles(df_scott_num, output_dir=out_dir_scott_num)
            self.assertIn("RMSE", figs_scott_num)
            self.assertIn("Interval_Width", figs_scott_num)
            self.assertTrue((out_dir_scott_num / "horizon_rmse_profile.png").exists())
            self.assertTrue((out_dir_scott_num / "horizon_interval_width_Combo_Z.png").exists())

            # Case 4: Non-consecutive horizons (e.g. H1, H3, H5)
            out_dir_non_consec = Path(tmpdir) / "non_consec"
            df_non_consec = pd.DataFrame([
                {
                    "Combination": "Model_Skip",
                    "RMSE_H1": 0.10, "RMSE_H3": 0.25, "RMSE_H5": 0.40,
                },
            ])
            figs_nc = plot_horizon_metric_profiles(df_non_consec, output_dir=out_dir_non_consec)
            self.assertIn("RMSE", figs_nc)
            self.assertTrue((out_dir_non_consec / "horizon_rmse_profile.png").exists())
            self.assertTrue((out_dir_non_consec / "horizon_rmse_Model_Skip.png").exists())

            # Case 5: Duplicate column representations preferring non-null columns
            out_dir_dup = Path(tmpdir) / "dup_cols"
            df_dup = pd.DataFrame([
                {
                    "Combination": "Model_Dup",
                    "RMSE_H1": np.nan,  # NaN in standard format
                    "RMSE H1": 0.33,    # Valid in variant format
                },
            ])
            figs_dup = plot_horizon_metric_profiles(df_dup, output_dir=out_dir_dup)
            self.assertIn("RMSE", figs_dup)
            self.assertTrue((out_dir_dup / "horizon_rmse_profile.png").exists())

            # Case 6: Mixed types in Combination column
            out_dir_mixed = Path(tmpdir) / "mixed_types"
            df_mixed = pd.DataFrame([
                {"Combination": 101, "RMSE_H1": 0.2},
                {"Combination": "Combo_Text", "RMSE_H1": 0.3},
            ])
            figs_mixed = plot_horizon_metric_profiles(df_mixed, output_dir=out_dir_mixed)
            self.assertIn("RMSE", figs_mixed)
            self.assertTrue((out_dir_mixed / "horizon_rmse_profile.png").exists())

    def test_generate_forecast_visualizations_without_raw_arrays(self):
        """Verify generate_forecast_visualizations generates horizon profiles even without raw arrays."""
        from tempo.analysis import generate_forecast_visualizations
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "fc_viz"
            df = pd.DataFrame([
                {
                    "Combination": "Model_1",
                    "Mean Interval Width": 0.45,
                    "Interval_Width_H1": 0.40,
                    "Interval_Width_H2": 0.50,
                    "RMSE_H1": 0.15,
                    "RMSE_H2": 0.22,
                },
                {
                    "Combination": "Model_2",
                    "Mean Interval Width": 0.55,
                    "Interval_Width_H1": 0.50,
                    "Interval_Width_H2": 0.60,
                    "RMSE_H1": 0.20,
                    "RMSE_H2": 0.28,
                },
            ])
            gen_files = generate_forecast_visualizations(df, output_dir=out_dir)
            self.assertGreater(len(gen_files), 0)
            self.assertTrue((out_dir / "horizon_interval_width_profile.png").exists())
            self.assertTrue((out_dir / "uncertainty_interval_width_all_combinations.png").exists())
            self.assertTrue((out_dir / "uncertainty_interval_width_mean_all_combinations.png").exists())

    def test_run_statistical_analysis_discrete_tau_with_array_inputs(self):
        """Verify run_statistical_analysis generates tau recovery plots from list/array inputs without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "tau_results.csv"
            out_dir = Path(tmpdir) / "analysis"
            df = pd.DataFrame([
                {
                    "Dataset": "drift_seed_42",
                    "Seed": 42,
                    "Extractor": "np_stat",
                    "Selector": "none",
                    "Combination": "np_stat_none",
                    "True Tau Values": json.dumps([0.1, 0.2, 0.3, 0.4, 0.5]),
                    "Test True Tau Values": json.dumps([0.1, 0.1, 0.2, 0.3, 0.4, 0.5]),
                    "Predicted Tau Values": json.dumps([0.11, 0.09, 0.21, 0.29, 0.42, 0.48]),
                    "Accuracy": 0.85,
                }
            ])
            df.to_csv(csv_path, index=False)
            res = run_statistical_analysis(
                csv_path=csv_path,
                output_dir=out_dir,
                task_type="regression",
                enable_ttests=False,
                enable_plots=True,
            )
            self.assertTrue((out_dir / "tau_recovery_np_stat_none.png").exists())
            self.assertTrue((out_dir / "tau_prediction_np_stat_none.png").exists())
            self.assertIn("tau_recovery_np_stat_none", res)

    def test_plot_tau_recovery_points_valid_and_edge_cases(self):
        """Verify plot_tau_recovery_points handles valid inputs and edge cases robustly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "tau_recovery.png"

            # 1. Standard 5-point discrete recovery
            taus = [0.1, 0.2, 0.3, 0.4, 0.5]
            preds = [0.11, 0.19, 0.32, 0.39, 0.51]
            fig = plot_tau_recovery_points(taus, preds, title="Tau Test", output_path=out_file)
            self.assertIsNotNone(fig)
            self.assertTrue(out_file.exists())

            # 2. NumPy arrays and Pandas Series
            fig_np = plot_tau_recovery_points(np.array(taus), np.array(preds))
            self.assertIsNotNone(fig_np)
            fig_pd = plot_tau_recovery_points(pd.Series(taus), pd.Series(preds))
            self.assertIsNotNone(fig_pd)

            # 3. Partial NaN / non-uniform finite values
            taus_nan = [0.1, np.nan, 0.3, 0.4, 0.5]
            preds_nan = [0.12, 0.21, np.nan, 0.38, 0.49]
            fig_nan = plot_tau_recovery_points(taus_nan, preds_nan)
            self.assertIsNotNone(fig_nan)

            # 4. Edge cases returning None
            self.assertIsNone(plot_tau_recovery_points([], []))
            self.assertIsNone(plot_tau_recovery_points([0.1, 0.2], [0.1]))
            self.assertIsNone(plot_tau_recovery_points(None, [0.1]))
            self.assertIsNone(plot_tau_recovery_points([0.1], None))
            self.assertIsNone(plot_tau_recovery_points([np.nan, np.nan], [np.nan, np.nan]))


if __name__ == "__main__":
    unittest.main()
