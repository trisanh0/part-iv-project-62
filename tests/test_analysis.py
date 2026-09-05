"""Unit tests for statistical analysis and evaluation engine."""

import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from tempo.analysis import pairwise_ttests, compute_summary_statistics, run_statistical_analysis


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


if __name__ == "__main__":
    unittest.main()
