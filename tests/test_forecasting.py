"""Unit tests for TEMPO time-series forecasting architecture.

Validates:
1. Synthetic forecasting dataset generation and standardized SDF validation.
2. Causal chronological temporal integrity and absence of future lookahead bias.
3. Sliding-window segmentation and 3D feature tensor / 2D target matrix conversions.
4. Backwards compatibility for forecast_horizon in segment_time_series.
5. Multi-output RandomForestRegressor and empirical tree-quantile interval estimation.
6. Horizon-wise metric calculations (RMSE, MAE, Coverage, Interval Width).
7. End-to-end unified benchmarking pipeline with task_type='forecasting'.
8. Publication-grade fan chart and horizon profile visualizations.
9. Scale boundaries, single-step edge cases (H=1), and error handling.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import RandomForestRegressor

from tempo.analysis import (
    generate_forecast_visualizations,
    metric_direction,
    plot_forecast_fan_chart,
    plot_horizon_metric_profiles,
    plot_interval_width_boxplot,
    plot_interval_width_mean_bar,
    run_statistical_analysis,
)
from tempo.benchmark import BakeoffRunner, PipelineConfig
from tempo.storage import (
    generate_simulated_forecasting_dataset,
    load_dataset,
    segment_forecasting_series,
    segment_time_series,
    to_forecasting_tensors,
    validate_export,
)


class TestForecasting(unittest.TestCase):
    def test_generate_and_validate_simulated_forecasting_dataset(self) -> None:
        """Verify synthetic forecasting dataset generation and standardized schema validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_dir = os.path.join(tmpdir, "simulated_forecasting")
            df_ts, df_targets = generate_simulated_forecasting_dataset(
                output_dir=ds_dir,
                n_series=5,
                series_len=120,
                history_len=40,
                forecast_horizon=10,
                test_size=0.25,
                seed=42,
            )

            ts_path = os.path.join(ds_dir, "time_series.parquet")
            tgt_path = os.path.join(ds_dir, "targets.parquet")
            self.assertTrue(os.path.exists(ts_path))
            self.assertTrue(os.path.exists(tgt_path))

            # Schema validation
            self.assertTrue(validate_export(ds_dir))

            # Verify DataFrame schemas and columns
            self.assertIn("sequence_id", df_ts.columns)
            self.assertIn("step", df_ts.columns)
            self.assertIn("value", df_ts.columns)

            self.assertIn("sequence_id", df_targets.columns)
            self.assertIn("split", df_targets.columns)
            self.assertIn("target", df_targets.columns)
            for h in range(10):
                self.assertIn(f"target_{h}", df_targets.columns)

            splits = set(df_targets["split"].unique().to_list())
            self.assertEqual(splits, {"train", "test"})

            # Test auto-loader
            loaded_ts, loaded_tgt = load_dataset("simulated_forecasting", processed_dir=tmpdir)
            self.assertEqual(loaded_ts.height, df_ts.height)
            self.assertEqual(loaded_tgt.height, df_targets.height)

    def test_causal_temporal_integrity_no_leakage(self) -> None:
        """Verify that training and testing partitions enforce strict causal chronological boundaries."""
        series_len = 200
        history_len = 40
        forecast_horizon = 10
        stride = 5
        test_size = 0.20

        # Monotonically increasing time points to verify indices exactly
        df_raw = pl.DataFrame({
            "series_id": np.zeros(series_len, dtype=np.int32),
            "time": np.arange(series_len, dtype=np.int32),
            "value": np.arange(series_len, dtype=np.float32),
            "target": np.arange(series_len, dtype=np.float32),
        })

        df_ts, df_targets = segment_forecasting_series(
            df=df_raw,
            history_len=history_len,
            forecast_horizon=forecast_horizon,
            stride=stride,
            time_col="time",
            feature_cols=["value"],
            target_col="target",
            group_col="series_id",
            test_size=test_size,
        )

        split_boundary = int(series_len * (1.0 - test_size))  # 160

        train_targets = df_targets.filter(pl.col("split") == "train")
        test_targets = df_targets.filter(pl.col("split") == "test")

        self.assertGreater(train_targets.height, 0)
        self.assertGreater(test_targets.height, 0)

        # In all training samples, forecast target values must be strictly < split_boundary
        for row in train_targets.iter_rows(named=True):
            tgt_vals = row["target"]
            self.assertEqual(len(tgt_vals), forecast_horizon)
            self.assertLessEqual(max(tgt_vals), split_boundary - 1)

        # In all testing samples, forecast target values must be strictly >= split_boundary
        for row in test_targets.iter_rows(named=True):
            tgt_vals = row["target"]
            self.assertEqual(len(tgt_vals), forecast_horizon)
            self.assertGreaterEqual(min(tgt_vals), split_boundary)

        # Verify zero temporal target overlap between train and test
        max_train_time = max(max(r["target"]) for r in train_targets.iter_rows(named=True))
        min_test_time = min(min(r["target"]) for r in test_targets.iter_rows(named=True))
        self.assertLess(max_train_time, min_test_time)

    def test_segment_forecasting_series_and_tensors(self) -> None:
        """Verify windowing and conversion to (n_samples, W, C) features and (n_samples, H) targets."""
        n_points = 150
        df_raw = pl.DataFrame({
            "series_id": np.repeat([0, 1], n_points // 2),
            "time": np.tile(np.arange(n_points // 2, dtype=np.int32), 2),
            "feat_1": np.random.randn(n_points).astype(np.float32),
            "feat_2": np.random.randn(n_points).astype(np.float32),
            "target": np.random.randn(n_points).astype(np.float32),
        })

        history_len = 25
        forecast_horizon = 8
        df_ts, df_targets = segment_forecasting_series(
            df=df_raw,
            history_len=history_len,
            forecast_horizon=forecast_horizon,
            stride=2,
            feature_cols=["feat_1", "feat_2"],
            target_col="target",
            group_col="series_id",
        )

        X, y = to_forecasting_tensors(df_ts, df_targets, ensure_3d=True)

        self.assertEqual(X.ndim, 3)
        self.assertEqual(X.shape[1], history_len)
        self.assertEqual(X.shape[2], 2)
        self.assertEqual(y.ndim, 2)
        self.assertEqual(y.shape[1], forecast_horizon)
        self.assertEqual(X.shape[0], y.shape[0])
        self.assertEqual(X.shape[0], df_targets.height)

        # Single-channel 2D tensor option
        X_2d, _ = to_forecasting_tensors(
            df_ts, df_targets, feature_cols=["feat_1"], ensure_3d=False
        )
        self.assertEqual(X_2d.ndim, 2)
        self.assertEqual(X_2d.shape, (df_targets.height, history_len))

    def test_segment_time_series_with_forecast_horizon(self) -> None:
        """Verify segment_time_series backwards-compatible forecast_horizon support."""
        df_raw = pl.DataFrame({
            "time": np.arange(300, dtype=np.int32),
            "sensor": np.random.randn(300).astype(np.float32),
            "target": np.random.randn(300).astype(np.float32),
        })

        df_ts, df_targets = segment_time_series(
            df_raw,
            window_size=50,
            stride=25,
            time_col="time",
            feature_cols=["sensor"],
            label_col="target",
            forecast_horizon=6,
        )

        self.assertIn("target", df_targets.columns)
        self.assertIn("target_0", df_targets.columns)
        self.assertIn("target_5", df_targets.columns)
        self.assertEqual(df_targets.height, (300 - 50 - 6) // 25 + 1)

    def test_multioutput_random_forest_and_tree_quantiles(self) -> None:
        """Verify multi-output RandomForestRegressor and empirical tree-quantile interval bounds."""
        n_samples = 80
        n_features = 15
        horizon = 6
        rng = np.random.default_rng(42)

        X = rng.normal(size=(n_samples, n_features))
        # Multi-output target
        y = np.column_stack([X[:, 0] * 1.5 + X[:, 1] * (h + 1) * 0.2 for h in range(horizon)])

        rf = RandomForestRegressor(n_estimators=30, random_state=42)
        rf.fit(X, y)

        preds = rf.predict(X)
        self.assertEqual(preds.shape, (n_samples, horizon))

        # Empirical tree quantiles
        tree_preds = np.stack([tree.predict(X) for tree in rf.estimators_], axis=0)
        self.assertEqual(tree_preds.shape, (30, n_samples, horizon))

        lower = np.percentile(tree_preds, 5.0, axis=0)
        upper = np.percentile(tree_preds, 95.0, axis=0)

        self.assertEqual(lower.shape, (n_samples, horizon))
        self.assertEqual(upper.shape, (n_samples, horizon))
        self.assertTrue(np.all(lower <= upper))

        # Coverage of training points
        coverage = np.mean((y >= lower) & (y <= upper))
        self.assertGreater(coverage, 0.80)

    def test_horizon_wise_metrics_calculation(self) -> None:
        """Verify step-by-step horizon metric formulas (RMSE, MAE, coverage, width)."""
        y_true = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
        ], dtype=np.float32)

        y_pred = np.array([
            [1.5, 2.0, 2.0],
            [1.0, 5.0, 6.0],
        ], dtype=np.float32)

        lower = y_pred - 1.0
        upper = y_pred + 1.0

        expected_rmse_0 = np.sqrt(0.625)
        expected_mae_0 = 0.75

        rmse_0 = np.sqrt(np.mean((y_true[:, 0] - y_pred[:, 0]) ** 2))
        mae_0 = np.mean(np.abs(y_true[:, 0] - y_pred[:, 0]))
        cov_0 = np.mean((y_true[:, 0] >= lower[:, 0]) & (y_true[:, 0] <= upper[:, 0]))
        width_0 = np.mean(upper[:, 0] - lower[:, 0])

        self.assertAlmostEqual(rmse_0, expected_rmse_0, places=5)
        self.assertAlmostEqual(mae_0, expected_mae_0, places=5)
        self.assertEqual(cov_0, 1.0)
        self.assertAlmostEqual(width_0, 2.0, places=5)

    def test_forecasting_benchmark_pipeline_end_to_end(self) -> None:
        """Verify full 5-stage benchmark execution with task_type='forecasting' and feature caching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_dir = Path(tmpdir) / "sim_fc"
            cache_dir = Path(tmpdir) / "cache"
            out_dir = Path(tmpdir) / "out"

            generate_simulated_forecasting_dataset(
                output_dir=str(ds_dir),
                n_series=3,
                series_len=90,
                history_len=30,
                forecast_horizon=6,
                test_size=0.25,
                seed=42,
            )

            cfg = PipelineConfig(
                dataset_paths=[str(ds_dir)],
                task_type="forecasting",
                history_len=30,
                forecast_horizon=6,
                extractors=["numpy_statistical"],
                selectors=[None, "select_k_best"],
                models=["random_forest"],
                n_splits=2,
                cache_backend="parquet",
                cache_dir=str(cache_dir),
                output_dir=str(out_dir),
                enable_ttests=False,
                enable_plots=True,
                enable_logging=False,
                seeds=[42],
            )

            runner = BakeoffRunner(cfg)
            df_res = runner.run()

            self.assertFalse(df_res.empty)
            self.assertEqual(len(df_res), 2)  # 1 extractor * 2 selectors * 1 model * 1 seed

            # Check required forecasting performance metrics
            self.assertIn("Forecast RMSE", df_res.columns)
            self.assertIn("Forecast MAE", df_res.columns)
            self.assertIn("Interval Coverage", df_res.columns)
            self.assertIn("Mean Interval Width", df_res.columns)

            # Check horizon-specific columns
            for h in range(1, 7):
                self.assertIn(f"RMSE_H{h}", df_res.columns)
                self.assertIn(f"MAE_H{h}", df_res.columns)
                self.assertIn(f"Coverage_H{h}", df_res.columns)
                self.assertIn(f"Interval_Width_H{h}", df_res.columns)

            # Check JSON serialization columns
            json_cols = [
                "History Values",
                "True Values",
                "Predictions",
                "Prediction Lower",
                "Prediction Upper",
            ]
            for jc in json_cols:
                self.assertIn(jc, df_res.columns)
                raw_json = df_res[jc].iloc[0]
                self.assertIsInstance(raw_json, str)
                parsed = json.loads(raw_json)
                self.assertIsInstance(parsed, list)
                self.assertGreater(len(parsed), 0)

            # Verify persistent feature store caching: running again without resuming combo hits feature store
            runner2 = BakeoffRunner(cfg)
            df_res2 = runner2.run(resume=False)
            self.assertTrue(df_res2["is_cached"].iloc[0])
            self.assertTrue(np.isnan(df_res2["Extraction Time (s)"].iloc[0]))
            self.assertTrue(np.isnan(df_res2["Extraction Peak RAM (MB)"].iloc[0]))

    def test_fan_chart_and_horizon_visualizations(self) -> None:
        """Verify fan charts, horizon metric profiles, and automated analysis generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "plots"
            out_dir.mkdir(parents=True)

            history = np.sin(np.linspace(0, 4 * np.pi, 50))
            true_future = np.sin(np.linspace(4 * np.pi, 5 * np.pi, 10))
            predictions = {
                "Model A": true_future + 0.1,
                "Model B": true_future - 0.15,
            }
            lower_bounds = {
                "Model A": true_future - 0.2,
                "Model B": true_future - 0.35,
            }
            upper_bounds = {
                "Model A": true_future + 0.4,
                "Model B": true_future + 0.05,
            }

            # 1. Direct fan chart
            fan_path = out_dir / "test_fan.png"
            fig = plot_forecast_fan_chart(
                history=history,
                true_future=true_future,
                predictions=predictions,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                title="Test Fan Chart",
                output_path=fan_path,
            )
            self.assertIsNotNone(fig)
            self.assertTrue(fan_path.exists())

            # 2. Horizon metric profiles
            df_benchmark = pd.DataFrame([
                {
                    "Combination": "Combo_A",
                    "RMSE_H1": 0.2, "RMSE_H2": 0.3, "RMSE_H3": 0.4,
                    "MAE_H1": 0.15, "MAE_H2": 0.25, "MAE_H3": 0.35,
                    "Coverage_H1": 0.95, "Coverage_H2": 0.90, "Coverage_H3": 0.85,
                    "Interval_Width_H1": 0.5, "Interval_Width_H2": 0.6, "Interval_Width_H3": 0.7,
                },
                {
                    "Combination": "Combo_B",
                    "RMSE_H1": 0.25, "RMSE_H2": 0.35, "RMSE_H3": 0.45,
                    "MAE_H1": 0.20, "MAE_H2": 0.30, "MAE_H3": 0.40,
                    "Coverage_H1": 0.92, "Coverage_H2": 0.88, "Coverage_H3": 0.82,
                    "Interval_Width_H1": 0.55, "Interval_Width_H2": 0.65, "Interval_Width_H3": 0.75,
                },
            ])
            profiles = plot_horizon_metric_profiles(df_benchmark, output_dir=out_dir)
            self.assertIn("RMSE", profiles)
            self.assertIn("MAE", profiles)
            self.assertTrue((out_dir / "horizon_rmse_profile.png").exists())

            # Direct testing of interval width visualizations
            df_widths = pd.DataFrame([
                {"Combination": "Combo_A", "Mean Interval Width": 0.50},
                {"Combination": "Combo_A", "Mean Interval Width": 0.55},
                {"Combination": "Combo_B", "Mean Interval Width": 0.70},
                {"Combination": "Combo_B", "Mean Interval Width": 0.75},
            ])
            fig_box = plot_interval_width_boxplot(df_widths, output_dir=out_dir)
            self.assertIsNotNone(fig_box)
            self.assertTrue((out_dir / "uncertainty_interval_width_all_combinations.png").exists())

            fig_bar = plot_interval_width_mean_bar(df_widths, output_dir=out_dir)
            self.assertIsNotNone(fig_bar)
            self.assertTrue((out_dir / "uncertainty_interval_width_mean_all_combinations.png").exists())

            # 3. Automated visualization decoding
            df_viz = pd.DataFrame([{
                "Dataset": "test_ds",
                "Task": "forecasting",
                "Seed": 42,
                "Combination": "np_stat + None",
                "Extractor": "np_stat",
                "Selector": "None",
                "Forecast RMSE": 0.25,
                "Forecast MAE": 0.18,
                "Interval Coverage": 0.90,
                "Mean Interval Width": 0.60,
                "History Values": json.dumps([history.tolist()]),
                "True Values": json.dumps([true_future.tolist()]),
                "Predictions": json.dumps([predictions["Model A"].tolist()]),
                "Prediction Lower": json.dumps([lower_bounds["Model A"].tolist()]),
                "Prediction Upper": json.dumps([upper_bounds["Model A"].tolist()]),
                "RMSE_H1": 0.2, "RMSE_H2": 0.3,
            }])
            gen_files = generate_forecast_visualizations(df_viz, output_dir=out_dir)
            self.assertGreater(len(gen_files), 0)
            for f in gen_files:
                self.assertTrue(os.path.exists(f))

            summary_csv_path = out_dir / "forecast_prediction_visualisation_summary.csv"
            self.assertTrue(summary_csv_path.exists())
            df_sum = pd.read_csv(summary_csv_path)
            self.assertIn("Combination", df_sum.columns)
            self.assertIn("Horizon", df_sum.columns)
            self.assertIn("True Mean", df_sum.columns)
            self.assertIn("Prediction Mean", df_sum.columns)
            self.assertEqual(len(df_sum), len(true_future))

            # 4. run_statistical_analysis with forecasting task
            csv_path = out_dir / "benchmark.csv"
            df_viz.to_csv(csv_path, index=False)
            analysis_outputs = run_statistical_analysis(
                csv_path=csv_path,
                output_dir=out_dir / "analysis",
                task_type="forecasting",
                enable_ttests=False,
                enable_plots=True,
            )
            self.assertIn("summary_forecasting", analysis_outputs)

    def test_boundary_and_edge_cases(self) -> None:
        """Verify scale boundaries, single-step forecasting (H=1), and error handling."""
        # Single-step forecast (H=1)
        df_raw = pl.DataFrame({
            "time": np.arange(80, dtype=np.int32),
            "sensor": np.random.randn(80).astype(np.float32),
            "target": np.random.randn(80).astype(np.float32),
        })
        df_ts, df_targets = segment_forecasting_series(
            df=df_raw,
            history_len=20,
            forecast_horizon=1,
            time_col="time",
            feature_cols=["sensor"],
            target_col="target",
        )
        X, y = to_forecasting_tensors(df_ts, df_targets)
        self.assertEqual(y.shape[1], 1)
        self.assertEqual(X.shape[1], 20)

        # Invalid history_len or forecast_horizon
        with self.assertRaises(ValueError):
            segment_forecasting_series(df_raw, history_len=0, forecast_horizon=5)

        with self.assertRaises(ValueError):
            segment_forecasting_series(df_raw, history_len=20, forecast_horizon=-1)

        with self.assertRaises(ValueError):
            segment_forecasting_series(df_raw, history_len=20, forecast_horizon=5, stride=0)

        with self.assertRaises(ValueError):
            segment_forecasting_series(df_raw, history_len=20, forecast_horizon=5, test_size=1.5)

        # Empty tensors
        empty_ts = pl.DataFrame({
            "sequence_id": pl.Series([], dtype=pl.Int32),
            "step": pl.Series([], dtype=pl.Int32),
            "sensor": pl.Series([], dtype=pl.Float32),
        })
        empty_targets = pl.DataFrame({
            "sequence_id": pl.Series([], dtype=pl.Int32),
            "target": pl.Series([], dtype=pl.Float32),
        })
        X_empty, y_empty = to_forecasting_tensors(empty_ts, empty_targets)
        self.assertEqual(X_empty.shape[0], 0)
        self.assertEqual(y_empty.shape[0], 0)

        # Metric direction verification
        self.assertEqual(metric_direction("Forecast RMSE"), "lower")
        self.assertEqual(metric_direction("Forecast MAE"), "lower")
        self.assertEqual(metric_direction("Mean Interval Width"), "lower")

        # Insufficient training partition raising ValueError (split < history_len + forecast_horizon)
        df_short = pl.DataFrame({
            "time": np.arange(50, dtype=np.int32),
            "sensor": np.random.randn(50).astype(np.float32),
            "target": np.random.randn(50).astype(np.float32),
        })
        with self.assertRaises(ValueError):
            segment_forecasting_series(
                df=df_short,
                history_len=35,
                forecast_horizon=10,
                time_col="time",
                feature_cols=["sensor"],
                target_col="target",
                test_size=0.2,  # split = 40 < 45
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                generate_simulated_forecasting_dataset(
                    output_dir=os.path.join(tmpdir, "invalid_len"),
                    series_len=50,
                    history_len=35,
                    forecast_horizon=10,
                    test_size=0.2,
                )

        # Multivariate default feature_cols=None preserves target column in historical features
        df_multi = pl.DataFrame({
            "time": np.arange(60, dtype=np.int32),
            "temp": np.random.randn(60).astype(np.float32),
            "humidity": np.random.randn(60).astype(np.float32),
            "target": np.random.randn(60).astype(np.float32),
        })
        df_ts_multi, df_tgt_multi = segment_forecasting_series(
            df=df_multi,
            history_len=20,
            forecast_horizon=5,
            time_col="time",
            feature_cols=None,
            target_col="target",
        )
        self.assertIn("target", df_ts_multi.columns)
        self.assertIn("temp", df_ts_multi.columns)
        self.assertIn("humidity", df_ts_multi.columns)
        X_multi, y_multi = to_forecasting_tensors(df_ts_multi, df_tgt_multi)
        self.assertEqual(X_multi.shape[2], 3)

        # String series_id and group preservation in df_targets
        df_str = pl.DataFrame({
            "subject": ["sub_A"] * 50 + ["sub_B"] * 50,
            "time": np.tile(np.arange(50, dtype=np.int32), 2),
            "reading": np.random.randn(100).astype(np.float32),
            "target": np.random.randn(100).astype(np.float32),
        })
        _, df_tgt_str = segment_forecasting_series(
            df=df_str,
            history_len=20,
            forecast_horizon=5,
            time_col="time",
            group_col="subject",
            feature_cols=["reading"],
            target_col="target",
        )
        self.assertIn("sub_A", df_tgt_str["series_id"].to_list())
        self.assertIn("sub_B", df_tgt_str["series_id"].to_list())
        self.assertIn("sub_A", df_tgt_str["group"].to_list())

    def test_single_step_forecasting_benchmark_rf_and_ridge(self) -> None:
        """Verify end-to-end BakeoffRunner on single-step H=1 with both RandomForest and Ridge."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_dir = Path(tmpdir) / "sim_fc_h1"
            out_dir = Path(tmpdir) / "out_h1"

            generate_simulated_forecasting_dataset(
                output_dir=str(ds_dir),
                n_series=2,
                series_len=80,
                history_len=25,
                forecast_horizon=1,
                test_size=0.25,
                seed=42,
            )

            cfg = PipelineConfig(
                dataset_paths=[str(ds_dir)],
                task_type="forecasting",
                history_len=25,
                forecast_horizon=1,
                extractors=["numpy_statistical"],
                selectors=[None],
                models=["random_forest", "ridge"],
                n_splits=2,
                output_dir=str(out_dir),
                enable_ttests=False,
                enable_plots=False,
                enable_logging=False,
                seeds=[42],
            )

            runner = BakeoffRunner(cfg)
            df_res = runner.run()

            self.assertEqual(len(df_res), 2)
            self.assertIn("RMSE_H1", df_res.columns)
            self.assertIn("MAE_H1", df_res.columns)
            self.assertNotIn("RMSE_H2", df_res.columns)
            self.assertFalse(df_res["Forecast RMSE"].isna().any())
            self.assertFalse(df_res["Interval Coverage"].isna().any())
            self.assertFalse(df_res["Mean Interval Width"].isna().any())

    def test_model_seed_propagation(self) -> None:
        """Verify that varying seeds in BakeoffRunner properly configures model estimators."""
        runner = BakeoffRunner(PipelineConfig(task_type="forecasting"))
        rf_42 = runner._get_model("random_forest", seed=42)
        rf_99 = runner._get_model("random_forest", seed=99)
        self.assertEqual(rf_42.random_state, 42)
        self.assertEqual(rf_99.random_state, 99)

        ridge_42 = runner._get_model("ridge", seed=42)
        ridge_99 = runner._get_model("ridge", seed=99)
        self.assertEqual(ridge_42.named_steps["ridge"].random_state, 42)
        self.assertEqual(ridge_99.named_steps["ridge"].random_state, 99)


if __name__ == "__main__":
    unittest.main()

