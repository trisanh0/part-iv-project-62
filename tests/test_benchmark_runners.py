"""Unit tests for TEMPO benchmark runner harnesses and telemetry drivers."""

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import polars as pl

from tempo.benchmark_efficient import (
    patched_roll_out_time_series,
    load_beed,
    load_pred_maintenance,
    load_dataset as load_efficient_dataset,
    benchmark_pandas_tsfresh_efficient,
    benchmark_numpy_efficient,
    run_in_process as run_in_process_efficient,
    run_benchmarks,
)
from tempo.benchmark_numba import (
    benchmark_numba_efficient,
    run_in_process as run_in_process_numba,
    run_numba_benchmarks,
)


class TestBenchmarkRunners(unittest.TestCase):
    def test_patched_roll_out_time_series(self):
        """Verify Pandas 3.0 rollout monkey patch under positive and negative directions."""
        df = pd.DataFrame({
            "id": [1, 1, 1, 2, 2, 2],
            "time": [0, 1, 2, 0, 1, 2],
            "val": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        })

        # 1. Forward rolling (rolling_direction > 0)
        res_pos = patched_roll_out_time_series(
            timeshift=2,
            grouped_data=df.groupby("id"),
            rolling_direction=1,
            max_timeshift=2,
            min_timeshift=0,
            column_sort="time",
            column_id="id",
        )
        self.assertIsInstance(res_pos, list)
        self.assertFalse(res_pos[0].empty)

        # 2. Backward rolling (rolling_direction < 0)
        res_neg = patched_roll_out_time_series(
            timeshift=1,
            grouped_data=df.groupby("id"),
            rolling_direction=-1,
            max_timeshift=2,
            min_timeshift=0,
            column_sort="time",
            column_id="id",
        )
        self.assertIsInstance(res_neg, list)

        # 3. No column_sort specified
        res_nosort = patched_roll_out_time_series(
            timeshift=2,
            grouped_data=df.groupby("id"),
            rolling_direction=1,
            max_timeshift=2,
            min_timeshift=0,
            column_sort=None,
            column_id="id",
        )
        self.assertIsInstance(res_nosort, list)

        # 4. Group where column_id was dropped by groupby apply (Pandas 3.0+ simulation)
        df_no_id = pd.DataFrame({
            "time": [0, 1, 2],
            "val": [10.0, 20.0, 30.0],
        })
        grouped_sim = df.groupby(["id", "time"])[["val"]]
        res_tuple_key = patched_roll_out_time_series(
            timeshift=2,
            grouped_data=grouped_sim,
            rolling_direction=1,
            max_timeshift=2,
            min_timeshift=0,
            column_sort=None,
            column_id="id",
        )
        self.assertIsInstance(res_tuple_key, list)

    def test_benchmark_efficient_dataset_loaders(self):
        """Verify CSV loaders in benchmark_efficient parse data without Pandas."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)

            # Test load_beed
            beed_dir = base_dir / "data" / "01_raw" / "beed"
            beed_dir.mkdir(parents=True)
            beed_csv = beed_dir / "BEED_Data.csv"
            with open(beed_csv, "w", encoding="utf-8") as f:
                f.write("y,X1,X2\n")
                f.write("1,0.5,1.5\n")
                f.write("0,0.8,1.2\n")

            data_beed, names_beed = load_beed(beed_csv)
            self.assertEqual(data_beed.shape, (2, 2))
            self.assertEqual(names_beed, ["X1", "X2"])

            # Test load_pred_maintenance
            pm_dir = base_dir / "data" / "01_raw" / "pred-maintenance"
            pm_dir.mkdir(parents=True)
            pm_csv = pm_dir / "ai4i2020.csv"
            with open(pm_csv, "w", encoding="utf-8") as f:
                f.write("UDI,Product ID,Type,Air temp,Process temp,Rotational speed,Torque,Tool wear,Machine failure,TWF,HDF,PWF,OSF,RNF\n")
                f.write("1,M100,L,298.1,308.6,1551,42.8,0,0,0,0,0,0,0\n")
                f.write("2,M101,M,298.2,308.7,1408,46.3,3,0,0,0,0,0,0\n")

            data_pm, names_pm = load_pred_maintenance(pm_csv)
            self.assertEqual(data_pm.shape, (2, 6))
            self.assertIn("Type", names_pm)
            self.assertEqual(data_pm[0, names_pm.index("Type")], 0.0)  # "L" -> 0.0
            self.assertEqual(data_pm[1, names_pm.index("Type")], 1.0)  # "M" -> 1.0

            # Test load_dataset dispatch
            d_beed, n_beed = load_efficient_dataset("beed", base_dir)
            self.assertEqual(d_beed.shape, (2, 2))

            d_pm, n_pm = load_efficient_dataset("pred-maintenance", base_dir)
            self.assertEqual(d_pm.shape, (2, 6))

            with self.assertRaises(ValueError):
                load_efficient_dataset("unknown_dataset", base_dir)

    def test_benchmark_efficient_extractors(self):
        """Verify benchmark extractors execute correctly on tiny array slices."""
        data = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ], dtype=np.float64)
        feature_names = ["feat_a", "feat_b"]

        res_pandas, cols_pandas = benchmark_pandas_tsfresh_efficient(
            data=data,
            feature_names=feature_names,
            max_timeshift=1,
        )
        self.assertIsInstance(res_pandas, np.ndarray)
        self.assertEqual(res_pandas.shape[0], 3)
        self.assertGreater(len(cols_pandas), 0)

        res_numpy, cols_numpy = benchmark_numpy_efficient(
            data=data,
            max_timeshift=1,
            raw_feature_names=feature_names,
        )
        self.assertIsInstance(res_numpy, np.ndarray)
        self.assertEqual(res_numpy.shape[0], 3)
        self.assertGreater(len(cols_numpy), 0)

    def test_benchmark_efficient_run_in_process(self):
        """Verify run_in_process telemetry collection and error handling in benchmark_efficient."""
        data = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ], dtype=np.float64)
        feature_names = ["feat_a", "feat_b"]

        # 1. Pandas
        res_pd, t_pd, mem_pd = run_in_process_efficient("pandas", data, feature_names, max_timeshift=1)
        self.assertEqual(res_pd.shape[0], 2)
        self.assertGreaterEqual(t_pd, 0.0)
        self.assertGreaterEqual(mem_pd, 0.0)

        # 2. NumPy
        res_np, t_np, mem_np = run_in_process_efficient("numpy", data, feature_names, max_timeshift=1)
        self.assertEqual(res_np.shape[0], 2)
        self.assertGreaterEqual(t_np, 0.0)
        self.assertGreaterEqual(mem_np, 0.0)

        # 3. Unsupported method raises ValueError
        with self.assertRaises(ValueError):
            run_in_process_efficient("unsupported_method", data, feature_names, max_timeshift=1)

    @patch("tempo.benchmark_efficient.run_in_process")
    @patch("tempo.benchmark_efficient.load_dataset")
    def test_benchmark_efficient_run_benchmarks_driver(self, mock_load, mock_run):
        """Verify run_benchmarks driver scaling loop executes quickly with mocked workloads."""
        mock_load.return_value = (np.zeros((25, 2)), ["a", "b"])
        mock_run.return_value = (np.zeros((10, 4)), 0.001, 0.5)

        # Driver should complete in <0.05s
        run_benchmarks()
        self.assertTrue(mock_load.called)
        self.assertTrue(mock_run.called)

    def test_benchmark_numba_efficient(self):
        """Verify benchmark_numba_efficient extracts features on small array."""
        data = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ], dtype=np.float64)
        feature_names = ["raw_0", "raw_1"]

        res, cols = benchmark_numba_efficient(data, feature_names)
        self.assertEqual(res.shape[0], 4)
        self.assertEqual(len(cols), res.shape[1])
        self.assertTrue(any(c.startswith("raw_0__") for c in cols))

    def test_benchmark_numba_run_in_process(self):
        """Verify run_in_process telemetry for all 3 methods in benchmark_numba."""
        data = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ], dtype=np.float64)
        feature_names = ["f0", "f1"]

        # 1. Pandas
        res_pd, t_pd, m_pd = run_in_process_numba("pandas", data, feature_names, max_timeshift=1)
        self.assertEqual(res_pd.shape[0], 2)

        # 2. NumPy Loop
        res_np, t_np, m_np = run_in_process_numba("numpy_loop", data, feature_names, max_timeshift=1)
        self.assertEqual(res_np.shape[0], 2)

        # 3. Numba JIT
        res_nb, t_nb, m_nb = run_in_process_numba("numba_jit", data, feature_names, max_timeshift=1)
        self.assertEqual(res_nb.shape[0], 2)

        # 4. Unknown method
        with self.assertRaises(ValueError):
            run_in_process_numba("unknown_engine", data, feature_names)

    @patch("tempo.benchmark_numba.run_in_process")
    @patch("tempo.benchmark_numba.load_dataset")
    def test_benchmark_numba_run_benchmarks_driver(self, mock_load, mock_run):
        """Verify run_numba_benchmarks driver scaling loop executes quickly with mocked workloads."""
        dummy_df = pl.DataFrame({
            "sequence_id": list(range(25)),
            "step": list(range(25)),
            "var_1": [1.0] * 25,
            "var_2": [2.0] * 25,
        })
        mock_load.return_value = (dummy_df, None)
        mock_run.return_value = (np.zeros((2, 10)), 0.001, 0.5)

        run_numba_benchmarks()
        self.assertTrue(mock_load.called)
        self.assertTrue(mock_run.called)


if __name__ == "__main__":
    unittest.main()
