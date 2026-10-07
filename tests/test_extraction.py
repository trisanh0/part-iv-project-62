"""Unit tests for TEMPO feature extraction modules, engines, and numerical robustness."""

import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import polars as pl

from tempo.extraction import (
    numpy_statistical_extractor,
    polars_statistical_extractor,
    tsfresh_extractor,
    numba_feature_extractor,
    numba_efficient_extractor,
    tsfel_extractor,
    fft_parameters,
)
from tempo.extraction.numba_engine import (
    _numba_basic_stats,
    _numba_autocorr_kernel,
    _numba_quantiles_kernel,
    _numba_linear_trend_kernel,
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
        self.assertAlmostEqual(df_feat.iloc[0]["var_0__minimum"], 1.0)
        self.assertAlmostEqual(df_feat.iloc[0]["var_0__maximum"], 5.0)

    def test_numba_efficient_extractor_3d_tensor(self):
        """Verify Numba efficient extractor operates on 3D tensors (N, T, P) with custom feature names."""
        X_3d = np.random.randn(3, 10, 2).astype(np.float64)
        raw_names = ["sensor_alpha", "sensor_beta"]
        df_feat = numba_efficient_extractor(X_3d, raw_feature_names=raw_names, n_fft_coeffs=2)

        self.assertIsInstance(df_feat, pd.DataFrame)
        self.assertEqual(df_feat.shape[0], 3)
        self.assertTrue(any(c.startswith("sensor_alpha__") for c in df_feat.columns))
        self.assertTrue(any(c.startswith("sensor_beta__") for c in df_feat.columns))

    def test_numba_efficient_extractor_invalid_dims(self):
        """Verify Numba efficient extractor raises ValueError on 1D or 4D array inputs."""
        X_1d = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            numba_efficient_extractor(X_1d)

        X_4d = np.ones((2, 5, 2, 2))
        with self.assertRaises(ValueError):
            numba_efficient_extractor(X_4d)

    def test_numba_efficient_extractor_fft_options(self):
        """Verify FFT coefficient parameter toggles and zero-padding when n_coeffs > max_k."""
        X = np.random.randn(4, 5).astype(np.float64)

        # n_fft_coeffs=0 skips FFT features entirely
        df_no_fft = numba_efficient_extractor(X, n_fft_coeffs=0)
        self.assertFalse(any("fft_coefficient" in c for c in df_no_fft.columns))

        # n_fft_coeffs=20 on T=5 zero-pads
        df_padded = numba_efficient_extractor(X, n_fft_coeffs=20)
        self.assertTrue(any("coeff_19" in c for c in df_padded.columns))

    def test_numba_individual_kernels(self):
        """Verify underlying JIT kernels using py_func and compiled execution on boundary cases."""
        # 1. _numba_basic_stats
        empty_arr = np.array([], dtype=np.float64)
        for fn in [_numba_basic_stats, _numba_basic_stats.py_func]:
            res_empty = fn(empty_arr)
            self.assertEqual(len(res_empty), 24)
            self.assertEqual(np.sum(res_empty), 0.0)

            single_arr = np.array([5.0], dtype=np.float64)
            res_single = fn(single_arr)
            self.assertEqual(res_single[0], 5.0)   # mean
            self.assertEqual(res_single[1], 0.0)   # std

            const_arr = np.array([3.0, 3.0, 3.0, 3.0], dtype=np.float64)
            res_const = fn(const_arr)
            self.assertLessEqual(res_const[1], 1e-12)
            self.assertEqual(res_const[8], 0.0)   # skewness
            self.assertEqual(res_const[9], 0.0)   # kurtosis

            var_larger_arr = np.array([0.0, 10.0], dtype=np.float64)
            res_var_larger = fn(var_larger_arr)
            self.assertEqual(res_var_larger[21], 1.0)  # var > std

            seq_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
            res_seq = fn(seq_arr)
            self.assertAlmostEqual(res_seq[0], 3.5)  # mean
            self.assertGreater(res_seq[1], 0.0)      # std

        # 2. _numba_autocorr_kernel
        for fn in [_numba_autocorr_kernel, _numba_autocorr_kernel.py_func]:
            self.assertEqual(np.sum(fn(single_arr, max_lag=5)), 0.0)
            self.assertEqual(np.sum(fn(const_arr, max_lag=5)), 0.0)
            short_arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
            res_ac = fn(short_arr, max_lag=10)
            self.assertEqual(len(res_ac), 10)
            # When lag >= len(x), correlation should be 0.0
            self.assertEqual(res_ac[5], 0.0)

        # 3. _numba_quantiles_kernel
        q_list = np.array([0.1, 0.5, 0.9], dtype=np.float64)
        for fn in [_numba_quantiles_kernel, _numba_quantiles_kernel.py_func]:
            res_q_empty = fn(empty_arr, q_list)
            self.assertEqual(np.sum(res_q_empty), 0.0)

            data_q = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
            res_q = fn(data_q, np.array([0.0, 0.5, 1.0], dtype=np.float64))
            self.assertAlmostEqual(res_q[0], 1.0)
            self.assertAlmostEqual(res_q[1], 3.0)
            self.assertAlmostEqual(res_q[2], 5.0)

        # 4. _numba_linear_trend_kernel
        for fn in [_numba_linear_trend_kernel, _numba_linear_trend_kernel.py_func]:
            self.assertEqual(np.sum(fn(single_arr)), 0.0)
            res_const_trend = fn(const_arr)
            self.assertEqual(res_const_trend[0], 0.0)  # slope is 0
            self.assertEqual(res_const_trend[1], 3.0)  # intercept is constant value
            self.assertEqual(res_const_trend[2], 0.0)  # rvalue is 0
            trend_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
            res_trend = fn(trend_data)
            self.assertAlmostEqual(res_trend[0], 1.0)  # slope
            self.assertAlmostEqual(res_trend[1], 1.0)  # intercept
            self.assertAlmostEqual(res_trend[2], 1.0)  # rvalue



    def test_fft_parameters_generator(self):
        """Verify fft_parameters generates complete real, imag, abs, angle attribute sets."""
        params = fft_parameters(n_coeffs=3)
        self.assertEqual(len(params), 12)
        attrs = {p["attr"] for p in params}
        self.assertEqual(attrs, {"real", "imag", "abs", "angle"})
        coeffs = {p["coeff"] for p in params}
        self.assertEqual(coeffs, {0, 1, 2})

    def test_tsfresh_extractor_fallbacks(self):
        """Verify TSFresh ID and sort column fallbacks, parameter options, and exceptions."""
        df_seq = pd.DataFrame({"sequence_id": [1, 1], "step": [0, 1], "val": [1.0, 2.0]})
        res_seq = tsfresh_extractor(df_seq, parameter_set="minimal")
        self.assertEqual(len(res_seq), 1)

        df_id = pd.DataFrame({"id": [1, 1], "time": [0, 1], "val": [1.0, 2.0]})
        res_id = tsfresh_extractor(df_id, parameter_set="minimal")
        self.assertEqual(len(res_id), 1)

        df_noid = pd.DataFrame({"val": [1.0, 2.0]})
        with self.assertRaises(KeyError):
            tsfresh_extractor(df_noid)

        df_nosort = pd.DataFrame({"id": [1, 1], "other": [0, 1], "val": [1.0, 2.0]})
        res_nosort = tsfresh_extractor(df_nosort, parameter_set="minimal")
        self.assertEqual(len(res_nosort), 1)

        with self.assertRaises(ValueError):
            tsfresh_extractor(df_seq, parameter_set="unsupported_param_set")

        res_eff = tsfresh_extractor(df_seq, parameter_set="efficient", fft_coefficients=2)
        self.assertGreater(res_eff.shape[1], 0)

        with patch("tempo.extraction.tsfresh_engine.extract_features") as mock_extract:
            mock_extract.return_value = pd.DataFrame({"f1": [1.0]})
            _ = tsfresh_extractor(df_seq, parameter_set="comprehensive")
            self.assertTrue(mock_extract.called)

    def test_tsfel_extractor_domains(self):
        """Verify TSFEL feature extraction across statistical and temporal domains."""
        sig = [np.sin(np.linspace(0, 10, 50))]
        df_stat = tsfel_extractor(sig, domain="statistical")
        self.assertIsInstance(df_stat, pd.DataFrame)
        self.assertEqual(df_stat.shape[0], 1)
        self.assertEqual(df_stat.shape[1], 31)

        df_temp = tsfel_extractor(sig, domain="temporal")
        self.assertIsInstance(df_temp, pd.DataFrame)
        self.assertEqual(df_temp.shape[0], 1)
        self.assertEqual(df_temp.shape[1], 14)

    def test_tsfel_extractor_deduplication(self):
        """Verify TSFEL duplicate feature name handling logic."""
        with patch("tsfel.get_features_by_domain") as mock_domain, \
             patch("tsfel.time_series_features_extractor") as mock_extractor:
            mock_domain.return_value = {
                "statistical": {
                    "mean": {"param": 1},
                    "mean_dup": {"param": 2},
                }
            }
            mock_extractor.return_value = pd.DataFrame([{"mean": 1.0}])
            sig = [np.array([1.0, 2.0, 3.0])]
            res = tsfel_extractor(sig, domain="statistical")
            self.assertEqual(res.shape[0], 1)

    def test_tsfel_extractor_spectral_upstream_error(self):
        """Verify TSFEL spectral domain raises ValueError due to upstream duplicate wavelet column bug."""
        sig = [np.sin(np.linspace(0, 10, 50))]
        with self.assertRaises(ValueError):
            tsfel_extractor(sig, domain="spectral")

    def test_polars_extractor_fallbacks(self):
        """Verify Polars extractor id fallback, missing id error, and auto-detection of value columns."""
        df_seq = pl.DataFrame({
            "sequence_id": [0, 0, 1, 1],
            "step": [0, 1, 0, 1],
            "val_a": [1.0, 2.0, 3.0, 4.0],
            "val_b": [10.0, 20.0, 30.0, 40.0],
        })
        res_seq = polars_statistical_extractor(df_seq)
        self.assertEqual(res_seq.height, 2)
        self.assertIn("val_a__mean", res_seq.columns)
        self.assertIn("val_b__mean", res_seq.columns)

        df_id = pl.DataFrame({
            "id": [0, 0, 1, 1],
            "time": [0, 1, 0, 1],
            "val": [1.0, 2.0, 3.0, 4.0],
        })
        res_id = polars_statistical_extractor(df_id)
        self.assertEqual(res_id.height, 2)
        self.assertIn("val__mean", res_id.columns)

        df_noid = pl.DataFrame({"sensor": [1.0, 2.0]})
        with self.assertRaises(KeyError):
            polars_statistical_extractor(df_noid)

    def test_numpy_extractor_variants(self):
        """Verify NumPy extractor with 2D/3D shapes, custom feature names, and dimension validation."""
        X_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
        res_2d_names = numpy_statistical_extractor(X_2d, feature_names=["channel_0"])
        self.assertIn("channel_0__mean", res_2d_names.columns)

        X_3d = np.array([[[1.0, 10.0], [2.0, 20.0]], [[3.0, 30.0], [4.0, 40.0]]])
        res_3d_default = numpy_statistical_extractor(X_3d)
        self.assertIn("ch_0__mean", res_3d_default.columns)
        self.assertIn("ch_1__mean", res_3d_default.columns)

        with self.assertRaises(ValueError):
            numpy_statistical_extractor(np.array([1.0, 2.0]))
        with self.assertRaises(ValueError):
            numpy_statistical_extractor(np.ones((2, 3, 2, 2)))


class TestExtractionNumericalRobustness(unittest.TestCase):
    def test_extractors_nan_and_inf_propagation(self):
        """Verify extractors handle NaN and Inf values without unhandled interpreter crashes."""
        X_pathological = np.array([
            [1.0, np.nan, 3.0, np.inf, -np.inf],
            [np.nan, 2.0, -np.inf, np.inf, 5.0],
        ], dtype=np.float64)

        res_np = numpy_statistical_extractor(X_pathological)
        self.assertEqual(res_np.shape[0], 2)

        res_nb = numba_efficient_extractor(X_pathological, n_fft_coeffs=2)
        self.assertEqual(res_nb.shape[0], 2)

        df_pl = pl.DataFrame({
            "sequence_id": [0, 0, 1, 1],
            "step": [0, 1, 0, 1],
            "val": [1.0, np.nan, np.inf, -np.inf],
        })
        res_pl = polars_statistical_extractor(df_pl)
        self.assertEqual(res_pl.height, 2)

    def test_extractors_zero_variance_signals(self):
        """Verify zero-variance signals result in 0.0 variance/spread metrics without division by zero."""
        X_const = np.full((3, 20), 5.0, dtype=np.float64)

        res_np = numpy_statistical_extractor(X_const)
        self.assertEqual(res_np.shape[0], 3)
        self.assertAlmostEqual(res_np.iloc[0]["std"], 0.0)

        res_nb = numba_efficient_extractor(X_const, n_fft_coeffs=2)
        self.assertEqual(res_nb.shape[0], 3)
        self.assertAlmostEqual(res_nb.iloc[0]["var_0__standard_deviation"], 0.0)
        self.assertAlmostEqual(res_nb.iloc[0]["var_0__skewness"], 0.0)
        self.assertAlmostEqual(res_nb.iloc[0]["var_0__kurtosis"], 0.0)
        self.assertAlmostEqual(res_nb.iloc[0]["var_0__autocorrelation__lag_1"], 0.0)
        self.assertAlmostEqual(res_nb.iloc[0]["var_0__linear_trend__attr_\"slope\""], 0.0)

        df_pl_const = pl.DataFrame({
            "sequence_id": [0, 0, 1, 1],
            "step": [0, 1, 0, 1],
            "val": [5.0, 5.0, 5.0, 5.0],
        })
        res_pl_const = polars_statistical_extractor(df_pl_const)
        self.assertEqual(res_pl_const.height, 2)
        self.assertAlmostEqual(res_pl_const["val__std"][0], 0.0)

    def test_extractors_single_step_series(self):
        """Verify single-step sequences (T=1) execute without T-1 zero division errors."""
        X_single = np.array([[42.0], [7.0]], dtype=np.float64)

        res_np = numpy_statistical_extractor(X_single)
        self.assertEqual(res_np.shape[0], 2)
        self.assertAlmostEqual(res_np.iloc[0]["mean"], 42.0)

        res_nb = numba_efficient_extractor(X_single, n_fft_coeffs=1)
        self.assertEqual(res_nb.shape[0], 2)
        self.assertAlmostEqual(res_nb.iloc[0]["var_0__mean"], 42.0)

    def test_extractors_empty_series(self):
        """Verify empty series (T=0) behavior across extractor engines."""
        X_empty = np.empty((3, 0), dtype=np.float64)

        with self.assertRaises(ValueError):
            numpy_statistical_extractor(X_empty)

        with self.assertRaises(ValueError):
            numba_efficient_extractor(X_empty, n_fft_coeffs=2)

        res_nb_no_fft = numba_efficient_extractor(X_empty, n_fft_coeffs=0)
        self.assertEqual(res_nb_no_fft.shape[0], 3)

        df_pl_empty = pl.DataFrame({
            "sequence_id": pl.Series([], dtype=pl.Int32),
            "step": pl.Series([], dtype=pl.Int32),
            "val": pl.Series([], dtype=pl.Float64),
        })
        res_pl_empty = polars_statistical_extractor(df_pl_empty)
        self.assertEqual(res_pl_empty.height, 0)



if __name__ == "__main__":
    unittest.main()
