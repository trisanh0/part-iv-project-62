"""Unit tests for TEMPO unified benchmarking engine."""

import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl

from tempo.benchmark import BakeoffRunner, PipelineConfig, ExtractorConfig, SelectorConfig
from tempo.storage import generate_simulated_dataset


class TestBenchmark(unittest.TestCase):
    def test_benchmark_import_and_config(self):
        """Verify PipelineConfig and BakeoffRunner instantiate cleanly."""
        cfg = PipelineConfig(
            dataset_paths=[],
            task_type="classification",
            extractors=["numpy_statistical"],
            selectors=["select_k_best"],
            models=["random_forest"],
            n_splits=2,
        )
        runner = BakeoffRunner(cfg)
        self.assertIsNotNone(runner)

    def test_multichannel_tsfel_extraction(self):
        """Verify multi-channel TSFEL extracts features across all channels with channel prefixes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a 2-channel time-series dataset
            n_seq = 4
            seq_len = 20
            df_ts = pl.DataFrame({
                "sequence_id": np.repeat(np.arange(n_seq, dtype=np.int32), seq_len),
                "step": np.tile(np.arange(seq_len, dtype=np.int32), n_seq),
                "ch_a": np.random.randn(n_seq * seq_len).astype(np.float32),
                "ch_b": np.random.randn(n_seq * seq_len).astype(np.float32),
            })
            
            cfg = PipelineConfig(
                dataset_paths=[],
                cache_dir=tmpdir,
                cache_backend="memory",
                enable_logging=False,
            )
            runner = BakeoffRunner(cfg)
            features, stats, is_cached = runner._extract_features(df_ts, "tsfel", "test_ds")

            self.assertFalse(is_cached)
            self.assertEqual(features.shape[0], n_seq)
            # Channel prefixes should exist for channel 0 and channel 1
            has_ch0 = any(c.startswith("ch0_") for c in features.columns)
            has_ch1 = any(c.startswith("ch1_") for c in features.columns)
            self.assertTrue(has_ch0, "Expected features with ch0_ prefix")
            self.assertTrue(has_ch1, "Expected features with ch1_ prefix")

    def test_group_cv_in_benchmark_run(self):
        """Verify GroupKFold is used when subject/group metadata exists in targets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_dir = Path(tmpdir) / "group_ds"
            ds_dir.mkdir(parents=True)
            
            n_seq = 12
            seq_len = 15
            df_ts = pl.DataFrame({
                "sequence_id": np.repeat(np.arange(n_seq, dtype=np.int32), seq_len),
                "step": np.tile(np.arange(seq_len, dtype=np.int32), n_seq),
                "sensor": np.random.randn(n_seq * seq_len).astype(np.float32),
            })
            # 3 subjects, each having 4 sequences
            df_targets = pl.DataFrame({
                "sequence_id": np.arange(n_seq, dtype=np.int32),
                "target": np.array([0, 1, 0, 1] * 3, dtype=np.int32),
                "subject_id": np.repeat([101, 102, 103], 4).astype(np.int32),
            })
            df_ts.write_parquet(ds_dir / "time_series.parquet")
            df_targets.write_parquet(ds_dir / "targets.parquet")

            cfg = PipelineConfig(
                dataset_paths=[str(ds_dir)],
                task_type="classification",
                extractors=["numpy_statistical"],
                selectors=[None],
                models=["random_forest"],
                n_splits=3,
                cache_backend="memory",
                cache_dir=str(Path(tmpdir) / "cache"),
                output_dir=str(Path(tmpdir) / "out"),
                enable_ttests=False,
                enable_plots=False,
                enable_logging=False,
            )
            runner = BakeoffRunner(cfg)
            df_res = runner.run()

            self.assertFalse(df_res.empty)
            self.assertEqual(len(df_res), 1)
            # Verify telemetry disaggregation
            self.assertIn("fit_time_seconds", df_res.columns)
            self.assertIn("inference_latency_ms", df_res.columns)
            self.assertIn("is_cached", df_res.columns)
            self.assertFalse(df_res["is_cached"].iloc[0])
            self.assertFalse(np.isnan(df_res["Extraction Time (s)"].iloc[0]))

    def test_cached_telemetry_cleanliness(self):
        """Verify cached feature loads set is_cached=True and extraction metrics to NaN."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_dir = Path(tmpdir) / "cache_ds"
            ds_dir.mkdir(parents=True)
            
            n_seq = 6
            seq_len = 10
            df_ts = pl.DataFrame({
                "sequence_id": np.repeat(np.arange(n_seq, dtype=np.int32), seq_len),
                "step": np.tile(np.arange(seq_len, dtype=np.int32), n_seq),
                "sensor": np.random.randn(n_seq * seq_len).astype(np.float32),
            })
            df_targets = pl.DataFrame({
                "sequence_id": np.arange(n_seq, dtype=np.int32),
                "target": np.array([0, 1, 0, 1, 0, 1], dtype=np.int32),
            })
            df_ts.write_parquet(ds_dir / "time_series.parquet")
            df_targets.write_parquet(ds_dir / "targets.parquet")

            cfg = PipelineConfig(
                dataset_paths=[str(ds_dir)],
                task_type="classification",
                extractors=["numpy_statistical"],
                selectors=[None],
                models=["random_forest"],
                n_splits=2,
                cache_backend="memory",
                cache_dir=str(Path(tmpdir) / "cache"),
                output_dir=str(Path(tmpdir) / "out"),
                enable_ttests=False,
                enable_plots=False,
                enable_logging=False,
            )
            runner = BakeoffRunner(cfg)
            # First run: uncached
            df_res1 = runner.run()
            self.assertFalse(df_res1["is_cached"].iloc[0])
            self.assertFalse(np.isnan(df_res1["Extraction Time (s)"].iloc[0]))

            # Second run: cached (without resuming completed combo)
            df_res2 = runner.run(resume=False)
            self.assertTrue(df_res2["is_cached"].iloc[0])
            # Extraction Time and RAM must be NaN when cached to avoid polluting benchmarks
            self.assertTrue(np.isnan(df_res2["Extraction Time (s)"].iloc[0]))
            self.assertTrue(np.isnan(df_res2["Extraction Peak RAM (MB)"].iloc[0]))


    def test_null_selection_honesty_and_jaccard(self):

        """Verify that when zero features survive selection, benchmark reports 0.0 Jaccard stability, 100% reduction, and flags fallback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_dir = Path(tmpdir) / "null_sel_ds"
            ds_dir.mkdir(parents=True)
            
            n_seq = 8
            seq_len = 10
            df_ts = pl.DataFrame({
                "sequence_id": np.repeat(np.arange(n_seq, dtype=np.int32), seq_len),
                "step": np.tile(np.arange(seq_len, dtype=np.int32), n_seq),
                "sensor": np.random.randn(n_seq * seq_len).astype(np.float32),
            })
            df_targets = pl.DataFrame({
                "sequence_id": np.arange(n_seq, dtype=np.int32),
                "target": np.array([0, 1] * 4, dtype=np.int32),
            })
            df_ts.write_parquet(ds_dir / "time_series.parquet")
            df_targets.write_parquet(ds_dir / "targets.parquet")

            # A selector config with sample_ratio so small or fdr threshold that no features survive
            # We can use subsampled selector with base_selector that returns empty set
            selector_spec = {
                "name": "subsampled",
                "sample_ratio": 0.5,
                "base_selector": lambda X, y: pd.DataFrame(index=X.index),
            }

            cfg = PipelineConfig(
                dataset_paths=[str(ds_dir)],
                task_type="classification",
                extractors=["numpy_statistical"],
                selectors=[selector_spec],
                models=["random_forest"],
                n_splits=2,
                cache_backend="memory",
                cache_dir=str(Path(tmpdir) / "cache"),
                output_dir=str(Path(tmpdir) / "out"),
                enable_ttests=False,
                enable_plots=False,
                enable_logging=False,
            )
            runner = BakeoffRunner(cfg)
            df_res = runner.run()

            self.assertFalse(df_res.empty)
            self.assertTrue(df_res["fallback_triggered"].iloc[0])
            self.assertEqual(df_res["Selection Stability (Jaccard)"].iloc[0], 0.0)
            self.assertEqual(df_res["Feature Reduction (%)"].iloc[0], 100.0)
            self.assertEqual(df_res["N Selected Features"].iloc[0], 0.0)

    def test_group_cv_single_group_fallback(self):
        """Verify Group CV cleanly falls back to standard CV without crashing when n_groups < 2."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_dir = Path(tmpdir) / "single_group_ds"
            ds_dir.mkdir(parents=True)

            n_seq = 6
            seq_len = 10
            df_ts = pl.DataFrame({
                "sequence_id": np.repeat(np.arange(n_seq, dtype=np.int32), seq_len),
                "step": np.tile(np.arange(seq_len, dtype=np.int32), n_seq),
                "sensor": np.random.randn(n_seq * seq_len).astype(np.float32),
            })
            # All 6 sequences have the same group ID (only 1 unique group)
            df_targets = pl.DataFrame({
                "sequence_id": np.arange(n_seq, dtype=np.int32),
                "target": np.array([0, 1] * 3, dtype=np.int32),
                "group": np.ones(n_seq, dtype=np.int32),
            })
            df_ts.write_parquet(ds_dir / "time_series.parquet")
            df_targets.write_parquet(ds_dir / "targets.parquet")

            cfg = PipelineConfig(
                dataset_paths=[str(ds_dir)],
                task_type="classification",
                extractors=["numpy_statistical"],
                selectors=[None],
                models=["random_forest"],
                n_splits=2,
                cache_backend="memory",
                cache_dir=str(Path(tmpdir) / "cache"),
                output_dir=str(Path(tmpdir) / "out"),
                enable_ttests=False,
                enable_plots=False,
                enable_logging=False,
            )
            runner = BakeoffRunner(cfg)
            # Must run cleanly without ValueError: k-fold requires at least n_splits=2
            df_res = runner.run()
            self.assertFalse(df_res.empty)
            self.assertEqual(len(df_res), 1)


if __name__ == "__main__":
    unittest.main()

