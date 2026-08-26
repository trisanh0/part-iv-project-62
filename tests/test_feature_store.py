"""Unit tests for TEMPO persistent feature storage and caching."""

import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

from tempo.storage.feature_store import FeatureStore


class TestFeatureStore(unittest.TestCase):
    def test_feature_store_parquet(self):
        """Verify Parquet storage read/write and slicing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeatureStore(storage_dir=tmpdir, backend="parquet")
            df = pd.DataFrame(np.random.randn(30, 20), columns=[f"f_{i}" for i in range(20)])
            
            self.assertFalse(store.exists("ds1", "ext1", {"p": 1}))
            saved_path = store.save(df, "ds1", "ext1", {"p": 1})
            self.assertIsNotNone(saved_path)
            self.assertTrue(saved_path.exists())
            self.assertTrue(store.exists("ds1", "ext1", {"p": 1}))

            # Test full load
            loaded = store.load("ds1", "ext1", {"p": 1})
            self.assertEqual(loaded.shape, (30, 20))

            # Test vertical columnar slice
            sliced = store.load("ds1", "ext1", {"p": 1}, feature_subset=["f_0", "f_3"])
            self.assertEqual(sliced.shape, (30, 2))
            self.assertEqual(list(sliced.columns), ["f_0", "f_3"])

    def test_feature_store_hdf5(self):
        """Verify HDF5 storage read/write and slicing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeatureStore(storage_dir=tmpdir, backend="hdf5")
            df = pd.DataFrame(np.random.randn(25, 15), columns=[f"f_{i}" for i in range(15)])

            store.save(df, "ds2", "numba", {"n_fft": 25})
            self.assertTrue(store.exists("ds2", "numba", {"n_fft": 25}))

            loaded = store.load("ds2", "numba", {"n_fft": 25}, feature_subset=["f_1", "f_5", "f_9"])
            self.assertEqual(loaded.shape, (25, 3))

    def test_feature_store_benchmark_io(self):
        """Verify I/O benchmark returns metrics for backends."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame(np.random.randn(40, 30), columns=[f"col_{i}" for i in range(30)])
            res = FeatureStore.benchmark_io(df, temp_dir=os.path.join(tmpdir, "io_bench"))
            self.assertIn("parquet", res)
            self.assertGreater(res["parquet"]["write_time_sec"], 0.0)
            self.assertGreater(res["parquet"]["read_time_sec"], 0.0)


if __name__ == "__main__":
    unittest.main()
