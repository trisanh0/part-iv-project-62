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


    def test_feature_store_provenance_and_invalidation(self):
        """Verify cache key formatting, invalidation on dataset/extractor changes, and metadata files."""
        from tempo.storage.feature_store import compute_dataset_fingerprint, compute_extractor_version
        import polars as pl
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeatureStore(storage_dir=tmpdir, backend="parquet")
            df_feat = pd.DataFrame(np.random.randn(20, 10), columns=[f"f_{i}" for i in range(10)])

            # 1. Test cache key generation with optional provenance hashes
            base_key = store.get_cache_key("ds_test", "numba_efficient", {"p": 1})
            self.assertEqual(base_key, "ds_test__numba_efficient__262ae956981d")

            keyed_with_hashes = store.get_cache_key(
                "ds_test", "numba_efficient", {"p": 1}, dataset_hash="dshash123456", extractor_version="v1.1.0"
            )
            self.assertEqual(keyed_with_hashes, "ds_test__numba_efficient__262ae956981d__ds_dshash12__ext_v1.1.0")

            # 2. Test saving and metadata sidecar generation
            saved_file = store.save(
                df_feat,
                "ds_test",
                "numba_efficient",
                params={"p": 1},
                dataset_hash="dshash123456",
                extractor_version="v1.1.0",
            )
            self.assertIsNotNone(saved_file)
            self.assertTrue(saved_file.exists())

            meta_file = Path(tmpdir) / f"{keyed_with_hashes}.meta.json"
            self.assertTrue(meta_file.exists())
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(meta["dataset_name"], "ds_test")
            self.assertEqual(meta["extractor_name"], "numba_efficient")
            self.assertEqual(meta["dataset_hash"], "dshash123456")
            self.assertEqual(meta["extractor_version"], "v1.1.0")
            self.assertEqual(meta["shape"], [20, 10])

            # 3. Test cache hit with matching provenance
            self.assertTrue(
                store.exists(
                    "ds_test",
                    "numba_efficient",
                    {"p": 1},
                    dataset_hash="dshash123456",
                    extractor_version="v1.1.0",
                )
            )

            # 4. Test cache miss on modified dataset hash (invalidation)
            self.assertFalse(
                store.exists(
                    "ds_test",
                    "numba_efficient",
                    {"p": 1},
                    dataset_hash="different_hash",
                    extractor_version="v1.1.0",
                )
            )

            # 5. Test cache miss on modified extractor version (invalidation)
            self.assertFalse(
                store.exists(
                    "ds_test",
                    "numba_efficient",
                    {"p": 1},
                    dataset_hash="dshash123456",
                    extractor_version="v2.0.0",
                )
            )

    def test_compute_dataset_fingerprint_file_and_dataframe(self):
        """Verify compute_dataset_fingerprint reflects file modification and in-memory dataframe changes."""
        from tempo.storage.feature_store import compute_dataset_fingerprint
        import polars as pl
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test_data.parquet"
            df1 = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10, 20, 30]})
            df1.write_parquet(p)

            fp1 = compute_dataset_fingerprint(p)
            self.assertIsInstance(fp1, str)
            self.assertEqual(len(fp1), 12)

            # Modify file by writing additional rows and changing timestamp
            time.sleep(0.01)
            df2 = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10, 20, 30, 40]})
            df2.write_parquet(p)
            fp2 = compute_dataset_fingerprint(p)
            self.assertNotEqual(fp1, fp2)

            # In-memory DataFrame fingerprinting
            fp_df1 = compute_dataset_fingerprint(df1)
            fp_df2 = compute_dataset_fingerprint(df2)
            self.assertNotEqual(fp_df1, fp_df2)

            # Pandas DataFrame fingerprinting
            df_pd1 = df1.to_pandas()
            df_pd2 = df2.to_pandas()
            self.assertNotEqual(compute_dataset_fingerprint(df_pd1), compute_dataset_fingerprint(df_pd2))


if __name__ == "__main__":
    unittest.main()
