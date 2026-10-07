"""
Persistent Feature Storage and Caching Engine for the TEMPO Framework.

Provides unified on-disk (Apache Parquet, HDF5) and in-memory caching for
extracted feature matrices to eliminate redundant feature extraction across
large-scale, multi-fold benchmarking runs.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BackendType = Literal["parquet", "hdf5", "memory", "none"]


def _compute_parameter_hash(params: Optional[Dict[str, Any]]) -> str:
    """Generate a deterministic MD5 hash for extraction parameters."""
    if not params:
        return "default"
    try:
        # Convert dict to deterministic JSON string
        encoded = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
        return hashlib.md5(encoded).hexdigest()[:12]
    except Exception:
        return str(hash(frozenset(str(params))))[:12]


def compute_dataset_fingerprint(
    dataset: Union[str, Path, pd.DataFrame, Any],
) -> str:
    """Compute a deterministic hash fingerprint representing dataset identity and content state.

    Supports file paths, directory paths containing Parquet files, and in-memory DataFrames.
    Uses file size and nanosecond modification timestamps for local filesystem artifacts,
    and schema, dimension, and sampled boundary checksums for in-memory data.
    """
    if isinstance(dataset, (str, Path)):
        p = Path(dataset)
        if p.exists():
            if p.is_dir():
                # Hash constituent time_series.parquet / targets.parquet or all parquet files
                candidates = sorted(p.glob("*.parquet"))
                if not candidates:
                    candidates = sorted(p.glob("*"))
                if candidates:
                    sig_parts = []
                    for c in candidates:
                        try:
                            st = c.stat()
                            sig_parts.append(f"{c.name}:{st.st_size}:{st.st_mtime_ns}")
                        except OSError:
                            pass
                    if sig_parts:
                        return hashlib.sha256(";".join(sig_parts).encode("utf-8")).hexdigest()[:12]
            else:
                try:
                    st = p.stat()
                    sig = f"{p.name}:{st.st_size}:{st.st_mtime_ns}"
                    return hashlib.sha256(sig.encode("utf-8")).hexdigest()[:12]
                except OSError:
                    pass
        # Fallback for named datasets or nonexistent paths: hash canonical string name
        return hashlib.sha256(str(dataset).strip().lower().encode("utf-8")).hexdigest()[:12]

    # In-memory DataFrame (Polars or Pandas)
    try:
        if hasattr(dataset, "schema") and hasattr(dataset, "shape"):
            # Polars DataFrame
            n_rows, n_cols = dataset.shape
            col_types = [f"{c}:{t}" for c, t in dataset.schema.items()]
            # Sample first, middle, last rows if available
            sample_vals = []
            if n_rows > 0:
                sample_idx = [0, n_rows // 2, n_rows - 1] if n_rows >= 3 else list(range(n_rows))
                try:
                    sample_df = dataset[sample_idx]
                    sample_vals = [str(row) for row in sample_df.iter_rows()]
                except Exception:
                    pass
            sig_str = f"pl:{n_rows}:{n_cols}:{','.join(col_types)}:{';'.join(sample_vals)}"
            return hashlib.sha256(sig_str.encode("utf-8")).hexdigest()[:12]

        elif hasattr(dataset, "dtypes") and hasattr(dataset, "shape"):
            # Pandas DataFrame
            n_rows, n_cols = dataset.shape
            col_types = [f"{c}:{t}" for c, t in dataset.dtypes.items()]
            sample_vals = []
            if n_rows > 0:
                sample_idx = [0, n_rows // 2, n_rows - 1] if n_rows >= 3 else list(range(n_rows))
                try:
                    sample_vals = [str(dataset.iloc[i].to_dict()) for i in sample_idx]
                except Exception:
                    pass
            sig_str = f"pd:{n_rows}:{n_cols}:{','.join(col_types)}:{';'.join(sample_vals)}"
            return hashlib.sha256(sig_str.encode("utf-8")).hexdigest()[:12]
    except Exception:
        pass

    return hashlib.sha256(str(type(dataset)).encode("utf-8")).hexdigest()[:12]


def compute_extractor_version(
    extractor: Union[str, Callable, Any],
) -> str:
    """Compute a deterministic version string or bytecode/name hash for an extractor."""
    if isinstance(extractor, str):
        # Known extractor versions in TEMPO framework
        known_versions: Dict[str, str] = {
            "numba_efficient": "v1.1.0",
            "numba": "v1.1.0",
            "polars_statistics": "v1.0.0",
            "polars": "v1.0.0",
            "numpy_statistical": "v1.0.0",
            "numpy": "v1.0.0",
            "tsfel": "v1.0.0",
            "tsfresh_minimal": "v1.0.0",
            "tsfresh_efficient": "v1.0.0",
            "tsfresh_comprehensive": "v1.0.0",
            "tsfresh": "v1.0.0",
        }
        clean = extractor.lower().strip()
        ver = known_versions.get(clean)
        if ver:
            return ver
        return hashlib.sha256(clean.encode("utf-8")).hexdigest()[:8]

    if callable(extractor):
        try:
            import inspect
            src = inspect.getsource(extractor)
            return hashlib.sha256(src.encode("utf-8")).hexdigest()[:8]
        except Exception:
            name = getattr(extractor, "__name__", str(extractor))
            return hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]

    name = getattr(extractor, "name", str(extractor))
    return hashlib.sha256(str(name).encode("utf-8")).hexdigest()[:8]


class FeatureStore:
    """Persistent storage manager for caching time-series feature matrices."""

    def __init__(
        self,
        storage_dir: Union[str, Path] = "data/03_processed/feature_store",
        backend: BackendType = "parquet",
    ):
        """Initialise the feature store.
        
        Args:
            storage_dir: Base directory path for storing feature cache files.
            backend: Storage format backend ('parquet', 'hdf5', 'memory', 'none').
        """
        self.storage_dir = Path(storage_dir)
        self.backend: BackendType = backend.lower()  # type: ignore
        self._memory_cache: Dict[str, pd.DataFrame] = {}

        if self.backend in ("parquet", "hdf5"):
            self.storage_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_key(
        self,
        dataset_name: str,
        extractor_name: str,
        params: Optional[Dict[str, Any]] = None,
        dataset_hash: Optional[str] = None,
        extractor_version: Optional[str] = None,
    ) -> str:
        """Construct a unique cache key from dataset, extractor, parameter config, and provenance hashes.

        Args:
            dataset_name: Dataset identifier.
            extractor_name: Extractor identifier.
            params: Optional parameter dictionary.
            dataset_hash: Optional deterministic dataset content/file fingerprint.
            extractor_version: Optional extractor version or code hash.

        Returns:
            Canonical cache key string.
        """
        param_hash = _compute_parameter_hash(params)
        clean_ds = dataset_name.lower().replace(" ", "_").replace("/", "_")
        clean_ext = extractor_name.lower().replace(" ", "_").replace("/", "_")
        parts = [clean_ds, clean_ext, param_hash]
        if dataset_hash is not None:
            parts.append(f"ds_{dataset_hash[:8]}")
        if extractor_version is not None:
            parts.append(f"ext_{extractor_version[:8]}")
        return "__".join(parts)

    def exists(
        self,
        dataset_name: str,
        extractor_name: str,
        params: Optional[Dict[str, Any]] = None,
        dataset_hash: Optional[str] = None,
        extractor_version: Optional[str] = None,
    ) -> bool:
        """Check if a cached feature matrix exists for the given configuration."""
        if self.backend == "none":
            return False

        key = self.get_cache_key(
            dataset_name,
            extractor_name,
            params,
            dataset_hash=dataset_hash,
            extractor_version=extractor_version,
        )

        if self.backend == "memory":
            return key in self._memory_cache

        if self.backend == "parquet":
            path = self.storage_dir / f"{key}.parquet"
            return path.exists()

        if self.backend == "hdf5":
            path = self.storage_dir / f"{key}.h5"
            return path.exists()

        return False

    def save(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        dataset_name: str,
        extractor_name: str,
        params: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
        sample_ids: Optional[np.ndarray] = None,
        dataset_hash: Optional[str] = None,
        extractor_version: Optional[str] = None,
    ) -> Optional[Path]:
        """Save an extracted feature matrix to the configured storage backend.
        
        Args:
            features: 2D feature matrix (DataFrame or NumPy array).
            dataset_name: Name of the input dataset.
            extractor_name: Name of the extractor engine.
            params: Dictionary of extractor parameters.
            feature_names: Optional list of column names if features is an ndarray.
            sample_ids: Optional index array for row identifiers.
            dataset_hash: Optional deterministic dataset content/file fingerprint.
            extractor_version: Optional extractor version or code hash.
            
        Returns:
            Path to saved file (for disk backends) or None (for memory/none backends).
        """
        if self.backend == "none":
            return None

        key = self.get_cache_key(
            dataset_name,
            extractor_name,
            params,
            dataset_hash=dataset_hash,
            extractor_version=extractor_version,
        )

        # Standardise to pandas DataFrame
        if isinstance(features, np.ndarray):
            if feature_names is None:
                feature_names = [f"feat_{i}" for i in range(features.shape[1])]
            df = pd.DataFrame(features, columns=feature_names)
            if sample_ids is not None:
                df.index = sample_ids
        else:
            df = features.copy()

        # Metadata dictionary for provenance auditability
        meta: Dict[str, Any] = {
            "key": key,
            "dataset_name": dataset_name,
            "extractor_name": extractor_name,
            "dataset_hash": dataset_hash,
            "extractor_version": extractor_version,
            "params": params or {},
            "shape": list(df.shape),
            "columns": [str(c) for c in df.columns],
            "saved_at": time.time(),
        }

        if self.backend == "memory":
            self._memory_cache[key] = df
            logger.debug("Cached %d features in memory for key: %s", df.shape[1], key)
            return None

        if self.backend == "parquet":
            file_path = self.storage_dir / f"{key}.parquet"
            meta_path = self.storage_dir / f"{key}.meta.json"
            # Ensure column names are strings for Parquet schema
            df.columns = [str(c) for c in df.columns]
            df.to_parquet(file_path, engine="pyarrow", index=True)
            try:
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
            except Exception as e:
                logger.warning("Could not write cache metadata file %s: %s", meta_path, e)
            logger.debug("Saved Parquet feature matrix: %s (shape: %s)", file_path, df.shape)
            return file_path

        if self.backend == "hdf5":
            try:
                import h5py
            except ImportError as err:
                raise ImportError(
                    "The 'h5py' package is required for HDF5 feature caching. "
                    "Install it via: pip install h5py"
                ) from err

            file_path = self.storage_dir / f"{key}.h5"
            meta_path = self.storage_dir / f"{key}.meta.json"
            with h5py.File(file_path, "w") as h5f:
                h5f.create_dataset(
                    "data",
                    data=df.to_numpy(dtype=np.float64),
                    compression="gzip",
                    compression_opts=4,
                )
                h5f.create_dataset(
                    "feature_names",
                    data=np.array([str(c).encode("utf-8") for c in df.columns]),
                )
                index_vals = df.index.to_numpy()
                if np.issubdtype(index_vals.dtype, np.number):
                    h5f.create_dataset("index", data=index_vals)
                else:
                    h5f.create_dataset(
                        "index",
                        data=np.array([str(idx).encode("utf-8") for idx in index_vals]),
                    )
            try:
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
            except Exception as e:
                logger.warning("Could not write cache metadata file %s: %s", meta_path, e)
            logger.debug("Saved HDF5 feature matrix: %s (shape: %s)", file_path, df.shape)
            return file_path

        return None

    def load(
        self,
        dataset_name: str,
        extractor_name: str,
        params: Optional[Dict[str, Any]] = None,
        row_indices: Optional[np.ndarray] = None,
        feature_subset: Optional[List[str]] = None,
        dataset_hash: Optional[str] = None,
        extractor_version: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load feature matrix with optional horizontal and vertical slicing.
        
        Args:
            dataset_name: Name of the input dataset.
            extractor_name: Name of the extractor engine.
            params: Dictionary of extractor parameters.
            row_indices: Optional integer or label positional index slice (horizontal).
            feature_subset: Optional list of specific feature column names (vertical).
            dataset_hash: Optional deterministic dataset content/file fingerprint.
            extractor_version: Optional extractor version or code hash.
            
        Returns:
            Sliced pandas DataFrame containing requested features and rows.
        """
        key = self.get_cache_key(
            dataset_name,
            extractor_name,
            params,
            dataset_hash=dataset_hash,
            extractor_version=extractor_version,
        )

        if not self.exists(
            dataset_name,
            extractor_name,
            params,
            dataset_hash=dataset_hash,
            extractor_version=extractor_version,
        ):
            raise FileNotFoundError(
                f"Feature cache not found for key: {key} (backend: {self.backend})"
            )

        if self.backend == "memory":
            df = self._memory_cache[key]

        elif self.backend == "parquet":
            file_path = self.storage_dir / f"{key}.parquet"
            if feature_subset is not None:
                # Optimized vertical columnar slice at read time
                try:
                    df = pd.read_parquet(
                        file_path,
                        columns=[str(c) for c in feature_subset],
                        engine="pyarrow",
                    )
                except Exception:
                    df = pd.read_parquet(file_path, engine="pyarrow")
            else:
                df = pd.read_parquet(file_path, engine="pyarrow")

        elif self.backend == "hdf5":
            import h5py
            file_path = self.storage_dir / f"{key}.h5"
            with h5py.File(file_path, "r") as h5f:
                raw_data = h5f["data"][:]
                feat_names = [f.decode("utf-8") for f in h5f["feature_names"][:]]
                if "index" in h5f:
                    raw_idx = h5f["index"][:]
                    if raw_idx.dtype.kind == "S":
                        idx = [x.decode("utf-8") for x in raw_idx]
                    else:
                        idx = raw_idx
                else:
                    idx = None

            df = pd.DataFrame(raw_data, columns=feat_names, index=idx)

        else:
            raise ValueError(f"Unsupported or disabled backend: {self.backend}")

        # Apply vertical slice if not already applied
        if feature_subset is not None and list(df.columns) != feature_subset:
            available_cols = [c for c in feature_subset if c in df.columns]
            df = df[available_cols]

        # Apply horizontal row slice
        if row_indices is not None:
            if hasattr(row_indices, "dtype") and np.issubdtype(row_indices.dtype, np.integer):
                df = df.iloc[row_indices]
            else:
                df = df.loc[df.index.intersection(row_indices)]

        return df

    def clear(self, dataset_name: Optional[str] = None) -> None:
        """Clear cached files from memory and disk."""
        if dataset_name is None:
            self._memory_cache.clear()
            if self.storage_dir.exists():
                for f in self.storage_dir.glob("*.*"):
                    if f.suffix in (".parquet", ".h5", ".hdf5"):
                        try:
                            f.unlink()
                        except Exception:
                            pass
        else:
            clean_ds = dataset_name.lower().replace(" ", "_").replace("/", "_")
            keys_to_del = [k for k in self._memory_cache if k.startswith(clean_ds)]
            for k in keys_to_del:
                del self._memory_cache[k]
            if self.storage_dir.exists():
                for f in self.storage_dir.glob(f"{clean_ds}__*"):
                    try:
                        f.unlink()
                    except Exception:
                        pass

    @staticmethod
    def benchmark_io(
        sample_df: pd.DataFrame,
        temp_dir: Union[str, Path] = "scratch/io_benchmark",
    ) -> Dict[str, Dict[str, float]]:
        """Benchmark write speed, read speed, and disk footprint for Parquet vs HDF5.
        
        Args:
            sample_df: Test DataFrame to evaluate.
            temp_dir: Directory for writing temporary benchmark files.
            
        Returns:
            Dictionary comparing metrics across backends.
        """
        temp_path = Path(temp_dir)
        temp_path.mkdir(parents=True, exist_ok=True)
        results: Dict[str, Dict[str, float]] = {}

        # 1. Parquet Benchmark
        pq_file = temp_path / "bench.parquet"
        t0 = time.perf_counter()
        sample_df.columns = [str(c) for c in sample_df.columns]
        sample_df.to_parquet(pq_file, engine="pyarrow", index=True)
        pq_write_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = pd.read_parquet(pq_file, engine="pyarrow")
        pq_read_time = time.perf_counter() - t0
        pq_size_mb = pq_file.stat().st_size / (1024.0 ** 2)

        results["parquet"] = {
            "write_time_sec": round(pq_write_time, 4),
            "read_time_sec": round(pq_read_time, 4),
            "file_size_mb": round(pq_size_mb, 4),
        }

        # 2. HDF5 Benchmark
        try:
            import h5py
            h5_file = temp_path / "bench.h5"
            t0 = time.perf_counter()
            with h5py.File(h5_file, "w") as h5f:
                h5f.create_dataset(
                    "data",
                    data=sample_df.to_numpy(dtype=np.float64),
                    compression="gzip",
                    compression_opts=4,
                )
                h5f.create_dataset(
                    "feature_names",
                    data=np.array([str(c).encode("utf-8") for c in sample_df.columns]),
                )
            h5_write_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            with h5py.File(h5_file, "r") as h5f:
                _ = h5f["data"][:]
            h5_read_time = time.perf_counter() - t0
            h5_size_mb = h5_file.stat().st_size / (1024.0 ** 2)

            results["hdf5"] = {
                "write_time_sec": round(h5_write_time, 4),
                "read_time_sec": round(h5_read_time, 4),
                "file_size_mb": round(h5_size_mb, 4),
            }
        except Exception as e:
            logger.warning("HDF5 I/O benchmark skipped: %s", e)

        # Cleanup
        try:
            if pq_file.exists():
                pq_file.unlink()
            if (temp_path / "bench.h5").exists():
                (temp_path / "bench.h5").unlink()
        except Exception:
            pass

        return results
