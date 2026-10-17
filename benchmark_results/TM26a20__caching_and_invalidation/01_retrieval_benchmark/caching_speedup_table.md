# Feature Generation vs. Cached Parquet Retrieval Latency

| Extractor | $N$ (Windows) | Features ($D$) | Generation Time (s) | Cached Retrieval (Median s) | Retrieval Speedup | Cache Size (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `polars_statistics` | 50 | 15 | 0.0090 s | 0.99 ms | **9.1x** | 0.02 MB |
| `polars_statistics` | 100 | 15 | 0.0014 s | 1.17 ms | **1.2x** | 0.02 MB |
| `polars_statistics` | 200 | 15 | 0.0023 s | 1.04 ms | **2.2x** | 0.04 MB |
| `polars_statistics` | 400 | 15 | 0.0023 s | 1.23 ms | **1.9x** | 0.06 MB |
| `polars_statistics` | 800 | 15 | 0.0032 s | 1.22 ms | **2.6x** | 0.12 MB |
| `polars_statistics` | 1600 | 15 | 0.0053 s | 1.04 ms | **5.0x** | 0.23 MB |
| `numpy_statistical` | 50 | 15 | 0.0004 s | 1.20 ms | **0.3x** | 0.02 MB |
| `numpy_statistical` | 100 | 15 | 0.0005 s | 1.23 ms | **0.4x** | 0.02 MB |
| `numpy_statistical` | 200 | 15 | 0.0004 s | 1.14 ms | **0.4x** | 0.04 MB |
| `numpy_statistical` | 400 | 15 | 0.0006 s | 1.11 ms | **0.6x** | 0.06 MB |
| `numpy_statistical` | 800 | 15 | 0.0010 s | 0.97 ms | **1.0x** | 0.12 MB |
| `numpy_statistical` | 1600 | 15 | 0.0015 s | 1.03 ms | **1.5x** | 0.23 MB |
| `numba_efficient` | 50 | 441 | 0.4919 s | 11.86 ms | **41.5x** | 0.51 MB |
| `numba_efficient` | 100 | 441 | 0.0026 s | 12.17 ms | **0.2x** | 0.69 MB |
| `numba_efficient` | 200 | 441 | 0.0049 s | 12.43 ms | **0.4x** | 1.04 MB |
| `numba_efficient` | 400 | 441 | 0.0098 s | 12.60 ms | **0.8x** | 1.74 MB |
| `numba_efficient` | 800 | 441 | 0.0187 s | 11.60 ms | **1.6x** | 3.17 MB |
| `numba_efficient` | 1600 | 441 | 0.0357 s | 12.39 ms | **2.9x** | 6.06 MB |
| `tsfel` | 50 | 93 | 0.2588 s | 2.73 ms | **95.0x** | 0.08 MB |
| `tsfel` | 100 | 93 | 0.4734 s | 2.66 ms | **177.9x** | 0.10 MB |
| `tsfel` | 200 | 93 | 0.9770 s | 2.87 ms | **340.2x** | 0.15 MB |
| `tsfel` | 400 | 93 | 1.9346 s | 3.03 ms | **637.7x** | 0.25 MB |
| `tsfel` | 800 | 93 | 4.2255 s | 2.97 ms | **1423.0x** | 0.44 MB |
| `tsfel` | 1600 | 93 | 8.1559 s | 2.99 ms | **2730.4x** | 0.83 MB |
| `tsfresh_minimal` | 50 | 30 | 3.1804 s | 1.36 ms | **2337.7x** | 0.03 MB |
| `tsfresh_minimal` | 100 | 30 | 3.6678 s | 1.56 ms | **2347.6x** | 0.04 MB |
| `tsfresh_minimal` | 200 | 30 | 3.1167 s | 1.28 ms | **2430.3x** | 0.07 MB |
| `tsfresh_minimal` | 400 | 30 | 2.7731 s | 1.42 ms | **1948.1x** | 0.12 MB |
| `tsfresh_minimal` | 800 | 30 | 3.0742 s | 1.52 ms | **2026.1x** | 0.21 MB |
| `tsfresh_efficient` | 50 | 1431 | 4.9677 s | 35.30 ms | **140.7x** | 1.67 MB |
| `tsfresh_efficient` | 100 | 1431 | 6.7291 s | 37.34 ms | **180.2x** | 2.15 MB |
| `tsfresh_efficient` | 200 | 1431 | 10.0898 s | 34.55 ms | **292.0x** | 3.10 MB |
| `tsfresh_efficient` | 400 | 1431 | 13.1393 s | 36.40 ms | **360.9x** | 5.02 MB |