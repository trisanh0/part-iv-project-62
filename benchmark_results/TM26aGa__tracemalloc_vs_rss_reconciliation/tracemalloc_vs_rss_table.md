# Hardware Telemetry Audit: tracemalloc vs. Process-Tree RSS

**Git Commit**: `70a5d3cfe970a217bb53fc58d9a36f37264732a1`  
**Environment**: `macOS-27.0.1-arm64-arm-64bit` | Python `3.11.16` | Logical Cores: `8`  
**Dataset**: AI4I 2020 Predictive Maintenance ($N=199, W=100, p=6$ multivariate channels)  

### Side-by-Side Telemetry Comparison Table

| Extractor Backend | Features ($D$) | Runtime (s) | Baseline RSS (MB) | Peak RSS (MB) | Net RSS (MB) | `tracemalloc` Peak (MB) | Missed Memory (MB) | RSS / `tracemalloc` Ratio | Blind Spot Mechanism |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`polars_statistics`** | 30 | 0.02~s | 301.8 | **305.4** | 3.5 | 0.07 | 305.3 (100.0%) | **4177.6$\times$** | CPython allocator misses Rust Apache Arrow contiguous chunk memory |
| **`numpy_statistical`** | 30 | 0.13~s | 284.9 | **286.9** | 2.0 | 0.29 | 286.6 (99.9%) | **991.8$\times$** | CPython allocator misses C-contiguous array buffer re-allocations |
| **`numba_efficient`** | 882 | 8.31~s | 303.3 | **386.7** | 83.4 | 26.85 | 359.9 (93.1%) | **14.4$\times$** | CPython allocator misses LLVM JIT native heap & SIMD scratch arrays |
| **`tsfel`** | 186 | 16.29~s | 302.7 | **310.5** | 7.8 | 2.15 | 308.4 (99.3%) | **144.5$\times$** | Iterative spectral feature extraction partially tracked via pure Python |
| **`tsfresh_minimal`** | 60 | 3.38~s | 303.2 | **1249.1** | 945.9 | 6.59 | 1242.5 (99.5%) | **189.5$\times$** | Misses child worker processes spawned by MultiprocessingDistributor |
| **`tsfresh_efficient (Single-Process)`** | 4,662 | 268.57~s | 301.8 | **609.0** | 307.2 | 231.82 | 377.2 (61.9%) | **2.6$\times$** | Tracks single-process heap objects; misses C-extension buffer overhead |
| **`tsfresh_efficient (Multi-Process Pool)`** | 4,662 | 14.64~s | 303.4 | **1814.6** | 1511.2 | 236.51 | 1578.1 (87.0%) | **7.7$\times$** | Misses memory allocated across child worker processes in process tree |

---

### Reconciliation of Discrepant tsfresh Memory Figures

1. **1,829 MB (Multi-Process Process-Tree Peak RSS)**:
   - Observed when `tsfresh` runs in its default parallel configuration (`MultiprocessingDistributor` across all CPU worker processes).
   - High memory is driven by intermediate Pandas DataFrame unpivoting (`.melt()`), which converts $199 \times 100 \times 6 = 119,400$ values into long-format entity rows, followed by IPC pickling and serialization across child worker pools.
   - `tracemalloc` reports only the parent thread heap, completely missing all child process memory allocations.

2. **~50 MB / 49.3 MB (Single-Process Net RSS Increase)**:
   - Observed in single-process sequential sweeps ($N=400, W=100, p=3$) in `TM26a19__06_scaling_curves`.
   - Represents purely **Net Resident Memory** ($\Delta \text{RSS} = \text{Peak RSS} - \text{Baseline RSS}$), where baseline interpreter and loaded dependencies accounted for $313.9$~MB, and total peak was $363.2$~MB.
   - This measures the isolated incremental memory of single-threaded feature generation without inter-process duplication.

3. **Over 2 GB / 2,136 MB to 10,670 MB (Langevin Large-Trajectory Recovery)**:
   - Measured in `TM26a1X__langevin` during large trajectory evaluations.
   - At $L=10,000$ steps per trajectory ($N=200$), peak RSS reached $2,136.7$~MB ($2.1$~GB).
   - At $L=100,000$ steps per trajectory ($N=200$), peak RSS reached $10,669.9$~MB ($10.7$~GB), requiring $44.9$ minutes.
   - Reflects the unconstrained scaling of Pandas tabular indexing when applied to high-frequency dynamical simulation sequences.
