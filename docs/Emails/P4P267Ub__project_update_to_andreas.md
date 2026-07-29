Subject: Project Update - TEMPO Progress & Benchmarks

Hi Andreas,

Hope you're doing well.

Since our meeting today was cancelled, Scott and I wanted to send through a quick update on our progress over the past week and where we're heading next.

### Key Progress & Benchmarks

1. **Mid-Year Report & Presentation Visuals**
   - Drafted the Mid-Year Report (`Mid-Year Report - tqhu735.md`).
   - Generated publication-ready visual assets for data architecture flow, memory trade-off analysis, and Fourier coefficient scaling.

2. **High-Performance Numba JIT Feature Extractor**
   - Implemented a custom Numba JIT-compiled extraction engine in `tempo` replicating 1:1 `tsfresh` `EfficientFCParameters`.
   - Achieved an **~1,800x speedup** over standard `tsfresh` execution (~0.013s vs ~23.5s per batch), drastically cutting computational and memory overhead.

3. **Data Standardization & Single-Pass NeSI Architecture**
   - Merged `coefficient-benchmark-testing` and `standardise-data` into `main`.
   - Formulated a single-pass feature matrix caching architecture for NeSI: pre-computing feature vectors once into binary storage (`.parquet`/`.npy`) so ~10,000 experiment combinations take vertical/horizontal slices without re-extracting features.

4. **Fourier Coefficient Cutoff Scaling**
   - Evaluated scaling across $n \in \{10, 25, 50, 75, 100\}$ Fourier coefficients.
   - Identified that cutoffs beyond 50 coefficients yield diminishing returns (<0.1% accuracy gain) while increasing memory footprint.

### Next Steps

- Deploying initial tracer-bullet pipeline on NeSI with single-pass feature caching.
- Configuring the Time-Series Classification Bake-Off cross-validation framework (10 repeated CV runs, $\ge 100$ hyperparameter optimization iterations per fold).

Please let us know if you have any feedback on the report draft or if you would like to arrange a brief call sometime next week.

Best,

Trisan & Scott
