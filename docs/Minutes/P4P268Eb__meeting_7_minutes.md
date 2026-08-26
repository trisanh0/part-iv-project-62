# Meeting Minutes: Project #62 — Time-Series ML Model Selection Framework

**Project:** Part IV Project 62 — TEMPO Framework  
**Attendees:** Tri San Huynh, Scott Mounsey, Assoc. Prof. Dr.-Ing. Oliver Kempa-Liehr  
**Date:** August 14, 2026  
**Document ID:** `P4P268Eb`  

---

## 1. Fourier Feature Extraction & GPU Acceleration
- **FFT Coefficient Truncation:** `tsfresh` `EfficientFCParameters` extracts the first 50 Fourier coefficients by default. For a time series with 1,000 observations, `scipy.fft` computes 1,000 coefficients and discards the remaining 950.
- **Computation Overhead:** Decreasing the requested number of Fourier coefficients does not reduce computation time, as matrix multiplication in `scipy.fft` computes all coefficients regardless.
- **GPU Acceleration:** Because FFT is fundamentally matrix multiplication, performance can be scaled by offloading FFT computations to GPUs.
- **Execution Thresholds:** Benchmark variations in time-series length ($N$) and sequence counts to determine exact performance thresholds for splitting execution between CPU, Numba JIT, and GPU kernels.

---

## 2. Feature Storage Architecture
- **In-Memory Caching vs. Disk Persistence:** In-memory class-based feature caching (`ExtractorCache`) works well in RAM, but large-scale runs require persistent disk storage.
- **Storage Formats:** Investigate HDF5 (`h5py`) and Parquet formats for high-throughput disk storage of large feature matrices instead of relying on Pandas in-memory DataFrames.

---

## 3. Streaming Data, Windowing & Feature Selection Workflow
- **Recurrent Feature Extraction:** For streaming sensor data (e.g. 1 MHz signal segmented into 1,000-point windows), extracting feature vectors per window produces a secondary time series of features. This $800 \times 800$ feature space often carries distinct physical interpretations.
- **Stacked Feature Selection Workflow:**
  1. Subsample the full raw dataset while preserving target class distributions.
  2. Perform full feature extraction on the subsample.
  3. Apply statistical feature selection (`FeatureSelector`) to identify significant features.
  4. Apply the resulting feature mask to process the remainder of the dataset efficiently.

---

## 4. TSFresh Feature Selection & P-Value Inspection
- **FDR Thresholding:** `tsfresh` `FeatureSelector` defaults to a False Discovery Rate (FDR) level of $q = 0.05$ (or stricter levels down to $10^{-10}$).
- **Scikit-Learn API Pattern:**
  ```python
  from tsfresh.transformers import FeatureSelector
  import pandas as pd

  # Fit feature selector with specified FDR level
  select = FeatureSelector(fdr_level=0.05)
  select.fit(X, y)
  X_selected = select.transform(X)

  # Inspect feature p-values as a Pandas Series
  p_values = pd.Series(select.p_values, index=select.feature_names)
  X_relevant = X[select.relevant_features]
  ```

---

## 5. Project Timeline & Scope
- **Development Freeze:** Continue heavy feature and pipeline implementation for 2 more weeks, followed by a code freeze for new features.
- **Report Writing Target:** Complete the first draft of the Part IV Research Report at the start of the mid-semester break.
- **Benchmarking Scope:** Align with the Time-Series Classification Bake-Off methodology. Focus execution on a single local machine / VM before scaling.
- **Forecasting Metrics:** For forecasting tasks, use Root Mean Squared Error (RMSE) as the default objective, with Mean Absolute Percentage Error (MAPE) and uncertainty metrics as secondary evaluation criteria.
- **Packaging:** Explore PyScaffold for Python package deployment.

---

## 6. Action Items

| Task                                                                             | Owner(s)       | Timeframe      |
| :------------------------------------------------------------------------------- | :------------- | :------------- |
| Investigate GPU acceleration for FFT feature calculations                        | Scott          | Next Sprint    |
| Benchmark time-series length & sequence count thresholds for Numba/GPU split     | Scott & Trisan | Next Sprint    |
| Explore HDF5 (`h5py`) and Parquet storage formats for persistent feature caching | Scott & Trisan | Next Sprint    |
| Implement subsampled feature selection workflow for large datasets               | Trisan         | Next Sprint    |
| Draft Part IV Project Report sections ahead of mid-semester break                | Scott & Trisan | Start of Break |
