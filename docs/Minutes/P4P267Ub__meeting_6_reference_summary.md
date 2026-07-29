# TEMPO Progress Summary & Algorithm Benchmark Reference
**Date**: July 30, 2026  
**Document RAG ID**: `P4P267Ub__meeting_6_reference_summary`

---

## 1. Key Accomplishments Since Last Meeting

- **Custom Numba JIT Feature Extraction Engine**: Built a Numba-compiled extraction module replicating 1:1 `tsfresh` `EfficientFCParameters`. Achieved an ~1,800x execution speedup compared to standard Python `tsfresh`.
- **Mid-Year Report Submission Draft**: Authored `Mid-Year Report - tqhu735.md` with accompanying publication-ready SVG/PNG figures for memory trade-offs, architecture flows, and Fourier coefficient scaling.
- **Data Standardization & Pipeline Integration**: Merged `coefficient-benchmark-testing` and `standardise-data` into `main`. Unified $X_{\text{train}}, X_{\text{test}}, y_{\text{train}}, y_{\text{test}}$ partitions across random seeds.
- **NeSI Single-Pass Database Strategy**: Formulated single-pass feature extraction architecture to avoid re-extracting feature matrices across ~10,000 NeSI experiment combinations.

---

## 2. Cross-Package Extractor & Selector Benchmarks

Data evaluated across 10 repeated runs on benchmark dataset splits:

| Extractor Preset | Feature Selector | Prediction Accuracy (Mean ± Std) | Extraction Time (s) | Selection Time (s) | Total Pipeline Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TSFresh Efficient (50 FFT)** | TSFresh | **0.9793 ± 0.0104** | 22.41 | 4.76 | 27.49 |
| **TSFresh Efficient (50 FFT)** | Boruta | 0.9775 ± 0.0101 | 22.49 | 29.93 | 52.65 |
| **TSFresh Efficient (50 FFT)** | SelectKBest | 0.9695 ± 0.0116 | 22.58 | 1.44 | 24.16 |
| **TSFresh Efficient (50 FFT)** | None | 0.9760 ± 0.0088 | 22.64 | 0.00 | 23.13 |
| **TSFEL** | TSFresh | 0.9788 ± 0.0078 | 3.08 | 3.81 | 7.05 |
| **TSFEL** | Boruta | 0.9765 ± 0.0071 | 3.14 | 9.27 | 12.52 |
| **TSFEL** | SelectKBest | 0.9780 ± 0.0075 | 3.29 | 0.08 | 3.50 |
| **TSFEL** | None | 0.9780 ± 0.0073 | 3.06 | 0.00 | 3.19 |
| **TSFresh Minimal** | TSFresh | 0.9540 ± 0.0082 | 9.28 | 4.34 | 13.79 |
| **TSFresh Minimal** | None | 0.9553 ± 0.0095 | 8.62 | 0.00 | 8.80 |
| **Statistics Only** | Boruta | 0.9543 ± 0.0092 | 0.0090 | 0.56 | 0.68 |
| **Statistics Only** | None | 0.9543 ± 0.0092 | 0.0089 | 0.00 | 0.13 |

---

## 3. Fourier Coefficient Cutoff ($n$) Scaling

Evaluation of varying Fourier coefficient cutoffs ($n \in \{10, 25, 50, 75, 100\}$) using `TSFresh` selection:

| Extractor Configuration | $n$ Cutoff | Mean Accuracy | Extraction Time (s) | Total Pipeline Time (s) |
| :--- | :--- | :--- | :--- | :--- |
| **TSFresh Comprehensive** | 10 | 0.9773 | 27.04 | 32.12 |
| **TSFresh Comprehensive** | 25 | 0.9788 | 27.21 | 32.29 |
| **TSFresh Comprehensive** | 50 | 0.9793 | 27.35 | 32.47 |
| **TSFresh Comprehensive** | 75 | 0.9775 | 27.37 | 32.54 |
| **TSFresh Comprehensive** | 100 | **0.9797** | 27.44 | 32.63 |
| **TSFresh Efficient** | 10 | 0.9790 | 22.89 | 27.93 |
| **TSFresh Efficient** | 25 | 0.9770 | 23.27 | 28.36 |
| **TSFresh Efficient** | 50 | **0.9793** | 23.47 | 28.60 |
| **TSFresh Efficient** | 75 | 0.9778 | 23.62 | 28.77 |
| **TSFresh Efficient** | 100 | 0.9780 | 23.84 | 29.07 |

*Key finding*: Increasing $n$ above 50 Fourier coefficients yields diminishing returns in accuracy (< 0.1% gain) while steadily increasing computation and memory footprint.

---

## 4. Numba JIT Engine vs. Standard TSFresh Performance

Comparison of feature extraction execution time for standard `tsfresh` vs. `tempo` custom Numba JIT implementation (`EfficientFCParameters` equivalent):

| Metric | Standard TSFresh (Python/Pandas) | TEMPO Custom Numba Engine | Improvement |
| :--- | :--- | :--- | :--- |
| **Batch Execution Time** | ~23.47 s | ~0.013 s | **~1,800x faster** |
| **Memory Allocation Overhead** | High (Pandas object wrapping) | Minimal (Contiguous NumPy arrays) | Peak memory cut > 70% |
| **Parallel CPU Core Scaling** | Process pool overhead | Direct C-level loop parallelization | Linear scaling |

---

## 5. NeSI Experiment & Data Storage Strategy

1. **Single-Pass Feature Extraction**: Extract complete feature matrix (~700 features per window/time series) once and store in binary format (`.parquet` / `.npy`).
2. **Vertical & Horizontal Slicing**:
   - *Horizontal Slicing*: Separate train/test sample splits per fold/seed.
   - *Vertical Slicing*: Select specific feature parameter blocks (e.g., Fourier subset vs. full matrix) on demand.
3. **Bake-Off Experimental Design**:
   - 10 repeated cross-validation runs per dataset.
   - $\ge 100$ hyperparameter optimization iterations per fold.
