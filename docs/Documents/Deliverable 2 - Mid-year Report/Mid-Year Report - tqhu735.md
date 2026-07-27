# Project #62: Time-series Machine Learning Model Selection Framework
## Mid-Year Report
## Trisan Ethan Q Huynh, partnered with Scott Mouat; supervised by Andreas Kempa-Liehr

---

## ABSTRACT

Time-series machine learning (ML) workflows require complex, multi-stage pipelines encompassing windowing, feature extraction, feature selection, and classification. A primary friction point in optimising these workflows is the computational overhead and memory footprint of automated feature extraction libraries, such as TSFresh. This report evaluates technical progress toward constructing a unified, high-performance benchmarking framework (TEMPO) capable of systematically comparing feature extraction and selection packages within a single pipeline. 

An alternative feature extraction engine was implemented in Python using direct 2D NumPy array slicing to bypass Pandas DataFrame overheads. Telemetry benchmarks across the BEED and PRED-MAINTENANCE datasets demonstrated an 11x reduction in peak memory footprint (26.93 MB vs 301.29 MB for N=100 rows on BEED); however, the unvectorised CPython iteration loop incurred a severe execution speed penalty (0.1x to 0.5x relative speedup), indicating that compiled backends are strictly required. Furthermore, Fourier coefficient truncation experiments demonstrated that reducing extracted frequency terms from 100 down to 25 per variable eliminates spectral feature redundancy without degrading Random Forest classification accuracy. Finally, comparative evaluations identified TSFEL and native Polars functime extractors as computationally efficient baselines for benchmark datasets.

---

## 1. INTRODUCTION

Engineering and scientific applications increasingly rely on continuous time-series data captured from physical sensors and monitoring systems. Analysing time-series data using machine learning (ML) requires complex, multi-stage pipelines encompassing windowing, feature extraction, feature selection, and classification. Automated feature extraction libraries, such as TSFresh, TSFEL, and catch22, map raw temporal signals into high-dimensional feature spaces across statistical, temporal, and spectral domains. 

However, these open-source libraries are often developed, optimised, and evaluated in isolation. Currently, researchers lack a standardised pipeline to compare configurations – such as swapping feature extractors or selection algorithms – and evaluate their impact on predictive accuracy, execution latency, and memory footprint.

Project #62 (Time-series ML Model Selection Framework, designated TEMPO) addresses this opportunity by constructing an integrated, high-performance benchmarking framework to cross-validate time-series ML pipelines for a given dataset. This report summarises progress accomplished during Semester 1, specifically presenting: 
•	an evaluation of a Pandas-free, NumPy-bound feature extraction engine and Polars DataFrame wrappers to reduce memory usage; 
•	an investigation into limiting Fourier coefficients to quantify feature redundancy; 
•	a cross-package benchmark comparing feature extraction and selection libraries within a single pipeline; and
•	a standardised data format for package development.

---

## 2. FEATURE EXTRACTION MEMORY BENCHMARKS

### 2.1 Motivation
A key goal of TEMPO is to enable reproducible comparisons of time-series packages within a single pipeline. However, default open-source implementations introduce notable computational overhead. TSFresh, for instance, converts input data into long-format Pandas DataFrames using unpivoting (melt) operations and joins. While this makes it easy to use, reliance on Pandas structures generates millions of temporary DataFrame objects. This can lead to high memory spikes that can exceed hardware limits when processing large datasets.

To evaluate low-overhead feature extraction alternatives, we developed a testing harness to measure execution time and peak memory consumption when replacing Pandas DataFrames with direct NumPy array slicing.


### 2.2 Benchmarking Setup
The benchmarking script developed compares the standard pandas-based TSFresh pipeline against a custom 1D NumPy slicer. Both approaches were tested with TSFresh's EfficientFCParameters configuration, which generates ~780 candidate features per time-series variable.

The NumPy extraction engine parses EfficientFCParameters into a lookup registry of calculator functions and keyword arguments. Input data matrices ("X"∈"R" ^(N×P)) are read directly without pandas index structures, and feature calculators are executed sequentially across 1D temporal array slices.

Execution time was measured using the high-precision time.perf_counter(), and memory usage was profiled using Python’s tracemalloc library, which tracks peak heap allocation in MB. Garbage collection was run prior to each evaluation to ensure clean baselines.

### 2.3 Results
The benchmarking suite was evaluated across two datasets:
    - BEED: a multi-sensor dataset comprising 8,000 rows and 16 raw input variables (~12,500 extracted features under EfficientFCParameters).
    - pred-maintenance: the AI4I 2020 Predictive Maintenance dataset comprising 10,000 rows and 7 raw input variables (~5,450 extracted features).

Feature extraction was tested across scaling row counts ("N"=20,50,100). The resulting telemetry metrics are detailed in Table 1 below.

#### Table 1: Telemetry Metrics across Scaling Boundaries for BEED and PRED-MAINTENANCE Datasets

| Dataset | Scale ($N$ Rows) | Extraction Method | Execution Time ($s$) | Peak Memory ($MB$) | Speedup Ratio |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **BEED** | 20 | Pandas Baseline | 6.4825 | 59.72 | 1.0x (Ref) |
| | 20 | NumPy Loop | 24.1083 | 6.71 | 0.3x |
| | 50 | Pandas Baseline | 9.2947 | 151.66 | 1.0x (Ref) |
| | 50 | NumPy Loop | 61.4560 | 14.30 | 0.2x |
| | 100 | Pandas Baseline | 14.6419 | 301.29 | 1.0x (Ref) |
| | 100 | NumPy Loop | 125.0795 | 26.93 | 0.1x |
| **PRED-MAINTENANCE** | 20 | Pandas Baseline | 4.9662 | 28.53 | 1.0x (Ref) |
| | 20 | NumPy Loop | 10.8361 | 3.38 | 0.5x |
| | 50 | Pandas Baseline | 6.5653 | 72.17 | 1.0x (Ref) |
| | 50 | NumPy Loop | 27.6909 | 6.75 | 0.2x |
| | 100 | Pandas Baseline | 9.1673 | 143.33 | 1.0x (Ref) |
| | 100 | NumPy Loop | 53.8820 | 12.36 | 0.2x |

### 2.4 Performance Trade-offs
The results show a clear trade-off between memory usage and execution speed:

Firstly, the NumPy engine achieved an 11x reduction in peak memory consumption. At "N"=100 rows on BEED, peak memory dropped from 301.29 MB (pandas) to 26.93 MB (NumPy). This confirms that avoiding DataFrame unpivoting successfully removes memory allocation spikes.

However, the unvectorised NumPy loop ran significantly slower (0.1x to 0.5x speedup relative to pandas). At "N"=100 rows on BEED, execution time increased from 14.64 s to 125.08 s.

While pandas baseline functions use C-optimised vectorised operations for DataFrames, iterating through hundreds of calculator functions in a standard Python for-loop adds heavy interpreter overhead. As a result, while direct NumPy slicing successfully bounds memory, the loop iterations needed are too slow for production use.

We also evaluated Polars as a memory-efficient DataFrame alternative. While Polars speeds up file loading and data unpivoting, passing Polars DataFrames into TSFresh still requires converting back to pandas (`.to_pandas()`) because TSFresh internally hardcodes pandas index structures. Consequently, using Polars as a wrapper alone does not resolve TSFresh's peak memory bottleneck. As such, future work will explore compiled backends (such as Numba or C-extensions) or native Polars expression evaluation to retain low memory bounds without sacrificing speed.

---

## 3. FOURIER COEFFICIENT TRUNCATION

### 3.1 Fourier Feature Redundancy
Discrete Fourier Transform (DFT) features account for a large portion of TSFresh's feature space. By default, TSFresh extracts 400 Fourier coefficient features per variable, comprising 100 frequency coefficients calculated across real, imaginary, magnitude, and phase angle components.

In physical sensor signals, lower-frequency coefficients capture main trends, whereas higher-frequency terms often represent noise. Extracting 100 coefficients across all four representations increases feature matrix size without necessarily improving classification accuracy. To evaluate whether Fourier extraction can be streamlined, tests were run to measure accuracy and extraction time when limiting the number of retained Fourier coefficients across all four spectral components.

### 3.2 Truncation Methodology
Using the testing notebook TEMPO26tUa_tsfreshcoefficienttest.ipynb, Fourier coefficient limits were evaluated across "n"=5, 10, 25, 50, 75, 100 under Efficient and Comprehensive TSFresh parameter sets, alongside a Minimal parameter set baseline ("n"=0, excluding Fourier terms).

Testing was performed on synthetic harmonic signals and real-world human gait data (walking.csv and running.csv). Extracted features were filtered using TSFresh feature selection (select_features) and evaluated with a Random Forest classifier (200 trees) under a stratified 70/30 train/test split.

### 3.3 Results
Table 2 and Figures 1 and 2 detail the feature counts, extraction times (s), selection times (s), and classification accuracies (%) on the gait dataset under varying Fourier coefficient truncation limits.

#### Table 2: Extraction and Classification Metrics under Fourier Coefficient Truncation

| Preset | Fourier Coeffs ($n$) | Extracted Features | Selected Features | Extraction Time ($s$) | Selection Time ($s$) | Accuracy ($\%$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Minimal** | 0 | 40 | 31 | 3.21 | 2.92 | 78.72% |
| **Efficient** | 5 | 1510 | 657 | 18.10 | 3.94 | 78.72% |
| | 10 | 1590 | 673 | 19.19 | 3.99 | 79.79% |
| | 25 | 1798 | 723 | 22.10 | 4.63 | 77.66% |
| | 50 | 1798 | 723 | 22.72 | 4.23 | 77.66% |
| | 75 | 1798 | 723 | 18.92 | 4.13 | 77.66% |
| | 100 | 1798 | 723 | 19.23 | 4.07 | 77.66% |
| **Comprehensive** | 5 | 1534 | 671 | 20.84 | 4.01 | 79.79% |
| | 10 | 1614 | 687 | 22.07 | 4.10 | 77.66% |
| | 25 | 1822 | 737 | 21.09 | 4.22 | 80.85% |
| | 50 | 1822 | 737 | 21.93 | 4.33 | 80.85% |
| | 75 | 1822 | 737 | 21.39 | 4.22 | 80.85% |
| | 100 | 1822 | 737 | 22.00 | 4.19 | 80.85% |

```
[FIGURE PLACEHOLDER: Prediction Accuracy vs Fourier Coefficient Truncation Level]
Caption: Figure 1. Random Forest classification accuracy across Minimal, Efficient, and Comprehensive TSFresh configurations as a function of Fourier coefficient count (n).
```

```
[FIGURE PLACEHOLDER: Feature Extraction Time vs Fourier Coefficient Truncation Level]
Caption: Figure 2. Wall-clock feature extraction execution time (s) across extraction presets under varying Fourier coefficient cutoffs.
```

### 3.4 Discussion
The results highlight three main observations:
1.  Increasing the number of Fourier coefficients beyond "n"=25 provided no additional accuracy gains, as shown in Figure 1. Classification accuracy plateaued at 77.66% for Efficient ("n"≥25) and 80.85% for Comprehensive ("n"≥25).
2.  The extracted feature counts (1,798 for Efficient, 1,822 for Comprehensive) and selected counts (723 and 737) remained constant for "n"≥25. Higher-order terms beyond "n"=25 likely return empty or duplicate values that are automatically removed during initial data cleaning.
3.  The Minimal preset extracted 40 features in 3.21 s (Figure 2) and achieved 78.72% accuracy—matching the accuracy of Efficient while taking less than one-sixth of the extraction time.

From these results, we can establish that limiting Fourier coefficients to "n"=25 or "n"=50 effectively reduces feature space dimension without significantly impacting model performance.

---

## 4. ALTERNATIVE PACKAGE TESTS

### 4.1 Evaluated Package Combinations
To establish cross-package benchmarks, tests were run comparing different feature extraction libraries and feature selection methods.

The evaluated extractors were:
    - TSFEL: a dedicated Python library extracting temporal, spectral, and statistical features.
    - Statistics: a custom Python extractor calculating 5 basic summary metrics (mean, standard deviation, minimum, maximum, and energy).
    - Polars functime: a native Polars expression extractor (`group_by("id").agg(...)`) combining summary statistics with `functime` time-series metrics.
    - TSFresh Minimal: a TSFresh preset calculating summary statistics without Fourier terms.
    - TSFresh Efficient (50 FFT): a TSFresh preset with Fourier coefficients capped at "n"=50.

The evaluated selectors were:
    - SelectKBest: an ANOVA F-value filter selector retaining top 20 features.
    - Boruta: a wrapper selector using Random Forest importance (BorutaPy).
    - TSFresh: built-in statistical hypothesis testing (select_features).
    - None: retaining all extracted features without selection.

### 4.2 Results
Pairing the 4 extractors with the 4 selectors produced 16 distinct pipeline combinations. Figures 3 and 4 compare extraction times and predictive accuracies across feature extraction packages, while Figure 5 compares feature selection runtime overheads. Figure 6 presents total wall-clock execution time across all 16 pipeline configurations.

```
[FIGURE PLACEHOLDER: Extraction Time by Feature Extraction Library]
Caption: Figure 3. Comparative feature extraction execution time (s) across TSFEL, Custom Statistics, TSFresh Minimal, and TSFresh Efficient (50 FFT).
```

```
[FIGURE PLACEHOLDER: Classification Accuracy by Feature Extraction Library]
Caption: Figure 4. Random Forest classification accuracy across feature extraction libraries.
```

```
[FIGURE PLACEHOLDER: Selection Time by Feature Selection Algorithm]
Caption: Figure 5. Execution time (s) for Boruta, TSFresh hypothesis selector, SelectKBest, and unselected baselines.
```

```
[FIGURE PLACEHOLDER: Total Pipeline Runtime by Extractor-Selector Combination]
Caption: Figure 6. Combined total execution wall-clock time (s) across all 16 extractor-selector pairings.
```

### 4.3 Method Comparison
Overall, TSFEL without feature selection offered a good balance of speed and accuracy (Figures 3 and 4). It required about half the extraction time of TSFresh Minimal while matching TSFresh Efficient accuracy (~0.2% improvement).

In addition, native Polars feature extraction using expression aggregations (`group_by("id").agg(...)`) and `functime` time-series extensions achieved 72.1% classification accuracy on benchmark data. Because operations run natively in Rust/C++ backends without converting to pandas or Python loops, it provides a fast, zero-copy alternative for baseline feature generation.

The simple custom Statistics extractor was ~1,000 times faster than TSFresh Efficient (0.0097 s average extraction time, Figure 3), but classification accuracy was ~4% lower (Figure 4).

Applying feature selection added computational runtime (Boruta ~10 s, TSFresh ~4 s, SelectKBest ~0.4 s; Figure 5) without improving classification accuracy on the benchmark dataset. As shown in Figure 6, total pipeline runtime was additive across extraction, selection, and inference phases. Furthermore, while omitting feature selection retained a larger feature space, downstream model inference latency increased by less than 0.2 s, confirming that feature selection runtime overheads provide no net efficiency gain for datasets of this scale.

---

## 5. STANDARDISED DATASET ARCHITECTURE

### 5.1 Unified Data Format
To decouple feature extraction engines from dataset-specific formatting, a standardised dataset ingestion pipeline was implemented using Polars and Apache Parquet binary storage (convert_datasets.py on the standardise-data branch). 

Raw benchmark datasets (such as BEED, PRED-MAINTENANCE, and synthetic harmonic series) are transformed into a long-format binary schema comprising two compressed Parquet files per dataset:
    - time_series.parquet: contains standard integer identifiers ("id", "time") paired with clean, zero-imputed numerical signal variables.
    - targets.parquet: contains matching sample identifiers ("id") paired with classification target labels ("target").

This binary abstraction provides fast, low-overhead IO while enforcing explicit schema validation prior to feature extraction. Standardising datasets into uniform Parquet tables ensures that any time-series dataset can be passed directly into TSFresh, TSFEL, or custom TEMPO extractors without requiring manual, dataset-specific formatting wrappers.

---

## 6. CONCLUSIONS AND FUTURE WORK

### 6.1 Summary
-   Measured pandas memory overhead in TSFresh and showed an 11x peak memory reduction using direct NumPy array slicing.
-   Showed that capping Fourier coefficients at "n"=25 or "n"=50 removes redundant features without reducing accuracy.
-   Used benchmarking to identify TSFEL without feature selection as an efficient (but incomplete) alternative to TSFresh.
-   Tested native Polars expression aggregations (`functime`) as a low-memory, zero-copy baseline achieving 72.1% accuracy.

### 6.2 Next Steps
-   Transition the Python iteration loop to Numba JIT compilation/C extensions (or similar) to eliminate runtime delays while minimising memory usage.
-   Finalise dataset standardisation scripts and methods.
-   Verify findings against standardised benchmark data.
-   Expand Polars zero-copy DataFrame evaluation to combine fast, vectorised execution with lower memory usage across larger datasets.

---

## REFERENCES

1. O. O. Aremu, D. Hyland-Wood, and P. R. McAree, "A machine learning approach to circumventing the curse of dimensionality in discontinuous time series machine data," *Reliab. Eng. Syst. Saf.*, vol. 195, p. 106706, Mar. 2020.
2. O. Gorodetskaya, Y. Gobareva, and M. Koroteev, "A Machine Learning Pipeline for Forecasting Time Series in the Banking Sector," *Economies*, vol. 9, no. 4, p. 205, Dec. 2021.
3. M. Barandas *et al.*, "TSFEL: Time Series Feature Extraction Library," *SoftwareX*, vol. 11, p. 100456, Jan. 2020.
4. M. Christ, N. Braun, J. Neuffer, and A. W. Kempa-Liehr, "Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests (tsfresh – A Python package)," *Neurocomputing*, vol. 307, pp. 72–77, Sep. 2018.
5. T. Henderson and B. D. Fulcher, "An Empirical Evaluation of Time-Series Feature Sets," *arXiv preprint arXiv:2110.10914*, 2021.
6. C. H. Lubba, S. S. Sethi, P. Knaute, S. R. Schultz, B. D. Fulcher, and N. S. Jones, "catch22: CAnonical Time-series CHaracteristics," *Data Min. Knowl. Discov.*, vol. 33, no. 6, pp. 1821–1852, Nov. 2019.
7. I. Revin, V. A. Potemkin, N. R. Balabanov, and N. O. Nikitin, "Automated machine learning approach for time series classification pipelines using evolutionary optimization," *Knowl.-Based Syst.*, vol. 268, p. 110483, May 2023.

---

## APPENDIX A: CODE SCRIPTS & REPOSITORY MAPPINGS

- Memory Telemetry Harness: `src/tempo/benchmark_efficient.py`
- Fourier Coefficient Truncation Experiments: `src/classification-examples/TEMPO26tUa_tsfreshcoefficienttest.ipynb`
- Alternative Library Extractors & Selectors: `src/classification-examples/my_extractors.py`
- Dataset Standardisation Pipeline: `scratch/convert_datasets.py` (on `standardise-data` branch)
