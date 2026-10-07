# TEMPO Codebase Reference & Agent Routing Architecture
**Document ID:** `TM269Ha__codebase_architecture_reference.md`  
**Target Audience:** Report Writing Agents & Automated LLM Systems  
**Purpose:** Pre-indexed semantic routing and factual reference. Eliminates full-tree repository searches.

---

## 1. Executive System Overview

`TEMPO` (*Time-series Evaluation for Model Performance Optimisation*) is a modular Python 3.11 benchmarking framework developed for University of Auckland Part IV Engineering Science Research Project 62.

It quantifies the Pareto frontier among:
1. **Feature extraction throughput** (speedup vs peak RAM allocation).
2. **Feature selection stability** (statistical hypothesis filters vs Boruta shadow-features vs subsampling).
3. **Downstream predictive accuracy** (classification, extrinsic regression, multi-step autoregressive forecasting).
4. **Hardware resource footprints** (process-tree RSS, CPU thread saturation, GPU VRAM).

### Core Licensing & Technical Constraints
- Exclusively MIT, BSD, or Apache 2.0 dependencies (Rule 01). No GPL/LGPL (e.g. strict avoidance of `hctsa` MATLAB runtime or GPL Python bindings).
- Scale-agnostic architecture: zero hardcoded sample lengths or channel counts.
- Contexere RAG indexing (`PIyymDc[_x]__keyword.ext`) for all research artefacts (scripts, configs, notebooks, minutes, data files).

---

## 2. Directory Tree & Fast File Map

```text
part-iv-project-62/
├── configs/                     # YAML & JSON benchmark matrix definitions
├── data/                        # Standardized binary Parquet storage
│   ├── 01_raw/                  # Unprocessed raw CSV/TXT datasets
│   ├── 02_interim/              # Intermediary cleaned sensor tables
│   └── 03_processed/            # Final partitioned time_series.parquet & targets.parquet
├── docs/
│   ├── CODEBASE_MAP.md          # Symbolic link to this reference document
│   ├── TM269Ha__codebase_architecture_reference.md # Primary AI reference document
│   ├── Documents/               # Academic briefs & Department handbooks
│   ├── Emails/                  # Formal supervisor communications
│   └── Minutes/                 # Contexere-indexed meeting minutes (P4P26*)
├── notebooks/                   # Legacy & exploratory research notebooks (TM26*, DS26*)
├── packages/
│   └── contexere/               # Standalone RAG CLI package (`nxt` executable)
├── presentation_figures/        # Exported high-res PNG/SVG pipeline & benchmark diagrams
├── report/                      # Master LaTeX thesis (ENGSCI 700 Report)
│   ├── figures/                 # Embedded PDF/PNG/TeX figures
│   ├── sections/                # Modular report chapters (00_abstract to 99_appendices)
│   ├── main.tex                 # LaTeX entrypoint
│   └── references.bib           # APA-7th BibTeX bibliography
├── scripts/
│   └── figures/                 # Dedicated Matplotlib/Seaborn reproduction scripts
├── src/tempo/                   # Core production package
│   ├── __init__.py              # Package entrypoint & Contexere export hooks
│   ├── analysis.py              # Paired t-tests, FDR, Cohen's d, CD diagrams
│   ├── benchmark.py             # 5-Stage unified harness (BakeoffRunner)
│   ├── export.py                # Deterministic Contexere filename & figure exporter
│   ├── telemetry.py             # Background thread psutil/pynvml resource sampler
│   ├── experiments/             # Standalone reproducible execution runs
│   ├── extraction/              # Modular feature extraction engines
│   ├── selection/               # Statistical hypothesis & wrapper feature filters
│   └── storage/                 # Parquet contract, window segmentation, tensor conversion
└── tests/                       # Pytest test suite ensuring 100% contract compliance
```

---

## 3. The 5-Stage TEMPO Pipeline Architecture

```
+---------------------------------------------------------------------------------------------------+
| Stage 1: Data Ingestion & Partitioning (tempo.storage)                                            |
| - Standardised Parquet dual-file schema (time_series.parquet + targets.parquet)                    |
| - Sliding window segmentation: segment_time_series(), segment_forecasting_series()                |
| - Leakage-free split: StratifiedKFold, StratifiedGroupKFold, GroupKFold, Temporal Train/Test       |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| Stage 2: Modular Feature Extraction & Caching (tempo.extraction, tempo.storage.feature_store)     |
| - Numba JIT (~780 features, njit fastmath parallel loops, zero DataFrame unpivoting)              |
| - Polars Expressions (lazy zero-copy aggregations: mean, std, min, max, energy)                   |
| - NumPy 1D/2D Slicing (vectorised array operations)                                               |
| - TSFEL Engine (statistical, spectral, temporal domains)                                          |
| - TSFresh Engine (Minimal, Efficient, Comprehensive, parameterised FFT truncation)                 |
| - FeatureStore: on-disk Parquet caching keyed by MD5(dataset + extractor + params)               |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| Stage 3: Feature Selection & Stability Analysis (tempo.selection)                                 |
| - Benjamini-Hochberg FDR hypothesis testing (tsfresh_selector)                                    |
| - Univariate Ranking: SelectKBest (ANOVA F-statistic, Mutual Information)                         |
| - Shadow-Feature Wrapper: Boruta (boruta_selector) with Random Forest importance                  |
| - SubsampledFeatureSelector: 2-stage fitting on subset, projected across full matrix               |
| - Stability Metric: Pairwise Jaccard similarity across CV folds                                   |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| Stage 4: Downstream Predictive Modelling & Latency Profiling (tempo.benchmark)                    |
| - Classification: RandomForestClassifier, LogisticRegression                                      |
| - Regression: RandomForestRegressor, Ridge                                                        |
| - Forecasting: Autoregressive tabular horizons, prediction intervals (alpha=0.10)                |
| - Latency: Split into extraction time, selection time, fit time (s), inference latency (ms/sample)|
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| Stage 5: Statistical Evaluation & Visualisation (tempo.analysis)                                  |
| - Paired Student's t-tests with Benjamini-Hochberg FDR adjustments across pipelines               |
| - Cohen's d effect sizes (negligible, small, medium, large)                                      |
| - (Note: CD diagrams and Bayesian statistics exist in results but code is not in this module)       |
| - Publication Plots: Boxplots (linear & log), Pareto frontiers, Speedup vs Accuracy ratios        |
+---------------------------------------------------------------------------------------------------+
                                                  ^
                                                  | (Continuous Sampling)
+---------------------------------------------------------------------------------------------------+
| Telemetry & Hardware Resource Profiling Daemon (tempo.telemetry)                                  |
| - Background sampling thread (20 Hz, 50 ms intervals)                                             |
| - Process-tree Resident Set Size (RSS in MB via psutil), avoiding CPython heap-only blindspots    |
| - CPU core utilisation (%) & GPU VRAM / SM utilisation (pynvml)                                   |
| - Environment snapshot: environment.json (CPU topology, RAM, Git commit, OS, package versions)   |
+---------------------------------------------------------------------------------------------------+
```

---

## 4. Module Specifications & Internal Call Graph

### A. Data Ingestion & Storage (`src/tempo/storage/`)
- [`dataset.py`](file:///Users/trisanh/Library/CloudStorage/GoogleDrive-trisanhuynh@gmail.com/Other%20computers/My%20laptop/2026/ENGSCI%20700/part-iv-project-62/src/tempo/storage/dataset.py)
  - `load_dataset(path)`: Loads `time_series.parquet` and `targets.parquet`. Returns `(df_ts, df_targets)`.
  - `to_numpy_tensor(df_ts)`: Converts long Polars/Pandas sequence table into 3D NumPy array `(n_instances, seq_len, n_channels)` without Python iteration.
  - `validate_export(path)`: Verifies schema types, non-null values, and strictly matching `sequence_id` indices.
  - Converters: `convert_predictive_maintenance()`, `convert_beed()`, `convert_uci_har()`, `convert_appliances_energy()`, `convert_beijing_pm25()`, `convert_gas_sensor_drift()`.
- [`segmentation.py`](file:///Users/trisanh/Library/CloudStorage/GoogleDrive-trisanhuynh@gmail.com/Other%20computers/My%20laptop/2026/ENGSCI%20700/part-iv-project-62/src/tempo/storage/segmentation.py)
  - `segment_time_series(df, window_size, stride)`: Sliding-window generator for classification and regression.
  - `segment_forecasting_series(df, history_len, forecast_horizon)`: Chronologically ordered sliding-window generator enforcing causal temporal separation.
- [`feature_store.py`](file:///Users/trisanh/Library/CloudStorage/GoogleDrive-trisanhuynh@gmail.com/Other%20computers/My%20laptop/2026/ENGSCI%20700/part-iv-project-62/src/tempo/storage/feature_store.py)
  - `FeatureStore(storage_dir, backend="parquet"|"memory"|"none")`: Persistent on-disk cache for extracted feature matrices.
  - `get_cache_key(dataset, extractor, params)`: Hashes config dictionary into deterministic MD5 keys.

### B. Feature Extraction Engines (`src/tempo/extraction/`)
- [`numba_engine.py`](file:///Users/trisanh/Library/CloudStorage/GoogleDrive-trisanhuynh@gmail.com/Other%20computers/My%20laptop/2026/ENGSCI%20700/part-iv-project-62/src/tempo/extraction/numba_engine.py)
  - `numba_efficient_extractor(X, feature_names)`: JIT-compiled C-speed extraction of ~780 TSFresh Efficient features.
  - Direct array kernels: `_numba_basic_stats()`, `_numba_autocorr_kernel()`, `_numba_quantiles_kernel()`, `_vectorized_fft_full_kernel()`. Completely eliminates Pandas unpivoting (`melt`) overhead.
- [`polars_engine.py`](file:///Users/trisanh/Library/CloudStorage/GoogleDrive-trisanhuynh@gmail.com/Other%20computers/My%20laptop/2026/ENGSCI%20700/part-iv-project-62/src/tempo/extraction/polars_engine.py)
  - `polars_statistical_extractor(df)`: Ultra-low latency summary features computed via native Polars expressions (zero-copy memory).
- [`numpy_engine.py`](file:///Users/trisanh/Library/CloudStorage/GoogleDrive-trisanhuynh@gmail.com/Other%20computers/My%20laptop/2026/ENGSCI%20700/part-iv-project-62/src/tempo/extraction/numpy_engine.py)
  - `numpy_statistical_extractor(X)`: Baseline vectorised numpy operations across 2D/3D tensors.
- [`tsfel_engine.py`](file:///Users/trisanh/Library/CloudStorage/GoogleDrive-trisanhuynh@gmail.com/Other%20computers/My%20laptop/2026/ENGSCI%20700/part-iv-project-62/src/tempo/extraction/tsfel_engine.py)
  - `tsfel_extractor(sequences, domain="statistical"|"spectral"|"temporal")`: TSFEL domain wrappers with automated deduplication of overlapping feature signatures.
- [`tsfresh_engine.py`](file:///Users/trisanh/Library/CloudStorage/GoogleDrive-trisanhuynh@gmail.com/Other%20computers/My%20laptop/2026/ENGSCI%20700/part-iv-project-62/src/tempo/extraction/tsfresh_engine.py)
  - `tsfresh_extractor(df, parameter_set, fft_coefficients)`: TSFresh wrapper.
  - `fft_parameters(n_coeffs)`: Parameter dictionary builder supporting truncated Fourier coefficient extraction ($n \in [5, 100]$).

### C. Feature Selection Algorithms (`src/tempo/selection/`)
- [`statistical.py`](file:///Users/trisanh/Library/CloudStorage/GoogleDrive-trisanhuynh@gmail.com/Other%20computers/My%20laptop/2026/ENGSCI%20700/part-iv-project-62/src/tempo/selection/statistical.py)
  - `tsfresh_selector(X, y, fdr_level)`: Benjamini-Hochberg FDR hypothesis testing over individual feature-target correlations.
  - `select_k_best(X, y, k, score_func)`: Top-$k$ ANOVA $F$-test or mutual information ranking.
- [`wrappers.py`](file:///Users/trisanh/Library/CloudStorage/GoogleDrive-trisanhuynh@gmail.com/Other%20computers/My%20laptop/2026/ENGSCI%20700/part-iv-project-62/src/tempo/selection/wrappers.py)
  - `boruta_selector(X, y, max_iter=20)`: Iterative all-relevant wrapper feature selector using Random Forest shadow attributes.
- [`subsampled.py`](file:///Users/trisanh/Library/CloudStorage/GoogleDrive-trisanhuynh@gmail.com/Other%20computers/My%20laptop/2026/ENGSCI%20700/part-iv-project-62/src/tempo/selection/subsampled.py)
  - `SubsampledFeatureSelector(base_selector, sample_ratio)`: High-throughput 2-stage selector. Fits expensive hypothesis tests or Boruta on small subsample (e.g. 10%) and projects mask to full dataset.

### D. Benchmarking Harness & Execution (`src/tempo/`)
- [`benchmark.py`](file:///Users/trisanh/Library/CloudStorage/GoogleDrive-trisanhuynh@gmail.com/Other%20computers/My%20laptop/2026/ENGSCI%20700/part-iv-project-62/src/tempo/benchmark.py)
  - `BakeoffRunner(config)`: Master orchestrator executing combinations of Extractors $\times$ Selectors $\times$ Models across $K$ folds.
  - `PipelineConfig`: Dataclass schema loading YAML/JSON experiment definitions.
  - Tracks extraction time, selection time, training time, inference latency per instance, memory RSS, and accuracy/RMSE.
- [`telemetry.py`](file:///Users/trisanh/Library/CloudStorage/GoogleDrive-trisanhuynh@gmail.com/Other%20computers/My%20laptop/2026/ENGSCI%20700/part-iv-project-62/src/tempo/telemetry.py)
  - `ResourceTracker()`: Context manager wrapping code execution blocks. Spawns `ResourceMonitor` thread polling at 50 ms.
  - Measures true OS Resident Set Size (`rss`), peak RAM increase ($\Delta \text{RSS}$), CPU core %, GPU VRAM (MB).
  - `log_system_info()`: Outputs hardware topology and Git state to `environment.json`.
- [`analysis.py`](file:///Users/trisanh/Library/CloudStorage/GoogleDrive-trisanhuynh@gmail.com/Other%20computers/My%20laptop/2026/ENGSCI%20700/part-iv-project-62/src/tempo/analysis.py)
  - `run_statistical_analysis()`: Comprehensive post-benchmark processor.
  - Produces paired Student's $t$-tests with Benjamini-Hochberg FDR adjustments (`pairwise_ttests`).
  - Calculates Cohen's $d$ effect sizes (`cohens_d`).
  - Generates speedup ratio plots and log-scale boxplots.
  - (Note: CD diagrams and Bayesian statistics are located in `benchmark_results/` but are currently generated externally).

---

## 5. Report Mapping & Evidence Locator

Use this cross-reference to write or verify sections in `report/sections/*.tex`:

| Report Section | Key Topic / Claims | Primary Code / File Reference | Empirical Data / Figure Source |
| :--- | :--- | :--- | :--- |
| **01 Introduction** (`01_introduction.tex`) | Motivation, high-dimensional telemetry, partner roles (Trisan vs Scott), 7 objectives. | `src/tempo/__init__.py`, `docs/Minutes/` | `docs/Documents/Project Brief.pdf`, `report/for-reference/` |
| **02 Lit Review** (`02_literature_review.tex`) | Comparison of `tsfresh`, `TSFEL`, `catch22`, `hctsa`; FDR Benjamini-Yekutieli; GPU DTW. | `report/references.bib` | `report/for-reference/00 Literature Review/` |
| **03 Methodology: Architecture** (`03_methodology.tex` L4-33) | 5-stage pipeline decomposition, `scikit-learn` estimator interface, deterministic hashing. | `src/tempo/benchmark.py`, `src/tempo/storage/feature_store.py` | `report/figures/fig_tempo_architecture.tex`, `presentation_figures/tempo_pipeline_flowchart.png` |
| **03 Methodology: Storage** (`03_methodology.tex` L34-44) | Parquet dual-file schema (`time_series.parquet`, `targets.parquet`), zero-copy Polars arrays. | `src/tempo/storage/dataset.py`, `src/tempo/storage/segmentation.py` | `presentation_figures/standardised_data_architecture_flow.png`, `tests/test_storage.py` |
| **03 Methodology: Telemetry** (`03_methodology.tex` L45-54) | Process-tree RSS vs `tracemalloc`, background thread (20 Hz), GPU VRAM (`pynvml`), environment log. | `src/tempo/telemetry.py` | `benchmark_results/environment.json`, `tests/test_telemetry.py` |
| **04 Results: Memory** (`04_results.tex` L4-51) | Pandas vs 1D NumPy slicing on BEED and AI4I ($11\times$ RAM drop, unvectorized Python loop latency). | `src/tempo/extraction/numpy_engine.py`, `notebooks/TM26sRa__beed_tsfresh.ipynb` | `report/tables/` (Table 1), `presentation_figures/memory_optimisation_tradeoff.png` |
| **04 Results: Fourier** (`04_results.tex` L54-106) | Truncating FFT coefficients ($n=5$ to $100$) on Gait sensor data. $n \ge 25$ yields no accuracy gains. | `src/tempo/extraction/tsfresh_engine.py` (`fft_parameters`), `notebooks/TM26tUa*` | `report/figures/fig1_2_fourier_accuracy_and_time.png` (Table 2) |
| **04 Results: 16-Pair Matrix** (`04_results.tex` L109-156) | 4 extractors $\times$ 4 selectors. TSFEL vs tsfresh runtime; Boruta selection latency bottleneck. | `src/tempo/experiments/extractor_selector_experiment.py` | `report/figures/fig3_extractor_latency.png` to `fig6_total_pipeline_latency.png` |
| **04 Results: Unified Bake-off** (`04_results.tex` update) | Numba JIT acceleration, multi-dataset UCR results, CD diagrams, paired t-tests, Bayesian sign test. | `src/tempo/benchmark.py`, `src/tempo/analysis.py`, `src/tempo/extraction/numba_engine.py` | `benchmark_results/TM268Ua__andreas_meeting_bakeoff/analysis/`, `benchmark_results/TM269Ga*/` |
| **Pedagogical Running Problem** (Meeting 8 / Sec 3 & 4) | AI4I 2020 predictive maintenance running example (W=100, S=50, binary failure vs continuous tool wear). | `src/tempo/experiments/TM269Ga__ai4i_running_benchmark.py` | `configs/TM269Ga__ai4i_running_classification.yaml`, `benchmark_results/TM269Ga*/` |

---

## 6. Configured Benchmark Matrices & Artifact Locations

### Key Experiment Configurations (`configs/`)
1. `TM268Ua__andreas_meeting_bakeoff.yaml`: Comprehensive multi-extractor and selector bake-off on benchmark datasets.
2. `TM269Ga__ai4i_running_classification.yaml` & `regression.yaml`: Pedagogical running toy problem scripts.
3. `TM268Xa__extractor_selector_classification.yaml`: Matrix evaluating 501 pipeline runs across diverse configurations.

### Empirical Data & Statistical Outputs (`benchmark_results/`)
- **Main Bakeoff Analysis:** `benchmark_results/TM268Ua__andreas_meeting_bakeoff/analysis/`
  - Critical difference diagrams: `cd_diagram_extractor_accuracy.png`, `cd_diagram_selector_accuracy.png`
  - Paired t-test tables: `extractor_accuracy_ttests.csv`, `extractor_extraction_time_s_ttests.csv`, `combination_accuracy_ttests.csv`
  - Bayesian statistics: `extractor_accuracy_bayesian.csv`, `selector_accuracy_bayesian.csv`
  - Speedup and resource distributions: `performance_ratio_extractor_speedup.png`, `extraction_peak_ram_mb_by_extractor_log.png`
- **AI4I Running Problem:** `benchmark_results/TM269Ga__ai4i_running_benchmark/`
  - Telemetry summaries and physical sensor importance rankings: `ai4i_running_benchmark_summary.csv`, `audit/`

---

## 7. How to Direct Report-Writing Agents

When prompting a report-writing agent for a specific section:
1. **Provide the Exact File Paths**: Route the agent directly to the relevant files mapped in Section 5 above.
2. **Cite Numerical Telemetry Directly**: Direct the agent to the corresponding CSV in `benchmark_results/*/analysis/` rather than letting it search or estimate metrics.
3. **Enforce Rule 02 (Report Writing Style)**: Strictly NZ/British English (`optimise`, `modelling`), third-person passive for methodology/physics, first-person active for personal team contributions, setup-action-result-impact paragraph flow.
4. **Alphanumeric Box Referencing**: Refer directly to labeled boxes in `fig_tempo_architecture.tex` (Box A through Box I, T1 through T3).
