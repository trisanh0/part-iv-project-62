# Benchmark Results: Overnight Classification Experiment (TM26a5a)

**Document Identifier**: `TM26a5a__overnight_classification`  
**Execution Timestamp**: 2026-10-05 23:47 to 2026-10-06 01:13 NZDT  
**Target Architecture**: Apple M1 Pro (10 Cores, 16 GB Unified RAM), macOS  
**Total Executed Configurations**: 288 (3 Datasets $\times$ 4 Extractors $\times$ 6 Selectors $\times$ 2 Downstream Models $\times$ 2 Random Seeds)  
**Status**: Completed successfully without fatal runtime errors or OOM interruptions.

---

## 1. Executive Summary

This overnight benchmarking run comprehensively assessed feature extraction efficiency, feature selection behaviour, and downstream classification fidelity across three diverse domain datasets:
1. `beed` (Honeybee bioacoustic sensor monitoring, binary classification)
2. `pred-maintenance-w100-cls` (NASA turbofan failure regime windows, multi-class degradation classification)
3. `har` (Human Activity Recognition 6-axis inertial motion telemetry, 6-class classification)

All four feature extractors (`numba_efficient`, `polars_statistics`, `numpy_statistical`, and `tsfresh_minimal`) and six selection strategies (`None` [Baseline], `variance_threshold`, `select_k_best`, `l1`, `tree_importance`, `subsampled` [Two-stage FDR]) completed evaluation across dual estimators (`random_forest`, `logistic_regression`).

### Key Quantitative Takeaways
- **Extraction Champion**: `numba_efficient` (with Welch FFT spectral binning $N_{\text{fft}}=25$) attained the highest overall classification accuracy across all datasets ($\mathbf{0.9593 \pm 0.0361}$, peak accuracy $\mathbf{0.9935}$ on `beed`), decisively surpassing purely time-domain moment statistical baselines (`numpy_statistical`: $0.9460$, `polars_statistics`: $0.9460$, `tsfresh_minimal`: $0.9456$).
- **Selection Fidelity & Sparsity**: 
  - `subsampled` (Two-stage Benjamini-Hochberg FDR filtering) emerged as the best feature selector overall: it pruned **47.63%** of extracted features while improving accuracy over unselected baselines to $\mathbf{0.9554}$ (versus baseline unselected $\mathbf{0.9533}$), achieving an average selection execution time of $3.41$ seconds.
  - `l1` (SAGA/Lasso regularised linear model selection) achieved **48.58%** feature reduction with high predictive accuracy ($0.9530$) and strong fold stability (Jaccard index $0.670$).
  - `select_k_best` (Univariate ANOVA F-value filtering) suffered substantial accuracy degradation ($0.9263$) despite aggressive 64.07% pruning, demonstrating that univariate ranking discards interdependent multi-channel acoustic and sensor features.
- **Model Comparison**: `random_forest` consistently outperformed linear `logistic_regression` across extracted time-series representations ($\mathbf{0.9620 \pm 0.0181}$ vs. $\mathbf{0.9364 \pm 0.0276}$).

---

## 2. Statistical Aggregations

### 2.1 Performance by Feature Extractor
| Feature Extractor | Mean Accuracy | Std Dev | Min Accuracy | Max Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **`numba_efficient`** | **0.9593** | 0.0361 | 0.8187 | **0.9935** |
| `numpy_statistical` | 0.9460 | 0.0228 | 0.8945 | 0.9873 |
| `polars_statistics` | 0.9460 | 0.0228 | 0.8943 | 0.9873 |
| `tsfresh_minimal` | 0.9456 | 0.0195 | 0.8854 | 0.9808 |

### 2.2 Performance by Feature Selector
| Feature Selector | Mean Accuracy | Feature Reduction (%) | Selection Stability (Jaccard) | Mean Selection Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| **`subsampled` (FDR)** | **0.9554** | **47.63%** | 0.6144 | 3.406 s |
| `variance_threshold` | 0.9540 | 5.52% | 0.9999 | 0.020 s |
| `None` (Baseline) | 0.9533 | 0.00% | 1.0000 | 0.000 s |
| `tree_importance` | 0.9533 | 0.00% | 1.0000 | 0.012 s |
| `l1` (SAGA/L1) | 0.9530 | **48.58%** | **0.6695** | 13.958 s |
| `select_k_best` | 0.9263 | 64.07% | 0.8138 | 0.020 s |

### 2.3 Performance by Evaluation Dataset
| Dataset | Modality / Task | Mean Accuracy | Min Accuracy | Max Accuracy |
| :--- | :--- | :---: | :---: | :---: |
| `beed` | Bioacoustic soundscapes (Binary) | 0.9614 | 0.8653 | 0.9935 |
| `pred-maintenance-w100-cls` | Turbofan sensor windows (Multiclass) | 0.9433 | 0.9197 | 0.9547 |
| `har` | 6-axis Smartphone IMU (6-class) | 0.9429 | 0.8187 | 0.9809 |

---

## 3. Analysis Artifacts & Critical Difference Plots

All generated analysis charts, boxplots, and paired two-sample Benjamini-Hochberg corrected $t$-test tables are archived in `benchmark_results/TM26a5a__overnight_classification/analysis/`:
- **Critical Difference (CD) Diagrams**:
  - Extractor Rank CD Diagram: `analysis/cd_diagram_extractor_accuracy_classification.png`
  - Selector Rank CD Diagram: `analysis/cd_diagram_selector_accuracy_classification.png`
- **Distribution Boxplots**:
  - Accuracy by Extractor: `analysis/accuracy_classification_by_extractor.png`
  - Accuracy by Selector: `analysis/accuracy_classification_by_selector.png`
- **Statistical Significance Tables**:
  - Extractor Pairwise $t$-tests: `analysis/extractor_accuracy_classification_ttests.csv`
  - Selector Pairwise $t$-tests: `analysis/combination_accuracy_classification_ttests.csv`
