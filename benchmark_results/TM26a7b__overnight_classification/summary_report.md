# Benchmark Results: Overnight Critical Bifurcation Transitions & Chemical Sensor Drift (TM26a7b)

**Document Identifier**: `TM26a7b__overnight_classification`  
**Execution Timestamp**: 2026-10-07 00:53 to 2026-10-07 01:05 NZDT (~12 minutes runtime)  
**Target Architecture**: Apple M1 Pro (10 Cores, 16 GB Unified RAM), macOS  
**Total Executed Configurations**: 192 (2 Datasets $\times$ 4 Extractors $\times$ 6 Selectors $\times$ 2 Downstream Models $\times$ 2 Random Seeds)  
**Status**: Completed successfully with 100% convergence across all stratified splits, telemetry logs, and automated statistical post-processing.

---

## 1. Executive Summary

This benchmark evaluated the TEMPO pipeline on two challenging non-stationary classification problems:
1. `drift-bifurcation-cls`: Critical slowing down and phase transitions across the sub-critical ($\tau < \tau_c$) vs super-critical ($\tau \ge \tau_c$) regime in Langevin dynamical systems ($\tau \in [2.5, 4.5]$, $\tau_c = 3.333$).
2. `gas-sensor-drift-sub`: Real-world 6-class chemical sensor drift array across long-term environmental degradation ($N=600$ balanced multi-channel sequences).

---

## 2. Key Quantitative Findings

### 2.1 Feature Extractor Performance
| Feature Extractor | Mean Accuracy | Drift Bifurcation ($\tau < \tau_c$) | Chemical Sensor Drift |
| :--- | :---: | :---: | :---: |
| **`numba_efficient`** | **0.8475** | 0.7383 | **0.9566** |
| `polars_statistics` | 0.8115 | **0.7592** | 0.8638 |
| `numpy_statistical` | 0.8115 | **0.7592** | 0.8638 |
| `tsfresh_minimal` | 0.8134 | 0.7542 | 0.8727 |

*Critical Scientific Finding*:
- **Statistical Moments Win on Bifurcation Transitions**: Just as observed in regression parameter recovery (`TM26a6a`), probability density moments (`polars_statistics` and `numpy_statistical`) outperformed spectral features for detecting bifurcation state crossings (**$0.7592$** vs $0.7383$).
- **Spectral Features Dominate Multi-Channel Sensor Drift**: On the multi-channel gas sensor array, `numba_efficient` decisively outperformed moment-based extractors (**$0.9566$** vs $0.8638$, an absolute gain of $+9.3\%$).

### 2.2 Feature Selection Behaviour
| Feature Selector | Mean Accuracy | Feature Reduction (%) | Selection Stability (Jaccard) |
| :--- | :---: | :---: | :---: |
| **`l1` (Lasso / SAGA)** | **0.8323** | 46.25% | 0.7187 |
| `variance_threshold` | 0.8322 | 8.87% | **1.0000** |
| `tree_importance` (ExtraTrees) | 0.8248 | 50.00% | 0.7378 |
| `None` (Baseline) | 0.8242 | 0.00% | **1.0000** |
| `subsampled` (FDR) | 0.8072 | **73.12%** | 0.3350 |
| `select_k_best` | 0.7974 | 60.10% | 0.8200 |

*Findings*:
- $L_1$-penalized selection achieved the highest downstream accuracy ($0.8323$) while eliminating $46.3\%$ of noisy candidate features with strong fold stability ($J = 0.7187$).
- `tree_importance` successfully pruned $50.0\%$ of features with zero warnings or fallbacks following the dispatch update.

---

## 3. Analysis Artifacts Generated

Generated artifacts and statistical CSVs are archived in `benchmark_results/TM26a7b__overnight_classification/analysis/`:
- **Critical Difference Diagrams**: `cd_diagram_accuracy_classification_extractor.png` and `cd_diagram_accuracy_classification_selector.png`
- **Distribution Boxplots**: `accuracy_classification_by_extractor.png` and `accuracy_classification_by_selector.png`
- **Benjamini-Hochberg Pairwise $t$-Tests**: `extractor_accuracy_classification_ttests.csv`, `selector_accuracy_classification_ttests.csv`
