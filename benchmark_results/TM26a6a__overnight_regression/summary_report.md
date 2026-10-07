# Benchmark Results: Overnight Regression & Parameter Estimation Experiment (TM26a6a)

**Document Identifier**: `TM26a6a__overnight_regression`  
**Execution Timestamp**: 2026-10-06 01:23 to 2026-10-06 02:00 NZDT (~37 minutes runtime)  
**Target Architecture**: Apple M1 Pro (10 Cores, 16 GB Unified RAM), macOS  
**Total Executed Configurations**: 384 (4 Datasets $\times$ 4 Extractors $\times$ 6 Selectors $\times$ 2 Downstream Regressors $\times$ 2 Random Seeds)  
**Status**: Completed successfully with 100% convergence across all splits, telemetry logs, and automated statistical post-processing.

---

## 1. Executive Summary

Following the classification bakeoff (`TM26a5a`), this overnight regression bakeoff addressed the primary project objective set during Meeting 9: evaluating feature extraction and selection on **continuous physical degradation tracking** and **stochastic dynamical system parameter estimation** ($\tau$).

### Evaluated Benchmark Datasets
1. `drift-bifurcation-reg`: Langevin drift-bifurcation stochastic differential equation simulation for estimating the critical bifurcation parameter $\tau \in [3.5, 4.5]$.
2. `pred-maintenance-w100-reg`: NASA turbofan CMAPSS-derived continuous Remaining Useful Life (RUL) regression across sensor temporal windows.
3. `beijing-pm25`: Environmental continuous meteorological pollution regression (hourly $\text{PM}_{2.5}$ concentration).
4. `appliances-energy`: Low-energy building environmental monitoring energy demand regression.

---

## 2. Key Quantitative Findings

### 2.1 Feature Extractor Performance Across All Tasks
| Feature Extractor | Mean RMSE $\downarrow$ | Mean MAE $\downarrow$ | Mean $R^2 \uparrow$ |
| :--- | :---: | :---: | :---: |
| **`numba_efficient`** | **46.93** | **30.13** | **0.2740** |
| `numpy_statistical` | 54.15 | 36.16 | 0.2316 |
| `polars_statistics` | 54.15 | 36.16 | 0.2316 |
| `tsfresh_minimal` | 54.95 | 36.72 | 0.2074 |

*Finding*: Just as in classification, `numba_efficient` decisively outperforms statistical moments on extrinsic regression, lowering overall RMSE by **13.3%** and MAE by **16.7%** relative to the numpy/polars baselines.

### 2.2 Feature Selection Fidelity and Dimensionality Reduction
| Feature Selector | Mean RMSE $\downarrow$ | Feature Reduction (%) | Selection Stability (Jaccard) | Mean Selection Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| **`select_k_best`** | **51.19** | **57.65%** | **0.7494** | **0.017 s** |
| **`l1` (Lasso)** | 52.40 | 57.61% | 0.6256 | 0.155 s |
| `subsampled` (FDR) | 52.91 | **90.13%** | 0.2291 | 3.053 s |
| `None` (Baseline) | 52.92 | 0.00% | 1.0000 | 0.000 s |
| `tree_importance` | 52.92 | 0.00% | 1.0000 | 0.013 s |
| `variance_threshold` | 52.95 | 8.92% | 0.9993 | 0.019 s |

*Finding*: 
- In regression, univariate $F$-regression filtering (`select_k_best`) and $L_1$-penalised linear selection (`l1`) improved regression performance over unpruned baselines (RMSE $51.19$ vs $52.92$) while eliminating over **57.6%** of uninformative features.
- `subsampled` pruned **90.1%** of features without worsening the RMSE compared to baseline ($52.91$ vs $52.92$).

### 2.3 Model Comparison
| Model | Mean RMSE $\downarrow$ | Mean MAE $\downarrow$ | Mean $R^2 \uparrow$ |
| :--- | :---: | :---: | :---: |
| **`random_forest`** | **49.56** | **31.39** | **0.3057** |
| `ridge` | 55.54 | 38.19 | 0.1667 |

---

## 3. Langevin Drift-Bifurcation Parameter Estimation ($\tau$) Analysis

A core theoretical hypothesis from Andreas is whether statistical and spectral features can recover the true dynamical bifurcation control parameter $\tau \in [3.5, 4.5]$ from noisy velocity trajectories ($v(t)$).

### Results on `drift-bifurcation-reg`:
| Strategy / Extractor | Tau RMSE $\downarrow$ | Tau MAE $\downarrow$ | Tau $R^2 \uparrow$ |
| :--- | :---: | :---: | :---: |
| **`numpy_statistical`** | **0.2751** | **0.2262** | **0.2558** |
| **`polars_statistics`** | **0.2751** | **0.2262** | **0.2558** |
| `tsfresh_minimal` | 0.2752 | 0.2261 | 0.2555 |
| `numba_efficient` | 0.3066 | 0.2487 | 0.0419 |

*Critical Theoretical Finding*:
- Unlike generic sensor datasets where FFT spectral features dominate, **Langevin drift bifurcation parameter estimation is driven by time-domain statistical moments (variance, skewness, kurtosis)**. As $\tau \to \tau_c$, critical slowing down and non-Gaussian fluctuations appear primarily in probability density moments rather than fixed frequency bins.
- Combining statistical moment extractors (`polars_statistics` or `numpy_statistical`) with `select_k_best` or `l1` yields a solid recovery metric ($\text{Tau } R^2 \approx \mathbf{0.28}$, $\text{Tau RMSE} \approx \mathbf{0.27}$).

---

## 4. Analysis Artifacts Generated

All generated statistical distributions, paired $t$-tests, and diagnostic plots are saved in `benchmark_results/TM26a6a__overnight_regression/analysis/`:
- **Langevin $\tau$ Recovery Diagnostics**:
  - `tau_estimation_scatter_*.png` (Scatter plots comparing true $\tau$ vs predicted $\hat{\tau}$ for each extractor/selector pair)
  - `tau_rmse_regression_by_extractor.png`
  - `tau_r2_regression_by_extractor.png`
  - `tau_rmse_regression_by_selector.png`
- **Critical Difference & Distribution Plots**:
  - `rmse_regression_by_extractor.png`
  - `r2_regression_by_extractor.png`
  - `rmse_regression_by_selector.png`
- **Statistical Significance Tables**:
  - `extractor_rmse_regression_ttests.csv`
  - `extractor_tau_rmse_regression_ttests.csv`
  - `combination_tau_r2_regression_ttests.csv`
