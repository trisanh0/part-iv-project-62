# Langevin Drift-Parameter Recovery Experiment Summary (TM26a1X)

**Git Commit Hash**: `70a5d3cfe970a217bb53fc58d9a36f37264732a1`  
**Execution Environment**: Darwin (macOS 15.x / arm64 Apple Silicon), Python 3.11.16, float64 precision  
**Longest Completed Trajectory Length**: $L = 1,000,000$  
**Summary Figure**: `benchmark_results/TM26a1X__langevin/figures/langevin_recovery_summary.png`  

---

## Step 0: Diagnostics and Literature Audit

1. **Simulator and Default Parameters**:
   - **Simulator**: `tsfresh.examples.driftbif_simulation.velocity` wrapped in `load_driftbif`.
   - **Control parameter $\tau$ range**: Drawn from `sample_tau(n, kappa_3=0.3, ratio=0.5, rel_increase=0.15)`.
     With critical bifurcation point $\tau_c = 1 / \kappa_3 = 3.3333$, the sampled range is:
     $$\tau \in [3.0833, 3.5833]$$
     This forms a symmetric window of width $0.5$ centered at $\tau_c = 3.3333$.
   - **Default parameters**: Temporal increment $\Delta t = 0.05$, shape parameter $Q = 1950.0$, noise amplitude $R = 3 \times 10^{-4}$, inverse bifurcation parameter $\kappa_3 = 0.3$.
   - **Target**: Continuous recovery of parameter $\tau$ (regression).

2. **Friedrich and Langevin Docstrings & Notebook Parameters**:
   - `friedrich_coefficients`: Fits a polynomial of order $m$ to conditional average velocity drift $\tilde{h}(x)$ computed across $r$ quantiles.
   - `max_langevin_fixed_point`: Computes the largest positive fixed point $\mathrm{argmax}_x \{h(x) = 0\}$ of the fitted polynomial.
   - **Notebook parameters**: $m = 3$ (order-3 cubic polynomial) and $r = 30$ (30 quantile bins).
   - **Docstring Warning**: Both docstrings explicitly state:
     > *"For short time-series this method is highly dependent on the parameters."*

3. **Inclusion in `EfficientFCParameters`**:
   - **Yes**. Both `friedrich_coefficients` (coefficients 0, 1, 2, 3 with $m=3, r=30$) and `max_langevin_fixed_point` ($m=3, r=30$) are included in `EfficientFCParameters`.

4. **Previous TEMPO Langevin Benchmark Diagnostics**:
   - **Simulator**: `tsfresh.examples.driftbif_simulation.velocity` (wrapped in `generate_drift_bifurcation_dataset`).
   - **Samples per trajectory**: $L = 500$ time steps per trajectory.
   - **Number of trajectories**:
     - $N = 1,000$ trajectories in `TM26a9a` (`drift-bifurcation-reg-1k`) and `TM26a15a`.
     - $N = 200$ trajectories in `TM26a6a` (`drift-bifurcation-reg`).
   - **$\tau$ range**:
     - $\tau \in [3.4, 4.2]$ in `TM26a9a` (restricted supercritical regime).
     - $\tau \in [3.5, 4.5]$ in `TM26a6a`.
   - **Time step**: $\Delta t = 0.05$.

---

## Feature Extraction Telemetry

| Length $L$ | Trajectories ($N$) | Feature Set | Extraction Time (s) | Peak Memory (MB) | Feature Count | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1,000 | 200 | `polars_statistics` | 0.02 | 355.1 | 5 | completed |
| 1,000 | 200 | `numba_efficient` | 0.02 | 424.6 | 447 | completed |
| 1,000 | 200 | `langevin_features` | 1.60 | 427.2 | 5 | completed |
| 1,000 | 200 | `tsfresh_efficient` | 37.87 | 1701.4 | 777 | completed |
| 10,000 | 200 | `polars_statistics` | 0.02 | 486.5 | 5 | completed |
| 10,000 | 200 | `numba_efficient` | 0.16 | 503.3 | 447 | completed |
| 10,000 | 200 | `langevin_features` | 1.97 | 506.3 | 5 | completed |
| 10,000 | 200 | `tsfresh_efficient` | 102.73 | 2136.7 | 777 | completed |
| 100,000 | 200 | `polars_statistics` | 0.26 | 1388.2 | 5 | completed |
| 100,000 | 200 | `numba_efficient` | 1.91 | 1390.6 | 447 | completed |
| 100,000 | 200 | `langevin_features` | 5.48 | 1393.0 | 5 | completed |
| 100,000 | 200 | `tsfresh_efficient` | 2693.33 | 10669.9 | 777 | completed |
| 1,000,000 | 100 | `polars_statistics` | 0.70 | 4341.5 | 5 | completed |
| 1,000,000 | 100 | `numba_efficient` | 9.09 | 4351.0 | 447 | completed |
| 1,000,000 | 100 | `langevin_features` | 12.69 | 4352.2 | 5 | completed |
| 1,000,000 | 100 | `tsfresh_efficient` | 0.00 | 0.0 | 0 | skipped (>90 min limit) |

---

## Benchmark Results Table

Values report out-of-fold cross-validation metrics across 5 folds ($\text{mean} \pm \text{std}$).

| Length $L$ | Feature Set | Selection | Model | Validation $R^2$ | RMSE | MAE | Extraction Time (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1,000 | `polars_statistics` | none | mean_baseline | -0.0127 ± 0.0144 | 0.1474 ± 0.0011 | 0.1292 ± 0.0010 | 0.02 |
| 1,000 | `polars_statistics` | none | ridge | 0.0654 ± 0.0611 | 0.1415 ± 0.0041 | 0.1223 ± 0.0039 | 0.02 |
| 1,000 | `polars_statistics` | none | random_forest | -0.1098 ± 0.0620 | 0.1542 ± 0.0036 | 0.1340 ± 0.0044 | 0.02 |
| 1,000 | `polars_statistics` | fdr | mean_baseline | -0.0127 ± 0.0144 | 0.1474 ± 0.0011 | 0.1292 ± 0.0010 | 0.02 |
| 1,000 | `polars_statistics` | fdr | ridge | 0.0679 ± 0.0640 | 0.1413 ± 0.0042 | 0.1220 ± 0.0040 | 0.02 |
| 1,000 | `polars_statistics` | fdr | random_forest | -0.3062 ± 0.1380 | 0.1672 ± 0.0093 | 0.1466 ± 0.0092 | 0.02 |
| 1,000 | `numba_efficient` | none | mean_baseline | -0.0127 ± 0.0144 | 0.1474 ± 0.0011 | 0.1292 ± 0.0010 | 0.02 |
| 1,000 | `numba_efficient` | none | ridge | -0.4121 ± 0.1538 | 0.1738 ± 0.0095 | 0.1504 ± 0.0076 | 0.02 |
| 1,000 | `numba_efficient` | none | random_forest | -0.0187 ± 0.0533 | 0.1478 ± 0.0035 | 0.1278 ± 0.0030 | 0.02 |
| 1,000 | `numba_efficient` | fdr | mean_baseline | -0.0127 ± 0.0144 | 0.1474 ± 0.0011 | 0.1292 ± 0.0010 | 0.02 |
| 1,000 | `numba_efficient` | fdr | ridge | -0.4121 ± 0.1538 | 0.1738 ± 0.0095 | 0.1504 ± 0.0076 | 0.02 |
| 1,000 | `numba_efficient` | fdr | random_forest | -0.0187 ± 0.0533 | 0.1478 ± 0.0035 | 0.1278 ± 0.0030 | 0.02 |
| 1,000 | `langevin_features` | none | mean_baseline | -0.0127 ± 0.0144 | 0.1474 ± 0.0011 | 0.1292 ± 0.0010 | 1.60 |
| 1,000 | `langevin_features` | none | ridge | -0.0529 ± 0.0284 | 0.1503 ± 0.0021 | 0.1311 ± 0.0028 | 1.60 |
| 1,000 | `langevin_features` | none | random_forest | -0.1825 ± 0.1562 | 0.1590 ± 0.0116 | 0.1365 ± 0.0119 | 1.60 |
| 1,000 | `tsfresh_efficient` | none | mean_baseline | -0.0127 ± 0.0144 | 0.1474 ± 0.0011 | 0.1292 ± 0.0010 | 37.87 |
| 1,000 | `tsfresh_efficient` | none | ridge | -0.4235 ± 0.2232 | 0.1742 ± 0.0119 | 0.1522 ± 0.0094 | 37.87 |
| 1,000 | `tsfresh_efficient` | none | random_forest | 0.0101 ± 0.0429 | 0.1457 ± 0.0032 | 0.1264 ± 0.0027 | 37.87 |
| 1,000 | `tsfresh_efficient` | fdr | mean_baseline | -0.0127 ± 0.0144 | 0.1474 ± 0.0011 | 0.1292 ± 0.0010 | 37.87 |
| 1,000 | `tsfresh_efficient` | fdr | ridge | -0.4235 ± 0.2232 | 0.1742 ± 0.0119 | 0.1522 ± 0.0094 | 37.87 |
| 1,000 | `tsfresh_efficient` | fdr | random_forest | 0.0101 ± 0.0429 | 0.1457 ± 0.0032 | 0.1264 ± 0.0027 | 37.87 |
| 10,000 | `polars_statistics` | none | mean_baseline | -0.0127 ± 0.0144 | 0.1474 ± 0.0011 | 0.1292 ± 0.0010 | 0.02 |
| 10,000 | `polars_statistics` | none | ridge | 0.5680 ± 0.0875 | 0.0959 ± 0.0105 | 0.0772 ± 0.0093 | 0.02 |
| 10,000 | `polars_statistics` | none | random_forest | 0.6098 ± 0.0680 | 0.0913 ± 0.0085 | 0.0754 ± 0.0069 | 0.02 |
| 10,000 | `polars_statistics` | fdr | mean_baseline | -0.0127 ± 0.0144 | 0.1474 ± 0.0011 | 0.1292 ± 0.0010 | 0.02 |
| 10,000 | `polars_statistics` | fdr | ridge | 0.5750 ± 0.0841 | 0.0951 ± 0.0101 | 0.0766 ± 0.0090 | 0.02 |
| 10,000 | `polars_statistics` | fdr | random_forest | 0.6188 ± 0.0743 | 0.0901 ± 0.0093 | 0.0742 ± 0.0076 | 0.02 |
| 10,000 | `numba_efficient` | none | mean_baseline | -0.0127 ± 0.0144 | 0.1474 ± 0.0011 | 0.1292 ± 0.0010 | 0.16 |
| 10,000 | `numba_efficient` | none | ridge | 0.2276 ± 0.0798 | 0.1286 ± 0.0073 | 0.1065 ± 0.0055 | 0.16 |
| 10,000 | `numba_efficient` | none | random_forest | 0.5798 ± 0.0937 | 0.0945 ± 0.0108 | 0.0782 ± 0.0092 | 0.16 |
| 10,000 | `numba_efficient` | fdr | mean_baseline | -0.0127 ± 0.0144 | 0.1474 ± 0.0011 | 0.1292 ± 0.0010 | 0.16 |
| 10,000 | `numba_efficient` | fdr | ridge | 0.6062 ± 0.0879 | 0.0915 ± 0.0108 | 0.0746 ± 0.0096 | 0.16 |
| 10,000 | `numba_efficient` | fdr | random_forest | 0.6032 ± 0.0715 | 0.0920 ± 0.0088 | 0.0754 ± 0.0080 | 0.16 |
| 10,000 | `langevin_features` | none | mean_baseline | -0.0127 ± 0.0144 | 0.1474 ± 0.0011 | 0.1292 ± 0.0010 | 1.97 |
| 10,000 | `langevin_features` | none | ridge | -0.0699 ± 0.3324 | 0.1499 ± 0.0238 | 0.1265 ± 0.0205 | 1.97 |
| 10,000 | `langevin_features` | none | random_forest | 0.2918 ± 0.1798 | 0.1225 ± 0.0159 | 0.0991 ± 0.0142 | 1.97 |
| 10,000 | `tsfresh_efficient` | none | mean_baseline | -0.0127 ± 0.0144 | 0.1474 ± 0.0011 | 0.1292 ± 0.0010 | 102.73 |
| 10,000 | `tsfresh_efficient` | none | ridge | 0.2741 ± 0.0744 | 0.1246 ± 0.0065 | 0.1030 ± 0.0051 | 102.73 |
| 10,000 | `tsfresh_efficient` | none | random_forest | 0.5807 ± 0.0777 | 0.0945 ± 0.0091 | 0.0779 ± 0.0080 | 102.73 |
| 10,000 | `tsfresh_efficient` | fdr | mean_baseline | -0.0127 ± 0.0144 | 0.1474 ± 0.0011 | 0.1292 ± 0.0010 | 102.73 |
| 10,000 | `tsfresh_efficient` | fdr | ridge | 0.5568 ± 0.1517 | 0.0963 ± 0.0169 | 0.0792 ± 0.0153 | 102.73 |
| 10,000 | `tsfresh_efficient` | fdr | random_forest | 0.6040 ± 0.0671 | 0.0919 ± 0.0085 | 0.0759 ± 0.0076 | 102.73 |
| 100,000 | `polars_statistics` | none | mean_baseline | -0.0127 ± 0.0144 | 0.1474 ± 0.0011 | 0.1292 ± 0.0010 | 0.26 |
| 100,000 | `polars_statistics` | none | ridge | 0.9131 ± 0.0132 | 0.0431 ± 0.0036 | 0.0357 ± 0.0029 | 0.26 |
| 100,000 | `polars_statistics` | none | random_forest | 0.9305 ± 0.0121 | 0.0385 ± 0.0034 | 0.0305 ± 0.0024 | 0.26 |
| 100,000 | `polars_statistics` | fdr | mean_baseline | -0.0127 ± 0.0144 | 0.1474 ± 0.0011 | 0.1292 ± 0.0010 | 0.26 |
| 100,000 | `polars_statistics` | fdr | ridge | 0.9141 ± 0.0128 | 0.0428 ± 0.0035 | 0.0355 ± 0.0028 | 0.26 |
| 100,000 | `polars_statistics` | fdr | random_forest | 0.9296 ± 0.0138 | 0.0386 ± 0.0040 | 0.0309 ± 0.0032 | 0.26 |
| 100,000 | `numba_efficient` | none | mean_baseline | -0.0127 ± 0.0144 | 0.1474 ± 0.0011 | 0.1292 ± 0.0010 | 1.91 |
| 100,000 | `numba_efficient` | none | ridge | 0.8873 ± 0.0123 | 0.0491 ± 0.0028 | 0.0401 ± 0.0022 | 1.91 |
| 100,000 | `numba_efficient` | none | random_forest | 0.9249 ± 0.0103 | 0.0401 ± 0.0030 | 0.0322 ± 0.0027 | 1.91 |
| 100,000 | `numba_efficient` | fdr | mean_baseline | -0.0127 ± 0.0144 | 0.1474 ± 0.0011 | 0.1292 ± 0.0010 | 1.91 |
| 100,000 | `numba_efficient` | fdr | ridge | 0.9180 ± 0.0101 | 0.0419 ± 0.0024 | 0.0343 ± 0.0022 | 1.91 |
| 100,000 | `numba_efficient` | fdr | random_forest | 0.9251 ± 0.0118 | 0.0400 ± 0.0033 | 0.0321 ± 0.0028 | 1.91 |
| 100,000 | `langevin_features` | none | mean_baseline | -0.0127 ± 0.0144 | 0.1474 ± 0.0011 | 0.1292 ± 0.0010 | 5.48 |
| 100,000 | `langevin_features` | none | ridge | 0.9054 ± 0.0147 | 0.0449 ± 0.0032 | 0.0372 ± 0.0024 | 5.48 |
| 100,000 | `langevin_features` | none | random_forest | 0.9069 ± 0.0207 | 0.0444 ± 0.0049 | 0.0357 ± 0.0038 | 5.48 |
| 100,000 | `tsfresh_efficient` | none | mean_baseline | -0.0127 ± 0.0144 | 0.1474 ± 0.0011 | 0.1292 ± 0.0010 | 2693.33 |
| 100,000 | `tsfresh_efficient` | none | ridge | 0.9046 ± 0.0024 | 0.0452 ± 0.0006 | 0.0371 ± 0.0008 | 2693.33 |
| 100,000 | `tsfresh_efficient` | none | random_forest | 0.9262 ± 0.0104 | 0.0397 ± 0.0029 | 0.0319 ± 0.0024 | 2693.33 |
| 100,000 | `tsfresh_efficient` | fdr | mean_baseline | -0.0127 ± 0.0144 | 0.1474 ± 0.0011 | 0.1292 ± 0.0010 | 2693.33 |
| 100,000 | `tsfresh_efficient` | fdr | ridge | 0.9145 ± 0.0133 | 0.0427 ± 0.0031 | 0.0351 ± 0.0028 | 2693.33 |
| 100,000 | `tsfresh_efficient` | fdr | random_forest | 0.9258 ± 0.0098 | 0.0398 ± 0.0026 | 0.0319 ± 0.0021 | 2693.33 |
| 1,000,000 | `polars_statistics` | none | mean_baseline | -0.0297 ± 0.0358 | 0.1480 ± 0.0151 | 0.1306 ± 0.0126 | 0.70 |
| 1,000,000 | `polars_statistics` | none | ridge | 0.9690 ± 0.0074 | 0.0256 ± 0.0044 | 0.0210 ± 0.0034 | 0.70 |
| 1,000,000 | `polars_statistics` | none | random_forest | 0.9893 ± 0.0063 | 0.0141 ± 0.0023 | 0.0116 ± 0.0018 | 0.70 |
| 1,000,000 | `polars_statistics` | fdr | mean_baseline | -0.0297 ± 0.0358 | 0.1480 ± 0.0151 | 0.1306 ± 0.0126 | 0.70 |
| 1,000,000 | `polars_statistics` | fdr | ridge | 0.9693 ± 0.0070 | 0.0255 ± 0.0043 | 0.0209 ± 0.0033 | 0.70 |
| 1,000,000 | `polars_statistics` | fdr | random_forest | 0.9897 ± 0.0057 | 0.0140 ± 0.0021 | 0.0116 ± 0.0017 | 0.70 |
| 1,000,000 | `numba_efficient` | none | mean_baseline | -0.0297 ± 0.0358 | 0.1480 ± 0.0151 | 0.1306 ± 0.0126 | 9.09 |
| 1,000,000 | `numba_efficient` | none | ridge | 0.9687 ± 0.0084 | 0.0255 ± 0.0037 | 0.0208 ± 0.0031 | 9.09 |
| 1,000,000 | `numba_efficient` | none | random_forest | 0.9850 ± 0.0044 | 0.0174 ± 0.0009 | 0.0145 ± 0.0007 | 9.09 |
| 1,000,000 | `numba_efficient` | fdr | mean_baseline | -0.0297 ± 0.0358 | 0.1480 ± 0.0151 | 0.1306 ± 0.0126 | 9.09 |
| 1,000,000 | `numba_efficient` | fdr | ridge | 0.9826 ± 0.0060 | 0.0188 ± 0.0026 | 0.0156 ± 0.0020 | 9.09 |
| 1,000,000 | `numba_efficient` | fdr | random_forest | 0.9865 ± 0.0052 | 0.0164 ± 0.0014 | 0.0135 ± 0.0012 | 9.09 |
| 1,000,000 | `langevin_features` | none | mean_baseline | -0.0297 ± 0.0358 | 0.1480 ± 0.0151 | 0.1306 ± 0.0126 | 12.69 |
| 1,000,000 | `langevin_features` | none | ridge | 0.9781 ± 0.0050 | 0.0215 ± 0.0036 | 0.0174 ± 0.0034 | 12.69 |
| 1,000,000 | `langevin_features` | none | random_forest | 0.9773 ± 0.0071 | 0.0214 ± 0.0013 | 0.0177 ± 0.0011 | 12.69 |

---

## Most Frequently Selected Features under FDR by Feature Set

### Trajectory Length $L = 100,000$ (Direct Competition with `tsfresh_efficient`)

#### 1. Feature Set: `tsfresh_efficient` (Total 129 Features Selected)
*Top 15 Most Frequently Selected Features*:

| Rank | Feature Identifier | Selection Frequency (out of 5 Folds) |
| :---: | :--- | :---: |
| 1 | `velocity__abs_energy` | 5/5 |
| 2 | `velocity__fft_coefficient__attr_"abs"__coeff_11` | 5/5 |
| 3 | `velocity__ratio_beyond_r_sigma__r_3` | 5/5 |
| 4 | `velocity__change_quantiles__f_agg_"var"__isabs_False__qh_0.4__ql_0.0` | 5/5 |
| 5 | `velocity__change_quantiles__f_agg_"var"__isabs_True__qh_0.6__ql_0.2` | 5/5 |
| 6 | `velocity__fft_coefficient__attr_"abs"__coeff_5` | 5/5 |
| 7 | `velocity__change_quantiles__f_agg_"mean"__isabs_True__qh_0.4__ql_0.0` | 5/5 |
| 8 | `velocity__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.0` | 5/5 |
| 9 | `velocity__fft_coefficient__attr_"abs"__coeff_8` | 5/5 |
| 10 | `velocity__quantile__q_0.7` | 5/5 |

*Langevin Features Selected under FDR in `tsfresh_efficient`*:
- `velocity__friedrich_coefficients__coeff_2__m_3__r_30`: **5/5 folds (100% selected)**
- `velocity__max_langevin_fixed_point__m_3__r_30`: **5/5 folds (100% selected)**

#### 2. Feature Set: `numba_efficient` ($L = 100,000$, Top 10 Features)

| Rank | Feature Identifier | Selection Frequency (out of 5 Folds) |
| :---: | :--- | :---: |
| 1 | `var_0__abs_energy` | 5/5 |
| 2 | `var_0__root_mean_square` | 5/5 |
| 3 | `var_0__autocorrelation__lag_2` | 5/5 |
| 4 | `var_0__standard_deviation` | 5/5 |
| 5 | `var_0__variance` | 5/5 |
| 6 | `var_0__autocorrelation__lag_1` | 5/5 |
| 7 | `var_0__autocorrelation__lag_9` | 5/5 |
| 8 | `var_0__autocorrelation__lag_7` | 5/5 |
| 9 | `var_0__autocorrelation__lag_8` | 5/5 |
| 10 | `var_0__autocorrelation__lag_3` | 5/5 |

---

### Trajectory Length $L = 1,000,000$ (Longest Completed Length)

#### 1. Feature Set: `polars_statistics`

| Rank | Feature Identifier | Selection Frequency (out of 5 Folds) |
| :---: | :--- | :---: |
| 1 | `velocity__energy` | 5/5 |
| 2 | `velocity__std` | 5/5 |
| 3 | `velocity__min` | 5/5 |
| 4 | `velocity__max` | 5/5 |

#### 2. Feature Set: `numba_efficient`

| Rank | Feature Identifier | Selection Frequency (out of 5 Folds) |
| :---: | :--- | :---: |
| 1 | `var_0__abs_energy` | 5/5 |
| 2 | `var_0__root_mean_square` | 5/5 |
| 3 | `var_0__autocorrelation__lag_2` | 5/5 |
| 4 | `var_0__standard_deviation` | 5/5 |
| 5 | `var_0__variance` | 5/5 |
| 6 | `var_0__autocorrelation__lag_1` | 5/5 |
| 7 | `var_0__autocorrelation__lag_9` | 5/5 |
| 8 | `var_0__autocorrelation__lag_7` | 5/5 |
| 9 | `var_0__autocorrelation__lag_8` | 5/5 |
| 10 | `var_0__autocorrelation__lag_3` | 5/5 |

---

## Unexpected Findings & Observations (Empirical Summary)

1. **Recovery Transition Threshold Across Length**:
   - At $L = 1,000$ ($N=200$), all feature sets fail to recover $\tau$, yielding $R^2 \le 0.068$ and negative $R^2$ values for unselected tree and linear models.
   - At $L = 10,000$, $R^2$ transitions sharply from near-zero to $\sim 0.61$ (`polars_statistics`: $0.6188$, `numba_efficient`: $0.6062$, `tsfresh_efficient`: $0.6040$).
   - At $L = 100,000$, all feature sets achieve $R^2 > 0.90$.
   - At $L = 1,000,000$, models achieve $R^2 = 0.978$ to $0.989$ ($\text{RMSE} = 0.014$ on a parameter spanning $[3.08, 3.58]$).

2. **Performance of Generic Baseline Statistics vs Dedicated Langevin Estimators**:
   - `polars_statistics` (5 baseline summary statistics: mean, std, min, max, energy) matches or slightly outperforms the full 777-feature `tsfresh_efficient` and 5-feature `langevin_features` across all lengths:
     - At $L = 10,000$: $R^2 = 0.6188$ (`polars_statistics`) vs $0.6040$ (`tsfresh_efficient`) vs $0.2918$ (`langevin_features`).
     - At $L = 100,000$: $R^2 = 0.9305$ (`polars_statistics`) vs $0.9262$ (`tsfresh_efficient`) vs $0.9069$ (`langevin_features`).
     - At $L = 1,000,000$: $R^2 = 0.9897$ (`polars_statistics`) vs $0.9865$ (`numba_efficient`) vs $0.9781$ (`langevin_features`).

3. **FDR Selection of Friedrich & Langevin Features**:
   - In `tsfresh_efficient` at $L = 100,000$, out of 777 candidates, 129 features survived FDR hypothesis testing ($q=0.05$).
   - Both `velocity__friedrich_coefficients__coeff_2__m_3__r_30` and `velocity__max_langevin_fixed_point__m_3__r_30` survived FDR filtering across 100% (5/5) of training folds.

4. **Extraction Runtime & Memory Scaling**:
   - At $L = 100,000$ ($N = 200$), `tsfresh_efficient` required 2,693.33 seconds (44.9 minutes) and reached 10,669.9 MB peak RAM.
   - Under the same dataset, `numba_efficient` required 1.91 seconds (1,390.6 MB RAM) and `langevin_features` required 5.48 seconds (1,393.0 MB RAM), achieving an extraction speedup of $1,410\times$ and $491\times$ respectively with equivalent model accuracy ($R^2 \approx 0.925$).
   - At $L = 1,000,000$, `tsfresh_efficient` was skipped according to the 90-minute safety threshold rule. `polars_statistics` extracted in 0.70 seconds, `numba_efficient` in 9.09 seconds, and `langevin_features` in 12.69 seconds.

---

## Artifact and Figure Locations

- **Two-Panel Figure**: `benchmark_results/TM26a1X__langevin/figures/langevin_recovery_summary.png`
- **Raw Benchmark Results**: `benchmark_results/TM26a1X__langevin/benchmark_results.csv`
- **Extraction Telemetry**: `benchmark_results/TM26a1X__langevin/extraction_telemetry.csv`
- **FDR Feature Selection Log**: `benchmark_results/TM26a1X__langevin/fdr_selected_features.csv`
- **System Environment Metadata**: `benchmark_results/TM26a1X__langevin/environment.json`
- **Git Commit Hash**: `benchmark_results/TM26a1X__langevin/git_hash.txt`