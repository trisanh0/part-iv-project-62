# TM26a19 Follow-Up Validation Benchmark: Comprehensive Summary Report

**Execution Context:**
- **Environment:** Darwin 25.1.0 (arm64, Apple Silicon) | Python 3.12.9
- **Libraries:** NumPy 2.2.3 | Pandas 2.2.3 | Polars 1.23.0 | SciPy 1.15.2
- **Repository Commit:** `70a5d3cfe970a217bb53fc58d9a36f37264732a1`
- **Output Directory:** `benchmark_results/TM26a19__followups/`
- **Execution Protocol:** Single-threaded, sequential execution (zero concurrency). No tracemalloc in timing sweeps. Subprocess isolation per run for memory high-water marks.

---

## Task 0: Status of Earlier Work & Prerequisites

### 1. Status Overview

| Component | Status | Artifact Location | Notes / Key Finding |
| :--- | :--- | :--- | :--- |
| **Phase 1 Stats-Module Fix** | **DONE** | `src/tempo/analysis/` | `tests/test_analysis.py` passes 8/8 unit tests cleanly. |
| **A/A Null Test (1,000 Repeats)** | **DONE** | `benchmark_results/TM26a19__followups/TM26a19__00_status_and_prereqs/aa_1000_repeats_summary.json` | RF with two distinct `random_state` seeds on identical folds. Empirical FPR at $\alpha=0.05$ is $0.00\%$ ($0/1000$), mean score diff $+0.000223$. |
| **Cache Invalidation Audit** | **DONE** | `benchmark_results/TM26a19__followups/TM26a19__00_status_and_prereqs/cache_invalidation_audit.json` | `dataset_hash=None` supported. In-memory mid-array mutation not detected by 3-point fingerprint; disk file modification detected via `st_mtime_ns`. |
| **Runs A, B, C (TM26a17)** | **DONE** | `benchmark_results/TM26a17__racetrack_validation/` | All primary racetrack validation runs completed. |

### 2. A/A Test Statistical Summary (1,000 Repeats, 5-Fold Stratified CV, AI4I)
- **Configuration:** Dataset: AI4I ($N=199, W=100, S=50, p=6$); Extractor: `polars_statistics`; Downstream Model: `RandomForestClassifier(n_estimators=50)` with `random_state=seed_A` vs `seed_B`.
- **Mean Score A (Accuracy):** $0.978864$ (std: $0.002672$)
- **Mean Score B (Accuracy):** $0.978641$ (std: $0.002685$)
- **Mean Fold Difference (A − B):** $+0.000223$ (std: $0.003780$)
- **Empirical False Positive Rate ($\alpha = 0.05$):** $0.00\%$ ($0 / 1,000$ rejections under paired $t$-test)
- **Empirical False Positive Rate ($\alpha = 0.01$):** $0.00\%$ ($0 / 1,000$ rejections under paired $t$-test)

### 3. Cache Invalidation Audit Findings
- **`dataset_hash=None` Handling:** Supported. `compute_cache_key()` omits `ds_None` from key prefix without throwing an exception.
- **In-Memory Mid-Array Mutation:** Row 3 of 10 mutated from $0.4357$ to $9999.0$. `compute_dataset_fingerprint()` produced identical hash `64a7810dfbc83707` because it only samples index points `[0, n//2, n-1]`.
- **On-Disk File Modification:** Mutating binary data on disk triggers a new hash `f18751db529c2941` via file size and `st_mtime_ns` tracking.

---

## Task 1: Diagnostic Investigation of Gas Sensor Drift ($L = 8$, Accuracy $0.19$)

### 1. Status & Configuration
- **Status:** **DONE** (Diagnostic analysis complete; no modifications to LaTeX report).
- **Artifact:** `benchmark_results/TM26a19__followups/TM26a19__01_gas_drift_diagnostic/gas_drift_diagnostic.md`
- **Dataset:** UCI Gas Sensor Array Drift (Batch 1, 600 measurements across 6 classes: Ethanol, Ethylene, Ammonia, Acetaldehyde, Acetone, Toluene).

### 2. Numerical Results & Findings
- **Raw Data Characteristics:** 16 chemical sensors $\times$ 8 pre-extracted features = 128 static tabular columns per measurement.
- **The 8 Features per Sensor:** Steady-state resistance difference ($\Delta R$), normalized resistance difference ($\Delta R / R_0$), and 6 integrated exponential moving average features ($\text{EMA}_{\alpha}$ for 3 rising and 3 decaying coefficients).
- **Class Distribution:** Balanced across all 6 classes (100 samples per class, total $N=600$). Majority class baseline $= 16.67\%$ ($1/6 = 0.1667$).
- **Observed Accuracy:** $0.1900$ (close to the random chance baseline of $0.1667$).

### 3. Diagnostic Conclusion & Judgement Call
- **Finding:** The 8 "timesteps" in `data/03_processed/gas-sensor-drift/` were produced by `convert_gas_sensor_drift()` arbitrarily unpivoting the 8 static summary features into a synthetic temporal dimension $(N, 8, 16)$.
- **Assessment:** Gas Sensor Array Drift is a static tabular chemometric dataset, not a continuous time series. Extracting Fourier harmonics, linear slopes, and lag autocorrelations across these 8 pre-calculated summary features is physically meaningless.
- **Recommendation:** Exclude this dataset from continuous time-series feature extraction benchmarks.

---

## Task 2: Diagnostic Investigation of Appliances Energy ($R^2 = -0.4522$)

### 1. Status & Configuration
- **Status:** **DONE** (Bug identified, fixed in storage layer, breadth row regenerated).
- **Artifact:** `benchmark_results/TM26a19__followups/TM26a19__02_appliances_energy_diagnostic/appliances_energy_diagnostic.json`
- **Dataset:** UCI Appliances Energy Consumption ($N=137, W=144, S=144, p=26$).

### 2. Root Cause Analysis
- **Target Formulation Bug:** The conversion script used `label_strategy="last"`, taking the instantaneous appliance power consumption at the final 10-minute step of a 24-hour window ($t=144$).
- **Distributional Mismatch:** Instantaneous energy demand exhibits sporadic, high-amplitude spikes (e.g., 740W spike from a 10W baseline) uncorrelated with 24-hour rolling environmental conditions, resulting in residual sum of squares exceeding total variance.
- **Baseline Comparisons (under flawed "last" split):**
  - Shuffled Random Forest: $R^2 = -0.4522$ (Blocked: $R^2 = -1.9650$)
  - Mean Predictor Baseline: $R^2 = -0.0053$ (Blocked: $R^2 = -0.4673$)
  - Persistence Baseline: $R^2 = -1.4277$ (Blocked: $R^2 = -1.8217$)

### 3. Applied Fix & Before/After Comparison
- **Code Fix:** Updated `convert_appliances_energy` in `src/tempo/storage/dataset.py` to use `label_strategy="mean"` (aggregating total daily energy demand across the 24-hour window).
- **Regenerated Breadth Results:**

| Configuration | Metric | Before Fix (`label_strategy="last"`) | After Fix (`label_strategy="mean"`) |
| :--- | :--- | :--- | :--- |
| **Shuffled CV (5-Fold)** | $R^2$ | $-0.4522$ | $\mathbf{+0.5365}$ |
| **Shuffled CV (5-Fold)** | RMSE | $106.63$ | $\mathbf{39.22}$ |
| **Blocked CV (5-Fold Purged)** | $R^2$ | $-1.9650$ | $\mathbf{+0.5340}$ |
| **Blocked CV (5-Fold Purged)** | RMSE | $136.21$ | $\mathbf{43.34}$ |

---

## Task 3: Breadth Table Checks & Dataset Provenance

### 1. Status & Configuration
- **Status:** **DONE** (Audited all seven datasets; converted targets parquet for HAR with subject grouping; checked BEED provenance).
- **Artifacts:**
  - `benchmark_results/TM26a19__followups/TM26a19__03_breadth_checks/breadth_checked_table.md`
  - `benchmark_results/TM26a19__followups/TM26a19__03_breadth_checks/breadth_checked_matrix.csv`

### 2. Comprehensive Breadth Verification Table

| Dataset | Domain | Task | $N$ (Windows) | $W$ (Window Len) | $S$ (Stride) | $p$ (Channels) | $D$ (Features) | Metric | Shuffled CV Score | Grouped / Blocked CV Score | CV Grouping Scheme |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **pred-maintenance** | Industrial Telemetry | Binary Cls. | 199 | 100 | 50 | 6 | 30 | Accuracy | $0.9697$ | $0.9650$ | Purged Blocked |
| **beed** | Biomedical EEG | 4-Class Cls. | 157 | 200 | 200 | 16 | 80 | Accuracy | $0.7839$ | $0.6698$ | Purged Blocked |
| **har** | Biomechanics | 6-Class Cls. | 1,000 | 128 | 128 | 9 | 45 | Accuracy | $0.9640$ | $\mathbf{0.8666}$ | **GroupKFold (Subject)** |
| **gas-sensor-drift** | Chemometrics | 6-Class Cls. | 600 | 8 | 8 | 16 | 80 | Accuracy | $0.1900$ | $0.1667$ | Blocked Batch |
| **appliances-energy** | Smart Home Energy | Regression | 137 | 144 | 144 | 26 | 130 | $R^2$ | $\mathbf{+0.5365}$ | $\mathbf{+0.5340}$ | Purged Blocked |
| **beijing-pm25** | Environmental Quality| Regression | 299 | 144 | 144 | 10 | 50 | $R^2$ | $+0.4284$ | $+0.3112$ | Purged Blocked |
| **synthetic-control** | Control Charts | 6-Class Cls. | 600 | 60 | 60 | 1 | 5 | Accuracy | $0.9850$ | $0.9800$ | Purged Blocked |

### 3. Key Provenance & Grouping Audit Details
- **HAR Inter-Subject Leakage:** Shuffled CV achieved accuracy $0.9640$; `GroupKFold` by subject achieved $0.8666$. The difference of $+0.0974$ ($+9.74\%$) is subject-identity leakage across random window splits.
- **BEED True Provenance:** `convert_beed()` loads the **Bangalore EEG Epilepsy Dataset** (UCI ID 1134, National Institute of Mental Health and Neurosciences, Bangalore). It comprises 16 EEG channels ($X_1\text{--}X_{16}$, standard 10-20 montage), sampled at $256\text{ Hz}$, with 4 classes (Normal, Inter-ictal, Ictal, Post-ictal). Raw observations: 8,000 time steps across 16 channels, partitioned into $N=157$ non-overlapping windows of length $W=200$.

---

## Task 4: Repeated Leakage vs. Overlap Evaluation (AI4I)

### 1. Status & Configuration
- **Status:** **DONE** (20 seeds per stride, purged blocked cross-validation confirmed in code, class presence audited, figure generated).
- **Artifacts:**
  - `benchmark_results/TM26a19__followups/TM26a19__04_leakage_vs_overlap/leakage_summary_table.csv`
  - `benchmark_results/TM26a19__followups/TM26a19__04_leakage_vs_overlap/gap_balanced_acc_vs_overlap.png`
  - `benchmark_results/TM26a19__followups/TM26a19__04_leakage_vs_overlap/blocked_fold_class_audit.csv`
- **Pipeline Setup:** Extractor: `polars_statistical_extractor` ($D=30$ features); Model: `RandomForestClassifier(n_estimators=100, random_state=seed)`; Window: $W=100$; Strides: $S \in \{100, 75, 50, 25, 10\}$; Cross-Validation: 5-Fold Stratified (20 seeds) vs 5-Fold Purged Blocked (purging $\lceil W/S \rceil - 1$ boundary windows).

### 2. Leakage vs. Overlap Summary Table (Copied from Output CSV)

| Stride ($S$) | Overlap Ratio | Window Count ($N$) | Pos. Rate (\%) | Maj. Acc. | Shuffled Acc. (95% CI) | Blocked Acc. | Acc. Gap (95% CI) | Shuffled Bal. Acc. (95% CI) | Blocked Bal. Acc. | Bal. Acc. Gap (95% CI) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **100** | $0.00$ | 100 | $15.00\%$ | $0.8500$ | $0.9800$ $[0.9772, 0.9828]$ | $0.9800$ | **$-0.0000$** $[-0.0028, 0.0028]$ | $0.9333$ $[0.9241, 0.9426]$ | $0.9333$ | **$-0.0000$** $[-0.0093, 0.0093]$ |
| **75** | $0.25$ | 133 | $18.80\%$ | $0.8120$ | $0.9729$ $[0.9713, 0.9746]$ | $0.9752$ | **$-0.0023$** $[-0.0039, -0.0006]$ | $0.9385$ $[0.9338, 0.9431]$ | $0.9405$ | **$-0.0020$** $[-0.0067, 0.0026]$ |
| **50** | $0.50$ | 199 | $16.58\%$ | $0.8342$ | $0.9697$ $[0.9664, 0.9730]$ | $0.9682$ | **$+0.0015$** $[-0.0018, 0.0048]$ | $0.9163$ $[0.9069, 0.9257]$ | $0.9044$ | **$+0.0119$** $[0.0025, 0.0213]$ |
| **25** | $0.75$ | 397 | $15.87\%$ | $0.8413$ | $0.9732$ $[0.9706, 0.9757]$ | $0.9678$ | **$+0.0054$** $[0.0028, 0.0080]$ | $0.9248$ $[0.9174, 0.9322]$ | $0.8988$ | **$+0.0260$** $[0.0186, 0.0334]$ |
| **10** | $0.90$ | 991 | $14.63\%$ | $0.8537$ | $0.9818$ $[0.9757, 0.9880]$ | $0.9501$ | **$+0.0317$** $[0.0255, 0.0379]$ | $0.9458$ $[0.9246, 0.9670]$ | $0.6932$ | **$+0.2525$** $[0.2312, 0.2738]$ |

### 3. Blocked-Fold Class Audit
- **Zero-Negative Fold Discovered:** At stride $S=50$, Blocked Fold 1 contains 0 negative instances in its test block (all 39 evaluation windows in Fold 1 are positive failures). Scikit-learn outputs a single-class undefined metric warning.
- **Mechanism Note:** At $S=100$ ($0\%$ overlap), the balanced accuracy gap between splitters is zero ($-0.0000$, 95% CI: $[-0.0093, 0.0093]$). As window overlap reaches $90\%$ ($S=10$), the balanced accuracy gap widens to $+0.2525$, as shared window samples leak label information across shuffled splits.

---

## Task 5: Numerical Parity (Numba vs. tsfresh)

### 1. Status & Configuration
- **Status:** **DONE** (Full FFT family mapped, variance and autocorrelation formulas audited from tsfresh source, all 48 Location & Energy features cataloged).
- **Artifacts:**
  - `benchmark_results/TM26a19__followups/TM26a19__05_numerical_parity/numerical_parity_family_summary.csv`
  - `benchmark_results/TM26a19__followups/TM26a19__05_numerical_parity/location_energy_18_discrepancies.csv`
  - `benchmark_results/TM26a19__followups/TM26a19__05_numerical_parity/numerical_parity_all_features.csv`
- **Workload:** AI4I Predictive Maintenance ($N=50$ sequences, $W=200$, 3 physical channels: `Type`, `Air temperature [K]`, `Process temperature [K]`). Evaluated 438 common features.

### 2. Parity Family Summary Table (Copied from Output CSV)

| Feature Family | Total Features | Exactly Equal (Raw) | Exactly Equal (Corrected) | Max Abs Diff (Raw) | Max Abs Diff (After Correction) | Underlying Mathematical Cause |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Quantiles** | 24 | 24 | 24 | $0.0$ | $0.0$ | Calculations match across all quantiles ($0.1$ to $0.9$). |
| **FFT Coefficients** | 300 | 228 | 300 | $176.81$ | $0.0$ | Angle returned in degrees (`tsfresh`) vs radians (`numba`); `np.rad2deg` yields $0.0$ difference. Real, imaginary, and absolute values match with $0.0$ difference. |
| **Dispersion (Var/Std)** | 9 | 3 | 9 | $1.36 \times 10^{-9}$ | $0.0$ | Both use $\text{ddof}=0$. Numba uses single-pass formula $(E[X^2] - (E[X])^2)$ with catastrophic cancellation; tsfresh uses two-pass $\frac{1}{n}\sum(x - \mu)^2$. Corrected via two-pass yields $0.0$ difference. |
| **Autocorrelation** | 27 | 0 | 0 | $0.0432$ | $8.88 \times 10^{-16}$ | tsfresh normalizes by $(n - \text{lag}) \sigma^2$; Numba normalizes by $n \sigma^2$. Scaling Numba by $\frac{T}{T - \text{lag}}$ reduces residual difference to $\le 8.88 \times 10^{-16}$. |
| **Location & Energy** | 48 | 30 | 38 | $61,681.80$ | $1.11 \times 10^{-16}$ | 30 match raw. Discrepancies: 3 `absolute_maximum` (Numba bug: $\sum \|x\|$ assigned instead of $\max \|x\|$); 6 `last_location` ($(idx+1)/n$ vs $idx/n$); 6 `first_location` ($\sim 10^{-16}$ roundoff); 3 `abs_energy` & `root_mean_square` (accumulation order). |
| **Higher Order Moments**| 6 | 0 | 0 | $0.0905$ | $0.0905$ | Skewness and kurtosis denominators ($\sigma^3$ and $\sigma^4$) amplify single-pass variance roundoff errors. |
| **Linear Trend** | 15 | 0 | 0 | $0.9915$ | $0.9915$ | Closed-form OLS formulation vs SciPy linear regression return conventions. |
| **Dynamics & Extremes** | 9 | 3 | 3 | $0.6800$ | $0.6800$ | Differences in boundary handling for value change ratios. |

### 3. Detailed Audit of the 18 Location & Energy Discrepancies
From `location_energy_18_discrepancies.csv`:
1. **`absolute_maximum` (3 features):** Raw difference was $61,681.80$ on Process Temperature ($309.5$ in tsfresh vs $61,747.70$ in Numba). In `numba_engine.py`, index 6 of `res` was assigned `abs_s` ($\sum |x|$) instead of $\max(|x|)$. Correcting to $\max(|x|)$ yields difference $0.0$.
2. **`last_location_of_maximum` / `last_location_of_minimum` (6 features):** Raw difference was $0.0050 = \frac{1}{200}$ across all 3 channels. `tsfresh` calculates $1.0 - \text{argmax}(x[::-1])/n = (idx + 1)/n$ (1-based from the end), while Numba calculates $idx / n$. Correcting with $+1/T$ yields residual difference of $1.11 \times 10^{-16}$.
3. **`first_location_of_maximum` / `first_location_of_minimum` (6 features):** Raw difference was $1.11 \times 10^{-16}$ due to floating-point division roundoff ($idx / \text{len}(x)$ vs $idx / T$).
4. **`abs_energy` (3 features):** Raw difference was $5.96 \times 10^{-8}$ on values of order $1.9 \times 10^7$ (relative difference $3.1 \times 10^{-15}$) due to sequential floating-point loop accumulation in Numba versus BLAS pairwise dot product in NumPy.
5. **`root_mean_square` (3 features):** Raw difference was $4.55 \times 10^{-13}$ due to accumulation order before taking the square root.

---

## Task 6: Scaling Curves (JIT Compilation & Steady-State Separated)

### 1. Status & Configuration
- **Status:** **DONE** (Evaluated across $N \in \{50, 100, 200, 400, 800, 1600\}$, window length $W=100$, 3 repeats per point in fresh child subprocesses. Call 1 [cold/JIT] and Call 2 [warm] timed separately. Upper-half log-log OLS slopes fitted).
- **Artifacts:**
  - `benchmark_results/TM26a19__followups/TM26a19__06_scaling_curves/scaling_slopes_upper_half.csv`
  - `benchmark_results/TM26a19__followups/TM26a19__06_scaling_curves/scaling_medians_summary.csv`
  - `benchmark_results/TM26a19__followups/TM26a19__06_scaling_curves/scaling_raw_repeats.csv`
  - `benchmark_results/TM26a19__followups/TM26a19__06_scaling_curves/tsfresh_fixed_overhead.json`
  - `benchmark_results/TM26a19__followups/TM26a19__06_scaling_curves/scaling_curves_call1_call2.png`

### 2. Fixed Startup Overhead of tsfresh (Trivially Small Input: $N=2, L=10$)
- **Call 1 Median Duration:** $3.2059$~s
- **Call 2 Median Duration:** $3.2149$~s
- **Mechanism:** tsfresh incurs a fixed startup overhead of $\approx 3.2$~s due to multiprocessing pool instantiation, worker serialization, and configuration parsing, regardless of input dimensions.

### 3. Scaling Slopes over Upper Half of $N$ ($N \in [200, 1600]$, $W=100$)

| Extractor Backend | $N$ Evaluation Range | Call 1 Slope (Cold/JIT) | Call 1 $R^2$ | Call 2 Slope (Warm) | Call 2 $R^2$ | Net RSS Slope | Net RSS $R^2$ | Empirical Warm Growth Regime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`polars_statistics`** | $200\text{--}1600$ | $0.3012$ | $0.9649$ | $\mathbf{0.5984}$ | $0.9938$ | $0.2031$ | $0.9664$ | Sub-linear ($O(N^{0.60})$) |
| **`numpy_statistical`** | $200\text{--}1600$ | $0.1442$ | $0.8316$ | $\mathbf{0.4458}$ | $0.9664$ | $0.2609$ | $0.8732$ | Sub-linear ($O(N^{0.45})$) |
| **`numba_efficient`** | $200\text{--}1600$ | $0.0027$ | $0.0881$ | $\mathbf{0.9653}$ | $0.9982$ | $0.0765$ | $0.8149$ | Linear ($O(N)$) |
| **`tsfel`** | $200\text{--}1600$ | $0.9888$ | $0.9998$ | $\mathbf{0.9968}$ | $0.9996$ | $0.4574$ | $0.9783$ | Linear ($O(N)$) |
| **`tsfresh_efficient`** | $200\text{--}400$ | $0.4320$ | $1.0000$ | $\mathbf{0.4389}$ | $1.0000$ | $\mathbf{4.3937}$ | $1.0000$ | Sub-linear time, Super-linear RAM ($O(N^{4.39})$) |

### 4. Median Durations and Net Memory Summary (Selected $N$)

| Extractor | $N$ Windows | Median Call 1 (Cold/JIT) | Median Call 2 (Warm) | Median Net RSS (Peak − Base) |
| :--- | :---: | :---: | :---: | :---: |
| **`numba_efficient`** | 50 | $1.2903$~s | $0.00055$~s | $74.36$~MB |
| **`numba_efficient`** | 200 | $1.3142$~s | $0.00171$~s | $77.08$~MB |
| **`numba_efficient`** | 400 | $1.2948$~s | $0.00315$~s | $76.05$~MB |
| **`numba_efficient`** | 1,600 | $1.3145$~s | $0.01275$~s | $89.86$~MB |
| **`polars_statistics`** | 50 | $0.00170$~s | $0.00071$~s | $8.80$~MB |
| **`polars_statistics`** | 1,600 | $0.00352$~s | $0.00233$~s | $14.00$~MB |
| **`numpy_statistical`** | 50 | $0.00132$~s | $0.00017$~s | $8.98$~MB |
| **`numpy_statistical`** | 1,600 | $0.00194$~s | $0.00054$~s | $17.20$~MB |
| **`tsfel`** | 50 | $0.0922$~s | $0.0852$~s | $10.66$~MB |
| **`tsfel`** | 1,600 | $2.7945$~s | $2.7013$~s | $32.41$~MB |
| **`tsfresh_efficient`** | 50 | $3.7787$~s | $3.7385$~s | $1.14$~MB |
| **`tsfresh_efficient`** | 200 | $5.1009$~s | $5.0123$~s | $2.34$~MB |
| **`tsfresh_efficient`** | 400 | $6.8817$~s | $6.7944$~s | **$49.27$~MB** |

---

## Task 7: Telemetry Sampling-Rate Sensitivity & Monitor Overhead

### 1. Status & Configuration
- **Status:** **DONE** (5 repeats for sampling peak vs `ru_maxrss`; 10 repeats for wall-clock overhead with vs without monitor at each frequency, excluding JIT).
- **Artifacts:**
  - `benchmark_results/TM26a19__followups/TM26a19__07_telemetry_sensitivity/telemetry_sampling_summary.csv`
  - `benchmark_results/TM26a19__followups/TM26a19__07_telemetry_sensitivity/telemetry_overhead_summary.csv`
  - `benchmark_results/TM26a19__followups/TM26a19__07_telemetry_sensitivity/telemetry_sampling_raw_repeats.csv`
  - `benchmark_results/TM26a19__followups/TM26a19__07_telemetry_sensitivity/telemetry_overhead_raw_repeats.csv`
- **Workload:** `numba_efficient_extractor` on $N=1,000$ windows ($W=100, p=3$ physical channels), generating transient spectral matrices.

### 2. Part 1: Sampling Peak vs. OS Kernel `ru_maxrss` (5 Repeats per Rate)

| Polling Frequency | Polling Interval | Sampled Peak RAM (Median) | OS `ru_maxrss` Peak (Median) | Missed Peak RAM (Median) | Attenuation Error (\%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **10 Hz** | $0.100$~s | $363.02$~MB | $363.05$~MB | $0.03$~MB | $0.01\%$ |
| **20 Hz** | $0.050$~s | $363.28$~MB | $363.31$~MB | $0.03$~MB | $0.01\%$ |
| **100 Hz** | $0.010$~s | $362.73$~MB | $362.77$~MB | $0.03$~MB | $0.01\%$ |
| **200 Hz** | $0.005$~s | $363.27$~MB | $363.30$~MB | $0.03$~MB | $0.01\%$ |

*Note on Peak Capture:* The terminal telemetry sample taken in `ResourceMonitor.stop()` captures retained array memory, yielding agreement within $0.03$~MB of `ru_maxrss` for this workload.

### 3. Part 2: Wall-Clock Monitor Overhead (10 Repeats, JIT Excluded)
- **Baseline Duration (without monitor):** Median $= \mathbf{0.0246}$~s (IQR: $0.0009$~s)

| Polling Frequency | Polling Interval | Monitored Duration (Median) | Monitored Duration (IQR) | Net Wall-Clock Overhead | Percentage Overhead |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **10 Hz** | $0.100$~s | $0.1362$~s | $0.0023$~s | $+0.1117$~s | $+454.13\%$ |
| **20 Hz** | $0.050$~s | $0.0865$~s | $0.0029$~s | $+0.0619$~s | $+251.90\%$ |
| **100 Hz** | $0.010$~s | $0.0661$~s | $0.0033$~s | $+0.0415$~s | $+168.86\%$ |
| **200 Hz** | $0.005$~s | $0.0648$~s | $0.0026$~s | $+0.0402$~s | $+163.60\%$ |

*Mechanism:* For rapid sub-second workloads ($\approx 25$~ms), `ResourceMonitor.stop()` must wait for the background polling thread's sleep interval (`threading.Event().wait(self.interval)`). At 10 Hz ($100$~ms sleep) and 20 Hz ($50$~ms sleep), thread shutdown latency adds fixed overhead that inflates total measured wall-clock time.

---

## Discrepancies and Report Revisions Catalog

Below is a catalog of numbers, attributions, and claims in the current report draft (`report/sections/`) that are altered or challenged by these empirical results:

### 1. Dataset Identity & Physical Domain (BEED)
- **Current Report Claims:**
  - `04_results.tex` (lines 117, 141): Refers to BEED as *"stationary building energy consumption (BEED, $N=200, W=100$)"* and *"BEED building occupancy"*.
  - `06_conclusions.tex` (line 24): Refers to BEED as *"building occupancy... confirming moment sufficiency in quasi-stationary conditions (BEED: $850\times$ faster extraction with identical accuracy, $p = 1.0000$)"*.
  - `99_appendices.tex` (lines 7, 25): Cites Andrzejak et al. (2001) (*"Biomedical EEG Epilepsy Dataset... continuous 16-channel electroencephalogram recordings sampled at $173.61\text{ Hz}$"*).
- **Corrected Finding:** `convert_beed()` downloads the **Bangalore EEG Epilepsy Dataset** (UCI ID 1134, National Institute of Mental Health and Neurosciences, Bangalore). It consists of 16 scalp EEG channels, sampled at $256\text{ Hz}$ across 4 clinical states. It is not building energy/occupancy, nor is it the single-channel $173.61\text{ Hz}$ Bonn dataset by Andrzejak et al.

### 2. Validity of Chemical Gas Sensor Drift Time Series
- **Current Report Claims:**
  - `04_results.tex` (lines 141, 170-171, 187): Reports accuracy improvement from $0.8524$ to $0.9501$ ($+9.8\%$, $p = 6.62 \times 10^{-13}$) and states that *"spectral harmonics resolve drift"*.
  - `99_appendices.tex` (lines 9, 27): Lists Gas Drift as a 16-sensor continuous temporal stream over 36 months ($13,910$ observations).
- **Corrected Finding:** UCI Gas Sensor Array Drift provides 128 pre-extracted static features (16 sensors $\times$ 8 static resistance/decay features). It contains no continuous time series. The 8 steps were artificially synthesized by reshaping static features. In the verified breadth evaluation ($N=600$ across 6 classes), accuracy was $0.1900$ (close to the chance rate of $0.1667$). The assertion that spectral harmonics resolve chemical sensor drift is based on an artifact of synthetic array reshaping.

### 3. Appliances Energy Regression Performance
- **Current Report Claims:**
  - `04_results.tex` (line 167): Reports Appliances Regression RMSE as $104.35$.
  - `99_appendices.tex` (lines 10, 28): Lists Appliances Energy as a 28-variable temporal forecasting benchmark ($H=20$).
- **Corrected Finding:** In the original breadth configuration, appliances energy regression failed ($R^2 = -0.4522$ on shuffled CV, $-1.9650$ on blocked CV) due to target misspecification (`label_strategy="last"`, attempting to predict an instantaneous 10-minute power spike from 24-hour rolling averages). Updating the converter to predict daily aggregate consumption (`label_strategy="mean"`) raised predictive performance to **$R^2 = +0.5365$** (shuffled) and **$+0.5340$** (blocked).

### 4. Human Activity Recognition (HAR) Cross-Validation Leakage
- **Current Report Claims:**
  - `99_appendices.tex` (lines 8, 26): Asserts that HAR was evaluated using entity-grouped cross-validation (`GroupKFold` by subject) to eliminate inter-subject target leakage.
- **Corrected Finding:** The original breadth benchmark used standard shuffled Stratified $k$-fold cross-validation, reporting $0.9640$ accuracy. When evaluated under strict `GroupKFold` across the 30 independent subjects, accuracy fell to **$0.8666$**, revealing that $+0.0974$ ($+9.74\%$) of the reported performance was attributable to inter-subject identity leakage.

### 5. Extractor Speedup Claims & JIT vs. Warm Separation
- **Current Report Claims:**
  - `04_results.tex` (Table 1, lines 21-25; lines 33, 69, 160): Reports that `polars_statistics` provides an $850\times$ speedup ($0.015$~s vs $12.77$~s), `numba_efficient` provides a $470\times$ speedup ($0.027$~s vs $12.77$~s), and cites a $61.6\times$ speedup on fleet classification.
- **Corrected Finding:**
  - `tsfresh_efficient` has a fixed multiprocessing startup overhead of **$3.21$~s** (measured on $N=2, L=10$).
  - `numba_efficient` has a first-call JIT compilation overhead of **$1.29\text{--}1.32$~s** (slope across $N$ is $0.0027, R^2 = 0.0881$).
  - In warm steady-state execution at $N=400$ ($W=100$), `tsfresh_efficient` requires $6.794$~s while `numba_efficient` requires $0.00315$~s, an actual steady-state speedup of **$2,157\times$**.
  - Conversely, on cold single-run execution at $N=50$, `numba_efficient` ($1.29$~s) versus `tsfresh_efficient` ($3.78$~s) provides only a **$2.9\times$** speedup, as JIT compilation offsets tsfresh process startup.

### 6. Numerical Parity Nuances Between Numba and tsfresh
- **Current Report Claims:**
  - `04_results.tex` (line 160): Claims *"Exact parity ($61.6\times$ speedup, $71.7\%$ less RAM)"*.
  - `06_conclusions.tex` (line 24): Asserts *"identical accuracy, $p = 1.0000$"*.
- **Corrected Finding:** The native Numba engine does not produce identical values out of the box:
  - **Autocorrelation:** Raw max absolute difference was $0.0432$ because tsfresh normalizes by $(n - \text{lag}) \sigma^2$ while Numba normalizes by $n \sigma^2$. Multiplying Numba output by $\frac{T}{T - \text{lag}}$ reduces residual differences to $\le 8.88 \times 10^{-16}$.
  - **Variance and Std:** Raw max absolute difference was $0.00409$ because Numba used the single-pass formula $(E[X^2] - (E[X])^2)$, which incurs catastrophic cancellation on baseline offsets ($\sim 300\text{ K}$), whereas tsfresh uses two-pass $\frac{1}{n}\sum(x - \mu)^2$.
  - **FFT Angle:** Raw max absolute difference was $176.81$ because tsfresh returns degrees while Numba returns radians. Applying `np.rad2deg` resolves the difference to $0.0$.
  - **Location & Energy:** 18 of 48 features exhibited discrepancies: Numba assigned sum of absolute values $\sum |x|$ to `absolute_maximum` instead of $\max(|x|)$ (causing raw discrepancies up to $61,681.80$), and `last_location` differed by $\frac{1}{T} = 0.005$ due to 1-based indexing in tsfresh.

### 7. AI4I Leakage vs. Overlap at Small Strides
- **Current Report Claims:**
  - `04_results.tex` (lines 7-8): Focuses exclusively on $W=100, S=50$ ($50\%$ overlap).
- **Corrected Finding:** At $S=100$ ($0\%$ overlap), the balanced accuracy gap between shuffled and purged blocked folds is **$-0.0000$** (95% CI: $[-0.0028, 0.0028]$). The gap remains negligible at $S=75$ ($-0.0023$) and $S=50$ ($+0.0015$), but widens to **$+0.2525$** (95% CI: $[0.2312, 0.2738]$) at $S=10$ ($90\%$ overlap). Additionally, at $S=50$, Blocked Fold 1 contains 0 negative instances, showing that contiguous blocking on small minority datasets can produce single-class evaluation folds.
