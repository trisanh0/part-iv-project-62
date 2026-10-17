# A/A Null Testing Calibration & Undefined t-Statistic Audit (Task 3)

**Date**: 2026-10-15 22:14:51 UTC  
**Dataset**: pred-maintenance (AI4I)  
**Evaluation Scope**: 1000 independent trials (5000 cross-validation fold evaluations)  
**Execution Duration**: 56.51 seconds  

---

## 1. Problem Statement & Mathematical Context

In algorithmic benchmarking, hypothesis testing engines must satisfy two fundamental properties:
1. **Conservative Protection Under Noise**: The test must not declare a statistically significant superiority claim purely as an artefact of random initialisation differences.
2. **Nominal Calibration**: Under continuous null conditions ($H_0$), the empirical false-positive rate (Type I error rate) must match the specified nominal significance threshold $\alpha$ (typically $5.0\%$ at $\alpha = 0.05$).

Earlier evaluations of TEMPO's statistical testing suite using discrete classification accuracy on AI4I yielded an empirical false-positive rate of **$0.00\%$**. This audit addresses how undefined $t$-values arise, how they are accounted for by the testing harness, and how continuous metrics resolve the discrete truncation boundary.

---

## 2. Part A: Discrete Classification Audit & Undefined t-Statistics

### Mechanism of Undefined t-Values
When two Random Forest models ($A$ and $B$) are trained on identical folds with differing random seeds, their prediction vectors on small test folds frequently coincide:
* In **665 out of 1000 trials (66.5\%)**, Model A and Model B produced **bit-for-bit identical predictions across all 199 samples**.
* The paired fold difference vector was $\Delta = [0, 0, 0, 0, 0]$.
* The sample standard deviation of differences was identically zero ($s_\Delta = 0$).
* The Student's $t$-statistic is therefore mathematically indeterminate:
  $$t = \frac{\bar{\Delta}}{s_\Delta / \sqrt{K}} = \frac{0}{0} \implies \text{NaN}$$

### Harness Handling in TEMPO
In `tempo.analysis.pairwise_ttests` and evaluation runners:
```python
# Undefined t-statistics (s_Delta == 0) are mapped to p = 1.0 (non-rejection of H0)
t_stat, p_val = stats.ttest_rel(scores_a, scores_b)
p_values.append(1.0 if np.isnan(p_val) else float(p_val))
```
* Rather than discarding indeterminate trials—which would bias the denominator—TEMPO maps $0/0 \to p = 1.0$.
* This treats identical performance as non-significant, preserving all 1000 trials in the denominator.

### The Discrete Truncation Boundary
In the remaining 335 trials where predictions differed:
* Differences are restricted to integer counts of misclassified samples out of $\approx 40$ samples per fold.
* Across all trials, the minimum valid observed $p$-value was **0.1778** (corresponding to a 2-sample divergence across folds).
* Observed discrete $p$-value levels: `[np.float64(0.1778), np.float64(0.1778), np.float64(0.1779), np.float64(0.1779), np.float64(0.208), np.float64(0.208), np.float64(0.21), np.float64(0.3739), np.float64(0.3739), np.float64(0.3739), np.float64(0.6953), np.float64(0.704), np.float64(0.988), np.float64(0.988), np.float64(0.988), np.float64(1.0), np.float64(1.0)]`.
* Because the support of the discrete $p$-value distribution is lower-bounded at $p \ge 0.1778$, **it is mathematically impossible to observe $p < 0.05$ under discrete classification accuracy**.
* Result: Empirical FPR = **0.0\%** at $\alpha = 0.05$.

---

## 3. Part B: Continuous Metric Calibration

When evaluated on a continuous metric (Brier score / quadratic scoring rule on posterior class probabilities $\hat{p}_i \in [0, 1]$) across the **exact same 1,000 seed pairs and folds**:
* Decision tree bootstrap subsampling and feature selection seed jitter induce continuous variations in probability estimates.
* The fold difference vector $\Delta_k \in \mathbb{R}$ has continuous support and non-zero variance ($s_\Delta > 0$).
* **Undefined $t$-statistics drop to zero (0 trials, 0.0\%)**.
* Empirical Rejections at $\alpha = 0.05$: **48 / 1000 (4.8\%)** (nominal expectation: 5.0%).
* Empirical Rejections at $\alpha = 0.01$: **8 / 1000 (0.8\%)** (nominal expectation: 1.0%).
* Kolmogorov-Smirnov Test vs Uniform(0, 1): $D = 0.038, p = 0.1086$ (confirming $p \sim \mathcal{U}(0, 1)$).

Synthetic continuous Gaussian benchmark reference ($H_0: \Delta \sim \mathcal{N}(0, \sigma^2)$):
* Empirical FPR at $\alpha = 0.05$: **4.3\%** (43/1,000).
* Empirical FPR at $\alpha = 0.01$: **0.9\%** (9/1,000).

---

## 4. Summary Table

| Metric Type | Evaluation Measure | Undefined $t$ Count ($0/0$) | Min $p$-Value | Empirical FPR ($\alpha=0.05$) | Empirical FPR ($\alpha=0.01$) | Calibration Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Discrete** | Classification Accuracy | 665 (66.5\%) | 0.1778 | **0.0\%** | **0.0\%** | Hyper-conservative (Discrete floor $p \ge 0.1778$) |
| **Continuous** | Brier Score (Probabilities) | **0 (0.0\%)** | 0.000217 | **4.8\%** | **0.8\%** | **Nominally Calibrated ($p \sim \mathcal{U}(0, 1)$)** |
| **Continuous** | Synthetic Gaussian Residuals | **0 (0.0\%)** | — | **4.3\%** | **0.9\%** | **Nominally Calibrated ($p \sim \mathcal{U}(0, 1)$)** |
