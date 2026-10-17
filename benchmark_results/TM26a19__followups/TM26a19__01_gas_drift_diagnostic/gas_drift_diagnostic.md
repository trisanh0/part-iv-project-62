# Task 1: Investigation of Gas Sensor Array Drift (`gas-sensor-drift-sub`)

## 1. Executive Diagnostic Finding
The dataset `gas-sensor-drift-sub` is **not a valid time series for TEMPO**. The 8 "timesteps" ($L=8$) are an artifact of reshaped tabular features, which explains why cross-validated accuracy ($0.1900$) is near chance level ($0.1667$).

---

## 2. Ingestion & Pre-Processing Code Audit
Inspection of `convert_gas_sensor_drift()` in `src/tempo/storage/dataset.py` reveals:
```python
# Lines 38-45 in src/tempo/storage/dataset.py
arr_features = np.array(all_features, dtype=np.float32)  # Shape: (N, 128)
seq_len = 8
n_channels = 16
tensor_gas = arr_features.reshape(n_samples, seq_len, n_channels)
```

The UCI Gas Sensor Array Drift dataset (Vergara et al., 2012) contains 16 distinct chemical sensors. For each sensor, the original authors pre-extracted **8 static scalar features**:
1. Steady-state resistance difference ($\Delta R$)
2. Normalized steady-state resistance difference ($\Delta R / R_0$)
3. Exponential moving average (EMA) rising feature ($\alpha = 0.1$)
4. EMA rising feature ($\alpha = 0.01$)
5. EMA rising feature ($\alpha = 0.001$)
6. EMA decaying feature ($\alpha = 0.1$)
7. EMA decaying feature ($\alpha = 0.01$)
8. EMA decaying feature ($\alpha = 0.001$)

Total features: $16 \times 8 = 128$ tabular scalar attributes.

The conversion function arbitrarily reshaped the 128-element feature vector into an array of shape $(N, 8, 16)$, treating the 8 distinct feature types as consecutive temporal steps ($t \in [0, \dots, 7]$).

---

## 3. Class Counts & Baseline Rates
* **Dataset Size**: $N = 600$ sample recordings across 10 temporal batches.
* **Class Count**: 6 classes (Ethanol, Ethylene, Ammonia, Acetaldehyde, Acetone, Toluene).
* **Class Distribution**:
  * Class 1: 100 samples (16.67%)
  * Class 2: 100 samples (16.67%)
  * Class 3: 100 samples (16.67%)
  * Class 4: 100 samples (16.67%)
  * Class 5: 100 samples (16.67%)
  * Class 6: 100 samples (16.67%)
* **Majority-Class Accuracy Rate**: **$16.67\%$** ($\frac{1}{6}$).
* **Breadth Run Score**: **$0.1900$** ($\approx$ random guessing / chance).

---

## 4. Evaluative Conclusion
Extracting rolling time-series features (e.g. mean, variance, linear slope, FFT) across 8 heterogeneous scalar features destroys their physical meaning. For example, computing a "linear trend" across steady-state resistance and three decaying exponential parameters is mathematically meaningless.

**Recommendation**: Exclude `gas-sensor-drift-sub` from benchmark suites evaluating temporal feature extraction, or replace it with true raw sensor stream datasets.
