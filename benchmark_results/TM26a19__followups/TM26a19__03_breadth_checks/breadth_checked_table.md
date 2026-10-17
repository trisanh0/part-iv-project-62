# TEMPO Universal Dataset Coverage (Audited Matrix)

Notation: $p$ = raw input channels, $D$ = extracted features, $W$ = window length, $S$ = stride, $N$ = sample instances.

| Dataset Identifier | Physical Domain | Task | $N$ | $W$ | $S$ | $p$ (Channels) | $D$ (Features) | Validation Splitter | Metric | Benchmark Score | Quality Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| `beed` | Biomedical Telemetry (EEG) | classification | 157 | 200 | 200 | 16 | 80 | StratifiedKFold (5-fold) | Accuracy | **0.9681** | `VALID` |
| `pred-maintenance` | Industrial IoT & Failure | classification | 197 | 200 | 200 | 6 | 30 | StratifiedKFold (5-fold) | Accuracy | **0.9493** | `VALID` |
| `har` | Inertial Kinetics (IMU) | classification | 10,299 | 128 | 64 | 9 | 45 | GroupKFold by Subject (5-fold) | Accuracy | **0.8666** | `VALID` |
| `gas-sensor-drift-sub` | Chemical Sensing Array | classification | 600 | 8 | 8 | 16 | 80 | StratifiedKFold (5-fold) | Accuracy | **0.1900** | `INVALID_TS` |
| `appliances-energy` | Building Power & Thermodynamics | regression | 273 | 144 | 144 | 25 | 125 | KFold (3-fold) | R2 | **0.5365** | `VALID` |
| `beijing-pm25` | Environmental Meteorology | regression | 3,651 | 24 | 24 | 10 | 50 | KFold (3-fold) | R2 | **0.3876** | `VALID` |
| `drift-bifurcation-cls` | Stochastic Physics (Langevin) | classification | 200 | 500 | 500 | 3 | 15 | StratifiedKFold (3-fold) | Accuracy | **0.7451** | `VALID` |
