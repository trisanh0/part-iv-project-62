# TEMPO Universal Dataset Coverage (Breadth Smoke Benchmark)

| Dataset Identifier | Physical Domain | Task | $N$ (Instances) | $L$ (Steps) | $D$ (Channels) | $P$ (Features) | Metric | CV Score | Total Runtime | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `beed` | Biomedical Telemetry (EEG) | classification | 157 | 200 | 16 | 80 | Accuracy | **0.9681** | 0.25s | `SUCCESS` |
| `pred-maintenance` | Industrial IoT & Failure | classification | 197 | 200 | 6 | 30 | Accuracy | **0.9493** | 0.14s | `SUCCESS` |
| `har` | Inertial Kinetics (IMU) | classification | 10,299 | 128 | 9 | 45 | Accuracy | **0.894** | 0.31s | `SUCCESS` |
| `gas-sensor-drift-sub` | Chemical Sensing Array | classification | 600 | 8 | 16 | 80 | Accuracy | **0.19** | 0.16s | `SUCCESS` |
| `appliances-energy` | Building Power & Thermodynamics | regression | 273 | 144 | 25 | 125 | R2 | **-0.4522** | 0.25s | `SUCCESS` |
| `beijing-pm25` | Environmental Meteorology | regression | 3,651 | 24 | 10 | 50 | R2 | **0.3876** | 0.29s | `SUCCESS` |
| `drift-bifurcation-cls` | Stochastic Physics (Langevin) | classification | 200 | 500 | 3 | 15 | Accuracy | **0.7451** | 0.17s | `SUCCESS` |
