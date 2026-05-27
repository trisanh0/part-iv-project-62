# TEMPO: Time-series Evaluation for Model Performance Optimisation

TEMPO is a framework for evaluating, benchmarking, and cross-validating time-series machine learning pipelines. It searches the pipeline configuration space to find the optimal trade-off (Pareto frontier) between model accuracy and execution latency.

This project is developed as part of the University of Auckland Department of Engineering Science Part IV Research Project (Project 62).

---

## 1. Project Scope

TEMPO evaluates variations across time-series pipelines:
* **Data structures**: Comparing memory usage and processing speed across NumPy arrays, Polars DataFrames, and `fastpandas`.
* **Feature extraction**: Scaling feature extraction using `tsfresh`.
* **Feature selection**: Testing distribution-free statistical filters, benchmarking the baseline Mann-Whitney U test against the Cramér-von Mises goodness-of-fit statistic.
* **Model training**: Evaluating downstream machine learning classifiers and cross-validation loops.

The codebase is designed to scale from local testing to processing up to 1,000,000 sequences and features on cloud infrastructure (UoA Nectar VM).

---

## 2. Directory Layout

* `.agents/` - AI assistant configuration, rules, and session state.
* `data/` - Local data directory (ignored by git).
  * `data/01_raw/` - Raw source data and sensor logs.
  * `data/02_interim/` - Intermediate resampled or segmented sequences.
  * `data/03_processed/` - Processed feature matrices for training.
* `docs/` - Obsidian vault root containing administrative files and meeting minutes.
* `notebooks/` - Jupyter notebooks for exploratory analysis and sandbox experiments.
  * `notebooks/examples/` - Example notebooks and tutorials.
* `src/tempo/` - Core production package, installed as an editable Python package.
* `tests/` - Unit tests using `pytest`.

---

## 3. Getting Started

### Prerequisites
* Python 3.11
* GNU Make

### Installation
To create the virtual environment and install dependencies:
```bash
make create_environment
```

To activate the environment:
```bash
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

---

## 4. Local Data Management

The `data/` directory is not tracked by version control. To run pipelines locally:
1. Obtain the raw dataset.
2. Save it under `data/01_raw/`.
3. Reference paths in notebooks using `pathlib` relative to the project root:

```python
from pathlib import Path
data_path = Path.cwd().parent / "data" / "01_raw" / "beed" / "BEED_Data.csv"
```

---

## 5. Development Governance

Contributors must adhere to the coding guidelines in `.agents/rules/01-architecture.md` and maintain session states in `.agents/context.md`. Key policies include:
* No GNU (GPL/LGPL) licensed dependencies.
* Mandatory validation of sequence ordering during parallel processing.
* Simple, explicit, and well-documented Python code.

Before concluding a coding session, update the session state in section 5 of `.agents/context.md`.