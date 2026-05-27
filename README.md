# TEMPO: Time-series Evaluation for Model Performance Optimisation

TEMPO (Time-series Evaluation for Model Performance Optimisation) is a high-performance computing framework engineered to systematically evaluate, benchmark, and cross-validate time-series machine learning pipelines. The framework treats the entire pipeline configuration space as an optimisation problem, mapping the Pareto frontier to find the ideal balance between model predictive accuracy and execution runtime latency.

The project is developed as part of the University of Auckland Department of Engineering Science Part IV Research Project number 62.

---

## 1. Project Architecture and Problem Scope

Time-series feature engineering configurations introduce major computational complexity and memory constraints at scale. TEMPO provides a scale-agnostic framework built to explore and evaluate variations across every phase of execution:
* Data structure comparisons evaluating memory overhead and processing speed across NumPy arrays, Polars DataFrames, and Fastpandas configurations.
* Feature extraction scaling using the tsfresh algorithm library.
* Alternative distribution-free statistical feature selection filters, benchmarking the baseline Mann-Whitney U test against the Cramér-von Mises goodness-of-fit statistic to isolate signal from high-dimensional noise.
* Downstream machine learning classification model evaluation and cross-validation execution loops.

The core package is architected to scale seamlessly from small local evaluation datasets to processing up to 1,000,000 sequences and features on remote cloud infrastructure (University of Auckland Nectar VM environment).

---

## 2. Codebase Directory Layout

The repository splits documentation, raw exploratory pipelines, input data files, and the production package layer:

* .agents/ - Contains internal system context records, background governance rules, and task execution runbooks used by AI coding assistants.
* data/ - Local data storage silos. This directory is strictly isolated from version control tracking via .gitignore rules to prevent large datasets from entering remote history.
  - data/01_raw/ - Unaltered source data files and local sensor logs.
  - data/02_interim/ - Resampled, aligned, or sliding-window segmented sequences.
  - data/03_processed/ - Compiled binary feature matrices ready for machine learning model training loops.
* docs/ - Academic knowledge repository and root folder for the project Obsidian vault. Contains administrative files under Documents/ and meeting tracking logs under Minutes/.
* notebooks/ - Laboratory scratchpad housing exploratory notebooks running initial data processing and feature engineering evaluations.
  - notebooks/examples/ - Isolated subdirectory reserved for library vendor documentation tutorials and supervisor-provided demonstration scripts.
* src/tempo/ - The production source package directory. Configured as an editable Python installation via pyproject.toml and mapped into the local environment.
* tests/ - Validation directory reserved for pytest suites mirroring the core package modules.

---

## 3. Getting Started

### Prerequisites
* Python 3.11
* GNU Make

### Environment Initialisation
The environment recipe is shared via configuration requirements, but individual virtual environments run locally on machine-specific binaries. Run the following command from the repository root to create your isolated virtual environment and compile the required dependency stack:

```bash
make create_environment
```

To activate the newly generated environment, run:

```bash
source .venv/bin/activate  # On macOS/Linux
.venv\Scripts\activate     # On Windows
```

---

## 4. Local Data Management Workflow

Because the data directory is entirely local and omitted from version control tracking, cloning the repository will provide only the empty directory scaffolding.

To run the exploratory pipelines:
1. Obtain the raw data archives from the designated team cloud storage configuration or network file link.
2. Manually copy the raw time-series source data files into your local data/01_raw/ directory.
3. Access files from notebooks by prepending relative root scaling links, or by constructing OS-agnostic paths using the pathlib module:

```python
from pathlib import Path
data_path = Path.cwd().parent / "data" / "01_raw" / "beed" / "BEED_Data.csv"
```

---

## 5. Development Governance and AI Continuity

This repository integrates an automated context management system for AI-collaborative engineering.

Before introducing code changes or initiating development loops, ensure your AI assistant processes .agents/context.md and .agents/rules/01-architecture.md. These files enforce explicit guardrails, including:
* Absolute prohibition of non-permissive GNU licenses (GPL, LGPL).
* Mandatory row-ordering preservation checks during parallel code execution blocks.
* Clean code styling constraints that forbid inline emojis and arbitrary Title Case.

At the conclusion of an active coding session, execute the defined heartbeat runbook configuration to update section 5 of the context file, ensuring state continuity is preserved for subsequent development cycles.