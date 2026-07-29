# TEMPO

**Time-series Evaluation for Model Performance Optimisation**

TEMPO is a benchmarking framework for evaluating time-series machine learning pipelines. It measures processing speed, memory usage, and predictive accuracy across data structures (NumPy, Polars, pandas), `tsfresh` feature extraction, statistical feature selection filters, and downstream classifiers.

Developed as part of the University of Auckland Department of Engineering Science Part IV Research Project (Project 62).

## Repository Structure

* `src/tempo/` — Core Python package.
* `notebooks/` — Experiment and analysis notebooks.
* `data/` — Local data directory (`01_raw`, `02_interim`, `03_processed`). Ignored by git.
* `docs/` — Project documentation and meeting minutes.
* `tests/` — Unit test suite.

## Quick Start

### Prerequisites
* Python 3.11
* GNU Make

### Installation
```bash
make create_environment
source .venv/bin/activate  # macOS/Linux
```

To run tests:
```bash
pytest
```

## Research Artefacts & Tracking

This repository follows the [Contexere](https://github.com/kempa-liehr/contexere) naming scheme (`PIyymDc[_x]__keyword`) for research notebooks, datasets, exported figures, and meeting minutes.

To inspect indexed research outputs:
```bash
make rag-summary
```