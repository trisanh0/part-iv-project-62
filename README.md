# TEMPO

Time-series Evaluation for Model Performance Optimisation

TEMPO is a benchmarking framework for evaluating time-series machine learning pipelines. It benchmarks feature extraction (Polars statistical primitives, Numba JIT, tsfresh, tsfel, and tsfeatures), feature selection (variance threshold, mutual information, F-tests, L1 regularization, tree importance, Boruta, and subsampling), and downstream predictive models. Supported tasks include classification, continuous regression, extrinsic parameter estimation for Langevin dynamics, and direct multi-output forecasting with quantile prediction intervals.

Hardware telemetry tracks execution time, CPU load, and peak memory usage across each pipeline stage. Intermediate representations can be cached to a Parquet feature store to avoid redundant computation.

Developed as part of the University of Auckland Department of Engineering Science Part IV Research Project (Project 62).

## Repository structure

* `src/tempo/`: core Python package containing extraction, selection, storage, telemetry, benchmarking, and analysis modules.
* `configs/`: YAML and JSON pipeline configurations.
* `benchmark_results/`: curated benchmark logs, evaluation summaries, and generated plots.
* `data/`: local data directory (`01_raw`, `02_interim`, `03_processed`), ignored by git.
* `notebooks/`: experiment and exploratory analysis notebooks.
* `docs/`: project documentation and meeting minutes.
* `tests/`: unit test suite.

## Quick start

### Prerequisites
* Python 3.11 or 3.12
* GNU Make

### Installation
```bash
make create_environment
source .venv/bin/activate
```

### Running tests
```bash
pytest
```

### Running benchmarks
Run a quick dry-run verification:
```bash
python -m tempo.benchmark --dry-run
```

Run a configuration-driven bakeoff:
```bash
python -m tempo.benchmark --config configs/TM26a5a__overnight_classification.yaml
```

## Research artefacts and tracking

This repository follows the [Contexere](https://github.com/kempa-liehr/contexere) naming scheme (`PIyymDc[_x]__keyword`) for research notebooks, datasets, exported figures, and meeting minutes.

Inspect indexed research outputs:
```bash
make rag-summary
```