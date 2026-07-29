30-07-2026

## Mid-Year Report and Presentation Figures
- Review draft of Deliverable 2 Mid-Year Report (`Mid-Year Report - tqhu735.md`).
- Discuss feedback on generated vector/raster visual assets (data architecture flow diagram, memory trade-off curves, Fourier coefficient subplots).

## Numba JIT Extraction and Performance Engineering
- Present custom Numba JIT-compiled extraction engine replicating `tsfresh` `EfficientFCParameters` with ~1,800x speedup.
- Review execution telemetry and peak memory profiling results.
- Modularized `tempo` library structure and initial unit test suite.

## NeSI Data Storage Architecture and Caching
- Discuss single-pass feature matrix caching schema for NeSI (~10,000 experiment combinations).
- Store 100% of feature extractions once, performing vertical/horizontal slicing during feature selection rather than re-extracting per run.
- Review standardized dataset splits ($X_{\text{train}}, X_{\text{test}}, y_{\text{train}}, y_{\text{test}}$ across random seeds).

## Time-Series Classification Bake-Off Methodology
- Review methodology alignment with Time-Series Classification Bake-Off (10 repeated CV runs per dataset, $\ge 100$ hyperparameter optimization iterations per fold).
- Design pipeline configuration interface for feature extraction, selection, and classifier combinations.

## Action Items
- 

## Unprocessed
