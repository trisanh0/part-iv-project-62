# TEMPO Display Day: Benchmark Results Database & Query Subsystem

This directory contains the consolidated benchmark results database, pre-compiled runtime catalogue, ingestion pipeline, and query interface for the **TEMPO Display Day Interactive Demonstration**.

---

## 1. Directory Structure

```
demo/
├── data/
│   ├── demo_benchmarks.db    # Relational SQLite database with raw runs and aggregated views
│   └── demo_catalog.json     # Pre-compiled hierarchical JSON catalogue for O(1) in-memory lookups
├── compile_db.py             # Reproducible ingestion and harmonization compiler script
├── query.py                  # Strongly typed Python query API (DemoQueryEngine)
└── README.md                 # This documentation file
```

---

## 2. Ingestion & Compilation Pipeline

To recompile the SQLite database and JSON catalogue from canonical experiment CSVs in `benchmark_results/`:

```bash
python demo/compile_db.py
```

### Ingestion Rules
1. **Source Discovery**: Scans all `benchmark_results/**/benchmark_results.csv` files, strictly skipping obsolete or archived runs in `benchmark_results/archive/`.
2. **Harmonization**:
   - Selector strings are canonicalised: empty strings, `null`, and `NaN` map to `'None'`.
   - Aliases are normalised: `select_k_best` $\to$ `select_k_best_anova`, `subsampled` $\to$ `subsampled_fdr`.
   - Missing latency metrics (`fit_time_seconds`, `inference_latency_ms`) are backfilled into standard columns.
3. **Aggregation with Variance Bounds**:
   - Groups by `(track_id, task_type, extractor, selector, model)`.
   - Computes empirical `mean`, `std`, `min`, `max`, and `sample_count` across runs.
   - Calculates total pipeline latency:
     $$\text{Total Latency} = t_{\text{extraction}} + t_{\text{selection}} + t_{\text{fit}}$$
   - Calculates speedup multipliers and memory reductions relative to `tsfresh_efficient`:
     $$\text{Speedup} = \frac{t_{\text{tsfresh}}}{t_{\text{current}}}$$
     $$\text{RAM Reduction} = \left(1 - \frac{\text{RAM}_{\text{current}}}{\text{RAM}_{\text{tsfresh}}}\right) \times 100\%$$
   - Calculates composite Pareto scores:
     $$\text{Score} = 100 \cdot S_{\text{norm}} - 15 \cdot \log_{10}(\max(t_{\text{tot}}, 10^{-3})) - 5 \cdot \left(\frac{\text{RAM}_{\text{MB}}}{1000}\right)$$
   - Identifies non-dominated multi-objective Pareto configurations.

---

## 3. Database Schema (`demo_benchmarks.db`)

### `tracks`
Metadata for each dataset / racecourse challenge.

| Column | Type | Description |
| :--- | :--- | :--- |
| `track_id` | `TEXT PRIMARY KEY` | Canonical dataset identifier (e.g. `beed`, `pred-maintenance-w100-cls`). |
| `display_name` | `TEXT` | Human-readable title for presentation displays and posters. |
| `domain` | `TEXT` | Physical application domain (e.g. Bioacoustics, Industrial IoT). |
| `task_type` | `TEXT` | Task category: `classification`, `regression`, or `forecasting`. |
| `primary_metric` | `TEXT` | Primary evaluation metric (`accuracy`, `r2`, `tau_r2`). |
| `is_curated_flagship` | `INTEGER` | `1` for the 4 core championship tracks with dense combinations; `0` for extended tracks. |
| `description` | `TEXT` | Concise summary of the physical problem. |
| `baseline_extractor`| `TEXT` | Standard baseline extractor (typically `tsfresh_efficient`). |
| `baseline_model` | `TEXT` | Standard baseline classifier/regressor (`random_forest`). |

### `aggregated_configs`
Pre-calculated statistics for each unique `(track, extractor, selector, model)` combination.

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | `INTEGER PRIMARY KEY` | Auto-incrementing identifier. |
| `track_id` | `TEXT` | Foreign key referencing `tracks(track_id)`. |
| `task_type` | `TEXT` | Task category. |
| `extractor` | `TEXT` | Feature extraction framework. |
| `selector` | `TEXT` | Feature selection algorithm (`None`, `fdr`, `boruta`, etc.). |
| `model` | `TEXT` | Downstream estimator (`random_forest`, `logistic_regression`, `ridge`). |
| `sample_count` | `INTEGER` | Number of experimental seed repetitions recorded. |
| `extraction_time_mean`| `REAL` | Mean feature extraction duration in seconds. |
| `extraction_time_min` | `REAL` | Minimum extraction duration recorded across seeds. |
| `extraction_time_max` | `REAL` | Maximum extraction duration recorded across seeds. |
| `extraction_time_std` | `REAL` | Standard deviation of extraction duration. |
| `extraction_ram_mean` | `REAL` | Mean peak RAM consumption in MB. |
| `extraction_ram_min`  | `REAL` | Minimum peak RAM recorded across seeds. |
| `extraction_ram_max`  | `REAL` | Maximum peak RAM recorded across seeds. |
| `selection_time_mean` | `REAL` | Mean feature selection duration in seconds. |
| `fit_time_mean`       | `REAL` | Mean model training duration in seconds. |
| `inference_latency_mean` | `REAL` | Mean per-sample inference latency in milliseconds. |
| `total_latency_mean`  | `REAL` | Mean end-to-end pipeline latency in seconds. |
| `n_extracted_mean`    | `INTEGER` | Median feature count before selection. |
| `n_selected_mean`     | `INTEGER` | Median feature count after selection. |
| `score_mean`          | `REAL` | Mean validation accuracy (classification) or $R^2$ / $\tau\text{-}R^2$ (regression). |
| `score_std`           | `REAL` | Standard deviation of validation score across folds/seeds. |
| `score_min`           | `REAL` | Minimum score observed. |
| `score_max`           | `REAL` | Maximum score observed. |
| `speedup_vs_tsfresh`  | `REAL` | Speedup multiplier relative to `tsfresh_efficient` on the same track. |
| `ram_reduction_pct`   | `REAL` | Peak memory percentage reduction relative to `tsfresh_efficient`. |
| `is_pareto_optimal`   | `INTEGER` | `1` if the configuration belongs to the non-dominated Pareto frontier. |
| `pareto_composite_score` | `REAL` | Composite weighted utility score. |

### `raw_benchmark_runs`
Contains all individual raw rows (3,072 entries) from every canonical run CSV for verification and audit trails.

---

## 4. Runtime JSON Catalogue (`demo_catalog.json`)

The JSON catalogue provides an $O(1)$ key lookup tree structured as:
```json
{
  "version": "1.0.0",
  "generated_at_utc": "2026-10-11T...",
  "flagship_tracks": [
    "beed",
    "pred-maintenance-w100-cls",
    "gas-sensor-drift-sub",
    "drift-bifurcation-reg-1k"
  ],
  "tracks": {
    "beed": {
      "track_id": "beed",
      "display_name": "BEED Bioacoustics Grand Prix",
      "available_extractors": ["numba_efficient", "polars_statistics", ...],
      "available_selectors": ["None", "fdr", "select_k_best_anova", ...],
      "available_models": ["logistic_regression", "random_forest"],
      "pareto_frontier": [
        "numba_efficient|None|random_forest",
        ...
      ],
      "configs": {
        "numba_efficient|None|random_forest": {
          "extraction_time_s": 1.7168,
          "extraction_time_bounds": [1.69, 1.74],
          "extraction_ram_mb": 389.7,
          "score": 0.9879,
          "speedup_vs_tsfresh": 14.18,
          "ram_reduction_pct": 76.86,
          "is_pareto_optimal": true,
          "pareto_composite_score": 93.04
        }
      }
    }
  }
}
```

---

## 5. Python Query Interface Usage

```python
from demo.query import DemoQueryEngine

# 1. Initialise query engine (loads in-memory cache by default)
engine = DemoQueryEngine()

# 2. List curated championship tracks
tracks = engine.list_tracks(flagship_only=True)
for track in tracks:
    print(f"{track.track_id}: {track.display_name} ({track.domain})")

# 3. Retrieve valid dropdown options for a track
options = engine.get_valid_options("beed")
print("Available extractors:", options["extractors"])
print("Available selectors:", options["selectors"])
print("Available models:", options["models"])

# 4. Lookup a specific user selection
config = engine.get_config(
    track_id="beed",
    extractor="numba_efficient",
    selector="fdr",
    model="random_forest",
)
if config:
    print(f"Accuracy: {config.score:.4f} (bounds: [{config.score_min:.4f}, {config.score_max:.4f}])")
    print(f"Speedup vs tsfresh: {config.speedup_vs_tsfresh:.1f}x")
    print(f"RAM Reduction: {config.ram_reduction_pct:.1f}%")

# 5. Fetch Pareto frontier for post-race screen
pareto_points = engine.get_pareto_frontier("beed")
for p in pareto_points:
    print(f"{p.key}: Score={p.score:.3f}, Latency={p.total_latency_s:.3f}s, RAM={p.extraction_ram_mb:.1f}MB")

# 6. Execute direct custom SQL for examiner questions
rows = engine.execute_custom_query(
    "SELECT extractor, AVG(speedup_vs_tsfresh) FROM aggregated_configs WHERE is_curated_flagship=1 GROUP BY extractor"
)
```
