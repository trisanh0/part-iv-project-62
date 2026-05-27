# Meeting Agenda: 28 May 2026

## 1. Objectives
* Close out action items from the previous review session with Ivan.
* Present and discuss the new time-series feature extraction benchmarking framework and performance breakthrough.
* Align on NumPy vs. Polars integration ownership and scaling roadmap.

---

## 2. Agenda Items

### Part I: Review of Action Items (Meeting with Ivan)s
* **Master's Thesis**: Confirm receipt and key takeaways from Ivan's research thesis.
* **Baseline Pipelines & Conceptual Draft**: Status update on the conceptual pipeline sketch and comparison of the four baseline models.
* **Polars vs. NumPy Ownership**: Confirm progress boundaries (NumPy baseline benchmarked, Polars trial status).
s
### Part II: The NumPy Vectorization Breakthrough (New Results)
* **Mathematical Parity**: Show validation success. The NumPy implementations produce identical feature matrices (`np.allclose < 1e-5`) to standard `tsfresh`.
* **Execution Wall-Time Speedup**:
  * **NumPy Loop**: Bypassing Pandas DataFrame wrappers yields a **2x to 30x speedup** depending on scale.
  * **NumPy Vectorized (`sliding_window_view`)**: Bypassing Python loops entirely and using zero-copy sliding window views yields an **average 1000x to 1200x speedup** (reducing 23.1 seconds of execution to **19 milliseconds** at 10,000 rows).
* **Memory Footprint Telemetry**:
  * Eliminating duplicate Pandas row-rolling structures cuts peak memory footprints by **88% to 96%** (reducing peak RAM usage from **269 MB to 31 MB** at full scale).

### Part III: Next Steps and Scaling Roadmap
* **Polars Trial Integration**: Discuss how a Polars-based rolling extraction compares (Scott).
* **Scaling Boundaries**: Strategy for scaling test sizes beyond 10,000 rows to 100,000+ sequences on cloud VM infrastructure (UoA Nectar).
* **Production Packaging**: Plans for incorporating the high-performance sliding window modules into the `src/tempo` package layer.

---

## 3. Talking Points & Discussion Starters
1. *"Since the C-level vectorized NumPy approach produces mathematically identical outputs to tsfresh in a fraction of a millisecond, is there any remaining reason to retain standard tsfresh rolling routines in our core package?"*
2. *"How should we handle feature calculations that cannot be easily vectorized (e.g., highly iterative or bespoke calculators)? Should we fall back to the NumPy loop method or Polars?"*
3. *"With peak memory usage dropped by over 90%, how does this impact our cluster deployment resource calculations on UoA Nectar VM?"*
