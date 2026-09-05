# Meeting Minutes: Project #62 — Time-Series ML Model Selection Framework
**Date:** September 2, 2026  
**Document ID:** `P4P2692a`  

---

## 1. Hardware Telemetry, Profiling & GPU Monitoring
- **Background Resource Monitoring:** Scott implemented background system telemetry using `psutil`, capturing CPU, RAM, and memory usage samples at fixed time steps throughout benchmark execution.
- **GPU Profiling on Apple Silicon:** `psutil` does not capture Apple Silicon / MPS or CUDA GPU utilisation natively. Dedicated platform-specific monitoring libraries (e.g., macOS `powermetrics`, `pynvml`, or Metal profiling hooks) need to be investigated.
- **FLOPs Computation:** Explore profiling libraries (such as `ptflops` or `thop` if applicable) to compute theoretical floating-point operations (FLOPs) across feature extraction routines.
- **Hardware Metadata Logging:** Ensure benchmark CSV outputs explicitly capture system host hardware specifications (CPU model, core counts, RAM capacity, OS, and GPU architecture) for reproducible benchmarking across local machines and cloud nodes (NeSI / NECTAR CoNZ).
- **Log-Scale Visualisation:** For runtimes and memory footprints with wide dynamic ranges, apply log-transformations to retain legibility across microsecond vs multi-minute operations.

---

## 2. Statistical Evaluation & Performance Visualisation
- **Critical Difference (CD) Diagrams:** Incorporate `scikit-posthocs` to generate critical difference diagrams (Nemenyi / Wilcoxon signed-rank tests) for ranking extractors and selectors across multiple benchmark datasets.
- **Dataset Sample Thresholds for CD Analysis:** Valid post-hoc critical difference tests require a minimum of $N \ge 10$ datasets, with $N \ge 30$ recommended for robust statistical power. Discuss dataset sample sizes in the methodology and discussion sections.
- **Bayesian Hypothesis Testing:** Integrate `Bambi` (or `PyMC`) to perform Bayesian $t$-tests, providing posterior probabilities for whether performance metric means differ significantly across pipelines, complementing classical null hypothesis significance testing.
- **Relative Performance Ratio Plots:** Generate performance ratio diagrams comparing extractor/selector combinations relative to baseline configurations.

---

## 3. Report Writing, Architecture Diagrams & Pedagogical Structure
- **Motivation & Literature Review:** Establish a clear research gap in the introduction and literature review: define existing TSML frameworks, identify why automated feature extraction pipelines fail at scale, and justify the necessity of TEMPO.
- **Architecture Overview & Component Labelling:** Include an architectural overview diagram with explicit alphanumeric labels on each stage and subsystem box (e.g., Box A, Box B) for unambiguous reference in the report body text.
- **Pedagogical Running Toy Problem:** Develop a lightweight, continuous toy problem running across all chapters to demonstrate ingestion, windowing, extraction, selection, and inference step-by-step.
- **Self-Contained Figure Captions:** Assume the reader cannot immediately parse visual plots. Every figure must have a descriptive 2–3 sentence caption explaining what is plotted, the operational context, and the primary empirical takeaway.
- **Writing Tone & Authenticity:** Maintain rigorous academic, human-written prose, avoiding artificial or generic AI filler phrasing.

---

## 4. Action Items

| Task | Owner(s) | Timeframe |
| :--- | :--- | :--- |
| Investigate Apple Silicon / Metal GPU telemetry integration | Scott | Next Sprint |
| Log host system hardware metadata (CPU model, RAM, OS) in benchmark runner | Trisan | Next Sprint |
| Implement `scikit-posthocs` Critical Difference (CD) diagrams in `tempo.analysis` | Trisan | Next Sprint |
| Prototype Bayesian $t$-test / probabilistic comparison module with `Bambi` | Scott & Trisan | Next Sprint |
| Expand benchmark suite to $\ge 10\text{--}30$ UCR/UEA datasets for CD plots | Trisan | Next Sprint |
| Add alphanumeric reference labels to `tempo_pipeline_flowchart` boxes | Trisan | Next Sprint |
| Draft running toy problem pipeline script and write initial report methodology sections | Scott & Trisan | Next Sprint |

---

## 5. Unprocessed

- Scott worked on the implementation of the memory monitoring throughout the run. Used psutils to define a monitor, and every given time step it takes a reading of CPU/GPU/memory usage, etc. over the run time. 
- For GPU utilisation, I would need a different library.
- Some of the stuff took much longer to run, so I log-transformed everything as needed.
- There may be a library that could compute the number of FLOPs.
- Look into libraries that could use the GPU on Apple Silicon.
- Right now, saving to a CSV with all information.
- In the CPU, specify what the system is being tested. 
- When describing the figure in the report, add labels to the boxes to be referred to.
- Plots can visualise the performance ratio of different algorithms.
- scikit-posthocs critical difference plots.
- Bayesian t-test – Bambi gives a probabilistic interpretation. Gives you probability that the means are different.
- These performance plotting should be difference.
- Move towards the documentation of what we have done.
- Literature review to talk about what exists, why this is needed, why the existsing frameworks arent needed for TSML. By end of intro, need a clear reason why we are doing this.
- Good to have one section overviewing the architecture.
- Can do a toy problem throughout, assessing the need of tempo throughout.
- One chapter doing another chapter to communicate a comparison of two things.
- Good to guide yourself through the process. 
- Avoid AI! Says the AI man.
- Start by describing an image. Assume that the reader cannot see the plots. Every plot should have a figure caption of 2-3 sentences such that if the person has a reason for why the figure is in the report. Use the caption to write the readers view and don’t be ambiguous.
- Jump back and forth keep writing.
- NECTAR CoNZ. 
- For posthocs, 30 would be good, 10 would be minimum. Can talk about it in the discussion. Can critically assess your work.
