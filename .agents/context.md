# Project Context and Codebase Master Reference

## 1. Document Roadmap and Context Locations
For deep-dive technical context, academic literature, or chronological tracking, look into the following specific files within the repository layout:
* For course regulations, grading criteria, and research compendium standards, read docs/Documents/Deliverable 1 - Scope, Objectives, Literature Review/P4P Handbook 2026 ESB.pdf.
* For the initial problem definition, baseline goals, and administrative boundaries, read docs/Documents/Deliverable 1 - Scope, Objectives, Literature Review/Project Brief.pdf.
* For the exhaustive scientific literature review, mathematical justifications, and early methodology outlines, read docs/Documents/Deliverable 1 - Scope, Objectives, Literature Review/Project Scope, Objectives, and Literature Review - Trisan Ethan Q Huynh.pdf.
* For the historical trajectory of engineering decisions and project milestones, review all markdown files sequentially inside docs/Minutes/.


## 2. Scientific Problem Domain and Scope
* Objective: Design, implement, and benchmark an automated evaluation framework that searches across a multi-dimensional pipeline configuration space to discover the optimal tradeoff frontier between model predictive performance and execution runtime.
* The Optimization Problem: Time-series pipelines involve multiple dependent phases, including window segmentation, feature extraction, statistical feature selection, and model training. Maximizing accuracy often introduces exponential computational latency and memory bloat. Conversely, minimizing execution time can result in underfit models that miss critical temporal signals. The framework must systematically evaluate these configurations to find the most efficient performance-to-time ratio.
* Pipeline Configuration Variables: The engine is designed to explore and evaluate variations across every phase of the pipeline. This includes comparing alternative data structures (such as NumPy arrays, Polars DataFrames, and Fastpandas), scaling feature extraction scopes, testing alternative feature selection filters (including the baseline Mann-Whitney U test and the Cramér-von Mises goodness-of-fit statistic), and testing diverse downstream machine learning architectures.
* Computational Scaling: The framework must operate under a scale-agnostic architecture. It is initially validated on small local datasets using a laptop, but the core engine must scale seamlessly to process up to 1,000,000 data points and 1,000,000 series when deployed on cloud computing infrastructure later in the project lifecycle.

## 3. Project Personnel and Roles
* Lead System Architect: Trisan. Responsible for repository architecture, data structure enforcement, infrastructure portability, rule governance, and implemnetaiton of the pipeline.
* Academic Research Partner: Scott. Core collaborator focused on pipeline exploration, initial model configuration, testing framework features, and notebook experimentation.
* Supervisor: Andreas. Big proponent of tsfresh, wants the project to centre around it.
* Technical Advisor: Ivan. Has experience replacing pandas with numpy within tsfresh. 

## 4. Deep-Dive Codebase and Directory Layout
* .agents/: The artificial intelligence configuration layer. Contains background instructions, permanent style and licensing safeguards, and targeted workflow scripts to ensure automated development matches the exact standards of the human architecture team.
* data/: Main data engine silo. This entire folder structure is strictly isolated by git configuration rules to prevent heavy production datasets from corrupting version control histories.
  - data/01_raw/: The pristine entry zone. Contains un-altered, original source sequence files, including local sensor logs (such as Phyphox accelerometer coordinates) and external benchmark data repositories.
  - data/02_interim/: The alignment zone. Reserved for intermediate, structured arrays that have undergone cleaning, uniform resampling, time-synchronization, or sliding window segmentation.
  - data/03_processed/: The execution zone. Holds serialized binary matrix extractions (such as .npy or .parquet arrays) representing the final computed feature vectors ready for machine learning model training loops.
* docs/: The academic knowledge management repository and active Obsidian vault root. Pointing the Obsidian desktop application directly at this folder surfaces the entire markdown logging system natively.
  - docs/Documents/: Core administrative repository housing pdf handbooks, report templates, lit reviews, and system onboarding materials.
  - docs/Minutes/: Chronological logging vault housing kickoff data, meeting agendas, and detailed review summaries from advisor sessions.
* notebooks/: The data science sandbox. Contains scratchpad notebooks used to run baseline data engineering, model training, and tsfresh package tests.
  - notebooks/examples/: Subfolder used for generic guidance of the ML process.
* src/tempo/: The formal production package environment. Mapped into the virtual environment using an editable python installation layout. It contains no execution code files yet, functioning as a clean blueprint ready for the systematic abstraction of the notebook pipelines.
* tests/: Dedicated validation directory. Built to house pytest modules mirroring the future mathematical and loading functions written inside src/tempo/.

## 5. Dynamic Session State and Agent Heartbeat
This section must be systematically updated by the agent at the conclusion of every single session to maintain state continuity across development boundaries.

* Last Active Session Date: 2026-06-30
* Current Completed Milestones:
  - Integrated classification sandbox scripts and notebooks (including benchmark execution, Polars vs. Pandas data ingestion comparison, and model training practices) into the workspace.
  - Refactored workspace structure by relocating all classification practice scripts and notebooks into the dedicated `src/classification-examples/` directory to preserve package namespace cleanliness.
  - Logged meeting minutes for Meeting 4 detailing optimization pathways, parallelisation, dataset standardisation, and Fourier coefficient scaling experiments.
  - Prepared draft emails for conference presentation inquiry and inter-semester break logistics.
* Active Working Constraints:
  - Dataset standardisation: the benchmark dataset must be standardised, documented, and frozen (focusing initially on balanced binary classification) prior to supervisor review.
  - Indexing integrity: parallel time-series processing must explicitly guarantee mathematical preservation of row indices to prevent positional alignment errors.
  - Dimension independence: core utility functions must not hardcode dimensions, maintaining scale-agnostic operation between local testing and cloud scaling (up to 1,000,000 data points).
  - Licensing limitations: no GNU GPL or LGPL third-party libraries or algorithms may be introduced.
* Next Steps:
  - Evaluate Ivan's approach of directly invoking SciPy and NumPy implementations to bypass Pandas translation overhead in feature extraction.
  - Execute systematic runtime and accuracy scaling benchmarks across varying time-series lengths and Fourier coefficient counts (comparing 50, 100, and 200 coefficients).
  - Explore Dask dataframes as a mechanism to scale parallelised time-series operations.
  - Freeze the primary binary classification dataset and review it with Andreas.
  - Adopt the Contexere framework template and structure within the workspace configuration.
  - Establish a feature subselection tool to evaluate model predictive accuracy against pipeline execution time.