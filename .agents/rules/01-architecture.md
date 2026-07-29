---
trigger: always_on
---

# Repository Architecture and Coding Governance

## 1. Licensing Constraints
* Do not introduce, import, or depend on any third-party tools, libraries, or algorithms utilizing a GNU license (GPL, LGPL).
* Stick exclusively to libraries with highly permissive licenses such as MIT, BSD, or Apache 2.0.

## 2. Telemetry and Caching
* For core pipeline loops and heavy feature extraction blocks, track execution time and peak memory consumption when evaluating pipeline changes.
* Favour caching intermediate pipeline data states to binary files (such as .npy or .parquet) where useful to prevent redundant computing overhead.

## 3. Concurrency and Data Integrity Check
* If implementing parallel processing or multi-threading across time-series sequences, explicitly verify that data row tracking and indexing are mathematically preserved.
* If positional indexing is required downstream, prioritize ordered mapping utilities (such as Python's multiprocessing.Pool.imap) to explicitly guarantee order conservation.

## 4. Scale-Agnostic Design
* Do not hardcode fixed dataset dimensions, sample lengths, or array shapes into core utility functions.
* Ensure all processing modules are parameterised to seamlessly handle scaling boundaries, allowing identical code to run smoothly on small local testing configurations or massive cloud cluster environments.

## 5. Code Philosophy
* Write code that is simple, clear, and easy to interpret.
* Avoid overly dense one-liners, deeply nested list comprehensions, or clever language hacks purely to save lines of code.
* Favor explicit code structures over implicit behavior.
* If a function is complex, break it down into smaller, readable steps rather than packing it into a cryptic abstraction.

## 6. Pythonic Conventions
* Always use is None or is not None when checking for singletons or missing values; never use comparison operators like == None.
* Use built-in structures and libraries correctly, such as leveraging context managers (with statements) for file handles, using dict lookups cleanly, and choosing explicit loops over unreadable lambda transformations.
* Anticipate edge cases, such as NaN values returned during feature extraction, and handle them explicitly using clean, descriptive exceptions or logging rather than letting them silently break downstream execution.

## 7. Professional Code Documentation Standards
* Maintain direct, professional, academic engineering commentary throughout the codebase.
* Never use emojis inside code files, docstrings, or commit messages.
* Avoid excessive or arbitrary Title Case in descriptive text and inline explanations.
* Write explicit type hints and concise, meaningful docstrings explaining the underlying mathematics.
* Avoid redundant comments that simply state what a line of code is doing visually.

## 8. Contexere RAG Naming Conventions & Repository Governance
* All project documentation, meeting minutes, experiment notebooks, and reference dataset artifacts must comply with the Contexere RAG index format: `PIyymDc[_x]__keyword`.
* **Project Identifier (`PI`) Prefixes**:
  - `P4P`: Part IV Project administrative docs, agendas, and meeting minutes stored in `docs/Minutes/` (e.g. `P4P26s7a__meeting_with_ivan_minutes.md`).
  - `TM`: TEMPO project codebase, experiment notebooks, and benchmark scripts stored in `notebooks/` (e.g. `TM26sRa__beed_tsfresh.ipynb`).
  - `DS`: Data Science reference material, external literature, and foundational benchmark datasets in `notebooks/examples/` (e.g. `DS26sRa__lab1_scikit_learn.ipynb`).
* **Format Breakdown**:
  - `yy`: Two-digit year (e.g., `26` for 2026).
  - `m`: One-character month in hexadecimal or sequence code (`1`-`9`, `a`=October, `b`=November, `c`=December, or month code).
  - `D`: One-character day identifier (`1`-`9`, `A`-`V` for days 1 through 31).
  - `c`: One-character document sequence or chunk identifier (e.g. `a`, `b`).
  - `[_x]`: Optional sub-chunk or revision index suffix.
  - `__keyword`: Double-underscore separator followed by concise, descriptive `snake_case` keyword tags.
* **Commit Requirement**: Developers may use default file names during initial drafting; however, all files MUST be renamed into compliance before git commit and pull request merge.