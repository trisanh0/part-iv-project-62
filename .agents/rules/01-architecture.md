---
trigger: always_on
---

# Repository Architecture and Coding Governance

## 1. Licensing Constraints
* Do not introduce, import, or depend on any third-party tools, libraries, or algorithms utilizing a GNU license (GPL, LGPL).
* Stick exclusively to libraries with highly permissive licenses such as MIT, BSD, or Apache 2.0.

## 2. Telemetry and Caching
* For core pipeline loops and heavy feature extraction blocks, track execution time and peak memory consumption when evaluating pipeline changes.
* Favor caching intermediate pipeline data states to binary files (such as .npy or .parquet) where useful to prevent redundant computing overhead.

## 3. Concurrency and Data Integrity Check
* If implementing parallel processing or multi-threading across time-series sequences, explicitly verify that data row tracking and indexing are mathematically preserved.
* If positional indexing is required downstream, prioritize ordered mapping utilities (such as Python's multiprocessing.Pool.imap) to explicitly guarantee order conservation.

## 4. Scale-Agnostic Design
* Do not hardcode fixed dataset dimensions, sample lengths, or array shapes into core utility functions.
* Ensure all processing modules are parameterized to seamlessly handle scaling boundaries, allowing identical code to run smoothly on small local testing configurations or massive cloud cluster environments.

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
* Maintain direct, professional, academic-grade engineering commentary throughout the codebase.
* Do not use emojis inside code files, docstrings, or commit messages.
* Avoid excessive or arbitrary Title Case in descriptive text and inline explanations.
* Write explicit type hints and concise, meaningful docstrings explaining the underlying mathematics.
* Avoid redundant comments that simply state what a line of code is doing visually.