---
name: contexere
description: >
  Guide for applying Oliver Kempa-Liehr's Contexere RAG naming convention (PIyymDc[_x]__keyword)
  and using the `nxt` CLI and `tempo.export` helpers across the TEMPO codebase.
  Use when creating, renaming, or saving research artefacts (notebooks, datasets, figures, meeting minutes).
---

# Contexere Research Artefact Naming Skill

This skill defines the Contexere Research Artefact Group (RAG) naming conventions and workflows for the TEMPO project.

## 1. Naming Formula

All research artefacts (notebooks, data files, figures, meeting minutes) follow the formula:

`PIyymDc[_x]__keyword.ext`

### Component Breakdown
- **`PI`**: Project Identifier ($\ge 2$ uppercase letters).
  - `P4P`: Part IV Project administrative docs, meeting agendas, and minutes (`docs/Minutes/`).
  - `TM`: TEMPO pipeline codebase, experiment notebooks, and benchmark scripts (`notebooks/`, `src/tempo/`).
  - `DS`: Reference datasets and external literature datasets (`data/01_raw/`, `notebooks/examples/`).
  - `FE`: Processed feature extraction matrices (`data/03_processed/`).
- **`yy`**: 2-digit year (e.g. `26` for 2026).
- **`m`**: 1-character month identifier:
  - Jan–Sep: `1`–`9`
  - Oct: `a` | Nov: `b` | Dec: `c`
- **`D`**: 1-character day identifier:
  - Days 1–9: `1`–`9`
  - Days 10–31: `A`–`V` (`A`=10, `B`=11, `C`=12, ..., `S`=28, `T`=29, `U`=30, `V`=31).
- **`c`**: 1-character daily counter (`a`, `b`, `c`, ...) enumerating items created on that day.
- **`[_x]`**: Optional parent RAG reference suffix (e.g., `_uTb` or `_s7a`).
- **`__keyword`**: Double-underscore separator followed by concise `snake_case` tag.

---

## 2. CLI Tooling (`nxt`)

The `contexere` package installs the `nxt` executable CLI tool.

### Useful Commands
- **Summarize Repository RAGs**:
  ```bash
  make rag-summary  # or: nxt --summary
  ```
- **Suggest Next Available RAG ID**:
  ```bash
  nxt -g TM
  ```
- **Clone and Track Notebook**:
  ```bash
  nxt notebooks/TM26sRa__beed_tsfresh.ipynb --keywords feature_selection
  ```
- **Clone with Parent Reference**:
  ```bash
  nxt notebooks/TM26sRa__beed_tsfresh.ipynb --reference sRa --keywords pareto_plots
  ```

---

## 3. Python Export Helpers (`src/tempo/export.py`)

In Python scripts and notebooks, export figures and dataframes using TEMPO's Contexere export helpers:

```python
from tempo.export import save_figure, save_dataframe

# Save Matplotlib/Seaborn or Plotly figure with auto-generated RAG name
save_figure(fig, prefix="TM", keyword="pareto_frontier", output_dir="presentation_figures")

# Save Polars/Pandas DataFrame to parquet or csv with RAG name
save_dataframe(df, prefix="FE", keyword="tsfresh_features", output_dir="data/03_processed")
```
