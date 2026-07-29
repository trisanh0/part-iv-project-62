"""TEMPO: Time-series Evaluation for Model Performance Optimisation.

A framework for evaluating, benchmarking, and cross-validating time-series
machine learning pipelines across feature extraction engines, selection filters,
and model architectures.
"""

__version__ = "0.1.0"

from tempo.export import (
    generate_rag_filename,
    save_figure,
    save_dataframe,
)

__all__ = [
    "generate_rag_filename",
    "save_figure",
    "save_dataframe",
]

