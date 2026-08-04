"""Native Polars expression feature extraction engine."""

from typing import List, Optional
import polars as pl


def polars_statistical_extractor(
    df: pl.DataFrame,
    id_col: str = "sequence_id",
    value_cols: Optional[List[str]] = None,
) -> pl.DataFrame:
    """Extract summary statistical features using native Polars aggregations.

    Args:
        df: Polars DataFrame in long format.
        id_col: Identifier column name.
        value_cols: List of numerical signal columns to aggregate.

    Returns:
        Polars DataFrame of aggregated feature metrics per entity ID.
    """
    if id_col not in df.columns:
        if "id" in df.columns:
            id_col = "id"
        else:
            raise KeyError(f"Identifier column '{id_col}' not found in DataFrame columns.")

    if value_cols is None:
        exclude_cols = {id_col, "step", "time"}
        value_cols = [c for c in df.columns if c not in exclude_cols]

    exprs = []
    for col in value_cols:
        exprs.extend([
            pl.col(col).mean().alias(f"{col}__mean"),
            pl.col(col).std().alias(f"{col}__std"),
            pl.col(col).min().alias(f"{col}__min"),
            pl.col(col).max().alias(f"{col}__max"),
            (pl.col(col) ** 2).sum().alias(f"{col}__energy"),
        ])

    return df.group_by(id_col).agg(exprs).sort(id_col)
