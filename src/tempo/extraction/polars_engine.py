"""Native Polars expression feature extraction engine."""

import polars as pl


def polars_statistical_extractor(
    df: pl.DataFrame,
    id_col: str = "id",
    value_cols: list[str] | None = None,
) -> pl.DataFrame:
    """Extract summary statistical features using native Polars aggregations.

    Args:
        df: Polars DataFrame in long format.
        id_col: Identifier column name.
        value_cols: List of numerical signal columns to aggregate.

    Returns:
        Polars DataFrame of aggregated feature metrics per entity ID.
    """
    if value_cols is None:
        value_cols = [c for c in df.columns if c not in (id_col, "time")]

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
