"""Contexere RAG export helper functions for figures, datasets, and metadata."""

import os
import datetime
from typing import Any, Optional, Union


def _format_rag_date_components(date_obj: datetime.date) -> tuple[str, str, str]:
    """Encode datetime into Contexere RAG date components (yy, m, D).

    Args:
        date_obj: Date instance to encode.

    Returns:
        Tuple containing 2-digit year string, 1-character month string,
        and 1-character day identifier string.
    """
    yy = date_obj.strftime("%y")

    month = date_obj.month
    if 1 <= month <= 9:
        m = str(month)
    elif month == 10:
        m = "a"
    elif month == 11:
        m = "b"
    elif month == 12:
        m = "c"
    else:
        m = str(month)

    day = date_obj.day
    if 1 <= day <= 9:
        d = str(day)
    elif 10 <= day <= 31:
        d = chr(ord("A") + day - 10)
    else:
        d = str(day)

    return yy, m, d


def generate_rag_filename(
    prefix: str,
    keyword: str,
    sequence: str = "a",
    extension: str = "",
    date: Optional[Union[datetime.date, datetime.datetime, str]] = None,
) -> str:
    """Generate a filename adhering to Contexere RAG standard naming convention.

    Format: `PIyymDc[_x]__keyword.ext`

    Args:
        prefix: Project or domain prefix identifier (e.g. 'P4P', 'TM', 'DS').
        keyword: Descriptive snake_case tag for the asset.
        sequence: Chunk or sequence version identifier (default 'a').
        extension: File extension (e.g. 'png', 'parquet', '.csv').
        date: Date object or string. Defaults to current date if None.

    Returns:
        Formated Contexere RAG filename string.
    """
    if date is None:
        date_obj = datetime.date.today()
        yy, m, d = _format_rag_date_components(date_obj)
        rag_tag = f"{yy}{m}{d}"
    elif isinstance(date, (datetime.date, datetime.datetime)):
        yy, m, d = _format_rag_date_components(date)
        rag_tag = f"{yy}{m}{d}"
    elif isinstance(date, str):
        if len(date) == 10 and date[4] == "-" and date[7] == "-":
            try:
                parsed_date = datetime.date.fromisoformat(date)
                yy, m, d = _format_rag_date_components(parsed_date)
                rag_tag = f"{yy}{m}{d}"
            except ValueError:
                rag_tag = date
        else:
            rag_tag = date
    else:
        date_obj = datetime.date.today()
        yy, m, d = _format_rag_date_components(date_obj)
        rag_tag = f"{yy}{m}{d}"

    filename = f"{prefix}{rag_tag}{sequence}__{keyword}"

    if extension:
        if not extension.startswith("."):
            extension = f".{extension}"
        filename += extension

    return filename


def save_figure(
    fig: Any,
    prefix: str,
    keyword: str,
    output_dir: str = ".",
    sequence: str = "a",
    extension: str = "png",
    date: Optional[Union[datetime.date, datetime.datetime, str]] = None,
    **kwargs: Any,
) -> str:
    """Export a Matplotlib or Plotly figure using Contexere RAG naming convention.

    Args:
        fig: Figure object (Matplotlib or Plotly).
        prefix: Asset prefix (e.g. 'TM', 'DS').
        keyword: Asset keyword description.
        output_dir: Destination directory path.
        sequence: Sequence tag.
        extension: File format extension.
        date: Optional date.
        **kwargs: Additional keyword arguments passed to save function.

    Returns:
        Destination file path string.
    """
    filename = generate_rag_filename(
        prefix=prefix,
        keyword=keyword,
        sequence=sequence,
        extension=extension,
        date=date,
    )
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    if hasattr(fig, "savefig"):
        fig.savefig(filepath, **kwargs)
    elif hasattr(fig, "write_image"):
        fig.write_image(filepath, **kwargs)
    else:
        raise AttributeError("Provided figure object does not support savefig or write_image methods.")

    return filepath


def save_dataframe(
    df: Any,
    prefix: str,
    keyword: str,
    output_dir: str = ".",
    sequence: str = "a",
    extension: str = "parquet",
    date: Optional[Union[datetime.date, datetime.datetime, str]] = None,
    **kwargs: Any,
) -> str:
    """Export a DataFrame (Polars or Pandas) using Contexere RAG naming convention.

    Args:
        df: DataFrame object (Polars or Pandas).
        prefix: Asset prefix (e.g. 'TM', 'DS').
        keyword: Asset keyword description.
        output_dir: Destination directory path.
        sequence: Sequence tag.
        extension: File format extension ('parquet', 'pq', or 'csv').
        date: Optional date.
        **kwargs: Additional arguments passed to export writer.

    Returns:
        Destination file path string.

    Raises:
        ValueError: If extension is not supported for DataFrame export.
        TypeError: If provided object is not a supported DataFrame type.
    """
    ext_clean = extension.lower().lstrip(".")
    if ext_clean not in ("parquet", "pq", "csv"):
        raise ValueError(
            f"Unsupported DataFrame export format/extension: '{extension}'. "
            "Supported formats are 'parquet', 'pq', and 'csv'."
        )

    has_polars_parquet = hasattr(df, "write_parquet")
    has_pandas_parquet = hasattr(df, "to_parquet")
    has_polars_csv = hasattr(df, "write_csv")
    has_pandas_csv = hasattr(df, "to_csv")

    if not (has_polars_parquet or has_pandas_parquet or has_polars_csv or has_pandas_csv):
        raise TypeError(
            f"Provided object of type '{type(df).__name__}' is not a supported DataFrame "
            "(must be a Polars or Pandas DataFrame)."
        )

    filename = generate_rag_filename(
        prefix=prefix,
        keyword=keyword,
        sequence=sequence,
        extension=extension,
        date=date,
    )
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    if ext_clean in ("parquet", "pq"):
        if has_polars_parquet:
            df.write_parquet(filepath, **kwargs)
        elif has_pandas_parquet:
            try:
                df.to_parquet(filepath, **kwargs)
            except (ImportError, ModuleNotFoundError):
                import polars as pl
                pl.DataFrame(df).write_parquet(filepath, **kwargs)
        else:
            raise TypeError("Provided DataFrame object does not support Parquet export.")
    elif ext_clean == "csv":
        if has_polars_csv:
            df.write_csv(filepath, **kwargs)
        elif has_pandas_csv:
            if "index" not in kwargs:
                kwargs["index"] = False
            df.to_csv(filepath, **kwargs)
        else:
            raise TypeError("Provided DataFrame object does not support CSV export.")

    return filepath
