"""Unit tests for TEMPO Contexere RAG export module."""

import os
import tempfile
import datetime
import unittest
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt

from tempo.export import (
    generate_rag_filename,
    save_figure,
    save_dataframe,
)


class TestExportModule(unittest.TestCase):
    """Test suite for RAG export helper functions."""

    def test_generate_rag_filename_default_date(self):
        """Verify RAG filename generation using default current date."""
        filename = generate_rag_filename(prefix="TM", keyword="test_experiment")
        self.assertTrue(filename.startswith("TM26"))
        self.assertIn("__test_experiment", filename)

    def test_generate_rag_filename_custom_date(self):
        """Verify RAG filename generation with explicit date and extension."""
        custom_date = datetime.date(2026, 9, 27)
        filename = generate_rag_filename(
            prefix="DS",
            keyword="benchmark_results",
            sequence="b",
            extension="parquet",
            date=custom_date,
        )
        self.assertEqual(filename, "DS269Rb__benchmark_results.parquet")

    def test_generate_rag_filename_iso_string_date(self):
        """Verify RAG filename generation using ISO format string date."""
        filename = generate_rag_filename(
            prefix="P4P",
            keyword="meeting_notes",
            sequence="a",
            extension="md",
            date="2026-10-15",
        )
        self.assertEqual(filename, "P4P26aFa__meeting_notes.md")

    def test_save_figure_matplotlib(self):
        """Verify exporting a Matplotlib figure using RAG naming."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])

        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = save_figure(
                fig=fig,
                prefix="TM",
                keyword="accuracy_curve",
                output_dir=tmp_dir,
                extension="png",
            )
            self.assertTrue(os.path.exists(filepath))
            self.assertTrue(os.path.basename(filepath).endswith("__accuracy_curve.png"))
        plt.close(fig)

    def test_save_dataframe_polars(self):
        """Verify exporting a Polars DataFrame using RAG naming."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = save_dataframe(
                df=df,
                prefix="TM",
                keyword="features",
                output_dir=tmp_dir,
                extension="parquet",
            )
            self.assertTrue(os.path.exists(filepath))
            loaded_df = pl.read_parquet(filepath)
            self.assertEqual(loaded_df.shape, (3, 2))

    def test_save_dataframe_pandas(self):
        """Verify exporting a Pandas DataFrame using RAG naming."""
        df = pd.DataFrame({"x": [10.0, 20.0], "y": [30.0, 40.0]})

        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = save_dataframe(
                df=df,
                prefix="DS",
                keyword="metrics",
                output_dir=tmp_dir,
                extension="csv",
            )
            self.assertTrue(os.path.exists(filepath))
            loaded_df = pd.read_csv(filepath)
            self.assertEqual(loaded_df.shape, (2, 2))

    def test_generate_rag_filename_date_boundaries(self):
        """Verify date encoding for months 11 ('b') and 12 ('c'), and day 31 ('V')."""
        date_nov = datetime.date(2026, 11, 15)
        filename_nov = generate_rag_filename(
            prefix="TM",
            keyword="nov_test",
            sequence="a",
            date=date_nov,
        )
        self.assertTrue(filename_nov.startswith("TM26bF"))

        date_dec = datetime.date(2026, 12, 15)
        filename_dec = generate_rag_filename(
            prefix="TM",
            keyword="dec_test",
            sequence="a",
            date=date_dec,
        )
        self.assertTrue(filename_dec.startswith("TM26cF"))

        date_dec31 = datetime.date(2026, 12, 31)
        filename_dec31 = generate_rag_filename(
            prefix="TM",
            keyword="boundary_day",
            sequence="a",
            date=date_dec31,
        )
        self.assertTrue(filename_dec31.startswith("TM26cV"))

    def test_save_dataframe_unsupported_extension(self):
        """Verify save_dataframe raises ValueError for unsupported extensions."""
        df = pl.DataFrame({"a": [1, 2]})
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                save_dataframe(df=df, prefix="TM", keyword="invalid", output_dir=tmp_dir, extension="json")
            with self.assertRaises(ValueError):
                save_dataframe(df=df, prefix="TM", keyword="invalid", output_dir=tmp_dir, extension=".txt")

    def test_save_dataframe_invalid_object_type(self):
        """Verify save_dataframe raises TypeError for invalid DataFrame object types."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(TypeError):
                save_dataframe(df="not_a_dataframe", prefix="TM", keyword="invalid", output_dir=tmp_dir, extension="parquet")
            with self.assertRaises(TypeError):
                save_dataframe(df=12345, prefix="TM", keyword="invalid", output_dir=tmp_dir, extension="csv")
            with self.assertRaises(TypeError):
                save_dataframe(df=[1, 2, 3], prefix="TM", keyword="invalid", output_dir=tmp_dir, extension="parquet")

    def test_save_dataframe_polars_csv(self):
        """Verify exporting a Polars DataFrame as CSV using RAG naming."""
        df = pl.DataFrame({"col1": [10, 20], "col2": ["foo", "bar"]})

        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = save_dataframe(
                df=df,
                prefix="TM",
                keyword="polars_csv_export",
                output_dir=tmp_dir,
                extension="csv",
            )
            self.assertTrue(os.path.exists(filepath))
            self.assertTrue(filepath.endswith(".csv"))
            loaded_df = pl.read_csv(filepath)
            self.assertEqual(loaded_df.shape, (2, 2))

    def test_save_dataframe_pandas_parquet(self):
        """Verify exporting a Pandas DataFrame as Parquet using RAG naming."""
        df = pd.DataFrame({"alpha": [1.5, 2.5], "beta": [3.5, 4.5]})

        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = save_dataframe(
                df=df,
                prefix="DS",
                keyword="pandas_parquet_export",
                output_dir=tmp_dir,
                extension="parquet",
            )
            self.assertTrue(os.path.exists(filepath))
            self.assertTrue(filepath.endswith(".parquet"))
            loaded_df = pl.read_parquet(filepath)
            self.assertEqual(loaded_df.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
