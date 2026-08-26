"""
Statistical Analysis CLI Entrypoint for the TEMPO Framework.

Executes multi-hypothesis testing (paired t-tests with Benjamini-Hochberg FDR)
and generates summary boxplots from benchmark CSV outputs.
"""

import argparse
import glob
import os
from pathlib import Path
import sys

from tempo.analysis import run_statistical_analysis


def main():
    parser = argparse.ArgumentParser(description="TEMPO Statistical Benchmark Analysis Engine")
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to input benchmark CSV file")
    parser.add_argument("--output", "-o", type=str, default="benchmark_results/analysis", help="Output directory")
    parser.add_argument("--task", "-t", type=str, choices=["classification", "regression"], default="classification")
    parser.add_argument("--no-ttests", action="store_true", help="Disable multi-hypothesis paired t-tests")
    parser.add_argument("--no-plots", action="store_true", help="Disable boxplot figure generation")

    args = parser.parse_args()

    input_file = args.input
    if input_file is None or not os.path.exists(input_file):
        # Auto-detect newest benchmark_results*.csv
        csv_candidates = sorted(glob.glob("benchmark_results/benchmark_results*.csv") + glob.glob("benchmark_results*.csv"))
        if csv_candidates:
            input_file = csv_candidates[-1]
            print(f"Auto-selected newest benchmark CSV: {input_file}")
        else:
            print("No benchmark results CSV found. Run a benchmark first via: python -m tempo.benchmark")
            sys.exit(1)

    print(f"Running TEMPO Statistical Analysis on: {input_file}")
    outputs = run_statistical_analysis(
        csv_path=input_file,
        output_dir=args.output,
        task_type=args.task,
        enable_ttests=not args.no_ttests,
        enable_plots=not args.no_plots,
    )
    print(f"Analysis complete. Results written to: {args.output}")


if __name__ == "__main__":
    main()
