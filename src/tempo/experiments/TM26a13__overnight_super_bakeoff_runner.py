#!/usr/bin/env python3
"""
TEMPO Andreas Scope & NeSI Overnight Super-Bakeoff Orchestrator (TM26a13).

Evaluates:
  Stage 1: Multi-Domain Classification Bakeoff (TM268Ua)
           - Datasets: beed, pred-maintenance, har
           - Extractors: numba_efficient (FFT 10, 25, 50), tsfresh_efficient (FFT 25),
                         tsfresh_minimal, polars_statistics
           - Selectors: None, select_k_best, fdr, subsampled (10%, 20%, 50%)
           - Models: random_forest, logistic_regression across Seeds [42, 123]
           - Total: 432 configurations

  Stage 2: Extrinsic Regression Bakeoff (andreas_medium_regression)
           - Datasets: appliances-energy, beijing-pm25
           - Extractors: numba_efficient (FFT 10, 25, 50), tsfresh_efficient (FFT 25),
                         tsfresh_minimal, polars_statistics, tsfel
           - Selectors: None, select_k_best, fdr, subsampled (10%, 20%)
           - Models: random_forest, ridge across Seed [42]
           - Total: 140 configurations

Total Workload: 572 evaluated configurations across 5 major datasets.
Estimated Runtime: 3.5 to 5.0 hours (finishes well before wake-up).
"""

import datetime
import logging
from pathlib import Path
import sys
import time

from tempo.benchmark import BakeoffRunner, PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tempo.overnight_super_bakeoff")


def run_stage(config_path: str, stage_name: str) -> None:
    logger.info("=" * 75)
    logger.info("STARTING STAGE: %s", stage_name)
    logger.info("Config Path: %s", config_path)
    logger.info("=" * 75)

    t0 = time.time()
    config = PipelineConfig.from_yaml(config_path)
    runner = BakeoffRunner(config)
    results_df = runner.run()
    elapsed = time.time() - t0

    logger.info(
        "STAGE COMPLETE: %s | Evaluated %d configurations in %.1f seconds (%.2f hours)",
        stage_name,
        len(results_df),
        elapsed,
        elapsed / 3600.0,
    )


def main() -> None:
    start_time = datetime.datetime.now()
    logger.info("=" * 75)
    logger.info("TEMPO Overnight Super-Bakeoff initiated at %s", start_time.isoformat())
    logger.info("=" * 75)

    stage1_cfg = "configs/TM268Ua__andreas_meeting_bakeoff.yaml"
    stage2_cfg = "configs/andreas_medium_regression.yaml"

    try:
        run_stage(stage1_cfg, "Stage 1: Multi-Domain Classification (TM268Ua)")
    except Exception as e:
        logger.exception("CRITICAL FAILURE in Stage 1 (Classification): %s", e)
        sys.exit(1)

    try:
        run_stage(stage2_cfg, "Stage 2: Extrinsic Regression (Andreas Scope)")
    except Exception as e:
        logger.exception("CRITICAL FAILURE in Stage 2 (Regression): %s", e)
        sys.exit(1)

    total_elapsed = (datetime.datetime.now() - start_time).total_seconds()
    logger.info("=" * 75)
    logger.info(
        "ALL OVERNIGHT SUPER-BAKEOFF STAGES COMPLETED IN %.1f seconds (%.2f hours)",
        total_elapsed,
        total_elapsed / 3600.0,
    )
    logger.info("=" * 75)


if __name__ == "__main__":
    main()
