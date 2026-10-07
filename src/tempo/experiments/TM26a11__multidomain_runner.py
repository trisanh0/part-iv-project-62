#!/usr/bin/env python3
"""
TEMPO Multidomain Unified 5x6 Matrix Orchestrator (TM26a11).

Evaluates the complete 5 Extractor x 6 Selector x 2 Model x 2 Seed matrix across:
  Stage 1: Classification on NASA Turbofan AI4I Failure Modes (pred-maintenance-w100-cls)
  Stage 2: Continuous Remaining Useful Life (RUL) Degradation Regression (pred-maintenance-w100-reg)

Generated artifacts conform to Contexere index: TM26a11__multidomain_unified_matrix.
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
logger = logging.getLogger("tempo.multidomain_runner")


def run_stage(config_path: str, stage_name: str) -> None:
    logger.info("=" * 70)
    logger.info("STARTING STAGE: %s", stage_name)
    logger.info("Config Path: %s", config_path)
    logger.info("=" * 70)

    t0 = time.time()
    config = PipelineConfig.from_yaml(config_path)
    runner = BakeoffRunner(config)
    results_df = runner.run()
    elapsed = time.time() - t0

    logger.info(
        "STAGE COMPLETE: %s | Evaluated %d configurations in %.1f seconds (%.2f minutes)",
        stage_name,
        len(results_df),
        elapsed,
        elapsed / 60.0,
    )


def main() -> None:
    start_time = datetime.datetime.now()
    logger.info("TEMPO TM26a11 Orchestrator started at %s", start_time.isoformat())

    stage1_cfg = "configs/TM26a11a__multidomain_classification.yaml"
    stage2_cfg = "configs/TM26a11b__multidomain_regression.yaml"

    try:
        run_stage(stage1_cfg, "TM26a11a (Classification: pred-maintenance-w100-cls)")
    except Exception as e:
        logger.exception("CRITICAL FAILURE in Stage 1 (Classification): %s", e)
        sys.exit(1)

    try:
        run_stage(stage2_cfg, "TM26a11b (Regression: pred-maintenance-w100-reg)")
    except Exception as e:
        logger.exception("CRITICAL FAILURE in Stage 2 (Regression): %s", e)
        sys.exit(1)

    total_elapsed = (datetime.datetime.now() - start_time).total_seconds()
    logger.info("=" * 70)
    logger.info(
        "ALL TM26a11 STAGES COMPLETED IN %.1f seconds (%.2f minutes)",
        total_elapsed,
        total_elapsed / 60.0,
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
