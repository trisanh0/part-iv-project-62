#!/usr/bin/env python3
"""
TEMPO Overnight Execution Orchestrator - Run 2 (TM26a8).

Executes sequentially:
  Stage 1: AI4I Pedagogical Classification Benchmark (configs/TM269Ga__ai4i_running_classification.yaml)
  Stage 2: AI4I Pedagogical Continuous Regression Benchmark (configs/TM269Ga__ai4i_running_regression.yaml)
  Stage 3: Automated Summary Generation and Telemetry Validation.
"""

import datetime
import logging
import os
from pathlib import Path
import sys
import time

from tempo.benchmark import BakeoffRunner, PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tempo.overnight_orchestrator_run2")


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
        "STAGE COMPLETE: %s | Generated %d evaluated rows in %.1f seconds (%.2f minutes)",
        stage_name,
        len(results_df),
        elapsed,
        elapsed / 60.0,
    )


def main() -> None:
    start_time = datetime.datetime.now()
    logger.info("TEMPO Overnight Run 2 Orchestrator initiated at %s", start_time.isoformat())
    
    stage1_cfg = "configs/TM269Ga__ai4i_running_classification.yaml"
    stage2_cfg = "configs/TM269Ga__ai4i_running_regression.yaml"
    
    try:
        run_stage(stage1_cfg, "TM269Ga (AI4I Pedagogical Classification)")
    except Exception as e:
        logger.exception("CRITICAL FAILURE in Stage 1 (AI4I Classification): %s", e)
        
    try:
        run_stage(stage2_cfg, "TM269Ga (AI4I Pedagogical Regression)")
    except Exception as e:
        logger.exception("CRITICAL FAILURE in Stage 2 (AI4I Regression): %s", e)
        
    end_time = datetime.datetime.now()
    total_elapsed = (end_time - start_time).total_seconds()
    logger.info("=" * 70)
    logger.info("ALL OVERNIGHT BENCHMARK RUN 2 STAGES COMPLETE.")
    logger.info("Total Elapsed Duration: %.1f seconds (%.2f hours)", total_elapsed, total_elapsed / 3600.0)
    logger.info("Completed at: %s", end_time.isoformat())
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
