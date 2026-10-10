#!/usr/bin/env python3
"""
TEMPO Fleet Bakeoff & Langevin Physical Interpretability Runner (TM26a15).

Executes the second-wave overnight bakeoff targeting an ~8:00 AM NZDT completion:
  Stage 1: Langevin Bifurcation Physical Parameter Recovery Regression (TM26a15a)
           - drift-bifurcation-reg-1k (N=1,000 empirical Langevin samples across tau in [3.4, 4.2])
           - numba_efficient vs tsfresh_efficient vs tsfel vs polars_statistics across 6 selectors, 2 regressors (48 configs)
           - Direct fulfillment of Meeting 9 Action Item 48 (Andreas Kempa-Liehr)

  Stage 2: Cross-Domain Multi-Library Classification Bakeoff (TM26a15b)
           - drift-bifurcation-cls, pred-maintenance-w100-cls, beed, gas-sensor-drift-sub
           - numba_efficient vs tsfresh_efficient vs tsfel vs polars_statistics across 6 selectors, 2 classifiers (192 configs)

  Stage 3: Continuous Energy Degradation & Power Demand Regression (TM26a15c)
           - appliances-energy
           - numba_efficient vs tsfresh_efficient vs tsfel vs polars_statistics across 5 selectors, 2 regressors (40 configs)

Total Evaluated Workload: 280 configurations across 6 physical datasets.
Conforms to Contexere RAG index: TM26a15__overnight_fleet_bakeoff.
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
logger = logging.getLogger("tempo.fleet_bakeoff_runner")


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
    logger.info("TEMPO TM26a15 Fleet Bakeoff Orchestrator initiated at %s", start_time.isoformat())
    logger.info("=" * 75)

    stage1_cfg = "configs/TM26a15a__langevin_recovery_reg.yaml"
    stage2_cfg = "configs/TM26a15b__multidomain_classification.yaml"
    stage3_cfg = "configs/TM26a15c__energy_regression.yaml"

    try:
        run_stage(stage1_cfg, "Stage 1: Langevin Physical Parameter Recovery (TM26a15a)")
    except Exception as e:
        logger.exception("CRITICAL FAILURE in Stage 1 (Langevin Regression): %s", e)
        sys.exit(1)

    try:
        run_stage(stage2_cfg, "Stage 2: Cross-Domain Multi-Library Classification (TM26a15b)")
    except Exception as e:
        logger.exception("CRITICAL FAILURE in Stage 2 (Classification): %s", e)
        sys.exit(1)

    try:
        run_stage(stage3_cfg, "Stage 3: Continuous Energy Demand Regression (TM26a15c)")
    except Exception as e:
        logger.exception("CRITICAL FAILURE in Stage 3 (Energy Regression): %s", e)
        sys.exit(1)

    total_elapsed = (datetime.datetime.now() - start_time).total_seconds()
    logger.info("=" * 75)
    logger.info(
        "ALL TM26a15 FLEET BAKEOFF STAGES COMPLETED IN %.1f seconds (%.2f hours)",
        total_elapsed,
        total_elapsed / 3600.0,
    )
    logger.info("=" * 75)


if __name__ == "__main__":
    main()
