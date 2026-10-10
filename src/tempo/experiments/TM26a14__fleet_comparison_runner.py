#!/usr/bin/env python3
"""
TEMPO Fleet Comparison & Literature Alignment Runner (TM26a14).

Executes the comprehensive overnight bakeoff:
  Stage 1: Multi-Domain Classification (configs/TM26a14a__fleet_classification.yaml)
           - gas-sensor-drift-sub, drift-bifurcation-cls, pred-maintenance-w100-cls, beed
           - tsfresh_efficient vs numba_efficient across 6 selectors, 2 classifiers, 2 seeds (192 configs)
  Stage 2: Continuous Degradation & Dynamical Regression (configs/TM26a14b__fleet_regression.yaml)
           - pred-maintenance-w100-reg, drift-bifurcation-reg, appliances-energy, beijing-pm25
           - tsfresh_efficient vs numba_efficient across 6 selectors, 2 regressors, 2 seeds (192 configs)
  Stage 3: Multi-Horizon Autoregressive Forecasting H=10 (configs/TM26a14c__fleet_forecasting.yaml)
           - simulated-forecasting, appliances-energy, beijing-pm25
           - tsfresh_efficient vs numba_efficient vs polars_statistics across 4 selectors, 2 models, 2 seeds (144 configs)

Total Evaluated Workload: 528 configurations across 8 physical datasets.
Conforms to Contexere RAG index: TM26a14__fleet_comparison.
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
logger = logging.getLogger("tempo.fleet_comparison_runner")


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
    logger.info("TEMPO TM26a14 Fleet Comparison Orchestrator initiated at %s", start_time.isoformat())
    logger.info("=" * 75)

    stage1_cfg = "configs/TM26a14a__fleet_classification.yaml"
    stage2_cfg = "configs/TM26a14b__fleet_regression.yaml"
    stage3_cfg = "configs/TM26a14c__fleet_forecasting.yaml"

    try:
        run_stage(stage1_cfg, "Stage 1: Multi-Domain Classification (TM26a14a)")
    except Exception as e:
        logger.exception("CRITICAL FAILURE in Stage 1 (Classification): %s", e)
        sys.exit(1)

    try:
        run_stage(stage2_cfg, "Stage 2: Continuous & Dynamical Regression (TM26a14b)")
    except Exception as e:
        logger.exception("CRITICAL FAILURE in Stage 2 (Regression): %s", e)
        sys.exit(1)

    try:
        run_stage(stage3_cfg, "Stage 3: Multi-Horizon Autoregressive Forecasting (TM26a14c)")
    except Exception as e:
        logger.exception("CRITICAL FAILURE in Stage 3 (Forecasting): %s", e)
        sys.exit(1)

    total_elapsed = (datetime.datetime.now() - start_time).total_seconds()
    logger.info("=" * 75)
    logger.info(
        "ALL TM26a14 FLEET COMPARISON STAGES COMPLETED IN %.1f seconds (%.2f hours)",
        total_elapsed,
        total_elapsed / 3600.0,
    )
    logger.info("=" * 75)


if __name__ == "__main__":
    main()
