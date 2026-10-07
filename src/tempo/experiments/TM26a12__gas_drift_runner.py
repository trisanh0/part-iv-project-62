#!/usr/bin/env python3
"""
TEMPO Gas Sensor Drift Unified Extractor & Selector Matrix Runner (TM26a12).

Evaluates the 4 Extractor x 6 Selector x 2 Model x 2 Seed matrix (96 configurations)
on real-world chemical sensor drift (gas-sensor-drift-sub).

Conforms to Contexere RAG index: TM26a12__gas_drift_unified_matrix.
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
logger = logging.getLogger("tempo.gas_drift_runner")


def main() -> None:
    start_time = datetime.datetime.now()
    logger.info("TEMPO TM26a12 Gas Sensor Drift Matrix Runner initiated at %s", start_time.isoformat())

    config_path = "configs/TM26a12__gas_drift_unified_matrix.yaml"
    t0 = time.time()
    config = PipelineConfig.from_yaml(config_path)
    runner = BakeoffRunner(config)
    results_df = runner.run()
    elapsed = time.time() - t0

    logger.info(
        "RUN COMPLETE: Evaluated %d configurations in %.1f seconds (%.2f minutes)",
        len(results_df),
        elapsed,
        elapsed / 60.0,
    )


if __name__ == "__main__":
    main()
