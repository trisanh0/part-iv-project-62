"""Unit tests for the TEMPO Display Day Benchmark Results Query Engine."""

from __future__ import annotations

import os
import time
import pytest

from demo.query import ConfigMetrics, DemoQueryEngine, TrackInfo


@pytest.fixture(scope="module")
def engine() -> DemoQueryEngine:
    """Fixture providing initialized DemoQueryEngine."""
    db_path = "demo/data/demo_benchmarks.db"
    json_path = "demo/data/demo_catalog.json"
    assert os.path.exists(db_path), "demo_benchmarks.db must exist before running tests"
    assert os.path.exists(json_path), "demo_catalog.json must exist before running tests"
    return DemoQueryEngine(db_path=db_path, json_path=json_path)


def test_list_tracks(engine: DemoQueryEngine) -> None:
    """Verify listing of tracks, including flagship filtering."""
    all_tracks = engine.list_tracks(flagship_only=False)
    assert len(all_tracks) >= 8, f"Expected at least 8 tracks, got {len(all_tracks)}"

    flagship_tracks = engine.list_tracks(flagship_only=True)
    assert len(flagship_tracks) == 4, f"Expected exactly 4 flagship tracks, got {len(flagship_tracks)}"

    flagship_ids = {t.track_id for t in flagship_tracks}
    expected_ids = {"beed", "pred-maintenance-w100-cls", "gas-sensor-drift-sub", "drift-bifurcation-reg-1k"}
    assert flagship_ids == expected_ids, f"Mismatch in flagship track IDs: {flagship_ids}"


def test_track_metadata_fields(engine: DemoQueryEngine) -> None:
    """Verify metadata fields on curated tracks."""
    beed = engine.get_track("beed")
    assert beed is not None
    assert beed.track_id == "beed"
    assert beed.domain == "Bioacoustics"
    assert beed.task_type == "classification"
    assert beed.primary_metric == "accuracy"
    assert beed.is_curated_flagship is True


def test_get_valid_options(engine: DemoQueryEngine) -> None:
    """Verify available extractor, selector, and model options for flagship tracks."""
    opts = engine.get_valid_options("beed")
    assert "extractors" in opts and len(opts["extractors"]) >= 4
    assert "selectors" in opts and len(opts["selectors"]) >= 4
    assert "models" in opts and len(opts["models"]) >= 2

    assert "numba_efficient" in opts["extractors"]
    assert "tsfresh_efficient" in opts["extractors"]
    assert "None" in opts["selectors"]
    assert "random_forest" in opts["models"]


def test_get_config_metrics_lookup(engine: DemoQueryEngine) -> None:
    """Verify specific metric lookup and data integrity."""
    cfg = engine.get_config("beed", "numba_efficient", "None", "random_forest")
    assert cfg is not None
    assert isinstance(cfg, ConfigMetrics)
    assert cfg.track_id == "beed"
    assert cfg.extractor == "numba_efficient"
    assert cfg.selector == "None"
    assert cfg.model == "random_forest"

    # Physics and performance checks
    assert cfg.extraction_time_s > 0.0
    assert cfg.extraction_ram_mb > 0.0
    assert 0.8 <= cfg.score <= 1.0  # BEED accuracy should be high
    assert cfg.speedup_vs_tsfresh > 1.0  # Numba should be faster than tsfresh baseline

    # Bounds check
    assert cfg.extraction_time_min <= cfg.extraction_time_s <= cfg.extraction_time_max
    assert cfg.extraction_ram_min <= cfg.extraction_ram_mb <= cfg.extraction_ram_max
    assert cfg.score_min <= cfg.score <= cfg.score_max


def test_leaderboard_sorting(engine: DemoQueryEngine) -> None:
    """Verify leaderboard sorting behavior."""
    leaderboard_pareto = engine.get_leaderboard("beed", sort_by="pareto", limit=5)
    assert len(leaderboard_pareto) == 5
    for i in range(len(leaderboard_pareto) - 1):
        assert leaderboard_pareto[i].pareto_composite_score >= leaderboard_pareto[i + 1].pareto_composite_score

    leaderboard_speed = engine.get_leaderboard("beed", sort_by="speed", limit=5)
    assert len(leaderboard_speed) == 5
    for i in range(len(leaderboard_speed) - 1):
        assert leaderboard_speed[i].total_latency_s <= leaderboard_speed[i + 1].total_latency_s


def test_pareto_frontier(engine: DemoQueryEngine) -> None:
    """Verify Pareto frontier extraction."""
    pareto_configs = engine.get_pareto_frontier("beed")
    assert len(pareto_configs) >= 1
    for p in pareto_configs:
        assert p.is_pareto_optimal is True


def test_lookup_latency_performance(engine: DemoQueryEngine) -> None:
    """Verify that configuration lookups execute in sub-millisecond time."""
    n_iterations = 1000
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = engine.get_config("beed", "numba_efficient", "None", "random_forest")
    duration = time.perf_counter() - start
    avg_ms = (duration / n_iterations) * 1000.0

    # Ensure average lookup is well below 1 millisecond
    assert avg_ms < 1.0, f"Average lookup took {avg_ms:.4f} ms (expected < 1.0 ms)"


def test_custom_sql_query(engine: DemoQueryEngine) -> None:
    """Verify execution of arbitrary read-only SQL queries."""
    rows = engine.execute_custom_query("SELECT COUNT(*) AS total FROM raw_benchmark_runs")
    assert len(rows) == 1
    assert rows[0]["total"] >= 3000
