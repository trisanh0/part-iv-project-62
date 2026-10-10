"""TEMPO Display Day Benchmark Query Engine.

Provides a fast, strongly typed programmatic interface to query pre-computed
benchmark metrics, valid dropdown options, Pareto frontiers, and speedups
from the SQLite database (`demo/data/demo_benchmarks.db`) or runtime JSON catalogue
(`demo/data/demo_catalog.json`).
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class TrackInfo:
    """Metadata describing a benchmark challenge dataset / racing track."""

    track_id: str
    display_name: str
    domain: str
    task_type: str
    primary_metric: str
    is_curated_flagship: bool
    description: str
    baseline_extractor: str
    baseline_model: str


@dataclass(frozen=True)
class ConfigMetrics:
    """Pre-computed empirical metrics for a specific pipeline configuration."""

    track_id: str
    task_type: str
    extractor: str
    selector: str
    model: str
    sample_count: int
    extraction_time_s: float
    extraction_time_min: float
    extraction_time_max: float
    extraction_time_std: float
    extraction_ram_mb: float
    extraction_ram_min: float
    extraction_ram_max: float
    extraction_ram_std: float
    selection_time_s: float
    fit_time_s: float
    inference_latency_ms: float
    total_latency_s: float
    n_extracted_features: int
    n_selected_features: int
    score: float
    score_std: float
    score_min: float
    score_max: float
    speedup_vs_tsfresh: float
    ram_reduction_pct: float
    is_pareto_optimal: bool
    pareto_composite_score: float

    @property
    def key(self) -> str:
        """Formatted combination key."""
        return f"{self.extractor}|{self.selector}|{self.model}"


class DemoQueryEngine:
    """Query engine for the Display Day interactive demonstration."""

    def __init__(
        self,
        db_path: str = "demo/data/demo_benchmarks.db",
        json_path: str = "demo/data/demo_catalog.json",
        prefer_memory_cache: bool = True,
    ) -> None:
        """Initialise query engine with paths to SQLite DB and JSON catalogue.

        Args:
            db_path: Path to master SQLite database.
            json_path: Path to runtime JSON catalogue.
            prefer_memory_cache: If True, loads JSON catalogue into memory for O(1) lookups.
        """
        self.db_path = db_path
        self.json_path = json_path
        self.prefer_memory_cache = prefer_memory_cache
        self._catalog_cache: Optional[dict[str, Any]] = None

        if self.prefer_memory_cache and os.path.exists(self.json_path):
            with open(self.json_path, "r", encoding="utf-8") as f:
                self._catalog_cache = json.load(f)

    def _get_connection(self) -> sqlite3.Connection:
        """Open a read-only SQLite connection."""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found at '{self.db_path}'. Run demo/compile_db.py first.")
        con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        return con

    def list_tracks(self, flagship_only: bool = False) -> list[TrackInfo]:
        """List all available benchmark tracks.

        Args:
            flagship_only: If True, returns only the curated flagship championship tracks.
        """
        if self._catalog_cache is not None:
            tracks_dict = self._catalog_cache.get("tracks", {})
            results: list[TrackInfo] = []
            for t_id, data in tracks_dict.items():
                if flagship_only and not data.get("is_curated_flagship", False):
                    continue
                results.append(
                    TrackInfo(
                        track_id=t_id,
                        display_name=data.get("display_name", t_id),
                        domain=data.get("domain", "General"),
                        task_type=data.get("task_type", "classification"),
                        primary_metric=data.get("primary_metric", "accuracy"),
                        is_curated_flagship=data.get("is_curated_flagship", False),
                        description=data.get("description", ""),
                        baseline_extractor=data.get("baseline", {}).get("extractor", "tsfresh_efficient"),
                        baseline_model=data.get("baseline", {}).get("model", "random_forest"),
                    )
                )
            return sorted(results, key=lambda x: (not x.is_curated_flagship, x.display_name))

        with self._get_connection() as con:
            cur = con.cursor()
            query = "SELECT * FROM tracks"
            if flagship_only:
                query += " WHERE is_curated_flagship = 1"
            query += " ORDER BY is_curated_flagship DESC, display_name ASC"
            rows = cur.execute(query).fetchall()
            return [
                TrackInfo(
                    track_id=r["track_id"],
                    display_name=r["display_name"],
                    domain=r["domain"],
                    task_type=r["task_type"],
                    primary_metric=r["primary_metric"],
                    is_curated_flagship=bool(r["is_curated_flagship"]),
                    description=r["description"],
                    baseline_extractor=r["baseline_extractor"],
                    baseline_model=r["baseline_model"],
                )
                for r in rows
            ]

    def get_track(self, track_id: str) -> Optional[TrackInfo]:
        """Get metadata for a specific track by ID."""
        for t in self.list_tracks(flagship_only=False):
            if t.track_id == track_id:
                return t
        return None

    def get_valid_options(self, track_id: str) -> dict[str, list[str]]:
        """Return the available extractors, selectors, and models for a given track."""
        if self._catalog_cache is not None:
            track = self._catalog_cache.get("tracks", {}).get(track_id)
            if track is not None:
                return {
                    "extractors": track.get("available_extractors", []),
                    "selectors": track.get("available_selectors", []),
                    "models": track.get("available_models", []),
                }

        with self._get_connection() as con:
            cur = con.cursor()
            exts = [
                r[0]
                for r in cur.execute(
                    "SELECT DISTINCT extractor FROM aggregated_configs WHERE track_id = ? ORDER BY extractor",
                    (track_id,),
                ).fetchall()
            ]
            sels = [
                r[0]
                for r in cur.execute(
                    "SELECT DISTINCT selector FROM aggregated_configs WHERE track_id = ? ORDER BY selector",
                    (track_id,),
                ).fetchall()
            ]
            mods = [
                r[0]
                for r in cur.execute(
                    "SELECT DISTINCT model FROM aggregated_configs WHERE track_id = ? ORDER BY model",
                    (track_id,),
                ).fetchall()
            ]
            return {
                "extractors": exts,
                "selectors": sels,
                "models": mods,
            }

    def get_config(
        self,
        track_id: str,
        extractor: str,
        selector: str = "None",
        model: str = "random_forest",
    ) -> Optional[ConfigMetrics]:
        """Lookup pre-computed empirical metrics for a specific pipeline combination."""
        # Normalize selector string
        if selector is None or str(selector).lower() in ("none", "null", ""):
            selector = "None"

        # Check memory cache first
        if self._catalog_cache is not None:
            track_dict = self._catalog_cache.get("tracks", {}).get(track_id)
            if track_dict is not None:
                key = f"{extractor}|{selector}|{model}"
                cfg = track_dict.get("configs", {}).get(key)
                if cfg is not None:
                    time_bounds = cfg.get("extraction_time_bounds", [cfg["extraction_time_s"], cfg["extraction_time_s"]])
                    ram_bounds = cfg.get("extraction_ram_bounds", [cfg["extraction_ram_mb"], cfg["extraction_ram_mb"]])
                    score_bounds = cfg.get("score_bounds", [cfg["score"], cfg["score"]])
                    return ConfigMetrics(
                        track_id=track_id,
                        task_type=track_dict.get("task_type", "classification"),
                        extractor=extractor,
                        selector=selector,
                        model=model,
                        sample_count=cfg.get("sample_count", 1),
                        extraction_time_s=cfg["extraction_time_s"],
                        extraction_time_min=time_bounds[0],
                        extraction_time_max=time_bounds[1],
                        extraction_time_std=0.0,
                        extraction_ram_mb=cfg["extraction_ram_mb"],
                        extraction_ram_min=ram_bounds[0],
                        extraction_ram_max=ram_bounds[1],
                        extraction_ram_std=0.0,
                        selection_time_s=cfg.get("selection_time_s", 0.0),
                        fit_time_s=cfg.get("fit_time_s", 0.0),
                        inference_latency_ms=cfg.get("inference_latency_ms", 0.0),
                        total_latency_s=cfg.get("total_latency_s", cfg["extraction_time_s"]),
                        n_extracted_features=cfg.get("n_extracted", 0),
                        n_selected_features=cfg.get("n_selected", 0),
                        score=cfg["score"],
                        score_std=cfg.get("score_std", 0.0),
                        score_min=score_bounds[0],
                        score_max=score_bounds[1],
                        speedup_vs_tsfresh=cfg.get("speedup_vs_tsfresh", 1.0),
                        ram_reduction_pct=cfg.get("ram_reduction_pct", 0.0),
                        is_pareto_optimal=cfg.get("is_pareto_optimal", False),
                        pareto_composite_score=cfg.get("pareto_composite_score", 0.0),
                    )

        # Fallback to direct SQLite query
        with self._get_connection() as con:
            cur = con.cursor()
            query = """
            SELECT * FROM aggregated_configs
            WHERE track_id = ? AND extractor = ? AND selector = ? AND model = ?
            LIMIT 1
            """
            row = cur.execute(query, (track_id, extractor, selector, model)).fetchone()
            if row is None:
                return None
            return ConfigMetrics(
                track_id=row["track_id"],
                task_type=row["task_type"],
                extractor=row["extractor"],
                selector=row["selector"],
                model=row["model"],
                sample_count=row["sample_count"],
                extraction_time_s=row["extraction_time_mean"],
                extraction_time_min=row["extraction_time_min"],
                extraction_time_max=row["extraction_time_max"],
                extraction_time_std=row["extraction_time_std"],
                extraction_ram_mb=row["extraction_ram_mean"],
                extraction_ram_min=row["extraction_ram_min"],
                extraction_ram_max=row["extraction_ram_max"],
                extraction_ram_std=row["extraction_ram_std"],
                selection_time_s=row["selection_time_mean"],
                fit_time_s=row["fit_time_mean"],
                inference_latency_ms=row["inference_latency_mean"],
                total_latency_s=row["total_latency_mean"],
                n_extracted_features=row["n_extracted_mean"],
                n_selected_features=row["n_selected_mean"],
                score=row["score_mean"],
                score_std=row["score_std"],
                score_min=row["score_min"],
                score_max=row["score_max"],
                speedup_vs_tsfresh=row["speedup_vs_tsfresh"],
                ram_reduction_pct=row["ram_reduction_pct"],
                is_pareto_optimal=bool(row["is_pareto_optimal"]),
                pareto_composite_score=row["pareto_composite_score"],
            )

    def get_leaderboard(
        self,
        track_id: str,
        sort_by: str = "pareto",
        limit: int = 10,
    ) -> list[ConfigMetrics]:
        """Get sorted leaderboard for a given track.

        Args:
            track_id: Identifier of the track.
            sort_by: 'pareto' (default), 'speed', 'accuracy', or 'ram'.
            limit: Maximum rows to return.
        """
        order_col_map = {
            "pareto": "pareto_composite_score DESC",
            "speed": "total_latency_mean ASC",
            "accuracy": "score_mean DESC",
            "score": "score_mean DESC",
            "ram": "extraction_ram_mean ASC",
        }
        order_clause = order_col_map.get(sort_by, "pareto_composite_score DESC")

        with self._get_connection() as con:
            cur = con.cursor()
            query = f"""
            SELECT * FROM aggregated_configs
            WHERE track_id = ?
            ORDER BY {order_clause}
            LIMIT ?
            """
            rows = cur.execute(query, (track_id, limit)).fetchall()
            return [
                ConfigMetrics(
                    track_id=r["track_id"],
                    task_type=r["task_type"],
                    extractor=r["extractor"],
                    selector=r["selector"],
                    model=r["model"],
                    sample_count=r["sample_count"],
                    extraction_time_s=r["extraction_time_mean"],
                    extraction_time_min=r["extraction_time_min"],
                    extraction_time_max=r["extraction_time_max"],
                    extraction_time_std=r["extraction_time_std"],
                    extraction_ram_mb=r["extraction_ram_mean"],
                    extraction_ram_min=r["extraction_ram_min"],
                    extraction_ram_max=r["extraction_ram_max"],
                    extraction_ram_std=r["extraction_ram_std"],
                    selection_time_s=r["selection_time_mean"],
                    fit_time_s=r["fit_time_mean"],
                    inference_latency_ms=r["inference_latency_mean"],
                    total_latency_s=r["total_latency_mean"],
                    n_extracted_features=r["n_extracted_mean"],
                    n_selected_features=r["n_selected_mean"],
                    score=r["score_mean"],
                    score_std=r["score_std"],
                    score_min=r["score_min"],
                    score_max=r["score_max"],
                    speedup_vs_tsfresh=r["speedup_vs_tsfresh"],
                    ram_reduction_pct=r["ram_reduction_pct"],
                    is_pareto_optimal=bool(r["is_pareto_optimal"]),
                    pareto_composite_score=r["pareto_composite_score"],
                )
                for r in rows
            ]

    def get_pareto_frontier(self, track_id: str) -> list[ConfigMetrics]:
        """Fetch all configurations lying on the non-dominated Pareto frontier."""
        with self._get_connection() as con:
            cur = con.cursor()
            query = """
            SELECT * FROM aggregated_configs
            WHERE track_id = ? AND is_pareto_optimal = 1
            ORDER BY score_mean DESC
            """
            rows = cur.execute(query, (track_id,)).fetchall()
            return [
                ConfigMetrics(
                    track_id=r["track_id"],
                    task_type=r["task_type"],
                    extractor=r["extractor"],
                    selector=r["selector"],
                    model=r["model"],
                    sample_count=r["sample_count"],
                    extraction_time_s=r["extraction_time_mean"],
                    extraction_time_min=r["extraction_time_min"],
                    extraction_time_max=r["extraction_time_max"],
                    extraction_time_std=r["extraction_time_std"],
                    extraction_ram_mb=r["extraction_ram_mean"],
                    extraction_ram_min=r["extraction_ram_min"],
                    extraction_ram_max=r["extraction_ram_max"],
                    extraction_ram_std=r["extraction_ram_std"],
                    selection_time_s=r["selection_time_mean"],
                    fit_time_s=r["fit_time_mean"],
                    inference_latency_ms=r["inference_latency_mean"],
                    total_latency_s=r["total_latency_mean"],
                    n_extracted_features=r["n_extracted_mean"],
                    n_selected_features=r["n_selected_mean"],
                    score=r["score_mean"],
                    score_std=r["score_std"],
                    score_min=r["score_min"],
                    score_max=r["score_max"],
                    speedup_vs_tsfresh=r["speedup_vs_tsfresh"],
                    ram_reduction_pct=r["ram_reduction_pct"],
                    is_pareto_optimal=True,
                    pareto_composite_score=r["pareto_composite_score"],
                )
                for r in rows
            ]

    def execute_custom_query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        """Execute arbitrary read-only SQL queries for ad-hoc inspection."""
        with self._get_connection() as con:
            cur = con.cursor()
            rows = cur.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
