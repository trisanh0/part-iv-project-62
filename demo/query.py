"""Read-only queries pinned to one validated catalogue release."""
from __future__ import annotations
import json
import hashlib
import math
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional
from demo.scoring import SCORING_VERSION, performance_order

SCHEMA_VERSION = 2
DATA_DIR = Path(__file__).resolve().parent / "data"

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
    speedup_vs_tsfresh: Optional[float]
    ram_reduction_pct: Optional[float]
    is_pareto_optimal: bool
    pareto_composite_score: float

    @property
    def key(self) -> str:
        """Formatted combination key."""
        return f"{self.extractor}|{self.selector}|{self.model}"


class DemoQueryEngine:
    def __init__(self, db_path=None, json_path=None, prefer_memory_cache=True, manifest_path=None):
        if db_path is None and json_path is None:
            manifest_file = Path(manifest_path or DATA_DIR / "current.json")
            manifest = json.loads(manifest_file.read_text())
            if not re.fullmatch(r"[a-f0-9]{20}", manifest["build_id"]):
                raise ValueError("Invalid catalogue build identifier")
            release = manifest_file.parent / "builds" / manifest["build_id"]
            db_path, json_path = release / "demo_benchmarks.db", release / "demo_catalog.json"
            expected = {"demo_benchmarks.db", "demo_catalog.json", manifest["audit_report"]}
            if set(manifest["sha256"]) != expected or any(Path(name).name != name for name in expected):
                raise ValueError("Invalid release artifact manifest")
            for name in sorted(expected):
                if hashlib.sha256((release / name).read_bytes()).hexdigest() != manifest["sha256"][name]:
                    raise ValueError("Catalogue checksum mismatch")
        if db_path is None or json_path is None:
            raise ValueError("Provide both database and JSON paths")
        self.db_path = str(db_path)
        self.json_path = str(json_path)
        catalogue = json.loads(Path(json_path).read_text())
        if catalogue.get("schema_version") != SCHEMA_VERSION or catalogue.get("scoring_version") != SCORING_VERSION:
            raise ValueError("Unsupported catalogue version; rebuild required")
        self.build_id = catalogue["build_id"]
        with self._get_connection() as con:
            metadata = dict(con.execute("SELECT key, value FROM metadata").fetchall())
            if metadata != {"build_id": self.build_id, "schema_version": str(SCHEMA_VERSION), "scoring_version": SCORING_VERSION}:
                raise ValueError("Catalogue/database version mismatch")
            db_configs = {r["key"]: json.loads(r["payload"]) for r in con.execute("SELECT key, payload FROM aggregated_configs")}
            if db_configs != catalogue["configs"]:
                raise ValueError("Catalogue/database configurations differ")
            db_tracks = {r["track_id"]: json.loads(r["payload"]) for r in con.execute("SELECT track_id,payload FROM tracks")}
            if db_tracks != catalogue["tracks"]:
                raise ValueError("Catalogue/database tracks differ")
        self._tracks = {key: TrackInfo(**data) for key, data in catalogue["tracks"].items()}
        self._configs = {key: ConfigMetrics(**data) for key, data in catalogue["configs"].items()}
        for key, c in self._configs.items():
            if key != f"{c.track_id}|{c.key}" or c.track_id not in self._tracks:
                raise ValueError("Invalid configuration identity")
            if c.sample_count < 1 or c.extraction_time_s <= 0 or c.extraction_ram_mb <= 0:
                raise ValueError("Invalid measurement counts or costs")
            if any(isinstance(v, float) and not math.isfinite(v) for v in asdict(c).values()):
                raise ValueError("Non-finite catalogue measurement")
        self.prefer_memory_cache = prefer_memory_cache

    @contextmanager
    def _get_connection(self):
        con = sqlite3.connect(Path(self.db_path).resolve().as_uri() + "?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            yield con
        finally:
            con.close()

    def list_tracks(self, flagship_only=False):
        return sorted((t for t in self._tracks.values() if not flagship_only or t.is_curated_flagship),
                      key=lambda t: (not t.is_curated_flagship, t.display_name))

    def get_track(self, track_id):
        return self._tracks.get(track_id)

    def get_all_configs(self, track_id):
        return sorted((c for c in self._configs.values() if c.track_id == track_id), key=lambda c: c.key)

    def get_valid_options(self, track_id):
        configs = self.get_all_configs(track_id)
        return {"extractors": sorted({c.extractor for c in configs}),
                "selectors": sorted({c.selector for c in configs}),
                "models": sorted({c.model for c in configs}),
                "configurations": [{"extractor": c.extractor, "selector": c.selector, "model": c.model} for c in configs]}

    def get_config(self, track_id, extractor, selector="None", model="random_forest"):
        if selector is None or str(selector).lower() in ("none", "null", ""):
            selector = "None"
        return self._configs.get(f"{track_id}|{extractor}|{selector}|{model}")

    def get_leaderboard(self, track_id, sort_by="pareto", limit=10):
        if sort_by not in ("pareto", "speed", "accuracy", "score", "ram"):
            raise ValueError("Unknown leaderboard sort")
        if not 1 <= limit <= 1000:
            raise ValueError("Limit must be between 1 and 1000")
        def order(c):
            value = {"pareto": -c.pareto_composite_score, "speed": c.total_latency_s,
                     "accuracy": -performance_order(c.task_type, c.score),
                     "score": -performance_order(c.task_type, c.score), "ram": c.extraction_ram_mb}[sort_by]
            return value, c.key
        return sorted(self.get_all_configs(track_id), key=order)[:limit]

    def get_pareto_frontier(self, track_id):
        return [c for c in self.get_all_configs(track_id) if c.is_pareto_optimal]

    def execute_custom_query(self, sql, params=()):
        with self._get_connection() as con:
            allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
            con.set_authorizer(lambda action, *_: sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY)
            return [dict(r) for r in con.execute(sql, params).fetchall()]
