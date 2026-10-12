"""TEMPO Display Day Booth Room Manager and Game State Coordinator.

Coordinates multiplayer room lifecycle, player registration, pipeline configuration,
race physics parameter calculation, and Pareto scoring without external game engines.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from demo.query import ConfigMetrics, DemoQueryEngine


class GamePhase(str, Enum):
    """Lifecycle phases for the demonstration booth room."""

    LOBBY = "LOBBY"
    BUILDING = "BUILDING"
    COUNTDOWN = "COUNTDOWN"
    RACING = "RACING"
    RESULTS = "RESULTS"


CAR_COLORS: list[str] = [
    "#FF4D4D",  # Neon Red
    "#00E5FF",  # Neon Cyan
    "#76FF03",  # Neon Lime
    "#FFD600",  # Neon Amber
    "#D500F9",  # Neon Magenta
    "#FF9100",  # Neon Orange
]


@dataclass
class PlayerSlot:
    """State and configuration for a participant connected to the booth."""

    client_id: str
    nickname: str
    avatar_color: str
    car_index: int = 0
    is_spectator: bool = False
    is_bot: bool = False
    is_ready: bool = False
    pipeline: dict[str, str] = field(default_factory=dict)
    metrics: Optional[ConfigMetrics] = None
    pareto_score: float = 0.0
    awards: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert player slot to serialization dictionary."""
        metrics_dict: Optional[dict[str, Any]] = None
        if self.metrics is not None:
            metrics_dict = {
                "track_id": self.metrics.track_id,
                "task_type": self.metrics.task_type,
                "extractor": self.metrics.extractor,
                "selector": self.metrics.selector,
                "model": self.metrics.model,
                "extraction_time_s": self.metrics.extraction_time_s,
                "extraction_ram_mb": self.metrics.extraction_ram_mb,
                "fit_time_s": self.metrics.fit_time_s,
                "total_latency_s": self.metrics.total_latency_s,
                "score": self.metrics.score,
                "speedup_vs_tsfresh": self.metrics.speedup_vs_tsfresh,
                "ram_reduction_pct": self.metrics.ram_reduction_pct,
                "is_pareto_optimal": self.metrics.is_pareto_optimal,
                "pareto_composite_score": self.metrics.pareto_composite_score,
            }

        return {
            "client_id": self.client_id,
            "nickname": self.nickname,
            "avatar_color": self.avatar_color,
            "car_index": self.car_index,
            "is_spectator": self.is_spectator,
            "is_bot": self.is_bot,
            "is_ready": self.is_ready,
            "pipeline": self.pipeline,
            "metrics": metrics_dict,
            "pareto_score": self.pareto_score,
            "awards": self.awards,
        }


class BoothRoomManager:
    """State manager for the unified poster display room."""

    def __init__(
        self,
        query_engine: DemoQueryEngine,
        initial_track_id: str = "beed",
        max_active_racers: int = 4,
        race_duration_seconds: float = 14.0,
        countdown_seconds: float = 3.0,
        admin_pin: str = "7000",
    ) -> None:
        """Initialize the booth room manager with empirical query engine.

        Args:
            query_engine: Initialized DemoQueryEngine instance.
            initial_track_id: Identifier for the default flagship benchmark track.
            max_active_racers: Maximum simultaneous racers before overflow into spectator queue.
            race_duration_seconds: Visual playback duration for a full heat.
            countdown_seconds: Visual countdown duration prior to race start.
            admin_pin: Passcode for administrative and presenter actions.
        """
        self.engine = query_engine
        self.current_track_id = initial_track_id
        self.max_active_racers = max_active_racers
        self.race_duration_seconds = race_duration_seconds
        self.countdown_seconds = countdown_seconds
        self.admin_pin = admin_pin

        self.phase: GamePhase = GamePhase.LOBBY
        self.players: dict[str, PlayerSlot] = {}
        self.phase_start_time_utc: float = time.time()
        self.race_start_timestamp_ms: int = 0
        self.race_results_cache: Optional[dict[str, Any]] = None
        self.bot_counter: int = 0

        # Weights for composite Pareto scoring: accuracy * 100 - log(latency) * 15 - (RAM / 1000) * 5
        self.pareto_weights = {
            "accuracy_weight": 100.0,
            "latency_log_weight": 15.0,
            "ram_scale_weight": 5.0,
        }

    def _allocate_car_index(self) -> int:
        """Find the lowest available car slot index for an active racer."""
        used_indices = {p.car_index for p in self.players.values() if not p.is_spectator}
        for idx in range(self.max_active_racers):
            if idx not in used_indices:
                return idx
        return len(used_indices)

    def add_player(
        self,
        client_id: str,
        nickname: str,
        avatar_color: Optional[str] = None,
    ) -> PlayerSlot:
        """Register a new connection into the booth room.

        Args:
            client_id: Unique WebSocket connection or session ID.
            nickname: Human-readable moniker chosen by visitor.
            avatar_color: Hex color string. If omitted, assigned from standard palette.
        """
        active_count = sum(1 for p in self.players.values() if not p.is_spectator)
        is_spectator = (active_count >= self.max_active_racers) or (self.phase in (GamePhase.COUNTDOWN, GamePhase.RACING))

        car_idx = 0 if is_spectator else self._allocate_car_index()
        color = avatar_color if avatar_color is not None else CAR_COLORS[car_idx % len(CAR_COLORS)]

        default_pipeline = self._get_default_pipeline()
        default_metrics = self.engine.get_config(
            track_id=self.current_track_id,
            extractor=default_pipeline["extractor"],
            selector=default_pipeline["selector"],
            model=default_pipeline["model"],
        )

        slot = PlayerSlot(
            client_id=client_id,
            nickname=nickname.strip() or f"Racer {car_idx + 1}",
            avatar_color=color,
            car_index=car_idx,
            is_spectator=is_spectator,
            is_bot=False,
            is_ready=False,
            pipeline=default_pipeline,
            metrics=default_metrics,
        )
        self.players[client_id] = slot
        return slot

    def remove_player(self, client_id: str) -> Optional[PlayerSlot]:
        """Remove a player connection and re-index spectator queue if in lobby."""
        slot = self.players.pop(client_id, None)
        if slot is not None and not slot.is_spectator and self.phase == GamePhase.LOBBY:
            self._promote_spectators()
        return slot

    def _promote_spectators(self) -> None:
        """Promote queued spectators into active racer slots."""
        active_count = sum(1 for p in self.players.values() if not p.is_spectator)
        if active_count >= self.max_active_racers:
            return

        for p in self.players.values():
            if p.is_spectator and not p.is_bot:
                p.is_spectator = False
                p.car_index = self._allocate_car_index()
                p.avatar_color = CAR_COLORS[p.car_index % len(CAR_COLORS)]
                active_count += 1
                if active_count >= self.max_active_racers:
                    break

    def _get_default_pipeline(self) -> dict[str, str]:
        """Return canonical default pipeline configuration for the current track."""
        track = self.engine.get_track(self.current_track_id)
        baseline_extractor = track.baseline_extractor if track is not None else "tsfresh_efficient"
        baseline_model = track.baseline_model if track is not None else "random_forest"
        options = self.engine.get_valid_options(self.current_track_id)

        extractor = "numba_efficient" if "numba_efficient" in options["extractors"] else baseline_extractor
        selector = "None" if "None" in options["selectors"] else options["selectors"][0]
        model = baseline_model if baseline_model in options["models"] else options["models"][0]

        return {"extractor": extractor, "selector": selector, "model": model}

    def update_pipeline(
        self,
        client_id: str,
        extractor: str,
        selector: str,
        model: str,
    ) -> Optional[PlayerSlot]:
        """Update a participant's pipeline configuration and resolve pre-computed metrics.

        Args:
            client_id: Unique participant ID.
            extractor: Feature extraction algorithm name.
            selector: Feature selection strategy name.
            model: Downstream predictive estimator name.
        """
        player = self.players.get(client_id)
        if player is None:
            return None

        # Normalize selector
        normalized_selector = "None" if str(selector).lower() in ("none", "null", "") else selector

        metrics = self.engine.get_config(
            track_id=self.current_track_id,
            extractor=extractor,
            selector=normalized_selector,
            model=model,
        )

        player.pipeline = {
            "extractor": extractor,
            "selector": normalized_selector,
            "model": model,
        }
        player.metrics = metrics
        return player

    def set_player_ready(self, client_id: str, is_ready: bool = True) -> bool:
        """Mark a participant as ready to race.

        Returns True if all active racers are ready and the room transitions to COUNTDOWN.
        """
        player = self.players.get(client_id)
        if player is None or player.is_spectator:
            return False

        player.is_ready = is_ready

        # Check if all active participants are ready
        active_players = [p for p in self.players.values() if not p.is_spectator]
        if active_players and all(p.is_ready for p in active_players):
            if self.phase in (GamePhase.LOBBY, GamePhase.BUILDING):
                self.start_countdown()
                return True
        return False

    def add_bot(
        self,
        nickname: Optional[str] = None,
        extractor: str = "tsfresh_efficient",
        selector: str = "None",
        model: str = "random_forest",
    ) -> Optional[PlayerSlot]:
        """Inject a simulated bot with empirical metrics into an available racer slot."""
        active_count = sum(1 for p in self.players.values() if not p.is_spectator)
        if active_count >= self.max_active_racers:
            return None

        self.bot_counter += 1
        client_id = f"bot_{self.bot_counter}_{int(time.time())}"
        car_idx = self._allocate_car_index()
        bot_name = nickname or f"Benchmark: {extractor}"

        metrics = self.engine.get_config(
            track_id=self.current_track_id,
            extractor=extractor,
            selector=selector,
            model=model,
        )

        bot_slot = PlayerSlot(
            client_id=client_id,
            nickname=bot_name,
            avatar_color=CAR_COLORS[car_idx % len(CAR_COLORS)],
            car_index=car_idx,
            is_spectator=False,
            is_bot=True,
            is_ready=True,
            pipeline={"extractor": extractor, "selector": selector, "model": model},
            metrics=metrics,
        )
        self.players[client_id] = bot_slot
        return bot_slot

    def change_track(self, track_id: str) -> bool:
        """Change the active competition track and update cached metrics for all connected players."""
        track = self.engine.get_track(track_id)
        if track is None:
            return False

        self.current_track_id = track_id
        options = self.engine.get_valid_options(track_id)

        # Refresh metrics for existing participants
        for p in self.players.values():
            ext = p.pipeline.get("extractor", options["extractors"][0])
            sel = p.pipeline.get("selector", "None")
            mod = p.pipeline.get("model", options["models"][0])

            if ext not in options["extractors"]:
                ext = options["extractors"][0]
            if sel not in options["selectors"]:
                sel = "None" if "None" in options["selectors"] else options["selectors"][0]
            if mod not in options["models"]:
                mod = options["models"][0]

            p.pipeline = {"extractor": ext, "selector": sel, "model": mod}
            p.metrics = self.engine.get_config(track_id, ext, sel, mod)
            p.is_ready = p.is_bot

        self.phase = GamePhase.LOBBY
        self.race_results_cache = None
        return True

    def start_building(self) -> None:
        """Transition room from LOBBY to BUILDING phase."""
        self.phase = GamePhase.BUILDING
        self.phase_start_time_utc = time.time()
        for p in self.players.values():
            p.is_ready = p.is_bot

    def start_countdown(self) -> None:
        """Transition room to COUNTDOWN phase and schedule race start."""
        self.phase = GamePhase.COUNTDOWN
        self.phase_start_time_utc = time.time()
        self.race_start_timestamp_ms = int((self.phase_start_time_utc + self.countdown_seconds) * 1000)

    def start_racing(self) -> None:
        """Transition room to active RACING phase."""
        self.phase = GamePhase.RACING
        self.phase_start_time_utc = time.time()

    def calculate_pareto_score(self, metrics: ConfigMetrics) -> float:
        """Compute composite utility score from empirical accuracy, latency, and memory metrics.

        Score = w_acc * accuracy - w_lat * ln(max(total_latency_s, 0.001)) - w_ram * (RAM_MB / 1000)
        """
        w_acc = self.pareto_weights["accuracy_weight"]
        w_lat = self.pareto_weights["latency_log_weight"]
        w_ram = self.pareto_weights["ram_scale_weight"]

        lat = max(metrics.total_latency_s, 0.001)
        score_component = w_acc * metrics.score
        latency_penalty = w_lat * math.log(lat)
        ram_penalty = w_ram * (metrics.extraction_ram_mb / 1000.0)

        return float(score_component - latency_penalty - ram_penalty)

    def finish_race(self) -> dict[str, Any]:
        """Compute post-race leaderboards, assign champion badges, and format results payload."""
        self.phase = GamePhase.RESULTS
        self.phase_start_time_utc = time.time()

        active_players = [p for p in self.players.values() if not p.is_spectator and p.metrics is not None]

        # Calculate scores
        for p in active_players:
            p.pareto_score = self.calculate_pareto_score(p.metrics)  # type: ignore[arg-type]
            p.awards = []

        if active_players:
            # Assign Speed Champion (lowest total latency)
            speed_champion = min(active_players, key=lambda x: x.metrics.total_latency_s)  # type: ignore[union-attr]
            speed_champion.awards.append("speed_champion")

            # Assign Accuracy Champion (highest score)
            acc_champion = max(active_players, key=lambda x: x.metrics.score)  # type: ignore[union-attr]
            acc_champion.awards.append("accuracy_champion")

            # Assign Green Compute Champion (lowest RAM)
            ram_champion = min(active_players, key=lambda x: x.metrics.extraction_ram_mb)  # type: ignore[union-attr]
            ram_champion.awards.append("green_compute_champion")

            # Assign Overall Pareto Champion
            pareto_champion = max(active_players, key=lambda x: x.pareto_score)
            pareto_champion.awards.append("pareto_champion")

        # Sort leaderboard by pareto score descending
        ranked_players = sorted(active_players, key=lambda x: x.pareto_score, reverse=True)

        # Retrieve full track Pareto frontier for background scatter plot
        pareto_configs = self.engine.get_pareto_frontier(self.current_track_id)
        frontier_points = [
            {
                "key": f"{c.extractor}|{c.selector}|{c.model}",
                "extractor": c.extractor,
                "selector": c.selector,
                "model": c.model,
                "latency_s": c.total_latency_s,
                "score": c.score,
                "ram_mb": c.extraction_ram_mb,
                "speedup_vs_tsfresh": c.speedup_vs_tsfresh,
                "ram_reduction_pct": c.ram_reduction_pct,
                "is_pareto_optimal": True,
            }
            for c in pareto_configs
        ]

        results_payload = {
            "track_id": self.current_track_id,
            "rankings": [p.to_dict() for p in ranked_players],
            "pareto_frontier": frontier_points,
            "recorded_at_utc": time.time(),
        }
        self.race_results_cache = results_payload
        return results_payload

    def reset_to_lobby(self) -> None:
        """Reset the room state to LOBBY and clear ready flags."""
        self.phase = GamePhase.LOBBY
        self.phase_start_time_utc = time.time()
        self.race_start_timestamp_ms = 0
        self.race_results_cache = None

        # Clean up temporary bots
        bot_ids = [cid for cid, p in self.players.items() if p.is_bot]
        for bid in bot_ids:
            self.players.pop(bid, None)

        # Reset player readiness and promote spectators if slots opened
        self._promote_spectators()
        for p in self.players.values():
            p.is_ready = False
            p.awards = []

    def get_state_snapshot(self) -> dict[str, Any]:
        """Generate a complete JSON-serializable snapshot of the booth room for client synchronization."""
        track = self.engine.get_track(self.current_track_id)
        options = self.engine.get_valid_options(self.current_track_id)

        track_info = {
            "track_id": self.current_track_id,
            "display_name": track.display_name if track is not None else self.current_track_id,
            "domain": track.domain if track is not None else "General",
            "task_type": track.task_type if track is not None else "classification",
            "primary_metric": track.primary_metric if track is not None else "accuracy",
            "description": track.description if track is not None else "",
            "baseline_extractor": track.baseline_extractor if track is not None else "tsfresh_efficient",
            "baseline_model": track.baseline_model if track is not None else "random_forest",
            "options": options,
        }

        active_racers = [p.to_dict() for p in self.players.values() if not p.is_spectator]
        spectators = [p.to_dict() for p in self.players.values() if p.is_spectator]

        return {
            "phase": self.phase.value,
            "track": track_info,
            "active_racers": active_racers,
            "spectators": spectators,
            "max_active_racers": self.max_active_racers,
            "race_duration_seconds": self.race_duration_seconds,
            "countdown_seconds": self.countdown_seconds,
            "race_start_timestamp_ms": self.race_start_timestamp_ms,
            "server_time_utc": time.time(),
            "results": self.race_results_cache,
        }
