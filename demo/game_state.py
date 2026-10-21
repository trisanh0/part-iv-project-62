"""Synchronous room transitions; a heat owns immutable entries and metrics."""
from __future__ import annotations
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from demo.query import ConfigMetrics, DemoQueryEngine
from demo.race import RACE_RULES_VERSION, RaceSchedule, build_schedules
from demo.scoring import SCORING_VERSION, composite_score, performance_order, performance_utility

PROTOCOL_VERSION = 2
VEHICLE_TYPES = ('formula', 'hatchback', 'motorbike')
CAR_COLORS = ['#FF4D4D', '#00E5FF', '#76FF03', '#FFD600', '#D500F9', '#FF9100']


class GamePhase(str, Enum):
    LOBBY = 'LOBBY'
    BUILDING = 'BUILDING'
    COUNTDOWN = 'COUNTDOWN'
    RACING = 'RACING'
    RESULTS = 'RESULTS'


def metrics_dict(c):
    if c is None:
        return None
    return {**{k: getattr(c, k) for k in ('track_id', 'task_type', 'extractor', 'selector', 'model',
            'extraction_time_s', 'extraction_ram_mb', 'selection_time_s', 'fit_time_s', 'total_latency_s', 'score',
            'n_extracted_features', 'n_selected_features', 'sample_count',
            'speedup_vs_tsfresh', 'ram_reduction_pct', 'is_pareto_optimal', 'pareto_composite_score')},
            'performance_utility': performance_utility(c.task_type, c.score)}


@dataclass
class PlayerSlot:
    client_id: str
    nickname: str
    avatar_color: str
    car_index: int = 0
    vehicle: str = 'formula'
    preferred_color: Optional[str] = None
    is_spectator: bool = False
    is_bot: bool = False
    is_ready: bool = False
    preparing_next: bool = False
    reviewing_results: bool = False
    last_results: Optional[dict] = field(default=None, repr=False)
    ready_at: float = 0
    pipeline: dict[str, str] = field(default_factory=dict)
    metrics: Optional[ConfigMetrics] = None
    pareto_score: float = 0.0
    awards: list[str] = field(default_factory=list)
    connected: bool = True
    disconnected_at: Optional[float] = None

    def to_dict(self):
        return {k: (metrics_dict(v) if k == 'metrics' else v.copy() if isinstance(v, (dict, list)) else v)
                for k, v in vars(self).items() if k not in ('disconnected_at', 'last_results', 'ready_at')}


@dataclass(frozen=True)
class RaceEntry:
    client_id: str
    nickname: str
    avatar_color: str
    car_index: int
    is_bot: bool
    metrics: ConfigMetrics
    vehicle: str = 'formula'

    def to_dict(self):
        return dict(client_id=self.client_id, nickname=self.nickname, avatar_color=self.avatar_color,
                    car_index=self.car_index, vehicle=self.vehicle, is_bot=self.is_bot, is_spectator=False, is_ready=True,
                    metrics=metrics_dict(self.metrics), pipeline={k: getattr(self.metrics, k) for k in ('extractor', 'selector', 'model')})


@dataclass(frozen=True)
class Heat:
    heat_id: str
    track_id: str
    build_id: str
    entries: tuple[RaceEntry, ...]
    duration_seconds: float
    race_start_timestamp_ms: int
    schedules: tuple[RaceSchedule, ...]
    race_rules_version: str = RACE_RULES_VERSION
    scoring_version: str = SCORING_VERSION


class BoothRoomManager:
    def __init__(self, query_engine: DemoQueryEngine, initial_track_id='gas-sensor-drift-sub', max_active_racers=4,
                 race_duration_seconds=27.0, countdown_seconds=7.0, admin_pin='', grid_window_seconds=20.0):
        if not query_engine.get_track(initial_track_id):
            raise ValueError('Default track is not in the eligible catalogue')
        if not 2 <= max_active_racers <= len(CAR_COLORS):
            raise ValueError('Racer capacity must be between 2 and 6')
        if not math.isfinite(race_duration_seconds) or not 5 <= race_duration_seconds <= 60:
            raise ValueError('Race duration must be between 5 and 60 seconds')
        if not math.isfinite(countdown_seconds) or not 0 <= countdown_seconds <= 30:
            raise ValueError('Invalid countdown')
        if not math.isfinite(grid_window_seconds) or not 1 <= grid_window_seconds <= 120:
            raise ValueError('Grid entry window must be between 1 and 120 seconds')
        self.engine, self.current_track_id = query_engine, initial_track_id
        self.max_active_racers, self.race_duration_seconds = max_active_racers, race_duration_seconds
        self.countdown_seconds, self.admin_pin = countdown_seconds, admin_pin
        self.phase = GamePhase.LOBBY
        self.players = {}
        self.phase_start_time_utc = time.time()
        self.race_start_timestamp_ms = 0
        self.race_results_cache = None
        self.heat = None
        self.revision = 0
        self.bot_counter = 0
        self.grid_window_seconds = grid_window_seconds
        self.grid_close_timestamp_ms = 0
        self.grid_open_timestamp_ms = 0

    def touch(self):
        self.revision += 1

    @property
    def admission_open(self):
        """The grid is locked only while a heat is starting or running."""
        return self.phase not in (GamePhase.COUNTDOWN, GamePhase.RACING)

    def require_editable(self):
        if self.phase not in (GamePhase.LOBBY, GamePhase.BUILDING):
            raise ValueError('Reset to the lobby before editing this heat')

    def is_current_entrant(self, client_id):
        return (not self.admission_open and self.heat is not None
                and any(entry.client_id == client_id for entry in self.heat.entries))

    def _ready_grid(self):
        return [p for p in self.players.values() if p.connected and p.is_ready
                and not p.is_spectator and not p.reviewing_results]

    def _allocate_car_index(self, exclude=None):
        used = {p.car_index for p in self._ready_grid() if p.client_id != exclude}
        return next(i for i in range(self.max_active_racers) if i not in used)

    def ensure_grid_window(self):
        if self.admission_open and self._ready_grid():
            if not self.grid_close_timestamp_ms:
                self.grid_open_timestamp_ms = round(time.time() * 1000)
                self.grid_close_timestamp_ms = self.grid_open_timestamp_ms + round(self.grid_window_seconds * 1000)
                self.touch()
        elif self.admission_open and self.grid_close_timestamp_ms:
            self.grid_close_timestamp_ms = self.grid_open_timestamp_ms = 0
            self.touch()

    def set_grid_window(self, seconds):
        if not math.isfinite(seconds) or not 1 <= seconds <= 120:
            raise ValueError('Grid entry window must be between 1 and 120 seconds')
        self.grid_window_seconds = seconds
        if self.grid_close_timestamp_ms:
            self.grid_close_timestamp_ms = self.grid_open_timestamp_ms + round(seconds * 1000)
        self.touch()

    def _assign_color(self, player, reserved=None):
        used = ({p.avatar_color for p in self._ready_grid() if p.client_id != player.client_id}
                if reserved is None else reserved)
        preferred = player.preferred_color or player.avatar_color
        player.avatar_color = preferred if preferred in CAR_COLORS and preferred not in used else next(c for c in CAR_COLORS if c not in used)

    def update_appearance(self, client_id, vehicle, color):
        self.require_player_editable(client_id)
        if vehicle not in VEHICLE_TYPES or color not in CAR_COLORS:
            raise ValueError('Choose an available vehicle and palette colour')
        p = self.players[client_id]
        p.vehicle, p.preferred_color, p.avatar_color = vehicle, color, color
        p.is_ready = p.is_spectator = False
        self._promote_spectators()
        self.ensure_grid_window()
        self.touch()

    def presenter_snapshot(self):
        state = self.get_state_snapshot()
        roster = []
        queue = sorted((p for p in self.players.values() if p.is_spectator), key=lambda p: p.ready_at)
        for p in self.players.values():
            status = ('Racing' if self.is_current_entrant(p.client_id) else 'Offline' if not p.connected
                      else 'Reviewing results' if p.reviewing_results else 'Queued' if p.is_spectator
                      else 'Ready' if p.is_ready else 'Building')
            roster.append({**p.to_dict(), 'status': status,
                           'queue_position': queue.index(p) + 1 if p in queue else None})
        state['roster'] = roster
        return state

    def _get_default_pipeline(self):
        c = min(self.engine.get_all_configs(self.current_track_id),
                key=lambda c: (c.extractor != 'numpy_statistical', c.selector != 'None', c.model != 'logistic_regression', c.key))
        return {k: getattr(c, k) for k in ('extractor', 'selector', 'model')}

    def add_player(self, client_id, nickname, avatar_color=None):
        if client_id in self.players:
            return self.players[client_id]
        if not isinstance(nickname, str) or not 1 <= len(nickname.strip()) <= 20:
            raise ValueError('Nickname must contain 1–20 characters')
        if avatar_color is not None and not re.fullmatch(r'#[0-9a-fA-F]{6}', avatar_color):
            raise ValueError('Invalid colour')
        self.expire_disconnected()
        index = sum(not p.is_bot for p in self.players.values()) % self.max_active_racers
        pipeline = self._get_default_pipeline()
        p = PlayerSlot(client_id, nickname.strip(), avatar_color or CAR_COLORS[index], index,
                       preferred_color=avatar_color or CAR_COLORS[index],
                       is_spectator=False, preparing_next=True,
                       pipeline=pipeline,
                       metrics=self.engine.get_config(self.current_track_id, **pipeline))
        self.players[client_id] = p
        self.touch()
        return p

    def remove_player(self, client_id):
        p = self.players.pop(client_id, None)
        if self.admission_open:
            self._promote_spectators()
            self.ensure_grid_window()
        self.touch()
        return p

    def disconnect(self, client_id):
        if p := self.players.get(client_id):
            p.connected, p.is_ready, p.disconnected_at = False, False, time.monotonic()
            p.is_spectator = False
            self._promote_spectators()
            self.ensure_grid_window()
            self.touch()

    def reconnect(self, client_id):
        if p := self.players.get(client_id):
            p.connected, p.disconnected_at = True, None
            self.touch()

    def expire_disconnected(self, now=None):
        now = time.monotonic() if now is None else now
        for cid, p in list(self.players.items()):
            if (not p.connected and not self.is_current_entrant(cid) and p.disconnected_at is not None
                    and now - p.disconnected_at >= 30):
                self.remove_player(cid)

    def _promote_spectators(self):
        if not self.admission_open:
            return
        waiting = sorted((p for p in self.players.values() if p.is_spectator and p.connected and p.is_ready),
                         key=lambda p: p.ready_at)
        for p in waiting:
            if len(self._ready_grid()) >= self.max_active_racers:
                break
            p.car_index = self._allocate_car_index(p.client_id)
            p.is_spectator = False
            self._assign_color(p)
            p.preparing_next = True
        self.ensure_grid_window()

    def require_player_editable(self, client_id):
        player = self.players.get(client_id)
        if player is None or player.reviewing_results or self.is_current_entrant(client_id):
            raise ValueError('Open your build before editing it')

    def update_pipeline(self, client_id, extractor, selector, model):
        self.require_player_editable(client_id)
        p = self.players.get(client_id)
        if p is None or p.is_spectator:
            raise ValueError('Join an active slot first')
        c = self.engine.get_config(self.current_track_id, extractor, selector, model)
        if c is None:
            raise ValueError('Unsupported pipeline combination')
        p.pipeline = {k: getattr(c, k) for k in ('extractor', 'selector', 'model')}
        p.metrics, p.is_ready, p.is_spectator = c, False, False
        self._promote_spectators()
        self.ensure_grid_window()
        self.touch()
        return p

    def set_player_ready(self, client_id, is_ready=True):
        self.require_player_editable(client_id)
        p = self.players.get(client_id)
        if p is None or p.metrics is None or not p.connected:
            raise ValueError('A connected participant with a valid pipeline is required')
        if is_ready and not p.is_ready:
            p.ready_at = time.monotonic()
        p.is_ready = is_ready
        p.is_spectator = False
        if is_ready:
            if not self.admission_open or len(self._ready_grid()) > self.max_active_racers:
                p.is_spectator = True
            else:
                p.car_index = self._allocate_car_index(p.client_id)
                self._assign_color(p)
        self._promote_spectators()
        self.ensure_grid_window()
        self.touch()
        return bool(self.grid_close_timestamp_ms)

    def add_bot(self, nickname=None, extractor=None, selector=None, model=None):
        if not self.admission_open:
            self.require_editable()
        if len(self._ready_grid()) >= self.max_active_racers:
            return None
        configs = self.engine.get_all_configs(self.current_track_id)
        if any(x is not None for x in (extractor, selector, model)):
            configs = [c for c in configs if all(value is None or getattr(c, key) == value for key, value in
                       (('extractor', extractor), ('selector', selector), ('model', model)))]
        if not configs:
            raise ValueError('No eligible bot configuration')
        baseline = [c for c in configs if c.extractor in ('tsfresh_efficient', 'tsfresh_minimal')]
        c = min(baseline, key=lambda c: (c.extractor != 'tsfresh_efficient', c.selector != 'None', c.model != 'random_forest', c.key)) if baseline else min(configs, key=lambda c: (-c.total_latency_s, c.key))
        self.bot_counter += 1
        name = nickname or 'Benchmark bot'
        p = self.add_player(f'bot_{self.bot_counter}', name)
        p.metrics, p.is_bot, p.is_ready = c, True, True
        p.car_index = self._allocate_car_index(p.client_id)
        self._assign_color(p)
        p.pipeline = {k: getattr(c, k) for k in ('extractor', 'selector', 'model')}
        self.ensure_grid_window()
        return p

    def change_track(self, track_id):
        if not self.admission_open:
            raise ValueError('End the current heat before changing challenge')
        if self.engine.get_track(track_id) is None:
            return False
        self.grid_close_timestamp_ms = self.grid_open_timestamp_ms = 0
        self.current_track_id = track_id
        for p in self.players.values():
            p.reviewing_results, p.last_results, p.is_spectator = False, None, False
            pipeline = p.pipeline if self.engine.get_config(track_id, **p.pipeline) else self._get_default_pipeline()
            p.pipeline, p.metrics, p.is_ready = pipeline.copy(), self.engine.get_config(track_id, **pipeline), p.is_bot
        self.heat, self.race_results_cache, self.race_start_timestamp_ms = None, None, 0
        self.phase, self.phase_start_time_utc = GamePhase.LOBBY, time.time()
        self.touch()
        return True

    def start_building(self):
        self.require_editable()
        self.phase, self.phase_start_time_utc = GamePhase.BUILDING, time.time()
        for p in self.players.values():
            p.is_ready = p.is_bot
        self.ensure_grid_window()
        self.touch()

    def start_countdown(self, ready_only=False):
        if not self.admission_open:
            return False
        if ready_only:
            active = self._ready_grid()
            if not active:
                self.grid_close_timestamp_ms = 0
                return False
        else:
            active = [p for p in self.players.values() if p.connected and not p.reviewing_results
                      and not p.is_spectator][:self.max_active_racers]
        for index, player in enumerate(active):
            player.car_index = index
            reserved = {p.avatar_color for p in active[:index]}
            if player.avatar_color not in CAR_COLORS or player.avatar_color in reserved:
                self._assign_color(player, reserved)
            player.is_ready = True
            player.preparing_next = False
        while len(active) < 2:
            active.append(self.add_bot())
        self.grid_close_timestamp_ms = self.grid_open_timestamp_ms = 0
        self.phase, self.phase_start_time_utc = GamePhase.COUNTDOWN, time.time()
        self.race_start_timestamp_ms = int((self.phase_start_time_utc + self.countdown_seconds) * 1000)
        entries = tuple(RaceEntry(p.client_id, p.nickname, p.avatar_color, p.car_index, p.is_bot, p.metrics, p.vehicle) for p in active)
        self.heat = Heat(uuid.uuid4().hex, self.current_track_id, self.engine.build_id,
                         entries, self.race_duration_seconds, self.race_start_timestamp_ms,
                         build_schedules(entries, self.race_duration_seconds))
        self.touch()
        return True

    def start_racing(self):
        if self.phase != GamePhase.COUNTDOWN or not self.heat:
            raise ValueError('Countdown required')
        self.phase, self.phase_start_time_utc = GamePhase.RACING, self.race_start_timestamp_ms / 1000
        self.touch()

    def calculate_pareto_score(self, metrics):
        return composite_score(metrics.task_type, metrics.score, metrics.total_latency_s, metrics.extraction_ram_mb)

    def finish_race(self):
        if self.phase == GamePhase.RESULTS:
            return self.race_results_cache
        if self.phase != GamePhase.RACING or not self.heat:
            raise ValueError('Running heat required')
        self.phase, self.phase_start_time_utc = GamePhase.RESULTS, time.time()
        entries = self.heat.entries
        schedules = dict(zip((p.client_id for p in entries), self.heat.schedules))
        ranked = sorted(entries, key=lambda p: (schedules[p.client_id].finish_ms, p.client_id))
        awards = {
            'speed_champion': lambda p: -p.metrics.total_latency_s,
            'accuracy_champion': lambda p: performance_order(p.metrics.task_type, p.metrics.score),
            'green_compute_champion': lambda p: -p.metrics.extraction_ram_mb,
            'pareto_champion': lambda p: self.calculate_pareto_score(p.metrics),
        }
        rows = []
        position = 1
        for index, p in enumerate(ranked):
            schedule = schedules[p.client_id]
            if index and schedule.finish_ms != schedules[ranked[index - 1].client_id].finish_ms:
                position = index + 1
            row = p.to_dict()
            row.update(race_schedule=schedule.to_dict(), finish_position=position,
                       tied=sum(s.finish_ms == schedule.finish_ms for s in self.heat.schedules) > 1)
            row.update(pareto_score=self.calculate_pareto_score(p.metrics), awards=[name for name, key in awards.items() if key(p) == max(map(key, entries))])
            rows.append(row)
        self.race_results_cache = {'heat_id': self.heat.heat_id, 'build_id': self.heat.build_id,
            'scoring_version': SCORING_VERSION, 'race_rules_version': self.heat.race_rules_version,
            'track_id': self.heat.track_id, 'rankings': rows,
            'pareto_frontier': [dict(key=c.key, latency_s=c.total_latency_s, ram_mb=c.extraction_ram_mb, **metrics_dict(c))
                                for c in self.engine.get_pareto_frontier(self.heat.track_id)], 'recorded_at_utc': time.time()}
        for entry in entries:
            player = self.players.get(entry.client_id)
            if player is not None and not player.is_bot:
                player.last_results = self.race_results_cache
                player.reviewing_results = True
                player.is_ready = player.is_spectator = player.preparing_next = False
        for client_id, player in list(self.players.items()):
            if player.is_bot:
                del self.players[client_id]
        self.expire_disconnected()
        self._promote_spectators()
        self.touch()
        return self.race_results_cache

    def request_race_again(self, client_id: str, heat_id: Optional[str]) -> None:
        """Re-enter setup from this participant's own completed race."""
        player = self.players.get(client_id)
        if (player is None or not player.connected or player.is_bot or player.last_results is None
                or player.last_results['heat_id'] != heat_id or self.is_current_entrant(client_id)):
            raise ValueError('Race again requires your completed heat')
        player.reviewing_results = False
        player.preparing_next = True
        player.is_ready = player.is_spectator = False
        self.touch()

    def reset_to_lobby(self):
        self.phase, self.phase_start_time_utc = GamePhase.LOBBY, time.time()
        self.grid_close_timestamp_ms = self.grid_open_timestamp_ms = 0
        self.race_start_timestamp_ms, self.heat, self.race_results_cache = 0, None, None
        for cid, p in list(self.players.items()):
            if p.is_bot:
                del self.players[cid]
            else:
                p.is_ready, p.awards, p.preparing_next = False, [], False
                p.reviewing_results, p.last_results, p.is_spectator = False, None, False
        self.expire_disconnected()
        self._promote_spectators()
        self.touch()

    def get_state_snapshot(self, client_id=None):
        from dataclasses import asdict
        track = asdict(self.engine.get_track(self.current_track_id))
        track['options'] = self.engine.get_valid_options(self.current_track_id)
        active = ([p.to_dict() for p in self.heat.entries] if self.heat and self.phase in (GamePhase.COUNTDOWN, GamePhase.RACING)
                  else [p.to_dict() for p in self.players.values() if p.connected and not p.is_spectator and not p.reviewing_results])
        if self.heat and self.phase in (GamePhase.COUNTDOWN, GamePhase.RACING):
            for row, schedule in zip(active, self.heat.schedules):
                row['race_schedule'] = schedule.to_dict()
        participant = self.players.get(client_id)
        personal = participant.to_dict() if participant else None
        if personal is not None:
            personal['view'] = ('results' if participant.reviewing_results else 'race' if self.is_current_entrant(client_id)
                                else 'waiting' if participant.is_spectator else 'configuration')
        return dict(vehicle_types=list(VEHICLE_TYPES), car_colors=CAR_COLORS.copy(), settings=dict(race_duration_seconds=self.race_duration_seconds, grid_window_seconds=self.grid_window_seconds, track_id=self.current_track_id), participant=personal, personal_results=participant.last_results if participant else None,
                    grid_close_timestamp_ms=self.grid_close_timestamp_ms, grid_window_seconds=self.grid_window_seconds,
                    participants=[p.to_dict() for p in self.players.values() if p.connected and not p.reviewing_results and not self.is_current_entrant(p.client_id)],
                    race_rules_version=self.heat.race_rules_version if self.heat else RACE_RULES_VERSION,
                    protocol_version=PROTOCOL_VERSION, revision=self.revision, heat_id=self.heat.heat_id if self.heat else None,
                    build_id=self.engine.build_id, scoring_version=SCORING_VERSION, phase=self.phase.value, track=track,
                    active_racers=active, spectators=[p.to_dict() for p in sorted(self.players.values(), key=lambda p: p.ready_at) if p.is_spectator],
                    max_active_racers=self.max_active_racers, race_duration_seconds=self.heat.duration_seconds if self.heat and not self.admission_open else self.race_duration_seconds,
                    countdown_seconds=self.countdown_seconds, phase_start_time_utc=self.phase_start_time_utc,
                    race_start_timestamp_ms=self.race_start_timestamp_ms, server_time_utc=time.time(), results=self.race_results_cache)
