"""Unit tests for the TEMPO Display Day Booth Room Manager and Game State."""

from __future__ import annotations

import pytest

from demo.game_state import BoothRoomManager, GamePhase, PlayerSlot
from demo.query import DemoQueryEngine


@pytest.fixture
def query_engine() -> DemoQueryEngine:
    """Fixture providing initialized query engine with in-memory catalogue."""
    return DemoQueryEngine(prefer_memory_cache=True)


@pytest.fixture
def room_manager(query_engine: DemoQueryEngine) -> BoothRoomManager:
    """Fixture providing initialized room manager on the default track."""
    return BoothRoomManager(
        query_engine=query_engine,
        initial_track_id="beed",
        max_active_racers=4,
        race_duration_seconds=10.0,
        countdown_seconds=2.0,
        admin_pin="7000",
    )


def test_initial_state(room_manager: BoothRoomManager) -> None:
    """Verify initial state of the booth room manager."""
    assert room_manager.phase == GamePhase.LOBBY
    assert room_manager.current_track_id == "beed"
    assert len(room_manager.players) == 0
    snapshot = room_manager.get_state_snapshot()
    assert snapshot["phase"] == "LOBBY"
    assert snapshot["track"]["track_id"] == "beed"
    assert len(snapshot["active_racers"]) == 0
    assert len(snapshot["spectators"]) == 0


def test_player_registration_and_color_assignment(room_manager: BoothRoomManager) -> None:
    """Verify player registration, nickname cleanup, and car index assignment."""
    p1 = room_manager.add_player("client_1", "Alice")
    assert p1.client_id == "client_1"
    assert p1.nickname == "Alice"
    assert not p1.is_spectator
    assert p1.car_index == 0
    assert p1.metrics is not None

    p2 = room_manager.add_player("client_2", "Bob")
    assert p2.client_id == "client_2"
    assert p2.car_index == 1
    assert p2.avatar_color != p1.avatar_color

    assert len(room_manager.players) == 2


def test_capacity_and_spectator_overflow(room_manager: BoothRoomManager) -> None:
    """Verify that players exceeding max_active_racers join as spectators."""
    for i in range(4):
        p = room_manager.add_player(f"client_{i}", f"Player {i}")
        assert not p.is_spectator

    # 5th player should be spectator
    p5 = room_manager.add_player("client_4", "Player 4")
    assert p5.is_spectator

    snapshot = room_manager.get_state_snapshot()
    assert len(snapshot["active_racers"]) == 4
    assert len(snapshot["spectators"]) == 1


def test_spectator_promotion_on_disconnect(room_manager: BoothRoomManager) -> None:
    """Verify that when an active player disconnects in the lobby, a spectator is promoted."""
    for i in range(4):
        room_manager.add_player(f"client_{i}", f"Player {i}")

    p5 = room_manager.add_player("client_4", "Spectator Player")
    assert p5.is_spectator

    # Remove active player 1
    removed = room_manager.remove_player("client_1")
    assert removed is not None
    assert not removed.is_spectator

    # Spectator should now be promoted
    promoted = room_manager.players["client_4"]
    assert not promoted.is_spectator
    assert promoted.car_index == 1


def test_pipeline_update_and_metric_resolution(room_manager: BoothRoomManager) -> None:
    """Verify updating player pipeline and resolving empirical metrics."""
    p = room_manager.add_player("c1", "Alice")
    updated = room_manager.update_pipeline(
        client_id="c1",
        extractor="numba_efficient",
        selector="None",
        model="random_forest",
    )
    assert updated is not None
    assert updated.metrics is not None
    assert updated.metrics.extractor == "numba_efficient"
    assert updated.metrics.total_latency_s < 5.0
    assert updated.metrics.score > 0.95


def test_ready_flow_and_auto_countdown(room_manager: BoothRoomManager) -> None:
    """Verify that when all active players are ready, the room enters countdown."""
    p1 = room_manager.add_player("c1", "Alice")
    p2 = room_manager.add_player("c2", "Bob")

    triggered = room_manager.set_player_ready("c1", True)
    assert not triggered
    assert room_manager.phase == GamePhase.LOBBY

    triggered = room_manager.set_player_ready("c2", True)
    assert triggered
    assert room_manager.phase == GamePhase.COUNTDOWN
    assert room_manager.race_start_timestamp_ms > 0


def test_add_bot(room_manager: BoothRoomManager) -> None:
    """Verify injecting a benchmark bot into the race."""
    p1 = room_manager.add_player("c1", "Solo Human")
    bot = room_manager.add_bot(
        nickname="Benchmark: tsfresh",
        extractor="tsfresh_efficient",
        selector="None",
        model="random_forest",
    )
    assert bot is not None
    assert bot.is_bot
    assert bot.is_ready
    assert bot.car_index == 1
    assert bot.metrics is not None
    assert bot.metrics.extractor == "tsfresh_efficient"

    # Now when human readies, countdown triggers
    triggered = room_manager.set_player_ready("c1", True)
    assert triggered
    assert room_manager.phase == GamePhase.COUNTDOWN


def test_change_track(room_manager: BoothRoomManager) -> None:
    """Verify changing competition track refreshes player metrics and resets state."""
    p = room_manager.add_player("c1", "Alice")
    room_manager.set_player_ready("c1", True)

    changed = room_manager.change_track("pred-maintenance-w100-cls")
    assert changed
    assert room_manager.current_track_id == "pred-maintenance-w100-cls"
    assert room_manager.phase == GamePhase.LOBBY
    assert not p.is_ready
    assert p.metrics is not None
    assert p.metrics.track_id == "pred-maintenance-w100-cls"


def test_race_finish_and_award_assignment(room_manager: BoothRoomManager) -> None:
    """Verify finishing a heat, computing Pareto scores, and awarding badges."""
    p1 = room_manager.add_player("c1", "Numba Fast")
    room_manager.update_pipeline("c1", "numba_efficient", "None", "random_forest")

    p2 = room_manager.add_player("c2", "tsfresh Heavy")
    room_manager.update_pipeline("c2", "tsfresh_efficient", "None", "random_forest")

    room_manager.start_racing()
    results = room_manager.finish_race()

    assert room_manager.phase == GamePhase.RESULTS
    assert "rankings" in results
    assert len(results["rankings"]) == 2

    r1 = results["rankings"][0]
    r2 = results["rankings"][1]

    # Numba should beat tsfresh in Pareto score due to massive speedup
    assert r1["nickname"] == "Numba Fast"
    assert "speed_champion" in r1["awards"]
    assert "green_compute_champion" in r1["awards"]
    assert "pareto_champion" in r1["awards"]


def test_reset_to_lobby(room_manager: BoothRoomManager) -> None:
    """Verify resetting room state to lobby cleans up bots and resets readiness."""
    room_manager.add_player("c1", "Alice")
    room_manager.add_bot("Bot")
    room_manager.start_racing()
    room_manager.finish_race()

    room_manager.reset_to_lobby()
    assert room_manager.phase == GamePhase.LOBBY
    assert len(room_manager.players) == 1
    assert "c1" in room_manager.players
    assert not room_manager.players["c1"].is_ready
