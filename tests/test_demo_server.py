"""Integration tests for the TEMPO Display Day REST and WebSocket server."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from demo.server import app, room_manager


@pytest.fixture(autouse=True)
def reset_server_state() -> None:
    """Ensure room manager is reset before each test."""
    room_manager.reset_to_lobby()
    room_manager.change_track("beed")


def test_index_route() -> None:
    """Verify that root URL returns the HTML application shell."""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "TEMPO" in response.text


def test_api_tracks() -> None:
    """Verify tracks endpoint returns metadata and current track ID."""
    with TestClient(app) as client:
        response = client.get("/api/tracks")
        assert response.status_code == 200
        data = response.json()
        assert "tracks" in data
        assert len(data["tracks"]) >= 4
        assert data["current_track_id"] == "beed"


def test_api_options() -> None:
    """Verify track options endpoint returns valid algorithm components."""
    with TestClient(app) as client:
        response = client.get("/api/options/beed")
        assert response.status_code == 200
        data = response.json()
        assert "options" in data
        opts = data["options"]
        assert "numba_efficient" in opts["extractors"]
        assert "random_forest" in opts["models"]

        # 404 for invalid track
        bad_response = client.get("/api/options/nonexistent_track")
        assert bad_response.status_code == 404


def test_api_pareto_data() -> None:
    """Verify pareto endpoint returns full dots and non-dominated frontier."""
    with TestClient(app) as client:
        response = client.get("/api/pareto/beed")
        assert response.status_code == 200
        data = response.json()
        assert "all_points" in data
        assert "pareto_frontier" in data
        assert len(data["all_points"]) > 0
        assert len(data["pareto_frontier"]) > 0


def test_api_leaderboard() -> None:
    """Verify leaderboard endpoint returns ranked configurations."""
    with TestClient(app) as client:
        response = client.get("/api/leaderboard/beed?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert "leaderboard" in data
        assert len(data["leaderboard"]) <= 5


def test_admin_pin_security() -> None:
    """Verify that admin actions require correct PIN."""
    with TestClient(app) as client:
        # Invalid PIN
        resp = client.post(
            "/api/admin/action",
            json={"pin": "wrong_pin", "action": "reset_lobby"},
        )
        assert resp.status_code == 403

        # Valid PIN
        resp = client.post(
            "/api/admin/action",
            json={"pin": "7000", "action": "reset_lobby"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_admin_add_bot_and_change_track() -> None:
    """Verify presenter actions for adding bots and switching tracks."""
    with TestClient(app) as client:
        resp = client.post(
            "/api/admin/action",
            json={
                "pin": "7000",
                "action": "add_bot",
                "bot_name": "Test Benchmark Bot",
                "extractor": "tsfresh_efficient",
                "selector": "None",
                "model": "random_forest",
            },
        )
        assert resp.status_code == 200
        assert "added" in resp.json()["message"]

        # Check track change
        resp = client.post(
            "/api/admin/action",
            json={
                "pin": "7000",
                "action": "change_track",
                "track_id": "pred-maintenance-w100-cls",
            },
        )
        assert resp.status_code == 200
        assert room_manager.current_track_id == "pred-maintenance-w100-cls"


def test_websocket_lifecycle() -> None:
    """Verify WebSocket handshake, join, pipeline update, and readiness toggle."""
    with TestClient(app) as client:
        with client.websocket_connect("/ws/game") as ws:
            # First frame should be INIT
            init_msg = ws.receive_json()
            assert init_msg["type"] == "INIT"
            assert "client_id" in init_msg["data"]

            # Send join action
            ws.send_json({"action": "join", "nickname": "Test Racer"})
            state_msg = ws.receive_json()
            assert state_msg["type"] == "STATE_UPDATE"
            active_racers = state_msg["data"]["active_racers"]
            assert len(active_racers) == 1
            assert active_racers[0]["nickname"] == "Test Racer"

            # Send pipeline update
            ws.send_json({
                "action": "set_pipeline",
                "extractor": "numba_efficient",
                "selector": "None",
                "model": "random_forest",
            })
            update_msg = ws.receive_json()
            assert update_msg["type"] == "STATE_UPDATE"
            pipe = update_msg["data"]["active_racers"][0]["pipeline"]
            assert pipe["extractor"] == "numba_efficient"

            # Send ready
            ws.send_json({"action": "set_ready", "ready": True})
            ready_msg = ws.receive_json()
            # If solitary racer, set_ready triggers countdown
            assert ready_msg["type"] in ("STATE_UPDATE", "COUNTDOWN_START")


def test_websocket_leave_action() -> None:
    """Verify that a player can voluntarily leave the heat via WebSocket."""
    with TestClient(app) as client:
        with client.websocket_connect("/ws/game") as ws:
            ws.receive_json()  # INIT
            ws.send_json({"action": "join", "nickname": "Leaver"})
            join_msg = ws.receive_json()
            assert len(join_msg["data"]["active_racers"]) == 1

            ws.send_json({"action": "leave"})
            leave_msg = ws.receive_json()
            assert len(leave_msg["data"]["active_racers"]) == 0


def test_websocket_grandstand_actions() -> None:
    """Verify Grandstand actions over WebSocket: add_bot, change_track, reset_lobby."""
    with TestClient(app) as client:
        with client.websocket_connect("/ws/game") as ws:
            ws.receive_json()  # INIT

            # Add bot
            ws.send_json({"action": "add_bot", "extractor": "tsfresh_efficient"})
            bot_msg = ws.receive_json()
            assert len(bot_msg["data"]["active_racers"]) == 1
            assert bot_msg["data"]["active_racers"][0]["is_bot"] is True

            # Change track
            ws.send_json({"action": "change_track", "track_id": "pred-maintenance-w100-cls"})
            track_msg = ws.receive_json()
            assert track_msg["data"]["track"]["track_id"] == "pred-maintenance-w100-cls"

            # Reset lobby
            ws.send_json({"action": "reset_lobby"})
            reset_msg = ws.receive_json()
            assert reset_msg["data"]["phase"] == "LOBBY"
            assert len(reset_msg["data"]["active_racers"]) == 0


def test_dual_client_presentation_flow() -> None:
    """Verify simultaneous dual-client flow: Phone participant vs Laptop Grandstand."""
    with TestClient(app) as client:
        # Client 1: Laptop Grandstand (Observer)
        with client.websocket_connect("/ws/game") as ws_laptop:
            ws_laptop.receive_json()  # INIT

            # Client 2: Phone Racer
            with client.websocket_connect("/ws/game") as ws_phone:
                ws_phone.receive_json()  # INIT

                # Phone racer joins
                ws_phone.send_json({"action": "join", "nickname": "Trisan"})
                phone_state = ws_phone.receive_json()
                laptop_state = ws_laptop.receive_json()

                assert len(phone_state["data"]["active_racers"]) == 1
                assert phone_state["data"]["active_racers"][0]["nickname"] == "Trisan"
                assert len(laptop_state["data"]["active_racers"]) == 1

                # Phone configures pipeline
                ws_phone.send_json({
                    "action": "set_pipeline",
                    "extractor": "numba_efficient",
                    "selector": "None",
                    "model": "random_forest",
                })
                phone_pipe_update = ws_phone.receive_json()
                ws_laptop.receive_json()
                metrics = phone_pipe_update["data"]["active_racers"][0]["metrics"]
                assert metrics is not None
                assert metrics["total_latency_s"] > 0

                # Grandstand on laptop clicks START RACE NOW
                ws_laptop.send_json({"action": "start_race"})

                # Both receive COUNTDOWN_START and state update
                # Server auto-injected Benchmark: tsfresh bot for head-to-head racing
                msg_phone = ws_phone.receive_json()
                msg_laptop = ws_laptop.receive_json()
                assert msg_phone["type"] in ("STATE_UPDATE", "COUNTDOWN_START")
                assert msg_laptop["type"] in ("STATE_UPDATE", "COUNTDOWN_START")


