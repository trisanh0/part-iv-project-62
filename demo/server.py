"""TEMPO Display Day Demo Server.

FastAPI application providing REST endpoints, WebSocket synchronization,
and static asset delivery for the interactive presentation demonstration.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from demo.game_state import BoothRoomManager, GamePhase
from demo.query import DemoQueryEngine

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="TEMPO Display Day Demo",
    description="Synchronized multiplayer time-series machine learning racing demonstration.",
    version="1.0.0",
)

# Allow local cross-origin connections for multi-device development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data and state engines
query_engine = DemoQueryEngine(prefer_memory_cache=True)
room_manager = BoothRoomManager(
    query_engine=query_engine,
    initial_track_id=os.getenv("DEMO_DEFAULT_TRACK", "beed"),
    max_active_racers=int(os.getenv("DEMO_MAX_ACTIVE_RACERS", "4")),
    race_duration_seconds=float(os.getenv("DEMO_RACE_DURATION_SECONDS", "14.0")),
    countdown_seconds=float(os.getenv("DEMO_COUNTDOWN_SECONDS", "3.0")),
    admin_pin=os.getenv("DEMO_ADMIN_PIN", "7000"),
)

# Active WebSocket connections mapping: client_id -> WebSocket
connections: dict[str, WebSocket] = {}
connections_lock = asyncio.Lock()
race_task: Optional[asyncio.Task] = None


async def broadcast_event(event_type: str, data: dict[str, Any]) -> None:
    """Broadcast a structured JSON event to all connected WebSocket clients."""
    payload = {"type": event_type, "data": data, "timestamp_utc": time.time()}
    async with connections_lock:
        stale_clients = []
        for client_id, ws in connections.items():
            try:
                await ws.send_json(payload)
            except Exception:
                stale_clients.append(client_id)

        for client_id in stale_clients:
            connections.pop(client_id, None)


async def broadcast_state() -> None:
    """Broadcast full room snapshot to all connected clients."""
    snapshot = room_manager.get_state_snapshot()
    await broadcast_event("STATE_UPDATE", snapshot)


async def execute_race_lifecycle() -> None:
    """Manage the timed progression through countdown, active racing, and results."""
    # Ensure at least 2 racers exist for an engaging heat
    active_players = [p for p in room_manager.players.values() if not p.is_spectator]
    if len(active_players) == 0:
        room_manager.add_bot(nickname="TEMPO (Numba)", extractor="numba_efficient", selector="None", model="random_forest")
        room_manager.add_bot(nickname="Baseline (tsfresh)", extractor="tsfresh_efficient", selector="None", model="random_forest")
    elif len(active_players) == 1:
        room_manager.add_bot(nickname="Benchmark: tsfresh", extractor="tsfresh_efficient", selector="None", model="random_forest")

    # Start countdown and immediately sync all clients
    room_manager.start_countdown()
    await broadcast_state()
    await broadcast_event("COUNTDOWN_START", {
        "countdown_seconds": room_manager.countdown_seconds,
        "race_start_timestamp_ms": room_manager.race_start_timestamp_ms,
    })
    await asyncio.sleep(room_manager.countdown_seconds)

    if room_manager.phase != GamePhase.COUNTDOWN:
        return

    # Racing phase
    room_manager.start_racing()
    snapshot = room_manager.get_state_snapshot()
    await broadcast_event("RACE_START", {
        "duration_seconds": room_manager.race_duration_seconds,
        "racers": snapshot["active_racers"],
        "track_id": room_manager.current_track_id,
    })
    await broadcast_state()

    # Active racing duration
    await asyncio.sleep(room_manager.race_duration_seconds)

    if room_manager.phase != GamePhase.RACING:
        return

    # Results phase
    results = room_manager.finish_race()
    await broadcast_event("RACE_FINISH", results)
    await broadcast_state()


def trigger_race_lifecycle() -> None:
    """Schedule the async race lifecycle task."""
    global race_task
    if race_task is not None and not race_task.done():
        race_task.cancel()
    race_task = asyncio.create_task(execute_race_lifecycle())


# --- REST API Endpoints ---

class AdminActionRequest(BaseModel):
    """Schema for authenticated presenter administration commands."""

    pin: str = Field(..., description="Presenter admin passcode.")
    action: str = Field(..., description="Action to execute: change_track, start_building, start_race, add_bot, reset_lobby, set_duration.")
    track_id: Optional[str] = None
    bot_name: Optional[str] = None
    extractor: Optional[str] = None
    selector: Optional[str] = None
    model: Optional[str] = None
    duration_seconds: Optional[float] = None


@app.get("/api/tracks")
async def list_tracks() -> JSONResponse:
    """Retrieve metadata for all benchmark competition tracks."""
    tracks = query_engine.list_tracks(flagship_only=False)
    payload = [
        {
            "track_id": t.track_id,
            "display_name": t.display_name,
            "domain": t.domain,
            "task_type": t.task_type,
            "primary_metric": t.primary_metric,
            "is_curated_flagship": t.is_curated_flagship,
            "description": t.description,
            "baseline_extractor": t.baseline_extractor,
            "baseline_model": t.baseline_model,
        }
        for t in tracks
    ]
    return JSONResponse(content={"tracks": payload, "current_track_id": room_manager.current_track_id})


@app.get("/api/options/{track_id}")
async def get_track_options(track_id: str) -> JSONResponse:
    """Retrieve available algorithm dropdown options for a given track."""
    track = query_engine.get_track(track_id)
    if track is None:
        raise HTTPException(status_code=404, detail=f"Track '{track_id}' not found.")
    options = query_engine.get_valid_options(track_id)
    return JSONResponse(content={"track_id": track_id, "options": options})


@app.get("/api/pareto/{track_id}")
async def get_track_pareto_data(track_id: str) -> JSONResponse:
    """Retrieve all evaluated points and the non-dominated Pareto frontier for plotting."""
    all_configs = query_engine.get_all_configs(track_id)
    pareto_configs = query_engine.get_pareto_frontier(track_id)

    def serialize(cfg):
        return {
            "key": f"{cfg.extractor}|{cfg.selector}|{cfg.model}",
            "extractor": cfg.extractor,
            "selector": cfg.selector,
            "model": cfg.model,
            "latency_s": cfg.total_latency_s,
            "score": cfg.score,
            "ram_mb": cfg.extraction_ram_mb,
            "speedup_vs_tsfresh": cfg.speedup_vs_tsfresh,
            "ram_reduction_pct": cfg.ram_reduction_pct,
            "is_pareto_optimal": cfg.is_pareto_optimal,
            "pareto_composite_score": cfg.pareto_composite_score,
        }

    return JSONResponse(
        content={
            "track_id": track_id,
            "all_points": [serialize(c) for c in all_configs],
            "pareto_frontier": [serialize(c) for c in pareto_configs],
        }
    )


@app.get("/api/leaderboard/{track_id}")
async def get_track_leaderboard(track_id: str, sort_by: str = "pareto", limit: int = 10) -> JSONResponse:
    """Retrieve top empirical configurations for a track."""
    entries = query_engine.get_leaderboard(track_id, sort_by=sort_by, limit=limit)
    payload = [
        {
            "extractor": e.extractor,
            "selector": e.selector,
            "model": e.model,
            "latency_s": e.total_latency_s,
            "score": e.score,
            "ram_mb": e.extraction_ram_mb,
            "speedup": e.speedup_vs_tsfresh,
            "pareto_score": e.pareto_composite_score,
        }
        for e in entries
    ]
    return JSONResponse(content={"track_id": track_id, "leaderboard": payload})


@app.get("/api/state")
async def get_state() -> JSONResponse:
    """Retrieve immediate room snapshot."""
    return JSONResponse(content=room_manager.get_state_snapshot())


@app.post("/api/admin/action")
async def execute_admin_action(body: AdminActionRequest) -> JSONResponse:
    """Execute authenticated presenter admin control actions."""
    if body.pin != room_manager.admin_pin:
        raise HTTPException(status_code=403, detail="Invalid admin PIN.")

    action = body.action
    if action == "change_track":
        if not body.track_id:
            raise HTTPException(status_code=400, detail="Missing track_id.")
        success = room_manager.change_track(body.track_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Track '{body.track_id}' not found.")
        await broadcast_state()
        return JSONResponse(content={"status": "ok", "message": f"Track changed to {body.track_id}"})

    elif action == "start_building":
        room_manager.start_building()
        await broadcast_state()
        return JSONResponse(content={"status": "ok", "message": "Moved to BUILDING phase"})

    elif action == "start_race":
        room_manager.start_countdown()
        trigger_race_lifecycle()
        await broadcast_state()
        return JSONResponse(content={"status": "ok", "message": "Countdown triggered"})

    elif action == "add_bot":
        extractor = body.extractor or "tsfresh_efficient"
        selector = body.selector or "None"
        model = body.model or "random_forest"
        bot = room_manager.add_bot(
            nickname=body.bot_name,
            extractor=extractor,
            selector=selector,
            model=model,
        )
        if bot is None:
            raise HTTPException(status_code=400, detail="Lobby is full. Cannot add bot.")
        await broadcast_state()
        return JSONResponse(content={"status": "ok", "message": f"Bot '{bot.nickname}' added."})

    elif action == "reset_lobby":
        global race_task
        if race_task is not None and not race_task.done():
            race_task.cancel()
        room_manager.reset_to_lobby()
        await broadcast_state()
        return JSONResponse(content={"status": "ok", "message": "Lobby reset."})

    elif action == "set_duration":
        if body.duration_seconds and body.duration_seconds > 0:
            room_manager.race_duration_seconds = body.duration_seconds
            await broadcast_state()
            return JSONResponse(content={"status": "ok", "message": f"Duration set to {body.duration_seconds}s"})
        raise HTTPException(status_code=400, detail="Invalid duration_seconds.")

    else:
        raise HTTPException(status_code=400, detail=f"Unknown admin action: {action}")


# --- WebSocket Real-Time Interface ---

@app.websocket("/ws/game")
async def websocket_game_endpoint(websocket: WebSocket) -> None:
    """Bi-directional WebSocket connection handler for participants and spectator screens."""
    await websocket.accept()
    client_id = f"client_{uuid.uuid4().hex[:8]}"

    async with connections_lock:
        connections[client_id] = websocket

    try:
        # Send initial handshake with client ID and current state
        init_payload = {
            "type": "INIT",
            "data": {
                "client_id": client_id,
                "state": room_manager.get_state_snapshot(),
            },
            "timestamp_utc": time.time(),
        }
        await websocket.send_json(init_payload)

        while True:
            msg = await websocket.receive_json()
            action = msg.get("action")

            if action == "join":
                nickname = msg.get("nickname", "").strip() or "Visitor"
                color = msg.get("color")
                room_manager.add_player(client_id, nickname=nickname, avatar_color=color)
                await broadcast_state()

            elif action == "leave":
                room_manager.remove_player(client_id)
                await broadcast_state()

            elif action == "set_pipeline":
                extractor = msg.get("extractor", "numba_efficient")
                selector = msg.get("selector", "None")
                model = msg.get("model", "random_forest")
                room_manager.update_pipeline(client_id, extractor=extractor, selector=selector, model=model)
                await broadcast_state()

            elif action == "set_ready":
                is_ready = bool(msg.get("ready", True))
                triggered = room_manager.set_player_ready(client_id, is_ready=is_ready)
                if triggered:
                    trigger_race_lifecycle()
                await broadcast_state()

            elif action == "start_race":
                trigger_race_lifecycle()
                await broadcast_state()

            elif action == "reset_lobby":
                if race_task is not None and not race_task.done():
                    race_task.cancel()
                room_manager.reset_to_lobby()
                await broadcast_state()

            elif action == "add_bot":
                room_manager.add_bot(
                    nickname=msg.get("nickname"),
                    extractor=msg.get("extractor", "tsfresh_efficient"),
                    selector=msg.get("selector", "None"),
                    model=msg.get("model", "random_forest"),
                )
                await broadcast_state()

            elif action == "change_track":
                t_id = msg.get("track_id")
                if t_id:
                    room_manager.change_track(t_id)
                    await broadcast_state()

            elif action == "ping":
                await websocket.send_json({"type": "PONG", "timestamp_utc": time.time()})

            elif action == "admin":
                pin = msg.get("pin")
                if pin == room_manager.admin_pin:
                    admin_act = msg.get("admin_action")
                    if admin_act == "start_race":
                        trigger_race_lifecycle()
                        await broadcast_state()
                    elif admin_act == "reset":
                        if race_task is not None and not race_task.done():
                            race_task.cancel()
                        room_manager.reset_to_lobby()
                        await broadcast_state()
                    elif admin_act == "change_track":
                        t_id = msg.get("track_id")
                        if t_id:
                            room_manager.change_track(t_id)
                            await broadcast_state()
                    elif admin_act == "add_bot":
                        room_manager.add_bot(
                            nickname=msg.get("nickname"),
                            extractor=msg.get("extractor", "tsfresh_efficient"),
                            selector=msg.get("selector", "None"),
                            model=msg.get("model", "random_forest"),
                        )
                        await broadcast_state()

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        async with connections_lock:
            connections.pop(client_id, None)
        room_manager.remove_player(client_id)
        await broadcast_state()


# --- Static Files & Index ---

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_index() -> FileResponse:
    """Serve the root web interface."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Static index.html not yet generated.")
    return FileResponse(index_path)
