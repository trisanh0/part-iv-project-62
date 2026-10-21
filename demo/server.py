"""Single-worker booth server with validated commands and resumable participants."""
from __future__ import annotations
import asyncio
import hmac
import json
import hashlib
import mimetypes
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlsplit
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from demo.game_state import BoothRoomManager, GamePhase, PROTOCOL_VERSION, ALL_READY_COUNTDOWN_SECONDS, DEFAULT_SETTINGS
from demo.query import DemoQueryEngine

logger = logging.getLogger('demo.server')
STATIC_DIR = Path(__file__).resolve().parent / 'static'


def snapshot_frontend(directory: Path):
    """Pin one complete frontend so imports and geometry cannot mix releases."""
    files = {path.relative_to(directory).as_posix(): path.read_bytes()
             for path in sorted(directory.rglob('*')) if path.is_file()}
    digest = hashlib.sha256()
    for name, content in files.items():
        digest.update(name.encode() + b'\0' + hashlib.sha256(content).digest())
    return digest.hexdigest()[:20], files


FRONTEND_VERSION, FRONTEND_FILES = snapshot_frontend(STATIC_DIR)
query_engine = DemoQueryEngine()
room_manager = BoothRoomManager(query_engine, initial_track_id=os.getenv('DEMO_DEFAULT_TRACK', 'gas-sensor-drift-sub'),
    max_active_racers=int(os.getenv('DEMO_MAX_ACTIVE_RACERS', '6')),
    race_duration_seconds=float(os.getenv('DEMO_RACE_DURATION_SECONDS', '27')),
    countdown_seconds=float(os.getenv('DEMO_COUNTDOWN_SECONDS', '7')),
    grid_window_seconds=float(os.getenv('DEMO_GRID_WINDOW_SECONDS', '20')),
    laps=int(os.getenv('DEMO_LAPS', '2')))


class Command(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    action: Literal['join', 'leave', 'set_pipeline', 'set_ready', 'ping', 'start_race', 'reset_lobby',
                    'change_track', 'add_bot', 'set_duration', 'start_building', 'race_again',
                    'set_grid_window', 'remove_participant', 'set_appearance', 'set_laps', 'set_countdown', 'reset_settings']
    heat_id: Optional[str] = Field(default=None, min_length=1, max_length=64)
    pin: Optional[str] = Field(default=None, max_length=128)
    nickname: Optional[str] = Field(default=None, min_length=1, max_length=20)
    bot_name: Optional[str] = Field(default=None, min_length=1, max_length=20)
    color: Optional[str] = Field(default=None, pattern=r'^#[0-9a-fA-F]{6}$')
    extractor: Optional[str] = Field(default=None, min_length=1, max_length=100)
    selector: Optional[str] = Field(default=None, min_length=1, max_length=100)
    model: Optional[str] = Field(default=None, min_length=1, max_length=100)
    track_id: Optional[str] = Field(default=None, min_length=1, max_length=100)
    vehicle: Optional[Literal['formula', 'hatchback', 'motorbike', 'camper']] = None
    target_client_id: Optional[str] = Field(default=None, min_length=1, max_length=100)
    interrupt_heat: bool = False
    grid_window_seconds: Optional[float] = Field(default=None, ge=1, le=120, allow_inf_nan=False)
    laps: Optional[int] = Field(default=None, ge=2, le=6)
    countdown_seconds: Optional[float] = Field(default=None, ge=3, le=15, allow_inf_nan=False)
    ready: bool = True
    duration_seconds: Optional[float] = Field(default=None, ge=5, le=60, allow_inf_nan=False)


class Hello(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    action: Literal['hello']
    protocol_version: int
    resume_token: Optional[str] = Field(default=None, max_length=128)


@dataclass
class Session:
    client_id: str
    token: str
    disconnected_at: Optional[float] = None


@dataclass
class Connection:
    websocket: WebSocket
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=32))
    writer: Optional[asyncio.Task] = None
    closing: bool = False


sessions: dict[str, Session] = {}
connections: dict[str, Connection] = {}
race_task = None
command_lock = asyncio.Lock()
background_tasks = set()
PRESENTER_ACTIONS = {'start_race', 'reset_lobby', 'change_track', 'add_bot', 'set_duration', 'start_building', 'set_grid_window', 'remove_participant', 'set_laps', 'set_countdown', 'reset_settings'}


def spawn(coro):
    task = asyncio.create_task(coro)
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    return task


def envelope(kind, data):
    return {'type': kind, 'data': data, 'protocol_version': PROTOCOL_VERSION,
            'revision': room_manager.revision, 'heat_id': room_manager.heat.heat_id if room_manager.heat else None,
            'timestamp_utc': time.time()}


async def close_connection(connection, code=1013):
    connection.closing = True
    with suppress(Exception):
        await asyncio.wait_for(connection.websocket.close(code=code), 2)


def enqueue(connection, payload):
    if connection.closing:
        return
    try:
        connection.queue.put_nowait(payload)
    except asyncio.QueueFull:
        connection.closing = True
        logger.warning('connection_queue_overflow')
        spawn(close_connection(connection))


async def writer_loop(connection):
    try:
        while True:
            payload = await connection.queue.get()
            await asyncio.wait_for(connection.websocket.send_json(payload), 2)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning('connection_send_failed')
        await close_connection(connection)


async def broadcast_event(kind, data):
    payload = envelope(kind, data)
    for connection in tuple(connections.values()):
        enqueue(connection, payload)


async def broadcast_state():
    for client_id, connection in tuple(connections.items()):
        enqueue(connection, envelope('STATE_UPDATE', room_manager.get_state_snapshot(client_id)))


async def execute_race_lifecycle(heat):
    try:
        await broadcast_state()
        await broadcast_event('COUNTDOWN_START', {'countdown_seconds': heat.countdown_seconds,
            'race_start_timestamp_ms': heat.race_start_timestamp_ms})
        await asyncio.sleep(max(0, heat.race_start_timestamp_ms / 1000 - time.time()))
        if room_manager.heat is not heat or room_manager.phase != GamePhase.COUNTDOWN:
            return
        room_manager.start_racing()
        logger.info('heat_started heat_id=%s', heat.heat_id)
        snapshot = room_manager.get_state_snapshot()
        await broadcast_event('RACE_START', {'duration_seconds': heat.duration_seconds, 'laps': heat.laps,
            'race_rules_version': heat.race_rules_version, 'racers': snapshot['active_racers'],
            'track_id': heat.track_id, 'race_start_timestamp_ms': heat.race_start_timestamp_ms})
        await broadcast_state()
        await asyncio.sleep(max(0, heat.race_start_timestamp_ms / 1000 + heat.duration_seconds - time.time()))
        if room_manager.heat is not heat or room_manager.phase != GamePhase.RACING:
            return
        results = room_manager.finish_race()
        logger.info('heat_finished heat_id=%s', heat.heat_id)
        await broadcast_event('RACE_FINISH', results)
        await broadcast_state()
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception('heat_failed')
        room_manager.reset_to_lobby()
        await broadcast_event('ERROR', {'code': 'heat_failed', 'message': 'Heat interrupted; returned to lobby'})
        await broadcast_state()


async def execute_ready_races():
    """Collect ready entrants without making builders or result viewers wait."""
    while True:
        async with command_lock:
            if room_manager.all_configurators_ready():
                room_manager.start_countdown(ready_only=True, countdown_seconds=ALL_READY_COUNTDOWN_SECONDS)
        if room_manager.phase == GamePhase.COUNTDOWN and room_manager.heat:
            await execute_race_lifecycle(room_manager.heat)
            continue
        deadline = room_manager.grid_close_timestamp_ms
        if not deadline:
            return
        await asyncio.sleep(max(0, deadline / 1000 - time.time()))
        async with command_lock:
            if deadline != room_manager.grid_close_timestamp_ms:
                continue
            if not room_manager.start_countdown(ready_only=True):
                await broadcast_state()
                return


def trigger_race_lifecycle():
    global race_task
    if race_task is not None and not race_task.done():
        return
    race_task = spawn(execute_ready_races())


async def restart_grid_timer():
    if room_manager.admission_open:
        await cancel_race()
        if room_manager.all_configurators_ready():
            room_manager.start_countdown(ready_only=True, countdown_seconds=ALL_READY_COUNTDOWN_SECONDS)
        if room_manager.grid_close_timestamp_ms or room_manager.phase == GamePhase.COUNTDOWN:
            trigger_race_lifecycle()


async def cancel_race():
    global race_task
    if race_task and not race_task.done():
        race_task.cancel()
        with suppress(asyncio.CancelledError):
            await race_task
    race_task = None


async def dispatch(command, client_id=None):
    """One policy boundary shared by HTTP and WebSocket commands."""
    action = command.action
    if action in PRESENTER_ACTIONS:
        if not room_manager.admin_pin or not hmac.compare_digest((command.pin or '').encode(), room_manager.admin_pin.encode()):
            logger.warning('command_rejected action=%s reason=unauthorized', action)
            raise HTTPException(403, 'Presenter authentication required')
    elif client_id is None:
        raise HTTPException(400, 'Participant commands require a WebSocket session')
    async with command_lock:
        try:
            if action == 'join':
                room_manager.add_player(client_id, command.nickname or 'Visitor', command.color)
            elif action == 'leave':
                room_manager.remove_player(client_id)
            elif action == 'set_pipeline':
                if any(v is None for v in (command.extractor, command.selector, command.model)):
                    raise ValueError('All three pipeline components are required')
                room_manager.update_pipeline(client_id, command.extractor, command.selector, command.model)
            elif action == 'set_appearance':
                room_manager.update_appearance(client_id, command.vehicle, command.color)
            elif action == 'set_ready':
                if room_manager.set_player_ready(client_id, command.ready):
                    trigger_race_lifecycle()
            elif action == 'race_again':
                room_manager.request_race_again(client_id, command.heat_id)
            elif action == 'start_race':
                if room_manager.admission_open:
                    await cancel_race()
                    if not room_manager.start_countdown(ready_only=True):
                        raise ValueError('No ready racers to start')
                trigger_race_lifecycle()
            elif action == 'reset_lobby':
                await cancel_race()
                room_manager.reset_to_lobby()
            elif action == 'change_track':
                if not command.track_id or not query_engine.get_track(command.track_id):
                    raise ValueError('Unknown eligible track')
                if command.track_id == room_manager.current_track_id:
                    return
                if not room_manager.admission_open:
                    if not command.interrupt_heat:
                        raise ValueError('Applying now ends the current heat. Confirm to continue.')
                    await cancel_race()
                    room_manager.reset_to_lobby()
                if not command.track_id or not room_manager.change_track(command.track_id):
                    raise ValueError('Unknown eligible track')
            elif action == 'start_building':
                room_manager.start_building()
            elif action == 'add_bot':
                if room_manager.add_bot(command.bot_name or command.nickname, command.extractor, command.selector, command.model) is None:
                    raise ValueError('Lobby is full')
            elif action == 'set_duration':
                if command.duration_seconds is None:
                    raise ValueError('Duration is required')
                if command.duration_seconds == room_manager.race_duration_seconds:
                    return
                if not room_manager.admission_open:
                    if not command.interrupt_heat:
                        raise ValueError('Applying now ends the current heat. Confirm to continue.')
                    await cancel_race()
                    room_manager.reset_to_lobby()
                room_manager.race_duration_seconds = command.duration_seconds
                room_manager.touch()
            elif action in ('set_laps', 'set_countdown', 'reset_settings'):
                if action == 'set_laps' and command.laps is None:
                    raise ValueError('Lap count is required')
                if action == 'set_countdown' and command.countdown_seconds is None:
                    raise ValueError('Countdown duration is required')
                if not room_manager.admission_open:
                    if not command.interrupt_heat:
                        raise ValueError('Applying now ends the current heat. Confirm to continue.')
                    await cancel_race()
                    room_manager.reset_to_lobby()
                if action == 'set_laps':
                    room_manager.laps = command.laps
                elif action == 'set_countdown':
                    room_manager.countdown_seconds = command.countdown_seconds
                else:
                    defaults = DEFAULT_SETTINGS
                    if room_manager.current_track_id != defaults['track_id']:
                        room_manager.change_track(defaults['track_id'])
                    room_manager.race_duration_seconds = defaults['race_duration_seconds']
                    room_manager.laps = defaults['laps']
                    room_manager.countdown_seconds = defaults['countdown_seconds']
                    room_manager.set_grid_window(defaults['grid_window_seconds'])
                room_manager.touch()
            elif action == 'set_grid_window':
                if command.grid_window_seconds is None:
                    raise ValueError('Grid entry window is required')
                room_manager.set_grid_window(command.grid_window_seconds)
            elif action == 'remove_participant':
                if not command.target_client_id or command.target_client_id not in room_manager.players:
                    raise ValueError('Participant has already left')
                room_manager.remove_player(command.target_client_id)
                for token, session in list(sessions.items()):
                    if session.client_id == command.target_client_id:
                        sessions.pop(token, None)
                target = connections.get(command.target_client_id)
                if target:
                    enqueue(target, envelope('REMOVED', {'message': 'The presenter removed you from the grid. You can join again.'}))
            elif action == 'ping':
                return
        except ValueError as error:
            logger.info('command_rejected action=%s reason=invalid_state_or_value', action)
            raise HTTPException(409, str(error)) from None
        await restart_grid_timer()
        await broadcast_state()


async def expire_sessions():
    while True:
        await asyncio.sleep(1)
        async with command_lock:
            before = room_manager.revision
            room_manager.expire_disconnected()
            for token, session in list(sessions.items()):
                # Protect frozen entrants through the race; results use the usual
                # reconnect grace so abandoned places cannot block the next grid.
                retained = (not room_manager.admission_open and room_manager.heat
                            and any(e.client_id == session.client_id for e in room_manager.heat.entries))
                if (not retained and session.client_id not in connections and session.disconnected_at is not None
                        and time.monotonic() - session.disconnected_at >= 30):
                    sessions.pop(token, None)
            if room_manager.revision != before:
                await restart_grid_timer()
                await broadcast_state()


@asynccontextmanager
async def lifespan(app):
    pin = os.getenv('DEMO_ADMIN_PIN', '')
    if not 4 <= len(pin) <= 128:
        raise RuntimeError('Set DEMO_ADMIN_PIN to a private 4–128 character presenter code')
    room_manager.admin_pin = pin
    reaper = spawn(expire_sessions())
    logger.info('catalogue_loaded build_id=%s tracks=%s', query_engine.build_id, len(query_engine.list_tracks()))
    try:
        yield
    finally:
        await cancel_race()
        for connection in list(connections.values()):
            await close_connection(connection, 1001)
        tasks = list(background_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        connections.clear()
        sessions.clear()
        room_manager.players.clear()
        room_manager.reset_to_lobby()
        room_manager.admin_pin = ''


app = FastAPI(title='TEMPO Display Day Demo', version='2.0.0', lifespan=lifespan)


@app.get('/api/tracks')
async def list_tracks():
    return {'tracks': [asdict(t) for t in query_engine.list_tracks()], 'current_track_id': room_manager.current_track_id}


@app.get('/api/options/{track_id}')
async def get_track_options(track_id: str):
    if not query_engine.get_track(track_id):
        raise HTTPException(404, 'Unknown track')
    return {'track_id': track_id, 'options': query_engine.get_valid_options(track_id)}


def serialize(c):
    from demo.game_state import metrics_dict
    return dict(metrics_dict(c), key=c.key, latency_s=c.total_latency_s, ram_mb=c.extraction_ram_mb)


@app.get('/api/pareto/{track_id}')
async def get_track_pareto_data(track_id: str):
    if not query_engine.get_track(track_id):
        raise HTTPException(404, 'Unknown track')
    return {'track_id': track_id, 'all_points': list(map(serialize, query_engine.get_all_configs(track_id))),
            'pareto_frontier': list(map(serialize, query_engine.get_pareto_frontier(track_id)))}


@app.get('/api/leaderboard/{track_id}')
async def get_track_leaderboard(track_id: str, sort_by: Literal['pareto', 'speed', 'score', 'accuracy', 'ram']='pareto', limit: int=Query(10, ge=1, le=1000)):
    if not query_engine.get_track(track_id):
        raise HTTPException(404, 'Unknown track')
    return {'track_id': track_id, 'leaderboard': [dict(**serialize(c), speedup=c.speedup_vs_tsfresh, pareto_score=c.pareto_composite_score)
            for c in query_engine.get_leaderboard(track_id, sort_by, limit)]}


@app.get('/api/state')
async def get_state():
    return room_manager.get_state_snapshot()


class AdminCredentials(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    pin: str = Field(min_length=1, max_length=128)


@app.post('/api/admin/unlock')
async def unlock_admin(body: AdminCredentials):
    if not room_manager.admin_pin or not hmac.compare_digest(body.pin.encode(), room_manager.admin_pin.encode()):
        raise HTTPException(403, 'Incorrect password')
    return {'status': 'ok'}


@app.post('/api/admin/state')
async def get_presenter_state(body: AdminCredentials):
    await unlock_admin(body)
    return room_manager.presenter_snapshot()


@app.post('/api/admin/action')
async def execute_admin_action(body: Command):
    await dispatch(body)
    return {'status': 'ok', 'message': f'{body.action} completed', 'state': room_manager.presenter_snapshot()}


@app.websocket('/ws/game')
async def websocket_game_endpoint(websocket: WebSocket):
    origin = websocket.headers.get('origin')
    if origin and urlsplit(origin).netloc != websocket.headers.get('host'):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    connection, client_id = None, None
    try:
        try:
            hello = Hello.model_validate(await asyncio.wait_for(websocket.receive_json(), 5))
            if hello.protocol_version != PROTOCOL_VERSION:
                raise ValueError('version')
        except (ValueError, ValidationError, asyncio.TimeoutError):
            await asyncio.wait_for(websocket.send_json(envelope('ERROR', {'code': 'reload_required', 'message': 'Reload the page to use protocol version 2'})), 2)
            await websocket.close(code=1008)
            return
        session = sessions.get(hello.resume_token)
        if session is not None and session.disconnected_at is not None:
            retained = (not room_manager.admission_open and room_manager.heat
                        and any(e.client_id == session.client_id for e in room_manager.heat.entries))
            if not retained and time.monotonic() - session.disconnected_at >= 30:
                room_manager.remove_player(session.client_id)
                sessions.pop(session.token, None)
                session = None
        resumed = session is not None
        if session is None:
            session = Session('client_' + secrets.token_hex(12), secrets.token_urlsafe(32))
            sessions[session.token] = session
        client_id = session.client_id
        old = connections.get(client_id)
        connection = Connection(websocket)
        connections[client_id] = connection
        if old:
            spawn(close_connection(old, 1000))
        async with command_lock:
            session.disconnected_at = None
            room_manager.reconnect(client_id)
            await restart_grid_timer()
            connection.writer = spawn(writer_loop(connection))
            enqueue(connection, envelope('INIT', {'client_id': client_id, 'resume_token': session.token, 'state': room_manager.get_state_snapshot(client_id)}))
            if resumed:
                await broadcast_state()
        while True:
            try:
                command = Command.model_validate(await websocket.receive_json())
                if connections.get(client_id) is not connection:
                    break
                if command.action == 'ping':
                    enqueue(connection, envelope('PONG', {}))
                else:
                    await dispatch(command, client_id)
            except (ValidationError, ValueError):
                enqueue(connection, envelope('ERROR', {'code': 'invalid_command', 'message': 'Invalid command fields'}))
            except HTTPException as error:
                enqueue(connection, envelope('ERROR', {'code': str(error.status_code), 'message': error.detail}))
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception('connection_handler_failed')
    finally:
        if connection:
            if connection.writer:
                connection.writer.cancel()
                with suppress(asyncio.CancelledError):
                    await connection.writer
            async with command_lock:
                if connections.get(client_id) is connection:
                    connections.pop(client_id, None)
                    session.disconnected_at = time.monotonic()
                    room_manager.disconnect(client_id)
                    await restart_grid_timer()
                    await broadcast_state()



@app.middleware('http')
async def frontend_cache_policy(request, call_next):
    response = await call_next(request)
    if request.url.path in ('/', '/admin') or request.url.path.startswith(('/static/', '/api/')):
        response.headers['Cache-Control'] = 'no-store'
    return response


@app.get('/assets/{version}/{asset_path:path}')
async def serve_frontend_asset(version: str, asset_path: str):
    if version != FRONTEND_VERSION or asset_path not in FRONTEND_FILES or asset_path.endswith('.html'):
        raise HTTPException(404, 'Frontend version unavailable; refresh the page')
    media_type = mimetypes.guess_type(asset_path)[0] or 'application/octet-stream'
    return Response(FRONTEND_FILES[asset_path], media_type=media_type,
                    headers={'Cache-Control': 'public, max-age=31536000, immutable'})


# Retain old URLs for compatibility, but new pages use a versioned module graph.
app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')


@app.get('/')
async def serve_index():
    html = FRONTEND_FILES['index.html'].decode().replace('/static/', f'/assets/{FRONTEND_VERSION}/')
    return HTMLResponse(html, headers={'Cache-Control': 'no-store'})


@app.get('/admin')
async def serve_presenter():
    html = FRONTEND_FILES['presenter.html'].decode().replace('/static/', f'/assets/{FRONTEND_VERSION}/')
    return HTMLResponse(html, headers={'Cache-Control': 'no-store'})
