"""Transport authorization, session recovery, and real two-client heat tests."""
import asyncio
from dataclasses import replace
import pytest
from starlette.testclient import TestClient
import demo.server as server
from demo.server import app, room_manager

@pytest.fixture(autouse=True)
def reset(monkeypatch):
    monkeypatch.setenv('DEMO_ADMIN_PIN', 'test-presenter')
    room_manager.players.clear()
    room_manager.reset_to_lobby()
    room_manager.change_track('beed')
    room_manager.countdown_seconds = .01
    room_manager.race_duration_seconds = 5


def hello(ws, token=None):
    ws.send_json({'action':'hello','protocol_version':2,'resume_token':token})
    return receive(ws,'INIT')['data']


def receive(ws, kind):
    for _ in range(50):
        msg=ws.receive_json()
        if msg['type']==kind:return msg
    raise AssertionError(f'No {kind} event')


def admin(client, action, **kwargs):
    return client.post('/api/admin/action',json={'action':action,'pin':'test-presenter',**kwargs})


def test_http_read_routes_and_validation():
    with TestClient(app) as client:
        assert 'TEMPO' in client.get('/').text
        assert client.get('/admin').status_code==200
        for path in ('/api/state','/api/tracks','/api/options/beed','/api/pareto/beed','/api/leaderboard/beed'):
            assert client.get(path).status_code==200
        assert client.get('/api/options/missing').status_code==404
        assert client.get('/api/leaderboard/beed?limit=-1').status_code==422
        assert client.get('/api/leaderboard/beed?sort_by=invalid').status_code==422

@pytest.mark.parametrize('action',sorted(server.PRESENTER_ACTIONS))
def test_authorization_both_transports(action):
    with TestClient(app) as client:
        before=room_manager.get_state_snapshot()['revision']
        assert client.post('/api/admin/action',json={'action':action,'pin':'wrong'}).status_code==403
        with client.websocket_connect('/ws/game') as ws:
            hello(ws)
            ws.send_json({'action':action})
            assert receive(ws,'ERROR')['data']['code']=='403'
            assert room_manager.revision==before


def test_presenter_actions():
    with TestClient(app) as client:
        assert admin(client,'add_bot').status_code==200
        assert admin(client,'change_track',track_id='har').status_code==200
        assert admin(client,'set_duration',duration_seconds=6.0).status_code==200
        assert admin(client,'reset_lobby').status_code==200
        assert not room_manager.players
        assert admin(client,'set_duration',duration_seconds=100).status_code==422


def test_bad_protocol_and_messages():
    with TestClient(app) as client:
        with client.websocket_connect('/ws/game') as ws:
            ws.send_json({'action':'hello','protocol_version':1})
            assert receive(ws,'ERROR')['data']['code']=='reload_required'
        with client.websocket_connect('/ws/game') as ws:
            hello(ws)
            for payload in ([], {'action':'join','nickname':23}, {'action':'join','color':'red'}, {'action':'set_ready','ready':'false'}):
                ws.send_json(payload)
                assert receive(ws,'ERROR')['data']['code']=='invalid_command'
            ws.send_text('not json')
            assert receive(ws,'ERROR')['data']['code']=='invalid_command'
            ws.send_json({'action':'ping'})
            assert receive(ws,'PONG')


def test_pipeline_rejection_preserves_state():
    with TestClient(app) as client:
        with client.websocket_connect('/ws/game') as ws:
            hello(ws)
            ws.send_json({'action':'join','nickname':'<b>test</b>'})
            state=receive(ws,'STATE_UPDATE')['data']
            prior=state['active_racers'][0]
            ws.send_json({'action':'set_pipeline','extractor':'invalid','selector':'None','model':'random_forest'})
            assert receive(ws,'ERROR')['data']['code']=='409'
            assert room_manager.get_state_snapshot()['active_racers'][0]['metrics']==prior['metrics']
            assert prior['nickname']=='<b>test</b>'  # rendering must use textContent
            ws.send_json({'action':'leave'})
            assert receive(ws,'STATE_UPDATE')['data']['active_racers']==[]


def test_lobby_resume_and_socket_replacement():
    with TestClient(app) as client:
        with client.websocket_connect('/ws/game') as ws:
            initial=hello(ws)
            ws.send_json({'action':'join','nickname':'Alice'})
            receive(ws,'STATE_UPDATE')
            with client.websocket_connect('/ws/game') as replacement:
                restored=hello(replacement,initial['resume_token'])
                assert restored['client_id']==initial['client_id']
                receive(replacement,'STATE_UPDATE')
                replacement.send_json({'action':'ping'})
                assert receive(replacement,'PONG')
        with client.websocket_connect('/ws/game') as resumed:
            restored=hello(resumed,initial['resume_token'])
            assert len(restored['state']['active_racers'])==1
            assert restored['state']['active_racers'][0]['connected']
            assert restored['client_id']==initial['client_id']


def test_real_heat_two_clients_reconnect_results():
    with TestClient(app) as client:
        with client.websocket_connect('/ws/game') as display:
            hello(display)
            with client.websocket_connect('/ws/game') as phone:
                identity=hello(phone)
                phone.send_json({'action':'join','nickname':'Alice'})
                receive(phone,'STATE_UPDATE');receive(display,'STATE_UPDATE')
                phone.send_json({'action':'set_ready','ready':True})
                race=receive(display,'RACE_START')
                heat_id=race['heat_id']
                assert race['data']['race_start_timestamp_ms']>0
                assert admin(client,'start_race').status_code==200
                assert room_manager.heat.heat_id==heat_id
                assert admin(client,'change_track',track_id='har').status_code==409
            with client.websocket_connect('/ws/game') as phone:
                restored=hello(phone,identity['resume_token'])
                assert restored['state']['heat_id']==heat_id
                assert restored['state']['phase']=='RACING'
                finished=receive(phone,'RACE_FINISH')['data']
                assert any(p['client_id']==identity['client_id'] for p in finished['rankings'])
                assert len(finished['rankings'])==2
                assert receive(display,'RACE_FINISH')['data']==finished


def test_reset_cancels_task():
    with TestClient(app) as client:
        assert admin(client,'start_race').status_code==200
        old_task=server.race_task
        assert admin(client,'reset_lobby').status_code==200
        assert old_task.done()
        assert room_manager.heat is None and room_manager.phase.value=='LOBBY'
        assert admin(client,'start_race').status_code==200
        assert server.race_task is not old_task
    assert not server.background_tasks and not server.connections


def test_slow_client_isolation_and_queue_overflow():
    class Socket:
        def __init__(self,slow=False):self.slow,self.sent,self.closed=slow,[],False
        async def send_json(self,payload):
            if self.slow:await asyncio.Event().wait()
            self.sent.append(payload)
        async def close(self,code=1000):self.closed=True
    async def check():
        slow,fast=server.Connection(Socket(True)),server.Connection(Socket())
        server.connections.update(slow=slow,fast=fast)
        tasks=[asyncio.create_task(server.writer_loop(c)) for c in (slow,fast)]
        try:
            await server.broadcast_event('TEST',{})
            await asyncio.sleep(.02)
            assert fast.websocket.sent and not slow.websocket.sent
            for _ in range(40):server.enqueue(slow,server.envelope('TEST',{}))
            await asyncio.sleep(.02)
            assert slow.closing and slow.websocket.closed
        finally:
            server.connections.clear()
            for t in tasks:t.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)
            if server.background_tasks:await asyncio.gather(*list(server.background_tasks),return_exceptions=True)
    asyncio.run(check())


def test_pin_required(monkeypatch):
    monkeypatch.delenv('DEMO_ADMIN_PIN')
    with pytest.raises(RuntimeError,match='DEMO_ADMIN_PIN'):
        with TestClient(app):pass


def test_send_timeout_closes_only_slow_socket():
    class Socket:
        closed=False
        async def send_json(self,payload):await asyncio.Event().wait()
        async def close(self,code=1000):self.closed=True
    async def check():
        connection=server.Connection(Socket())
        server.enqueue(connection,server.envelope('TEST',{}))
        await asyncio.wait_for(server.writer_loop(connection),3)
        assert connection.websocket.closed and connection.closing
    asyncio.run(check())


def test_resume_token_is_private_and_unicode_pin_is_rejected_cleanly():
    with TestClient(app) as client:
        with client.websocket_connect('/ws/game') as ws:
            data=hello(ws)
            ws.send_json({'action':'join','nickname':'Alice'})
            assert data['resume_token'] not in str(receive(ws,'STATE_UPDATE'))
            ws.send_json({'action':'reset_lobby','pin':'wrong-🔑'})
            assert receive(ws,'ERROR')['data']['code']=='403'
            ws.send_json({'action':'ping'})
            assert receive(ws,'PONG')


def test_participant_race_again_without_presenter_code():
    with TestClient(app) as client:
        with client.websocket_connect('/ws/game') as ws:
            hello(ws)
            ws.send_json({'action': 'join', 'nickname': 'Alice'})
            receive(ws, 'STATE_UPDATE')
            room_manager.start_countdown()
            room_manager.start_racing()
            room_manager.finish_race()
            heat_id = room_manager.heat.heat_id
            ws.send_json({'action': 'race_again', 'heat_id': heat_id})
            state = receive(ws, 'STATE_UPDATE')['data']
            assert state['phase'] == 'RESULTS'
            assert state['active_racers'][0]['preparing_next']
            assert state['active_racers'][0]['nickname'] == 'Alice'
            assert state['heat_id'] == heat_id
            ws.send_json({'action': 'reset_lobby'})
            assert receive(ws, 'ERROR')['data']['code'] == '403'


def test_admin_unlock_validates_without_changing_room():
    with TestClient(app) as client:
        before = room_manager.revision
        assert client.post('/api/admin/unlock', json={'pin': 'wrong'}).status_code == 403
        assert client.post('/api/admin/unlock', json={'pin': 'test-presenter'}).status_code == 200
        assert room_manager.revision == before
        assert client.post('/api/admin/action', json={'action': 'change_track', 'track_id': 'har'}).status_code == 403


def test_frontend_versions_cover_transitive_imports_and_disable_html_cache(tmp_path):
    import re
    from urllib.parse import urljoin
    with TestClient(app) as client:
        root = client.get('/')
        assert root.headers['cache-control'] == 'no-store'
        script = re.search(r'src="([^"]+/js/app.js)"', root.text).group(1)
        assert script.startswith(f'/assets/{server.FRONTEND_VERSION}/')
        pending, seen = [script], set()
        while pending:
            path = pending.pop()
            if path in seen:
                continue
            seen.add(path)
            response = client.get(path)
            assert response.status_code == 200
            assert 'immutable' in response.headers['cache-control']
            for dependency in re.findall(r'''from ['"](\./[^'"]+)['"]''', response.text):
                pending.append(urljoin(path, dependency))
        track = f'/assets/{server.FRONTEND_VERSION}/js/track.js'
        assert track in seen
        assert 'CIRCUIT_HEIGHT = 1400' in client.get(track).text
        css = re.search(r'href="([^"]+/css/style.css)"', root.text).group(1)
        assert client.get(css).status_code == 200
        assert client.get('/admin').headers['cache-control'] == 'no-store'
        assert client.get('/static/js/track.js').headers['cache-control'] == 'no-store'
        assert client.get('/assets/old-version/js/track.js').status_code == 404
    (tmp_path / 'track.js').write_text('old geometry')
    first_version, frozen = server.snapshot_frontend(tmp_path)
    (tmp_path / 'track.js').write_text('new geometry')
    second_version, _ = server.snapshot_frontend(tmp_path)
    assert first_version != second_version
    assert frozen['track.js'] == b'old geometry'


def test_six_racers_queue_promotion_and_local_next_round():
    from contextlib import ExitStack
    previous_capacity = room_manager.max_active_racers
    room_manager.max_active_racers = 6

    def state_until(ws, predicate):
        for _ in range(100):
            event = ws.receive_json()
            if event['type'] == 'STATE_UPDATE' and predicate(event['data']):
                return event['data']
        raise AssertionError('Expected room update was not delivered')

    try:
        with TestClient(app) as client, ExitStack() as stack:
            sockets = []
            ids = []
            for i in range(7):
                ws = stack.enter_context(client.websocket_connect('/ws/game'))
                init = hello(ws)
                ids.append(init['client_id'])
                ws.send_json({'action': 'join', 'nickname': f'Racer {i}'})
                state_until(ws, lambda s, cid=ids[-1]: any(p['client_id'] == cid for p in s['active_racers'] + s['spectators']))
                sockets.append(ws)
            snapshot = room_manager.get_state_snapshot()
            assert len(snapshot['active_racers']) == 6
            assert snapshot['spectators'][0]['client_id'] == ids[6]
            for ws in sockets[:6]:
                ws.send_json({'action': 'set_ready', 'ready': True})
            results = receive(sockets[0], 'RACE_FINISH')['data']
            assert len(results['rankings']) == 6
            sockets[0].send_json({'action': 'race_again', 'heat_id': results['heat_id']})
            state = state_until(sockets[0], lambda s: any(p['client_id'] == ids[0] and p.get('preparing_next') for p in s['active_racers']))
            assert state['phase'] == 'RESULTS'
            assert not next(p for p in state['active_racers'] if p['client_id'] == ids[1])['preparing_next']
            assert admin(client, 'reset_lobby').status_code == 200
            sockets[5].send_json({'action': 'leave'})
            promoted = state_until(sockets[6], lambda s: any(p['client_id'] == ids[6] for p in s['active_racers']))
            assert not promoted['spectators']
    finally:
        room_manager.max_active_racers = previous_capacity
