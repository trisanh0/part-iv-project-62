"""Room invariants, immutable heats, and reconnection behaviour."""
from dataclasses import FrozenInstanceError
import pytest
from demo.query import DemoQueryEngine
from demo.game_state import BoothRoomManager, GamePhase

@pytest.fixture
def room():
    return BoothRoomManager(DemoQueryEngine())

def start(room):
    room.start_countdown()
    room.start_racing()

def test_initial_state(room):
    state = room.get_state_snapshot()
    assert state['phase'] == 'LOBBY'
    assert state['race_duration_seconds'] == 27
    assert state['countdown_seconds'] == 7
    assert state['protocol_version'] == 2
    assert state['build_id'] == room.engine.build_id
    assert 'phase_start_time_utc' in state
    assert state['track']['track_id'] == 'gas-sensor-drift-sub'


def test_guided_builds_have_real_measurements(room):
    from itertools import product
    for extractor, selector, model in product(
        ['numpy_statistical', 'numba_efficient', 'tsfresh_minimal'],
        ['None', 'select_k_best_anova', 'subsampled_fdr'],
        ['logistic_regression', 'random_forest'],
    ):
        c = room.engine.get_config(room.current_track_id, extractor, selector, model)
        assert c is not None and c.sample_count > 0
        assert 0 < c.n_selected_features <= c.n_extracted_features


def test_accuracy_changes_winner_and_schedule_survives_disconnect(room):
    room.add_player('fast', 'Fast')
    room.add_player('accurate', 'Accurate')
    room.update_pipeline('fast', 'numpy_statistical', 'None', 'logistic_regression')
    room.update_pipeline('accurate', 'numba_efficient', 'None', 'logistic_regression')
    start(room)
    snapshot = room.get_state_snapshot()
    schedules = {p['client_id']: p['race_schedule'] for p in snapshot['active_racers']}
    assert schedules['fast']['driving_ms'] < schedules['accurate']['driving_ms']
    assert schedules['fast']['pit_ms'] > schedules['accurate']['pit_ms']
    assert schedules['fast']['finish_ms'] > schedules['accurate']['finish_ms']
    assert max(s['finish_ms'] for s in schedules.values()) == 27000
    room.disconnect('fast')
    room.reconnect('fast')
    assert room.get_state_snapshot()['active_racers'] == snapshot['active_racers']
    results = room.finish_race()
    assert results['rankings'][0]['client_id'] == 'accurate'
    assert results['race_rules_version'] == snapshot['race_rules_version']
    assert all(r['race_schedule'] == schedules[r['client_id']] for r in results['rankings'])


@pytest.mark.parametrize('duration', [5, 27, 60])
def test_six_schedules_share_ties_and_exact_deadline(duration):
    room = BoothRoomManager(DemoQueryEngine(), max_active_racers=6, race_duration_seconds=duration)
    for i in range(6):
        room.add_player(str(i), 'Twenty character nam')
    start(room)
    for schedule in room.heat.schedules:
        assert schedule.driving_ms + schedule.pit_ms == schedule.finish_ms == duration * 1000
        assert 0 < schedule.pit_entry_ms < schedule.pit_stop_ms
        assert schedule.pit_stop_ms <= schedule.pit_departure_ms < schedule.pit_exit_ms < schedule.finish_ms
    rows = room.finish_race()['rankings']
    assert all(row['finish_position'] == 1 and row['tied'] for row in rows)


def test_perfect_accuracy_and_nonclassification_rules(room):
    from dataclasses import replace
    from demo.race import build_schedules
    room.add_player('a', 'A')
    start(room)
    entry = room.heat.entries[0]
    perfect = replace(entry, metrics=replace(entry.metrics, score=1))
    schedule, = build_schedules((perfect,), 27)
    assert schedule.uses_accuracy and schedule.pit_ms == 0
    for task, score in [('regression', -15), ('forecasting', 500)]:
        schedule, = build_schedules((replace(entry, metrics=replace(entry.metrics, task_type=task, score=score)),), 27)
        assert not schedule.uses_accuracy and schedule.pit_ms == 0
        assert schedule.finish_ms == 27000

def test_player_registration_and_idempotency(room):
    p = room.add_player('a', ' Alice ')
    assert p.nickname == 'Alice' and p.car_index == 0 and p.metrics
    assert room.add_player('a', 'Again') is p
    assert len(room.players) == 1

@pytest.mark.parametrize('slot', range(4))
def test_capacity_and_spectator_promotion(room, slot):
    for i in range(5):
        room.add_player(str(i), f'Player {i}')
    assert room.players['4'].is_spectator
    room.remove_player(str(slot))
    assert not room.players['4'].is_spectator
    assert room.players['4'].car_index == slot
    assert sorted(p.car_index for p in room.players.values()) == [0, 1, 2, 3]

def test_pipeline_validation_and_readiness(room):
    p = room.add_player('a', 'A')
    room.add_player('b', 'B')
    assert not room.set_player_ready('a')
    old = p.metrics
    with pytest.raises(ValueError, match='Unsupported'):
        room.update_pipeline('a', 'missing', 'None', 'random_forest')
    assert p.metrics is old and p.is_ready
    room.update_pipeline('a', **p.pipeline)
    assert not p.is_ready
    room.set_player_ready('a')
    assert room.set_player_ready('b')
    assert room.phase == GamePhase.COUNTDOWN

def test_freeze_disconnect_mutation_and_duplicate_start(room):
    p = room.add_player('a', 'A')
    start(room)
    heat = room.heat
    assert not room.start_countdown()
    with pytest.raises(ValueError):
        room.update_pipeline('a', **p.pipeline)
    with pytest.raises(ValueError):
        room.change_track('har')
    with pytest.raises(FrozenInstanceError):
        heat.entries[0].nickname = 'changed'
    room.remove_player('a')
    results = room.finish_race()
    assert any(r['client_id'] == 'a' for r in results['rankings'])
    assert results['heat_id'] == heat.heat_id
    assert room.finish_race() is results

def test_bot_configuration_is_real(room):
    b = room.add_bot()
    assert b.is_bot and b.is_ready and b.metrics
    with pytest.raises(ValueError):
        room.add_bot(extractor='missing')

def test_all_tracks_defaults_and_switch(room):
    p = room.add_player('a', 'A')
    for track in room.engine.list_tracks():
        assert room.change_track(track.track_id)
        assert p.metrics and p.metrics.track_id == track.track_id
        assert not p.is_ready
    assert not room.change_track('missing')

def test_shared_awards_and_scoring(room):
    room.add_player('a', 'A')
    room.add_player('b', 'B')
    start(room)
    results = room.finish_race()
    a, b = results['rankings']
    assert a['pareto_score'] == b['pareto_score']
    assert set(a['awards']) == set(b['awards']) == {'speed_champion','accuracy_champion','green_compute_champion','pareto_champion'}
    assert a['client_id'] == 'a'
    assert a['pareto_score'] == room.heat.entries[0].metrics.pareto_composite_score

def test_disconnect_reservation_expiration_and_resume(room):
    for i in range(5): room.add_player(str(i), str(i))
    room.disconnect('0')
    then = room.players['0'].disconnected_at
    room.expire_disconnected(then + 29)
    assert '0' in room.players and room.players['4'].is_spectator
    room.reconnect('0')
    assert room.players['0'].connected and not room.players['0'].is_ready
    room.disconnect('0')
    room.expire_disconnected(room.players['0'].disconnected_at + 30)
    assert '0' not in room.players and not room.players['4'].is_spectator

def test_disconnected_slot_blocks_start(room):
    room.add_player('a', 'A'); room.add_player('b', 'B')
    room.disconnect('b')
    assert not room.set_player_ready('a')
    with pytest.raises(ValueError, match='connected'):room.start_countdown()

def test_reset_and_expiry_after_heat(room):
    room.add_player('a', 'A'); start(room)
    room.disconnect('a')
    room.players['a'].disconnected_at -= 31
    room.expire_disconnected()
    assert 'a' in room.players
    room.finish_race(); room.reset_to_lobby()
    assert not room.players and room.heat is None

def test_invalid_identity(room):
    for name in (' ', 'a' * 21):
        with pytest.raises(ValueError):room.add_player('a', name)
    with pytest.raises(ValueError):room.add_player('a', 'A', 'red" onload="bad')


def test_race_again_only_current_completed_heat_entrant(room):
    room.add_player('a', 'Alice')
    start(room)
    heat_id = room.heat.heat_id
    with pytest.raises(ValueError, match='completed heat'):
        room.request_race_again('a', heat_id)
    room.finish_race()
    with pytest.raises(ValueError, match='entrant'):
        room.request_race_again('viewer', heat_id)
    with pytest.raises(ValueError, match='completed heat'):
        room.request_race_again('a', 'stale-heat')
    room.request_race_again('a', heat_id)
    assert room.phase == GamePhase.RESULTS
    assert room.heat.heat_id == heat_id
    assert room.players['a'].preparing_next
    assert not room.players['a'].is_ready
    start(room)
    room.finish_race()
    with pytest.raises(ValueError, match='completed heat'):
        room.request_race_again('a', heat_id)


def test_next_round_preparation_is_per_player(room):
    room.add_player('a', 'Alice')
    room.add_player('b', 'Bob')
    start(room)
    results = room.finish_race()
    heat_id = room.heat.heat_id
    room.request_race_again('a', heat_id)
    room.update_pipeline('a', **room.players['a'].pipeline)
    assert not room.set_player_ready('a')
    assert room.phase == GamePhase.RESULTS
    assert room.race_results_cache is results
    assert not room.players['b'].preparing_next
    with pytest.raises(ValueError):
        room.update_pipeline('b', **room.players['b'].pipeline)
    room.request_race_again('b', heat_id)
    assert room.set_player_ready('b')
    assert room.phase == GamePhase.COUNTDOWN
    assert room.heat.heat_id != heat_id
