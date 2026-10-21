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
        room.set_player_ready(str(i))
    assert room.players['4'].is_spectator
    room.remove_player(str(slot))
    assert not room.players['4'].is_spectator
    assert room.players['4'].car_index == slot
    assert sorted(p.car_index for p in room._ready_grid()) == [0, 1, 2, 3]


def test_pipeline_validation_and_readiness(room):
    p = room.add_player('a', 'A')
    room.add_player('b', 'B')
    assert room.set_player_ready('a')
    assert room.grid_close_timestamp_ms > 0
    old = p.metrics
    with pytest.raises(ValueError, match='Unsupported'):
        room.update_pipeline('a', 'missing', 'None', 'random_forest')
    assert p.metrics is old and p.is_ready
    room.update_pipeline('a', **p.pipeline)
    assert not p.is_ready
    room.set_player_ready('a')
    assert room.set_player_ready('b')
    assert room.start_countdown(ready_only=True)
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
    for i in range(5):
        room.add_player(str(i), str(i))
        room.set_player_ready(str(i))
    assert room.players['4'].is_spectator
    room.disconnect('0')
    then = room.players['0'].disconnected_at
    assert not room.players['4'].is_spectator
    assert not room.players['0'].is_ready
    room.expire_disconnected(then + 29)
    assert '0' in room.players
    room.reconnect('0')
    assert room.players['0'].connected and not room.players['0'].is_ready
    room.disconnect('0')
    room.expire_disconnected(room.players['0'].disconnected_at + 30)
    assert '0' not in room.players


def test_disconnected_slot_blocks_start(room):
    room.add_player('a', 'A'); room.add_player('b', 'B')
    room.set_player_ready('a')
    room.set_player_ready('b')
    room.disconnect('b')
    assert room.start_countdown(ready_only=True)
    assert {p.client_id for p in room.heat.entries if not p.is_bot} == {'a'}


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
    with pytest.raises(ValueError, match='completed heat'):
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
    room.add_player('a', 'Alice'); room.add_player('b', 'Bob')
    start(room)
    results = room.finish_race()
    heat_id = room.heat.heat_id
    assert room.get_state_snapshot('b')['participant']['view'] == 'results'
    assert not room.get_state_snapshot()['active_racers']
    room.request_race_again('a', heat_id)
    room.update_pipeline('a', **room.players['a'].pipeline)
    assert room.set_player_ready('a')
    assert room.start_countdown(ready_only=True)
    assert {p.client_id for p in room.heat.entries if not p.is_bot} == {'a'}
    assert room.get_state_snapshot('b')['personal_results'] is results
    assert room.get_state_snapshot('b')['participant']['view'] == 'results'
    room.request_race_again('b', heat_id)
    assert room.get_state_snapshot('b')['participant']['view'] == 'configuration'
    room.update_pipeline('b', **room.players['b'].pipeline)


def test_arrival_during_results_can_build_without_resetting_other_screens(room):
    room.add_player('a', 'Alice')
    start(room)
    results = room.finish_race()
    newcomer = room.add_player('b', 'Bob')
    assert not newcomer.is_spectator and newcomer.preparing_next
    assert not newcomer.is_ready
    assert room.phase == GamePhase.RESULTS
    assert not room.players['a'].preparing_next
    assert room.race_results_cache is results
    room.update_pipeline('b', **newcomer.pipeline)
    assert room.set_player_ready('b')
    room.request_race_again('a', room.heat.heat_id)
    assert room.set_player_ready('a')
    room.start_countdown(ready_only=True)
    assert {entry.client_id for entry in room.heat.entries} == {'a', 'b'}


@pytest.mark.parametrize('join_phase', ['COUNTDOWN', 'RACING'])
def test_queue_opens_automatically_at_finish_and_preserves_fifo(room, join_phase):
    room.add_player('a', 'Alice'); room.add_player('b', 'Bob')
    room.start_countdown()
    if join_phase == 'RACING': room.start_racing()
    waiting = []
    for i in range(5):
        p = room.add_player(str(i), f'Waiting {i}')
        assert not p.is_spectator
        room.set_player_ready(p.client_id)
        waiting.append(p)
    assert all(p.is_spectator for p in waiting)
    if join_phase == 'COUNTDOWN': room.start_racing()
    results = room.finish_race()
    assert all(not p.is_spectator for p in waiting[:4])
    assert waiting[4].is_spectator
    assert room.players['a'].reviewing_results
    room.remove_player('0')
    assert not waiting[4].is_spectator
    assert room.race_results_cache is results
    assert len(results['rankings']) == 2


def test_results_release_bot_places_and_allow_a_repeat_solo_race(room):
    room.add_player('a', 'Alice')
    start(room)
    bot_id = next(p.client_id for p in room.heat.entries if p.is_bot)
    results = room.finish_race()
    assert bot_id not in room.players
    assert any(row['client_id'] == bot_id for row in results['rankings'])
    room.request_race_again('a', room.heat.heat_id)
    assert room.set_player_ready('a')
    room.start_countdown(ready_only=True)
    assert len(room.heat.entries) == 2
    assert any(p.is_bot for p in room.heat.entries)


def test_expired_race_reservations_release_at_finish_without_a_presenter_reset(room):
    for i in range(4):
        room.add_player(str(i), f'Racer {i}')
    start(room)
    waiting = room.add_player('next', 'Next racer')
    room.set_player_ready('next')
    room.disconnect('0')
    room.players['0'].disconnected_at -= 31
    room.expire_disconnected()
    assert '0' in room.players and waiting.is_spectator
    results = room.finish_race()
    assert '0' not in room.players
    assert not waiting.is_spectator and waiting.preparing_next
    assert any(row['client_id'] == '0' for row in results['rankings'])


def test_results_reconnect_grace_expires_and_promotes_waiters(room):
    for i in range(4): room.add_player(str(i), f'Racer {i}')
    start(room); room.finish_race()
    assert not room._ready_grid()
    for i in range(5):
        p = room.add_player(f'new{i}', f'Next {i}')
        room.set_player_ready(p.client_id)
    assert room.players['new4'].is_spectator
    room.disconnect('new0')
    assert not room.players['new4'].is_spectator
    disconnected = room.players['new0'].disconnected_at
    room.expire_disconnected(disconnected + 29)
    assert 'new0' in room.players
    room.expire_disconnected(disconnected + 30)
    assert 'new0' not in room.players



def test_only_ready_connected_people_affect_the_next_race(room):
    for i in range(12):
        room.add_player(str(i), f'Visitor {i}')
    assert not room._ready_grid() and not room.grid_close_timestamp_ms
    assert all(not p.is_spectator for p in room.players.values())
    room.set_player_ready('0')
    deadline = room.grid_close_timestamp_ms
    room.set_player_ready('1')
    assert room.grid_close_timestamp_ms == deadline
    room.disconnect('1')
    assert room.start_countdown(ready_only=True)
    assert {p.client_id for p in room.heat.entries if not p.is_bot} == {'0'}
    assert room.get_state_snapshot('2')['participant']['view'] == 'configuration'
    assert room.get_state_snapshot('1')['participant']['is_ready'] is False


def test_undo_and_disconnect_cancel_empty_entry_window(room):
    room.add_player('a', 'A')
    room.set_player_ready('a')
    assert room.grid_close_timestamp_ms
    room.set_player_ready('a', False)
    assert room.grid_close_timestamp_ms == 0
    room.set_player_ready('a')
    room.disconnect('a')
    assert room.grid_close_timestamp_ms == 0
    assert not room.start_countdown(ready_only=True)


def test_personal_results_survive_another_heat_and_reentry_is_explicit(room):
    room.add_player('a', 'A'); room.add_player('b', 'B')
    start(room)
    first = room.finish_race()
    room.request_race_again('b', first['heat_id'])
    room.set_player_ready('b')
    room.start_countdown(ready_only=True)
    assert room.get_state_snapshot('a')['participant']['view'] == 'results'
    assert room.get_state_snapshot('a')['personal_results'] is first
    assert room.get_state_snapshot('b')['participant']['view'] == 'race'
    room.start_racing()
    second = room.finish_race()
    assert second['heat_id'] != first['heat_id']
    assert room.get_state_snapshot('a')['personal_results'] is first
    assert room.get_state_snapshot('b')['personal_results'] is second
    room.request_race_again('a', first['heat_id'])
    assert room.get_state_snapshot('a')['participant']['view'] == 'configuration'
    assert not room.players['a'].is_ready


def test_queue_is_ordered_by_ready_time_instead_of_name_registration(room):
    room.add_player('early-builder', 'Early builder')
    room.add_player('racer', 'Racer')
    room.set_player_ready('racer')
    room.start_countdown(ready_only=True)
    later = room.add_player('later-builder', 'Later builder')
    room.set_player_ready(later.client_id)
    room.set_player_ready('early-builder')
    assert [p['client_id'] for p in room.get_state_snapshot()['spectators']] == ['later-builder', 'early-builder']


def test_twenty_second_window_updates_from_original_opening(room, monkeypatch):
    import demo.game_state as state
    clock = [100.0]
    monkeypatch.setattr(state.time, 'time', lambda: clock[0])
    assert room.grid_window_seconds == 20
    room.add_player('a', 'Alex')
    room.set_player_ready('a')
    assert room.grid_open_timestamp_ms == 100000
    assert room.grid_close_timestamp_ms == 120000
    clock[0] += 10
    room.set_grid_window(30)
    assert room.grid_close_timestamp_ms == 130000
    room.set_grid_window(5)
    assert room.grid_close_timestamp_ms == 105000  # The lifecycle must start immediately.
    room.set_player_ready('a', False)
    assert room.grid_close_timestamp_ms == room.grid_open_timestamp_ms == 0
    room.set_player_ready('a')
    assert room.grid_close_timestamp_ms == 115000
    with pytest.raises(ValueError):
        room.set_grid_window(0)


def test_preferred_colours_reserve_on_ready_and_survive_freeze(room):
    from demo.game_state import CAR_COLORS
    for cid in ['first-joined', 'first-ready', 'builder']:
        room.add_player(cid, cid)
        room.update_appearance(cid, 'motorbike', CAR_COLORS[1])
    room.set_player_ready('first-ready')
    room.set_player_ready('first-joined')
    assert room.players['first-ready'].avatar_color == CAR_COLORS[1]
    assert room.players['first-joined'].avatar_color != CAR_COLORS[1]
    before = {p.client_id: p.avatar_color for p in room._ready_grid()}
    room.start_countdown(ready_only=True)
    assert {p.client_id: p.avatar_color for p in room.heat.entries} == before
    assert all(p.vehicle == 'motorbike' for p in room.heat.entries)
    assert len({p.avatar_color for p in room.heat.entries}) == len(room.heat.entries)
    with pytest.raises(ValueError):
        room.update_appearance('first-ready', 'hatchback', CAR_COLORS[0])
    room.start_racing()
    results = room.finish_race()
    room.request_race_again('first-ready', results['heat_id'])
    assert room.players['first-ready'].vehicle == 'motorbike'
    assert room.players['first-ready'].preferred_color == CAR_COLORS[1]


def test_six_identical_colour_requests_queue_and_editing_releases_reservation():
    from demo.game_state import CAR_COLORS
    room = BoothRoomManager(DemoQueryEngine(), max_active_racers=6)
    for i in range(7):
        room.add_player(str(i), f'Racer {i}')
        room.update_appearance(str(i), 'hatchback', CAR_COLORS[0])
        room.set_player_ready(str(i))
    assert len({p.avatar_color for p in room._ready_grid()}) == 6
    assert room.players['6'].is_spectator
    room.update_appearance('0', 'formula', CAR_COLORS[2])
    assert not room.players['0'].is_ready
    assert not room.players['6'].is_spectator
    assert room.players['6'].avatar_color == CAR_COLORS[0]
    assert len({p.avatar_color for p in room._ready_grid()}) == 6
    room.start_countdown(ready_only=True)
    assert len({p.avatar_color for p in room.heat.entries}) == 6


def test_removed_frozen_entrant_finishes_without_reappearing(room):
    room.add_player('a', 'Alex')
    room.set_player_ready('a')
    room.start_countdown(ready_only=True)
    frozen = room.heat.entries
    room.remove_player('a')
    assert room.heat.entries == frozen
    room.reconnect('a')
    assert 'a' not in room.players
    room.start_racing()
    results = room.finish_race()
    assert any(r['client_id'] == 'a' for r in results['rankings'])
    assert 'a' not in room.players


def test_presenter_roster_includes_reviewers_offline_and_queue(room):
    room.add_player('a', 'Alex')
    room.set_player_ready('a')
    room.start_countdown(ready_only=True)
    room.add_player('b', 'Builder')
    room.add_player('q', 'Queued')
    room.set_player_ready('q')
    room.add_player('o', 'Offline')
    room.disconnect('o')
    statuses = {p['client_id']: p['status'] for p in room.presenter_snapshot()['roster']}
    assert statuses['a'] == 'Racing' and statuses['b'] == 'Building'
    assert statuses['q'] == 'Queued' and statuses['o'] == 'Offline'
    room.start_racing(); room.finish_race()
    roster = room.presenter_snapshot()['roster']
    assert next(p for p in roster if p['client_id'] == 'a')['status'] == 'Reviewing results'
    assert not any('last_results' in p or 'token' in p for p in roster)


def test_all_ready_counts_connected_builders_not_reviewers_offline_or_bots(room, monkeypatch):
    from demo.game_state import ALL_READY_COUNTDOWN_SECONDS
    import demo.game_state as state
    monkeypatch.setattr(state.time, 'time', lambda: 100.0)
    room.add_player('alex', 'Alex')
    room.add_player('sam', 'Sam')
    room.add_player('reviewer', 'Reviewer').reviewing_results = True
    room.add_player('offline', 'Offline')
    room.disconnect('offline')
    assert not room.all_configurators_ready()
    room.set_player_ready('alex')
    assert not room.all_configurators_ready()
    room.set_grid_window(60)
    assert room.grid_close_timestamp_ms == 160000
    room.set_player_ready('sam')
    assert room.all_configurators_ready()
    room.start_countdown(ready_only=True, countdown_seconds=ALL_READY_COUNTDOWN_SECONDS)
    state = room.get_state_snapshot('alex')
    assert state['countdown_seconds'] == 3
    assert state['race_start_timestamp_ms'] == 103000
    assert not state['grid_close_timestamp_ms']
    assert room.countdown_seconds == 7  # Normal starts retain the full lights sequence.
    assert not room.all_configurators_ready()


def test_all_ready_requires_real_people_and_handles_full_grid_queue(room):
    bot = room.add_bot()
    assert not room.all_configurators_ready()
    room.remove_player(bot.client_id)
    for i in range(5):
        room.add_player(str(i), f'Racer {i}')
        room.set_player_ready(str(i))
    assert room.players['4'].is_spectator
    assert room.all_configurators_ready()
    room.set_player_ready('0', False)
    assert not room.all_configurators_ready()
    room.disconnect('0')
    assert room.all_configurators_ready()
    room.reconnect('0')
    assert not room.all_configurators_ready()


@pytest.mark.parametrize('laps', [2, 3, 6])
def test_lap_count_freezes_and_scales_first_lap_pit_without_changing_finish(room, laps):
    from demo.race import build_schedules
    room.add_player('a', 'Alex')
    room.add_player('b', 'Sam')
    room.update_pipeline('b', 'numba_efficient', 'None', 'logistic_regression')
    room.laps = laps
    room.start_countdown()
    assert room.heat.laps == laps
    assert room.get_state_snapshot()['laps'] == laps
    baseline = build_schedules(room.heat.entries, room.race_duration_seconds)
    for scheduled, original in zip(room.heat.schedules, baseline):
        assert scheduled.finish_ms == original.finish_ms
        assert scheduled.pit_ms == original.pit_ms
        assert scheduled.pit_entry_ms == round(scheduled.driving_ms * .88 / laps)
        assert 0 < scheduled.pit_entry_ms < scheduled.pit_stop_ms <= scheduled.pit_departure_ms < scheduled.pit_exit_ms < scheduled.finish_ms
    room.laps = 2
    assert room.get_state_snapshot()['laps'] == laps  # Playback follows the frozen heat.
    room.start_racing()
    assert room.finish_race()['laps'] == laps
