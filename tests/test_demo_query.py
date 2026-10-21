"""Public query contract and stored catalogue consistency."""
import math
import time
import pytest
from demo.query import DemoQueryEngine
from demo.scoring import composite_score, performance_order

@pytest.fixture(scope='module')
def engine():return DemoQueryEngine()

def test_tracks_and_options(engine):
    assert engine.list_tracks()
    for t in engine.list_tracks():
        opts=engine.get_valid_options(t.track_id)
        assert len(opts['configurations']) >= 2
        for c in opts['configurations']:
            assert engine.get_config(t.track_id, **c)
        assert t.primary_metric

def test_metrics_and_scoring(engine):
    for t in engine.list_tracks():
        for c in engine.get_all_configs(t.track_id):
            assert c.extraction_time_s > 0 and c.extraction_ram_mb > 0
            assert c.extraction_time_min <= c.extraction_time_s <= c.extraction_time_max
            assert c.score_min <= c.score <= c.score_max
            assert c.total_latency_s == c.extraction_time_s + c.selection_time_s + c.fit_time_s
            assert c.pareto_composite_score == composite_score(c.task_type,c.score,c.total_latency_s,c.extraction_ram_mb)

def test_sorting_all_tasks(engine):
    for t in engine.list_tracks():
        for sort in ('pareto','speed','ram','score'):
            rows=engine.get_leaderboard(t.track_id,sort,1000)
            values=[{'pareto':-c.pareto_composite_score,'speed':c.total_latency_s,'ram':c.extraction_ram_mb,
                     'score':-performance_order(c.task_type,c.score)}[sort] for c in rows]
            assert values == sorted(values)
    forecasts=[t for t in engine.list_tracks() if t.task_type=='forecasting']
    assert forecasts
    for t in forecasts:
        assert engine.get_leaderboard(t.track_id,'score',1)[0].score == min(c.score for c in engine.get_all_configs(t.track_id))

def test_pareto_is_nondominated(engine):
    for t in engine.list_tracks():
        for c in engine.get_pareto_frontier(t.track_id):
            assert not any(performance_order(d.task_type,d.score) >= performance_order(c.task_type,c.score)
                           and d.total_latency_s <= c.total_latency_s and d.extraction_ram_mb <= c.extraction_ram_mb
                           and (d.score != c.score or d.total_latency_s < c.total_latency_s or d.extraction_ram_mb < c.extraction_ram_mb)
                           for d in engine.get_all_configs(t.track_id))

def test_lookup_performance_and_missing(engine):
    c=engine.get_all_configs('beed')[0]
    start=time.perf_counter()
    for _ in range(1000):assert engine.get_config('beed',c.extractor,c.selector,c.model)==c
    assert (time.perf_counter()-start)/1000 < .001
    assert engine.get_config('missing','missing') is None

def test_custom_sql_is_readonly(engine):
    assert engine.execute_custom_query('SELECT COUNT(*) AS total FROM raw_benchmark_runs')[0]['total'] > 0
    with pytest.raises(Exception):engine.execute_custom_query('DELETE FROM tracks')

def test_invalid_query(engine):
    with pytest.raises(ValueError):engine.get_leaderboard('beed','injection')
    with pytest.raises(ValueError):engine.get_leaderboard('beed',limit=-1)
