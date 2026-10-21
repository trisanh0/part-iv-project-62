"""Provenance boundaries, score semantics, and transactional publication."""
import json
import sqlite3
from pathlib import Path
from dataclasses import asdict
import pandas as pd
import pytest
from demo import compile_db as compiler
from demo.query import DemoQueryEngine
from demo.scoring import composite_score, performance_order


def config():
    return dict(dataset_paths=['data/beed'],task_type='classification', seeds=[42,101],
                extractors=[{'name':'numba_efficient','fft_coefficients':25}],
                selectors=[None,{'name':'fdr'}],models=['random_forest','logistic_regression'])


def row(**overrides):
    return {'Dataset':'beed','Task':'classification','Seed':42,'Extractor':'numba_efficient',
        'Selector':None,'Model':'random_forest','is_cached':False,'N Extracted Features':100,
        'N Selected Features':100,'Extraction Time (s)':2.,'Extraction Peak RAM (MB)':300.,
        'Selection Time (s)':0.,'Fit Time (s)':1.,'Accuracy':.9,**overrides}


def prepare(rows,cfg=None):
    return compiler.prepare_cohort(pd.DataFrame(rows),cfg or config(),'run/benchmark_results.csv')


def test_cached_reuse_has_exact_donor_and_does_not_mutate_raw():
    rows=[row(),row(Selector='fdr',is_cached=True,**{'Extraction Time (s)':None,'Extraction Peak RAM (MB)':None})]
    accepted,rejected=prepare(rows)
    assert len(accepted)==2 and not rejected
    assert accepted[1]['donor_row_id']==accepted[0]['row_id']
    assert not accepted[1]['directly_measured']
    assert rows[1]['Extraction Time (s)'] is None

@pytest.mark.parametrize('override',[
    {'Seed':101}, {'N Extracted Features':101}, {'Dataset':'other'},
    {'Extractor':'other'}, {'is_cached':False},
])
def test_no_broader_fallback(override):
    rows=[row(),row(Selector='fdr',is_cached=True,**{'Extraction Time (s)':None,'Extraction Peak RAM (MB)':None})]
    rows[1].update(override)
    accepted,rejected=prepare(rows)
    assert not any(r['Selector']=='fdr' for r in accepted)
    assert rejected


def test_ambiguous_parameters_excluded():
    cfg=config();cfg['extractors'].append({'name':'numba_efficient','fft_coefficients':10})
    accepted,rejected=prepare([row()],cfg)
    assert not accepted and rejected[0]['reason']=='ambiguous_or_unknown_component_settings'


def test_duplicate_seed_deduplication_and_conflict():
    accepted,rejected=prepare([row(),row()])
    assert len(accepted)==1 and rejected[0]['reason']=='duplicate_seed'
    accepted,rejected=prepare([row(),row(Accuracy=.8)])
    assert not accepted and all(r['reason']=='conflicting_duplicate_seed' for r in rejected)

@pytest.mark.parametrize('overrides',[
    {'Accuracy':float('nan')},{'Accuracy':1.1},{'Fit Time (s)':None},
    {'Extraction Peak RAM (MB)':-1},{'Selection Time (s)':-1},
])
def test_invalid_measurements_excluded(overrides):
    accepted,rejected=prepare([row(**overrides)])
    assert not accepted and rejected


def test_missing_selection_only_zero_for_none():
    accepted,_=prepare([row(**{'Selection Time (s)':None})])
    assert accepted[0]['selection']==0
    accepted,rejected=prepare([row(Selector='fdr',**{'Selection Time (s)':None})])
    assert not accepted


def test_raw_regression_scores_not_clamped():
    cfg=config();cfg.update(task_type='regression')
    accepted,_=prepare([row(Task='regression',R2=-5.)],cfg)
    assert compiler.aggregate(accepted)[0].score==-5
    assert composite_score('regression',-5,2,300)==composite_score('regression',-1,2,300)
    assert composite_score('forecasting',10,2,300)>composite_score('forecasting',100,2,300)


def test_schema_rejection():
    accepted,rejected=compiler.prepare_cohort(pd.DataFrame([{'length':100}]),config(),'other')
    assert not accepted and rejected[0]['reason']=='unsupported_schema'


def write_experiment(root, name, rows, cfg=None):
    import yaml
    output=root/'benchmark_results'/name
    output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(output/'benchmark_results.csv',index=False)
    configuration=cfg or config()
    configuration['output_dir']=f'benchmark_results/{name}'
    directory=root/'configs';directory.mkdir(exist_ok=True)
    (directory/f'{name}.yaml').write_text(yaml.safe_dump(configuration))


def test_publication_cohort_isolation_rebuild_and_failure(tmp_path,monkeypatch):
    root=tmp_path/'repo';root.mkdir()
    rows=[row(),row(Model='logistic_regression',Accuracy=.8)]
    write_experiment(root,'run_a',rows)
    write_experiment(root,'run_b',[row(**{'N Extracted Features':200}),row(Model='logistic_regression',**{'N Extracted Features':200})])
    output=tmp_path/'output'
    manifest=compiler.compile_all(root,output)
    engine=DemoQueryEngine(manifest_path=output/'current.json')
    configs=engine.get_all_configs('beed')
    assert len(configs)==2 and {c.n_extracted_features for c in configs}=={100}
    assert {c.sample_count for c in configs}=={1}
    original=(output/'current.json').read_bytes()
    assert compiler.compile_all(root,output)==manifest
    assert original==(output/'current.json').read_bytes()
    report=json.loads((output/'builds'/manifest['build_id']/manifest['audit_report']).read_text())
    assert report['selected_cohorts']['beed'].endswith('run_a/benchmark_results.csv')
    assert any(r['reason']=='unselected_cohort' for r in report['exclusions'])
    def fail(*args,**kwargs):raise ValueError('injected validation failure')
    monkeypatch.setattr(compiler,'DemoQueryEngine',fail)
    with pytest.raises(ValueError,match='injected'):compiler.compile_all(root,output)
    assert original==(output/'current.json').read_bytes()
    assert DemoQueryEngine(manifest_path=output/'current.json').build_id==manifest['build_id']


def test_mismatched_release_rejected(tmp_path):
    import shutil
    engine=DemoQueryEngine()
    db=tmp_path/'test.db';catalogue=tmp_path/'test.json'
    shutil.copyfile(engine.db_path,db);shutil.copyfile(engine.json_path,catalogue)
    payload=json.loads(catalogue.read_text());payload['build_id']='wrong';catalogue.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='mismatch'):DemoQueryEngine(db,catalogue)
