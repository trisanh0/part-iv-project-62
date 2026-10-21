"""Compile traceable single-experiment cohorts into an atomic demo release.

Run from the repository root: python -m demo.compile_db
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import sqlite3
import statistics
import tempfile
from collections import defaultdict, Counter
from dataclasses import asdict, replace
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import yaml
from demo.query import ConfigMetrics, TrackInfo, DemoQueryEngine, SCHEMA_VERSION, DATA_DIR
from demo.scoring import SCORING_VERSION, composite_score, performance_order

TRACK_METADATA: dict[str, dict[str, Any]] = {
    "beed": {
        "display_name": "BEED EEG Classification",
        "domain": "EEG",
        "task_type": "classification",
        "primary_metric": "accuracy",
        "is_curated_flagship": True,
        "description": "Classify EEG recordings from the BEED epilepsy dataset.",
        "baseline_extractor": "tsfresh_efficient",
        "baseline_model": "random_forest",
    },
    "pred-maintenance-w100-cls": {
        "display_name": "AI4I Predictive Maintenance",
        "domain": "Industrial IoT",
        "task_type": "classification",
        "primary_metric": "accuracy",
        "is_curated_flagship": True,
        "description": "Predict machine failure from the AI4I predictive maintenance dataset.",
        "baseline_extractor": "tsfresh_efficient",
        "baseline_model": "random_forest",
    },
    "gas-sensor-drift-sub": {
        "display_name": "Gas Sensor Drift Speedway",
        "domain": "Chemical Sensing",
        "task_type": "classification",
        "primary_metric": "accuracy",
        "is_curated_flagship": True,
        "description": "Multi-sensor drift classification across 6 chemical compounds with multi-month temporal degradation.",
        "baseline_extractor": "tsfresh_efficient",
        "baseline_model": "random_forest",
    },
    "har": {
        "display_name": "Human Kinematics Super-Sprint",
        "domain": "Biomechanics",
        "task_type": "classification",
        "primary_metric": "accuracy",
        "is_curated_flagship": True,
        "description": "Tri-axial smartphone accelerometer and gyroscope human activity recognition benchmark across 10,299 sequences.",
        "baseline_extractor": "tsfresh_efficient",
        "baseline_model": "random_forest",
    },
    "drift-bifurcation-reg-1k": {
        "display_name": "Langevin Bifurcation Autostrada",
        "domain": "Nonlinear Dynamics",
        "task_type": "regression",
        "primary_metric": "tau_r2",
        "is_curated_flagship": True,
        "description": "Supercritical pitchfork bifurcation parameter recovery (tau >= tau_c) across 1,000 stochastic trajectories.",
        "baseline_extractor": "tsfresh_efficient",
        "baseline_model": "random_forest",
    },
    "drift-bifurcation-cls": {
        "display_name": "Bifurcation Regime Switchway",
        "domain": "Nonlinear Dynamics",
        "task_type": "classification",
        "primary_metric": "accuracy",
        "is_curated_flagship": False,
        "description": "Binary classification of supercritical pitchfork bifurcation regimes across fluctuating potential wells.",
        "baseline_extractor": "tsfresh_efficient",
        "baseline_model": "random_forest",
    },
    "pred-maintenance-w100-reg": {
        "display_name": "Milling Tool Wear Oval",
        "domain": "Industrial IoT",
        "task_type": "regression",
        "primary_metric": "r2",
        "is_curated_flagship": False,
        "description": "Continuous tool wear degradation estimation across multivariate telemetry streams.",
        "baseline_extractor": "tsfresh_efficient",
        "baseline_model": "random_forest",
    },
    "appliances-energy": {
        "display_name": "Smart Home Energy Circuit",
        "domain": "Building Physics",
        "task_type": "regression",
        "primary_metric": "r2",
        "is_curated_flagship": False,
        "description": "Low-energy house electrical appliance power demand regression from weather and indoor ambient conditions.",
        "baseline_extractor": "tsfresh_efficient",
        "baseline_model": "random_forest",
    },
    "appliances-energy-forecasting": {
        "display_name": "Smart Home Demand Horizon",
        "domain": "Building Physics",
        "task_type": "forecasting",
        "primary_metric": "forecast_rmse",
        "is_curated_flagship": False,
        "description": "Multi-step forward forecasting of household appliance energy consumption.",
        "baseline_extractor": "tsfresh_efficient",
        "baseline_model": "random_forest",
    },
    "beijing-pm25": {
        "display_name": "Urban Atmospheric Driftway",
        "domain": "Environmental Science",
        "task_type": "regression",
        "primary_metric": "r2",
        "is_curated_flagship": False,
        "description": "Multivariate particulate matter (PM2.5) air pollution regression under atmospheric stagnation.",
        "baseline_extractor": "tsfresh_efficient",
        "baseline_model": "random_forest",
    },
    "beijing-pm25-forecasting": {
        "display_name": "Atmospheric PM2.5 Horizon",
        "domain": "Environmental Science",
        "task_type": "forecasting",
        "primary_metric": "forecast_rmse",
        "is_curated_flagship": False,
        "description": "Recursive multi-horizon aerosol particulate forecasting under seasonal weather drift.",
        "baseline_extractor": "tsfresh_efficient",
        "baseline_model": "random_forest",
    },
    "simulated-forecasting": {
        "display_name": "Synthetic AR Forecasting Loop",
        "domain": "Signal Processing",
        "task_type": "forecasting",
        "primary_metric": "forecast_rmse",
        "is_curated_flagship": False,
        "description": "Multi-horizon autoregressive time-series forecasting with additive stochastic shocks.",
        "baseline_extractor": "tsfresh_efficient",
        "baseline_model": "random_forest",
    },
}

SELECTOR_NORMALIZATION_MAP = {
    "select_k_best": "select_k_best_anova",
    "subsampled": "subsampled_fdr",
}

def normalize_selector(value: Any) -> str:
    """Normalize null, None, or empty selector strings to 'None' and standardize aliases."""
    if value is None or pd.isna(value):
        return "None"
    s = str(value).strip()
    if s == "" or s.lower() in ("none", "null", "nan"):
        return "None"
    return SELECTOR_NORMALIZATION_MAP.get(s, s)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value.item() if hasattr(value, 'item') else value


def finite(value, minimum=None):
    try:
        number = float(value)
        return math.isfinite(number) and (minimum is None or number >= minimum)
    except (TypeError, ValueError):
        return False


def settings_by_name(config, key):
    result = defaultdict(set)
    for item in config.get(key, []):
        if item is None:
            item = {'name': 'None'}
        elif isinstance(item, str):
            item = {'name': item}
        else:
            item = dict(item)
        name = normalize_selector(item['name']) if key == 'selectors' else item['name']
        item['name'] = name
        result[name].add(json.dumps(item, sort_keys=True))
    return result


def source_config(root, csv_path):
    """Prefer a recorded run-local snapshot; reject conflicting matches."""
    matches = []
    for base in (root / 'configs', root / 'benchmark_results'):
        for p in base.rglob('*.yaml'):
            if 'archive' in p.parts:
                continue
            config = yaml.safe_load(p.read_text())
            if isinstance(config, dict) and config.get('output_dir'):
                if (root / config['output_dir']).resolve() == csv_path.parent.resolve():
                    matches.append((p, config))
    local = [(p, c) for p, c in matches if p.is_relative_to(root / 'benchmark_results')]
    matches = local or matches
    if not matches:
        raise ValueError('missing_experiment_configuration')
    # Logging destinations are immaterial; actual experimental settings must agree.
    identities = {json.dumps(c, sort_keys=True) for _, c in matches}
    if len(identities) != 1:
        raise ValueError('conflicting_experiment_configurations')
    return sorted(matches, key=lambda x: str(x[0]))[0]


def prepare_cohort(frame, config, source):
    """Return eligible derived rows and per-row exclusion reasons; input stays raw."""
    required = ['Dataset', 'Task', 'Seed', 'Extractor', 'Selector', 'Model']
    if any(k not in frame for k in required):
        return [], [{'source': source, 'reason': 'unsupported_schema', 'rows': len(frame)}]
    exts, sels, models = (settings_by_name(config, k) for k in ('extractors', 'selectors', 'models'))
    prepared, rejected = [], []
    for i, raw in frame.iterrows():
        row = {k: safe_value(v) for k, v in raw.to_dict().items()}
        row['row_id'] = f'{source}:{i + 2}'
        row['source'] = source
        row['Selector'] = normalize_selector(row['Selector'])
        reason = None
        if any(row.get(k) is None for k in required):
            reason = 'missing_identity'
        elif row['Task'] != config.get('task_type'):
            reason = 'task_mismatch'
        elif not any(Path(p).name == row['Dataset'] for p in config.get('dataset_paths', [])):
            reason = 'dataset_not_in_experiment'
        elif not finite(row['Seed']) or float(row['Seed']) != int(row['Seed']) or int(row['Seed']) not in config.get('seeds', []):
            reason = 'seed_not_in_experiment'
        elif any(len(table.get(row[key], set())) != 1 for table, key in ((exts, 'Extractor'), (sels, 'Selector'), (models, 'Model'))):
            reason = 'ambiguous_or_unknown_component_settings'
        elif not finite(row.get('N Extracted Features'), 1):
            reason = 'missing_feature_count'
        if reason:
            rejected.append({'row_id': row['row_id'], 'reason': reason})
            continue
        row['settings'] = [next(iter(exts[row['Extractor']])), next(iter(sels[row['Selector']])), next(iter(models[row['Model']]))]
        if row['Task'] == 'forecasting' and row['Dataset'] in ('appliances-energy', 'beijing-pm25'):
            row['Dataset'] += '-forecasting'
        prepared.append(row)

    # A repeated seed must not be counted as an independent replicate.
    unique = []
    grouped = defaultdict(list)
    for row in prepared:
        grouped[tuple(row[k] for k in required)].append(row)
    for rows in grouped.values():
        signatures = {json.dumps({k: v for k, v in r.items() if k != 'row_id'}, sort_keys=True) for r in rows}
        if len(signatures) > 1:
            rejected.extend({'row_id': r['row_id'], 'reason': 'conflicting_duplicate_seed'} for r in rows)
        else:
            unique.append(rows[0])
            rejected.extend({'row_id': r['row_id'], 'reason': 'duplicate_seed'} for r in rows[1:])

    counts = defaultdict(set)
    for row in unique:
        counts[(row['Dataset'], row['Extractor'])].add(row['N Extracted Features'])
    donors = defaultdict(list)
    def donor_key(row):
        return row['Dataset'], row['Seed'], row['settings'][0], row['N Extracted Features']
    def measured(row):
        return (row.get('is_cached') is False and finite(row.get('Extraction Time (s)'), 1e-12)
                and finite(row.get('Extraction Peak RAM (MB)'), 1e-12))
    for row in unique:
        if measured(row):
            donors[donor_key(row)].append(row)
    eligible = []
    for row in unique:
        reason = None
        if len(counts[(row['Dataset'], row['Extractor'])]) != 1:
            reason = 'inconsistent_feature_count'
        donor = row if measured(row) else next(iter(sorted(donors[donor_key(row)], key=lambda r: r['row_id'])), None)
        if donor is None or (not measured(row) and row.get('is_cached') is not True):
            reason = reason or 'missing_measured_extraction'
        task = row['Task']
        metric = 'Accuracy' if task == 'classification' else 'Forecast RMSE' if task == 'forecasting' else 'Tau R2' if row['Dataset'].startswith('drift-bifurcation-reg') else 'R2'
        score = row.get(metric)
        if metric == 'Tau R2' and score is None:
            score = row.get('R2')
        selection = row.get('Selection Time (s)')
        if selection is None and row['Selector'] == 'None':
            selection = 0.0
        fit = row.get('Fit Time (s)')
        if fit is None:
            fit = row.get('fit_time_seconds')
        if not finite(score) or (task == 'forecasting' and float(score) < 0) or (task == 'classification' and not 0 <= float(score) <= 1):
            reason = reason or 'missing_or_invalid_score'
        if not finite(selection, 0) or not finite(fit, 0):
            reason = reason or 'missing_pipeline_cost'
        if reason:
            rejected.append({'row_id': row['row_id'], 'reason': reason})
            continue
        row.update(score=float(score), extraction=float(donor['Extraction Time (s)']), ram=float(donor['Extraction Peak RAM (MB)']),
                   selection=float(selection), fit=float(fit), donor_row_id=donor['row_id'], directly_measured=donor is row)
        eligible.append(row)
    return eligible, rejected


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['Dataset'], row['Task'], row['Extractor'], row['Selector'], row['Model'])].append(row)
    configs = []
    for (track, task, ext, sel, model), values in sorted(groups.items()):
        def stats(key):
            v = [r[key] for r in values]
            return statistics.mean(v), min(v), max(v), statistics.pstdev(v)
        et, emin, emax, estd = stats('extraction')
        ram, rmin, rmax, rstd = stats('ram')
        score, smin, smax, sstd = stats('score')
        st = statistics.mean(r['selection'] for r in values)
        ft = statistics.mean(r['fit'] for r in values)
        infer = [r.get('Inference Latency (ms)', r.get('inference_latency_ms')) for r in values]
        infer = [float(v) for v in infer if finite(v, 0)]
        nsel = [r.get('N Selected Features') for r in values if finite(r.get('N Selected Features'), 0)]
        configs.append(ConfigMetrics(track, task, ext, sel, model, len(values), et, emin, emax, estd, ram, rmin, rmax, rstd,
                        st, ft, statistics.mean(infer) if infer else 0.0, et + st + ft,
                        int(values[0]['N Extracted Features']), int(statistics.median(nsel)) if nsel else int(values[0]['N Extracted Features']),
                        score, sstd, smin, smax, None, None, False, composite_score(task, score, et + st + ft, ram)))
    return configs


def decorate(configs):
    results, tracks = [], {}
    for track in sorted({c.track_id for c in configs}):
        group = [c for c in configs if c.track_id == track]
        baseline = None
        for ext in ('tsfresh_efficient', 'tsfresh_minimal'):
            matches = [c for c in group if c.extractor == ext]
            if matches:
                baseline = min(matches, key=lambda c: (c.selector != 'None', c.model != 'random_forest', c.key))
                break
        meta = dict(TRACK_METADATA.get(track, {}))
        meta.update(track_id=track, task_type=group[0].task_type,
                    baseline_extractor=baseline.extractor if baseline else '', baseline_model=baseline.model if baseline else '')
        info = TrackInfo(**meta)
        tracks[track] = asdict(info)
        for c in group:
            dominated = any(performance_order(d.task_type, d.score) >= performance_order(c.task_type, c.score)
                            and d.total_latency_s <= c.total_latency_s and d.extraction_ram_mb <= c.extraction_ram_mb
                            and (d.score != c.score or d.total_latency_s < c.total_latency_s or d.extraction_ram_mb < c.extraction_ram_mb)
                            for d in group)
            results.append(replace(c, is_pareto_optimal=not dominated,
                           speedup_vs_tsfresh=baseline.extraction_time_s / c.extraction_time_s if baseline else None,
                           ram_reduction_pct=100 * (1 - c.extraction_ram_mb / baseline.extraction_ram_mb) if baseline else None))
    return results, tracks


def compile_all(repo_root='.', data_dir=None):
    root = Path(repo_root).resolve()
    target = Path(data_dir or DATA_DIR)
    sources, raw_rows, rejected, candidates = [], [], [], defaultdict(list)
    for path in sorted((root / 'benchmark_results').rglob('benchmark_results.csv')):
        if 'archive' in path.relative_to(root).parts:
            continue
        source = path.relative_to(root).as_posix()
        frame = pd.read_csv(path)
        for i, row in frame.iterrows():
            raw_rows.append({'row_id': f'{source}:{i + 2}', 'source': source,
                             'values': {k: safe_value(v) for k, v in row.to_dict().items() if k not in ('History Values', 'True Values', 'Predictions', 'Prediction Lower', 'Prediction Upper')}})
        entry = {'source': source, 'sha256': digest(path), 'rows': len(frame)}
        sources.append(entry)
        try:
            cp, config = source_config(root, path)
            entry.update(configuration=cp.relative_to(root).as_posix(), configuration_sha256=digest(cp))
            environment = path.parent / 'environment.json'
            if environment.exists():
                entry.update(environment=environment.relative_to(root).as_posix(), environment_sha256=digest(environment))
            rows, reasons = prepare_cohort(frame, config, source)
            rejected.extend(reasons)
            for track in sorted({r['Dataset'] for r in rows}):
                subset = [r for r in rows if r['Dataset'] == track]
                configs = aggregate(subset)
                candidates[track].append((source, configs, subset))
        except ValueError as error:
            rejected.append({'source': source, 'reason': str(error), 'rows': len(frame)})
    selected, selected_rows, selection = [], [], {}
    for track, cohorts in sorted(candidates.items()):
        cohorts.sort(key=lambda x: (-len(x[1]), -len({(r['Extractor'], r['Selector'], r['Model']) for r in x[2] if r['directly_measured']}), x[0]))
        source, configs, rows = cohorts[0]
        if len(configs) < 2 or track not in TRACK_METADATA:
            rejected.append({'track': track, 'reason': 'insufficient_configurations_or_unknown_track'})
            continue
        selection[track] = source
        selected.extend(configs)
        selected_rows.extend(rows)
        for other, _, _ in cohorts[1:]:
            rejected.append({'track': track, 'source': other, 'reason': 'unselected_cohort'})
    if not selected:
        raise ValueError('No eligible tracks; previous release retained')
    selected, tracks = decorate(selected)
    config_map = {f'{c.track_id}|{c.key}': asdict(c) for c in selected}
    audit = {'compiler_sha256': digest(__file__), 'sources': sources, 'selected_cohorts': selection, 'exclusions': rejected,
             'excluded_tracks': sorted((set(TRACK_METADATA) | {str(r['values'].get('Dataset')) for r in raw_rows if r['values'].get('Dataset')}) - set(tracks)),
             'raw_rows': len(raw_rows), 'eligible_rows': len(selected_rows), 'configurations': len(selected),
             'provenance': [{k: r[k] for k in ('row_id', 'Dataset', 'Extractor', 'Selector', 'Model', 'Seed', 'settings', 'donor_row_id', 'directly_measured')} for r in selected_rows]}
    signature = hashlib.sha256(json.dumps([config_map, tracks, audit, SCHEMA_VERSION, SCORING_VERSION], sort_keys=True, allow_nan=False).encode()).hexdigest()[:20]
    build_id = signature
    catalogue = {'schema_version': SCHEMA_VERSION, 'scoring_version': SCORING_VERSION, 'build_id': build_id,
                 'tracks': tracks, 'configs': config_map}
    target.mkdir(parents=True, exist_ok=True)
    builds = target / 'builds'
    builds.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.catalogue-', dir=builds) as temporary:
        stage = Path(temporary)
        (stage / 'demo_catalog.json').write_text(json.dumps(catalogue, indent=2, allow_nan=False) + '\n')
        today = datetime.now(ZoneInfo('Pacific/Auckland'))
        rag = f"TM{today:%y}{'123456789abc'[today.month-1]}{'123456789ABCDEFGHIJKLMNOPQRSTUV'[today.day-1]}a"
        report_name = f'{rag}__demo_catalogue_audit.json'
        (stage / report_name).write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
        with sqlite3.connect(stage / 'demo_benchmarks.db') as con:
            con.executescript('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL); CREATE TABLE tracks(track_id TEXT PRIMARY KEY,payload TEXT NOT NULL); CREATE TABLE aggregated_configs(key TEXT PRIMARY KEY,track_id TEXT NOT NULL,payload TEXT NOT NULL); CREATE TABLE raw_benchmark_runs(row_id TEXT PRIMARY KEY,source TEXT NOT NULL,payload TEXT NOT NULL); CREATE TABLE metric_provenance(row_id TEXT PRIMARY KEY,payload TEXT NOT NULL);')
            con.executemany('INSERT INTO metadata VALUES (?,?)', [('build_id', build_id), ('schema_version', str(SCHEMA_VERSION)), ('scoring_version', SCORING_VERSION)])
            con.executemany('INSERT INTO tracks VALUES (?,?)', [(k, json.dumps(v, allow_nan=False)) for k, v in tracks.items()])
            con.executemany('INSERT INTO aggregated_configs VALUES (?,?,?)', [(k, v['track_id'], json.dumps(v, allow_nan=False)) for k, v in config_map.items()])
            con.executemany('INSERT INTO raw_benchmark_runs VALUES (?,?,?)', [(r['row_id'], r['source'], json.dumps(r['values'], allow_nan=False)) for r in raw_rows])
            con.executemany('INSERT INTO metric_provenance VALUES (?,?)', [(r['row_id'], json.dumps(r)) for r in audit['provenance']])
            if con.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('Database integrity failed')
        con.close()
        DemoQueryEngine(stage / 'demo_benchmarks.db', stage / 'demo_catalog.json')
        release = builds / build_id
        if not release.exists():
            os.replace(stage, release)
        # Validate an existing content-addressed build before reusing it.
        DemoQueryEngine(release / 'demo_benchmarks.db', release / 'demo_catalog.json')
        manifest = {'build_id': build_id, 'audit_report': next(release.glob('*__demo_catalogue_audit.json')).name,
                    'sha256': {name: digest(release / name) for name in ('demo_benchmarks.db', 'demo_catalog.json', next(release.glob('*__demo_catalogue_audit.json')).name)}}
        fd, name = tempfile.mkstemp(prefix='.current-', dir=target)
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(manifest, f, indent=2)
                f.write('\n')
                f.flush()
                os.fsync(f.fileno())
            os.replace(name, target / 'current.json')
        finally:
            if os.path.exists(name):
                os.unlink(name)
    print(json.dumps({'build_id': build_id, 'tracks': len(tracks), 'configurations': len(selected), 'excluded_tracks': audit['excluded_tracks'], 'audit': str(release / report_name)}, indent=2))
    return manifest


if __name__ == '__main__':
    compile_all()
