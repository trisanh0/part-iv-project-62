/** Shared presentation rules for measured benchmark values and race positions. */
export const PRESETS = {
    numba: {title: 'Accelerated', description: 'Compiled features for efficient signal processing.', extractor: 'numba_efficient', model: 'random_forest'},
    tsfresh: {title: 'Standard', description: 'A comprehensive set of signal features.', extractor: 'tsfresh_efficient', model: 'random_forest'},
    polars: {title: 'Statistical', description: 'A compact set of statistical summaries.', extractor: 'polars_statistics', model: 'logistic_regression'},
};
const NAMES = {numba_efficient: 'Numba', numpy_statistical: 'NumPy summaries', polars_statistics: 'Polars', tsfresh_efficient: 'tsfresh Standard',
    tsfresh_minimal: 'tsfresh Minimal', None: 'All features', random_forest: 'Random forest', logistic_regression: 'Logistic regression', ridge: 'Ridge',
    l1: 'L1 regularisation', fdr: 'False discovery rate', subsampled_fdr: 'Subsampled FDR', select_k_best_anova: 'ANOVA k-best'};
export const GUIDED_STAGES = [
    {key: 'extractor', title: 'Extract', purpose: 'Turn sensor signals into numerical features.', choices: [
        {value: 'numpy_statistical', title: 'NumPy summaries', description: 'Five statistical summaries per channel.'},
        {value: 'numba_efficient', title: 'Numba', description: 'A broad set of compiled signal features.'},
        {value: 'tsfresh_minimal', title: 'tsfresh Minimal', description: 'A small feature set from the tsfresh library.'},
    ]},
    {key: 'selector', title: 'Select', purpose: 'Choose which extracted features reach the model.', choices: [
        {value: 'None', title: 'Keep all', description: 'Pass every extracted feature to the model.'},
        {value: 'select_k_best_anova', title: 'ANOVA top 25', description: 'Keep 25 features ranked by their relationship to the target.'},
        {value: 'subsampled_fdr', title: 'Subsampled FDR', description: 'Test feature relevance on a 10% sample of the training data.'},
    ]},
    {key: 'model', title: 'Predict', purpose: 'Learn to predict the target from the selected features.', choices: [
        {value: 'logistic_regression', title: 'Logistic regression', description: 'Learn a linear decision rule.'},
        {value: 'random_forest', title: 'Random forest', description: 'Combine predictions from many decision trees.'},
    ]},
];

export function guidedChoices(configurations, pipeline, stage) {
    return stage.choices.filter(choice => configurations.some(config => samePipeline(config, {...pipeline, [stage.key]: choice.value})));
}

export function measurementChange(current, previous) {
    if (!current || !previous || current.task_type !== previous.task_type) return '';
    const signed = (value, digits = 2) => `${value > 0 ? '+' : ''}${value.toFixed(digits)}`;
    const compute = current.total_latency_s - previous.total_latency_s;
    const computeLabel = `${compute > 0 ? '+' : compute < 0 ? '-' : ''}${formatSeconds(Math.abs(compute))}`;
    const quality = current.task_type === 'classification'
        ? `${signed((current.score - previous.score) * 100, 1)} percentage points accuracy`
        : `${signed(current.score - previous.score, 3)} ${metricInfo(current.task_type).suffix}`;
    return `${computeLabel} compute · ${quality} · ${signed(current.extraction_ram_mb - previous.extraction_ram_mb, 0)} MB peak RAM`;
}
export const algorithmName = key => NAMES[key] || String(key || '').replaceAll('_', ' ');
export const escapeText = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
export const samePipeline = (a, b) => !!a && !!b && ['extractor', 'selector', 'model'].every(k => a[k] === b[k]);
export function metricInfo(task) {
    if (task === 'classification') return {label: 'Prediction accuracy', suffix: '%', note: 'Accuracy is the share of correct predictions. Higher is better.'};
    if (task === 'forecasting') return {label: 'Forecast error', suffix: 'RMSE', note: 'RMSE measures forecast error. Lower is better; values use the challenge’s target units.'};
    return {label: 'Prediction fit', suffix: 'R²', note: 'R² measures fit against a constant prediction. Higher is better; negative values are possible.'};
}
export function formatPerformance(metrics) {
    if (!Number.isFinite(metrics?.score)) return '—';
    return metrics.task_type === 'classification' ? `${(metrics.score * 100).toFixed(1)}%` : `${metrics.score.toFixed(3)} ${metricInfo(metrics.task_type).suffix}`;
}
export function formatSeconds(value) {
    if (!Number.isFinite(value)) return '—';
    if (value > 0 && value < .01) return `${value.toPrecision(2)} s`;
    return `${value.toFixed(2)} s`;
}
export function finishPositions(rows) {
    const finish = row => row.race_schedule?.finish_ms ?? row.metrics.total_latency_s;
    const sorted = [...rows].sort((a, b) => finish(a) - finish(b) || a.client_id.localeCompare(b.client_id));
    let position = 1;
    return sorted.map((row, i) => {
        const time = finish(row);
        if (i && time !== finish(sorted[i - 1])) position = i + 1;
        const tied = sorted.some(other => other !== row && finish(other) === time);
        return {...row, finish_position: position, tied};
    });
}
export function participantView(state, clientId, pendingAction = null, watching = false) {
    const active = state?.active_racers?.find(p => p.client_id === clientId);
    const queued = state?.spectators?.find(p => p.client_id === clientId);
    if (queued) return 'waiting';
    if (!active && !watching) return pendingAction === 'join' ? 'joining' : 'landing';
    if (active?.preparing_next && state.phase === 'RESULTS') return 'configuration';
    if (state?.phase === 'COUNTDOWN' || state?.phase === 'RACING') return 'race';
    if (state?.phase === 'RESULTS') return 'results';
    return active ? 'configuration' : 'landing';
}
export function resolvePreset(configurations, preset) {
    const desired = PRESETS[preset];
    if (!desired) return null;
    return configurations.filter(c => c.extractor === desired.extractor).sort((a, b) =>
        Number(a.selector !== 'None') - Number(b.selector !== 'None') ||
        Number(a.model !== desired.model) - Number(b.model !== desired.model) ||
        a.selector.localeCompare(b.selector) || a.model.localeCompare(b.model))[0] || null;
}


export function fieldScrollDelta(rect, viewport, margin = 24) {
    const top = viewport.offsetTop + margin;
    const bottom = viewport.offsetTop + viewport.height - margin;
    if (rect.bottom > bottom) return rect.bottom - bottom;
    if (rect.top < top) return rect.top - top;
    return 0;
}

export function startLightsState(startMs, countdownSeconds, nowMs) {
    const remaining = startMs - nowMs;
    if (remaining <= 0) return {lit: 0, label: 'Race underway', visible: remaining > -600};
    const duration = Math.max(1, countdownSeconds * 1000);
    const interval = Math.min(1000, duration / 7);
    // A short dark introduction and a small per-heat hold variation. The shared
    // deadline seeds it, so every device (including reconnects) sees the same stage.
    const intro = interval * (.7 + (Math.trunc(startMs) % 1000) / 2000);
    const elapsed = duration - remaining;
    const lit = elapsed < intro ? 0 : Math.min(5, 1 + Math.floor((elapsed - intro) / interval));
    return {lit, label: 'Automatic start', visible: true};
}

/** Two-dimensional frontier for the speed/quality chart; memory is separate. */
export function speedQualityFrontier(points, taskType) {
    const quality = point => taskType === 'forecasting' ? -point.score : point.score;
    const sorted = points.filter(p => Number.isFinite(p.latency_s) && p.latency_s > 0 && Number.isFinite(p.score))
        .slice().sort((a, b) => a.latency_s - b.latency_s || quality(b) - quality(a));
    let best = -Infinity;
    return sorted.filter(point => {
        if (quality(point) <= best) return false;
        best = quality(point);
        return true;
    });
}
