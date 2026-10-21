import assert from 'node:assert/strict';
import test from 'node:test';

globalThis.window = {addEventListener() {}};
const {RaceCanvasRenderer} = await import('../demo/static/js/canvas_renderer.js');
const {PresenterApp} = await import('../demo/static/js/presenter.js');
const {DemoClientApp} = await import('../demo/static/js/app.js');

function renderer(latencies, duration = 14) {
    const r = Object.create(RaceCanvasRenderer.prototype);
    Object.assign(r, {
        resize() {}, displayWidth: 800, displayHeight: 450, totalLaps: 2,
        ctx: {clearRect() {}, fillRect() {}},
        circuit: {render: () => 1, sample: () => ({x: 0, y: 0, heading: 0, curvature: 0}), progressAtTime: laps => laps},
        _drawDriverMarker() {},
    });
    r.setupRace(latencies.map(total_latency_s => ({metrics: {total_latency_s}})), duration, 1000);
    return r;
}

test('all cars launch together and the final car finishes at the requested duration', () => {
    for (const duration of [5, 14, 60]) {
        const r = renderer([0.01, 10, 10000], duration);
        r.renderFrame(-1);
        assert.ok(r.racers.every(c => c.progress === 0));
        r.renderFrame(0.001);
        assert.ok(r.racers.every(c => c.progress > 0));
        assert.ok(r.racers[0].progress > r.racers[1].progress);
        r.renderFrame(duration - 0.001);
        assert.equal(r.racers[2].isFinished, false);
        r.renderFrame(duration);
        assert.ok(r.racers.every(c => c.isFinished));
        assert.equal(r.racers[2].finishTimeSec, duration);
        r.renderFrame(duration + 20);
        assert.equal(r.racers[2].finishTimeSec, duration);
    }
});

test('equal latencies tie; delayed playback reconstructs the same positions', () => {
    const r = renderer([10, 10]);
    r.renderFrame(7);
    assert.equal(r.racers[0].completionTimeSec, r.racers[1].completionTimeSec);
    const resumed = renderer([10, 10]);
    resumed.renderFrame(7);
    assert.deepEqual(resumed.racers, r.racers);
});

function element() {
    const classes = new Set();
    return {value: '', textContent: '', classList: {
        add: c => classes.add(c), remove: c => classes.delete(c),
        contains: c => classes.has(c),
        toggle(c, on) { if (on) classes.add(c); else classes.delete(c); },
    }};
}

test('close cars trade the lead, move forward, and retain exact finish order', () => {
    const r = renderer([10, 10.1, 10000]);
    const previous = [0, 0, 0];
    const leaders = new Set();
    for (let step = 0; step <= 14000; step++) {
        const time = step / 1000;
        r.renderFrame(time);
        r.racers.forEach((car, index) => {
            assert.ok(car.progress >= previous[index] - 1e-12, 'cars never reverse');
            if (time < car.completionTimeSec) assert.ok(car.progress < 2, 'no early finishes');
            previous[index] = car.progress;
        });
        if (time > 0.2 && time < r.racers[0].completionTimeSec - 0.2) {
            leaders.add(r.getStandings()[0]);
        }
    }
    assert.ok(leaders.has(r.racers[0]) && leaders.has(r.racers[1]), 'both close cars lead');
    assert.ok(!leaders.has(r.racers[2]), 'a much slower car does not lead');
    assert.ok(r.racers[0].finishTimeSec < r.racers[1].finishTimeSec);
    assert.deepEqual(r.getStandings(), r.racers);
    const resumed = renderer([10, 10.1, 10000]);
    r.renderFrame(3.37);
    resumed.renderFrame(3.37);
    assert.deepEqual(resumed.racers, r.racers);
});

test('curated track list keeps the library optional and preserves an active library track', async () => {
    const {selectDemoTracks} = await import('../demo/static/js/app.js');
    const ids = ['har', 'simulated-forecasting', 'beed', 'pred-maintenance-w100-reg', 'gas-sensor-drift-sub', 'pred-maintenance-w100-cls'];
    const tracks = ids.map(track_id => ({track_id}));
    assert.deepEqual(selectDemoTracks(tracks, 'beed').map(t => t.track_id),
        ['gas-sensor-drift-sub', 'pred-maintenance-w100-cls', 'beed']);
    assert.equal(selectDemoTracks(tracks, 'har').length, 4);
    assert.deepEqual(selectDemoTracks(tracks, 'beed', true), tracks);
});

test('new circuit closes smoothly with near-uniform arc-length samples', async () => {
    const {TrackCircuit} = await import('../demo/static/js/track.js');
    const circuit = new TrackCircuit();
    const first = circuit.sample(0), end = circuit.sample(1 - 1e-6);
    assert.ok(Math.hypot(first.x - end.x, first.y - end.y) < 0.1);
    assert.ok(Math.abs(first.heading - end.heading) < .05);
    assert.deepEqual(circuit.sample(0), circuit.sample(2));
    const distances = Array.from({length: 600}, (_, i) => {
        const a = circuit.sample(i / 600), b = circuit.sample((i + 1) / 600);
        return Math.hypot(a.x - b.x, a.y - b.y);
    });
    assert.ok(Math.max(...distances) / Math.min(...distances) < 1.02);
});


const {participantView, finishPositions, resolvePreset, formatPerformance, metricInfo} = await import('../demo/static/js/presentation.js');

test('fresh visitors stay on landing through every room phase, including event transitions', () => {
    for (const phase of ['LOBBY', 'BUILDING', 'COUNTDOWN', 'RACING', 'RESULTS']) {
        const state = {phase, active_racers: [], spectators: []};
        assert.equal(participantView(state, 'new'), 'landing');
        assert.equal(participantView(state, 'new', 'join'), 'joining');
        assert.equal(participantView({...state, spectators: [{client_id: 'new'}]}, 'new'), 'waiting');
    }
});

test('confirmed entrants resume their view and next-round preparation is local', () => {
    const state = {phase: 'RESULTS', active_racers: [{client_id: 'a', preparing_next: true}, {client_id: 'b'}], spectators: []};
    assert.equal(participantView(state, 'a'), 'configuration');
    assert.equal(participantView(state, 'b'), 'results');
    assert.equal(participantView({...state, phase: 'RACING'}, 'a'), 'race');
    assert.equal(participantView({...state, phase: 'LOBBY'}, 'b'), 'configuration');
    assert.equal(participantView({...state, phase: 'LOBBY', active_racers: [], spectators: [{client_id: 'b'}]}, 'b'), 'waiting');
    assert.equal(participantView({...state, phase: 'LOBBY', active_racers: [{client_id: 'b'}], spectators: []}, 'b'), 'configuration');
});

test('results reflect latency rather than composite award order, with full-precision ties', () => {
    const rows = [
        {client_id: 'slow', metrics: {total_latency_s: 20}, pareto_score: 100},
        {client_id: 'tie-b', metrics: {total_latency_s: 1}, pareto_score: 50},
        {client_id: 'close', metrics: {total_latency_s: 1.00001}},
        {client_id: 'tie-a', metrics: {total_latency_s: 1}, pareto_score: 60},
    ];
    const ranked = finishPositions(rows);
    assert.deepEqual(ranked.map(r => r.client_id), ['tie-a', 'tie-b', 'close', 'slow']);
    assert.deepEqual(ranked.map(r => r.finish_position), [1, 1, 3, 4]);
    assert.deepEqual(ranked.map(r => r.tied), [true, true, false, false]);
    assert.equal(rows[0].client_id, 'slow');
});

test('presets never substitute a different extractor and support task-specific models', () => {
    const configs = [{extractor: 'tsfresh_efficient', selector: 'None', model: 'ridge'}];
    assert.equal(resolvePreset(configs, 'numba'), null);
    assert.equal(resolvePreset(configs, 'tsfresh'), configs[0]);
    assert.match(formatPerformance({task_type: 'regression', score: -0.3}), /-0.300 R²/);
    assert.match(formatPerformance({task_type: 'forecasting', score: 20}), /20.000 RMSE/);
    assert.match(metricInfo('forecasting').note, /Lower is better/);
    assert.match(metricInfo('regression').note, /Higher is better/);
});

test('join requires a name and waits for server confirmation instead of navigating optimistically', () => {
    const prior = globalThis.document;
    const error = element();
    globalThis.document = {getElementById: () => error};
    try {
        const app = Object.create(DemoClientApp.prototype);
        app.view = 'landing'; app.inputNickname = {value: '   ', focus() {}};
        const commands = []; app._command = c => commands.push(c);
        app._onJoin();
        assert.equal(commands.length, 0);
        assert.match(error.textContent, /Enter a name/);
        app.inputNickname.value = '  Alex  '; app._onJoin();
        assert.deepEqual(commands, [{action: 'join', nickname: 'Alex'}]);
        assert.equal(app.view, 'landing');
        app.pending = {action: 'join'}; app._onJoin();
        assert.equal(commands.length, 1);
    } finally { globalThis.document = prior; }
});

test('readiness remains unchanged until confirmed by the server', () => {
    const app = Object.create(DemoClientApp.prototype);
    app.myPlayer = {is_ready: false};
    const commands = []; app._command = c => commands.push(c);
    app._onToggleReady();
    assert.equal(app.myPlayer.is_ready, false);
    assert.deepEqual(commands, [{action: 'set_ready', ready: true}]);
});

test('chart handles negative R², zero scores and lower-is-better forecast error', async () => {
    const {ParetoCanvasChart} = await import('../demo/static/js/pareto_chart.js');
    const chart = Object.create(ParetoCanvasChart.prototype);
    chart.allPoints = [{latency_s: 1, score: -3}, {latency_s: 2, score: 0}];
    chart.racerPoints = []; chart.displayWidth = 390; chart.displayHeight = 390;
    chart.padding = {left: 60, right: 30, top: 30, bottom: 50};
    chart.taskType = 'regression';
    assert.ok(chart._getBounds().minScore < -3);
    assert.ok(chart._toScreenCoords(1, 0).y < chart._toScreenCoords(1, -3).y);
    chart.taskType = 'forecasting';
    assert.ok(chart._toScreenCoords(1, -3).y < chart._toScreenCoords(1, 0).y);
});

test('duplicate joins are suppressed and unrelated updates cannot confirm a pending join', () => {
    const previousDocument = globalThis.document;
    globalThis.document = {getElementById: () => element()};
    const app = Object.create(DemoClientApp.prototype);
    let sent = 0;
    app._send = () => { sent++; return true; };
    app._syncScreenPhase = () => {};
    app._showError = () => {};
    try {
        app._command({action: 'join', nickname: 'Alex'});
        app._command({action: 'join', nickname: 'Alex'});
        assert.equal(sent, 1);
        app.myPlayer = null; app._confirmPending();
        assert.equal(app.pending.action, 'join');
        app.myPlayer = {nickname: 'Alex'}; app._confirmPending();
        assert.equal(app.pending, null);
    } finally { app._clearPending(); globalThis.document = previousDocument; }
});

test('live timing reports bounded elapsed time and shared positions for tied finishers', () => {
    const previousDocument = globalThis.document;
    globalThis.document = {createElement() {
        const node = element();
        node.children = [];
        node.appendChild = child => node.children.push(child);
        node.style = {setProperty() {}};
        return node;
    }};
    try {
        const r = renderer([10, 10]);
        r.racers.forEach((car, i) => { car.client_id = String(i); car.nickname = `Driver ${i}`; });
        r.viewerClientId = '1';
        r.leaderboard = {appendChild() {}};
        let telemetry;
        r.onTelemetry = value => { telemetry = value; };
        r.renderFrame(0);
        assert.equal(telemetry.position, null);
        r.renderFrame(14);
        assert.deepEqual(telemetry, {elapsed: 14, position: 1, lap: 2, finish: {position: 1, tied: true, timeMs: 14000}});
        r.renderFrame(100);
        assert.equal(telemetry.elapsed, 14);
        assert.equal(telemetry.position, 1);
    } finally { globalThis.document = previousDocument; }
});

test('six cars form a two-column grid and every row moves immediately', async () => {
    const {TrackCircuit} = await import('../demo/static/js/track.js');
    const r = renderer([1, 2, 3, 4, 5, 6]);
    r.circuit = new TrackCircuit();
    r.circuit.render = () => .35;
    let positions = [];
    r._drawDriverMarker = (ctx, x, y) => positions.push({x, y});
    r.renderFrame(0);
    const start = positions;
    assert.equal(new Set(start.map(p => `${p.x.toFixed(2)},${p.y.toFixed(2)}`)).size, 6);
    assert.ok(r.racers.every(car => car.progress === 0));
    for (let row = 0; row < 3; row++) {
        const a = start[row * 2], b = start[row * 2 + 1];
        assert.ok(Math.abs(Math.hypot(a.x - b.x, a.y - b.y) - 10.5) < .01);
    }
    positions = [];
    r.renderFrame(.001);
    assert.ok(positions.every((p, i) => Math.hypot(p.x - start[i].x, p.y - start[i].y) > 0));
});

test('name entry remains inside the visible keyboard viewport, including a panned viewport', async () => {
    const {fieldScrollDelta} = await import('../demo/static/js/presentation.js');
    const viewport = {offsetTop: 100, height: 300};
    assert.equal(fieldScrollDelta({top: 360, bottom: 412}, viewport), 36);
    assert.equal(fieldScrollDelta({top: 90, bottom: 142}, viewport), -34);
    assert.equal(fieldScrollDelta({top: 150, bottom: 202}, viewport), 0);
});

test('start lights fade in dark, illuminate one second apart, hold and extinguish together', async () => {
    const {startLightsState} = await import('../demo/static/js/presentation.js');
    const start = 10000, duration = 7, beginning = 3000;
    assert.equal(startLightsState(start, duration, beginning).lit, 0);
    assert.equal(startLightsState(start, duration, beginning + 699).lit, 0);
    for (let stage = 0; stage < 5; stage++) {
        assert.equal(startLightsState(start, duration, beginning + 700 + stage * 1000).lit, stage + 1);
    }
    assert.equal(startLightsState(start, duration, 9999).lit, 5);
    assert.equal(startLightsState(start, duration, 9000).label, 'Automatic start');
    assert.deepEqual(startLightsState(start, duration, start), {lit: 0, label: 'Race underway', visible: true});
    assert.equal(startLightsState(start, 0, start + 600).visible, false);
    // Short configured countdowns remain supported (e.g. transport tests).
    assert.equal(startLightsState(start, .01, start - 1).lit, 5);
});

test('speed/quality frontier excludes dominated memory winners and respects metric direction', async () => {
    const {speedQualityFrontier} = await import('../demo/static/js/presentation.js');
    const points = [
        {latency_s: 2, score: .8, ram_mb: 1},
        {latency_s: 1, score: .8, ram_mb: 100},
        {latency_s: 1, score: .7},
        {latency_s: 3, score: .9},
        {latency_s: 3, score: .9},
        {latency_s: 0, score: 1},
    ];
    const before = JSON.stringify(points);
    assert.deepEqual(speedQualityFrontier(points, 'classification'), [points[1], points[3]]);
    assert.equal(JSON.stringify(points), before);
    const forecast = [{latency_s: 1, score: 4}, {latency_s: 2, score: 2}, {latency_s: 3, score: 3}];
    assert.deepEqual(speedQualityFrontier(forecast, 'forecasting'), forecast.slice(0, 2));
    const regression = [{latency_s: 1, score: -3}, {latency_s: 2, score: 0}];
    assert.deepEqual(speedQualityFrontier(regression, 'regression'), regression);
});

test('start line and six-car grid sit in the middle of the main straight', async () => {
    const {TrackCircuit} = await import('../demo/static/js/track.js');
    const circuit = new TrackCircuit();
    const line = circuit.sample(0);
    assert.ok(Math.abs(line.x - 220) < .01);
    assert.ok(Math.abs(line.y - 740) < .01);
    assert.ok(Math.abs(line.heading + Math.PI / 2) < .01);
    // The grid extends 142 units behind the start, still well inside the straight.
    const behind = circuit.sample(1 - 142 / circuit.totalTrackLength);
    assert.ok(Math.abs(behind.x - 220) < 1);
    assert.ok(behind.y > 880 && behind.y < 884);
});

test('circuit pacing brakes into tight corners, accelerates out and crosses lap seams smoothly', async () => {
    const {TrackCircuit} = await import('../demo/static/js/track.js');
    const c = new TrackCircuit();
    const apex = Array.from({length: 600}, (_, i) => i / 600)
        .reduce((a, b) => c.sample(a).curvature > c.sample(b).curvature ? a : b);
    const timeAt = distance => {
        let low = 0, high = 1;
        for (let i = 0; i < 40; i++) {
            const mid = (low + high) / 2;
            if (c.progressAtTime(mid) < distance) low = mid;
            else high = mid;
        }
        return (low + high) / 2;
    };
    const speedAt = distance => {
        const t = timeAt(distance), step = .00001;
        return (c.progressAtTime(t + step) - c.progressAtTime(t - step)) / (2 * step);
    };
    assert.ok(speedAt(apex) < speedAt(.02) * .9);
    assert.ok(speedAt(apex) > speedAt(.02) * .55, 'tight corners retain a flowing pace');
    assert.ok(speedAt(apex - .01) < speedAt(apex - .03));
    assert.ok(speedAt(apex + .05) > speedAt(apex));
    assert.ok(Math.abs((c.progressAtTime(1) - c.progressAtTime(.99999)) -
        (c.progressAtTime(1.00001) - c.progressAtTime(1))) < .000001);
});

test('corner pacing retains simultaneous launch, monotonic motion, finish order and reconnect recovery', async () => {
    const {TrackCircuit} = await import('../demo/static/js/track.js');
    for (const duration of [5, 27, 60]) {
        const r = renderer([.01, 2, 2, 10000], duration);
        r.circuit = new TrackCircuit();
        r.circuit.render = () => .35;
        let previous = r.racers.map(() => 0);
        for (let tick = 1; tick <= 1000; tick++) {
            r.renderFrame(duration * tick / 1000);
            r.racers.forEach((car, i) => {
                assert.ok(car.progress >= previous[i]);
                assert.ok(car.progress > 0);
                assert.equal(car.isFinished, duration * tick / 1000 >= car.completionTimeSec);
            });
            previous = r.racers.map(c => c.progress);
        }
        assert.equal(r.racers.at(-1).finishTimeSec, duration);
        assert.ok(r.racers[0].finishTimeSec < r.racers[1].finishTimeSec);
        assert.equal(r.racers[1].finishTimeSec, r.racers[2].finishTimeSec);
        r.renderFrame(duration * .37);
        const saved = r.racers.map(c => c.progress);
        r.renderFrame(duration * .8);
        r.renderFrame(duration * .37);
        assert.deepEqual(r.racers.map(c => c.progress), saved);
    }
});

test('position, heading and pace velocity remain continuous across interpolation knots', async () => {
    const {TrackCircuit} = await import('../demo/static/js/track.js');
    const c = new TrackCircuit(), epsilon = 1e-7;
    for (let i = 0; i < c.sampleResolution; i++) {
        const a = c.sample(i / c.sampleResolution - epsilon);
        const b = c.sample(i / c.sampleResolution + epsilon);
        const angle = Math.atan2(Math.sin(b.heading - a.heading), Math.cos(b.heading - a.heading));
        assert.ok(Math.abs(angle) < .001, 'tangent does not snap at a geometry knot');
        assert.ok(Math.hypot(b.x - a.x, b.y - a.y) < .001);
        const t = c.paceTimes[i];
        const before = (c.progressAtTime(t) - c.progressAtTime(t - epsilon)) / epsilon;
        const after = (c.progressAtTime(t + epsilon) - c.progressAtTime(t)) / epsilon;
        assert.ok(Math.abs(before - after) / before < .001, 'pace does not step at a timing knot');
    }
});

test('six rendered cars flow through the right-side bends without near-stops or speed spikes', async () => {
    const {TrackCircuit} = await import('../demo/static/js/track.js');
    const r = renderer([10, 10.01, 10.02, 10.03, 10.04, 10.05], 27);
    r.circuit = new TrackCircuit(); r.circuit.render = () => 1;
    let positions, previous;
    const speeds = [];
    r._drawDriverMarker = (ctx, x, y, car) => positions.push({x, y, progress: car.progress});
    for (let tick = 0; tick <= 27 * 120; tick++) {
        positions = []; r.renderFrame(tick / 120);
        if (previous) positions.forEach((p, i) => {
            if (p.progress > .2 && p.progress < 1.9 && p.x > 450) {
                speeds.push(Math.hypot(p.x - previous[i].x, p.y - previous[i].y) * 120);
            }
        });
        previous = positions;
    }
    const sorted = speeds.sort((a, b) => a - b), median = sorted[Math.floor(sorted.length / 2)];
    assert.ok(sorted[0] > median * .4, 'inside lanes keep moving through the apex');
    assert.ok(sorted.at(-1) < median * 1.6, 'no discontinuous offset/heading jumps');
});

const {scheduledMotion, PIT_ENTRY_PROGRESS, PIT_EXIT_PROGRESS} = await import('../demo/static/js/race_motion.js');
const {GUIDED_STAGES, guidedChoices, measurementChange} = await import('../demo/static/js/presentation.js');
const {TrackCircuit} = await import('../demo/static/js/track.js');
const {samePipeline} = await import('../demo/static/js/presentation.js');

function pitSchedule(box = 0, pit = 3500) {
    const drive = 20000;
    const stop = Math.round(drive * (.48 + box * .008));
    return {driving_ms: drive, pit_ms: pit, pit_entry_ms: 8800, pit_stop_ms: stop,
        pit_departure_ms: stop + pit, pit_exit_ms: 11200 + pit,
        finish_ms: drive + pit, uses_accuracy: true, pit_box: box};
}

test('scheduled pit playback is monotonic, deterministic, continuous and exact at every checkpoint', () => {
    const circuit = new TrackCircuit();
    for (const box of [0, 1, 2, 3, 4, 5]) {
        for (const penalty of [0, 3500, 12000]) {
            const schedule = pitSchedule(box, penalty);
            let previous = 0;
            for (let time = 0; time <= schedule.finish_ms; time += 10) {
                const motion = scheduledMotion(schedule, time, circuit, box);
                assert.ok(motion.progress >= previous - 1e-9, `${box}: backwards at ${time}`);
                assert.ok(motion.progress <= 2);
                previous = motion.progress;
            }
            for (const time of [schedule.pit_entry_ms, schedule.pit_stop_ms, schedule.pit_departure_ms, schedule.pit_exit_ms]) {
                const left = scheduledMotion(schedule, time - .01, circuit, box);
                const right = scheduledMotion(schedule, time + .01, circuit, box);
                assert.ok(Math.abs(left.progress - right.progress) < .0001);
                const point = m => m.inPit ? circuit.samplePit(m.pitRoute) : circuit.sample(m.progress);
                assert.ok(Math.hypot(point(left).x-point(right).x, point(left).y-point(right).y) < .1);
            }
            assert.equal(scheduledMotion(schedule, schedule.finish_ms, circuit).progress, 2);
            assert.deepEqual(scheduledMotion(schedule, schedule.pit_stop_ms + penalty / 2, circuit),
                scheduledMotion(schedule, schedule.pit_stop_ms + penalty / 2, new TrackCircuit()));
        }
    }
    assert.ok(Math.hypot(circuit.samplePit(0).x-circuit.sample(PIT_ENTRY_PROGRESS).x,
        circuit.samplePit(0).y-circuit.sample(PIT_ENTRY_PROGRESS).y) < .01);
    assert.ok(Math.hypot(circuit.samplePit(1).x-circuit.sample(PIT_EXIT_PROGRESS).x,
        circuit.samplePit(1).y-circuit.sample(PIT_EXIT_PROGRESS).y) < .01);
});

test('authoritative schedules decide results regardless of compute or composite order', () => {
    const rows = [
        {client_id: 'fast', metrics: {total_latency_s: .01}, race_schedule: {finish_ms: 27000}},
        {client_id: 'accurate-b', metrics: {total_latency_s: 1}, race_schedule: {finish_ms: 24000}},
        {client_id: 'accurate-a', metrics: {total_latency_s: 1}, race_schedule: {finish_ms: 24000}},
    ];
    const ranked = finishPositions(rows);
    assert.deepEqual(ranked.map(r => r.client_id), ['accurate-a', 'accurate-b', 'fast']);
    assert.deepEqual(ranked.map(r => r.finish_position), [1, 1, 3]);
    assert.deepEqual(ranked.map(r => r.tied), [true, true, false]);
});

test('guided stages retain other components and never offer an unmeasured combination', () => {
    const pipeline = {extractor: 'numpy_statistical', selector: 'None', model: 'logistic_regression'};
    const configs = [pipeline, {...pipeline, extractor: 'numba_efficient'}, {...pipeline, selector: 'select_k_best_anova'}];
    for (const stage of GUIDED_STAGES) {
        for (const choice of guidedChoices(configs, pipeline, stage)) {
            assert.ok(configs.some(c => samePipeline(c, {...pipeline, [stage.key]: choice.value})));
        }
    }
    assert.equal(guidedChoices(configs, pipeline, GUIDED_STAGES[0]).length, 2);
    assert.equal(guidedChoices(configs, pipeline, GUIDED_STAGES[2]).length, 1);
});

test('build comparison uses percentage points and ignores histories from another participant, challenge or release', () => {
    const current = {task_type: 'classification', total_latency_s: 2, score: .97, extraction_ram_mb: 400};
    const previous = {...current, total_latency_s: 1, score: .90, extraction_ram_mb: 500};
    assert.equal(measurementChange(current, previous), '+1.00 s compute · +7.0 percentage points accuracy · -100 MB peak RAM');
    const app = Object.create(DemoClientApp.prototype);
    app.clientId = 'me'; app.state = {track: {track_id: 'gas'}, build_id: 'current'};
    app.buildHistory = [
        {client_id: 'me', track_id: 'gas', build_id: 'current', heat_id: 'old', metrics: previous},
        {client_id: 'someone', track_id: 'gas', build_id: 'current', heat_id: 'other'},
    ];
    assert.equal(app._previousBuild('new').heat_id, 'old');
    assert.equal(app._previousBuild('old'), undefined);
    app.state.build_id = 'different';
    assert.equal(app._previousBuild(), undefined);
});

test('pit entry and exit match physical velocity rather than jumping between speed profiles', () => {
    const circuit = new TrackCircuit();
    for (let box = 0; box < 6; box++) {
        const schedule = pitSchedule(box);
        const sample = time => {
            const motion = scheduledMotion(schedule, time, circuit);
            return motion.inPit ? circuit.samplePit(motion.pitRoute) : circuit.sample(motion.progress);
        };
        for (const time of [schedule.pit_entry_ms, schedule.pit_exit_ms]) {
            const a = sample(time-.01), b = sample(time), c = sample(time+.01);
            const before = Math.hypot(b.x-a.x, b.y-a.y);
            const after = Math.hypot(c.x-b.x, c.y-b.y);
            assert.ok(Math.abs(before / after - 1) < .01, `pit box ${box} velocity jump at ${time}`);
        }
    }
});

test('finish opens promoted and new visitors builds while previous entrants keep results', () => {
    const racers = [{client_id: 'previous', preparing_next: false}, {client_id: 'promoted', preparing_next: true},
        {client_id: 'new', preparing_next: true}];
    const results = {phase: 'RESULTS', active_racers: racers, spectators: []};
    assert.equal(participantView(results, 'previous'), 'results');
    assert.equal(participantView(results, 'promoted'), 'configuration');
    assert.equal(participantView(results, 'new'), 'configuration');
    assert.equal(participantView({...results, active_racers: racers.slice(0, 1), spectators: [{client_id: 'promoted'}]}, 'promoted'), 'waiting');
});

test('personal views keep builders and result reviewers out of an unrelated race', () => {
    for (const view of ['configuration', 'waiting', 'race', 'results']) {
        const state = {phase: 'RACING', participant: {client_id: 'me', view}, active_racers: [], spectators: []};
        assert.equal(participantView(state, 'me'), view);
    }
});

test('cars coast past the finish, decelerate and retain their exact classified finish', () => {
    const r = renderer([1, 2], 27);
    r.circuit = new TrackCircuit(); r.circuit.render = () => 1;
    const car = r.racers[0];
    r.renderFrame(car.completionTimeSec);
    assert.equal(car.progress, 2);
    assert.equal(car.visualProgress, 2);
    const finish = car.finishTimeSec;
    let previous = car.visualProgress;
    for (let i=1; i<=300; i++) {
        r.renderFrame(car.completionTimeSec + i/100);
        assert.ok(car.visualProgress >= previous);
        assert.equal(car.progress, 2);
        assert.equal(car.finishTimeSec, finish);
        previous = car.visualProgress;
    }
    const parked = car.visualProgress;
    assert.ok(parked > 2 && parked < 2.15);
    r.renderFrame(1000);
    assert.equal(car.visualProgress, parked);
});

test('explicitly removed participant stays on landing despite their frozen car', () => {
    const state = {phase: 'RACING', participant: null, active_racers: [{client_id: 'removed'}], spectators: []};
    assert.equal(participantView(state, 'removed'), 'landing');
    assert.equal(participantView(state, 'removed', 'join'), 'joining');
});

test('appearance changes require matching server confirmation', () => {
    const app = Object.create(DemoClientApp.prototype);
    app.pending = {action: 'set_appearance', vehicle: 'motorbike', color: '#5C9FA0'};
    app.myPlayer = {vehicle: 'formula', preferred_color: '#5C9FA0'};
    app._confirmPending();
    assert.ok(app.pending);
    app.myPlayer.vehicle = 'motorbike';
    app._confirmPending();
    assert.equal(app.pending, null);
});

test('presenter commands preserve unlock on rejected settings and ignore responses after locking', async () => {
    const previousDocument = globalThis.document;
    const nodes = new Map();
    globalThis.document = {getElementById(id) {
        if (!nodes.has(id)) nodes.set(id, {...element(), dataset: {dirty: 'true'}});
        return nodes.get(id);
    }};
    const app = Object.create(PresenterApp.prototype);
    Object.assign(app, {pin: 'correct', connected: true, managementConnected: true, busy: false, authGeneration: 0,
        updateButtons() {}, lock() { this.pin = ''; this.authGeneration++; }});
    let applied = null;
    app.applyState = state => { applied = state; };
    try {
        app.post = async () => { const error = new Error('Check the setting'); error.status = 409; throw error; };
        await app.command('set_grid_window', {grid_window_seconds: 0});
        assert.equal(app.pin, 'correct');
        assert.equal(nodes.get('control-status').textContent, 'Check the setting');
        let resolve;
        app.post = () => new Promise(r => { resolve = r; });
        const pending = app.command('set_grid_window', {grid_window_seconds: 20});
        assert.equal(app.busy, true);
        app.lock();
        resolve({state: {revision: 1}});
        await pending;
        assert.equal(applied, null);
        assert.equal(app.pin, '');
        app.pin = 'correct';
        app.post = async () => { const error = new Error('Unauthorized'); error.status = 403; throw error; };
        await app.command('set_duration', {duration_seconds: 10});
        assert.equal(app.pin, '');
    } finally { globalThis.document = previousDocument; }
});

test('configured lap counts retain smooth pit playback, exact finish and coasting', () => {
    for (const laps of [2, 3, 6]) {
        const circuit = new TrackCircuit();
        const drive = 20000, pit = 4000;
        const schedule = {driving_ms: drive, pit_ms: pit, pit_entry_ms: Math.round(drive * .88 / laps),
            pit_stop_ms: Math.round(drive * .96 / laps), pit_departure_ms: Math.round(drive * .96 / laps) + pit,
            pit_exit_ms: Math.round(drive * 1.12 / laps) + pit, finish_ms: drive + pit, uses_accuracy: true, pit_box: 0};
        let previous = 0;
        for (let time = 0; time <= schedule.finish_ms; time += 10) {
            const motion = scheduledMotion(schedule, time, circuit, 0, laps);
            assert.ok(motion.progress >= previous && motion.progress <= laps);
            previous = motion.progress;
        }
        const r = renderer([1], 24);
        r.circuit = circuit; r.circuit.render = () => 1;
        r.setupRace([{metrics: {total_latency_s: 1}, race_schedule: schedule}], 24, 1000, laps);
        r.renderFrame(0);
        assert.equal(r.racers[0].progress, 0);
        r.renderFrame(23.999);
        assert.equal(r.racers[0].isFinished, false);
        r.renderFrame(24);
        assert.equal(r.racers[0].progress, laps);
        assert.equal(r.racers[0].finishTimeSec, 24);
        r.renderFrame(25);
        assert.ok(r.racers[0].visualProgress > laps);
        assert.equal(r.racers[0].progress, laps);
        assert.equal(scheduledMotion({...schedule, uses_accuracy: false}, 24000, circuit, 0, laps).progress, laps);
    }
});

test('restoring defaults replaces an unapplied challenge selection even when options are cached', () => {
    const originalDocument = globalThis.document;
    const select = {value: 'beed', dataset: {signature: JSON.stringify([['gas-sensor-drift-sub', 'beed'], 'gas-sensor-drift-sub'])}};
    const elements = {'control-track': select, 'challenge-form': {dataset: {}}, 'control-library': {checked: false}};
    globalThis.document = {getElementById: id => elements[id]};
    const app = Object.create(PresenterApp.prototype);
    app.state = {track: {track_id: 'gas-sensor-drift-sub'}};
    app.tracks = [{track_id: 'gas-sensor-drift-sub'}, {track_id: 'beed'}];
    try {
        app.renderTrackOptions();
        assert.equal(select.value, 'gas-sensor-drift-sub');
    } finally { globalThis.document = originalDocument; }
});
