import {drawVehicle, VEHICLES, COLOR_NAMES} from './vehicles.js';
import {setupDisclosures} from './disclosures.js';
/** Mobile participant journey, driven by confirmed room state. */
import {RaceCanvasRenderer} from './canvas_renderer.js';
import {ParetoCanvasChart} from './pareto_chart.js';
import {algorithmName, escapeText, samePipeline, metricInfo, formatPerformance, formatSeconds,
    finishPositions, participantView, fieldScrollDelta, startLightsState,
    GUIDED_STAGES, guidedChoices, measurementChange} from './presentation.js';

const CURATED_TRACKS = ['gas-sensor-drift-sub', 'pred-maintenance-w100-cls', 'beed'];
export function selectDemoTracks(tracks, currentId, showAll = false) {
    if (showAll) return tracks;
    const rank = id => CURATED_TRACKS.includes(id) ? CURATED_TRACKS.indexOf(id) : CURATED_TRACKS.length;
    return tracks.filter(t => CURATED_TRACKS.includes(t.track_id) || t.track_id === currentId)
        .sort((a, b) => rank(a.track_id) - rank(b.track_id));
}
const raceTime = milliseconds => `${(milliseconds / 1000).toFixed(3)} s`;
const byId = id => document.getElementById(id);
function metricMarkup(metrics) {
    return `<span class="build-metrics"><span><span class="metric-label">Compute time</span><span class="metric-value">${formatSeconds(metrics?.total_latency_s ?? metrics?.latency_s)}</span></span><span><span class="metric-label">${metricInfo(metrics?.task_type).label}</span><span class="metric-value">${formatPerformance(metrics)}</span></span><span><span class="metric-label">Peak RAM</span><span class="metric-value">${Math.round(metrics?.extraction_ram_mb ?? metrics?.ram_mb ?? 0)} MB</span></span></span>`;
}
function loadHistory() {
    try { const value = JSON.parse(sessionStorage.getItem('tempoBuildHistory') || '[]'); return Array.isArray(value) ? value.slice(-2) : []; }
    catch { return []; }
}
const raceExplanation = task => task === 'classification'
    ? 'Faster compute gives more pace. Higher accuracy means less time in the pits.'
    : 'Race order uses compute time. Prediction quality is shown separately.';
function storageGet() { try { return sessionStorage.getItem('tempoResumeToken'); } catch { return null; } }
function storageSet(token) { try { sessionStorage.setItem('tempoResumeToken', token); } catch { /* In-memory play still works. */ } }

export class DemoClientApp {
    constructor() {
        this.socket = null;
        this.clientId = null;
        this.state = null;
        this.myPlayer = null;
        this.connected = false;
        this.pending = null;
        this.currentTrackId = null;
        this.serverOffsetMs = 0;
        this.watching = false;
        this.view = 'landing';
        this.buildHistory = loadHistory();
        this.screens = Object.fromEntries(['lobby', 'waiting', 'building', 'racing', 'results'].map(key => [key, byId(`screen-${key}`)]));
        this.inputNickname = byId('input-nickname');
        this.btnJoin = byId('btn-join');
        this.btnReady = byId('btn-ready');
        this.raceRenderer = new RaceCanvasRenderer(byId('race-canvas'), byId('live-leaderboard'), telemetry => {
            const minutes = Math.floor(telemetry.elapsed / 60).toString().padStart(2, '0');
            const seconds = (telemetry.elapsed % 60).toFixed(1).padStart(4, '0');
            byId('race-elapsed').textContent = `${minutes}:${seconds}`;
            byId('race-position').textContent = telemetry.position ? `P${telemetry.position}` : '—';
            byId('race-lap').textContent = `${telemetry.lap || '—'} / ${this.raceRenderer.totalLaps}`;
            if (telemetry.finish) {
                const notice = byId('race-finish-notice');
                notice.classList.remove('hidden');
                notice.classList.toggle('is-winner', telemetry.finish.position === 1);
                byId('finish-notice-title').textContent = telemetry.finish.position === 1
                    ? telemetry.finish.tied ? 'You share the win!' : 'You won!' : `Finished · P${telemetry.finish.position}`;
                byId('finish-notice-time').textContent = `${raceTime(telemetry.finish.timeMs)} race time`;
            }
        });
        this.waitingRenderer = new RaceCanvasRenderer(byId('waiting-canvas'), byId('waiting-leaderboard'));
        this.paretoChart = new ParetoCanvasChart(byId('pareto-canvas'), byId('pareto-tooltip'), byId('benchmark-inspector'));
        this._setupListeners();
        this.disclosures = setupDisclosures();
        this._connectWebSocket();
    }

    _setupListeners() {
        this._setupNameFocus();
        byId('join-form').addEventListener('submit', event => { event.preventDefault(); this._onJoin(); });
        byId('btn-leave-heat').addEventListener('click', () => this._command({action: 'leave'}));
        byId('btn-leave-queue').addEventListener('click', () => this._command({action: 'set_ready', ready: false}));
        for (const id of ['vehicle-choices', 'color-choices']) byId(id).addEventListener('keydown', event => {
            if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
            const buttons = [...byId(id).querySelectorAll('button:not(:disabled)')];
            const index = buttons.indexOf(event.target);
            if (index < 0) return;
            event.preventDefault();
            const direction = ['ArrowRight', 'ArrowDown'].includes(event.key) ? 1 : -1;
            const next = buttons[(index + direction + buttons.length) % buttons.length];
            next.focus(); next.click();
        });
        this.btnReady.addEventListener('click', () => this._onToggleReady());
        byId('btn-next-round').addEventListener('click', () => {
            if (this.myPlayer && !this.myPlayer.is_spectator) this._command({action: 'race_again', heat_id: (this.state.personal_results || this.state.results)?.heat_id});
            else { this.watching = false; this._syncScreenPhase(); }
        });
        byId('benchmark-explorer').addEventListener('toggle', event => {
            if (event.target.open) this._loadChart();
        });
        // Pausing a hidden tab is safe: elapsed time reconstructs the same positions.
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) { this.raceRenderer.stop(); this.waitingRenderer.stop(); }
            else this._syncScreenPhase();
        });
    }

    _setupNameFocus() {
        const ensureVisible = () => {
            if (document.activeElement !== this.inputNickname) return;
            const viewport = window.visualViewport || {offsetTop: 0, height: window.innerHeight};
            const delta = fieldScrollDelta(this.inputNickname.getBoundingClientRect(), viewport);
            if (Math.abs(delta) > 1) window.scrollBy({top: delta, behavior: 'auto'});
        };
        this.inputNickname.addEventListener('focus', () => {
            if (window.innerWidth > 700 && !window.matchMedia('(pointer: coarse)').matches) return;
            document.body.classList.add('name-entry');
            requestAnimationFrame(() => {
                if (document.activeElement !== this.inputNickname) return;
                this.inputNickname.scrollIntoView({block: 'center', behavior: 'auto'});
                ensureVisible();
            });
        });
        this.inputNickname.addEventListener('blur', () => document.body.classList.remove('name-entry'));
        for (const event of ['resize', 'scroll']) window.visualViewport?.addEventListener(event, ensureVisible);
    }

    _connectWebSocket() {
        const socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/game`);
        this.socket = socket;
        socket.onopen = () => socket.send(JSON.stringify({action: 'hello', protocol_version: 2, resume_token: storageGet()}));
        socket.onmessage = event => {
            if (this.socket !== socket) return;
            try { this._handleServerEvent(JSON.parse(event.data)); }
            catch (error) { console.error('Could not apply room update', error); }
        };
        socket.onclose = () => {
            if (this.socket !== socket) return;
            this.connected = false;
            const interrupted = !!this.pending;
            this._clearPending();
            this._connectionMessage(this.reloadRequired ? 'This page needs to be refreshed.' : 'Connection lost. Reconnecting…');
            if (interrupted) this._showError('Connection interrupted. Please try again when connected.');
            this._syncScreenPhase();
            if (!this.reloadRequired) this.reconnectTimer = setTimeout(() => this._connectWebSocket(), 1500);
        };
    }

    _connectionMessage(message) { byId('connection-status').textContent = message; }
    _errorId() { return this.view === 'results' ? 'results-error' : ['landing', 'joining'].includes(this.view) ? 'join-error' : 'config-error'; }
    _showError(message) { byId(this._errorId()).textContent = message; }
    _send(payload) {
        if (!this.connected || this.socket?.readyState !== 1) return false;
        this.socket.send(JSON.stringify(payload));
        return true;
    }
    _command(payload) {
        if (this.pending) return false;
        for (const id of ['join-error', 'config-error', 'results-error']) byId(id).textContent = '';
        if (!this._send(payload)) { this._showError('Connect to the track before trying again.'); return false; }
        this.pending = payload;
        this.pendingTimer = setTimeout(() => {
            this._clearPending(); this._showError('The request was not confirmed. Please try again.'); this._syncScreenPhase();
        }, 6000);
        this._syncScreenPhase();
        return true;
    }
    _clearPending() { clearTimeout(this.pendingTimer); this.pending = null; }
    _confirmPending() {
        const p = this.pending;
        if (!p) return;
        const confirmed = (p.action === 'join' && this.myPlayer) || (p.action === 'leave' && !this.myPlayer)
            || (p.action === 'set_pipeline' && samePipeline(this.myPlayer?.pipeline, p))
            || (p.action === 'set_appearance' && this.myPlayer?.vehicle === p.vehicle && this.myPlayer?.preferred_color === p.color)
            || (p.action === 'set_ready' && this.myPlayer?.is_ready === p.ready)
            || (p.action === 'race_again' && this.myPlayer?.preparing_next);
        if (confirmed) this._clearPending();
    }
    _handleServerEvent(event) {
        if (event.type === 'REMOVED') {
            this._clearPending();
            try { sessionStorage.removeItem('tempoResumeToken'); } catch { /* In-memory session. */ }
            this.state = null; this.myPlayer = null; this.clientId = null;
            this._syncScreenPhase();
            byId('join-error').textContent = event.data.message;
            this.socket.close();
            return;
        }
        if (event.type === 'ERROR') {
            if (event.data.code === 'reload_required') this.reloadRequired = true;
            this._clearPending(); this._showError(event.data.message); this._syncScreenPhase(); return;
        }
        if (event.type === 'INIT') {
            storageSet(event.data.resume_token);
            this.clientId = event.data.client_id;
            this.connected = true;
            this._connectionMessage('');
            this._updateState(event.data.state);
        } else if (event.type === 'STATE_UPDATE' && (!this.state || event.revision >= this.state.revision)) {
            this._updateState(event.data);
        }
        // Transition broadcasts are informational. Confirmed snapshots drive every view.
    }

    _updateState(state) {
        const oldTrack = this.currentTrackId;
        const previousPlayer = this.myPlayer;
        this.state = state;
        this.serverOffsetMs = state.server_time_utc * 1000 - Date.now();
        this.myPlayer = Object.hasOwn(state, 'participant') ? state.participant : state.active_racers.find(p => p.client_id === this.clientId) || state.spectators.find(p => p.client_id === this.clientId);
        if (oldTrack === state.track.track_id && previousPlayer?.metrics && this.myPlayer?.metrics
            && !samePipeline(previousPlayer.pipeline, this.myPlayer.pipeline)) {
            this.previousConfirmedMetrics = previousPlayer.metrics;
        }
        this._rememberCompletedBuild(state.personal_results || state.results);
        this._confirmPending();
        if (oldTrack !== state.track.track_id) {
            this.currentTrackId = state.track.track_id;
            this.previousConfirmedMetrics = null;
            this.guidedSignature = null;
            if (this.pending?.action === 'set_pipeline') this._clearPending();
        }
        this._syncScreenPhase();
    }

    _syncScreenPhase() {
        let next = participantView(this.state, this.clientId, this.pending?.action, this.watching);
        const personalResults = this.state?.personal_results || this.state?.results;
        const finishAt = this.raceRenderer.raceStartTimeMs + this.raceRenderer.raceDurationSec * 1000 + 2500;
        if (next === 'results' && this.view === 'race' && personalResults?.heat_id === this.raceRenderer.raceHeatId && Date.now() < finishAt) {
            next = 'race';
            clearTimeout(this.finishTransitionTimer);
            this.finishTransitionTimer = setTimeout(() => this._syncScreenPhase(), finishAt - Date.now());
        }
        const previous = this.view;
        this.view = next;
        const key = {landing: 'lobby', joining: 'lobby', configuration: 'building', race: 'racing', waiting: 'waiting', results: 'results'}[next];
        Object.entries(this.screens).forEach(([name, screen]) => screen?.classList.toggle('hidden', name !== key));
        document.body.dataset.view = next;
        this.btnJoin.disabled = !this.connected || !!this.pending;
        this.btnJoin.querySelector('span').textContent = this.pending?.action === 'join' ? 'Joining…' : 'Start';
        this._renderIdentity();
        if (next !== 'race') this.raceRenderer.stop();
        if (next !== 'configuration') clearInterval(this.gridTimer);
        if (next !== 'waiting') this.waitingRenderer.stop();
        if (next !== 'race' && next !== 'waiting') clearInterval(this.countdownTimer);
        if (next === 'configuration') this._renderConfiguration();
        if (next === 'waiting') this._renderWaiting();
        if (next === 'race') this._renderRace(this.raceRenderer);
        if (next === 'results') this._renderResults(personalResults);
        if (next !== previous && !(next === 'joining' || previous === 'joining')) {
            window.scrollTo(0, 0);
            document.activeElement?.blur();
            const heading = this.screens[key]?.querySelector('h1');
            if (heading) { heading.tabIndex = -1; heading.focus({preventScroll: true}); }
        }
    }
    _renderIdentity() {
        const header = byId('header-identity');
        header.textContent = this.myPlayer?.nickname || (this.watching ? 'Watching' : '');
        header.style.setProperty('--racer-color', this.myPlayer?.is_spectator ? '#4E2277' : this.myPlayer?.avatar_color || '#4E2277');
        this.raceRenderer.viewerClientId = this.clientId;
        const identity = byId('race-identity');
        const entrant = this.state?.active_racers.find(p => p.client_id === this.clientId) || (this.view === 'race' ? this.myPlayer : null);
        identity.textContent = entrant ? `YOU · ${entrant.nickname}` : 'Watching the race';
        identity.style.setProperty('--racer-color', entrant?.avatar_color || '#b9a4cf');
    }
    _onJoin() {
        if (this.pending) return;
        const nickname = this.inputNickname.value.trim();
        if (!nickname || nickname.length > 20) {
            byId('join-error').textContent = 'Enter a name between 1 and 20 characters.';
            this.inputNickname.focus(); return;
        }
        this._command({action: 'join', nickname});
    }
    _rememberCompletedBuild(results) {
        if (!results) return;
        const row = results.rankings.find(r => r.client_id === this.clientId);
        if (!row || this.buildHistory.some(b => b.heat_id === results.heat_id)) return;
        this.buildHistory = [...this.buildHistory, {heat_id: results.heat_id, track_id: results.track_id,
            build_id: results.build_id, client_id: this.clientId, metrics: row.metrics, pipeline: row.pipeline}].slice(-2);
        try { sessionStorage.setItem('tempoBuildHistory', JSON.stringify(this.buildHistory)); } catch { /* Memory-only comparison. */ }
    }
    _previousBuild(excludeHeat = null) {
        return [...this.buildHistory].reverse().find(b => b.client_id === this.clientId && b.heat_id !== excludeHeat
            && b.track_id === this.state.track.track_id && b.build_id === this.state.build_id);
    }
    _renderGuidedBuilder() {
        const configurations = this.state.track.options.configurations;
        const pipeline = this.myPlayer.pipeline;
        const signature = JSON.stringify([configurations, pipeline, !!this.pending, this.connected]);
        if (this.guidedSignature === signature) return;
        const root = byId('guided-builder');
        const open = root.querySelector('details[open]:not([data-expanded="false"])')?.dataset.stage
            || (document.querySelector('.car-customisation[open]:not([data-expanded="false"])') ? null : 'extractor');
        if (document.activeElement?.dataset.choice) {
            this.guidedFocus = {choice: document.activeElement.dataset.choice, stage: document.activeElement.dataset.stage};
        }
        const focused = this.guidedFocus?.choice;
        const focusedStage = this.guidedFocus?.stage;
        root.innerHTML = GUIDED_STAGES.map((stage, index) => {
            const choices = guidedChoices(configurations, pipeline, stage).map(choice => {
                if (choice.value !== 'select_k_best_anova' || this.state.track.track_id === 'gas-sensor-drift-sub') return choice;
                return {...choice, title: 'ANOVA k-best', description: 'Keep features most associated with the target.'};
            });
            const selectedName = choices.find(choice => choice.value === pipeline[stage.key])?.title || algorithmName(pipeline[stage.key]);
            return `<details class="build-stage" name="pipeline-build" data-stage="${stage.key}" ${open === stage.key ? 'open' : ''}><summary><span class="stage-number">0${index + 1}</span><span class="stage-heading">${stage.title}<small>${escapeText(selectedName)}</small></span><span class="stage-toggle" aria-hidden="true">+</span></summary><p class="help-text">${stage.key === 'model' && this.state.track.task_type !== 'classification' ? 'Choose the prediction model.' : stage.purpose}</p><div role="radiogroup" aria-label="${stage.title}">${choices.map(choice => `<button type="button" class="component-choice" role="radio" aria-checked="${pipeline[stage.key] === choice.value}" data-stage="${stage.key}" data-choice="${choice.value}" ${this.pending || !this.connected ? 'disabled' : ''}><span class="preset-radio" aria-hidden="true"></span><span><strong>${choice.title}</strong><small>${choice.description}</small></span></button>`).join('') || '<p class="measurement-note">This stage uses the component shown above.</p>'}</div></details>`;
        }).join('');
        root.querySelectorAll('button[data-choice]').forEach(button => {
            button.addEventListener('click', () => {
                if (this.pending) return;
                const choice = {...this.myPlayer.pipeline, [button.dataset.stage]: button.dataset.choice};
                if (configurations.some(c => samePipeline(c, choice))) this._command({action: 'set_pipeline', ...choice});
            });
            button.addEventListener('keydown', event => {
                if (!['ArrowDown', 'ArrowUp', 'ArrowLeft', 'ArrowRight'].includes(event.key)) return;
                event.preventDefault();
                const buttons = [...button.closest('[role="radiogroup"]').querySelectorAll('button:not(:disabled)')];
                const delta = ['ArrowDown', 'ArrowRight'].includes(event.key) ? 1 : -1;
                const next = buttons[(buttons.indexOf(button) + delta + buttons.length) % buttons.length];
                next?.focus(); next?.click();
            });
        });
        if (focused && !this.pending) {
            root.querySelector(`button[data-stage="${focusedStage}"][data-choice="${focused}"]:not(:disabled)`)?.focus({preventScroll: true});
            this.guidedFocus = null;
        }
        const custom = GUIDED_STAGES.some(stage => !stage.choices.some(c => c.value === pipeline[stage.key]));
        byId('custom-build-summary').classList.toggle('hidden', !custom);
        byId('custom-build-summary').textContent = 'Custom pipeline selected.';
        this.guidedSignature = signature;
    }
    _renderAppearance() {
        const player = this.myPlayer;
        const colors = this.state.car_colors || ['#FF4D4D', '#00E5FF', '#76FF03', '#FFD600', '#D500F9', '#FF9100'];
        const signature = JSON.stringify([player.vehicle, player.avatar_color, player.preferred_color, !!this.pending, this.connected]);
        if (signature === this.appearanceSignature) return;
        this.appearanceSignature = signature;
        byId('car-summary').textContent = `${VEHICLES.find(v => v.id === player.vehicle)?.name || 'Formula'} · ${COLOR_NAMES[colors.indexOf(player.avatar_color)] || 'Auto'}`;
        const choose = (vehicle, color, focus) => { this.appearanceFocus = focus; this._command({action: 'set_appearance', vehicle, color}); };
        const vehicles = byId('vehicle-choices');
        vehicles.replaceChildren(...VEHICLES.map(vehicle => {
            const button = document.createElement('button');
            button.type = 'button'; button.className = 'vehicle-choice'; button.setAttribute('role', 'radio');
            button.setAttribute('aria-checked', String((player.vehicle || 'formula') === vehicle.id));
            button.disabled = !this.connected || !!this.pending;
            const canvas = document.createElement('canvas'); canvas.width = 120; canvas.height = 64; canvas.setAttribute('aria-hidden', 'true');
            const ctx = canvas.getContext('2d'); ctx.translate(60, 32); drawVehicle(ctx, vehicle.id, player.avatar_color, 65);
            const name = document.createElement('span'); name.textContent = vehicle.name;
            button.append(canvas, name); button.addEventListener('click', () => choose(vehicle.id, player.preferred_color || player.avatar_color, {type: 'vehicle', value: vehicle.name}));
            return button;
        }));
        byId('color-choices').replaceChildren(...colors.map((color, index) => {
            const button = document.createElement('button'); button.type = 'button'; button.className = 'color-choice';
            button.setAttribute('role', 'radio'); button.setAttribute('aria-label', COLOR_NAMES[index]);
            button.setAttribute('aria-checked', String(player.avatar_color === color));
            button.style.setProperty('--racer-color', color); button.disabled = !this.connected || !!this.pending;
            button.addEventListener('click', () => choose(player.vehicle || 'formula', color, {type: 'color', value: COLOR_NAMES[index]}));
            return button;
        }));
        if (this.appearanceFocus && !this.pending) {
            const {type, value} = this.appearanceFocus;
            const root = byId(type === 'vehicle' ? 'vehicle-choices' : 'color-choices');
            [...root.querySelectorAll('button')].find(button => (button.getAttribute('aria-label') || button.textContent) === value)?.focus({preventScroll: true});
            this.appearanceFocus = null;
        }
        byId('car-color-note').textContent = player.is_ready && player.preferred_color && player.preferred_color !== player.avatar_color
            ? 'Your preferred colour was taken. This is your colour for the next race.' : '';
    }
    _renderConfiguration() {
        const track = this.state.track;
        byId('track-name').textContent = `${track.track_id === 'gas-sensor-drift-sub' ? 'Gas sensors · Identify six gases' : track.display_name} · ${track.task_type === 'classification' ? 'Classification' : track.task_type === 'forecasting' ? 'Forecasting' : 'Regression'}`;
        byId('track-desc').textContent = `${track.description} ${metricInfo(track.task_type).note}`;
        this._renderAppearance();
        this._renderGuidedBuilder();
        const metrics = this.myPlayer.metrics;
        const entries = [['Compute time', formatSeconds(metrics.total_latency_s)], [metricInfo(track.task_type).label, formatPerformance(metrics)], ['Peak RAM', `${Math.round(metrics.extraction_ram_mb)} MB`], ['Features kept / extracted', `${metrics.n_selected_features.toLocaleString()} / ${metrics.n_extracted_features.toLocaleString()}`]];
        byId('selected-measurements').innerHTML = entries.map(([label, value]) => `<dt>${label}</dt><dd>${value}</dd>`).join('');
        const previous = this.previousConfirmedMetrics || this._previousBuild()?.metrics;
        byId('build-change').textContent = previous ? `Since your ${this.previousConfirmedMetrics ? 'last selection' : 'last race'}: ${measurementChange(metrics, previous)}.` : '';
        byId('race-rules').textContent = raceExplanation(track.task_type);
        const racers = (this.state.participants || this.state.active_racers).filter(p => !p.is_bot);
        const readyForHeat = p => p.is_ready && p.connected !== false;
        const ready = racers.filter(readyForHeat).length;
        byId('readiness-count').textContent = `${ready} ready · ${racers.filter(p => !p.is_ready).length} building`;
        byId('builder-racers-list').replaceChildren();
        byId('ready-build-metrics').textContent = `${formatSeconds(metrics.total_latency_s)} compute · ${formatPerformance(metrics)} ${track.task_type === 'classification' ? 'accuracy' : ''} · ${Math.round(metrics.extraction_ram_mb)} MB RAM`.trim();
        const isReady = this.myPlayer.is_ready;
        this.btnReady.disabled = !this.connected || !!this.pending;
        this.btnReady.className = `button ${isReady ? 'button-secondary' : 'button-primary'}`;
        this.btnReady.textContent = this.pending?.action === 'set_ready' ? 'Confirming…' : isReady ? 'Undo ready' : 'Ready →';
        clearInterval(this.gridTimer);
        const updateGridTime = () => {
            const remaining = Math.max(0, Math.ceil((this.state.grid_close_timestamp_ms - (Date.now() + this.serverOffsetMs)) / 1000));
            byId('ready-status').textContent = this.pending?.action === 'set_pipeline' ? 'Confirming your build…'
                : this.state.grid_close_timestamp_ms ? `${isReady ? 'You’re in.' : 'Ready to join?'} Grid closes in ${remaining}s.`
                : isReady ? 'You’re ready for the next race.' : '';
        };
        updateGridTime();
        if (this.state.grid_close_timestamp_ms) this.gridTimer = setInterval(updateGridTime, 250);
    }
    _onToggleReady() { this._command({action: 'set_ready', ready: !this.myPlayer.is_ready}); }
    _renderWaiting() {
        byId('waiting-message').textContent = `${this.myPlayer.nickname}, you’re waiting for a place.`;
        const index = this.state.spectators.filter(p => p.connected !== false).findIndex(p => p.client_id === this.clientId);
        const reason = this.state.phase === 'COUNTDOWN' ? 'Race starting'
            : this.state.phase === 'RACING' ? 'Race in progress'
            : `Grid full · ${this.state.max_active_racers} places occupied`;
        byId('waiting-status').textContent = `${reason} · Queue position ${index + 1}`;
        const running = ['COUNTDOWN', 'RACING'].includes(this.state.phase);
        byId('waiting-race').classList.toggle('hidden', !running);
        if (running) this._renderRace(this.waitingRenderer);
        else this.waitingRenderer.stop();
        byId('btn-leave-queue').disabled = !this.connected || !!this.pending;
    }
    _renderRace(renderer) {
        const state = this.state;
        if (state.phase === 'RESULTS') return;
        byId('race-measurement-note').textContent = `${raceExplanation(state.track.task_type)} Times are scaled for the race.`;
        const key = `${state.heat_id}:${state.phase}`;
        renderer.viewerClientId = this.clientId;
        if (renderer.heatKey !== key || (state.phase === 'RACING' && !renderer.isRunning)) {
            renderer.stop(); renderer.heatKey = key; renderer.raceHeatId = state.heat_id;
            byId('race-finish-notice').classList.add('hidden');
            renderer.setupRace(state.active_racers, state.race_duration_seconds, state.race_start_timestamp_ms - this.serverOffsetMs, state.laps || 2);
            if (state.phase === 'RACING' && !document.hidden) renderer.start();
            else renderer.renderFrame(0);
        }
        byId('race-lap-count').textContent = `${renderer.totalLaps} laps`;
        byId('race-phase').textContent = state.phase === 'COUNTDOWN' ? 'Starting' : 'Live';
        clearInterval(this.countdownTimer);
        const update = () => {
            const lights = startLightsState(this.state.race_start_timestamp_ms,
                this.state.countdown_seconds, Date.now() + this.serverOffsetMs);
            byId('countdown-overlay').classList.toggle('hidden', !lights.visible);
            byId('countdown-overlay').classList.toggle('lights-out', Date.now() + this.serverOffsetMs >= this.state.race_start_timestamp_ms);
            if (byId('start-light-status').textContent !== lights.label) byId('start-light-status').textContent = lights.label;
            byId('start-lights').querySelectorAll('.start-light-column').forEach((light, index) => light.classList.toggle('is-lit', index < lights.lit));
            if (!lights.visible) clearInterval(this.countdownTimer);
        };
        update();
        if (Date.now() + this.serverOffsetMs < this.state.race_start_timestamp_ms + 600) this.countdownTimer = setInterval(update, 80);
    }

    _renderResults(results) {
        if (!results) return;
        const rows = finishPositions(results.rankings);
        const own = rows.find(r => r.client_id === this.clientId);
        byId('result-summary').textContent = own ? `You finished P${own.finish_position}${own.tied ? ' · tied' : ''}.` : 'Race complete.';
        const task = rows[0]?.metrics.task_type || this.state.track.task_type;
        byId('result-metric-note').textContent = `${task === 'classification' ? 'Higher accuracy is better.' : metricInfo(task).note} Equal race times share a position.`;
        byId('winner-banner').classList.toggle('hidden', !own || own.finish_position !== 1);
        byId('winner-banner').textContent = own?.tied ? 'You share the win!' : 'You won!';
        const previous = this._previousBuild(results.heat_id);
        byId('previous-build-comparison').textContent = own && previous ? `Since your last race: ${measurementChange(own.metrics, previous.metrics)}.` : '';
        if (this.resultSignature !== results.heat_id) {
            byId('podium-cards').innerHTML = rows.map(row => `<li class="result-row ${row.client_id === this.clientId ? 'is-you' : ''}"><div class="result-person"><span class="finish-position">${row.finish_position}${row.tied ? '=' : ''}</span><span class="racer-dot" data-color="${row.client_id}"></span><span>${escapeText(row.nickname)}</span>${row.client_id === this.clientId ? '<span class="you-label">You</span>' : ''}</div>${metricMarkup(row.metrics)}${row.race_schedule ? `<p class="race-breakdown">${row.race_schedule.uses_accuracy ? `Drive ${raceTime(row.race_schedule.driving_ms)} · Pit ${raceTime(row.race_schedule.pit_ms)} · ` : ''}<strong>Total ${raceTime(row.race_schedule.finish_ms)}</strong></p>` : ''}</li>`).join('');
            byId('podium-cards').querySelectorAll('[data-color]').forEach(dot => dot.style.setProperty('--racer-color', rows.find(r => r.client_id === dot.dataset.color).avatar_color));
            const details = byId('result-technical-content');
            details.replaceChildren(...rows.map(row => {
                const block = document.createElement('div');
                const title = document.createElement('h2'); title.textContent = row.nickname;
                const pipeline = document.createElement('p'); pipeline.className = 'help-text'; pipeline.textContent = Object.values(row.pipeline).map(algorithmName).join(' → ');
                const memory = document.createElement('p'); memory.className = 'help-text'; memory.textContent = `Peak extraction memory: ${Math.round(row.metrics.extraction_ram_mb)} MB`;
                block.append(title, pipeline, memory); return block;
            }));
            byId('benchmark-explorer').open = false;
            byId('result-technical').open = false;
            this.resultSignature = results.heat_id;
            this.chartHeat = null;
        }
        byId('btn-next-round').disabled = !this.connected || !!this.pending;
    }
    async _loadChart() {
        const results = this.state?.personal_results || this.state?.results;
        if (!results) return;
        const task = results.rankings[0]?.metrics.task_type || this.state.track.task_type;
        byId('chart-explanation').textContent = `Left is faster. Higher means ${task === 'forecasting' ? 'lower forecast error' : 'better predictions'}. The line marks pipelines that cannot improve on one axis without losing on the other.`;
        if (this.chartHeat === results.heat_id) { this.paretoChart.resize(); return; }
        try {
            const response = await fetch(`/api/pareto/${encodeURIComponent(results.track_id)}`);
            if (!response.ok) throw new Error('No benchmark data');
            const data = await response.json();
            if ((this.state?.personal_results || this.state?.results)?.heat_id !== results.heat_id) return;
            this.paretoChart.setData(data.all_points, data.pareto_frontier, finishPositions(results.rankings), task, this.clientId);
            this.chartHeat = results.heat_id;
        } catch { byId('chart-explanation').textContent = 'Chart unavailable. Close and reopen to retry.'; }
    }


}
window.addEventListener('DOMContentLoaded', () => { window.demoApp = new DemoClientApp(); });
