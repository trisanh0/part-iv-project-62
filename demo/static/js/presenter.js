import {RaceCanvasRenderer} from './canvas_renderer.js';
import {algorithmName, formatPerformance, formatSeconds, metricInfo, startLightsState} from './presentation.js';
import {drawVehicle, VEHICLES} from './vehicles.js';

const $ = id => document.getElementById(id);
const curated = ['gas-sensor-drift-sub', 'pred-maintenance-w100-cls', 'beed'];
const activeHeat = state => ['COUNTDOWN', 'RACING'].includes(state?.phase);
const phaseTitle = state => state?.phase === 'COUNTDOWN' ? 'Lights out shortly.' : state?.phase === 'RACING' ? 'On track.' : state?.grid_close_timestamp_ms ? 'The grid is forming.' : 'Waiting for racers.';

export class PresenterApp {
    constructor() {
        this.pin = ''; this.connected = false; this.busy = false; this.tab = 'dashboard';
        this.state = null; this.roster = []; this.tracks = []; this.authGeneration = 0;
        try { this.display = sessionStorage.getItem('tempoDisplayMode') === 'true'; } catch { this.display = false; }
        this.renderer = new RaceCanvasRenderer($('control-canvas'), $('control-leaderboard'), data => {
            $('control-timing').textContent = `${data.elapsed.toFixed(1)} s`;
        });
        this.renderer.showNames = true;
        this.listen(); this.syncAccess(); this.connect();
        this.clockTimer = setInterval(() => this.updateClock(), 100);
        fetch('/api/tracks', {cache: 'no-store'}).then(r => r.ok ? r.json() : Promise.reject()).then(data => {
            this.tracks = data.tracks; this.renderSettings();
        }).catch(() => { $('control-status').textContent = 'Challenge list unavailable. Reload to retry.'; });
    }
    listen() {
        $('unlock-form').addEventListener('submit', event => { event.preventDefault(); this.unlock(); });
        document.querySelectorAll('[data-tab]').forEach(button => button.addEventListener('click', () => this.setTab(button.dataset.tab)));
        $('display-mode').addEventListener('click', () => this.lock(true));
        $('lock-control').addEventListener('click', () => this.lock(false));
        $('unlock-display').addEventListener('click', () => {
            this.loginFromDisplay = true; $('control-workspace').classList.add('hidden'); $('control-login').classList.remove('hidden');
            $('back-to-display').classList.remove('hidden'); $('presenter-pin').focus();
        });
        $('back-to-display').addEventListener('click', () => { this.loginFromDisplay = false; this.syncAccess(); });
        $('fullscreen').addEventListener('click', async () => {
            try {
                if (document.fullscreenElement) await document.exitFullscreen();
                else if (document.documentElement.requestFullscreen) await document.documentElement.requestFullscreen();
                else $('control-connection').textContent = 'Full screen is not supported here. Add to Home Screen for a larger display.';
            } catch { $('control-connection').textContent = 'Full screen is unavailable in this browser.'; }
        });
        $('keep-awake').addEventListener('click', () => { this.wantWakeLock = !this.wantWakeLock; this.updateWakeLock(); });
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) this.renderer.stop();
            else { this.renderRace(true); this.poll(); this.updateWakeLock(); }
        });
        $('control-library').addEventListener('change', () => this.renderTrackOptions(true));
        for (const id of ['challenge-form', 'duration-form', 'window-form', 'laps-form', 'countdown-form']) {
            $(id).addEventListener('input', () => { $(id).dataset.dirty = 'true'; });
        }
        $('challenge-form').addEventListener('submit', event => { event.preventDefault(); this.applySetting('change_track', {track_id: $('control-track').value}, 'challenge-form'); });
        $('duration-form').addEventListener('submit', event => { event.preventDefault(); this.applySetting('set_duration', {duration_seconds: Number($('control-duration').value)}, 'duration-form'); });
        $('window-form').addEventListener('submit', event => { event.preventDefault(); this.applySetting('set_grid_window', {grid_window_seconds: Number($('control-window').value)}, 'window-form'); });
        $('laps-form').addEventListener('submit', event => { event.preventDefault(); this.applySetting('set_laps', {laps: Number($('control-laps').value)}, 'laps-form'); });
        $('countdown-form').addEventListener('submit', event => { event.preventDefault(); this.applySetting('set_countdown', {countdown_seconds: Number($('control-countdown').value)}, 'countdown-form'); });
        $('restore-defaults').addEventListener('click', () => this.confirm('Restore default settings?', `Gas sensors, 2 laps, a 27-second race, a 20-second entry window and 7-second lights.${activeHeat(this.state) ? ' This ends the current heat and returns racers to setup.' : ''}`, () => this.command('reset_settings', {interrupt_heat: activeHeat(this.state)}), 'Restore defaults'));
        $('start-ready').addEventListener('click', () => this.command('start_race'));
        $('add-bot').addEventListener('click', () => this.command('add_bot'));
        $('reset-room').addEventListener('click', () => this.confirm('Reset the room?', 'This ends the current heat and returns everyone to setup.', () => this.command('reset_lobby'), 'Reset room'));
        $('confirm-cancel').addEventListener('click', () => $('control-confirm').close());
        $('confirm-apply').addEventListener('click', () => { const action = this.confirmAction; this.confirmAction = null; $('control-confirm').close(); action?.(); });
    }
    setTab(tab) {
        if (!this.pin) return;
        this.tab = tab;
        for (const name of ['dashboard', 'people', 'settings']) $('panel-' + name).classList.toggle('hidden', name !== tab);
        document.querySelectorAll('[data-tab]').forEach(button => {
            if (button.dataset.tab === tab) button.setAttribute('aria-current', 'page'); else button.removeAttribute('aria-current');
        });
        if (tab === 'dashboard') this.renderRace(true); else this.renderer.stop();
    }
    syncAccess() {
        const unlocked = !!this.pin;
        document.body.classList.toggle('display-mode', this.display);
        $('control-login').classList.toggle('hidden', unlocked || this.display);
        $('control-workspace').classList.toggle('hidden', !unlocked && !this.display);
        $('unlock-display').classList.toggle('hidden', !this.display);
        $('back-to-display').classList.add('hidden');
        if (unlocked) this.setTab(this.tab);
        else if (this.display) {
            for (const name of ['dashboard', 'people', 'settings']) $('panel-' + name).classList.toggle('hidden', name !== 'dashboard');
            this.tab = 'dashboard';
        }
        this.updateButtons();
        if (this.display || unlocked) this.renderRace(true); else this.renderer.stop();
    }
    lock(display) {
        this.authGeneration++; this.pin = ''; this.roster = []; this.peopleSignature = null; this.display = display; this.loginFromDisplay = false;
        clearTimeout(this.pollTimer);
        $('people-groups').replaceChildren(); $('people-count').textContent = '';
        $('presenter-pin').value = ''; $('unlock-error').textContent = ''; $('control-status').textContent = ''; $('control-status').classList.remove('is-confirmed');
        try { sessionStorage.setItem('tempoDisplayMode', String(display)); } catch { /* Optional view preference. */ }
        this.syncAccess(); this.renderBuilds();
    }
    async unlock() {
        if (this.unlockBusy) return;
        this.unlockBusy = true;
        const generation = this.authGeneration;
        const pin = $('presenter-pin').value;
        $('unlock-form').querySelector('button').disabled = true;
        try {
            const state = await this.post('/api/admin/state', {pin});
            if (generation !== this.authGeneration) return;
            this.pin = pin; this.display = false; this.loginFromDisplay = false;
            $('presenter-pin').value = ''; $('unlock-error').textContent = '';
            try { sessionStorage.setItem('tempoDisplayMode', 'false'); } catch { /* Optional view preference. */ }
            this.applyState(state, true); this.syncAccess(); this.schedulePoll();
        } catch (error) { $('unlock-error').textContent = error.message; }
        finally { this.unlockBusy = false; $('unlock-form').querySelector('button').disabled = false; }
    }
    async post(url, body) {
        const response = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, cache: 'no-store', body: JSON.stringify(body), signal: AbortSignal.timeout(8000)});
        const data = await response.json();
        if (!response.ok) {
            const error = new Error(typeof data.detail === 'string' ? data.detail : 'Check the settings and try again.');
            error.status = response.status; throw error;
        }
        return data;
    }
    schedulePoll() { clearTimeout(this.pollTimer); if (this.pin) this.pollTimer = setTimeout(() => this.poll(), 1000); }
    async poll() {
        if (!this.pin || this.pollBusy) return;
        const generation = this.authGeneration;
        this.pollBusy = true;
        try {
            const state = await this.post('/api/admin/state', {pin: this.pin});
            if (generation === this.authGeneration && this.pin) this.applyState(state, true);
        } catch (error) {
            if (generation !== this.authGeneration) return;
            if (error.status === 403) { this.lock(false); $('unlock-error').textContent = 'Unlock again to manage the session.'; }
            else { $('control-status').classList.remove('is-confirmed'); $('control-status').textContent = 'Management connection lost. Retrying…'; }
            this.managementConnected = false; this.updateButtons();
        } finally { this.pollBusy = false; this.schedulePoll(); }
    }
    connect() {
        const socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/game`);
        this.socket = socket;
        socket.onopen = () => socket.send(JSON.stringify({action: 'hello', protocol_version: 2}));
        socket.onmessage = event => {
            if (this.socket !== socket) return;
            const message = JSON.parse(event.data);
            if (message.type === 'INIT' || message.type === 'STATE_UPDATE') {
                this.connected = true; $('control-connection').textContent = '';
                this.applyState(message.type === 'INIT' ? message.data.state : message.data);
            } else if (message.type === 'ERROR' && message.data.code === 'reload_required') {
                this.reloadRequired = true; $('control-connection').textContent = 'Reload this page to use the current version.';
            }
        };
        socket.onclose = () => {
            if (this.socket !== socket) return;
            this.connected = false; this.renderer.stop(); this.updateButtons();
            $('control-connection').textContent = this.reloadRequired ? 'Reload this page to continue.' : 'Display disconnected. Reconnecting…';
            if (!this.reloadRequired) this.reconnectTimer = setTimeout(() => this.connect(), 1500);
        };
    }
    applyState(state, management = false) {
        if (this.state && state.revision < this.state.revision) return;
        this.state = state;
        this.offset = state.server_time_utc * 1000 - Date.now();
        if (management) {
            this.roster = state.roster || [];
            if (!this.managementConnected && $('control-status').textContent === 'Management connection lost. Retrying…') $('control-status').textContent = '';
            this.managementConnected = true; this.renderPeople();
        }
        $('dashboard-phase').textContent = activeHeat(state) ? `Grand Prix / ${state.laps || 2} laps` : 'Next race';
        $('dashboard-title').textContent = phaseTitle(state);
        $('control-challenge').textContent = state.track.display_name;
        this.renderBuilds(); this.renderSettings(); this.renderRace(); this.updateButtons(); this.updateClock();
    }
    renderRace(force = false) {
        if (!this.state || this.tab !== 'dashboard' || (!this.pin && !this.display) || document.hidden) return;
        const running = activeHeat(this.state);
        const key = running ? `${this.state.heat_id}:${this.state.phase}` : 'idle';
        if (force || this.raceKey !== key) {
            this.raceKey = key; this.renderer.stop();
            this.renderer.setupRace(running ? this.state.active_racers : [], this.state.race_duration_seconds, this.state.race_start_timestamp_ms - this.offset, this.state.laps || 2);
            if (this.state.phase === 'RACING' && this.connected) this.renderer.start(); else this.renderer.renderFrame(0);
        }
        $('live-timing').classList.toggle('hidden', !running);
        $('control-lap-count').textContent = `${this.state.laps || 2} laps`;
        if (!running) $('control-timing').textContent = `${this.state.laps || 2} laps`;
    }
    updateClock() {
        if (!this.state) return;
        const now = Date.now() + this.offset;
        $('grid-clock').textContent = this.state.grid_close_timestamp_ms ? `Grid closes in ${Math.max(0, Math.ceil((this.state.grid_close_timestamp_ms - now) / 1000))}s` : '';
        const lights = startLightsState(this.state.race_start_timestamp_ms, this.state.countdown_seconds, now);
        $('control-lights').classList.toggle('hidden', this.state.phase !== 'COUNTDOWN' || !lights.visible);
        $('control-lights').querySelectorAll('.start-light-column').forEach((column, index) => column.classList.toggle('is-lit', index < lights.lit));
    }
    renderBuilds() {
        if (!this.state) return;
        const racing = activeHeat(this.state);
        const ready = this.state.participants.filter(p => p.is_ready && !p.is_spectator);
        const racers = racing ? this.state.active_racers : ready;
        $('builds-heading').textContent = racing ? 'On-track builds' : 'Next grid';
        $('grid-count').textContent = `${racers.length} / ${this.state.max_active_racers}`;
        $('control-builds').replaceChildren(...racers.map(racer => this.buildCard(racer)));
        if (!racers.length) { const p = document.createElement('p'); p.className = 'empty-grid'; p.textContent = 'Racers appear here when they press Ready.'; $('control-builds').append(p); }
        const results = this.state.results;
        $('last-result').classList.toggle('hidden', !results);
        $('last-result-list').replaceChildren(...(results?.rankings || []).map(racer => {
            const li = document.createElement('li'); const name = document.createElement('span'); name.textContent = `P${racer.finish_position}${racer.tied ? '=' : ''} · ${racer.nickname}`;
            const time = document.createElement('span'); time.className = 'tabular'; time.textContent = `${(racer.race_schedule.finish_ms / 1000).toFixed(3)} s`;
            li.append(name, time); return li;
        }));
    }
    buildCard(racer) {
        const article = document.createElement('article'); article.className = 'control-build';
        const title = document.createElement('p'); title.className = 'control-build-title';
        const canvas = document.createElement('canvas'); canvas.width = 84; canvas.height = 48; canvas.setAttribute('aria-hidden', 'true');
        const ctx = canvas.getContext('2d'); ctx.translate(42, 24); drawVehicle(ctx, racer.vehicle, racer.avatar_color, 62);
        const name = document.createElement('span'); name.textContent = racer.nickname; title.append(canvas, name);
        const pipeline = document.createElement('p'); pipeline.className = 'control-pipeline'; pipeline.textContent = ['extractor', 'selector', 'model'].map(key => algorithmName(racer.pipeline[key])).join(' → ');
        const metrics = document.createElement('div'); metrics.className = 'build-metrics';
        for (const [label, value] of [['Compute time', formatSeconds(racer.metrics.total_latency_s)], [metricInfo(racer.metrics.task_type).label, formatPerformance(racer.metrics)], ['Peak RAM', `${Math.round(racer.metrics.extraction_ram_mb)} MB`]]) {
            const cell = document.createElement('span'); const dt = document.createElement('span'); dt.className = 'metric-label'; dt.textContent = label;
            const dd = document.createElement('span'); dd.className = 'metric-value'; dd.textContent = value; cell.append(dt, dd); metrics.append(cell);
        }
        article.append(title, pipeline, metrics); return article;
    }
    renderPeople() {
        const signature = JSON.stringify(this.roster);
        if (this.peopleSignature === signature) return;
        this.peopleSignature = signature;
        $('people-count').textContent = String(this.roster.length);
        const groups = [];
        for (const status of ['Racing', 'Ready', 'Queued', 'Building', 'Reviewing results', 'Offline']) {
            const people = this.roster.filter(p => p.status === status).sort((a, b) => (a.queue_position || 0) - (b.queue_position || 0));
            if (!people.length) continue;
            const section = document.createElement('section'); section.className = 'people-group';
            const heading = document.createElement('h2'); heading.textContent = `${status} · ${people.length}`; section.append(heading);
            for (const person of people) {
                const row = document.createElement('div'); row.className = 'person-row';
                const content = document.createElement('div'); const name = document.createElement('strong'); name.textContent = person.nickname;
                const description = document.createElement('p'); description.textContent = `${person.is_bot ? 'Benchmark · ' : ''}${!person.connected ? 'Offline · ' : ''}${person.queue_position ? `Queue ${person.queue_position} · ` : ''}${VEHICLES.find(v => v.id === person.vehicle)?.name || 'Formula'} · ${['extractor', 'selector', 'model'].map(key => algorithmName(person.pipeline[key])).join(' → ')}`;
                const remove = document.createElement('button'); remove.type = 'button'; remove.className = 'text-button'; remove.textContent = 'Remove'; remove.setAttribute('aria-label', `Remove ${person.nickname}`);
                remove.disabled = !this.connected || this.busy || !this.managementConnected;
                remove.addEventListener('click', () => this.confirm(`Remove ${person.nickname}?`, status === 'Racing' ? 'Their car will finish this heat. Their device returns to the landing page.' : 'Their device returns to the landing page. They can join again.', () => this.command('remove_participant', {target_client_id: person.client_id}), 'Remove'));
                content.append(name, description); row.append(content, remove); section.append(row);
            }
            groups.push(section);
        }
        $('people-groups').replaceChildren(...groups);
        if (!groups.length) { const p = document.createElement('p'); p.className = 'empty-grid'; p.textContent = 'Nobody has joined yet.'; $('people-groups').append(p); }
    }
    renderTrackOptions(force = false) {
        if (!this.state || !this.tracks.length || (!force && $('challenge-form').dataset.dirty)) return;
        const selected = $('challenge-form').dataset.dirty ? $('control-track').value : this.state.track.track_id;
        const tracks = this.tracks.filter(track => $('control-library').checked || curated.includes(track.track_id) || track.track_id === selected || track.track_id === this.state.track.track_id);
        if (!$('control-library').checked) tracks.sort((a, b) => (curated.includes(a.track_id) ? curated.indexOf(a.track_id) : 3) - (curated.includes(b.track_id) ? curated.indexOf(b.track_id) : 3));
        const signature = JSON.stringify([tracks.map(t => t.track_id), selected]);
        if ($('control-track').dataset.signature === signature) { $('control-track').value = selected; return; }
        $('control-track').replaceChildren(...tracks.map(track => {
            const option = document.createElement('option'); option.value = track.track_id; option.textContent = track.display_name; option.selected = track.track_id === selected; return option;
        }));
        $('control-track').dataset.signature = signature;
    }
    renderSettings() {
        if (!this.state) return;
        this.renderTrackOptions();
        for (const [form, input, value] of [['duration-form', 'control-duration', this.state.settings.race_duration_seconds], ['window-form', 'control-window', this.state.settings.grid_window_seconds], ['laps-form', 'control-laps', this.state.settings.laps], ['countdown-form', 'control-countdown', this.state.settings.countdown_seconds]]) {
            if (!$(form).dataset.dirty && document.activeElement !== $(input)) $(input).value = value;
        }
    }
    updateButtons() {
        const disabled = !this.pin || !this.connected || !this.managementConnected || this.busy;
        document.querySelectorAll('.setting-block button, .setting-block input, .setting-block select, .person-row button').forEach(button => { button.disabled = disabled; });
        $('display-mode').disabled = !this.pin;
        $('start-ready').disabled = disabled || activeHeat(this.state) || !this.state?.participants.some(p => p.is_ready && !p.is_spectator);
        $('add-bot').disabled = disabled || activeHeat(this.state) || (this.state?.participants.filter(p => p.is_ready && !p.is_spectator).length >= this.state?.max_active_racers);
    }
    confirm(title, message, action, label = 'Apply now') {
        $('confirm-title').textContent = title; $('confirm-message').textContent = message; $('confirm-apply').textContent = label; this.confirmAction = action; $('control-confirm').showModal(); $('confirm-cancel').focus();
    }
    applySetting(action, payload, form) {
        const key = {change_track: 'track_id', set_duration: 'race_duration_seconds', set_grid_window: 'grid_window_seconds', set_laps: 'laps', set_countdown: 'countdown_seconds'}[action];
        const value = action === 'set_duration' ? payload.duration_seconds : payload[key];
        const unchanged = value === this.state.settings[key];
        if (unchanged) { delete $(form).dataset.dirty; $('control-status').textContent = 'Already applied.'; return; }
        const apply = interrupt => this.command(action, {...payload, interrupt_heat: interrupt}, form);
        if (activeHeat(this.state) && action !== 'set_grid_window') this.confirm('Apply now and end this heat?', 'Racers return to setup immediately with the new setting.', () => apply(true), 'Apply now');
        else apply(false);
    }
    async command(action, payload = {}, form = null) {
        if (this.busy || !this.pin || !this.connected || !this.managementConnected) return;
        this.busy = true; this.updateButtons(); $('control-status').classList.remove('is-confirmed'); $('control-status').textContent = 'Applying…';
        const generation = this.authGeneration;
        try {
            const result = await this.post('/api/admin/action', {pin: this.pin, action, ...payload});
            if (generation !== this.authGeneration || !this.pin) return;
            if (form) delete $(form).dataset.dirty;
            if (action === 'reset_settings') {
                for (const id of ['challenge-form', 'duration-form', 'window-form', 'laps-form', 'countdown-form']) delete $(id).dataset.dirty;
                $('control-library').checked = false;
            }
            this.applyState(result.state, true);
            this.showConfirmation(action, result.state);
        } catch (error) {
            if (generation !== this.authGeneration) return;
            if (error.status === 403) { this.lock(false); $('unlock-error').textContent = 'Unlock again to apply changes.'; }
            $('control-status').textContent = error.name === 'TimeoutError' ? 'No confirmation received. Check the current state before trying again.' : error.message;
        } finally { this.busy = false; this.updateButtons(); }
    }
    showConfirmation(action, state) {
        const settings = state.settings;
        const message = {
            change_track: `Challenge changed to ${state.track.display_name}.`,
            set_duration: `Race length changed to ${settings.race_duration_seconds} seconds.`,
            set_grid_window: `Entry window changed to ${settings.grid_window_seconds} seconds.`,
            set_laps: `Lap count changed to ${settings.laps}.`,
            set_countdown: `Normal starting lights changed to ${settings.countdown_seconds} seconds.`,
            reset_settings: 'Default settings restored: gas sensors, 2 laps, 27-second race, 20-second entry window and 7-second lights.',
            remove_participant: 'Participant removed.', add_bot: 'Benchmark car added.',
            reset_lobby: 'Room reset. Everyone is back in setup.', start_race: 'The race is starting.',
        }[action] || 'Settings updated.';
        const status = $('control-status');
        status.textContent = message;
        status.classList.remove('is-confirmed');
        requestAnimationFrame(() => { if (status.textContent === message) status.classList.add('is-confirmed'); });
    }
    async updateWakeLock() {
        try {
            if (!this.wantWakeLock) { await this.wakeLock?.release(); this.wakeLock = null; $('keep-awake').textContent = 'Keep awake'; return; }
            if (!navigator.wakeLock) { $('keep-awake').textContent = 'Wake lock unavailable'; return; }
            if (!document.hidden && (!this.wakeLock || this.wakeLock.released)) this.wakeLock = await navigator.wakeLock.request('screen');
            $('keep-awake').textContent = 'Keeping awake ✓';
        } catch { $('keep-awake').textContent = 'Wake lock unavailable'; }
    }
}
window.addEventListener('DOMContentLoaded', () => { window.presenterApp = new PresenterApp(); });
