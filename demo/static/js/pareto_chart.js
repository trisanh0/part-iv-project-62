import {algorithmName, formatPerformance, metricInfo, formatSeconds, speedQualityFrontier} from './presentation.js';

const latency = p => p.metrics?.total_latency_s ?? p.latency_s;
const score = p => p.metrics?.score ?? p.score;
const valid = p => Number.isFinite(latency(p)) && latency(p) > 0 && Number.isFinite(score(p));
const buildName = p => ['extractor', 'selector', 'model'].map(k => algorithmName(p.pipeline?.[k] ?? p[k])).join(' · ');

/** Decade gridlines with logarithmically spaced minor divisions and sparse labels. */
export function logarithmicTicks(logMin, logMax, pixelWidth) {
    const span = logMax - logMin;
    if (!Number.isFinite(span) || span <= 0 || pixelWidth <= 0) return [];
    const ticks = [];
    for (let exponent = Math.floor(logMin); exponent <= Math.ceil(logMax); exponent++) {
        for (let multiplier = 1; multiplier <= (span <= 4 ? 9 : 1); multiplier++) {
            const value = multiplier * 10 ** exponent;
            const position = (Math.log10(value) - logMin) / span;
            if (position < -1e-10 || position > 1 + 1e-10) continue;
            ticks.push({value, position, major: multiplier === 1, label: false,
                priority: multiplier === 1 ? 0 : [2, 5].includes(multiplier) ? 1 : 2});
        }
    }
    const labelled = [];
    for (const tick of [...ticks].sort((a, b) => a.priority - b.priority || a.value - b.value)) {
        if (tick.priority === 2 && span >= 1) continue;
        if (labelled.every(other => Math.abs(tick.position - other.position) * pixelWidth >= 54)) {
            tick.label = true; labelled.push(tick);
        }
    }
    return ticks;
}

/** Measured speed/quality comparison. Memory stays outside this two-axis frontier. */
export class ParetoCanvasChart {
    constructor(canvas, tooltipEl = null, inspector = null) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.tooltipEl = tooltipEl;
        this.inspector = inspector;
        this.allPoints = [];
        this.paretoPoints = [];
        this.racerPoints = [];
        this.inspectionPoints = [];
        this.selectedPoint = null;
        this.taskType = 'classification';
        this.padding = {top: 26, right: 14, bottom: 54, left: 65};
        canvas.addEventListener('click', e => {
            const rect = canvas.getBoundingClientRect();
            this._checkHitPoint(e.clientX - rect.left, e.clientY - rect.top);
        });
        inspector?.addEventListener('change', () => this._selectPoint(this.inspectionPoints[Number(inspector.value)]));
        window.addEventListener('resize', () => this.resize());
        if (typeof ResizeObserver !== 'undefined') {
            this.resizeObserver = new ResizeObserver(() => this.resize());
            this.resizeObserver.observe(canvas);
        }
    }

    resize() {
        const rect = this.canvas.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) return;
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = Math.round(rect.width * dpr);
        this.canvas.height = Math.round(rect.height * dpr);
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        this.displayWidth = rect.width;
        this.displayHeight = rect.height;
        this.render();
    }

    setData(allPoints, _paretoFrontier, racers = [], taskType = 'classification', viewerClientId = null) {
        this.taskType = taskType;
        this.allPoints = (allPoints || []).filter(valid);
        this.racerPoints = racers.filter(valid);
        this.paretoPoints = speedQualityFrontier(this.allPoints, taskType);
        this.inspectionPoints = [...this.racerPoints, ...this.allPoints];
        if (this.inspector) {
            this.inspector.replaceChildren();
            for (const [label, points] of [['This race', this.racerPoints], ['All benchmarked pipelines', this.allPoints]]) {
                if (!points.length) continue;
                const group = document.createElement('optgroup');
                group.label = label;
                for (const point of points) {
                    const option = document.createElement('option');
                    option.value = String(this.inspectionPoints.indexOf(point));
                    option.textContent = point.nickname ? `${point.nickname}${point.client_id === viewerClientId ? ' · you' : ''}` : buildName(point);
                    group.append(option);
                }
                this.inspector.append(group);
            }
            this.inspector.disabled = !this.inspectionPoints.length;
        }
        this.resize();
        this._selectPoint(this.racerPoints.find(p => p.client_id === viewerClientId) || this.inspectionPoints[0] || null);
    }

    _checkHitPoint(x, y) {
        let nearest = null, distance = 18;
        for (const point of this.inspectionPoints) {
            const pos = this._toScreenCoords(latency(point), score(point));
            const delta = Math.hypot(pos.x - x, pos.y - y);
            if (delta < distance) { nearest = point; distance = delta; }
        }
        if (nearest) this._selectPoint(nearest);
    }

    _selectPoint(point) {
        this.selectedPoint = point;
        if (this.inspector && point) this.inspector.value = String(this.inspectionPoints.indexOf(point));
        if (this.tooltipEl) {
            this.tooltipEl.hidden = !point;
            this.tooltipEl.replaceChildren();
            if (point) {
                const title = document.createElement('p');
                title.className = 'chart-selection-title';
                title.textContent = point.nickname || 'Benchmarked pipeline';
                const pipeline = document.createElement('p');
                pipeline.textContent = buildName(point);
                const measurements = document.createElement('dl');
                measurements.className = 'technical-measurements';
                const ram = point.metrics?.extraction_ram_mb ?? point.ram_mb ?? point.extraction_ram_mb;
                const values = [['Compute time', formatSeconds(latency(point))],
                    [metricInfo(this.taskType).label, formatPerformance({score: score(point), task_type: this.taskType})],
                    ['Extraction memory', Number.isFinite(ram) ? `${ram.toFixed(1)} MB` : '—']];
                for (const [label, value] of values) {
                    const dt = document.createElement('dt'), dd = document.createElement('dd');
                    dt.textContent = label; dd.textContent = value;
                    measurements.append(dt, dd);
                }
                this.tooltipEl.append(title, pipeline, measurements);
            }
        }
        this.render();
    }

    _getBounds() {
        const points = [...this.allPoints, ...this.racerPoints].filter(valid);
        const times = points.map(latency), scores = points.map(score);
        const low = scores.length ? Math.min(...scores) : 0;
        const high = scores.length ? Math.max(...scores) : 1;
        const pad = Math.max((high - low) * .12, Math.abs(high) * .02, .01);
        return {
            logMinLat: Math.log10(times.length ? Math.min(...times) * .7 : .01),
            logMaxLat: Math.log10(times.length ? Math.max(...times) * 1.5 : 100),
            minScore: this.taskType === 'classification' ? Math.max(0, low - pad) : low - pad,
            maxScore: this.taskType === 'classification' ? Math.min(1, high + pad) : high + pad,
        };
    }

    _toScreenCoords(lat, value) {
        const b = this._getBounds();
        const w = this.displayWidth || this.canvas.width;
        const h = this.displayHeight || this.canvas.height;
        const quality = (value - b.minScore) / (b.maxScore - b.minScore);
        return {
            x: this.padding.left + (Math.log10(lat) - b.logMinLat) / (b.logMaxLat - b.logMinLat) * (w - this.padding.left - this.padding.right),
            y: this.padding.top + (this.taskType === 'forecasting' ? quality : 1 - quality) * (h - this.padding.top - this.padding.bottom),
        };
    }

    render() {
        if (!this.displayWidth || !this.displayHeight) return;
        const ctx = this.ctx, w = this.displayWidth, h = this.displayHeight, p = this.padding;
        ctx.clearRect(0, 0, w, h);
        const b = this._getBounds(), plotW = w - p.left - p.right, plotH = h - p.top - p.bottom;
        ctx.fillStyle = '#F0ECF6';
        ctx.fillRect(p.left, p.top, plotW, plotH);
        ctx.font = '10px system-ui, sans-serif';
        ctx.fillStyle = '#71617E'; ctx.strokeStyle = '#DCD3E5'; ctx.lineWidth = 1;
        ctx.textAlign = 'right';
        for (let i = 0; i <= 4; i++) {
            const quality = this.taskType === 'forecasting' ? i / 4 : 1 - i / 4;
            const value = b.minScore + quality * (b.maxScore - b.minScore);
            const y = p.top + i / 4 * plotH;
            ctx.beginPath(); ctx.moveTo(p.left, y); ctx.lineTo(w - p.right, y); ctx.stroke();
            const label = this.taskType === 'classification' ? `${(value * 100).toFixed(0)}%` : Number(value.toPrecision(3)).toString();
            ctx.fillText(label, p.left - 8, y + 3);
        }
        ctx.textAlign = 'center';
        for (const tick of logarithmicTicks(b.logMinLat, b.logMaxLat, plotW)) {
            const x = p.left + tick.position * plotW;
            ctx.beginPath(); ctx.moveTo(x, p.top); ctx.lineTo(x, h - p.bottom);
            ctx.strokeStyle = tick.major ? '#CCBEDB' : '#E2DAEB';
            ctx.lineWidth = tick.major ? 1 : .65; ctx.stroke();
            if (tick.label) {
                ctx.beginPath(); ctx.moveTo(x, h - p.bottom); ctx.lineTo(x, h - p.bottom + 4);
                ctx.strokeStyle = '#927BA6'; ctx.lineWidth = 1; ctx.stroke();
                const label = `${Number(tick.value.toPrecision(2))} s`;
                const halfLabel = ctx.measureText(label).width / 2;
                const labelX = Math.max(halfLabel + 2, Math.min(w - halfLabel - 2, x));
                ctx.fillText(label, labelX, h - p.bottom + 19);
            }
        }
        ctx.beginPath(); ctx.moveTo(p.left, p.top); ctx.lineTo(p.left, h - p.bottom); ctx.lineTo(w - p.right, h - p.bottom);
        ctx.strokeStyle = '#927BA6'; ctx.lineWidth = 1.2; ctx.stroke();
        ctx.fillStyle = '#564760';
        ctx.fillText('Compute time · logarithmic scale', p.left + plotW / 2, h - 9);
        ctx.textAlign = 'left';
        ctx.fillText(`${metricInfo(this.taskType).label} · better ↑`, p.left, 13);

        const dot = (point, radius, colour) => {
            const pos = this._toScreenCoords(latency(point), score(point));
            ctx.beginPath(); ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
            ctx.fillStyle = colour; ctx.fill();
        };
        this.allPoints.forEach(point => dot(point, 2.75, '#AD9DBB'));
        ctx.beginPath();
        this.paretoPoints.forEach((point, i) => {
            const pos = this._toScreenCoords(point.latency_s, point.score);
            if (i) ctx.lineTo(pos.x, pos.y); else ctx.moveTo(pos.x, pos.y);
        });
        ctx.strokeStyle = '#4E2277'; ctx.lineWidth = 2; ctx.stroke();
        this.paretoPoints.forEach(point => dot(point, 3.5, '#4E2277'));
        // Segmented rings keep coincident participants visible at their true measurements.
        const groups = new Map();
        for (const point of this.racerPoints) {
            const key = `${latency(point)}:${score(point)}`;
            if (!groups.has(key)) groups.set(key, []);
            groups.get(key).push(point);
        }
        for (const group of groups.values()) group.forEach((point, i) => {
            const pos = this._toScreenCoords(latency(point), score(point));
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, 6, i / group.length * Math.PI * 2, (i + 1) / group.length * Math.PI * 2);
            ctx.strokeStyle = point.avatar_color || '#4E2277'; ctx.lineWidth = 3; ctx.stroke();
        });
        if (this.selectedPoint) {
            const pos = this._toScreenCoords(latency(this.selectedPoint), score(this.selectedPoint));
            ctx.beginPath(); ctx.arc(pos.x, pos.y, 11, 0, Math.PI * 2);
            ctx.strokeStyle = '#20182B'; ctx.lineWidth = 1; ctx.stroke();
        }
    }
}
