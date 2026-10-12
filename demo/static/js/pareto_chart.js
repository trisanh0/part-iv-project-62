/**
 * TEMPO Display Day Pareto Frontier Canvas Plotter.
 *
 * Renders an interactive, touch-friendly 2D scatter plot mapping empirical
 * Latency (log scale) vs Accuracy/Score, with Pareto frontier curve and racer positions.
 */

export class ParetoCanvasChart {
    /**
     * @param {HTMLCanvasElement} canvas
     * @param {HTMLElement} tooltipEl Optional DOM element for details display
     */
    constructor(canvas, tooltipEl = null) {
        this.canvas = canvas;
        this.ctx = canvas.getContext("2d");
        this.tooltipEl = tooltipEl;

        this.allPoints = [];
        this.paretoPoints = [];
        this.racerPoints = [];
        this.selectedPoint = null;

        this.padding = { top: 30, right: 30, bottom: 50, left: 60 };
        this._setupListeners();
    }

    _setupListeners() {
        const handleInteraction = (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const clientX = e.touches ? e.touches[0].clientX : e.clientX;
            const clientY = e.touches ? e.touches[0].clientY : e.clientY;
            const x = clientX - rect.left;
            const y = clientY - rect.top;

            this._checkHitPoint(x, y);
        };

        this.canvas.addEventListener("click", handleInteraction);
        this.canvas.addEventListener("touchstart", handleInteraction, { passive: true });

        window.addEventListener("resize", () => this.resize());
        if (typeof ResizeObserver !== "undefined") {
            this.resizeObserver = new ResizeObserver(() => this.resize());
            if (this.canvas.parentElement) {
                this.resizeObserver.observe(this.canvas.parentElement);
            }
            this.resizeObserver.observe(this.canvas);
        }
        setTimeout(() => this.resize(), 50);
    }

    /**
     * Measure and configure backing buffer dimensions.
     */
    resize() {
        const rect = this.canvas.getBoundingClientRect();
        let w = rect.width;
        let h = rect.height;

        if (w <= 0 || h <= 0) {
            const parent = this.canvas.parentElement;
            if (parent && parent.clientWidth > 0) {
                w = parent.clientWidth;
                h = parent.clientHeight > 0 ? parent.clientHeight : Math.round(w * (3 / 4));
            } else {
                w = Math.min(800, Math.max(320, (window.innerWidth || 600) - 32));
                h = Math.round(w * (3 / 4));
            }
        }

        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = Math.floor(w * dpr);
        this.canvas.height = Math.floor(h * dpr);

        if (this.ctx.resetTransform) {
            this.ctx.resetTransform();
        } else {
            this.ctx.setTransform(1, 0, 0, 1, 0, 0);
        }
        this.ctx.scale(dpr, dpr);
        this.displayWidth = w;
        this.displayHeight = h;

        this.render();
    }

    /**
     * Set data and draw chart.
     */
    setData(allPoints, paretoFrontier, racers = []) {
        this.resize();
        this.allPoints = allPoints || [];
        this.paretoPoints = (paretoFrontier || []).sort((a, b) => a.latency_s - b.latency_s);
        this.racerPoints = racers || [];
        this.selectedPoint = null;
        this.render();
    }

    _checkHitPoint(clickX, clickY) {
        const threshold = 18;
        let closest = null;
        let minDist = threshold;

        // Check racers first (highest priority)
        const candidates = [...this.racerPoints, ...this.paretoPoints, ...this.allPoints];

        for (const pt of candidates) {
            const lat = pt.metrics ? pt.metrics.total_latency_s : (pt.latency_s ?? pt.total_latency_s);
            const score = pt.metrics ? pt.metrics.score : pt.score;
            if (lat === undefined || score === undefined) continue;

            const screen = this._toScreenCoords(lat, score);
            const dist = Math.hypot(screen.x - clickX, screen.y - clickY);
            if (dist < minDist) {
                minDist = dist;
                closest = pt;
            }
        }

        this.selectedPoint = closest;
        this.render();

        if (this.tooltipEl && closest) {
            const title = closest.nickname ? `${closest.nickname} (${closest.pipeline?.extractor || closest.extractor || "TEMPO"})` : (closest.extractor || "Configuration");
            const sel = closest.pipeline?.selector || closest.selector || "None";
            const mod = closest.pipeline?.model || closest.model || "None";
            const lat = closest.metrics ? closest.metrics.total_latency_s : (closest.latency_s ?? closest.total_latency_s ?? 0);
            const score = closest.metrics ? closest.metrics.score : (closest.score ?? 0);
            const ram = closest.metrics ? closest.metrics.extraction_ram_mb : (closest.ram_mb ?? closest.extraction_ram_mb ?? 0);
            const speedup = closest.metrics ? closest.metrics.speedup_vs_tsfresh : (closest.speedup_vs_tsfresh ?? 1.0);

            this.tooltipEl.innerHTML = `
                <div style="font-weight: 700; font-size: 0.9rem; color: #1A1A2E; margin-bottom: 4px;">${title}</div>
                <div style="font-size: 0.75rem; color: #6B7185; line-height: 1.5;">
                    Selector: <span style="color: #1A1A2E;">${sel}</span> | Model: <span style="color: #1A1A2E;">${mod}</span><br>
                    Latency: <span style="color: #4E2277; font-family: ui-monospace, monospace;">${lat.toFixed(3)}s</span><br>
                    Score: <span style="color: #0F8B5B; font-family: ui-monospace, monospace;">${(score * 100).toFixed(1)}%</span><br>
                    Peak RAM: <span style="color: #D4A843; font-family: ui-monospace, monospace;">${ram.toFixed(1)} MB</span><br>
                    Speedup vs tsfresh: <span style="color: #D4A843; font-weight: 600;">${speedup.toFixed(1)}x</span>
                </div>
            `;
            this.tooltipEl.style.display = "block";
        } else if (this.tooltipEl) {
            this.tooltipEl.style.display = "none";
        }
    }

    _getBounds() {
        const allLatencies = [
            ...this.allPoints.map(p => p.latency_s),
            ...this.racerPoints.map(p => p.metrics?.total_latency_s || 1.0),
        ].filter(v => v > 0);

        const allScores = [
            ...this.allPoints.map(p => p.score),
            ...this.racerPoints.map(p => p.metrics?.score || 0.8),
        ];

        const minLat = allLatencies.length ? Math.min(...allLatencies) : 0.01;
        const maxLat = allLatencies.length ? Math.max(...allLatencies) : 100.0;
        const minScore = allScores.length ? Math.min(...allScores) : 0.8;
        const maxScore = allScores.length ? Math.max(...allScores) : 1.0;

        return {
            logMinLat: Math.log10(Math.max(0.001, minLat * 0.7)),
            logMaxLat: Math.log10(maxLat * 1.5),
            minScore: Math.max(0.0, minScore - 0.03),
            maxScore: Math.min(1.02, maxScore + 0.02),
        };
    }

    _toScreenCoords(lat, score) {
        const bounds = this._getBounds();
        const width = this.displayWidth || this.canvas.width;
        const height = this.displayHeight || this.canvas.height;

        const plotW = width - this.padding.left - this.padding.right;
        const plotH = height - this.padding.top - this.padding.bottom;

        const logLat = Math.log10(Math.max(0.001, lat));
        const normX = (logLat - bounds.logMinLat) / (bounds.logMaxLat - bounds.logMinLat);
        const normY = (score - bounds.minScore) / (bounds.maxScore - bounds.minScore);

        const x = this.padding.left + normX * plotW;
        const y = this.padding.top + (1.0 - normY) * plotH;

        return { x, y };
    }

    render() {
        if (!this.displayWidth || this.displayWidth <= 0 || !this.displayHeight || this.displayHeight <= 0) {
            this.resize();
        }
        const width = this.displayWidth || 800;
        const height = this.displayHeight || 600;
        const ctx = this.ctx;

        ctx.clearRect(0, 0, width, height);

        // Chart area background
        ctx.fillStyle = "#F7F8FA";
        ctx.fillRect(0, 0, width, height);

        const bounds = this._getBounds();
        const plotW = width - this.padding.left - this.padding.right;
        const plotH = height - this.padding.top - this.padding.bottom;

        // Grid lines & Axis Labels
        ctx.strokeStyle = "#E8E9ED";
        ctx.lineWidth = 1;

        // Horizontal score grids
        const scoreSteps = 5;
        ctx.fillStyle = "#8B8FA3";
        ctx.font = "10px system-ui, sans-serif";
        ctx.textAlign = "right";

        for (let i = 0; i <= scoreSteps; i++) {
            const scoreVal = bounds.minScore + (i / scoreSteps) * (bounds.maxScore - bounds.minScore);
            const y = this.padding.top + (1.0 - i / scoreSteps) * plotH;

            ctx.beginPath();
            ctx.moveTo(this.padding.left, y);
            ctx.lineTo(this.padding.left + plotW, y);
            ctx.stroke();

            ctx.fillText(`${(scoreVal * 100).toFixed(0)}%`, this.padding.left - 8, y + 3);
        }

        // Vertical Log-Latency grids
        ctx.textAlign = "center";
        const minDecade = Math.floor(bounds.logMinLat);
        const maxDecade = Math.ceil(bounds.logMaxLat);

        for (let d = minDecade; d <= maxDecade; d++) {
            const normX = (d - bounds.logMinLat) / (bounds.logMaxLat - bounds.logMinLat);
            if (normX >= 0 && normX <= 1) {
                const x = this.padding.left + normX * plotW;
                ctx.beginPath();
                ctx.moveTo(x, this.padding.top);
                ctx.lineTo(x, this.padding.top + plotH);
                ctx.stroke();

                const secVal = Math.pow(10, d);
                const label = secVal < 1 ? `${secVal.toFixed(2)}s` : `${secVal.toFixed(0)}s`;
                ctx.fillText(label, x, this.padding.top + plotH + 18);
            }
        }

        // Axis Titles
        ctx.fillStyle = "#6B7185";
        ctx.font = "bold 11px system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("Total Latency (Log Scale, s)", this.padding.left + plotW / 2, height - 12);

        ctx.save();
        ctx.translate(16, this.padding.top + plotH / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText("Score / Accuracy", 0, 0);
        ctx.restore();

        // 1. Draw all background configurations (grey circles)
        this.allPoints.forEach((pt) => {
            const coords = this._toScreenCoords(pt.latency_s, pt.score);
            ctx.fillStyle = "#C8CBD3";
            ctx.beginPath();
            ctx.arc(coords.x, coords.y, 3, 0, Math.PI * 2);
            ctx.fill();
        });

        // 2. Draw Pareto Frontier Curve (Cyan Line + shaded dots)
        if (this.paretoPoints.length > 1) {
            ctx.beginPath();
            this.paretoPoints.forEach((pt, idx) => {
                const coords = this._toScreenCoords(pt.latency_s, pt.score);
                if (idx === 0) ctx.moveTo(coords.x, coords.y);
                else ctx.lineTo(coords.x, coords.y);
            });
            ctx.strokeStyle = "#4E2277";
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        this.paretoPoints.forEach((pt) => {
            const coords = this._toScreenCoords(pt.latency_s, pt.score);
            ctx.fillStyle = "#4E2277";
            ctx.beginPath();
            ctx.arc(coords.x, coords.y, 5, 0, Math.PI * 2);
            ctx.fill();
        });

        // 3. Highlight Racers in completed heat
        this.racerPoints.forEach((r) => {
            const m = r.metrics;
            if (!m) return;
            const coords = this._toScreenCoords(m.total_latency_s, m.score);

            // Pulsing highlight circle
            ctx.fillStyle = r.avatar_color || "#D4A843";
            ctx.beginPath();
            ctx.arc(coords.x, coords.y, 9, 0, Math.PI * 2);
            ctx.fill();

            ctx.strokeStyle = "#1A1A2E";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(coords.x, coords.y, 11, 0, Math.PI * 2);
            ctx.stroke();

            // Label
            ctx.fillStyle = "#1A1A2E";
            ctx.font = "bold 11px system-ui, sans-serif";
            ctx.textAlign = "center";
            ctx.fillText(r.nickname, coords.x, coords.y - 15);
        });

        // 4. Selection ring
        if (this.selectedPoint) {
            const lat = this.selectedPoint.latency_s || this.selectedPoint.total_latency_s || 1.0;
            const score = this.selectedPoint.score || 0.9;
            const coords = this._toScreenCoords(lat, score);

            ctx.strokeStyle = "#D4A843";
            ctx.lineWidth = 2;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.arc(coords.x, coords.y, 16, 0, Math.PI * 2);
            ctx.stroke();
            ctx.setLineDash([]);
        }
    }
}
