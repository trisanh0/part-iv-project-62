/**
 * TEMPO open-wheel race and live timing renderer.
 *
 * Runs the 60 FPS requestAnimationFrame animation loop, mapping empirical
 * measured ML latency to race progress and authoritative finish times.
 */

import {drawVehicle} from './vehicles.js';
import {scheduledMotion, PIT_ENTRY_PROGRESS, PIT_EXIT_PROGRESS} from './race_motion.js';
import { TrackCircuit, CIRCUIT_WIDTH, CIRCUIT_HEIGHT, CIRCUIT_PALETTE } from "./track.js";

export class RaceCanvasRenderer {
    /**
     * @param {HTMLCanvasElement} canvas
     */
    constructor(canvas, leaderboard = null, onTelemetry = null) {
        this.canvas = canvas;
        this.leaderboard = leaderboard;
        this.onTelemetry = onTelemetry;
        this.ctx = canvas.getContext("2d");
        this.circuit = new TrackCircuit();

        this.racers = [];
        this.raceStartTimeMs = 0;
        this.raceDurationSec = 27.0;
        this.totalLaps = 2;
        this.isRunning = false;
        this.animationFrameId = null;

        this._setupResizeListener();
    }

    _setupResizeListener() {
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
                h = parent.clientHeight > 0 ? parent.clientHeight : Math.round(w * (9 / 16));
            } else {
                w = Math.min(1000, Math.max(320, (window.innerWidth || 800) - 32));
                h = Math.round(w * (9 / 16));
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

        if (!this.isRunning) {
            this.renderFrame(0);
        }
    }

    /**
     * Set active racers and race parameters.
     */
    setupRace(racers, durationSec = 27.0, startTimeMs = 0, totalLaps = 2) {
        this.totalLaps = totalLaps;
        this.resize();
        this.raceDurationSec = durationSec;
        this.raceStartTimeMs = startTimeMs || Date.now();

        // Derive deterministic pace from each entrant’s measured compute time.
        this.racers = racers.map((r, idx) => {
            const m = r.metrics || {};
            const extractionTime = Math.max(0.01, m.extraction_time_s || 1.0);
            const totalLatency = Math.max(0.01, m.total_latency_s || extractionTime);

            // Log compression preserves latency ordering without making fast cars vanish.
            const speedFactor = 1.0 / (1.0 + 0.12 * Math.log1p(totalLatency));

            return {
                ...r,
                speedFactor,
                pacePhase: (r.car_index ?? idx) * Math.PI * (3 - Math.sqrt(5)),
                totalLatency,
                extractionTime,
                progress: 0.0,
                lateralOffset: (idx % 2 === 0 ? -1 : 1) * 15,
                gridRow: Math.floor(idx / 2),
                isFinished: false,
                finishTimeSec: null,
            };
        });
        this.leaderboardRows = new Map();
        this.leaderboard?.replaceChildren();
        this.lastLeaderboardTick = null;
        const slowestSpeed = Math.min(...this.racers.map(car => car.speedFactor));
        for (const car of this.racers) {
            car.completionTimeSec = car.race_schedule ? car.race_schedule.finish_ms / 1000 : this.raceDurationSec * (slowestSpeed / car.speedFactor);
        }

    }

    /**
     * Start the continuous animation loop.
     */
    start() {
        this.stop();
        this.resize();
        this.isRunning = true;
        const tick = () => {
            if (!this.isRunning) return;
            const elapsedSec = (Date.now() - this.raceStartTimeMs) / 1000.0;
            this.renderFrame(elapsedSec);

            if (elapsedSec < this.raceDurationSec + 2.5) {
                this.animationFrameId = requestAnimationFrame(tick);
            } else {
                this.isRunning = false;
            }
        };
        this.animationFrameId = requestAnimationFrame(tick);
    }

    /**
     * Stop active animation loop.
     */
    stop() {
        this.isRunning = false;
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
    }

    /**
     * Render a single discrete frame given elapsed seconds since race start.
     */
    renderFrame(elapsedSec) {
        if (!this.displayWidth || this.displayWidth <= 0 || !this.displayHeight || this.displayHeight <= 0) {
            this.resize();
        }

        const width = this.displayWidth || 800;
        const height = this.displayHeight || 450;
        const ctx = this.ctx;

        ctx.clearRect(0, 0, width, height);

        // Carry the page surface through the circuit.
        ctx.fillStyle = CIRCUIT_PALETTE.paper;
        ctx.fillRect(0, 0, width, height);

        // Render circuit tarmac and obtain viewport coordinate scale
        const scale = this.circuit.render(ctx, width, height);
        const offsetX = (width - CIRCUIT_WIDTH * scale) / 2;
        const offsetY = (height - CIRCUIT_HEIGHT * scale) / 2;

        this.driverLabels = [];
        // Update and draw each competitor vehicle
        this.racers.forEach((car) => {
            // Every car moves from t=0; the slowest completes at the room deadline.
            const fraction = Math.min(1, Math.max(0, elapsedSec) / car.completionTimeSec);
            // One gentle pace wave per race. Speed modulation stays below 9%,
            // with no displacement or speed change at either endpoint.
            const variation = 0.009 * Math.sin(Math.PI * fraction) ** 2
                * Math.sin(2 * Math.PI * fraction + car.pacePhase);
            const elapsedLaps = this.totalLaps * (fraction + variation);
            // A common monotonic time-to-distance map slows corners while preserving
            // completion times, tied finishes and deterministic reconnect playback.
            const motion = car.race_schedule
                ? scheduledMotion(car.race_schedule, elapsedSec * 1000, this.circuit, car.pacePhase, this.totalLaps) : null;
            const lapProgress = motion ? motion.progress : this.circuit.progressAtTime(elapsedLaps);
            car.inPit = !!motion?.inPit;
            car.dwelling = !!motion?.dwelling;
            car.pitRemaining = motion?.pitRemaining || 0;
            car.progress = lapProgress;
            car.isFinished = lapProgress >= this.totalLaps;
            car.finishTimeSec = car.isFinished ? car.completionTimeSec : null;

            // Sample track coordinate at instantaneous progress
            // A two-column grid converges onto the race line after launch.
            // This visual grid spacing does not change measured progress or finish times.
            const launch = Math.min(1, lapProgress / .12);
            const gridFade = 1 - launch * launch * (3 - 2 * launch);
            const gridSpacing = (car.gridRow * 60) / (this.circuit.totalTrackLength || 4000);
            let visualProgress = lapProgress;
            if (car.isFinished && this.circuit.totalTrackLength) {
                const coastSeconds = 1.8 + (car.car_index || 0) * .1;
                const u = Math.min(1, Math.max(0, elapsedSec - car.completionTimeSec) / coastSeconds);
                const prior = car.race_schedule
                    ? scheduledMotion(car.race_schedule, (car.completionTimeSec - .001) * 1000, this.circuit, car.pacePhase, this.totalLaps).progress
                    : this.circuit.progressAtTime(this.totalLaps - this.totalLaps * .001 / car.completionTimeSec);
                const velocity = (this.totalLaps - prior) / .001;
                visualProgress += velocity * coastSeconds / 3 * (1 - (1-u)**3);
            }
            car.visualProgress = visualProgress;
            const pt = motion?.inPit ? this.circuit.samplePit(motion.pitRoute)
                : this.circuit.sample(visualProgress - gridSpacing * gridFade);

            const pitBlend = car.race_schedule?.uses_accuracy && !car.inPit
                ? Math.min(1, Math.abs(lapProgress - PIT_ENTRY_PROGRESS) / .025, Math.abs(lapProgress - PIT_EXIT_PROGRESS) / .025) : 1;
            const smoothPitBlend = pitBlend * pitBlend * (3 - 2 * pitBlend);
            const laneScale = car.inPit ? 0 : (gridFade + (1 - gridFade) * (pt.laneScale ?? 1)) * smoothPitBlend;
            const dynamicOffset = car.lateralOffset * laneScale;

            // Perpendicular unit vector to heading
            const perpX = -Math.sin(pt.heading);
            const perpY = Math.cos(pt.heading);

            const carVirtualX = pt.x + perpX * dynamicOffset;
            const carVirtualY = pt.y + perpY * dynamicOffset;

            // Transform virtual coordinate to canvas screen space
            const screenX = offsetX + carVirtualX * scale;
            const screenY = offsetY + carVirtualY * scale;

            this._drawDriverMarker(ctx, screenX, screenY, car, scale, pt.heading);
        });

        this._drawLabels(ctx, width, height);
        this._updateLeaderboard(elapsedSec);
    }

    /** Render a compact open-wheel silhouette aligned to the racing line. */
    _drawDriverMarker(ctx, x, y, car, scale, heading = 0) {
        const mine = !!this.viewerClientId && car.client_id === this.viewerClientId;
        const color = car.avatar_color || CIRCUIT_PALETTE.ink;
        const length = Math.max(20, Math.min(28, 52 * scale));
        const width = length * .44;
        ctx.save(); ctx.translate(x, y); ctx.rotate(heading);
        if (mine) {
            ctx.strokeStyle = CIRCUIT_PALETTE.ink; ctx.lineWidth = .9;
            ctx.beginPath(); ctx.ellipse(0, 0, length * .67, width, 0, 0, Math.PI * 2); ctx.stroke();
        }
        drawVehicle(ctx, car.vehicle || 'formula', color, length);
        ctx.restore();
        if (mine) this.driverLabels.push({x, y, mine, name: "YOU"});
    }

    _drawLabels(ctx, width, height) {
        const occupied = [];
        for (const label of this.driverLabels.sort((a, b) => Number(b.mine) - Number(a.mine))) {
            ctx.save(); ctx.font = `${label.mine ? 600 : 400} 12px system-ui`;
            const name = label.name.length > 14 ? label.name.slice(0, 13) + "…" : label.name;
            const text = label.mine ? "YOU" : name;
            const w = ctx.measureText(text).width, h = 18;
            let box;
            for (const [dx, dy] of [[12, -9], [-w - 12, -9], [12, 12], [-w - 12, 12], [12, -30], [-w - 12, -30], [12, 33], [-w - 12, 33]]) {
                const candidate = {x: Math.max(6, Math.min(width - w - 6, label.x + dx)),
                    y: Math.max(6, Math.min(height - h - 6, label.y + dy)), w, h};
                if (!occupied.some(b => candidate.x < b.x + b.w + 5 && candidate.x + w + 5 > b.x && candidate.y < b.y + b.h + 3 && candidate.y + h + 3 > b.y)) {
                    box = candidate; break;
                }
            }
            if (!box) { ctx.restore(); continue; }
            occupied.push(box);
            ctx.strokeStyle = CIRCUIT_PALETTE.labelLine; ctx.lineWidth = .7;
            const anchor = box.x > label.x ? box.x - 3 : box.x + w + 3;
            ctx.beginPath(); ctx.moveTo(label.x, label.y); ctx.lineTo(anchor, box.y + h / 2); ctx.stroke();
            // A paper halo keeps names legible when the circuit passes underneath.
            ctx.textAlign = "left"; ctx.lineJoin = "round";
            ctx.strokeStyle = CIRCUIT_PALETTE.paper; ctx.lineWidth = 4;
            ctx.strokeText(text, box.x, box.y + 13);
            ctx.fillStyle = label.mine ? CIRCUIT_PALETTE.ink : CIRCUIT_PALETTE.label;
            ctx.fillText(text, box.x, box.y + 13);
            ctx.restore();
        }
    }

    getStandings() {
        return [...this.racers].sort((a, b) =>
            b.progress - a.progress || a.completionTimeSec - b.completionTimeSec ||
            String(a.client_id || "").localeCompare(String(b.client_id || "")));
    }

    _updateLeaderboard(elapsedSec) {
        if (!this.leaderboard) return;
        const tick = Math.floor(Math.max(0, elapsedSec) * 10);
        if (tick === this.lastLeaderboardTick) return;
        this.lastLeaderboardTick = tick;
        const standings = this.getStandings();
        if (this.onTelemetry) {
            const own = standings.find(r => r.client_id === this.viewerClientId);
            this.onTelemetry({elapsed: Math.max(0, Math.min(elapsedSec, this.raceDurationSec)),
                position: own && elapsedSec > 0 ? standings.findIndex(r => r === own ||
                    (own.isFinished && r.isFinished && r.completionTimeSec === own.completionTimeSec)) + 1 : null,
                lap: own ? Math.min(this.totalLaps, Math.floor(own.progress) + 1) : null,
                finish: own?.isFinished ? {
                    position: 1 + this.racers.filter(r => r.completionTimeSec < own.completionTimeSec).length,
                    tied: this.racers.filter(r => r.completionTimeSec === own.completionTimeSec).length > 1,
                    timeMs: Math.round(own.completionTimeSec * 1000),
                } : null});
        }
        standings.forEach((car, index) => {
            let row = this.leaderboardRows.get(car);
            if (!row) {
                row = document.createElement("li");
                row.className = "live-racer";
                for (const className of ["live-position", "live-name", "live-progress"]) {
                    const cell = document.createElement("span");
                    cell.className = className;
                    row.appendChild(cell);
                }
                row.children[1].textContent = car.client_id === this.viewerClientId ? "You" : this.showNames ? car.nickname : car.is_bot ? "Benchmark" : `Racer ${car.car_index + 1}`;
                row.classList.toggle("is-you", !!this.viewerClientId && car.client_id === this.viewerClientId);
                row.style.setProperty("--racer-color", car.avatar_color || "#5C9FA0");
                this.leaderboardRows.set(car, row);
            }
            const previous = standings[index - 1];
            const tied = car.isFinished && previous?.isFinished && car.completionTimeSec === previous.completionTimeSec;
            row.children[0].textContent = tied ? "=" : `P${index + 1}`;
            row.children[2].textContent = car.isFinished
                ? `Finished · ${car.finishTimeSec.toFixed(3)}s`
                : car.inPit ? (car.dwelling ? `In pits · accuracy penalty ${car.pitRemaining.toFixed(1)}s` : `Pit lane · ${car.pitRemaining.toFixed(1)}s penalty`)
                : `Lap ${Math.min(this.totalLaps, Math.floor(car.progress) + 1)}/${this.totalLaps} · ${Math.floor(car.progress / this.totalLaps * 100)}%`;
            this.leaderboard.appendChild(row);
        });
    }
}
