/**
 * TEMPO Display Day 2D Canvas Race Renderer.
 *
 * Runs the 60 FPS requestAnimationFrame animation loop, mapping empirical
 * ML latency to vehicle throttle/staging delay, and classification score to cornering grip.
 */

import { TrackCircuit } from "./track.js";

export class RaceCanvasRenderer {
    /**
     * @param {HTMLCanvasElement} canvas
     */
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext("2d");
        this.circuit = new TrackCircuit();

        this.racers = [];
        this.raceStartTimeMs = 0;
        this.raceDurationSec = 14.0;
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
    setupRace(racers, durationSec = 14.0, startTimeMs = 0) {
        this.resize();
        this.raceDurationSec = durationSec;
        this.raceStartTimeMs = startTimeMs || Date.now();

        // Calculate dynamic physics weights for each racer based on their real metrics
        this.racers = racers.map((r, idx) => {
            const m = r.metrics || {};
            const extractionTime = Math.max(0.01, m.extraction_time_s || 1.0);
            const totalLatency = Math.max(0.01, m.total_latency_s || extractionTime);
            const score = Math.max(0.5, Math.min(1.0, m.score || 0.9));

            // Staging delay simulates extraction compute overhead before car launches
            // Logarithmic compression keeps max wait to 3.0s while showing clear disparity
            const launchDelaySec = Math.min(3.2, Math.max(0.2, Math.log(1 + extractionTime) * 0.7));

            // Run speed multiplier
            const availableDriveSec = Math.max(2.0, this.raceDurationSec - launchDelaySec);
            const speedFactor = 1.0 / (1.0 + 0.12 * Math.log(1 + totalLatency));

            return {
                ...r,
                launchDelaySec,
                availableDriveSec,
                speedFactor,
                score,
                totalLatency,
                extractionTime,
                progress: 0.0,
                lateralOffset: (idx - (racers.length - 1) / 2) * 10, // Lane grid spacing
                isFinished: false,
                finishTimeSec: null,
            };
        });
    }

    /**
     * Start the continuous animation loop.
     */
    start() {
        this.resize();
        this.isRunning = true;
        const tick = () => {
            if (!this.isRunning) return;
            const elapsedSec = (Date.now() - this.raceStartTimeMs) / 1000.0;
            this.renderFrame(elapsedSec);

            if (elapsedSec < this.raceDurationSec + 2.0) {
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

        // Draw background terrain
        ctx.fillStyle = "#F0F1F4";
        ctx.fillRect(0, 0, width, height);

        // Render circuit tarmac and obtain viewport coordinate scale
        const scale = this.circuit.render(ctx, width, height);
        const offsetX = (width - 1000 * scale) / 2;
        const offsetY = (height - 600 * scale) / 2;

        // Update and draw each competitor vehicle
        this.racers.forEach((car) => {
            // Compute vehicle progression
            let lapProgress = 0.0;
            let currentStatus = "EXTRACTING";

            if (elapsedSec > car.launchDelaySec) {
                const drivingElapsed = elapsedSec - car.launchDelaySec;
                // Target: fastest reaches 2.0 laps around raceDurationSec
                const completionRate = drivingElapsed / car.availableDriveSec;
                lapProgress = Math.min(this.totalLaps, completionRate * this.totalLaps * car.speedFactor);
                currentStatus = lapProgress >= this.totalLaps ? "FINISHED" : `LAP ${Math.min(this.totalLaps, Math.floor(lapProgress) + 1)}/${this.totalLaps}`;
            }

            car.progress = lapProgress;
            if (lapProgress >= this.totalLaps && !car.isFinished) {
                car.isFinished = true;
                car.finishTimeSec = elapsedSec;
            }

            // Sample track coordinate at instantaneous progress
            const pt = this.circuit.sample(lapProgress);

            // Compute cornering slip governed by (1 - Accuracy)
            // Low accuracy slides wide on sharp corners (high curvature)
            const slipMagnitude = (1.0 - car.score) * 60.0 * (pt.curvature * 15.0);
            const dynamicOffset = car.lateralOffset + slipMagnitude * Math.sin(elapsedSec * 6);

            // Perpendicular unit vector to heading
            const perpX = -Math.sin(pt.heading);
            const perpY = Math.cos(pt.heading);

            const carVirtualX = pt.x + perpX * dynamicOffset;
            const carVirtualY = pt.y + perpY * dynamicOffset;

            // Transform virtual coordinate to canvas screen space
            const screenX = offsetX + carVirtualX * scale;
            const screenY = offsetY + carVirtualY * scale;

            this._drawCarSprite(ctx, screenX, screenY, pt.heading, car, scale, currentStatus);
        });

        // Draw Telemetry Leaderboard Overlay
        this._drawLeaderboardOverlay(ctx, width, height, elapsedSec);
    }

    /**
     * Draw geometric top-down racecar sprite with wheels and nametag.
     */
    _drawCarSprite(ctx, x, y, heading, car, scale, status) {
        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(heading);

        const carLen = 28 * scale;
        const carWid = 16 * scale;

        // Tires (4 corners)
        ctx.fillStyle = "#B0B5C0";
        const tireLen = 8 * scale;
        const tireWid = 4 * scale;
        ctx.fillRect(-carLen / 2 + 2, -carWid / 2 - tireWid / 2, tireLen, tireWid);
        ctx.fillRect(carLen / 2 - tireLen - 2, -carWid / 2 - tireWid / 2, tireLen, tireWid);
        ctx.fillRect(-carLen / 2 + 2, carWid / 2 - tireWid / 2, tireLen, tireWid);
        ctx.fillRect(carLen / 2 - tireLen - 2, carWid / 2 - tireWid / 2, tireLen, tireWid);

        // Vehicle Chassis
        ctx.fillStyle = car.avatar_color || "#3B82F6";
        ctx.beginPath();
        ctx.roundRect(-carLen / 2, -carWid / 2, carLen, carWid, 4 * scale);
        ctx.fill();

        // Cockpit / Windshield
        ctx.fillStyle = "#FFFFFF";
        ctx.fillRect(-carLen * 0.1, -carWid * 0.3, carLen * 0.4, carWid * 0.6);

        // Front wing indicator
        ctx.fillStyle = "#1A1A2E";
        ctx.fillRect(carLen / 2 - 2, -carWid / 2, 2, carWid);

        ctx.restore();

        // Nametag & Status badge
        ctx.save();
        ctx.translate(x, y - 22 * scale);
        ctx.font = `bold ${Math.max(10, Math.floor(11 * scale))}px system-ui, sans-serif`;
        ctx.textAlign = "center";

        // Label background pill
        const labelText = `${car.nickname} (${car.pipeline?.extractor || "TEMPO"})`;
        const textWidth = ctx.measureText(labelText).width;
        ctx.fillStyle = "rgba(255, 255, 255, 0.92)";
        ctx.beginPath();
        ctx.roundRect(-textWidth / 2 - 6, -12, textWidth + 12, 16, 4);
        ctx.fill();

        // Label text
        ctx.fillStyle = car.avatar_color || "#1A1A2E";
        ctx.fillText(labelText, 0, 0);

        ctx.restore();
    }

    /**
     * Draw split-screen telemetry leaderboard in canvas upper-right corner.
     */
    _drawLeaderboardOverlay(ctx, width, height, elapsedSec) {
        const sorted = [...this.racers].sort((a, b) => b.progress - a.progress);

        ctx.save();
        const cardWidth = Math.min(260, width * 0.35);
        const cardX = width - cardWidth - 14;
        const cardY = 14;

        ctx.fillStyle = "rgba(255, 255, 255, 0.94)";
        ctx.strokeStyle = "#E2E4E9";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(cardX, cardY, cardWidth, 26 + sorted.length * 28, 8);
        ctx.fill();
        ctx.stroke();

        ctx.font = "bold 11px system-ui, sans-serif";
        ctx.fillStyle = "#6B7185";
        ctx.fillText("LIVE TELEMETRY STANDINGS", cardX + 12, cardY + 18);

        sorted.forEach((r, i) => {
            const rowY = cardY + 40 + i * 28;

            // Position badge
            ctx.fillStyle = i === 0 ? "#FBBF24" : i === 1 ? "#8B8FA3" : "#9CA3AF";
            ctx.font = "bold 12px system-ui, sans-serif";
            ctx.fillText(`P${i + 1}`, cardX + 12, rowY);

            // Car Color Dot
            ctx.fillStyle = r.avatar_color || "#3B82F6";
            ctx.beginPath();
            ctx.arc(cardX + 40, rowY - 4, 5, 0, Math.PI * 2);
            ctx.fill();

            // Name
            ctx.fillStyle = "#1A1A2E";
            ctx.font = "500 12px system-ui, sans-serif";
            const truncatedName = r.nickname.length > 10 ? r.nickname.substring(0, 9) + "…" : r.nickname;
            ctx.fillText(truncatedName, cardX + 52, rowY);

            // Score & Latency
            ctx.fillStyle = "#6B7185";
            ctx.font = "11px system-ui, sans-serif";
            ctx.textAlign = "right";
            const latencyStr = `${(r.totalLatency || 0).toFixed(1)}s`;
            ctx.fillText(latencyStr, cardX + cardWidth - 12, rowY);
            ctx.textAlign = "left";
        });

        ctx.restore();
    }
}
