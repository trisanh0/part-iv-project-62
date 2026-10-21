import {PIT_ENTRY_PROGRESS, PIT_EXIT_PROGRESS} from './race_motion.js';

/**
 * TEMPO Display Day 2D Circuit Track Geometry.
 *
 * Defines closed parametric bezier loops, racing lines, and apex curvature
 * for 2D canvas race rendering without external physics libraries.
 */

export const CIRCUIT_WIDTH = 1000;
export const CIRCUIT_HEIGHT = 1400;

export const CIRCUIT_PALETTE = Object.freeze({
    paper: '#FAF9FC', line: '#B8ADBF', ink: '#4E2277',
    asphalt: '#33343B', runoff: '#E2E7E2', kerb: '#A64E58',
    label: '#45374F', labelLine: '#C6BDCF',
});

export class TrackCircuit {
    /**
     * Initialize track with closed control points and normalized interpolation table.
     */
    constructor() {
        // Closed loop control waypoints scaled to a virtual 1000x1400 coordinate canvas
        // Original Grand Prix layout: pit straight, opening chicane, two fast
        // sweepers, an infield direction change and a broad final return.
        this.controlPoints = [
            {x: 220, y: 740}, {x: 220, y: 560}, {x: 220, y: 380},
            {x: 250, y: 230}, {x: 390, y: 145}, {x: 650, y: 145},
            {x: 810, y: 235}, {x: 840, y: 390}, {x: 730, y: 490},
            {x: 605, y: 520}, {x: 575, y: 640}, {x: 710, y: 720},
            {x: 815, y: 835}, {x: 750, y: 960}, {x: 585, y: 1020},
            {x: 520, y: 1135}, {x: 585, y: 1250}, {x: 470, y: 1310},
            {x: 325, y: 1270}, {x: 240, y: 1150}, {x: 220, y: 1030},
            {x: 220, y: 900},
        ];

        this.splineSamples = [];
        this.totalTrackLength = 0;
        this.sampleResolution = 600; // Number of equidistant arc-length lookup samples
        this._buildSplineLUT();
        this._buildPaceLUT();
        this._buildPitLane();
    }

    /**
     * Compute Catmull-Rom spline interpolation point between 4 control vertices.
     */
    _catmullRom(p0, p1, p2, p3, t) {
        const t2 = t * t;
        const t3 = t2 * t;

        const v0 = (p2.x - p0.x) * 0.5;
        const v1 = (p3.x - p1.x) * 0.5;
        const x = (2 * p1.x - 2 * p2.x + v0 + v1) * t3 +
                  (-3 * p1.x + 3 * p2.x - 2 * v0 - v1) * t2 +
                  v0 * t + p1.x;

        const u0 = (p2.y - p0.y) * 0.5;
        const u1 = (p3.y - p1.y) * 0.5;
        const y = (2 * p1.y - 2 * p2.y + u0 + u1) * t3 +
                  (-3 * p1.y + 3 * p2.y - 2 * u0 - u1) * t2 +
                  u0 * t + p1.y;

        return { x, y };
    }

    /**
     * Build arc-length parameterised Look-Up Table (LUT) for constant-speed traversal.
     */
    _buildSplineLUT() {
        const n = this.controlPoints.length;
        const rawPoints = [];
        let cumulativeLength = 0;

        for (let i = 0; i < n; i++) {
            const p0 = this.controlPoints[(i - 1 + n) % n];
            const p1 = this.controlPoints[i];
            const p2 = this.controlPoints[(i + 1) % n];
            const p3 = this.controlPoints[(i + 2) % n];

            const steps = 40;
            for (let step = 0; step < steps; step++) {
                const t = step / steps;
                const pt = this._catmullRom(p0, p1, p2, p3, t);

                if (rawPoints.length > 0) {
                    const last = rawPoints[rawPoints.length - 1];
                    const dist = Math.hypot(pt.x - last.x, pt.y - last.y);
                    cumulativeLength += dist;
                }
                rawPoints.push({ x: pt.x, y: pt.y, dist: cumulativeLength });
            }
        }
        const first = rawPoints[0], last = rawPoints[rawPoints.length - 1];
        cumulativeLength += Math.hypot(first.x - last.x, first.y - last.y);
        rawPoints.push({...first, dist: cumulativeLength});
        this.totalTrackLength = cumulativeLength;

        // Resample into equidistant intervals along arc length s in [0, 1]
        for (let j = 0; j <= this.sampleResolution; j++) {
            const targetDist = (j / this.sampleResolution) * this.totalTrackLength;
            let low = 0;
            let high = rawPoints.length - 1;

            while (low < high) {
                const mid = Math.floor((low + high) / 2);
                if (rawPoints[mid].dist < targetDist) {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }

            const idx = Math.max(1, low);
            const pPrev = rawPoints[idx - 1];
            const pNext = rawPoints[idx];
            const segmentSpan = Math.max(0.0001, pNext.dist - pPrev.dist);
            const alpha = Math.min(1.0, Math.max(0.0, (targetDist - pPrev.dist) / segmentSpan));

            const x = pPrev.x + (pNext.x - pPrev.x) * alpha;
            const y = pPrev.y + (pNext.y - pPrev.y) * alpha;

            this.splineSamples.push({ x, y });
        }
    }

    /** Gaussian filtering wraps around the lap instead of introducing a seam. */
    _smoothPeriodic(values, distance) {
        const sigma = distance / (this.totalTrackLength / this.sampleResolution);
        const radius = Math.ceil(3 * sigma);
        const weights = Array.from({length: radius * 2 + 1}, (_, i) =>
            Math.exp(-.5 * ((i - radius) / sigma) ** 2));
        const total = weights.reduce((sum, value) => sum + value, 0);
        return values.map((_, i) => {
            let sum = 0;
            for (let j = -radius; j <= radius; j++) {
                sum += values[(i + j + values.length) % values.length] * weights[j + radius];
            }
            return sum / total;
        });
    }

    /** Plan broad braking zones, rather than reacting to individual geometry samples. */
    _buildPaceLUT() {
        const count = this.sampleResolution;
        const step = this.totalTrackLength / count;
        const curvature = Array.from({length: count}, (_, i) => this.sample(i / count).curvature);
        this.laneCurvatures = this._smoothPeriodic(curvature, 20);
        const bends = this._smoothPeriodic(curvature, 60);
        let speeds = bends.map(bend => .62 + .38 / (1 + (bend / .008) ** 2));
        // Propagate braking and acceleration constraints across the lap boundary.
        for (let pass = 0; pass < 2; pass++) {
            for (let i = 0; i < count; i++) {
                const next = (i + 1) % count;
                speeds[next] = Math.min(speeds[next], Math.sqrt(speeds[i] ** 2 + 2 * .0012 * step));
            }
            for (let i = count - 1; i >= 0; i--) {
                const next = (i + 1) % count;
                speeds[i] = Math.min(speeds[i], Math.sqrt(speeds[next] ** 2 + 2 * .0018 * step));
            }
        }
        speeds = this._smoothPeriodic(speeds, 30);
        const times = [0];
        for (let i = 0; i < count; i++) {
            times.push(times[i] + 2 * step / (speeds[i] + speeds[(i + 1) % count]));
        }
        this.paceTimes = times.map(time => time / times[count]);
        this.paceSlopes = speeds.map(speed => speed * times[count] / this.totalTrackLength);
    }

    /** Smooth time-to-distance interpolation matches velocity on both sides of every knot. */
    progressAtTime(elapsedLaps) {
        const laps = Math.floor(elapsedLaps);
        const fraction = elapsedLaps - laps;
        let low = 0, high = this.sampleResolution;
        while (high - low > 1) {
            const mid = Math.floor((low + high) / 2);
            if (this.paceTimes[mid] <= fraction) low = mid;
            else high = mid;
        }
        const span = this.paceTimes[high] - this.paceTimes[low];
        const t = (fraction - this.paceTimes[low]) / span;
        const t2 = t * t, t3 = t2 * t;
        const start = low / this.sampleResolution;
        const end = high / this.sampleResolution;
        const slopeA = this.paceSlopes[low] * span;
        const slopeB = this.paceSlopes[high % this.sampleResolution] * span;
        return laps + (2 * t3 - 3 * t2 + 1) * start + (t3 - 2 * t2 + t) * slopeA
            + (-2 * t3 + 3 * t2) * end + (t3 - t2) * slopeB;
    }

    /** Inverse of the continuous pace map, used at fixed pit checkpoints. */
    timeAtProgress(progress) {
        this.checkpointTimes ||= new Map();
        if (this.checkpointTimes.has(progress)) return this.checkpointTimes.get(progress);
        let low = 0, high = 2;
        for (let i = 0; i < 35; i++) {
            const mid = (low + high) / 2;
            if (this.progressAtTime(mid) < progress) low = mid;
            else high = mid;
        }
        const result = (low + high) / 2;
        this.checkpointTimes.set(progress, result);
        return result;
    }

    _buildPitLane() {
        const entry = this.sample(PIT_ENTRY_PROGRESS);
        const exit = this.sample(PIT_EXIT_PROGRESS);
        this.pitEntry = entry; this.pitExit = exit;
        const points = [];
        const cubic = (a, b, c, d, t) => (1-t)**3*a + 3*(1-t)**2*t*b + 3*(1-t)*t*t*c + t**3*d;
        for (let i = 0; i <= 200; i++) {
            const t = i / 200;
            let x, y;
            if (t < .2) {
                const u = t / .2;
                x = cubic(entry.x, entry.x, 145, 145, u);
                y = cubic(entry.y, entry.y - 30, 960, 900, u);
            } else if (t <= .8) { x = 145; y = 900 - (t - .2) / .6 * 380; }
            else {
                const u = (t - .8) / .2;
                x = cubic(145, 145, exit.x, exit.x, u);
                y = cubic(520, 470, exit.y + 40, exit.y, u);
            }
            const previous = points.at(-1);
            points.push({x, y, distance: previous ? previous.distance + Math.hypot(x-previous.x, y-previous.y) : 0});
        }
        this.pitSamples = points;
    }

    samplePit(fraction) {
        const target = Math.max(0, Math.min(1, fraction)) * this.pitSamples.at(-1).distance;
        let low = 0, high = this.pitSamples.length - 1;
        while (high - low > 1) {
            const mid = Math.floor((low + high) / 2);
            if (this.pitSamples[mid].distance < target) low = mid;
            else high = mid;
        }
        const a = this.pitSamples[low], b = this.pitSamples[high];
        const u = (target - a.distance) / (b.distance - a.distance);
        return {x: a.x + (b.x-a.x)*u, y: a.y + (b.y-a.y)*u,
            heading: Math.atan2(b.y-a.y, b.x-a.x), laneScale: 0};
    }

    /** Cubic interpolation returns a value and its first two derivatives. */
    _interpolate(p0, p1, p2, p3, t) {
        const a = (-p0 + 3 * p1 - 3 * p2 + p3) / 2;
        const b = p0 - 2.5 * p1 + 2 * p2 - .5 * p3;
        const c = (p2 - p0) / 2;
        return {value: ((a * t + b) * t + c) * t + p1,
            first: (3 * a * t + 2 * b) * t + c, second: 6 * a * t + 2 * b};
    }

    /** Sample continuous position, tangent and curvature (per virtual distance unit). */
    sample(progress) {
        const count = this.sampleResolution;
        const s = ((progress % 1) + 1) % 1;
        const position = s * count;
        const index = Math.floor(position);
        const t = position - index;
        const indices = [-1, 0, 1, 2].map(offset => (index + offset + count) % count);
        const points = indices.map(i => this.splineSamples[i]);
        const x = this._interpolate(...points.map(p => p.x), t);
        const y = this._interpolate(...points.map(p => p.y), t);
        const length = Math.hypot(x.first, y.first);
        const curvature = Math.abs(x.first * y.second - y.first * x.second) / Math.max(length ** 3, 1e-9);
        // Fixed offsets can collapse an inside lane at a tight apex. Converging
        // smoothly toward the centre avoids near-stops and keeps cars on the road.
        const laneBend = this.laneCurvatures
            ? this._interpolate(...indices.map(i => this.laneCurvatures[i]), t).value : 0;
        return {x: x.value, y: y.value, heading: Math.atan2(y.first, x.first), curvature,
            laneScale: 1 / (1 + 100 * Math.max(0, laneBend))};
    }

    _traceRoad(ctx) {
        // Render the exact spline as a continuous cubic path. Hundreds of short
        // stroked segments can leave tessellation seams on mobile canvas GPUs.
        const points = this.controlPoints;
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 0; i < points.length; i++) {
            const p0 = points[(i - 1 + points.length) % points.length];
            const p1 = points[i];
            const p2 = points[(i + 1) % points.length];
            const p3 = points[(i + 2) % points.length];
            ctx.bezierCurveTo(p1.x + (p2.x - p0.x) / 6, p1.y + (p2.y - p0.y) / 6,
                p2.x - (p3.x - p1.x) / 6, p2.y - (p3.y - p1.y) / 6, p2.x, p2.y);
        }
        ctx.closePath();
    }

    /** Render a narrow racing surface, pit lane and restrained kerb details. */
    render(ctx, renderWidth, renderHeight) {
        const scale = Math.min(renderWidth / CIRCUIT_WIDTH, renderHeight / CIRCUIT_HEIGHT);
        ctx.save();
        ctx.translate((renderWidth - CIRCUIT_WIDTH * scale) / 2, (renderHeight - CIRCUIT_HEIGHT * scale) / 2);
        ctx.scale(scale, scale);

        // Restrained paddock and grandstand detail around an original circuit.
        ctx.fillStyle = '#EDF0EC';
        this._traceRoad(ctx); ctx.fill();
        ctx.save(); ctx.translate(492, 67); ctx.rotate(.02);
        ctx.fillStyle = '#D9D9DF'; ctx.fillRect(0, 0, 190, 25);
        ctx.strokeStyle = '#B5B6C0'; ctx.lineWidth = 1;
        for (let x = 8; x < 190; x += 14) { ctx.beginPath(); ctx.moveTo(x, 3); ctx.lineTo(x, 22); ctx.stroke(); }
        ctx.restore();
        ctx.strokeStyle = '#D7D8DF'; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(69, 520); ctx.lineTo(69, 885); ctx.stroke();

        // Pit lane follows the main straight; the buildings sit outside the race line.
        ctx.beginPath(); ctx.moveTo(this.pitEntry.x, this.pitEntry.y);
        ctx.bezierCurveTo(this.pitEntry.x, this.pitEntry.y - 30, 145, 960, 145, 900);
        ctx.lineTo(145, 520);
        ctx.bezierCurveTo(145, 470, this.pitExit.x, this.pitExit.y + 40, this.pitExit.x, this.pitExit.y);
        ctx.strokeStyle = CIRCUIT_PALETTE.line; ctx.lineWidth = 28; ctx.stroke();
        ctx.strokeStyle = '#DBDCE2'; ctx.lineWidth = 22; ctx.stroke();
        ctx.fillStyle = '#E2DCE8'; ctx.fillRect(83, 550, 32, 300);
        ctx.strokeStyle = CIRCUIT_PALETTE.line; ctx.lineWidth = 1.5;
        for (let y = 570; y < 850; y += 36) {
            ctx.beginPath(); ctx.moveTo(83, y); ctx.lineTo(115, y); ctx.stroke();
        }

        this._traceRoad(ctx);
        ctx.lineJoin = "round"; ctx.lineCap = "round";
        ctx.strokeStyle = CIRCUIT_PALETTE.runoff; ctx.lineWidth = 106; ctx.stroke();
        ctx.strokeStyle = '#C2C7C3'; ctx.lineWidth = 82; ctx.stroke();
        ctx.strokeStyle = CIRCUIT_PALETTE.paper; ctx.lineWidth = 78; ctx.stroke();
        ctx.lineCap = "butt";
        ctx.strokeStyle = CIRCUIT_PALETTE.kerb; ctx.setLineDash([14, 14]); ctx.stroke();
        ctx.setLineDash([]);
        ctx.strokeStyle = CIRCUIT_PALETTE.paper; ctx.lineWidth = 68; ctx.stroke();
        ctx.strokeStyle = CIRCUIT_PALETTE.asphalt; ctx.lineWidth = 64; ctx.stroke();

        const start = this.sample(0);
        ctx.save(); ctx.translate(start.x, start.y); ctx.rotate(start.heading);
        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 2; col++) {
                ctx.fillStyle = (row + col) % 2 ? CIRCUIT_PALETTE.paper : CIRCUIT_PALETTE.asphalt;
                ctx.fillRect(col * 7 - 7, row * 8 - 32, 7, 8);
            }
        }
        // Grid boxes behind the start are small, straight-edged timing cues.
        ctx.strokeStyle = '#B6ABBF'; ctx.lineWidth = 1.5;
        for (let row = 0; row < 3; row++) {
            for (const side of [-1, 1]) {
                const x = -30 - row * 56, y = side * 16;
                ctx.beginPath(); ctx.moveTo(x - 22, y - 7); ctx.lineTo(x, y - 7);
                ctx.lineTo(x, y + 7); ctx.lineTo(x - 22, y + 7); ctx.stroke();
            }
        }
        ctx.restore();
        ctx.fillStyle = '#77747E'; ctx.font = '15px system-ui'; ctx.textAlign = 'center';
        ctx.save(); ctx.translate(102, 710); ctx.rotate(-Math.PI/2); ctx.fillText('PIT LANE', 0, 0); ctx.restore();
        ctx.strokeStyle = '#8C8797'; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(178, 733); ctx.lineTo(178, 705); ctx.lineTo(262, 705); ctx.lineTo(262, 733); ctx.stroke();
        ctx.restore();
        return scale;
    }
}
