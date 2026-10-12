/**
 * TEMPO Display Day 2D Circuit Track Geometry.
 *
 * Defines closed parametric bezier loops, racing lines, and apex curvature
 * for 2D canvas race rendering without external physics libraries.
 */

export class TrackCircuit {
    /**
     * Initialize track with closed control points and normalized interpolation table.
     */
    constructor() {
        // Closed loop control waypoints scaled to a virtual 1000x600 coordinate canvas
        this.controlPoints = [
            { x: 180, y: 150 }, // Turn 1 entrance
            { x: 450, y: 120 }, // North high-speed straight
            { x: 750, y: 140 }, // Turn 2 approach
            { x: 880, y: 220 }, // Turn 2 hairpin apex
            { x: 850, y: 380 }, // East loop
            { x: 700, y: 440 }, // Chicane entrance
            { x: 620, y: 350 }, // Chicane flick
            { x: 480, y: 480 }, // South sweeping bend
            { x: 250, y: 500 }, // Final turn approach
            { x: 120, y: 400 }, // Final hairpin apex
            { x: 130, y: 250 }, // Main straightaway
        ];

        this.splineSamples = [];
        this.totalTrackLength = 0;
        this.sampleResolution = 600; // Number of equidistant arc-length lookup samples
        this._buildSplineLUT();
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

    /**
     * Sample continuous track coordinates and tangent angle at progress s in [0, 1].
     */
    sample(progress) {
        const s = ((progress % 1.0) + 1.0) % 1.0;
        const floatIdx = s * this.sampleResolution;
        const i0 = Math.floor(floatIdx);
        const i1 = (i0 + 1) % this.splineSamples.length;
        const frac = floatIdx - i0;

        const p0 = this.splineSamples[i0];
        const p1 = this.splineSamples[i1];

        const x = p0.x + (p1.x - p0.x) * frac;
        const y = p0.y + (p1.y - p0.y) * frac;

        // Forward finite difference for heading angle
        const dx = p1.x - p0.x;
        const dy = p1.y - p0.y;
        const heading = Math.atan2(dy, dx);

        // Approximate local curvature by second difference
        const i2 = (i1 + 1) % this.splineSamples.length;
        const p2 = this.splineSamples[i2];
        const dx2 = p2.x - p1.x;
        const dy2 = p2.y - p1.y;
        const headingNext = Math.atan2(dy2, dx2);
        let angleDiff = headingNext - heading;
        while (angleDiff > Math.PI) angleDiff -= Math.PI * 2;
        while (angleDiff < -Math.PI) angleDiff += Math.PI * 2;
        const curvature = Math.abs(angleDiff);

        return { x, y, heading, curvature };
    }

    /**
     * Render racetrack tarmac, curbs, kerbs, and starting grid onto target 2D context.
     */
    render(ctx, renderWidth, renderHeight) {
        const scaleX = renderWidth / 1000;
        const scaleY = renderHeight / 600;
        const scale = Math.min(scaleX, scaleY);

        ctx.save();
        ctx.translate((renderWidth - 1000 * scale) / 2, (renderHeight - 600 * scale) / 2);
        ctx.scale(scale, scale);

        const trackWidth = 56;

        // Outer gravel / grass boundary
        ctx.beginPath();
        for (let i = 0; i <= this.sampleResolution; i++) {
            const pt = this.splineSamples[i];
            if (i === 0) ctx.moveTo(pt.x, pt.y);
            else ctx.lineTo(pt.x, pt.y);
        }
        ctx.closePath();
        ctx.strokeStyle = "#D1D5DC";
        ctx.lineWidth = trackWidth + 14;
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        ctx.stroke();

        // Asphalt road surface
        ctx.strokeStyle = "#E2E4E9";
        ctx.lineWidth = trackWidth;
        ctx.stroke();

        // White racing center-line guides (dashed)
        ctx.beginPath();
        for (let i = 0; i <= this.sampleResolution; i++) {
            const pt = this.splineSamples[i];
            if (i === 0) ctx.moveTo(pt.x, pt.y);
            else ctx.lineTo(pt.x, pt.y);
        }
        ctx.strokeStyle = "#C0C4CE";
        ctx.lineWidth = 2;
        ctx.setLineDash([8, 12]);
        ctx.stroke();
        ctx.setLineDash([]);

        // Start / Finish Checkered Line
        const startPt = this.sample(0.0);
        ctx.save();
        ctx.translate(startPt.x, startPt.y);
        ctx.rotate(startPt.heading + Math.PI / 2);
        ctx.fillStyle = "#1A1A2E";
        ctx.fillRect(-trackWidth / 2, -4, trackWidth, 8);
        ctx.fillStyle = "#FFFFFF";
        const blocks = 8;
        const bw = trackWidth / blocks;
        for (let b = 0; b < blocks; b += 2) {
            ctx.fillRect(-trackWidth / 2 + b * bw, -4, bw, 4);
            ctx.fillRect(-trackWidth / 2 + (b + 1) * bw, 0, bw, 4);
        }
        ctx.restore();

        ctx.restore();
        return scale;
    }
}
