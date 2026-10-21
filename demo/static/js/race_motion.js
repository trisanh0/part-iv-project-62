/** Replay a frozen schedule. All times are relative to the shared lights-out. */
export const PIT_ENTRY_PROGRESS = .9318;
export const PIT_EXIT_PROGRESS = 1.0818;

const hermite = (u, a, b) => (3*u*u - 2*u*u*u) + (u*u*u - 2*u*u + u)*a + (u*u*u - u*u)*b;

export function scheduledMotion(schedule, elapsedMs, circuit, phase = 0) {
    const elapsed = Math.max(0, elapsedMs);
    const finish = schedule.finish_ms;
    if (elapsed >= finish) return {progress: 2, finished: true, inPit: false, pitRemaining: 0};
    if (!schedule.uses_accuracy) {
        return {progress: circuit.progressAtTime(2 * elapsed / finish), inPit: false, pitRemaining: 0};
    }
    const entry = schedule.pit_entry_ms, stop = schedule.pit_stop_ms;
    const departure = schedule.pit_departure_ms, exit = schedule.pit_exit_ms;
    const box = .27 + schedule.pit_box * .09;
    if (elapsed >= entry && elapsed < exit) {
        let route;
        if (elapsed < stop) {
            const u = (elapsed - entry) / (stop - entry);
            route = box * (u + u * u - u * u * u);
        } else if (elapsed < departure) route = box;
        else {
            const u = (elapsed - departure) / (exit - departure);
            route = box + (1 - box) * u * u * (2 - u);
        }
        return {progress: PIT_ENTRY_PROGRESS + route * (PIT_EXIT_PROGRESS - PIT_ENTRY_PROGRESS),
            pitRoute: route, inPit: true, dwelling: elapsed >= stop && elapsed < departure,
            pitRemaining: Math.max(0, departure - Math.max(elapsed, stop)) / 1000};
    }
    const before = elapsed < entry;
    const u = before ? elapsed / entry : (elapsed - exit) / (finish - exit);
    // A small wave between fixed checkpoints cannot move the pit or finish times.
    const start = before ? 0 : circuit.timeAtProgress(PIT_EXIT_PROGRESS);
    const end = before ? circuit.timeAtProgress(PIT_ENTRY_PROGRESS) : 2;
    const pitLength = circuit.pitSamples.at(-1).distance;
    const checkpoint = before ? end : start;
    const paceSlope = (circuit.progressAtTime(checkpoint + 1e-5) - circuit.progressAtTime(checkpoint - 1e-5)) / 2e-5;
    const mainSpeed = paceSlope * circuit.totalTrackLength * (end - start) / (before ? entry : finish - exit);
    const pitSpeed = before ? pitLength * box / (stop - entry) : pitLength * (1-box) / (exit-departure);
    const ratio = pitSpeed / mainSpeed;
    let warped = u;
    // Match pit-lane velocity by braking before entry and accelerating after exit.
    if (before && u > .8) warped = .8 + .2 * hermite((u-.8)/.2, 1, ratio);
    if (!before && u < .2) warped = .2 * hermite(u/.2, ratio, 1);
    const varied = warped + .004 * Math.sin(Math.PI * warped) ** 2 * Math.sin(2 * Math.PI * warped + phase);
    return {progress: circuit.progressAtTime(start + varied * (end - start)), inPit: false, pitRemaining: 0};
}
