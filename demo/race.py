"""Deterministic race rules derived from eligible benchmark measurements."""
from dataclasses import asdict, dataclass
import math

RACE_RULES_VERSION = 'accuracy-pit-v1'
PIT_ENTRY_PROGRESS = 0.92789
PIT_EXIT_PROGRESS = 1.08653


@dataclass(frozen=True)
class RaceSchedule:
    driving_ms: int
    pit_ms: int
    pit_entry_ms: int
    pit_stop_ms: int
    pit_departure_ms: int
    pit_exit_ms: int
    finish_ms: int
    uses_accuracy: bool
    pit_box: int

    def to_dict(self) -> dict:
        return asdict(self)


def build_schedules(entries: tuple, duration_seconds: float, laps: int = 3) -> tuple[RaceSchedule, ...]:
    """Compress costs equally; integer millisecond ties are shared race finishes."""
    if type(laps) is not int or not 2 <= laps <= 6:
        raise ValueError('Lap count must be between 2 and 6')
    costs = []
    for entry in entries:
        metrics = entry.metrics
        driving = 1 + 0.12 * math.log1p(max(0, metrics.total_latency_s))
        accuracy = metrics.task_type == 'classification'
        pit = 4 * (1 - min(1, max(0, metrics.score))) if accuracy else 0
        costs.append((driving, pit, accuracy))
    duration_ms = round(duration_seconds * 1000)
    scale = duration_ms / max(driving + pit for driving, pit, _ in costs)
    schedules = []
    for entry, (driving, pit, accuracy) in zip(entries, costs):
        finish_ms = round((driving + pit) * scale)
        pit_ms = round(pit * scale)
        driving_ms = finish_ms - pit_ms
        box = entry.car_index
        pit_lap_offset = laps - 2  # Enter near the end of the penultimate lap.
        stop_ms = round(driving_ms * (pit_lap_offset + 0.96 + box * 0.016) / laps)
        schedules.append(RaceSchedule(
            driving_ms, pit_ms, round(driving_ms * (pit_lap_offset + 0.88) / laps), stop_ms,
            stop_ms + pit_ms, round(driving_ms * (pit_lap_offset + 1.12) / laps) + pit_ms,
            finish_ms, accuracy, box))
    return tuple(schedules)
