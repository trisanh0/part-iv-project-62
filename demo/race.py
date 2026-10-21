"""Deterministic race rules derived from eligible benchmark measurements."""
from dataclasses import asdict, dataclass
import math

RACE_RULES_VERSION = 'accuracy-pit-v1'
PIT_ENTRY_PROGRESS = 0.9318
PIT_EXIT_PROGRESS = 1.0818


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


def build_schedules(entries: tuple, duration_seconds: float) -> tuple[RaceSchedule, ...]:
    """Compress costs equally; integer millisecond ties are shared race finishes."""
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
        stop_ms = round(driving_ms * (0.48 + box * 0.008))
        schedules.append(RaceSchedule(
            driving_ms, pit_ms, round(driving_ms * 0.44), stop_ms,
            stop_ms + pit_ms, round(driving_ms * 0.56) + pit_ms,
            finish_ms, accuracy, box))
    return tuple(schedules)
