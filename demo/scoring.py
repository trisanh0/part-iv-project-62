"""Shared, versioned metric semantics for compilation and live heats."""
import math

SCORING_VERSION = '2.0'


def performance_utility(task: str, score: float) -> float:
    if not math.isfinite(score):
        raise ValueError('Score must be finite')
    if task == 'classification':
        return min(1.0, max(0.0, score))
    if task == 'regression':
        return min(1.0, max(0.0, (score + 1.0) / 2.0))
    if task == 'forecasting' and score >= 0:
        return 1.0 / (1.0 + score / 100.0)
    raise ValueError('Invalid task or forecasting error')


def composite_score(task: str, score: float, latency: float, ram: float) -> float:
    if not all(math.isfinite(x) and x >= 0 for x in (latency, ram)):
        raise ValueError('Invalid resource measurements')
    return 100 * performance_utility(task, score) - 15 * math.log10(max(latency, .001)) - 5 * ram / 1000


def performance_order(task: str, score: float) -> float:
    """Larger is better, without clipping physical scores."""
    return -score if task == 'forecasting' else score
