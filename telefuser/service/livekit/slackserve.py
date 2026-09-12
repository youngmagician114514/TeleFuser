"""Pure control-plane primitives for the SlackServe baseline.

The real serving policy lives behind :class:`SchedulingPolicy`; this module
keeps the paper's profile and urgency calculations independent from LiveKit,
CUDA, and the shared scheduler state machine.  In particular, Elastic SP is
not modeled here: SP=2 is model parallelism for one stream, not a B=2 batch of
two independent sessions.
"""

from __future__ import annotations

import math
import statistics
from enum import IntEnum
from typing import Protocol, Sequence, TypeVar


_EPSILON = 1e-9


class SlackServeProfile(Protocol):
    """The measured fields required by bi-modal profile routing (BMPR)."""

    fidelity: str
    latency_seconds: float
    quality: float


ProfileT = TypeVar("ProfileT", bound=SlackServeProfile)


class SlackServeTier(IntEnum):
    """Ascending dispatch priority used by SlackServe."""

    URGENT = 0
    NORMAL = 1
    RELAXED = 2


def service_credit(
    *,
    playout_slack_seconds: float,
    remaining_seconds: float,
    next_chunk_seconds: float,
) -> float:
    """Return SlackServe service credit ``P - (R + T)``.

    ``P`` is buffered playout time and therefore cannot be negative.  A
    stalled stream consequently has credit ``-(R + T)`` rather than acquiring
    an artificial, unbounded priority from wall-clock time after its buffer
    reaches zero.
    """

    values = (playout_slack_seconds, remaining_seconds, next_chunk_seconds)
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("service-credit inputs must be finite")
    if remaining_seconds < 0 or next_chunk_seconds <= 0:
        raise ValueError("remaining_seconds must be non-negative and next_chunk_seconds positive")
    return max(0.0, float(playout_slack_seconds)) - (
        float(remaining_seconds) + float(next_chunk_seconds)
    )


def urgency_tier(
    credit_seconds: float,
    next_chunk_seconds: float,
    *,
    urgent_multiplier: float = 2.0,
    relaxed_multiplier: float = 4.0,
) -> SlackServeTier:
    """Classify one stream using the paper's ``2T``/``4T`` thresholds."""

    values = (credit_seconds, next_chunk_seconds, urgent_multiplier, relaxed_multiplier)
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("urgency-tier inputs must be finite")
    if next_chunk_seconds <= 0 or urgent_multiplier < 0:
        raise ValueError("next_chunk_seconds must be positive and urgent_multiplier non-negative")
    if relaxed_multiplier <= urgent_multiplier:
        raise ValueError("relaxed_multiplier must exceed urgent_multiplier")
    if credit_seconds < urgent_multiplier * next_chunk_seconds:
        return SlackServeTier.URGENT
    if credit_seconds > relaxed_multiplier * next_chunk_seconds:
        return SlackServeTier.RELAXED
    return SlackServeTier.NORMAL


def pareto_frontier(profiles: Sequence[ProfileT]) -> tuple[ProfileT, ...]:
    """Return non-dominated latency/quality points in deterministic order."""

    rows = tuple(profiles)
    frontier: list[ProfileT] = []
    for row in rows:
        dominated = any(
            other is not row
            and other.latency_seconds <= row.latency_seconds + _EPSILON
            and other.quality + _EPSILON >= row.quality
            and (
                other.latency_seconds < row.latency_seconds - _EPSILON
                or other.quality > row.quality + _EPSILON
            )
            for other in rows
        )
        if not dominated:
            frontier.append(row)
    return tuple(
        sorted(
            frontier,
            key=lambda row: (row.latency_seconds, -row.quality, row.fidelity),
        )
    )


def median_quality_floor(profiles: Sequence[SlackServeProfile]) -> float:
    """Return the paper's global median candidate quality floor."""

    if not profiles:
        raise ValueError("at least one profile is required")
    return float(statistics.median(row.quality for row in profiles))


def choose_bmpr_profile(
    profiles: Sequence[ProfileT],
    *,
    slack_budget_seconds: float,
    quality_floor: float | None = None,
) -> ProfileT:
    """Choose SlackServe's quality or recovery BMPR profile.

    Quality mode selects the highest-quality Pareto point that both meets the
    current playout budget and remains above the global quality floor.  If no
    such point exists, recovery mode selects the fastest point above the same
    floor, even when that point is predicted to miss the budget.
    """

    if not profiles:
        raise ValueError("at least one profile is required")
    if not math.isfinite(float(slack_budget_seconds)):
        raise ValueError("slack_budget_seconds must be finite")
    floor = median_quality_floor(profiles) if quality_floor is None else float(quality_floor)
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError("quality_floor must be positive and finite")
    candidates = tuple(row for row in pareto_frontier(profiles) if row.quality + _EPSILON >= floor)
    if not candidates:
        raise ValueError("no Pareto profile satisfies the quality floor")
    fitting = tuple(
        row
        for row in candidates
        if row.latency_seconds <= max(0.0, slack_budget_seconds) + _EPSILON
    )
    if fitting:
        return max(
            fitting,
            key=lambda row: (row.quality, -row.latency_seconds, row.fidelity),
        )
    return min(
        candidates,
        key=lambda row: (row.latency_seconds, -row.quality, row.fidelity),
    )


__all__ = [
    "SlackServeProfile",
    "SlackServeTier",
    "choose_bmpr_profile",
    "median_quality_floor",
    "pareto_frontier",
    "service_credit",
    "urgency_tier",
]
