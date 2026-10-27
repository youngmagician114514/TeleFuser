"""Finite-queue scheduling for dependency-coupled latent-step heads.

The worker has a small, memory-bounded active cohort. Clear its remaining
step DAG on one GPU, rather than pretending unselected heads run in parallel.
Only the first invocation is committed; arrivals and timings are reconsidered
at the next step boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Hashable, Sequence


@dataclass(frozen=True)
class PhaseStepGroup:
    session_id: str
    rows: int
    remaining_steps: int
    window: int
    compatibility: Hashable
    deadline: float


@dataclass(frozen=True)
class PhaseBatchPlan:
    sessions: tuple[str, ...]
    duration_seconds: float = 0.0
    clearing_seconds: float = 0.0
    total_completion_seconds: float = 0.0


def plan_phase_steps(
    groups: Sequence[PhaseStepGroup],
    *,
    max_rows: int,
    seconds: Callable[[int, int], float],
) -> PhaseBatchPlan:
    """Maximize ready-cohort service rate, then minimize total completion time.

    Quality and output per group are committed at admission, so minimizing
    the cohort's clearing time maximizes its quality-weighted supply rate.
    The secondary objective integrates unfinished Jobs over that real serial
    timeline. Deadline order resolves equal-cost choices. No rendezvous or
    arbitrary batching threshold is used.

    There are at most slots/3 active groups on an ABot actor. With four S4
    groups the exact DAG has only 5**4 states. All subset service costs are
    evaluated once, including mixed attention windows.
    """
    groups = tuple(sorted(groups, key=lambda group: (group.deadline, group.session_id)))
    if not groups:
        return PhaseBatchPlan(())
    count = len(groups)
    members: dict[int, tuple[int, ...]] = {}
    durations: dict[int, float] = {}
    for mask in range(1, 1 << count):
        indices = tuple(index for index in range(count) if mask & (1 << index))
        rows = sum(groups[index].rows for index in indices)
        if rows > max_rows or any(groups[index].compatibility != groups[indices[0]].compatibility for index in indices):
            continue
        members[mask] = indices
        durations[mask] = sum(groups[index].rows * seconds(groups[index].window, rows) for index in indices) / rows
    # Integer nanoseconds make mathematically equal schedules tie exactly;
    # floating summation order must not override the completion-time objective.
    ticks = {mask: max(1, round(duration * 1_000_000_000)) for mask, duration in durations.items()}
    unit_lower = tuple(
        min(
            ticks[mask] * groups[index].rows // sum(groups[i].rows for i in indices)
            for mask, indices in members.items()
            if index in indices
        )
        if any(index in indices for indices in members.values())
        else 0
        for index in range(count)
    )

    @lru_cache(maxsize=None)
    def solve(remaining: tuple[int, ...]) -> tuple[int, int, int]:
        available = sum(1 << index for index, steps in enumerate(remaining) if steps)
        if not available:
            return 0, 0, 0
        unfinished = available.bit_count()
        best = None
        mask = available
        while mask:
            if mask in durations:
                successor = list(remaining)
                for index in members[mask]:
                    successor[index] -= 1
                duration = ticks[mask]
                # Each later invocation costs at least the sum of its rows'
                # best measured service shares. This admissible bound prunes
                # slower plans without changing the optimum.
                lower = duration + sum(steps * unit for steps, unit in zip(successor, unit_lower))
                if best is not None and lower > best[0][0]:
                    mask = (mask - 1) & available
                    continue
                tail, flow, _ = solve(tuple(successor))
                # Every unfinished group waits for this same physical call.
                # It can only advance in an explicit later invocation.
                order = (duration + tail, unfinished * duration + flow, members[mask])
                if best is None or order < best[0]:
                    best = order, mask
            mask = (mask - 1) & available
        if best is None:
            raise ValueError("A ready dependency group exceeds the worker's latent-slot capacity")
        return best[0][0], best[0][1], best[1]

    clearing, flow, mask = solve(tuple(group.remaining_steps for group in groups))
    return PhaseBatchPlan(
        tuple(groups[index].session_id for index in members[mask]),
        ticks[mask] / 1_000_000_000,
        clearing / 1_000_000_000,
        flow / 1_000_000_000,
    )
