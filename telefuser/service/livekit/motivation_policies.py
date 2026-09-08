"""Pluggable global scheduling policies.

The policy objects decide *which* ready work should be offered to the shared
Motivation scheduler.  State transitions, candidate validation, reservation,
completion, migration, and worker execution remain owned by
``MotivationScheduler`` and ``MotivationRuntimeController``.

Keeping the policy boundary small is intentional: a baseline such as FIFO can
reuse the production feasibility and lease machinery without becoming a second
LiveKit execution implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class SchedulingSearchRequest:
    """Immutable search inputs passed from the scheduler coordinator."""

    now: float | None
    gpu_states: tuple[Any, ...] | None
    wait_seconds: float
    include_wait: bool
    allow_migrations: bool
    exclude_session_ids: tuple[str, ...]
    blocked_migration_session_ids: tuple[str, ...]


class SchedulingPolicy(Protocol):
    """Strategy interface for selecting the next dispatch candidate."""

    name: str

    def select(self, scheduler: Any, request: SchedulingSearchRequest) -> Any:
        """Select a candidate using the coordinator's shared primitives."""


class MotivationPolicy:
    """The profile/slack/quality-aware policy used by the current system."""

    name = "motivation"

    def select(self, scheduler: Any, request: SchedulingSearchRequest) -> Any:
        return scheduler._find_best_motivation(
            now=request.now,
            gpu_states=request.gpu_states,
            wait_seconds=request.wait_seconds,
            include_wait=request.include_wait,
            allow_migrations=request.allow_migrations,
            exclude_session_ids=request.exclude_session_ids,
            blocked_migration_session_ids=request.blocked_migration_session_ids,
            policy_name=self.name,
        )


class FIFOPolicy:
    """A deliberately simple FIFO baseline.

    FIFO is defined over released pending jobs, not every raw heartbeat.  The
    session state intentionally keeps only the newest pending action, so this
    policy selects the oldest currently visible action job and dispatches it as
    a singleton.  The coordinator still performs the normal profile,
    compatibility, GPU, migration, reservation, and stale-candidate checks.
    """

    name = "fifo"

    def select(self, scheduler: Any, request: SchedulingSearchRequest) -> Any:
        ready = scheduler._ready_jobs_for_fifo(now=request.now)
        excluded = set(request.exclude_session_ids)
        eligible = [(session_id, job) for session_id, job in ready if session_id not in excluded]

        # Action work is the baseline's primary demand.  Idle sentinels remain
        # available only when no action is ready, preserving the existing
        # consumption-gated idle semantics without letting idle work overtake
        # an action from another session.
        action_ready = [(session_id, job) for session_id, job in eligible if job.kind == "action"]
        ordered_pool = action_ready or eligible
        if ordered_pool:
            head_session_id, _head_job = min(
                ordered_pool,
                key=lambda item: (item[1].sequence, item[1].created_at, item[1].job_id),
            )
            excluded.update(
                session_id for session_id, _job in eligible if session_id != head_session_id
            )

        return scheduler._find_best_motivation(
            now=request.now,
            gpu_states=request.gpu_states,
            wait_seconds=request.wait_seconds,
            include_wait=request.include_wait,
            allow_migrations=request.allow_migrations,
            exclude_session_ids=tuple(sorted(excluded)),
            blocked_migration_session_ids=request.blocked_migration_session_ids,
            policy_name=self.name,
        )


_POLICY_TYPES: dict[str, type[SchedulingPolicy]] = {
    MotivationPolicy.name: MotivationPolicy,
    FIFOPolicy.name: FIFOPolicy,
}


def create_scheduling_policy(name: str) -> SchedulingPolicy:
    """Create a registered policy by name without coupling callers to classes."""
    normalized = str(name).strip().lower()
    policy_type = _POLICY_TYPES.get(normalized)
    if policy_type is None:
        available = ", ".join(sorted(_POLICY_TYPES))
        raise ValueError(f"unknown scheduling policy {name!r}; available policies: {available}")
    return policy_type()


__all__ = [
    "FIFOPolicy",
    "MotivationPolicy",
    "SchedulingPolicy",
    "SchedulingSearchRequest",
    "create_scheduling_policy",
]
