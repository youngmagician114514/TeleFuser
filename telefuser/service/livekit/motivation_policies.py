"""Pluggable global scheduling policies.

The policy objects decide *which* ready work should be offered to the shared
scheduling coordinator.  State transitions, candidate validation,
reservation, completion, and worker execution remain owned by
``MotivationScheduler`` and ``MotivationRuntimeController``.  Policies decide
independently whether migration is part of their algorithm: Motivation and
TurboServe may migrate, while FIFO is deliberately static.

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

    FIFO is defined over released *action* jobs, not every raw heartbeat.  It
    selects the oldest currently visible action and dispatches it as a B1
    singleton at one fixed fidelity.  It deliberately does not call the
    Motivation search, quality objective, fairness objective, idle-job path,
    or batch aggregation logic.  The scheduler/controller still own the
    transport-independent safety checks (owner-GPU availability, fixed-row
    memory, reservation, and stale candidates), but FIFO never invokes state
    migration.
    """

    name = "fifo"

    def select(self, scheduler: Any, request: SchedulingSearchRequest) -> Any:
        # Keep all FIFO-specific feasibility and deterministic target selection
        # in the scheduler coordinator.  In particular, do not narrow the
        # ready set and then call ``_find_best_motivation``: that still invokes
        # profile enumeration, quality/fairness scoring, and idle semantics.
        return scheduler._find_fifo_candidate(
            now=request.now,
            gpu_states=request.gpu_states,
            wait_seconds=request.wait_seconds,
            include_wait=request.include_wait,
            allow_migrations=request.allow_migrations,
            exclude_session_ids=request.exclude_session_ids,
            blocked_migration_session_ids=request.blocked_migration_session_ids,
            policy_name=self.name,
        )


class TurboServePolicy:
    """TurboServe-style placement and source-faithful singleton baseline.

    The reference TurboServe controller is a placement policy: it keeps
    sessions on their current workers, uses a least-loaded placement for new
    work, and performs a bounded min--max migration only when the predicted
    bottleneck reduction exceeds the migration cost.  It does not inspect the
    Motivation slack/quality objective or choose among quality points.  The
    coordinator still owns the common job lease and physical execution path;
    this class only selects a fixed-family B1 candidate through the dedicated
    ``_find_turboserve_candidate`` path.  The paper describes coalesced
    chunks, but the checked repository worker executes one session per step;
    this policy follows that implementation rather than inventing a new
    cross-session batching runtime.
    """

    name = "turboserve"

    def select(self, scheduler: Any, request: SchedulingSearchRequest) -> Any:
        return scheduler._find_turboserve_candidate(
            now=request.now,
            gpu_states=request.gpu_states,
            wait_seconds=request.wait_seconds,
            include_wait=request.include_wait,
            allow_migrations=request.allow_migrations,
            exclude_session_ids=request.exclude_session_ids,
            blocked_migration_session_ids=request.blocked_migration_session_ids,
            policy_name=self.name,
        )


_POLICY_TYPES: dict[str, type[SchedulingPolicy]] = {
    MotivationPolicy.name: MotivationPolicy,
    FIFOPolicy.name: FIFOPolicy,
    TurboServePolicy.name: TurboServePolicy,
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
    "TurboServePolicy",
    "SchedulingPolicy",
    "SchedulingSearchRequest",
    "create_scheduling_policy",
]
