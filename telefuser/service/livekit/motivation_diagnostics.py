"""Model-agnostic diagnostics for the Motivation scheduling control plane.

The scheduler emits compact search and dispatch summaries through a sink
interface.  Keeping collection outside the policy implementation makes the
diagnostics useful for offline tests, LiveKit serving, and future model
adapters without coupling the policy to a particular worker or pipeline.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from threading import RLock
from typing import Protocol

_BATCH_SIZES = (1, 2, 3, 4)


def _zero_batch_counts() -> dict[int, int]:
    return {size: 0 for size in _BATCH_SIZES}


@dataclass(frozen=True)
class MotivationSearchSummary:
    """One bounded snapshot of a candidate search."""

    observed_at: float
    snapshot_epoch: int
    ready_count: int
    ready_action_count: int
    ready_idle_count: int
    excluded_ready_count: int
    gpu_count: int
    include_wait: bool
    allow_migrations: bool
    wait_seconds: float
    enumerated_by_batch_size: Mapping[int, int]
    compatible_by_batch_size: Mapping[int, int]
    profiles_evaluated_by_batch_size: Mapping[int, int]
    feasible_by_batch_size: Mapping[int, int]
    rejected_by_reason: Mapping[str, int]
    not_selected_by_score: Mapping[int, int]
    selected_batch_size: int
    selected_wait: bool
    selected_gpu_id: str | None
    selected_fidelity: str | None
    selected_score: float | None
    selected_migration_count: int
    selected_session_ids: tuple[str, ...]
    selected_job_ids: tuple[str, ...]


@dataclass(frozen=True)
class MotivationDispatchSummary:
    """Outcome of trying to dispatch one policy candidate."""

    observed_at: float
    outcome: str
    reason: str
    batch_size: int
    gpu_id: str | None
    fidelity: str | None
    migration_count: int
    session_ids: tuple[str, ...]
    job_ids: tuple[str, ...]


class MotivationDiagnosticsSink(Protocol):
    """Small interface implemented by collectors or test probes."""

    def record_search(self, summary: MotivationSearchSummary) -> None:
        """Record one candidate-search summary."""

    def record_dispatch(self, summary: MotivationDispatchSummary) -> None:
        """Record one candidate dispatch outcome."""

    def snapshot(self) -> dict[str, object]:
        """Return JSON-compatible aggregate diagnostics."""


class NullMotivationDiagnostics:
    """No-op sink for callers that do not need scheduling diagnostics."""

    def record_search(self, summary: MotivationSearchSummary) -> None:
        del summary

    def record_dispatch(self, summary: MotivationDispatchSummary) -> None:
        del summary

    def snapshot(self) -> dict[str, object]:
        return {}


class MotivationDiagnosticsCollector:
    """Thread-safe bounded collector for scheduler diagnostics.

    Aggregate counters are retained for the whole process.  Recent search and
    dispatch records are optional and bounded so a long-running server cannot
    grow memory with workload size.
    """

    def __init__(self, *, recent_search_limit: int = 128, recent_dispatch_limit: int = 128) -> None:
        if recent_search_limit < 0 or recent_dispatch_limit < 0:
            raise ValueError("recent diagnostic limits must be non-negative")
        self._recent_searches: deque[MotivationSearchSummary] = deque(maxlen=recent_search_limit)
        self._recent_dispatches: deque[MotivationDispatchSummary] = deque(maxlen=recent_dispatch_limit)
        self._search_count = 0
        self._dispatch_count = 0
        self._ready_count = 0
        self._ready_action_count = 0
        self._ready_idle_count = 0
        self._excluded_ready_count = 0
        self._enumerated = Counter()
        self._compatible = Counter()
        self._profiles_evaluated = Counter()
        self._feasible = Counter()
        self._not_selected = Counter()
        self._rejected = Counter()
        self._selected = Counter()
        self._dispatch_outcomes = Counter()
        self._dispatch_batch_sizes = Counter()
        self._dispatch_reasons = Counter()
        self._lock = RLock()

    def record_search(self, summary: MotivationSearchSummary) -> None:
        with self._lock:
            self._search_count += 1
            self._ready_count += summary.ready_count
            self._ready_action_count += summary.ready_action_count
            self._ready_idle_count += summary.ready_idle_count
            self._excluded_ready_count += summary.excluded_ready_count
            self._add_batch_counts(self._enumerated, summary.enumerated_by_batch_size)
            self._add_batch_counts(self._compatible, summary.compatible_by_batch_size)
            self._add_batch_counts(self._profiles_evaluated, summary.profiles_evaluated_by_batch_size)
            self._add_batch_counts(self._feasible, summary.feasible_by_batch_size)
            self._add_batch_counts(self._not_selected, summary.not_selected_by_score)
            self._rejected.update(summary.rejected_by_reason)
            self._selected[str(summary.selected_batch_size)] += 1
            self._recent_searches.append(summary)

    def record_dispatch(self, summary: MotivationDispatchSummary) -> None:
        with self._lock:
            self._dispatch_count += 1
            self._dispatch_outcomes[summary.outcome] += 1
            self._dispatch_batch_sizes[summary.batch_size] += 1
            self._dispatch_reasons[summary.reason] += 1
            self._recent_dispatches.append(summary)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "search_count": self._search_count,
                "dispatch_count": self._dispatch_count,
                "mean_ready_count": (
                    self._ready_count / self._search_count if self._search_count else 0.0
                ),
                "mean_ready_action_count": (
                    self._ready_action_count / self._search_count if self._search_count else 0.0
                ),
                "mean_ready_idle_count": (
                    self._ready_idle_count / self._search_count if self._search_count else 0.0
                ),
                "mean_excluded_ready_count": (
                    self._excluded_ready_count / self._search_count if self._search_count else 0.0
                ),
                "enumerated_by_batch_size": self._batch_counts(self._enumerated),
                "compatible_by_batch_size": self._batch_counts(self._compatible),
                "profiles_evaluated_by_batch_size": self._batch_counts(self._profiles_evaluated),
                "feasible_by_batch_size": self._batch_counts(self._feasible),
                "not_selected_by_score": self._batch_counts(self._not_selected),
                "rejected_by_reason": dict(sorted(self._rejected.items())),
                "selected_batch_size": dict(sorted(self._selected.items())),
                "dispatch_outcomes": dict(sorted(self._dispatch_outcomes.items())),
                "dispatch_by_batch_size": self._batch_counts(self._dispatch_batch_sizes),
                "dispatch_reasons": dict(sorted(self._dispatch_reasons.items())),
                "recent_searches": [asdict(item) for item in self._recent_searches],
                "recent_dispatches": [asdict(item) for item in self._recent_dispatches],
            }

    @staticmethod
    def _add_batch_counts(target: Counter, values: Mapping[int, int]) -> None:
        for size in _BATCH_SIZES:
            target[size] += int(values.get(size, 0))

    @staticmethod
    def _batch_counts(values: Mapping[int, int]) -> dict[str, int]:
        return {str(size): int(values.get(size, 0)) for size in _BATCH_SIZES}


def empty_batch_counts() -> dict[int, int]:
    """Return the standard zero-valued batch-size map for a search summary."""

    return _zero_batch_counts()
