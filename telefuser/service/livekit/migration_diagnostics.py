"""Bounded observability for session-state migration.

The process-NCCL worker pool has two distinct layers of migration telemetry:
the runtime metrics collector records the public migration outcome, while this
module records the transport phases that explain *why* a migration succeeded
or failed.  The collector is deliberately independent of NCCL and has no
serving-side effect; it can therefore also be used by another state-transfer
backend.

Only bounded, low-cardinality aggregates are exported.  Recent records are
useful while debugging a live deployment, but are capped and do not contain
session identifiers.
"""

from __future__ import annotations

import contextlib
import math
import signal
import time
from collections import Counter, deque
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable

_MIGRATION_PHASES = (
    "drain",
    "pause",
    "export",
    "prepare_recv",
    "transfer",
    "commit_source",
    "route_commit",
)
_DEFAULT_PHASE = "other"
_MAX_ERROR_LENGTH = 512


def _phase_name(phase: str) -> str:
    """Normalize phase names so externally visible labels stay bounded."""

    normalized = str(phase).strip().lower().replace("-", "_")
    return normalized if normalized in _MIGRATION_PHASES else _DEFAULT_PHASE


def classify_migration_error(error: BaseException | str | None) -> str:
    """Map a transport exception to a stable, low-cardinality error class."""

    if error is None:
        return "unknown"
    text = str(error).lower()
    exception_name = type(error).__name__.lower() if isinstance(error, BaseException) else ""
    if "cancel" in text or "cancel" in exception_name:
        return "cancelled"
    if "timed out" in text or "timeout" in text or "timeout" in exception_name:
        return "timeout"
    if "out of memory" in text or "cuda oom" in text:
        return "oom"
    if "worker process" in text and "not alive" in text:
        return "worker_unavailable"
    if "worker" in text and ("exited" in text or "dead" in text):
        return "worker_unavailable"
    if "nccl" in text or "nccl" in exception_name:
        return "nccl"
    if "ownership" in text or "route" in text:
        return "ownership"
    return "runtime"


def _safe_error(error: BaseException | str | None) -> str | None:
    """Return a bounded, one-line error suitable for a JSON snapshot/log."""

    if error is None:
        return None
    text = str(error).replace("\n", " ").strip()
    if not text and isinstance(error, BaseException):
        text = type(error).__name__
    if len(text) > _MAX_ERROR_LENGTH:
        return text[: _MAX_ERROR_LENGTH - 3] + "..."
    return text


def _finite_nonnegative(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return number if math.isfinite(number) and number >= 0 else 0.0


def _nonnegative_int(value: object) -> int:
    """Convert malformed byte/count values to a safe non-negative integer."""

    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, number)


@dataclass
class _ActiveTransfer:
    transfer_id: str
    source_worker_id: str
    target_worker_id: str
    started_at: float
    state_bytes: int
    phase: str | None = None
    phase_started_at: float | None = None
    failed_phase: str | None = None
    transport_report: dict[str, Any] | None = None


@dataclass
class _PhaseAggregate:
    count: int = 0
    success: int = 0
    failures: int = 0
    total_ms: float = 0.0


class MigrationDiagnostics:
    """Thread-safe, bounded migration lifecycle and phase telemetry.

    The methods intentionally tolerate duplicate/missing lifecycle calls.  A
    diagnostics collector must never make a serving request fail merely
    because an optional metric was emitted out of order.
    """

    def __init__(
        self,
        *,
        recent_limit: int = 32,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if recent_limit < 0:
            raise ValueError("recent_limit must be non-negative")
        self._clock = clock
        self._recent: deque[dict[str, Any]] = deque(maxlen=recent_limit)
        self._active: dict[str, _ActiveTransfer] = {}
        self._phases: dict[str, _PhaseAggregate] = {phase: _PhaseAggregate() for phase in _MIGRATION_PHASES}
        self._phases[_DEFAULT_PHASE] = _PhaseAggregate()
        self._attempts = 0
        self._successes = 0
        self._failures = 0
        self._aborted = 0
        self._rejected = 0
        self._total_duration_ms = 0.0
        self._successful_duration_ms = 0.0
        # Transfer-level and phase-level errors are kept separate: one failed
        # migration may have both a failed phase and a rollback error, and
        # combining them would make failure totals look artificially doubled.
        self._errors: Counter[str] = Counter()
        self._phase_errors: Counter[str] = Counter()
        self._worker_exits = 0
        self._worker_exit_codes: Counter[str] = Counter()
        self._worker_exit_errors: Counter[str] = Counter()
        self._last: dict[str, Any] | None = None
        self._last_failure: dict[str, Any] | None = None
        self._last_worker_exit: dict[str, Any] | None = None
        self._lock = RLock()

    def begin(
        self,
        transfer_id: str,
        *,
        source_worker_id: str,
        target_worker_id: str,
        state_bytes: int = 0,
    ) -> None:
        """Start tracking one accepted transfer."""

        now = self._now()
        with self._lock:
            self._attempts += 1
            self._active[str(transfer_id)] = _ActiveTransfer(
                transfer_id=str(transfer_id),
                source_worker_id=str(source_worker_id),
                target_worker_id=str(target_worker_id),
                started_at=now,
                state_bytes=_nonnegative_int(state_bytes),
            )

    def set_state_bytes(self, transfer_id: str, state_bytes: object) -> None:
        """Update the transferred-state size once export metadata is available."""

        try:
            value = int(state_bytes)
        except (TypeError, ValueError, OverflowError):
            return
        if value < 0:
            return
        with self._lock:
            active = self._active.get(str(transfer_id))
            if active is not None:
                active.state_bytes = value

    def set_transport_report(self, transfer_id: str, report: object) -> None:
        """Attach bounded per-group transfer timing from the target worker."""
        if not isinstance(report, dict):
            return
        raw_groups = report.get("groups")
        groups: list[dict[str, Any]] = []
        if isinstance(raw_groups, list):
            for raw in raw_groups[:64]:
                if not isinstance(raw, dict):
                    continue
                groups.append(
                    {
                        "name": str(raw.get("name", "unknown"))[:64],
                        "layer_index": (
                            int(raw["layer_index"])
                            if isinstance(raw.get("layer_index"), int)
                            else None
                        ),
                        "bytes": _nonnegative_int(raw.get("bytes", 0)),
                        "duration_ms": round(_finite_nonnegative(raw.get("duration_ms", 0.0)), 3),
                    }
                )
        sanitized = {
            "total_bytes": _nonnegative_int(report.get("total_bytes", 0)),
            "total_duration_ms": round(_finite_nonnegative(report.get("total_duration_ms", 0.0)), 3),
            "groups": groups,
        }
        raw_progress = report.get("progress")
        if isinstance(raw_progress, dict):
            sanitized["progress"] = {
                "layer_count": _nonnegative_int(raw_progress.get("layer_count", 0)),
                "ready_layers": _nonnegative_int(raw_progress.get("ready_layers", 0)),
                "complete": bool(raw_progress.get("complete", False)),
                "failed": bool(raw_progress.get("failed", False)),
                "first_layer_ready_ms": (
                    round(_finite_nonnegative(raw_progress.get("first_layer_ready_ms", 0.0)), 3)
                    if raw_progress.get("first_layer_ready_ms") is not None
                    else None
                ),
                "transfer_complete_ms": (
                    round(_finite_nonnegative(raw_progress.get("transfer_complete_ms", 0.0)), 3)
                    if raw_progress.get("transfer_complete_ms") is not None
                    else None
                ),
                "first_compute_residual_wait_ms": round(
                    _finite_nonnegative(raw_progress.get("first_compute_residual_wait_ms", 0.0)),
                    3,
                ),
                "host_wait_ms": round(_finite_nonnegative(raw_progress.get("host_wait_ms", 0.0)), 3),
                "wait_calls": _nonnegative_int(raw_progress.get("wait_calls", 0)),
                "complete_host_wait_ms": round(
                    _finite_nonnegative(raw_progress.get("complete_host_wait_ms", 0.0)),
                    3,
                ),
                "complete_wait_calls": _nonnegative_int(raw_progress.get("complete_wait_calls", 0)),
            }
        with self._lock:
            active = self._active.get(str(transfer_id))
            if active is not None:
                active.transport_report = sanitized

    def phase_started(self, transfer_id: str, phase: str) -> None:
        """Mark the beginning of a transport phase."""

        with self._lock:
            active = self._active.get(str(transfer_id))
            if active is None:
                return
            active.phase = _phase_name(phase)
            active.phase_started_at = self._now()

    def phase_finished(
        self,
        transfer_id: str,
        phase: str,
        *,
        success: bool,
        duration_seconds: float | None = None,
        error: BaseException | str | None = None,
    ) -> None:
        """Record one phase result and its elapsed time."""

        normalized = _phase_name(phase)
        with self._lock:
            active = self._active.get(str(transfer_id))
            if duration_seconds is None:
                started = active.phase_started_at if active is not None else None
                duration_seconds = self._now() - started if started is not None else 0.0
            duration_ms = _finite_nonnegative(duration_seconds) * 1000.0
            aggregate = self._phases[normalized]
            aggregate.count += 1
            aggregate.total_ms += duration_ms
            if success:
                aggregate.success += 1
            else:
                aggregate.failures += 1
                self._phase_errors[classify_migration_error(error)] += 1
                if active is not None:
                    active.failed_phase = normalized
            if active is not None and active.phase == normalized:
                active.phase = None
                active.phase_started_at = None

    def finish(
        self,
        transfer_id: str,
        *,
        outcome: str,
        error: BaseException | str | None = None,
    ) -> bool:
        """Finish a transfer and return whether an active record was found."""

        transfer_key = str(transfer_id)
        normalized_outcome = str(outcome).strip().lower()
        if normalized_outcome not in {"success", "failure", "aborted"}:
            normalized_outcome = "failure"
        now = self._now()
        with self._lock:
            active = self._active.pop(transfer_key, None)
            if active is None:
                return False
            duration_ms = _finite_nonnegative(now - active.started_at) * 1000.0
            if normalized_outcome == "success":
                self._successes += 1
                self._successful_duration_ms += duration_ms
            elif normalized_outcome == "aborted":
                self._aborted += 1
            else:
                self._failures += 1
            self._total_duration_ms += duration_ms
            error_text = _safe_error(error)
            error_kind = classify_migration_error(error) if normalized_outcome != "success" else None
            if error_kind is not None:
                self._errors[error_kind] += 1
            record: dict[str, Any] = {
                "transfer_id": active.transfer_id,
                "source_worker_id": active.source_worker_id,
                "target_worker_id": active.target_worker_id,
                "outcome": normalized_outcome,
                "duration_ms": round(duration_ms, 3),
                "state_bytes": active.state_bytes,
                "transport_report": active.transport_report,
                "failed_phase": (
                    active.failed_phase or active.phase if normalized_outcome != "success" else None
                ),
                "error_kind": error_kind,
                "error": error_text,
                "completed_at": now,
            }
            self._last = record
            if normalized_outcome != "success":
                self._last_failure = record
            self._recent.append(dict(record))
            return True

    def reject(
        self,
        *,
        source_worker_id: str,
        target_worker_id: str,
        reason: BaseException | str,
    ) -> None:
        """Record a request rejected before a transfer token was allocated."""

        now = self._now()
        error_text = _safe_error(reason)
        error_kind = classify_migration_error(reason)
        with self._lock:
            self._rejected += 1
            self._errors[error_kind] += 1
            record = {
                "transfer_id": None,
                "source_worker_id": str(source_worker_id),
                "target_worker_id": str(target_worker_id),
                "outcome": "rejected",
                "duration_ms": 0.0,
                "state_bytes": 0,
                "failed_phase": "admission",
                "error_kind": error_kind,
                "error": error_text,
                "completed_at": now,
            }
            self._last = record
            self._last_failure = record
            self._recent.append(dict(record))

    def record_worker_exit(
        self,
        worker_id: str,
        exitcode: int | None,
        *,
        error: BaseException | str | None = None,
    ) -> None:
        """Record a process exit with active-transfer and startup-error context."""

        try:
            numeric_code = None if exitcode is None else int(exitcode)
            code = "unknown" if numeric_code is None else str(numeric_code)
        except (TypeError, ValueError, OverflowError):
            numeric_code = None
            code = "unknown"
        exit_signal = None
        if numeric_code is not None and numeric_code < 0:
            with contextlib.suppress(ValueError):
                exit_signal = signal.Signals(-numeric_code).name
        elif numeric_code == 137:
            # Shells commonly report SIGKILL as 128 + signal number.
            exit_signal = "SIGKILL"
        error_text = _safe_error(error)
        error_kind = classify_migration_error(error) if error is not None else "process_exit"
        now = self._now()
        with self._lock:
            self._worker_exits += 1
            self._worker_exit_codes[code] += 1
            self._worker_exit_errors[error_kind] += 1
            active = [
                {
                    "transfer_id": item.transfer_id,
                    "source_worker_id": item.source_worker_id,
                    "target_worker_id": item.target_worker_id,
                    "phase": item.phase,
                }
                for item in self._active.values()
            ]
            self._last_worker_exit = {
                "worker_id": str(worker_id),
                "exit_code": code,
                "error_kind": error_kind,
                "error": error_text,
                "exit_signal": exit_signal,
                "active_transfers": active,
                "observed_at": now,
            }

    def snapshot(self) -> dict[str, Any]:
        """Return JSON-compatible aggregate and bounded recent telemetry."""

        with self._lock:
            phase_snapshot = {
                phase: {
                    "count": aggregate.count,
                    "success": aggregate.success,
                    "failures": aggregate.failures,
                    "total_ms": round(aggregate.total_ms, 3),
                    "average_ms": round(aggregate.total_ms / aggregate.count, 3) if aggregate.count else 0.0,
                }
                for phase, aggregate in self._phases.items()
            }
            return {
                "attempts_total": self._attempts,
                "success_total": self._successes,
                "failure_total": self._failures,
                "aborted_total": self._aborted,
                "rejected_total": self._rejected,
                "active": len(self._active),
                "active_transfers": [
                    {
                        "transfer_id": item.transfer_id,
                        "source_worker_id": item.source_worker_id,
                        "target_worker_id": item.target_worker_id,
                        "phase": item.phase,
                        "state_bytes": item.state_bytes,
                    }
                    for item in self._active.values()
                ],
                "total_duration_ms": round(self._total_duration_ms, 3),
                "average_duration_ms": round(
                    self._total_duration_ms / (self._successes + self._failures + self._aborted), 3
                )
                if self._successes + self._failures + self._aborted
                else 0.0,
                "successful_duration_ms": round(self._successful_duration_ms, 3),
                "average_success_duration_ms": round(self._successful_duration_ms / self._successes, 3)
                if self._successes
                else 0.0,
                "error_counts": dict(sorted(self._errors.items())),
                "phase_error_counts": dict(sorted(self._phase_errors.items())),
                "phase_timings": phase_snapshot,
                "worker_exits_total": self._worker_exits,
                "worker_exits_by_code": dict(sorted(self._worker_exit_codes.items())),
                "worker_exit_error_counts": dict(sorted(self._worker_exit_errors.items())),
                "last": dict(self._last) if self._last is not None else None,
                "last_failure": dict(self._last_failure) if self._last_failure is not None else None,
                "last_worker_exit": dict(self._last_worker_exit) if self._last_worker_exit is not None else None,
                "recent": [dict(item) for item in self._recent],
            }

    def _now(self) -> float:
        try:
            return _finite_nonnegative(self._clock())
        except Exception:
            return time.monotonic()


__all__ = ["MigrationDiagnostics", "classify_migration_error"]
