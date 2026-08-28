"""Asynchronous session-state migration coordination.

The policy must be able to account for a future ``migration_ready_at`` while
the current GPU continues to serve other work.  This module provides a small
backend-neutral state machine for that control path.  A backend may implement
true pre-copy plus a short boundary commit, or use the compatibility adapter
for TeleFuser's existing blocking, chunk-boundary-safe router API.
"""

from __future__ import annotations

import concurrent.futures
import math
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

MigrationState = Literal["precopied", "ready", "committing", "completed", "aborted", "failed"]


@dataclass(frozen=True)
class MigrationRequest:
    """Immutable request submitted by the global scheduler."""

    session_id: str
    source_gpu: str
    target_gpu: str
    requested_at: float
    estimated_ready_at: float
    state_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.session_id or not self.source_gpu or not self.target_gpu:
            raise ValueError("session_id and GPU identifiers must be non-empty")
        if self.source_gpu == self.target_gpu:
            raise ValueError("migration source and target must differ")
        for name in ("requested_at", "estimated_ready_at"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.estimated_ready_at < self.requested_at:
            raise ValueError("estimated_ready_at cannot precede requested_at")
        if self.state_bytes < 0:
            raise ValueError("state_bytes must be non-negative")


class AsyncMigrationBackend(Protocol):
    """Backend contract for asynchronous pre-copy and boundary commit."""

    def begin(self, request: MigrationRequest) -> Any:
        """Start state transfer and return an opaque operation handle."""

    def ready(self, operation: Any) -> bool:
        """Return whether the target has a consistent snapshot to commit."""

    def commit(self, operation: Any) -> None:
        """Atomically switch ownership at a safe chunk boundary."""

    def abort(self, operation: Any) -> None:
        """Cancel the transfer and release target-side reservations."""


@dataclass(frozen=True)
class MigrationRecord:
    """Observable migration state returned to the scheduler and metrics."""

    migration_id: str
    request: MigrationRequest
    state: MigrationState
    started_at: float
    ready_at: float | None = None
    completed_at: float | None = None
    error: str | None = None


@dataclass
class _MigrationOperation:
    record: MigrationRecord
    backend: AsyncMigrationBackend
    operation: Any


class AsyncMigrationManager:
    """Track non-blocking state transfer and atomic ownership commit.

    ``begin`` is expected to be non-blocking for a true pre-copy backend.  The
    manager never changes scheduler ownership during ``precopied`` or ``ready``;
    callers must invoke ``commit`` only after the target is ready and the source
    has reached a chunk boundary.  All state transitions are serialized, while
    backend transfer itself can use its own copy/NCCL stream.
    """

    def __init__(self, *, max_concurrent: int = 1, clock: Callable[[], float] = time.monotonic) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be positive")
        self.max_concurrent = max_concurrent
        self._clock = clock
        self._active: dict[str, _MigrationOperation] = {}
        self._finished: dict[str, MigrationRecord] = {}
        self._lock = threading.RLock()

    def begin(self, request: MigrationRequest, backend: AsyncMigrationBackend) -> MigrationRecord:
        """Start one transfer, rejecting duplicate session migrations."""
        with self._lock:
            if request.session_id in self._active:
                raise RuntimeError(f"session {request.session_id!r} already has an active migration")
            if len(self._active) >= self.max_concurrent:
                raise RuntimeError("maximum concurrent migrations reached")
            operation = backend.begin(request)
            migration_id = uuid.uuid4().hex
            now = self._clock()
            record = MigrationRecord(
                migration_id=migration_id,
                request=request,
                state="precopied",
                started_at=now,
            )
            self._active[request.session_id] = _MigrationOperation(record, backend, operation)
            return record

    def poll(self, session_id: str, *, now: float | None = None) -> MigrationRecord:
        """Advance a transfer to ``ready`` once its backend reports readiness."""
        with self._lock:
            item = self._active.get(session_id)
            if item is None:
                for record in self._finished.values():
                    if record.request.session_id == session_id:
                        return record
                raise KeyError(session_id)
            record = item.record
            if record.state == "precopied" and item.backend.ready(item.operation):
                ready_at = self._clock() if now is None else now
                item.record = MigrationRecord(
                    migration_id=record.migration_id,
                    request=record.request,
                    state="ready",
                    started_at=record.started_at,
                    ready_at=ready_at,
                )
            return item.record

    def commit(self, session_id: str, *, now: float | None = None) -> MigrationRecord:
        """Commit a ready transfer and publish the new owner to the caller."""
        with self._lock:
            item = self._active.get(session_id)
            if item is None:
                raise KeyError(session_id)
            record = item.record
            if record.state != "ready":
                raise RuntimeError(f"migration {record.migration_id} is not ready: {record.state}")
            item.record = MigrationRecord(
                migration_id=record.migration_id,
                request=record.request,
                state="committing",
                started_at=record.started_at,
                ready_at=record.ready_at,
            )
            try:
                item.backend.commit(item.operation)
            except Exception as exc:
                failed = MigrationRecord(
                    migration_id=record.migration_id,
                    request=record.request,
                    state="failed",
                    started_at=record.started_at,
                    ready_at=record.ready_at,
                    completed_at=self._clock() if now is None else now,
                    error=repr(exc),
                )
                self._active.pop(session_id, None)
                self._finished[record.migration_id] = failed
                raise
            completed = MigrationRecord(
                migration_id=record.migration_id,
                request=record.request,
                state="completed",
                started_at=record.started_at,
                ready_at=record.ready_at,
                completed_at=self._clock() if now is None else now,
            )
            self._active.pop(session_id, None)
            self._finished[record.migration_id] = completed
            return completed

    def abort(self, session_id: str, *, reason: str = "cancelled", now: float | None = None) -> MigrationRecord:
        """Abort a precopy/ready migration and release backend state."""
        with self._lock:
            item = self._active.pop(session_id, None)
            if item is None:
                raise KeyError(session_id)
            record = item.record
            try:
                item.backend.abort(item.operation)
            finally:
                aborted = MigrationRecord(
                    migration_id=record.migration_id,
                    request=record.request,
                    state="aborted",
                    started_at=record.started_at,
                    ready_at=record.ready_at,
                    completed_at=self._clock() if now is None else now,
                    error=reason,
                )
                self._finished[record.migration_id] = aborted
            return aborted

    def active(self) -> tuple[MigrationRecord, ...]:
        """Return active transfers without exposing backend operation handles."""
        with self._lock:
            return tuple(item.record for item in self._active.values())

    def get(self, migration_id: str) -> MigrationRecord:
        """Return a finished migration record by ID."""
        with self._lock:
            return self._finished[migration_id]


class RouterMigrationBackend:
    """Compatibility backend for the existing blocking pipeline router.

    The router call executes in a private thread, so the scheduler and other
    GPUs are not blocked.  It still quiesces the source at a chunk boundary
    internally; a future NCCL pre-copy backend can implement the same protocol
    with a true overlap and a shorter final commit.
    """

    def __init__(self, router: Any, *, executor: concurrent.futures.Executor | None = None) -> None:
        self.router = router
        self._executor = executor or concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="telefuser-migration",
        )
        self._owns_executor = executor is None

    def begin(self, request: MigrationRequest) -> concurrent.futures.Future[Any]:
        return self._executor.submit(
            self.router.migrate_session,
            request.session_id,
            request.target_gpu,
        )

    @staticmethod
    def ready(operation: concurrent.futures.Future[Any]) -> bool:
        return operation.done()

    @staticmethod
    def commit(operation: concurrent.futures.Future[Any]) -> None:
        operation.result()

    @staticmethod
    def abort(operation: concurrent.futures.Future[Any]) -> None:
        operation.cancel()

    def close(self) -> None:
        """Stop the private executor when the owning runtime shuts down."""
        if self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)

