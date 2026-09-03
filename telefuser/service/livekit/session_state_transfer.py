"""Backend-neutral Session State Transfer coordination.

The runtime keeps policy decisions, session state ownership, and the concrete
transport separate.  A transfer backend may implement a CPU snapshot, direct
NCCL transfer, or a future paged/layered copy without changing the motivation
controller.  ``SessionStateTransferManager`` only owns the lifecycle and
version boundary; it never performs tensor copies itself.
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

SessionStateTransferState = Literal[
    "precopied",
    "ready",
    "committing",
    "streaming",
    "completed",
    "aborted",
    "failed",
]


@dataclass(frozen=True)
class SessionStateTransferRequest:
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
            raise ValueError("transfer source and target must differ")
        for name in ("requested_at", "estimated_ready_at"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.estimated_ready_at < self.requested_at:
            raise ValueError("estimated_ready_at cannot precede requested_at")
        if self.state_bytes < 0:
            raise ValueError("state_bytes must be non-negative")


class SessionStateTransferBackend(Protocol):
    """Backend contract for asynchronous pre-copy and boundary commit."""

    def begin(self, request: SessionStateTransferRequest) -> Any:
        """Start a transfer and return an opaque operation handle."""

    def ready(self, operation: Any) -> bool:
        """Return whether the target has a consistent state to commit."""

    def commit(self, operation: Any) -> None:
        """Publish the target ownership at a safe session boundary."""

    def abort(self, operation: Any) -> None:
        """Cancel the transfer and release target-side reservations."""

    # A progressive backend may additionally expose ``done(operation)`` and
    # ``finalize(operation)``.  In that case ``commit`` publishes a target
    # whose first layer is ready, while the manager retains the operation in
    # ``streaming`` until the residual copy has either completed or failed.


@dataclass(frozen=True)
class SessionStateTransferRecord:
    """Observable transfer state returned to policy and runtime metrics."""

    transfer_id: str
    request: SessionStateTransferRequest
    state: SessionStateTransferState
    started_at: float
    ready_at: float | None = None
    completed_at: float | None = None
    error: str | None = None

    @property
    def migration_id(self) -> str:
        """Compatibility view for callers that still use migration terminology."""
        return self.transfer_id


@dataclass
class _TransferOperation:
    record: SessionStateTransferRecord
    backend: SessionStateTransferBackend
    operation: Any


class SessionStateTransferManager:
    """Track non-blocking state transfers and atomic ownership commits.

    ``begin`` must be non-blocking for a real asynchronous backend.  The
    manager serializes lifecycle transitions while the backend owns copy
    streams, page allocation, and transport-specific synchronization.
    """

    def __init__(self, *, max_concurrent: int = 1, clock: Callable[[], float] = time.monotonic) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be positive")
        self.max_concurrent = max_concurrent
        self._clock = clock
        self._active: dict[str, _TransferOperation] = {}
        self._finished: dict[str, SessionStateTransferRecord] = {}
        self._lock = threading.RLock()

    def begin(
        self,
        request: SessionStateTransferRequest,
        backend: SessionStateTransferBackend,
    ) -> SessionStateTransferRecord:
        """Start one transfer, rejecting duplicate or over-capacity requests."""
        with self._lock:
            if request.session_id in self._active:
                raise RuntimeError(f"session {request.session_id!r} already has an active transfer")
            if len(self._active) >= self.max_concurrent:
                raise RuntimeError("maximum concurrent transfers reached")
            operation = backend.begin(request)
            transfer_id = uuid.uuid4().hex
            record = SessionStateTransferRecord(
                transfer_id=transfer_id,
                request=request,
                state="precopied",
                started_at=self._clock(),
            )
            self._active[request.session_id] = _TransferOperation(record, backend, operation)
            return record

    def poll(self, session_id: str, *, now: float | None = None) -> SessionStateTransferRecord:
        """Advance readiness and finalize a progressive residual transfer."""
        with self._lock:
            item = self._active.get(session_id)
            if item is None:
                for record in self._finished.values():
                    if record.request.session_id == session_id:
                        return record
                raise KeyError(session_id)
            record = item.record
            if record.state == "precopied" and item.backend.ready(item.operation):
                item.record = SessionStateTransferRecord(
                    transfer_id=record.transfer_id,
                    request=record.request,
                    state="ready",
                    started_at=record.started_at,
                    ready_at=self._clock() if now is None else now,
                )
            elif record.state == "streaming":
                done = getattr(item.backend, "done", None)
                if callable(done) and done(item.operation):
                    completed_at = self._clock() if now is None else now
                    finalize = getattr(item.backend, "finalize", None)
                    try:
                        if callable(finalize):
                            finalize(item.operation)
                    except Exception as exc:
                        failed = SessionStateTransferRecord(
                            transfer_id=record.transfer_id,
                            request=record.request,
                            state="failed",
                            started_at=record.started_at,
                            ready_at=record.ready_at,
                            completed_at=completed_at,
                            error=repr(exc),
                        )
                        self._active.pop(session_id, None)
                        self._finished[record.transfer_id] = failed
                        return failed
                    completed = SessionStateTransferRecord(
                        transfer_id=record.transfer_id,
                        request=record.request,
                        state="completed",
                        started_at=record.started_at,
                        ready_at=record.ready_at,
                        completed_at=completed_at,
                    )
                    self._active.pop(session_id, None)
                    self._finished[record.transfer_id] = completed
                    return completed
            return item.record

    def commit(self, session_id: str, *, now: float | None = None) -> SessionStateTransferRecord:
        """Commit a ready transfer and publish the new owner to the caller."""
        with self._lock:
            item = self._active.get(session_id)
            if item is None:
                raise KeyError(session_id)
            record = item.record
            if record.state != "ready":
                raise RuntimeError(f"transfer {record.transfer_id} is not ready: {record.state}")
            item.record = SessionStateTransferRecord(
                transfer_id=record.transfer_id,
                request=record.request,
                state="committing",
                started_at=record.started_at,
                ready_at=record.ready_at,
            )
            try:
                item.backend.commit(item.operation)
            except Exception as exc:
                failed = SessionStateTransferRecord(
                    transfer_id=record.transfer_id,
                    request=record.request,
                    state="failed",
                    started_at=record.started_at,
                    ready_at=record.ready_at,
                    completed_at=self._clock() if now is None else now,
                    error=repr(exc),
                )
                self._active.pop(session_id, None)
                self._finished[record.transfer_id] = failed
                raise
            done = getattr(item.backend, "done", None)
            if callable(done) and not done(item.operation):
                streaming = SessionStateTransferRecord(
                    transfer_id=record.transfer_id,
                    request=record.request,
                    state="streaming",
                    started_at=record.started_at,
                    ready_at=record.ready_at,
                )
                item.record = streaming
                return streaming
            finalize = getattr(item.backend, "finalize", None)
            if callable(finalize):
                try:
                    finalize(item.operation)
                except Exception as exc:
                    failed = SessionStateTransferRecord(
                        transfer_id=record.transfer_id,
                        request=record.request,
                        state="failed",
                        started_at=record.started_at,
                        ready_at=record.ready_at,
                        completed_at=self._clock() if now is None else now,
                        error=repr(exc),
                    )
                    self._active.pop(session_id, None)
                    self._finished[record.transfer_id] = failed
                    raise
            completed = SessionStateTransferRecord(
                transfer_id=record.transfer_id,
                request=record.request,
                state="completed",
                started_at=record.started_at,
                ready_at=record.ready_at,
                completed_at=self._clock() if now is None else now,
            )
            self._active.pop(session_id, None)
            self._finished[record.transfer_id] = completed
            return completed

    def abort(
        self,
        session_id: str,
        *,
        reason: str = "cancelled",
        now: float | None = None,
    ) -> SessionStateTransferRecord:
        """Abort a transfer and release backend state."""
        with self._lock:
            item = self._active.pop(session_id, None)
            if item is None:
                raise KeyError(session_id)
            record = item.record
            try:
                item.backend.abort(item.operation)
            finally:
                aborted = SessionStateTransferRecord(
                    transfer_id=record.transfer_id,
                    request=record.request,
                    state="aborted",
                    started_at=record.started_at,
                    ready_at=record.ready_at,
                    completed_at=self._clock() if now is None else now,
                    error=reason,
                )
                self._finished[record.transfer_id] = aborted
            return aborted

    def active(self) -> tuple[SessionStateTransferRecord, ...]:
        """Return active transfers without exposing backend operation handles."""
        with self._lock:
            return tuple(item.record for item in self._active.values())

    def get(self, transfer_id: str) -> SessionStateTransferRecord:
        """Return a finished transfer record by ID."""
        with self._lock:
            return self._finished[transfer_id]


class ThreadedRouterSessionStateTransferBackend:
    """Adapt a blocking router to the Session State Transfer contract.

    This compatibility backend is intentionally isolated from the policy.  A
    true paged/layered backend can replace it without changing callers.
    """

    def __init__(self, router: Any, *, executor: concurrent.futures.Executor | None = None) -> None:
        self.router = router
        self._executor = executor or concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="telefuser-session-transfer",
        )
        self._owns_executor = executor is None

    def begin(self, request: SessionStateTransferRequest) -> concurrent.futures.Future[Any]:
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
