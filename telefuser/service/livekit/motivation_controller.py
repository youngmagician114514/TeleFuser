"""Runtime bridge from LiveKit events to the motivation policy core.

The controller is intentionally opt-in.  Existing TeleFuser services keep
their worker-local scheduling behavior until a caller supplies this bridge and
a worker dispatch callback.  This makes the migration from heuristic serving
to the global policy incremental and testable.
"""

from __future__ import annotations

import concurrent.futures
import math
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .async_migration import (
    AsyncMigrationBackend,
    AsyncMigrationManager,
    MigrationRecord,
    MigrationRequest,
)
from .motivation_scheduler import (
    ActionJob,
    DispatchCandidate,
    MotivationScheduler,
    SessionSchedulingState,
)


@dataclass(frozen=True)
class DispatchLease:
    """Reserved policy decision handed to a worker executor."""

    candidate: DispatchCandidate
    jobs: tuple[ActionJob, ...]
    reserved_at: float


DispatchCallback = Callable[[DispatchLease], None]
MigrationBackendFactory = Callable[[MigrationRequest], AsyncMigrationBackend]
SearchExecutor = concurrent.futures.Executor


class MotivationRuntimeController:
    """Connect event callbacks, global search, migration, and execution.

    ``schedule_once`` is safe to call from an event loop or a background
    scheduler thread.  It starts at most one migration per call, allowing the
    next invocation to observe readiness without blocking another GPU.  A
    candidate with a remote owner is therefore prepared first and dispatched
    only after migration commits and the candidate is searched again.
    """

    def __init__(
        self,
        scheduler: MotivationScheduler,
        *,
        dispatch: DispatchCallback,
        migration_manager: AsyncMigrationManager | None = None,
        migration_backend_factory: MigrationBackendFactory | None = None,
        search_executor: SearchExecutor | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.scheduler = scheduler
        self.dispatch = dispatch
        self.migration_manager = migration_manager
        self.migration_backend_factory = migration_backend_factory
        self._search_executor = search_executor
        self._owns_search_executor = False
        self._clock = clock
        self._lock = threading.RLock()

    def on_session_registered(
        self,
        session_id: str,
        *,
        owner_gpu: str,
        now: float | None = None,
        slack_seconds: float | None = None,
        quality: float | None = None,
        compatibility_key: Iterable[object] = (),
        active: bool = True,
    ) -> SessionSchedulingState:
        """Register a retained session at the runtime event boundary."""
        return self.scheduler.register_session(
            session_id,
            owner_gpu=owner_gpu,
            now=self._observed(now),
            slack_seconds=slack_seconds,
            quality=quality,
            compatibility_key=compatibility_key,
            active=active,
        )

    def on_session_departed(self, session_id: str, *, now: float | None = None) -> None:
        """Remove future policy work while retaining any in-flight job."""
        self.scheduler.mark_departed(session_id, now=self._observed(now))

    def on_action(
        self,
        session_id: str,
        controls: Iterable[str],
        *,
        now: float | None = None,
        release: bool = True,
    ) -> tuple[ActionJob | None, bool]:
        """Forward one action update and preserve empty-to-nonempty semantics."""
        return self.scheduler.submit_action(
            session_id,
            controls,
            now=self._observed(now),
            release=release,
        )

    def on_gpu_update(
        self,
        gpu_id: str,
        *,
        free_at: float | None = None,
        memory_free_gb: float | None = None,
        available: bool | None = None,
        now: float | None = None,
    ) -> None:
        """Publish worker timing facts and invalidate stale searches."""
        self.scheduler.update_gpu(
            gpu_id,
            free_at=free_at,
            memory_free_gb=memory_free_gb,
            available=available,
            now=self._observed(now),
        )

    def on_migration_ready(
        self,
        session_id: str,
        *,
        target_gpu: str,
        ready_at: float,
        now: float | None = None,
    ) -> None:
        """Publish an externally observed migration-ready timestamp."""
        self.scheduler.set_migration_ready(
            session_id,
            target_gpu=target_gpu,
            ready_at=ready_at,
            now=self._observed(now),
        )

    def search_async(
        self,
        *,
        now: float | None = None,
        wait_seconds: float = 0.0,
    ) -> concurrent.futures.Future[DispatchCandidate | None]:
        """Search in a background executor while the GPU runs another lease.

        The returned future contains only a snapshot candidate.  It never
        reserves a job; callers must pass the result to
        :meth:`dispatch_candidate`, which performs the current-state check.
        """
        observed_at = self._observed(now)
        with self._lock:
            if self._search_executor is None:
                self._search_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="telefuser-motivation-search",
                )
                self._owns_search_executor = True
            executor = self._search_executor
        def search() -> DispatchCandidate | None:
            # The task may start after an event callback has advanced policy
            # time.  Search from the scheduler's current timeline rather than
            # failing with a backwards-time error; dispatch_candidate still
            # rejects the captured epoch if the state changed meanwhile.
            search_at = max(observed_at, self.scheduler.current_time)
            try:
                return self.scheduler.find_best(
                    now=search_at,
                    wait_seconds=wait_seconds,
                    include_wait=True,
                )
            except ValueError as exc:
                # A concurrent event can advance the scheduler between the
                # time read above and find_best acquiring its lock. Retry at
                # the newer timeline; any semantic change is still caught by
                # the candidate epoch check at dispatch time.
                if "monotonic" not in str(exc):
                    raise
                return self.scheduler.find_best(
                    now=self.scheduler.current_time,
                    wait_seconds=wait_seconds,
                    include_wait=True,
                )

        return executor.submit(search)

    def schedule_once(
        self,
        *,
        now: float | None = None,
        wait_seconds: float = 0.0,
    ) -> DispatchLease | None:
        """Search synchronously and dispatch one current candidate if possible."""
        observed_at = self._observed(now)
        candidate = self.scheduler.find_best(
            now=observed_at,
            wait_seconds=wait_seconds,
            include_wait=True,
        )
        return self.dispatch_candidate(candidate, now=observed_at) if candidate is not None else None

    def dispatch_candidate(
        self,
        candidate: DispatchCandidate,
        *,
        now: float | None = None,
    ) -> DispatchLease | None:
        """Validate and dispatch a candidate produced by sync or async search.

        A remote candidate first starts or completes asynchronous migration.  A
        fresh search is required after ownership changes, so no stale
        candidate is reserved across the migration boundary.
        """
        observed_at = self._observed(now)
        with self._lock:
            if candidate.wait:
                return None
            if candidate.migration_count:
                if not self._prepare_migration(candidate, now=observed_at):
                    return None
                # ``_prepare_migration`` commits at most one transfer. Search
                # again after the ownership/version boundary.
                return None
            if not self.scheduler.validate(candidate, now=observed_at):
                return None
            jobs = tuple(
                self.scheduler.session(session_id).ready_job(include_idle=True)
                for session_id in candidate.session_ids
            )
            if any(job is None for job in jobs):
                return None
            lease = DispatchLease(
                candidate=candidate,
                jobs=tuple(job for job in jobs if job is not None),
                reserved_at=observed_at,
            )
            self.scheduler.reserve(candidate, now=observed_at)
            self.dispatch(lease)
            return lease

    def close(self) -> None:
        """Release a controller-owned background search executor."""
        with self._lock:
            executor = self._search_executor if self._owns_search_executor else None
            self._search_executor = None
            self._owns_search_executor = False
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def on_completion(
        self,
        lease: DispatchLease,
        *,
        completed_at: float | None = None,
        quality: float | None = None,
    ) -> tuple[ActionJob, ...]:
        """Commit worker completion and release the profile memory reservation."""
        return self.scheduler.complete(
            lease.candidate,
            completed_at=self._observed(completed_at),
            quality=quality,
        )

    def poll_migrations(self, *, now: float | None = None) -> tuple[MigrationRecord, ...]:
        """Poll active transfers and return records that became ready/completed."""
        if self.migration_manager is None:
            return ()
        records: list[MigrationRecord] = []
        for record in self.migration_manager.active():
            records.append(self.migration_manager.poll(record.request.session_id, now=self._observed(now)))
        return tuple(records)

    def _prepare_migration(self, candidate: DispatchCandidate, *, now: float) -> bool:
        if self.migration_manager is None or self.migration_backend_factory is None:
            # A policy caller that has no migration backend must not silently
            # execute a remote-state candidate.  It can still use local GPU
            # candidates by supplying a migration-disabled estimator/config.
            return False
        assert candidate.gpu_id is not None
        for session_id in candidate.session_ids:
            state = self.scheduler.session(session_id)
            if state.owner_gpu == candidate.gpu_id:
                continue
            active = next(
                (
                    record
                    for record in self.migration_manager.active()
                    if record.request.session_id == session_id
                ),
                None,
            )
            if active is None:
                request = MigrationRequest(
                    session_id=session_id,
                    source_gpu=state.owner_gpu,
                    target_gpu=candidate.gpu_id,
                    requested_at=now,
                    estimated_ready_at=max(now, candidate.start_at),
                )
                self.migration_manager.begin(
                    request,
                    self.migration_backend_factory(request),
                )
                self.scheduler.set_migration_ready(
                    session_id,
                    target_gpu=candidate.gpu_id,
                    ready_at=request.estimated_ready_at,
                    now=now,
                )
                return False
            record = self.migration_manager.poll(session_id, now=now)
            if record.state != "ready":
                return False
            self.migration_manager.commit(session_id, now=now)
            self.scheduler.commit_migration(session_id, target_gpu=candidate.gpu_id, now=now)
            return False
        return True

    def _observed(self, now: float | None) -> float:
        observed = self._clock() if now is None else now
        if not math.isfinite(observed) or observed < 0:
            raise ValueError("observed time must be finite and non-negative")
        return observed

