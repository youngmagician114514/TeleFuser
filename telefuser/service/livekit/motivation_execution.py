"""LiveKit execution adapter for the motivation scheduling control plane.

The policy scheduler is transport agnostic.  This module supplies the small
adapter that turns a reserved ``(B, c, g)`` candidate into a bounded batch of
pipeline control messages and commits the reservation when model output has
entered the worker output path.  It intentionally does not reach into model
objects; workers and pipeline adapters remain the ownership boundary.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .motivation_batch_gate import MotivationBatchGate
from .motivation_controller import DispatchLease, MotivationRuntimeController

ControlDispatch = Callable[[DispatchLease, Sequence[tuple[str, dict[str, Any]]]], None]
ReleasePolicy = Callable[[dict[str, Any]], bool]

logger = logging.getLogger(__name__)
_EPSILON = 1e-9


@dataclass
class _LeaseProgress:
    """Track output arrival for one reserved batch."""

    lease: DispatchLease
    pending_session_ids: set[str] = field(default_factory=set)


def release_on_control_state(chunk: dict[str, Any]) -> bool:
    """Release a job for one complete heartbeat/control-state message.

    A caller that aggregates sub-second input transitions can provide a
    different policy. The bridge applies the one-second gate around this
    predicate, so a browser's immediate key transition updates latest state
    without creating another job before the next heartbeat window.
    """

    if chunk.get("type") == "control_state":
        return bool(chunk.get("controls"))
    return chunk.get("type") == "control"


class MotivationExecutionBridge:
    """Connect normalized worker events to a motivation runtime controller.

    ``on_control_message`` returns ``True`` when the bridge consumed the
    message.  A worker must then avoid forwarding it directly to the model;
    the bridge will release the latest action according to ``release_policy``
    and dispatch it as part of a globally selected batch.  ``stop`` messages
    are deliberately left to the normal worker path after the policy session
    is marked departed.

    The bridge is opt-in and expects the supplied controller to have one GPU
    state per worker identifier.  It can be used by both the in-process and
    parent-transport ``process-nccl`` worker pools.
    """

    def __init__(
        self,
        controller: MotivationRuntimeController,
        *,
        dispatch: ControlDispatch,
        release_policy: ReleasePolicy = release_on_control_state,
        heartbeat_interval_seconds: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        batch_gate: MotivationBatchGate | None = None,
        enable_batch_gate: bool = True,
    ) -> None:
        self.controller = controller
        self.dispatch = dispatch
        if heartbeat_interval_seconds <= 0 or not math.isfinite(heartbeat_interval_seconds):
            raise ValueError("heartbeat_interval_seconds must be positive and finite")
        self.release_policy = release_policy
        self._heartbeat_interval_seconds = float(heartbeat_interval_seconds)
        self._clock = clock
        if not isinstance(enable_batch_gate, bool):
            raise TypeError("enable_batch_gate must be a bool")
        self._batch_gate = (
            (
                batch_gate
                if batch_gate is not None
                else MotivationBatchGate(
                    controller.scheduler.profile_provider,
                    max_wait_seconds=self._heartbeat_interval_seconds,
                )
            )
            if enable_batch_gate
            else None
        )
        self._batch_gate_timer: threading.Timer | None = None
        self._batch_gate_deadline: float | None = None
        self._batch_gate_generation = 0
        self._batch_gate_wait_count = 0
        self._batch_gate_expiry_count = 0
        self._batch_gate_cancel_count = 0
        # Scheduling callbacks can be raised by dispatch/output events on
        # different worker threads, and a dispatch adapter may synchronously
        # emit another worker-free event. Serialize the bounded drain while
        # coalescing such reentrant requests instead of recursively scheduling.
        self._schedule_guard = threading.Lock()
        self._schedule_running = False
        self._schedule_requested = False
        self._schedule_force_requested = False
        self._drain_rounds_total = 0
        self._drain_limit_hits = 0
        self._drain_reentrant_events = 0
        self._drain_last_rounds = 0
        self._lock = threading.RLock()
        self._controls: dict[str, set[str]] = {}
        self._next_release_at: dict[str, float] = {}
        self._pipeline_to_session: dict[str, str] = {}
        self._leases: dict[str, _LeaseProgress] = {}
        self._job_to_lease: dict[str, str] = {}
        self._idle_wakeup_timers: dict[str, threading.Timer] = {}
        self._lease_sequence = 0
        self.controller.set_dispatch_callback(self._dispatch_lease)

    def register_session(self, session_id: str, *, owner_gpu: str, now: float | None = None) -> None:
        """Register one admitted HTTP session at pipeline-session creation."""

        self.controller.on_session_registered(
            session_id,
            owner_gpu=owner_gpu,
            now=self._observed(now),
        )
        with self._lock:
            self._controls.setdefault(session_id, set())

    def register_pipeline_session(self, session_id: str, pipeline_session_id: str) -> None:
        """Bind worker output identifiers back to an HTTP session."""

        with self._lock:
            self._pipeline_to_session[pipeline_session_id] = session_id

    def on_session_ready(self, session_id: str) -> None:
        """Start policy work after the worker has replayed pre-connect input."""
        del session_id
        self._schedule()

    def on_control_message(self, worker_id: str, session_id: str, chunk: dict[str, Any]) -> bool:
        """Consume one normalized worker control and maybe release a job."""

        del worker_id
        message_type = str(chunk.get("type", ""))
        if message_type == "stop":
            self.controller.on_session_departed(session_id, now=self._clock())
            return False
        if message_type not in {"control_state", "control"}:
            return False

        observed_at = self._clock()
        controls = self._next_controls(session_id, chunk)
        release = bool(self.release_policy(chunk))
        if self.release_policy is release_on_control_state:
            release = release and self._heartbeat_window_allows(session_id, observed_at)
        self.controller.on_action(
            session_id,
            controls,
            now=observed_at,
            # An empty state must reach the ABot service so it stops admitting
            # new model work; it never creates an action job.
            release=release and bool(controls),
        )
        if not controls:
            with self._lock:
                self._next_release_at.pop(session_id, None)
        if not controls:
            return False
        if release:
            self._schedule()
        return True

    def on_model_output(
        self,
        worker_id: str,
        pipeline_session_id: str,
        payload: dict[str, Any],
        session_runtime_metrics: dict[str, Any] | None = None,
    ) -> None:
        """Commit a lease after each selected session reaches model output."""

        del worker_id
        if payload.get("type") != "chunk":
            return
        with self._lock:
            session_id = self._pipeline_to_session.get(pipeline_session_id)
            lease_id = next(
                (
                    candidate_id
                    for candidate_id, progress in self._leases.items()
                    if session_id in progress.pending_session_ids
                ),
                None,
            )
            if lease_id is None or session_id is None:
                return
            progress = self._leases[lease_id]
            progress.pending_session_ids.discard(session_id)
            lease = progress.lease
            completed = not progress.pending_session_ids
            if completed:
                self._leases.pop(lease_id, None)
                for job in progress.lease.jobs:
                    self._job_to_lease.pop(job.job_id, None)
        if session_id is not None and isinstance(session_runtime_metrics, dict):
            compatibility_key = session_runtime_metrics.get("batch_compatibility_key")
            if isinstance(compatibility_key, str):
                self.controller.on_session_compatibility(
                    session_id,
                    (compatibility_key,),
                    now=self._clock(),
                )
        if not completed:
            return
        self.controller.on_completion(lease, completed_at=self._clock())
        self._schedule()

    def on_chunk_published(
        self,
        worker_id: str,
        session_id: str,
        frames: int,
        first_frame_at: float | None = None,
    ) -> None:
        """Wake idle scheduling after a generated chunk reaches the publisher."""
        del worker_id, frames, first_frame_at
        self._schedule()
        try:
            state = self.controller.scheduler.session(session_id)
        except KeyError:
            return
        remaining = max(0.0, state.idle_video_remaining_seconds)
        if remaining <= 1e-9:
            self._schedule()
            return
        with self._lock:
            previous = self._idle_wakeup_timers.pop(session_id, None)
            if previous is not None:
                previous.cancel()
            timer = threading.Timer(remaining, self._idle_wakeup, args=(session_id,))
            timer.daemon = True
            self._idle_wakeup_timers[session_id] = timer
            timer.start()

    def _idle_wakeup(self, session_id: str) -> None:
        """Re-run policy after the published idle chunk's playback duration."""
        with self._lock:
            self._idle_wakeup_timers.pop(session_id, None)
        self._schedule()

    def on_session_departed(self, session_id: str, *, now: float | None = None) -> None:
        """Stop future policy work while allowing an in-flight lease to drain."""

        self.controller.on_session_departed(session_id, now=self._observed(now))
        self._cancel_batch_gate()
        with self._lock:
            timer = self._idle_wakeup_timers.pop(session_id, None)
            if timer is not None:
                timer.cancel()
            # Keep the pipeline mapping until an in-flight output can commit
            # its reservation. The runtime owns final mapping cleanup.
            self._controls.pop(session_id, None)
            self._next_release_at.pop(session_id, None)
        # A departing session may have been the only candidate holding the
        # collection timer; immediately reconsider any other pending action.
        self._schedule()

    def close(self) -> None:
        """Close the controller's optional async-search executor."""

        self._cancel_batch_gate()
        with self._lock:
            timers = tuple(self._idle_wakeup_timers.values())
            self._idle_wakeup_timers.clear()
        for timer in timers:
            timer.cancel()
        self.controller.close()

    def _dispatch_lease(self, lease: DispatchLease) -> None:
        payloads = tuple(
            (
                job.session_id,
                {
                    "type": "control_state",
                    "controls": list(job.controls),
                    "motivation": {
                        "job_id": job.job_id,
                        "kind": job.kind,
                        "batch_size": lease.candidate.batch_size,
                        "fidelity": lease.candidate.fidelity,
                        "one_shot": True,
                        "gpu_id": lease.candidate.gpu_id,
                    },
                },
            )
            for job in lease.jobs
        )
        if not payloads:
            return
        with self._lock:
            self._lease_sequence += 1
            lease_id = f"lease:{self._lease_sequence:08d}"
            self._leases[lease_id] = _LeaseProgress(
                lease=lease,
                pending_session_ids={session_id for session_id, _ in payloads},
            )
            for job in lease.jobs:
                self._job_to_lease[job.job_id] = lease_id
        try:
            self.dispatch(lease, payloads)
        except Exception:
            with self._lock:
                self._leases.pop(lease_id, None)
                for job in lease.jobs:
                    if self._job_to_lease.get(job.job_id) == lease_id:
                        self._job_to_lease.pop(job.job_id, None)
            # A transport can disappear between candidate validation and the
            # parent queue write while a browser tears down. Keep the failure
            # visible at the policy boundary instead of silently losing it.
            logger.exception(
                "Motivation dispatch failed: lease=%s sessions=%s gpu=%s",
                lease_id,
                [session_id for session_id, _ in payloads],
                lease.candidate.gpu_id,
            )
            raise

    def schedule_wakeup(self) -> None:
        """Retry the policy after an external event completes."""
        self._schedule()

    def _schedule(self, *, force_batch_gate: bool = False) -> None:
        """Drain immediately runnable leases without allowing reentrant storms.

        A controller call reserves at most one GPU slot.  On a worker/action
        event, repeatedly invoke it while newly selected leases can start now;
        the number of rounds is bounded by the configured GPU count.  A
        singleton batch gate remains a hard stop for that round, so the drain
        never turns profile-driven aggregation into a busy loop.  Callbacks
        that arrive while dispatch is in progress are coalesced and observed
        by the next round.
        """
        with self._schedule_guard:
            if self._schedule_running:
                self._schedule_requested = True
                self._schedule_force_requested |= force_batch_gate
                self._drain_reentrant_events += 1
                return
            self._schedule_running = True

        rounds = 0
        force = force_batch_gate
        try:
            limit = self._immediate_drain_limit()
            while rounds < limit:
                rounds += 1
                lease = self._schedule_round(force_batch_gate=force)
                force = False
                with self._schedule_guard:
                    requested = self._schedule_requested
                    force = self._schedule_force_requested
                    self._schedule_requested = False
                    self._schedule_force_requested = False
                # A successful lease reserves one GPU timeline, so another
                # round can only fill a different currently-free slot.  If no
                # lease was produced, retry once only when a reentrant event
                # was coalesced during the round; otherwise a gate/no-work
                # result should return and let its timer/event wake us later.
                if lease is None and not requested:
                    break
            if rounds >= limit and (lease is not None or requested):
                with self._schedule_guard:
                    self._drain_limit_hits += 1
                logger.debug("Motivation immediate drain reached bound: rounds=%d", limit)
        finally:
            with self._schedule_guard:
                self._drain_rounds_total += rounds
                self._drain_last_rounds = rounds
                # Do not recurse after the bound. A later worker event will
                # retry; dropping the coalesced bit here guarantees that a
                # pathological synchronous callback cannot spin indefinitely.
                self._schedule_requested = False
                self._schedule_force_requested = False
                self._schedule_running = False

    def _schedule_round(self, *, force_batch_gate: bool) -> DispatchLease | None:
        """Run one policy search/dispatch round and return its lease, if any."""
        observed_at = self._clock()
        try:
            if not force_batch_gate and self._defer_singleton_action(observed_at):
                return None
            lease = self.controller.schedule_once(now=observed_at)
        except Exception:
            logger.exception("Motivation scheduling callback failed")
            return None
        if lease is not None:
            self._cancel_batch_gate()
            logger.debug(
                "Motivation lease dispatched: jobs=%s gpu=%s fidelity=%s",
                len(lease.jobs),
                lease.candidate.gpu_id,
                lease.candidate.fidelity,
            )
        return lease

    def _immediate_drain_limit(self) -> int:
        """Return a finite round bound based on the registered GPU count."""
        try:
            gpu_count = len(self.controller.scheduler.gpus())
        except (AttributeError, TypeError):
            gpu_count = 1
        return max(1, gpu_count)

    def _defer_singleton_action(self, observed_at: float) -> bool:
        """Arm a profile-derived gate before dispatching a singleton action.

        The gate is deliberately action-only. Idle sentinels, remote migration
        candidates, busy GPUs, and candidates whose predicted slack would be
        exhausted by the wait are dispatched through the normal controller
        path. A single deadline is retained across input updates so repeated
        heartbeats cannot postpone a pending action indefinitely.
        """
        gate = self._batch_gate
        if gate is None:
            return False
        scheduler = self.controller.scheduler
        has_action = any(
            not state.departed and state.pending_action is not None
            for state in scheduler.sessions()
        )
        if not has_action:
            self._cancel_batch_gate()
            return False
        try:
            pending_migrations = self.controller.pending_migration_sessions()
            search_at = max(observed_at, scheduler.current_time)
            blocked_migrations = self.controller.blocked_migration_sessions(now=search_at)
            candidate = scheduler.find_best(
                now=search_at,
                include_wait=False,
                allow_migrations=True,
                exclude_session_ids=pending_migrations,
                blocked_migration_session_ids=blocked_migrations,
            )
        except (AttributeError, KeyError, ValueError):
            # A transient state/version race is handled by the controller's
            # regular search path; never let the optional gate block service.
            return False
        if candidate is None:
            return self._batch_gate_is_armed(search_at)
        if not self._candidate_is_singleton_action(candidate, search_at):
            self._cancel_batch_gate()
            return False
        # A gate is only useful when this is the sole pending action. If an
        # independent action is already ready, dispatch immediately and let
        # the bounded drain fill the other free GPU(s); delaying it would make
        # a global singleton gate strand capacity for no aggregation benefit.
        if self._has_other_pending_actions(candidate.session_ids):
            self._cancel_batch_gate()
            return False
        assert candidate.gpu_id is not None
        try:
            wait_seconds = gate.wait_seconds(
                gpu_id=candidate.gpu_id,
                fidelity=candidate.fidelity,
            )
            # Keep a custom gate bounded by the bridge's one-heartbeat release
            # period as well as the default gate's profile cap.
            wait_seconds = min(wait_seconds, self._heartbeat_interval_seconds)
        except (KeyError, ValueError, TypeError):
            logger.debug("Motivation batch gate profile lookup failed", exc_info=True)
            wait_seconds = 0.0
        if wait_seconds <= _EPSILON or not self._batch_wait_preserves_slack(candidate, wait_seconds):
            self._cancel_batch_gate()
            return False
        with self._lock:
            deadline = self._batch_gate_deadline
        if deadline is not None and search_at >= deadline - _EPSILON:
            self._cancel_batch_gate()
            return False
        self._arm_batch_gate(search_at, wait_seconds)
        return True

    def _has_other_pending_actions(self, selected_session_ids: Sequence[str]) -> bool:
        """Return whether another session can provide action work now.

        The scheduler intentionally gives action jobs global priority over idle
        sentinels. Mirror that narrow rule here rather than probing a second
        candidate (which would duplicate policy scoring): an existing action
        on another session is enough reason to dispatch and drain immediately.
        """
        selected = set(selected_session_ids)
        return any(
            not state.departed
            and state.session_id not in selected
            and state.pending_action is not None
            and state.in_flight is None
            for state in self.controller.scheduler.sessions()
        )

    def _candidate_is_singleton_action(self, candidate: Any, observed_at: float) -> bool:
        """Check that a candidate is an immediately executable action B1."""
        if candidate.wait or candidate.batch_size != 1 or candidate.migration_count:
            return False
        if candidate.gpu_id is None or candidate.fidelity is None:
            return False
        ready_now = getattr(self.controller.scheduler, "candidate_ready_now", None)
        if callable(ready_now):
            if not ready_now(candidate, now=observed_at):
                return False
        elif candidate.start_at > observed_at + _EPSILON:
            return False
        session_id = candidate.session_ids[0]
        job_id = candidate.job_ids[0]
        try:
            state = self.controller.scheduler.session(session_id)
        except KeyError:
            return False
        job = state.ready_job(include_idle=True)
        return job is not None and job.job_id == job_id and job.kind == "action"

    def _batch_wait_preserves_slack(self, candidate: Any, wait_seconds: float) -> bool:
        """Reject a gate if waiting would make any active session unsafe.

        The wait delays playback for every session, not only the singleton
        candidate. Check the complete projected system state so a near-deadline
        unselected session can force immediate dispatch.
        """
        for state in self.controller.scheduler.sessions():
            if state.departed or not state.playback_active:
                continue
            projected = candidate.projected_slack.get(state.session_id)
            if projected is None or projected - wait_seconds <= _EPSILON:
                return False
            if state.slack_seconds - wait_seconds <= _EPSILON:
                return False
        return True

    def _arm_batch_gate(self, observed_at: float, wait_seconds: float) -> None:
        """Arm or tighten the singleton gate timer without extending its deadline."""
        desired_deadline = observed_at + wait_seconds
        old_timer: threading.Timer | None = None
        with self._lock:
            current_deadline = self._batch_gate_deadline
            if current_deadline is not None and current_deadline > observed_at + _EPSILON:
                deadline = min(current_deadline, desired_deadline)
            else:
                deadline = desired_deadline
            if (
                self._batch_gate_timer is not None
                and current_deadline is not None
                and abs(current_deadline - deadline) <= _EPSILON
            ):
                return
            old_timer = self._batch_gate_timer
            self._batch_gate_generation += 1
            generation = self._batch_gate_generation
            self._batch_gate_deadline = deadline
            self._batch_gate_wait_count += 1
            timer = threading.Timer(
                max(0.0, deadline - observed_at),
                self._batch_gate_wakeup,
                args=(generation,),
            )
            timer.daemon = True
            self._batch_gate_timer = timer
        if old_timer is not None:
            old_timer.cancel()
        timer.start()

    def _batch_gate_wakeup(self, generation: int) -> None:
        """Retry scheduling once the profile-derived collection window expires."""
        with self._lock:
            if generation != self._batch_gate_generation:
                return
            self._batch_gate_timer = None
            self._batch_gate_deadline = None
            self._batch_gate_expiry_count += 1
        self._schedule(force_batch_gate=True)

    def _batch_gate_is_armed(self, observed_at: float) -> bool:
        with self._lock:
            return (
                self._batch_gate_timer is not None
                and self._batch_gate_deadline is not None
                and observed_at < self._batch_gate_deadline - _EPSILON
            )

    def _cancel_batch_gate(self) -> None:
        """Cancel a pending collection timer and invalidate its callback."""
        with self._lock:
            self._batch_gate_generation += 1
            timer = self._batch_gate_timer
            self._batch_gate_timer = None
            self._batch_gate_deadline = None
            if timer is not None:
                self._batch_gate_cancel_count += 1
        if timer is not None:
            timer.cancel()

    def snapshot(self) -> dict[str, Any]:
        """Return bounded policy state for diagnosing scheduling progress."""
        with self._lock:
            leases = {
                lease_id: {
                    "session_ids": sorted(progress.pending_session_ids),
                    "job_ids": [job.job_id for job in progress.lease.jobs],
                    "gpu_id": progress.lease.candidate.gpu_id,
                    "fidelity": progress.lease.candidate.fidelity,
                    "reserved_at": progress.lease.reserved_at,
                }
                for lease_id, progress in self._leases.items()
            }
            controls = {session_id: sorted(values) for session_id, values in self._controls.items()}
            next_release_at = dict(self._next_release_at)
            batch_gate_deadline = self._batch_gate_deadline
            batch_gate_armed = self._batch_gate_timer is not None and batch_gate_deadline is not None
            batch_gate_stats = {
                "enabled": self._batch_gate is not None,
                "armed": batch_gate_armed,
                "deadline": batch_gate_deadline,
                "wait_count": self._batch_gate_wait_count,
                "expiry_count": self._batch_gate_expiry_count,
                "cancel_count": self._batch_gate_cancel_count,
            }
        with self._schedule_guard:
            immediate_drain_stats = {
                "running": self._schedule_running,
                "rounds_total": self._drain_rounds_total,
                "last_rounds": self._drain_last_rounds,
                "limit_hits": self._drain_limit_hits,
                "reentrant_events": self._drain_reentrant_events,
            }
        if batch_gate_armed and batch_gate_deadline is not None:
            batch_gate_stats["remaining_seconds"] = max(0.0, batch_gate_deadline - self._clock())
        else:
            batch_gate_stats["remaining_seconds"] = 0.0
        sessions = {}
        for state in self.controller.scheduler.sessions():
            sessions[state.session_id] = {
                "owner_gpu": state.owner_gpu,
                "departed": state.departed,
                "pending_action": state.pending_action.job_id if state.pending_action else None,
                "pending_idle": state.pending_idle.job_id if state.pending_idle else None,
                "in_flight": state.in_flight.job_id if state.in_flight else None,
                "latest_controls": list(state.latest_controls),
                "state_version": state.state_version,
                "slack_seconds": state.slack_seconds,
                "quality_ema": state.quality_ema,
                "migration_target_gpu": state.migration_target_gpu,
            }
        migration_manager = self.controller.migration_manager
        return {
            "epoch": self.controller.scheduler.epoch,
            "current_time": self.controller.scheduler.current_time,
            "leases": leases,
            "sessions": sessions,
            "controls": controls,
            "next_release_at": next_release_at,
            "batch_gate": batch_gate_stats,
            "immediate_drain": immediate_drain_stats,
            "diagnostics": self.controller.diagnostics_snapshot(),
            "active_migrations": [
                {
                    "session_id": record.request.session_id,
                    "source_gpu": record.request.source_gpu,
                    "target_gpu": record.request.target_gpu,
                    "state": record.state,
                }
                for record in (migration_manager.active() if migration_manager is not None else ())
            ],
        }

    def _heartbeat_window_allows(self, session_id: str, observed_at: float) -> bool:
        """Release the first input immediately, then at most once per heartbeat."""
        with self._lock:
            next_release_at = self._next_release_at.get(session_id)
            if next_release_at is not None and observed_at < next_release_at - 1e-9:
                return False
            self._next_release_at[session_id] = observed_at + self._heartbeat_interval_seconds
            return True

    def _next_controls(self, session_id: str, chunk: dict[str, Any]) -> tuple[str, ...]:
        with self._lock:
            controls = self._controls.setdefault(session_id, set())
            message_type = str(chunk.get("type", ""))
            if message_type == "control_state":
                values = chunk.get("controls", [])
                controls.clear()
                if isinstance(values, list):
                    controls.update(str(value) for value in values if str(value))
            elif message_type == "control":
                value = str(chunk.get("control", chunk.get("key", "")))
                event = str(chunk.get("event") or chunk.get("action") or "press").lower()
                if event == "press" and value:
                    controls.add(value)
                elif event in {"release", "up"}:
                    controls.discard(value)
                elif event in {"reset", "reset_pose"}:
                    controls.clear()
            return tuple(sorted(controls))

    def _observed(self, now: float | None) -> float:
        return self._clock() if now is None else now
