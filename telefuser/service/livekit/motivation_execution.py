"""LiveKit execution adapter for the motivation scheduling control plane.

The policy scheduler is transport agnostic.  This module supplies the small
adapter that turns a reserved ``(B, c, g)`` candidate into a bounded batch of
pipeline control messages and commits the reservation when model output has
entered the worker output path.  It intentionally does not reach into model
objects; workers and pipeline adapters remain the ownership boundary.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .motivation_controller import DispatchLease, MotivationRuntimeController

ControlDispatch = Callable[[DispatchLease, Sequence[tuple[str, dict[str, Any]]]], None]
ReleasePolicy = Callable[[dict[str, Any]], bool]


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
    ) -> None:
        self.controller = controller
        self.dispatch = dispatch
        if heartbeat_interval_seconds <= 0 or not math.isfinite(heartbeat_interval_seconds):
            raise ValueError("heartbeat_interval_seconds must be positive and finite")
        self.release_policy = release_policy
        self._heartbeat_interval_seconds = float(heartbeat_interval_seconds)
        self._clock = clock
        self._lock = threading.RLock()
        self._controls: dict[str, set[str]] = {}
        self._next_release_at: dict[str, float] = {}
        self._pipeline_to_session: dict[str, str] = {}
        self._leases: dict[str, _LeaseProgress] = {}
        self._job_to_lease: dict[str, str] = {}
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
            if progress.pending_session_ids:
                return
            self._leases.pop(lease_id, None)
            for job in progress.lease.jobs:
                self._job_to_lease.pop(job.job_id, None)
            lease = progress.lease
        self.controller.on_completion(lease, completed_at=self._clock())
        self._schedule()

    def on_session_departed(self, session_id: str, *, now: float | None = None) -> None:
        """Stop future policy work while allowing an in-flight lease to drain."""

        self.controller.on_session_departed(session_id, now=self._observed(now))
        with self._lock:
            # Keep the pipeline mapping until an in-flight output can commit
            # its reservation. The runtime owns final mapping cleanup.
            self._controls.pop(session_id, None)
            self._next_release_at.pop(session_id, None)

    def close(self) -> None:
        """Close the controller's optional async-search executor."""

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
            if job.kind == "action"
        )
        if not payloads:
            # Idle sentinels are policy-side work.  A model-specific runtime
            # may provide an idle executor later; never pretend that an idle
            # marker generated a video by sending an empty action to ABot.
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
        self.dispatch(lease, payloads)

    def _schedule(self) -> None:
        self.controller.schedule_once(now=self._clock())

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
