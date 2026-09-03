"""Global slack-aware scheduling primitives for stateful LiveKit sessions.

The module deliberately has no CUDA or LiveKit transport dependency.  It is
the control-plane part of the motivation scheduler: action events become
bounded session-local jobs, a global snapshot is searched for feasible
``(B, c, g)`` candidates, and a versioned reservation prevents an
asynchronously computed decision from dispatching stale state.

Model workers remain responsible for executing a reserved batch.  The worker
integration can therefore evolve independently from the policy and its CPU
unit tests.
"""

from __future__ import annotations

import csv
import itertools
import math
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from .motivation_diagnostics import (
    MotivationDiagnosticsSink,
    MotivationDispatchSummary,
    MotivationSearchSummary,
    NullMotivationDiagnostics,
    empty_batch_counts,
)

EPSILON = 1e-9
JobKind = Literal["action", "idle"]


@dataclass(frozen=True)
class MotivationProfile:
    """Measured execution point for one batch size and fidelity.

    ``gpu_id=None`` denotes a profile shared by homogeneous GPU replicas.
    A GPU-specific row takes precedence over a shared row.  Quality is the
    offline-table quality value used by the policy; the runtime does not run
    a visual evaluator in the dispatch critical path.
    """

    batch_size: int
    fidelity: str
    latency_seconds: float
    quality: float
    memory_gb: float
    output_seconds: float = 1.0
    p95_seconds: float | None = None
    gpu_id: str | None = None

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.batch_size > 4:
            raise ValueError("batch_size must be in [1, 4]")
        if not self.fidelity:
            raise ValueError("fidelity must be non-empty")
        if not math.isfinite(self.latency_seconds) or self.latency_seconds <= 0:
            raise ValueError("latency_seconds must be positive and finite")
        if not math.isfinite(self.quality) or self.quality <= 0:
            raise ValueError("quality must be positive and finite")
        if not math.isfinite(self.memory_gb) or self.memory_gb <= 0:
            raise ValueError("memory_gb must be positive and finite")
        if not math.isfinite(self.output_seconds) or self.output_seconds <= 0:
            raise ValueError("output_seconds must be positive and finite")
        if self.p95_seconds is not None and (
            not math.isfinite(self.p95_seconds) or self.p95_seconds <= 0
        ):
            raise ValueError("p95_seconds must be positive and finite when supplied")


def load_motivation_profiles_csv(
    path: str | Path,
    *,
    max_batch_size: int = 4,
    gpu_id: str | None = None,
    output_seconds: float = 1.0,
) -> StaticMotivationProfileTable:
    """Load the measured offline-table rows used by the policy.

    The loader accepts the ABot profile schema directly.  ``Q_world`` is the
    preferred quality column; if it is empty, the mean of available
    ``Q_action``, ``Q_temporal`` and ``Q_visual`` values is used.  The rows are
    tagged as homogeneous (or with ``gpu_id`` when supplied), so GPU-specific
    measurements can coexist with shared fallback rows.
    """
    if not 1 <= max_batch_size <= 4:
        raise ValueError("max_batch_size must be in [1, 4]")
    if output_seconds <= 0 or not math.isfinite(output_seconds):
        raise ValueError("output_seconds must be positive and finite")
    rows: list[MotivationProfile] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            batch_size = int(raw["B"])
            if batch_size > max_batch_size:
                continue
            latency_ms = float(raw["latency_ms"])
            p95_raw = raw.get("latency_p95_ms", "")
            p95_ms = float(p95_raw) if p95_raw not in {None, ""} else latency_ms
            memory_gb = float(raw["memory_GB"])
            quality_raw = raw.get("Q_world", "")
            if quality_raw in {None, ""}:
                quality_values = [
                    float(raw[key])
                    for key in ("Q_action", "Q_temporal", "Q_visual")
                    if raw.get(key, "") not in {None, ""}
                ]
                if not quality_values:
                    raise ValueError(f"profile row {raw.get('config', '<unknown>')} has no quality value")
                quality = sum(quality_values) / len(quality_values)
            else:
                quality = float(quality_raw)
            rows.append(
                MotivationProfile(
                    batch_size=batch_size,
                    fidelity=str(raw.get("config") or f"b{batch_size}"),
                    latency_seconds=latency_ms / 1000.0,
                    p95_seconds=p95_ms / 1000.0,
                    quality=quality,
                    memory_gb=memory_gb,
                    output_seconds=output_seconds,
                    gpu_id=gpu_id,
                )
            )
    if not rows:
        raise ValueError(f"no profiles with batch size <= {max_batch_size} found in {path}")
    return StaticMotivationProfileTable(rows)


class MotivationProfileProvider(Protocol):
    """Lookup interface used by the policy search."""

    def profiles_for(self, *, batch_size: int, gpu_id: str) -> Sequence[MotivationProfile]:
        """Return all measured fidelity points feasible for ``gpu_id``."""


class StaticMotivationProfileTable:
    """Immutable-style lookup table backed by measured offline profiles."""

    def __init__(self, profiles: Iterable[MotivationProfile]) -> None:
        rows = tuple(profiles)
        if not rows:
            raise ValueError("at least one profile is required")
        self._rows: dict[tuple[str | None, int, str], MotivationProfile] = {}
        for profile in rows:
            key = (profile.gpu_id, profile.batch_size, profile.fidelity)
            if key in self._rows:
                raise ValueError(f"duplicate profile row: {key!r}")
            self._rows[key] = profile

    def profiles_for(self, *, batch_size: int, gpu_id: str) -> tuple[MotivationProfile, ...]:
        """Return GPU-specific rows followed by shared homogeneous rows."""
        specific = {
            profile.fidelity: profile
            for (row_gpu, row_batch, _), profile in self._rows.items()
            if row_gpu == gpu_id and row_batch == batch_size
        }
        shared = {
            profile.fidelity: profile
            for (row_gpu, row_batch, _), profile in self._rows.items()
            if row_gpu is None and row_batch == batch_size
        }
        merged = dict(shared)
        merged.update(specific)
        return tuple(merged[fidelity] for fidelity in sorted(merged))


@dataclass(frozen=True)
class ActionJob:
    """One algorithm-side job candidate for a session."""

    job_id: str
    session_id: str
    kind: JobKind
    controls: tuple[str, ...]
    created_at: float
    state_version: int


@dataclass
class SessionSchedulingState:
    """Mutable control-plane state for one retained model session.

    There is at most one pending action job and one pending idle sentinel.  A
    newer action replaces an older pending action, but it never replaces an
    already generated idle video.  ``in_flight`` is allowed to coexist with a
    pending action because a control update cannot cancel a running model
    invocation.
    """

    session_id: str
    owner_gpu: str
    slack_seconds: float
    quality_ema: float
    quality_update_rate: float = 0.2
    compatibility_key: tuple[object, ...] = ()
    active: bool = True
    departed: bool = False
    playback_active: bool = True
    last_updated_at: float = 0.0
    quality_weight_seconds: float = 0.0
    latest_controls: tuple[str, ...] = ()
    pending_action: ActionJob | None = None
    pending_idle: ActionJob | None = None
    in_flight: ActionJob | None = None
    idle_video_remaining_seconds: float = 0.0
    state_version: int = 0
    migration_target_gpu: str | None = None
    migration_ready_at: float = 0.0

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must be non-empty")
        if not self.owner_gpu:
            raise ValueError("owner_gpu must be non-empty")
        if not math.isfinite(self.slack_seconds):
            raise ValueError("slack_seconds must be finite")
        if not math.isfinite(self.quality_ema) or self.quality_ema <= 0:
            raise ValueError("quality_ema must be positive and finite")
        if not 0 < self.quality_update_rate <= 1:
            raise ValueError("quality_update_rate must be in (0, 1]")
        if not 0 <= self.quality_weight_seconds:
            raise ValueError("quality_weight_seconds must be non-negative")
        if self.last_updated_at < 0 or not math.isfinite(self.last_updated_at):
            raise ValueError("last_updated_at must be finite and non-negative")

    @property
    def idle_video_outstanding(self) -> bool:
        """Whether an idle output is still waiting to be consumed."""
        return self.idle_video_remaining_seconds > EPSILON

    def advance_to(self, now: float) -> None:
        """Consume playback slack and any idle output until ``now``.

        Playback is intentionally independent of the latest action state:
        once frames enter the consumer queue, they continue to play after the
        user releases the controls.  A caller can set ``playback_active`` to
        false for an explicitly paused consumer.
        """
        if not math.isfinite(now) or now < self.last_updated_at - EPSILON:
            raise ValueError("now must be finite and monotonic")
        elapsed = max(0.0, now - self.last_updated_at)
        if self.playback_active and not self.departed:
            self.slack_seconds -= elapsed
            if self.idle_video_remaining_seconds > EPSILON:
                self.idle_video_remaining_seconds = max(
                    0.0, self.idle_video_remaining_seconds - elapsed
                )
        self.last_updated_at = now

    def submit_action(
        self,
        *,
        job_id: str,
        controls: Iterable[str],
        now: float,
        release: bool,
    ) -> bool:
        """Update the latest controls and optionally release an action job.

        Empty controls and non-release heartbeat updates do not create jobs.
        The return value reports whether the session's ready queue transitioned
        from no pending action to a pending action.  An in-flight job does not
        occupy that ready slot, so an action arriving during execution still
        invalidates a search while leaving the running job untouched.
        """
        self.advance_to(now)
        canonical = tuple(sorted({str(control) for control in controls if str(control)}))
        self.latest_controls = canonical
        if not release or not canonical or self.departed:
            return False
        # A not-yet-dispatched idle sentinel describes the old no-action state.
        # A new action makes it stale and drops only that pending sentinel; an
        # idle job already in flight is retained and its generated video remains
        # protected from overwrite.
        self.pending_idle = None
        had_pending = self.pending_action is not None
        self.state_version += 1
        self.pending_action = ActionJob(
            job_id=job_id,
            session_id=self.session_id,
            kind="action",
            controls=canonical,
            created_at=now,
            state_version=self.state_version,
        )
        return not had_pending

    def create_idle_job(self, *, job_id: str, now: float) -> ActionJob | None:
        """Create one idle sentinel if no action or idle output is pending."""
        self.advance_to(now)
        # A held latest action state still represents user work even when its
        # next heartbeat has not released a replacement job. Do not insert an
        # idle continuation into that gap; the next action heartbeat must win.
        if self.departed or self.latest_controls or self.pending_action is not None:
            return None
        if self.pending_idle is not None or self.in_flight is not None:
            return None
        if self.idle_video_outstanding:
            return None
        self.state_version += 1
        self.pending_idle = ActionJob(
            job_id=job_id,
            session_id=self.session_id,
            kind="idle",
            controls=(),
            created_at=now,
            state_version=self.state_version,
        )
        return self.pending_idle

    def ready_job(self, *, include_idle: bool) -> ActionJob | None:
        """Return the highest-priority job for this session."""
        if self.pending_action is not None:
            return self.pending_action
        if include_idle:
            return self.pending_idle
        return None

    def mark_dispatched(self, job: ActionJob) -> None:
        """Move a current pending job into the in-flight slot."""
        if self.in_flight is not None:
            raise RuntimeError(f"session {self.session_id} already has an in-flight job")
        if job.kind == "action":
            if self.pending_action is None or self.pending_action.job_id != job.job_id:
                raise RuntimeError(f"action job {job.job_id} is no longer pending")
            self.pending_action = None
        else:
            if self.pending_idle is None or self.pending_idle.job_id != job.job_id:
                raise RuntimeError(f"idle job {job.job_id} is no longer pending")
            self.pending_idle = None
        self.in_flight = job

    def rollback_dispatched(self, job: ActionJob) -> None:
        """Return a synchronously rejected dispatch to the ready queue.

        Dispatch rollback is deliberately narrower than completion: it does
        not advance slack, quality, or cache ownership.  A newer action may
        already occupy ``pending_action`` in an unusual reentrant adapter; in
        that case the latest action wins and the rejected older job is simply
        cleared from the in-flight slot.
        """
        if self.in_flight is None or self.in_flight.job_id != job.job_id:
            raise RuntimeError(f"job {job.job_id} is not in flight for session {self.session_id}")
        self.in_flight = None
        if job.kind == "action":
            if self.pending_action is None:
                self.pending_action = job
        elif self.pending_action is None and self.pending_idle is None and not self.departed:
            self.pending_idle = job

    def complete_job(self, *, completed_at: float, output_seconds: float, quality: float) -> ActionJob:
        """Complete the current job and update quality/idle consumption state."""
        self.advance_to(completed_at)
        job = self.in_flight
        if job is None:
            raise RuntimeError(f"session {self.session_id} has no in-flight job")
        if output_seconds <= 0 or not math.isfinite(output_seconds):
            raise ValueError("output_seconds must be positive and finite")
        if quality <= 0 or not math.isfinite(quality):
            raise ValueError("quality must be positive and finite")
        self.slack_seconds += output_seconds
        self.quality_ema = (1.0 - self.quality_update_rate) * self.quality_ema + (
            self.quality_update_rate * quality
        )
        self.quality_weight_seconds += output_seconds
        if job.kind == "idle":
            # The next idle sentinel is gated by actual playback consumption.
            self.idle_video_remaining_seconds = output_seconds
        self.in_flight = None
        return job

    def mark_departed(self, *, now: float) -> None:
        """Drop future work while allowing a reserved invocation to drain.

        A departure can race with a GPU invocation that was already reserved.
        Keep that in-flight job until the completion callback releases its GPU
        reservation; only future pending jobs are discarded.
        """
        self.advance_to(now)
        self.departed = True
        self.active = False
        self.pending_action = None
        self.pending_idle = None


@dataclass(frozen=True)
class GpuSchedulingState:
    """Scheduler-visible GPU timeline and memory facts."""

    gpu_id: str
    free_at: float = 0.0
    memory_free_gb: float = math.inf
    available: bool = True
    version: int = 0

    def __post_init__(self) -> None:
        if not self.gpu_id:
            raise ValueError("gpu_id must be non-empty")
        if not math.isfinite(self.free_at) or self.free_at < 0:
            raise ValueError("free_at must be finite and non-negative")
        if self.memory_free_gb != math.inf and (
            not math.isfinite(self.memory_free_gb) or self.memory_free_gb < 0
        ):
            raise ValueError("memory_free_gb must be non-negative and finite")


@dataclass(frozen=True)
class MigrationEstimate:
    """Predicted state-transfer readiness for one session and target GPU."""

    ready_at: float
    cost_seconds: float = 0.0
    required: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.ready_at) or self.ready_at < 0:
            raise ValueError("ready_at must be finite and non-negative")
        if not math.isfinite(self.cost_seconds) or self.cost_seconds < 0:
            raise ValueError("cost_seconds must be non-negative and finite")


class MigrationEstimator(Protocol):
    """Estimate asynchronous migration readiness without starting transfer."""

    def estimate(
        self,
        session: SessionSchedulingState,
        *,
        target_gpu: str,
        now: float,
    ) -> MigrationEstimate:
        """Return the target readiness and residual migration cost."""


class LocalMigrationEstimator:
    """Default estimator for local execution and fixed residual migration cost."""

    def __init__(self, *, migration_cost_seconds: float = 0.0) -> None:
        if migration_cost_seconds < 0 or not math.isfinite(migration_cost_seconds):
            raise ValueError("migration_cost_seconds must be non-negative and finite")
        self.migration_cost_seconds = float(migration_cost_seconds)

    def estimate(
        self,
        session: SessionSchedulingState,
        *,
        target_gpu: str,
        now: float,
    ) -> MigrationEstimate:
        if session.owner_gpu == target_gpu:
            return MigrationEstimate(ready_at=now, required=False)
        ready_at = max(now + self.migration_cost_seconds, session.migration_ready_at)
        return MigrationEstimate(
            ready_at=ready_at,
            cost_seconds=max(0.0, ready_at - now),
            required=True,
        )


@dataclass(frozen=True)
class MotivationSchedulerConfig:
    """Policy constants matching the revised paper simulator defaults."""

    max_batch_size: int = 4
    utility_cap_seconds: float = 3.0
    quality_ema: float = 0.2
    fairness_delta: float = 0.05
    lambda_quality: float = 2.0
    lambda_migration: float = 0.05
    initial_slack_seconds: float = 1.0
    initial_quality: float | None = None
    include_idle_jobs: bool = True
    migration_enabled: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.max_batch_size <= 4:
            raise ValueError("max_batch_size must be in [1, 4]")
        for name in (
            "utility_cap_seconds",
            "initial_slack_seconds",
            "fairness_delta",
            "lambda_quality",
            "lambda_migration",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.utility_cap_seconds <= 0 or self.initial_slack_seconds <= 0:
            raise ValueError("utility_cap_seconds and initial_slack_seconds must be positive")
        if not 0 < self.quality_ema <= 1:
            raise ValueError("quality_ema must be in (0, 1]")
        if self.initial_quality is not None and (
            not math.isfinite(self.initial_quality) or self.initial_quality <= 0
        ):
            raise ValueError("initial_quality must be positive and finite")


@dataclass(frozen=True)
class DispatchCandidate:
    """A complete policy candidate, including a deliberate wait option."""

    session_ids: tuple[str, ...]
    job_ids: tuple[str, ...]
    gpu_id: str | None
    fidelity: str | None
    profile: MotivationProfile | None
    start_at: float
    finish_at: float
    migration_count: int
    migration_seconds: float
    score: float
    projected_slack: Mapping[str, float]
    projected_quality: Mapping[str, float]
    snapshot_epoch: int
    session_versions: Mapping[str, int]
    gpu_version: int | None
    wait: bool = False

    @property
    def batch_size(self) -> int:
        return len(self.session_ids)

    @property
    def action_count(self) -> int:
        return sum(1 for job_id in self.job_ids if ":idle" not in job_id)


class MotivationScheduler:
    """Thread-safe global candidate search and versioned dispatch coordinator.

    This class owns policy state only.  A runtime adapter should feed it
    action/GPU/completion events and execute the returned reservation through
    the owning worker.  The adapter may run ``find_best`` in a background
    thread while the currently reserved GPU batch is executing.
    """

    def __init__(
        self,
        profile_provider: MotivationProfileProvider,
        *,
        config: MotivationSchedulerConfig | None = None,
        migration_estimator: MigrationEstimator | None = None,
        diagnostics: MotivationDiagnosticsSink | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.profile_provider = profile_provider
        self.config = config or MotivationSchedulerConfig()
        self.migration_estimator = migration_estimator or LocalMigrationEstimator()
        self._diagnostics = diagnostics or NullMotivationDiagnostics()
        self._clock = clock
        self._sessions: dict[str, SessionSchedulingState] = {}
        self._gpus: dict[str, GpuSchedulingState] = {}
        self._job_sequence = 0
        self._epoch = 0
        self._now = 0.0
        self._lock = threading.RLock()

    @property
    def epoch(self) -> int:
        """Return the state version used to invalidate asynchronous searches."""
        with self._lock:
            return self._epoch

    @property
    def current_time(self) -> float:
        """Return the latest monotonic policy time observed by the scheduler."""
        with self._lock:
            return self._now

    def add_gpu(self, state: GpuSchedulingState) -> None:
        """Register or replace a scheduler-visible GPU snapshot."""
        with self._lock:
            previous = self._gpus.get(state.gpu_id)
            version = state.version
            if previous is not None and (
                previous.free_at != state.free_at
                or previous.memory_free_gb != state.memory_free_gb
                or previous.available != state.available
            ):
                version = max(previous.version + 1, version)
            self._gpus[state.gpu_id] = GpuSchedulingState(
                gpu_id=state.gpu_id,
                free_at=state.free_at,
                memory_free_gb=state.memory_free_gb,
                available=state.available,
                version=version,
            )
            self._epoch += 1

    def update_gpu(
        self,
        gpu_id: str,
        *,
        free_at: float | None = None,
        memory_free_gb: float | None = None,
        available: bool | None = None,
        now: float | None = None,
    ) -> GpuSchedulingState:
        """Update one GPU and invalidate searches when its facts change."""
        with self._lock:
            current = self._gpus[gpu_id]
            if now is not None:
                self._advance_to(now)
            state = GpuSchedulingState(
                gpu_id=gpu_id,
                free_at=current.free_at if free_at is None else free_at,
                memory_free_gb=current.memory_free_gb if memory_free_gb is None else memory_free_gb,
                available=current.available if available is None else available,
                version=current.version + 1,
            )
            self._gpus[gpu_id] = state
            self._epoch += 1
            return state

    def register_session(
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
        """Register one retained session and its initial scheduling state."""
        observed_at = self._clock() if now is None else now
        with self._lock:
            if session_id in self._sessions:
                raise ValueError(f"session {session_id!r} is already registered")
            if owner_gpu not in self._gpus:
                raise KeyError(f"unknown owner GPU {owner_gpu!r}")
            initial_quality = quality
            if initial_quality is None:
                initial_quality = self.config.initial_quality
            if initial_quality is None:
                profiles = [
                    profile
                    for gpu_id in self._gpus
                    for batch_size in range(1, self.config.max_batch_size + 1)
                    for profile in self.profile_provider.profiles_for(batch_size=batch_size, gpu_id=gpu_id)
                ]
                if not profiles:
                    raise ValueError("profile provider has no quality values")
                initial_quality = max(profile.quality for profile in profiles)
            state = SessionSchedulingState(
                session_id=session_id,
                owner_gpu=owner_gpu,
                quality_update_rate=self.config.quality_ema,
                slack_seconds=(
                    self.config.initial_slack_seconds if slack_seconds is None else slack_seconds
                ),
                quality_ema=initial_quality,
                compatibility_key=tuple(compatibility_key),
                active=active,
                last_updated_at=observed_at,
            )
            self._sessions[session_id] = state
            self._epoch += 1
            self._now = max(self._now, observed_at)
            return state

    def submit_action(
        self,
        session_id: str,
        controls: Iterable[str],
        *,
        now: float | None = None,
        release: bool = True,
    ) -> tuple[ActionJob | None, bool]:
        """Submit an action update and report ``(job, empty_to_nonempty)``."""
        observed_at = self._clock() if now is None else now
        with self._lock:
            self._advance_to(observed_at)
            state = self._sessions[session_id]
            self._job_sequence += 1
            job_id = f"{session_id}:action:{self._job_sequence:08d}"
            before = state.pending_action
            invalidated = state.submit_action(
                job_id=job_id,
                controls=controls,
                now=observed_at,
                release=release,
            )
            job = state.pending_action if state.pending_action is not before else None
            if release and job is not None:
                # Replacement changes the session version, so a candidate that
                # captured the old job is rejected even if the global ready set
                # did not transition from empty to non-empty.
                self._epoch += 1 if invalidated else 0
            return job, invalidated

    def create_idle_job(
        self,
        session_id: str,
        *,
        now: float | None = None,
    ) -> ActionJob | None:
        """Create a consumption-gated idle sentinel for one session."""
        observed_at = self._clock() if now is None else now
        with self._lock:
            self._advance_to(observed_at)
            self._job_sequence += 1
            job = self._sessions[session_id].create_idle_job(
                job_id=f"{session_id}:idle:{self._job_sequence:08d}",
                now=observed_at,
            )
            if job is not None:
                self._epoch += 1
            return job

    def set_playback_active(self, session_id: str, active: bool, *, now: float | None = None) -> None:
        """Pause/resume slack consumption for an explicitly paused consumer."""
        observed_at = self._clock() if now is None else now
        with self._lock:
            self._advance_to(observed_at)
            state = self._sessions[session_id]
            if state.playback_active != active:
                state.playback_active = active
                self._epoch += 1

    def mark_departed(self, session_id: str, *, now: float | None = None) -> None:
        """Remove future candidates for a departed session."""
        observed_at = self._clock() if now is None else now
        with self._lock:
            self._advance_to(observed_at)
            self._sessions[session_id].mark_departed(now=observed_at)
            self._epoch += 1

    def commit_migration(
        self,
        session_id: str,
        *,
        target_gpu: str,
        now: float | None = None,
    ) -> None:
        """Commit scheduler ownership after the migration backend switches state."""
        observed_at = self._clock() if now is None else now
        with self._lock:
            self._advance_to(observed_at)
            if target_gpu not in self._gpus:
                raise KeyError(f"unknown target GPU {target_gpu!r}")
            state = self._sessions[session_id]
            if state.in_flight is not None:
                raise RuntimeError(f"cannot migrate session {session_id!r} while a job is in flight")
            state.owner_gpu = target_gpu
            state.migration_target_gpu = None
            state.migration_ready_at = observed_at
            self._epoch += 1

    def clear_migration(self, session_id: str, *, now: float | None = None) -> None:
        """Clear a failed asynchronous migration and permit a fresh search."""
        observed_at = self._clock() if now is None else now
        with self._lock:
            self._advance_to(observed_at)
            state = self._sessions[session_id]
            if state.migration_target_gpu is None and state.migration_ready_at == 0.0:
                return
            state.migration_target_gpu = None
            state.migration_ready_at = 0.0
            self._epoch += 1

    def set_migration_ready(
        self,
        session_id: str,
        *,
        target_gpu: str,
        ready_at: float,
        now: float | None = None,
    ) -> None:
        """Publish asynchronous migration readiness for candidate search."""
        observed_at = self._clock() if now is None else now
        if target_gpu not in self._gpus:
            raise KeyError(f"unknown target GPU {target_gpu!r}")
        with self._lock:
            self._advance_to(observed_at)
            state = self._sessions[session_id]
            state.migration_target_gpu = target_gpu
            state.migration_ready_at = ready_at
            self._epoch += 1

    def update_session_compatibility(
        self,
        session_id: str,
        compatibility_key: Iterable[object],
        *,
        now: float | None = None,
    ) -> None:
        """Publish the worker's current batch-compatibility state.

        The policy process cannot inspect a worker's KV cursor directly. A
        worker therefore reports a structural key after each completed
        invocation. Updating the key invalidates asynchronous searches based
        on the previous KV/layout state.
        """
        observed_at = self._clock() if now is None else now
        with self._lock:
            self._advance_to(observed_at)
            state = self._sessions[session_id]
            updated = tuple(compatibility_key)
            if state.compatibility_key == updated:
                return
            state.compatibility_key = updated
            self._epoch += 1

    def _advance_to(self, now: float) -> None:
        if not math.isfinite(now) or now < self._now - EPSILON:
            raise ValueError("scheduler time must be monotonic")
        for state in self._sessions.values():
            state.advance_to(now)
        self._now = now

    def _ready_jobs(self) -> tuple[tuple[SessionSchedulingState, ActionJob], ...]:
        action_ready = [
            (state, state.pending_action)
            for state in self._sessions.values()
            if (
                not state.departed
                and state.in_flight is None
                and state.pending_action is not None
            )
        ]
        # A session with an in-flight invocation is intentionally absent from
        # both ready lists. Its pending action remains stored on the session and
        # becomes runnable after completion releases the in-flight slot.
        # Action jobs have priority over idle sentinels globally.  This also
        # prevents an idle job from occupying a batch slot while action work is
        # waiting on another session.
        if action_ready:
            return tuple((state, job) for state, job in action_ready if job is not None)
        if not self.config.include_idle_jobs:
            return ()
        idle_ready = [
            (state, state.pending_idle)
            for state in self._sessions.values()
            if (
                not state.departed
                and state.in_flight is None
                and state.pending_idle is not None
            )
        ]
        return tuple((state, job) for state, job in idle_ready if job is not None)

    @staticmethod
    def _utility(slack: float, cap: float) -> float:
        """The simulator's ``U(P)=min(P, cap)`` utility."""
        return min(slack, cap)

    def _wait_candidate(self, *, now: float, wait_seconds: float) -> DispatchCandidate:
        wait = max(0.0, wait_seconds)
        projected_slack = {
            state.session_id: state.slack_seconds - (wait if state.playback_active else 0.0)
            for state in self._sessions.values()
            if not state.departed
        }
        score = sum(
            self._utility(value, self.config.utility_cap_seconds)
            for value in projected_slack.values()
        )
        return DispatchCandidate(
            session_ids=(),
            job_ids=(),
            gpu_id=None,
            fidelity=None,
            profile=None,
            start_at=now + wait,
            finish_at=now + wait,
            migration_count=0,
            migration_seconds=0.0,
            score=score,
            projected_slack=projected_slack,
            projected_quality={
                state.session_id: state.quality_ema
                for state in self._sessions.values()
                if not state.departed
            },
            snapshot_epoch=self._epoch,
            session_versions={
                state.session_id: state.state_version
                for state in self._sessions.values()
                if not state.departed
            },
            gpu_version=None,
            wait=True,
        )

    def find_best(
        self,
        *,
        now: float | None = None,
        gpu_states: Sequence[GpuSchedulingState] | None = None,
        wait_seconds: float = 0.0,
        include_wait: bool = True,
        allow_migrations: bool = True,
        exclude_session_ids: Iterable[str] = (),
        blocked_migration_session_ids: Iterable[str] = (),
    ) -> DispatchCandidate | None:
        """Enumerate and score all feasible candidates in a global snapshot.

        Only one job is selected from each session.  All members of a batch
        share one fidelity and one target GPU.  A candidate includes all
        non-departed sessions in its slack utility, not just selected members.
        blocked_migration_session_ids excludes candidates that would move one
        of those sessions; work on its current owner remains eligible.
        """
        observed_at = self._clock() if now is None else now
        excluded = set(exclude_session_ids)
        blocked_migrations = set(blocked_migration_session_ids)
        with self._lock:
            self._advance_to(observed_at)
            all_ready = self._ready_jobs()
            ready = tuple(item for item in all_ready if item[0].session_id not in excluded)
            if gpu_states is not None:
                gpus = tuple(gpu_states)
            else:
                gpus = tuple(self._gpus.values())
            candidates: list[DispatchCandidate] = []
            enumerated = empty_batch_counts()
            compatible = empty_batch_counts()
            profiles_evaluated = empty_batch_counts()
            feasible = empty_batch_counts()
            rejected: dict[str, int] = {}

            def reject(reason: str) -> None:
                rejected[reason] = rejected.get(reason, 0) + 1

            if ready:
                for gpu in gpus:
                    if not gpu.available:
                        reject("gpu_unavailable")
                        continue
                    for size in range(1, min(self.config.max_batch_size, len(ready)) + 1):
                        for members in itertools.combinations(ready, size):
                            enumerated[size] += 1
                            states = tuple(item[0] for item in members)
                            jobs = tuple(item[1] for item in members)
                            keys = {state.compatibility_key for state in states}
                            if len(keys) != 1:
                                reject("incompatible")
                                continue
                            compatible[size] += 1
                            profiles = tuple(self.profile_provider.profiles_for(
                                batch_size=size,
                                gpu_id=gpu.gpu_id,
                            ))
                            if not profiles:
                                reject("no_profile")
                                continue
                            profiles_evaluated[size] += len(profiles)
                            for profile in profiles:
                                if profile.memory_gb > gpu.memory_free_gb + EPSILON:
                                    reject("memory")
                                    continue
                                estimates = tuple(
                                    self.migration_estimator.estimate(
                                        state,
                                        target_gpu=gpu.gpu_id,
                                        now=observed_at,
                                    )
                                    for state in states
                                )
                                if any(
                                    estimate.required
                                    and (not self.config.migration_enabled or not allow_migrations)
                                    for estimate in estimates
                                ):
                                    reject("migration_disabled")
                                    continue
                                if any(
                                    estimate.required and state.session_id in blocked_migrations
                                    for state, estimate in zip(states, estimates)
                                ):
                                    reject("migration_policy")
                                    continue
                                migration_count = sum(estimate.required for estimate in estimates)
                                migration_seconds = sum(
                                    estimate.cost_seconds for estimate in estimates if estimate.required
                                )
                                start_at = max(
                                    observed_at,
                                    gpu.free_at,
                                    *(estimate.ready_at for estimate in estimates),
                                )
                                finish_at = start_at + profile.latency_seconds
                                duration = finish_at - observed_at
                                projected_slack = {
                                    state.session_id: state.slack_seconds
                                    - (duration if state.playback_active else 0.0)
                                    for state in self._sessions.values()
                                    if not state.departed
                                }
                                projected_quality = {
                                    state.session_id: state.quality_ema
                                    for state in self._sessions.values()
                                    if not state.departed
                                }
                                for state in states:
                                    projected_quality[state.session_id] = (
                                        (1.0 - state.quality_update_rate) * state.quality_ema
                                        + state.quality_update_rate * profile.quality
                                    )
                                    projected_slack[state.session_id] += profile.output_seconds
                                selected_quality = [projected_quality[state.session_id] for state in states]
                                system_quality = (
                                    sum(projected_quality.values()) / len(projected_quality)
                                    if projected_quality
                                    else 0.0
                                )
                                if any(
                                    value < system_quality - self.config.fairness_delta - EPSILON
                                    for value in selected_quality
                                ):
                                    reject("fairness")
                                    continue
                                feasible[size] += 1
                                score = sum(
                                    self._utility(value, self.config.utility_cap_seconds)
                                    for value in projected_slack.values()
                                )
                                score += self.config.lambda_quality * sum(selected_quality)
                                # Penalize both transfer count and predicted transfer time.
                                score -= self.config.lambda_migration * (migration_count + migration_seconds)
                                candidates.append(
                                    DispatchCandidate(
                                        session_ids=tuple(state.session_id for state in states),
                                        job_ids=tuple(job.job_id for job in jobs),
                                        gpu_id=gpu.gpu_id,
                                        fidelity=profile.fidelity,
                                        profile=profile,
                                        start_at=start_at,
                                        finish_at=finish_at,
                                        migration_count=migration_count,
                                        migration_seconds=migration_seconds,
                                        score=score,
                                        projected_slack=projected_slack,
                                        projected_quality=projected_quality,
                                        snapshot_epoch=self._epoch,
                                        session_versions={
                                            state.session_id: state.state_version for state in states
                                        },
                                        gpu_version=gpu.version,
                                    )
                                )
            if include_wait:
                candidates.append(self._wait_candidate(now=observed_at, wait_seconds=wait_seconds))
            selected = (
                max(
                    candidates,
                    key=lambda candidate: (
                        candidate.score,
                        candidate.action_count,
                        candidate.batch_size,
                        -candidate.finish_at,
                        -candidate.migration_count,
                    ),
                )
                if candidates
                else None
            )
            not_selected = dict(feasible)
            if selected is not None and not selected.wait:
                not_selected[selected.batch_size] = max(
                    0, not_selected[selected.batch_size] - 1
                )
            selected_batch_size = selected.batch_size if selected is not None else 0
            summary = MotivationSearchSummary(
                observed_at=observed_at,
                snapshot_epoch=self._epoch,
                ready_count=len(ready),
                ready_action_count=sum(1 for _, job in ready if job.kind == "action"),
                ready_idle_count=sum(1 for _, job in ready if job.kind == "idle"),
                excluded_ready_count=len(all_ready) - len(ready),
                gpu_count=len(gpus),
                include_wait=include_wait,
                allow_migrations=allow_migrations,
                wait_seconds=max(0.0, wait_seconds),
                enumerated_by_batch_size=enumerated,
                compatible_by_batch_size=compatible,
                profiles_evaluated_by_batch_size=profiles_evaluated,
                feasible_by_batch_size=feasible,
                rejected_by_reason=rejected,
                not_selected_by_score=not_selected,
                selected_batch_size=selected_batch_size,
                selected_wait=bool(selected.wait) if selected is not None else False,
                selected_gpu_id=selected.gpu_id if selected is not None else None,
                selected_fidelity=selected.fidelity if selected is not None else None,
                selected_score=selected.score if selected is not None else None,
                selected_migration_count=(selected.migration_count if selected is not None else 0),
                selected_session_ids=selected.session_ids if selected is not None else (),
                selected_job_ids=selected.job_ids if selected is not None else (),
            )
            try:
                self._diagnostics.record_search(summary)
            except Exception:
                # Diagnostics must never affect policy availability.
                pass
            return selected

    def record_dispatch_diagnostics(self, summary: MotivationDispatchSummary) -> None:
        """Publish a dispatch outcome without coupling policy to a logger."""
        try:
            self._diagnostics.record_dispatch(summary)
        except Exception:
            # A telemetry sink must never break reservation or worker dispatch.
            return

    def diagnostics_snapshot(self) -> dict[str, object]:
        """Return the injected diagnostics sink's bounded snapshot."""
        try:
            return self._diagnostics.snapshot()
        except Exception:
            return {}

    def validate(self, candidate: DispatchCandidate, *, now: float | None = None) -> bool:
        """Check that an asynchronously searched candidate is still current."""
        observed_at = self._clock() if now is None else now
        with self._lock:
            self._advance_to(observed_at)
            if candidate.wait:
                return candidate.snapshot_epoch == self._epoch
            if candidate.gpu_id is None or candidate.profile is None:
                return False
            # A new ready job can change the globally optimal batch even when
            # none of the candidate's selected sessions changed.
            if candidate.snapshot_epoch != self._epoch:
                return False
            gpu = self._gpus.get(candidate.gpu_id)
            if gpu is None or not gpu.available or gpu.version != candidate.gpu_version:
                return False
            for session_id, job_id, version in zip(
                candidate.session_ids,
                candidate.job_ids,
                (candidate.session_versions[sid] for sid in candidate.session_ids),
            ):
                state = self._sessions.get(session_id)
                if state is None or state.departed or state.state_version != version:
                    return False
                job = state.ready_job(include_idle=True)
                if job is None or job.job_id != job_id or state.in_flight is not None:
                    return False
            return True

    def candidate_ready_now(self, candidate: DispatchCandidate, *, now: float | None = None) -> bool:
        """Return whether a current candidate can start on its GPU immediately.

        ``validate`` deliberately checks only snapshot freshness, because a
        future candidate can still be useful for planning or asynchronous
        migration.  This helper adds the live GPU timeline check needed by a
        model dispatch path; callers should validate the candidate separately.
        """
        observed_at = self._clock() if now is None else now
        with self._lock:
            self._advance_to(observed_at)
            if (
                candidate.wait
                or candidate.gpu_id is None
                or not math.isfinite(candidate.start_at)
                or candidate.start_at > observed_at + EPSILON
            ):
                return False
            gpu = self._gpus.get(candidate.gpu_id)
            return gpu is not None and gpu.available and gpu.free_at <= observed_at + EPSILON

    def reserve(self, candidate: DispatchCandidate, *, now: float | None = None) -> DispatchCandidate:
        """Atomically reserve a candidate and move its jobs in-flight."""
        observed_at = self._clock() if now is None else now
        with self._lock:
            self._advance_to(observed_at)
            if candidate.wait:
                return candidate
            if not self.validate(candidate, now=observed_at):
                raise RuntimeError("stale motivation scheduling candidate")
            # ``find_best`` includes the predicted GPU timeline in
            # ``candidate.start_at`` so the policy can compare a future slot
            # against an immediately executable one.  That prediction is not
            # a reservation, however: only a worker-side completion event can
            # make a busy GPU available.  Keep this final guard in the
            # scheduler as a defence for callers that bypass the runtime
            # controller (and for races between search and dispatch).
            if not self.candidate_ready_now(candidate, now=observed_at):
                raise RuntimeError("motivation scheduling candidate is not ready")
            assert candidate.gpu_id is not None
            for session_id, job_id in zip(candidate.session_ids, candidate.job_ids):
                state = self._sessions[session_id]
                job = state.ready_job(include_idle=True)
                assert job is not None and job.job_id == job_id
                state.mark_dispatched(job)
                if state.owner_gpu != candidate.gpu_id:
                    state.migration_target_gpu = candidate.gpu_id
            gpu = self._gpus[candidate.gpu_id]
            self._gpus[candidate.gpu_id] = GpuSchedulingState(
                gpu_id=gpu.gpu_id,
                free_at=candidate.finish_at,
                memory_free_gb=max(0.0, gpu.memory_free_gb - candidate.profile.memory_gb),
                available=gpu.available,
                version=gpu.version + 1,
            )
            self._epoch += 1
            return candidate

    def rollback_reservation(
        self,
        candidate: DispatchCandidate,
        *,
        now: float | None = None,
    ) -> tuple[ActionJob, ...]:
        """Release a reservation whose physical dispatch failed synchronously."""
        if candidate.wait or candidate.profile is None or candidate.gpu_id is None:
            return ()
        observed_at = self._clock() if now is None else now
        with self._lock:
            self._advance_to(observed_at)
            jobs: list[ActionJob] = []
            for session_id, job_id in zip(candidate.session_ids, candidate.job_ids, strict=True):
                state = self._sessions[session_id]
                job = state.in_flight
                if job is None or job.job_id != job_id:
                    raise RuntimeError(f"motivation reservation for {job_id} is no longer active")
                state.rollback_dispatched(job)
                state.migration_target_gpu = None
                jobs.append(job)
            gpu = self._gpus[candidate.gpu_id]
            self._gpus[candidate.gpu_id] = GpuSchedulingState(
                gpu_id=gpu.gpu_id,
                free_at=observed_at,
                memory_free_gb=gpu.memory_free_gb + candidate.profile.memory_gb,
                available=gpu.available,
                version=gpu.version + 1,
            )
            self._epoch += 1
            return tuple(jobs)

    def complete(
        self,
        candidate: DispatchCandidate,
        *,
        completed_at: float | None = None,
        quality: float | None = None,
    ) -> tuple[ActionJob, ...]:
        """Commit a completed batch and release its GPU reservation."""
        if candidate.wait or candidate.profile is None or candidate.gpu_id is None:
            return ()
        observed_at = self._clock() if completed_at is None else completed_at
        with self._lock:
            self._advance_to(observed_at)
            profile = candidate.profile
            jobs: list[ActionJob] = []
            for session_id in candidate.session_ids:
                state = self._sessions[session_id]
                job = state.complete_job(
                    completed_at=observed_at,
                    output_seconds=profile.output_seconds,
                    quality=profile.quality if quality is None else quality,
                )
                state.owner_gpu = candidate.gpu_id
                state.migration_target_gpu = None
                state.migration_ready_at = observed_at
                jobs.append(job)
            gpu = self._gpus[candidate.gpu_id]
            self._gpus[candidate.gpu_id] = GpuSchedulingState(
                gpu_id=gpu.gpu_id,
                free_at=observed_at,
                memory_free_gb=gpu.memory_free_gb + profile.memory_gb,
                available=gpu.available,
                version=gpu.version + 1,
            )
            self._epoch += 1
            return tuple(jobs)

    def session(self, session_id: str) -> SessionSchedulingState:
        """Return a live session state for runtime adapters and telemetry."""
        with self._lock:
            return self._sessions[session_id]

    def sessions(self) -> tuple[SessionSchedulingState, ...]:
        """Return a stable tuple of live session state references."""
        with self._lock:
            return tuple(self._sessions.values())

    def gpus(self) -> tuple[GpuSchedulingState, ...]:
        """Return the latest GPU snapshot."""
        with self._lock:
            return tuple(self._gpus.values())
