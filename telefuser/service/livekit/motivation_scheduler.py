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

import bisect
import csv
import itertools
import math
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Protocol

from .motivation_diagnostics import (
    MotivationDiagnosticsSink,
    MotivationDispatchSummary,
    MotivationSearchSummary,
    NullMotivationDiagnostics,
    empty_batch_counts,
)
from .motivation_policies import (
    SchedulingPolicy,
    SchedulingSearchRequest,
    create_scheduling_policy,
)
from .profile_quality import (
    DEFAULT_QUALITY_REFERENCE_FIDELITY,
    normalize_profile_qualities,
    profile_batch_family,
)

EPSILON = 1e-9
JobKind = Literal["action", "idle"]


@dataclass(frozen=True)
class MotivationProfile:
    """Measured execution point for one batch size and fidelity.

    ``gpu_id=None`` denotes a profile shared by homogeneous GPU replicas.
    A GPU-specific row takes precedence over a shared row.  ``quality`` is the
    normalized semantic Q factor used by the policy; ``raw_quality`` retains
    the offline evaluator value when the row came from a CSV table.  The
    runtime does not run a visual evaluator in the dispatch critical path.
    """

    batch_size: int
    fidelity: str
    latency_seconds: float
    quality: float
    memory_gb: float
    output_seconds: float = 1.0
    p95_seconds: float | None = None
    gpu_id: str | None = None
    # Keep the evaluator's raw value alongside the normalized policy value for
    # diagnostics/audit.  Existing positional constructors remain compatible.
    raw_quality: float | None = None

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.batch_size > 4:
            raise ValueError("batch_size must be in [1, 4]")
        if not self.fidelity:
            raise ValueError("fidelity must be non-empty")
        if not math.isfinite(self.latency_seconds) or self.latency_seconds <= 0:
            raise ValueError("latency_seconds must be positive and finite")
        if not math.isfinite(self.quality) or self.quality <= 0:
            raise ValueError("quality must be positive and finite")
        if self.raw_quality is not None and (
            not math.isfinite(self.raw_quality) or self.raw_quality <= 0
        ):
            raise ValueError("raw_quality must be positive and finite when supplied")
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
    normalize_quality: bool = True,
    quality_reference_fidelity: str = DEFAULT_QUALITY_REFERENCE_FIDELITY,
    batch_invariant_quality: bool = True,
) -> StaticMotivationProfileTable:
    """Load the measured offline-table rows used by the policy.

    The loader accepts the ABot profile schema directly.  ``Q_world`` is the
    preferred quality column; if it is empty, the mean of available
    ``Q_action``, ``Q_temporal`` and ``Q_visual`` values is used.  By default,
    values are exposed as normalized ``Q`` factors relative to the explicit
    B1/S4/W18 reference, and all batch sizes in one S/W family share the B1
    quality.  Raw evaluator values remain on ``MotivationProfile.raw_quality``.
    A reduced custom table without an S4/W18 reference transparently falls
    back to raw values (and records that fact on the returned table).  The rows
    are tagged as homogeneous (or with ``gpu_id`` when supplied), so
    GPU-specific measurements can coexist with shared fallback rows.
    """
    if not 1 <= max_batch_size <= 4:
        raise ValueError("max_batch_size must be in [1, 4]")
    if output_seconds <= 0 or not math.isfinite(output_seconds):
        raise ValueError("output_seconds must be positive and finite")
    parsed_rows: list[tuple[int, str, float, float, float, float, float]] = []
    raw_quality_by_config: dict[str, float] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            batch_size = int(raw["B"])
            if batch_size > max_batch_size:
                continue
            latency_ms = float(raw["latency_ms"])
            p95_raw = raw.get("latency_p95_ms", "")
            p95_ms = float(p95_raw) if p95_raw not in {None, ""} else latency_ms
            memory_gb = float(raw["memory_GB"])
            fidelity = str(raw.get("config") or f"b{batch_size}")
            quality_raw = raw.get("Q_world", "")
            if quality_raw in {None, ""}:
                quality_values = [
                    float(raw[key])
                    for key in ("Q_action", "Q_temporal", "Q_visual")
                    if raw.get(key, "") not in {None, ""}
                ]
                if not quality_values:
                    raise ValueError(f"profile row {raw.get('config', '<unknown>')} has no quality value")
                raw_quality = sum(quality_values) / len(quality_values)
            else:
                raw_quality = float(quality_raw)
            if fidelity in raw_quality_by_config:
                raise ValueError(f"duplicate profile row: {fidelity!r}")
            raw_quality_by_config[fidelity] = raw_quality
            parsed_rows.append(
                (
                    batch_size,
                    fidelity,
                    latency_ms,
                    p95_ms,
                    raw_quality,
                    memory_gb,
                    output_seconds,
                )
            )
    if not parsed_rows:
        raise ValueError(f"no profiles with batch size <= {max_batch_size} found in {path}")

    if normalize_quality:
        quality_result = normalize_profile_qualities(
            raw_quality_by_config,
            reference_fidelity=quality_reference_fidelity,
            batch_invariant=batch_invariant_quality,
        )
    else:
        quality_result = normalize_profile_qualities(
            raw_quality_by_config,
            reference_fidelity=quality_reference_fidelity,
            batch_invariant=False,
        )
        # Explicitly preserve the old/raw semantics when normalization is
        # disabled, including the metadata flag used by reports.
        quality_result = quality_result.__class__(
            values=dict(raw_quality_by_config),
            reference_config=quality_result.reference_config,
            reference_raw=quality_result.reference_raw,
            normalized=False,
            batch_invariant=False,
        )

    rows: list[MotivationProfile] = []
    for batch_size, fidelity, latency_ms, p95_ms, raw_quality, memory_gb, output_seconds in parsed_rows:
        rows.append(
            MotivationProfile(
                batch_size=batch_size,
                fidelity=fidelity,
                latency_seconds=latency_ms / 1000.0,
                p95_seconds=p95_ms / 1000.0,
                quality=quality_result.values[fidelity],
                memory_gb=memory_gb,
                output_seconds=output_seconds,
                gpu_id=gpu_id,
                raw_quality=raw_quality,
            )
        )

    # Several historical ABot profile captures measured B1/B2/B4/B8 but
    # omitted B3. Treat that omission as a profile-data gap rather than
    # rejecting every three-session candidate. Interpolate the neighboring
    # measured points with a conservative quality value; physical execution
    # still validates the resulting batch and trace records the actual size.
    if max_batch_size >= 3:
        by_suffix: dict[str, dict[int, MotivationProfile]] = {}
        for row in rows:
            _, suffix = profile_batch_family(row.fidelity)
            by_suffix.setdefault(suffix, {})[row.batch_size] = row
        for suffix, neighbors in by_suffix.items():
            lower = neighbors.get(2)
            upper = neighbors.get(4)
            if lower is not None and upper is not None and 3 not in neighbors:
                latency = (lower.latency_seconds + upper.latency_seconds) / 2.0
                p95 = (
                    (lower.p95_seconds or lower.latency_seconds)
                    + (upper.p95_seconds or upper.latency_seconds)
                ) / 2.0
                memory = (lower.memory_gb + upper.memory_gb) / 2.0
                # Quality is a semantic family property.  Once the measured
                # rows have been normalized, batching must not perturb it;
                # use the lower-B row only as a fallback for reduced tables.
                quality = (
                    lower.quality
                    if quality_result.normalized and quality_result.batch_invariant
                    else min(lower.quality, upper.quality)
                )
                raw_quality = (
                    lower.raw_quality
                    if quality_result.normalized and quality_result.batch_invariant
                    else min(
                        value
                        for value in (lower.raw_quality, upper.raw_quality)
                        if value is not None
                    )
                )
                gpu_id_for_row = lower.gpu_id
                rows.append(
                    MotivationProfile(
                        batch_size=3,
                        fidelity=f"b3_{suffix}" if suffix else "b3",
                        latency_seconds=latency,
                        p95_seconds=p95,
                        quality=quality,
                        memory_gb=memory,
                        output_seconds=lower.output_seconds,
                        gpu_id=gpu_id_for_row,
                        raw_quality=raw_quality,
                    )
                )
    return StaticMotivationProfileTable(
        rows,
        quality_reference_config=quality_result.reference_config,
        quality_reference_raw=quality_result.reference_raw,
        quality_normalized=quality_result.normalized,
        batch_invariant_quality=quality_result.batch_invariant,
    )


class MotivationProfileProvider(Protocol):
    """Lookup interface used by the policy search."""

    def profiles_for(self, *, batch_size: int, gpu_id: str) -> Sequence[MotivationProfile]:
        """Return all measured fidelity points feasible for ``gpu_id``."""


class StaticMotivationProfileTable:
    """Immutable-style lookup table backed by measured offline profiles."""

    def __init__(
        self,
        profiles: Iterable[MotivationProfile],
        *,
        quality_reference_config: str | None = None,
        quality_reference_raw: float | None = None,
        quality_normalized: bool = False,
        batch_invariant_quality: bool = False,
    ) -> None:
        rows = tuple(profiles)
        if not rows:
            raise ValueError("at least one profile is required")
        self.quality_reference_config = quality_reference_config
        self.quality_reference_raw = quality_reference_raw
        self.quality_normalized = bool(quality_normalized)
        self.batch_invariant_quality = bool(batch_invariant_quality)
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
    sequence: int = 0


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
        sequence: int = 0,
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
            sequence=sequence,
        )
        return not had_pending

    def create_idle_job(self, *, job_id: str, now: float, sequence: int = 0) -> ActionJob | None:
        """Create one idle sentinel if no action or idle output is pending."""
        self.advance_to(now)
        # Match the paper simulator's per-session head semantics: once the
        # released action job has run, the session may contribute an idle
        # sentinel while it waits for the next heartbeat.  ``submit_action``
        # still drops a not-yet-dispatched sentinel when a fresh action is
        # released, so real action work always wins within the session.
        if self.departed or self.pending_action is not None:
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
            sequence=sequence,
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
    # ``memory_free_gb`` is the free baseline at the current timeline point.
    # A profile's ``memory_gb`` is a peak requirement for one serialized
    # invocation, not a permanent allocation.  Keep the in-flight peak
    # separately so a candidate projected after ``free_at`` can reclaim it;
    # otherwise every future B>1 candidate is incorrectly rejected while the
    # GPU is busy even though the previous invocation has already completed.
    reserved_memory_gb: float = 0.0

    def __post_init__(self) -> None:
        if not self.gpu_id:
            raise ValueError("gpu_id must be non-empty")
        if not math.isfinite(self.free_at) or self.free_at < 0:
            raise ValueError("free_at must be finite and non-negative")
        if self.memory_free_gb != math.inf and (
            not math.isfinite(self.memory_free_gb) or self.memory_free_gb < 0
        ):
            raise ValueError("memory_free_gb must be non-negative and finite")
        if not math.isfinite(self.reserved_memory_gb) or self.reserved_memory_gb < 0:
            raise ValueError("reserved_memory_gb must be non-negative and finite")


@dataclass(frozen=True)
class MigrationEstimate:
    """Predicted state-transfer readiness for one session and target GPU."""

    ready_at: float
    cost_seconds: float = 0.0
    required: bool = False
    first_layer_ready_seconds: float = 0.0
    transfer_seconds: float = 0.0
    drain_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.ready_at) or self.ready_at < 0:
            raise ValueError("ready_at must be finite and non-negative")
        if not math.isfinite(self.cost_seconds) or self.cost_seconds < 0:
            raise ValueError("cost_seconds must be non-negative and finite")
        for name in ("first_layer_ready_seconds", "transfer_seconds", "drain_seconds"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be non-negative and finite")


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
    """Estimate the route-ready critical path, not background transfer time.

    ``migration_cost_seconds`` remains as a compatibility alias for callers
    that only have one prior. New callers should provide the measured
    route-ready duration through the legacy ``first_layer_ready_seconds``
    field and optionally retain wire/drain telemetry. The field name remains
    stable for existing profiles, but its runtime value includes source
    quiesce, metadata export, target preparation, and first-layer DMA. The
    residual transfer and source cleanup are deliberately non-blocking.
    """

    def __init__(
        self,
        *,
        migration_cost_seconds: float = 0.0,
        first_layer_ready_seconds: float | None = None,
        transfer_seconds: float = 0.0,
        drain_seconds: float = 0.0,
    ) -> None:
        values = {
            "migration_cost_seconds": migration_cost_seconds,
            "transfer_seconds": transfer_seconds,
            "drain_seconds": drain_seconds,
        }
        if first_layer_ready_seconds is not None:
            values["first_layer_ready_seconds"] = first_layer_ready_seconds
        for name, value in values.items():
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be non-negative and finite")
        self.migration_cost_seconds = float(migration_cost_seconds)
        self.first_layer_ready_seconds = (
            self.migration_cost_seconds
            if first_layer_ready_seconds is None
            else float(first_layer_ready_seconds)
        )
        self.transfer_seconds = float(transfer_seconds)
        self.drain_seconds = float(drain_seconds)

    def record_phase_timings(
        self,
        *,
        first_layer_ready_seconds: float | None = None,
        transfer_seconds: float | None = None,
        drain_seconds: float | None = None,
    ) -> None:
        """Update online priors without ever learning full E2E drain time."""
        values = {
            "first_layer_ready_seconds": first_layer_ready_seconds,
            "transfer_seconds": transfer_seconds,
            "drain_seconds": drain_seconds,
        }
        for name, value in values.items():
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0):
                raise ValueError(f"{name} must be non-negative and finite")
        if first_layer_ready_seconds is not None:
            self.first_layer_ready_seconds = float(first_layer_ready_seconds)
        if transfer_seconds is not None:
            self.transfer_seconds = float(transfer_seconds)
        if drain_seconds is not None:
            self.drain_seconds = float(drain_seconds)

    def estimate(
        self,
        session: SessionSchedulingState,
        *,
        target_gpu: str,
        now: float,
    ) -> MigrationEstimate:
        if session.owner_gpu == target_gpu:
            return MigrationEstimate(ready_at=now, required=False)
        first_layer_ready = max(0.0, self.first_layer_ready_seconds)
        ready_at = max(now + first_layer_ready, session.migration_ready_at)
        return MigrationEstimate(
            ready_at=ready_at,
            # Scheduler cost ends when the target route is usable. Residual
            # wire completion and source cleanup overlap target compute.
            cost_seconds=max(0.0, ready_at - now),
            required=True,
            first_layer_ready_seconds=first_layer_ready,
            transfer_seconds=max(0.0, self.transfer_seconds),
            drain_seconds=max(0.0, self.drain_seconds),
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
    policy_name: str = "motivation"
    # FIFO is intentionally a fixed-execution baseline.  The paper-facing
    # baseline is pinned to the normalized S4/W18 B1 point; no quality/latency
    # trade-off is made at runtime.  A caller may still override this in a
    # reduced unit-test table, but production entrypoints use this default.
    fifo_fidelity: str | None = "b1_s4_w18_rho0_bf16"

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
        if not isinstance(self.policy_name, str) or not self.policy_name.strip():
            raise ValueError("policy_name must be a non-empty string")
        if self.fifo_fidelity is not None and (
            not isinstance(self.fifo_fidelity, str) or not self.fifo_fidelity.strip()
        ):
            raise ValueError("fifo_fidelity must be a non-empty string or None")


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
    policy_name: str = "motivation"

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
        policy: SchedulingPolicy | None = None,
    ) -> None:
        self.profile_provider = profile_provider
        requested_config = config or MotivationSchedulerConfig()
        self._policy = policy or create_scheduling_policy(requested_config.policy_name)
        # FIFO is a policy-level baseline, not Motivation with a one-session
        # candidate filter.  Normalize the coordinator knobs as a defensive
        # boundary so callers cannot accidentally re-enable idle work, larger
        # batches, or quality/fairness scoring by passing production defaults.
        if self._policy.name == "fifo":
            self.config = replace(
                requested_config,
                max_batch_size=1,
                include_idle_jobs=False,
                lambda_quality=0.0,
                fairness_delta=0.0,
                migration_enabled=False,
            )
        else:
            self.config = requested_config
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

    @property
    def policy_name(self) -> str:
        """Return the registered policy name used for candidate selection."""
        return self._policy.name

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
                reserved_memory_gb=state.reserved_memory_gb,
            )
            self._epoch += 1

    def _coerce_time_locked(self, observed_at: float) -> float:
        """Return a scheduler-monotonic timestamp.

        Runtime callbacks are delivered by several worker/event-loop threads.
        A callback can capture a timestamp, block while a large candidate
        search or an NCCL transfer runs, and only then acquire the scheduler
        lock.  In that case its timestamp is older than ``_now`` even though
        the event is perfectly valid.  Treat it as occurring at the current
        policy instant rather than turning an out-of-order callback into a
        scheduler-wide failure.  The logical timeline remains monotonic and
        callers still get strict validation for non-finite/negative values.
        """
        if not math.isfinite(observed_at) or observed_at < 0:
            raise ValueError("observed time must be finite and non-negative")
        return max(float(observed_at), self._now)

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
                reserved_memory_gb=current.reserved_memory_gb,
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
            observed_at = self._coerce_time_locked(observed_at)
            if session_id in self._sessions:
                raise ValueError(f"session {session_id!r} is already registered")
            if owner_gpu not in self._gpus:
                raise KeyError(f"unknown owner GPU {owner_gpu!r}")
            initial_quality = quality
            if initial_quality is None:
                initial_quality = self.config.initial_quality
            # FIFO does not maintain a quality objective.  Use a neutral
            # bookkeeping value instead of scanning the profile table and
            # selecting its maximum-quality row during session admission.
            # Actual profile quality remains available in the execution trace
            # for post-run reporting.
            if initial_quality is None and self.policy_name == "fifo":
                initial_quality = 1.0
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
            observed_at = self._advance_to(observed_at)
            state = self._sessions[session_id]
            self._job_sequence += 1
            job_id = f"{session_id}:action:{self._job_sequence:08d}"
            before = state.pending_action
            invalidated = state.submit_action(
                job_id=job_id,
                controls=controls,
                now=observed_at,
                release=release,
                sequence=self._job_sequence,
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
            observed_at = self._advance_to(observed_at)
            self._job_sequence += 1
            job = self._sessions[session_id].create_idle_job(
                job_id=f"{session_id}:idle:{self._job_sequence:08d}",
                now=observed_at,
                sequence=self._job_sequence,
            )
            if job is not None:
                self._epoch += 1
            return job

    def set_playback_active(self, session_id: str, active: bool, *, now: float | None = None) -> None:
        """Pause/resume slack consumption for an explicitly paused consumer."""
        observed_at = self._clock() if now is None else now
        with self._lock:
            observed_at = self._advance_to(observed_at)
            state = self._sessions[session_id]
            if state.playback_active != active:
                state.playback_active = active
                self._epoch += 1

    def mark_departed(self, session_id: str, *, now: float | None = None) -> None:
        """Remove future candidates for a departed session."""
        observed_at = self._clock() if now is None else now
        with self._lock:
            observed_at = self._advance_to(observed_at)
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
            observed_at = self._advance_to(observed_at)
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
            observed_at = self._advance_to(observed_at)
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
            observed_at = self._advance_to(observed_at)
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
            observed_at = self._advance_to(observed_at)
            state = self._sessions[session_id]
            updated = tuple(compatibility_key)
            if state.compatibility_key == updated:
                return
            state.compatibility_key = updated
            self._epoch += 1

    def _advance_to(self, now: float) -> float:
        """Advance the logical timeline and return its effective timestamp.

        ``now`` may be stale when an asynchronous callback was delayed behind
        another event.  Coercing it under the scheduler lock keeps all state
        updates ordered without fabricating a backwards jump.
        """
        effective_now = self._coerce_time_locked(now)
        for state in self._sessions.values():
            state.advance_to(effective_now)
        self._now = effective_now
        return effective_now

    def _ready_jobs(self) -> tuple[tuple[SessionSchedulingState, ActionJob], ...]:
        ready: list[tuple[SessionSchedulingState, ActionJob]] = []
        for state in self._sessions.values():
            # A session with an in-flight invocation is intentionally absent.
            # Its pending action remains stored and becomes runnable after the
            # current invocation releases the per-session slot.
            if state.departed or state.in_flight is not None:
                continue
            job = state.ready_job(include_idle=self.config.include_idle_jobs)
            if job is not None:
                ready.append((state, job))
        # Priority is per session, not global: ``ready_job`` returns an action
        # before that session's idle sentinel, while idle heads from other
        # sessions remain available as useful batch fillers. Candidate scoring
        # retains ``action_count`` as the deterministic action-first tie-break.
        return tuple(ready)

    def _find_fifo_candidate(
        self,
        *,
        now: float | None = None,
        gpu_states: Sequence[GpuSchedulingState] | None = None,
        wait_seconds: float = 0.0,
        include_wait: bool = True,
        allow_migrations: bool = True,
        exclude_session_ids: Iterable[str] = (),
        blocked_migration_session_ids: Iterable[str] = (),
        policy_name: str = "fifo",
    ) -> DispatchCandidate | None:
        """Select one oldest action with a fixed B1 execution point.

        This is intentionally a separate, linear policy path.  It never
        invokes :meth:`_find_best_motivation`, never creates an idle candidate,
        and never compares profile quality or scores.  The profile table is
        consulted only to obtain the one fixed execution row required by the
        physical worker (latency/memory/fidelity); the common scheduler still
        performs snapshot validation and reservation.
        """
        # These are Motivation-only inputs.  FIFO has a fixed singleton and a
        # static owner, so it never evaluates a wait, migration estimate, or
        # residence-cooldown rule.
        del include_wait, allow_migrations, blocked_migration_session_ids
        observed_at = self._clock() if now is None else now
        excluded = {str(value) for value in exclude_session_ids}

        # Take the same immutable snapshot used by the Motivation search.  A
        # candidate found outside the lock is still rejected by epoch/version
        # validation if a concurrent event changes the ready queue.
        with self._lock:
            observed_at = self._advance_to(observed_at)
            live_states = tuple(self._sessions.values())
            snapshot_states = tuple(replace(state) for state in live_states)
            snapshot_by_id = {state.session_id: state for state in snapshot_states}
            all_ready = tuple(
                (snapshot_by_id[state.session_id], job)
                for state, job in self._ready_jobs()
                if job.kind == "action"
            )
            ready = tuple(
                item for item in all_ready if item[0].session_id not in excluded
            )
            gpus = tuple(gpu_states) if gpu_states is not None else tuple(self._gpus.values())
            snapshot_epoch = self._epoch

        ordered_ready = tuple(
            sorted(
                ready,
                key=lambda item: (
                    item[1].sequence,
                    item[1].created_at,
                    item[1].job_id,
                ),
            )
        )

        # Look up one fixed row per owner GPU.  A reduced test/profile table
        # may contain only one B1 row; accepting that sole row keeps tests
        # useful without turning a missing fixed point into a quality choice.
        profile_cache: dict[str, MotivationProfile | None] = {}
        fallback_gpus: set[str] = set()

        def fixed_profile(gpu: GpuSchedulingState) -> MotivationProfile | None:
            if gpu.gpu_id in profile_cache:
                return profile_cache[gpu.gpu_id]
            profiles = tuple(
                self.profile_provider.profiles_for(batch_size=1, gpu_id=gpu.gpu_id)
            )
            if not profiles:
                profile_cache[gpu.gpu_id] = None
                return None
            requested = self.config.fifo_fidelity
            profile = next(
                (row for row in profiles if row.fidelity == requested),
                None,
            ) if requested is not None else None
            if profile is None and len(profiles) == 1:
                # A reduced test/profile table may contain only one B1 row.
                # Accept that sole row, but never silently choose among
                # multiple quality points when the requested fixed row is
                # absent.
                profile = profiles[0]
                fallback_gpus.add(gpu.gpu_id)
            profile_cache[gpu.gpu_id] = profile
            return profile

        enumerated = empty_batch_counts()
        compatible = empty_batch_counts()
        profiles_evaluated = empty_batch_counts()
        feasible = empty_batch_counts()
        rejected: dict[str, int] = {}

        def reject(reason: str, count: int = 1) -> None:
            rejected[reason] = rejected.get(reason, 0) + count

        best: tuple[
            tuple[float, int, str],
            SessionSchedulingState,
            ActionJob,
            GpuSchedulingState,
            MotivationProfile,
            float,
            float,
        ] | None = None

        # FIFO order is global over currently runnable actions.  A session is
        # permanently tied to its admitted owner GPU; if that owner is busy,
        # the candidate is projected for that GPU and the controller may look
        # for a newer action on another *already free* owner in its fallback
        # pass.  No remote placement or state transfer is attempted.
        gpu_by_id = {gpu.gpu_id: gpu for gpu in gpus}
        for state, job in ordered_ready:
            gpu = gpu_by_id.get(state.owner_gpu)
            if gpu is None:
                reject("owner_gpu_missing")
                continue
            enumerated[1] += 1
            compatible[1] += 1  # every singleton is structurally compatible
            if not gpu.available:
                reject("gpu_unavailable")
                continue
            profile = fixed_profile(gpu)
            profiles_evaluated[1] = len(profile_cache)
            if profile is None:
                reject("fifo_fidelity_missing")
                continue
            effective_memory_free_gb = gpu.memory_free_gb
            if (
                gpu.free_at > observed_at + EPSILON
                and gpu.memory_free_gb != math.inf
            ):
                effective_memory_free_gb += gpu.reserved_memory_gb
            if profile.memory_gb > effective_memory_free_gb + EPSILON:
                reject("memory")
                continue
            start_at = max(observed_at, gpu.free_at)
            finish_at = start_at + profile.latency_seconds
            target_key = (start_at, 0, gpu.gpu_id)
            feasible[1] += 1
            best = (
                target_key,
                state,
                job,
                gpu,
                profile,
                start_at,
                finish_at,
            )
            break

        selected: DispatchCandidate | None = None
        if best is not None:
            (
                _target_key,
                selected_state,
                selected_job,
                selected_gpu,
                selected_profile,
                selected_start,
                selected_finish,
            ) = best
            selected_gpu_id = selected_gpu.gpu_id
            selected = DispatchCandidate(
                session_ids=(selected_state.session_id,),
                job_ids=(selected_job.job_id,),
                gpu_id=selected_gpu_id,
                fidelity=selected_profile.fidelity,
                profile=selected_profile,
                start_at=selected_start,
                finish_at=selected_finish,
                migration_count=0,
                migration_seconds=0.0,
                # FIFO has no utility/quality score.  Keep the field at zero
                # so downstream telemetry cannot be mistaken for optimization.
                score=0.0,
                # These fields belong to Motivation's objective projection;
                # FIFO intentionally leaves them empty rather than computing
                # slack or quality deltas that it never uses.
                projected_slack={},
                projected_quality={},
                snapshot_epoch=snapshot_epoch,
                session_versions={selected_state.session_id: selected_state.state_version},
                gpu_version=selected_gpu.version,
                policy_name=policy_name,
            )

        if fallback_gpus:
            rejected["fifo_fidelity_fallback"] = len(fallback_gpus)
        not_selected = empty_batch_counts()
        if feasible[1] and selected is not None:
            not_selected[1] = max(0, feasible[1] - 1)
        summary = MotivationSearchSummary(
            observed_at=observed_at,
            snapshot_epoch=snapshot_epoch,
            ready_count=len(ready),
            ready_action_count=len(ready),
            ready_idle_count=0,
            excluded_ready_count=len(all_ready) - len(ready),
            gpu_count=len(gpus),
            include_wait=False,
            allow_migrations=False,
            wait_seconds=max(0.0, wait_seconds),
            enumerated_by_batch_size=enumerated,
            compatible_by_batch_size=compatible,
            profiles_evaluated_by_batch_size=profiles_evaluated,
            feasible_by_batch_size=feasible,
            rejected_by_reason=rejected,
            not_selected_by_score=not_selected,
            selected_batch_size=selected.batch_size if selected is not None else 0,
            selected_wait=False,
            selected_gpu_id=selected.gpu_id if selected is not None else None,
            selected_fidelity=selected.fidelity if selected is not None else None,
            selected_score=0.0 if selected is not None else None,
            selected_migration_count=(selected.migration_count if selected is not None else 0),
            selected_session_ids=selected.session_ids if selected is not None else (),
            selected_job_ids=selected.job_ids if selected is not None else (),
            policy_name=policy_name,
        )
        try:
            self._diagnostics.record_search(summary)
        except Exception:
            # Diagnostics must never affect policy availability.
            pass
        return selected

    @staticmethod
    def _utility(slack: float, cap: float) -> float:
        """The simulator's ``U(P)=min(P, cap)`` utility."""
        return min(slack, cap)

    def _wait_candidate(
        self,
        *,
        now: float,
        wait_seconds: float,
        states: Sequence[SessionSchedulingState] | None = None,
        snapshot_epoch: int | None = None,
        policy_name: str = "motivation",
    ) -> DispatchCandidate:
        """Build a wait candidate from either live state or a search snapshot."""
        source = tuple(self._sessions.values()) if states is None else tuple(states)
        active_states = tuple(state for state in source if not state.departed)
        wait = max(0.0, wait_seconds)
        projected_slack = {
            state.session_id: state.slack_seconds - (wait if state.playback_active else 0.0)
            for state in active_states
        }
        score = sum(self._utility(value, self.config.utility_cap_seconds) for value in projected_slack.values())
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
            projected_quality={state.session_id: state.quality_ema for state in active_states},
            snapshot_epoch=self._epoch if snapshot_epoch is None else snapshot_epoch,
            session_versions={state.session_id: state.state_version for state in active_states},
            gpu_version=None,
            wait=True,
            policy_name=policy_name,
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
        """Select a candidate through the configured policy strategy."""
        request = SchedulingSearchRequest(
            now=now,
            gpu_states=None if gpu_states is None else tuple(gpu_states),
            wait_seconds=float(wait_seconds),
            include_wait=bool(include_wait),
            allow_migrations=bool(allow_migrations),
            exclude_session_ids=tuple(str(value) for value in exclude_session_ids),
            blocked_migration_session_ids=tuple(
                str(value) for value in blocked_migration_session_ids
            ),
        )
        return self._policy.select(self, request)

    def _find_best_motivation(
        self,
        *,
        now: float | None = None,
        gpu_states: Sequence[GpuSchedulingState] | None = None,
        wait_seconds: float = 0.0,
        include_wait: bool = True,
        allow_migrations: bool = True,
        exclude_session_ids: Iterable[str] = (),
        blocked_migration_session_ids: Iterable[str] = (),
        policy_name: str = "motivation",
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

        # Advance and copy the policy state while holding the lock, then do the
        # expensive Cartesian-product search without it.  Event callbacks can
        # therefore register sessions and publish completions while a large
        # ready set is being scored; ``snapshot_epoch`` makes the result fail
        # closed at reservation time if anything changed meanwhile.
        with self._lock:
            observed_at = self._advance_to(observed_at)
            live_states = tuple(self._sessions.values())
            snapshot_states = tuple(replace(state) for state in live_states)
            snapshot_by_id = {state.session_id: state for state in snapshot_states}
            all_ready = tuple(
                (snapshot_by_id[state.session_id], job)
                for state, job in self._ready_jobs()
            )
            ready = tuple(item for item in all_ready if item[0].session_id not in excluded)
            gpus = tuple(gpu_states) if gpu_states is not None else tuple(self._gpus.values())
            snapshot_epoch = self._epoch

        active_states = tuple(state for state in snapshot_states if not state.departed)
        # Keep the hot loop on compact integer indexes.  Constructing state/job
        # tuples and projected dictionaries for every losing candidate was the
        # dominant cost once a trace had 20+ ready sessions.  The original
        # combinations order is retained so exact-score ties remain stable.
        ready_states = tuple(state for state, _ in ready)
        ready_jobs = tuple(job for _, job in ready)
        ready_keys = tuple(state.compatibility_key for state in ready_states)
        compatibility_groups: tuple[tuple[int, ...], ...] = tuple(
            tuple(index for index, key in enumerate(ready_keys) if key == group_key)
            for group_key in dict.fromkeys(ready_keys)
        )
        compatible_combinations: dict[int, tuple[tuple[int, ...], ...]] = {}
        for size in range(1, min(self.config.max_batch_size, len(ready)) + 1):
            compatible_combinations[size] = tuple(
                indexes
                for indexes in itertools.combinations(range(len(ready)), size)
                if all(ready_keys[index] == ready_keys[indexes[0]] for index in indexes[1:])
            )

        # ``best_record`` is a lightweight descriptor.  Full candidate maps
        # are materialized once, after the exhaustive score search has ended.
        best_record: tuple[
            tuple[float, int, int, float, int],
            tuple[int, ...],
            str,
            MotivationProfile,
            float,
            float,
            int,
            float,
        ] | None = None
        candidates_best_key: tuple[float, int, int, float, int] | None = None
        enumerated = empty_batch_counts()
        compatible = empty_batch_counts()
        profiles_evaluated = empty_batch_counts()
        feasible = empty_batch_counts()
        rejected: dict[str, int] = {}

        def reject(reason: str, count: int = 1) -> None:
            rejected[reason] = rejected.get(reason, 0) + count

        # Cache profile and migration lookups once per search.  Both are
        # independent of the selected member combination, while the old loop
        # repeated them for every profile and combination respectively.
        profile_cache: dict[tuple[str, int], tuple[MotivationProfile, ...]] = {}

        def profiles_for(gpu: GpuSchedulingState, size: int) -> tuple[MotivationProfile, ...]:
            cache_key = (gpu.gpu_id, size)
            profiles = profile_cache.get(cache_key)
            if profiles is None:
                profiles = tuple(self.profile_provider.profiles_for(batch_size=size, gpu_id=gpu.gpu_id))
                profile_cache[cache_key] = profiles
            return profiles

        # Baseline terms are common to every profile at a given predicted
        # duration.  Keep a sorted prefix representation for the normal
        # playback-active sessions.  The old hot loop scanned every active
        # session for every (profile, member-set) pair; at 24 sessions that
        # made the control-plane search compete with the GPU for ~100 ms per
        # call.  ``min(slack-duration, cap)`` is piecewise linear, so a
        # bisect/prefix lookup is exact (paused sessions remain a small
        # constant term).
        slack_cache: dict[float, float] = {}
        utility_cap = self.config.utility_cap_seconds
        lambda_quality = self.config.lambda_quality
        lambda_migration = self.config.lambda_migration
        fairness_delta = self.config.fairness_delta
        active_quality_sum = sum(state.quality_ema for state in active_states)
        active_slacks = tuple(state.slack_seconds for state in active_states)
        active_playback = tuple(state.playback_active for state in active_states)
        playback_slacks = sorted(
            slack for slack, playback in zip(active_slacks, active_playback, strict=True) if playback
        )
        playback_slack_prefix = [0.0]
        for slack in playback_slacks:
            playback_slack_prefix.append(playback_slack_prefix[-1] + slack)
        paused_utility = sum(
            slack if slack < utility_cap else utility_cap
            for slack, playback in zip(active_slacks, active_playback, strict=True)
            if not playback
        )

        def baseline_utility(duration: float) -> float:
            """Return exact system utility after ``duration`` seconds."""
            threshold = utility_cap + duration
            split = bisect.bisect_left(playback_slacks, threshold)
            return (
                playback_slack_prefix[split]
                - split * duration
                + (len(playback_slacks) - split) * utility_cap
                + paused_utility
            )

        active_index = {state.session_id: index for index, state in enumerate(active_states)}
        ready_active_index = tuple(active_index[state.session_id] for state in ready_states)
        ready_quality = tuple(state.quality_ema for state in ready_states)
        ready_quality_rate = tuple(state.quality_update_rate for state in ready_states)
        ready_action_flags = tuple(1 if job.kind == "action" else 0 for job in ready_jobs)

        # Migration estimates do not depend on batch size or fidelity.  Keep a
        # per-GPU vector and derive the small combination metadata once per
        # compatible member set instead of repeating reductions for every
        # profile row.
        estimates_by_gpu: dict[str, tuple[MigrationEstimate, ...]] = {}

        def estimates_for_gpu(
            gpu: GpuSchedulingState,
        ) -> tuple[MigrationEstimate, ...]:
            estimates = estimates_by_gpu.get(gpu.gpu_id)
            if estimates is None:
                estimates = tuple(
                    self.migration_estimator.estimate(
                        state,
                        target_gpu=gpu.gpu_id,
                        now=observed_at,
                    )
                    for state in ready_states
                )
                estimates_by_gpu[gpu.gpu_id] = estimates
            return estimates

        if ready:
            max_size = min(self.config.max_batch_size, len(ready))
            for gpu in gpus:
                if not gpu.available:
                    reject("gpu_unavailable")
                    continue
                gpu_estimates: tuple[MigrationEstimate, ...] | None = None
                for size in range(1, max_size + 1):
                    total_combinations = math.comb(len(ready), size)
                    enumerated[size] += total_combinations
                    combinations = compatible_combinations[size]
                    compatible[size] += len(combinations)
                    if total_combinations > len(combinations):
                        reject("incompatible", total_combinations - len(combinations))
                    profiles = profiles_for(gpu, size)
                    if not profiles:
                        for _ in combinations:
                            reject("no_profile")
                        continue
                    profiles_evaluated[size] += len(profiles) * len(combinations)
                    # A GPU has at most one physical invocation in flight.
                    # When this candidate is projected after that invocation's
                    # ``free_at`` boundary, the current peak reservation is
                    # released before the candidate starts.  The simulator
                    # applies the same serialized peak-memory constraint; use
                    # the reclaimed value for future slots instead of treating
                    # the previous batch's peak as a permanent allocation.
                    effective_memory_free_gb = gpu.memory_free_gb
                    if (
                        gpu.free_at > observed_at + EPSILON
                        and gpu.memory_free_gb != math.inf
                    ):
                        effective_memory_free_gb += gpu.reserved_memory_gb
                    usable_profiles = tuple(
                        profile
                        for profile in profiles
                        if profile.memory_gb <= effective_memory_free_gb + EPSILON
                    )
                    memory_rejected = len(profiles) - len(usable_profiles)
                    if memory_rejected:
                        reject("memory", memory_rejected * len(combinations))
                    if not usable_profiles:
                        continue
                    if gpu_estimates is None:
                        gpu_estimates = estimates_for_gpu(gpu)
                    profile_bounds: list[tuple[MotivationProfile, float]] = []
                    earliest_start = max(observed_at, gpu.free_at)
                    for profile in usable_profiles:
                        earliest_duration = earliest_start - observed_at + profile.latency_seconds
                        optimistic_baseline = baseline_utility(earliest_duration)
                        best_quality_sum = 0.0
                        for group in compatibility_groups:
                            if len(group) < size:
                                continue
                            group_qualities = sorted(
                                (
                                    ready_quality[index]
                                    + ready_quality_rate[index]
                                    * (profile.quality - ready_quality[index])
                                    for index in group
                                ),
                                reverse=True,
                            )
                            quality_sum = sum(group_qualities[:size])
                            if quality_sum > best_quality_sum:
                                best_quality_sum = quality_sum
                        profile_bounds.append(
                            (
                                profile,
                                optimistic_baseline
                                + size * profile.output_seconds
                                + lambda_quality * best_quality_sum,
                            )
                        )
                    for indexes in combinations:
                        migration_count = 0
                        migration_seconds = 0.0
                        max_ready_at = observed_at
                        blocked = False
                        for index in indexes:
                            estimate = gpu_estimates[index]
                            if estimate.required:
                                migration_count += 1
                                migration_seconds += estimate.cost_seconds
                                if ready_states[index].session_id in blocked_migrations:
                                    blocked = True
                            if estimate.ready_at > max_ready_at:
                                max_ready_at = estimate.ready_at
                        if migration_count and (not self.config.migration_enabled or not allow_migrations):
                            reject("migration_disabled", len(usable_profiles))
                            continue
                        if blocked:
                            reject("migration_policy", len(usable_profiles))
                            continue
                        start_at = max(observed_at, gpu.free_at, max_ready_at)
                        action_count = sum(ready_action_flags[index] for index in indexes)
                        active_count = len(active_states)
                        for profile, optimistic_bound in profile_bounds:
                            # A profile-level optimistic bound avoids walking
                            # every member combination once an earlier
                            # candidate is already strictly better.  The bound
                            # uses the earliest possible start, grants every
                            # selected member the full output increment, and
                            # ignores migration/fairness penalties, so pruning
                            # cannot remove a true winner.  It is especially
                            # effective for the 24-session startup burst,
                            # where a fast B1 profile dominates all larger
                            # batches under the initial one-second slack.
                            if (
                                candidates_best_key is not None
                                and optimistic_bound < candidates_best_key[0] - EPSILON
                            ):
                                continue
                            finish_at = start_at + profile.latency_seconds
                            duration = finish_at - observed_at
                            baseline_score = slack_cache.get(duration)
                            if baseline_score is None:
                                baseline_score = baseline_utility(duration)
                                slack_cache[duration] = baseline_score
                            score = baseline_score
                            quality_delta_sum = 0.0
                            selected_quality_sum = 0.0
                            selected_qualities: list[float] = []
                            quality = profile.quality
                            for index in indexes:
                                updated_quality = ready_quality[index] + ready_quality_rate[index] * (
                                    quality - ready_quality[index]
                                )
                                selected_qualities.append(updated_quality)
                                selected_quality_sum += updated_quality
                                quality_delta_sum += updated_quality - ready_quality[index]
                            system_quality = (
                                (active_quality_sum + quality_delta_sum) / active_count
                                if active_count
                                else 0.0
                            )
                            fairness_limit = system_quality - fairness_delta - EPSILON
                            if any(value < fairness_limit for value in selected_qualities):
                                reject("fairness")
                                continue
                            feasible[size] += 1
                            output_seconds = profile.output_seconds
                            for index in indexes:
                                active_idx = ready_active_index[index]
                                before = active_slacks[active_idx]
                                if active_playback[active_idx]:
                                    before -= duration
                                after = before + output_seconds
                                before_utility = before if before < utility_cap else utility_cap
                                after_utility = after if after < utility_cap else utility_cap
                                score += after_utility - before_utility
                            score += lambda_quality * selected_quality_sum
                            # Penalize both transfer count and predicted transfer time.
                            score -= lambda_migration * (migration_count + migration_seconds)
                            candidate_key = (
                                score,
                                action_count,
                                size,
                                -finish_at,
                                -migration_count,
                            )
                            if candidates_best_key is not None and candidate_key <= candidates_best_key:
                                continue
                            candidates_best_key = candidate_key
                            best_record = (
                                candidate_key,
                                indexes,
                                gpu.gpu_id,
                                profile,
                                start_at,
                                finish_at,
                                migration_count,
                                migration_seconds,
                            )

        # A wait is a valid outcome only when the search found no executable
        # model candidate.  In particular, ``wait_seconds=0`` is a no-op, not
        # an alternative service decision.  Comparing that no-op against a
        # model candidate is dangerous once the aggregate slack is negative:
        # the model candidate pays its compute duration for every active
        # session while adding output credit to only the selected members, so
        # the objective can prefer doing nothing indefinitely even though a
        # GPU is idle and work is ready.  The runtime has a separate singleton
        # batch gate for intentional short waits; keeping the policy itself
        # work-conserving prevents the 24-session trace from entering that
        # starvation state.
        selected: DispatchCandidate | None = None
        if best_record is not None:
            (
                best_key,
                best_indexes,
                best_gpu_id,
                best_profile,
                best_start_at,
                best_finish_at,
                best_migration_count,
                best_migration_seconds,
            ) = best_record
            candidates_best_key = best_key
            duration = best_finish_at - observed_at
            projected_slack = {}
            for state in active_states:
                value = state.slack_seconds - (duration if state.playback_active else 0.0)
                projected_slack[state.session_id] = value
            for index in best_indexes:
                session_id = ready_states[index].session_id
                projected_slack[session_id] += best_profile.output_seconds
            projected_quality = {
                state.session_id: state.quality_ema for state in active_states
            }
            for index in best_indexes:
                state = ready_states[index]
                projected_quality[state.session_id] = state.quality_ema + state.quality_update_rate * (
                    best_profile.quality - state.quality_ema
                )
            selected = DispatchCandidate(
                session_ids=tuple(ready_states[index].session_id for index in best_indexes),
                job_ids=tuple(ready_jobs[index].job_id for index in best_indexes),
                gpu_id=best_gpu_id,
                fidelity=best_profile.fidelity,
                profile=best_profile,
                start_at=best_start_at,
                finish_at=best_finish_at,
                migration_count=best_migration_count,
                migration_seconds=best_migration_seconds,
                score=best_key[0],
                projected_slack=projected_slack,
                projected_quality=projected_quality,
                snapshot_epoch=snapshot_epoch,
                session_versions={
                    ready_states[index].session_id: ready_states[index].state_version
                    for index in best_indexes
                },
                gpu_version=next(
                    (gpu.version for gpu in gpus if gpu.gpu_id == best_gpu_id),
                    None,
                ),
                policy_name=policy_name,
            )
        if include_wait and selected is None:
            wait_candidate = self._wait_candidate(
                now=observed_at,
                wait_seconds=wait_seconds,
                states=snapshot_states,
                snapshot_epoch=snapshot_epoch,
                policy_name=policy_name,
            )
            wait_key = (
                wait_candidate.score,
                wait_candidate.action_count,
                wait_candidate.batch_size,
                -wait_candidate.finish_at,
                -wait_candidate.migration_count,
            )
            selected = wait_candidate
            candidates_best_key = wait_key

        not_selected = dict(feasible)
        if selected is not None and not selected.wait:
            not_selected[selected.batch_size] = max(0, not_selected[selected.batch_size] - 1)
        summary = MotivationSearchSummary(
            observed_at=observed_at,
            snapshot_epoch=snapshot_epoch,
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
            selected_batch_size=selected.batch_size if selected is not None else 0,
            selected_wait=bool(selected.wait) if selected is not None else False,
            selected_gpu_id=selected.gpu_id if selected is not None else None,
            selected_fidelity=selected.fidelity if selected is not None else None,
            selected_score=selected.score if selected is not None else None,
            selected_migration_count=(selected.migration_count if selected is not None else 0),
            selected_session_ids=selected.session_ids if selected is not None else (),
            selected_job_ids=selected.job_ids if selected is not None else (),
            policy_name=policy_name,
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
            observed_at = self._advance_to(observed_at)
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
            if self.policy_name == "fifo":
                # A FIFO candidate is always one action on its admitted owner;
                # reject hand-built/stale candidates that try to smuggle in an
                # idle job, a batch, or a remote placement.
                if (
                    candidate.batch_size != 1
                    or candidate.migration_count
                    or candidate.session_ids[0] not in self._sessions
                    or self._sessions[candidate.session_ids[0]].owner_gpu != candidate.gpu_id
                ):
                    return False
            for session_id, job_id, version in zip(
                candidate.session_ids,
                candidate.job_ids,
                (candidate.session_versions[sid] for sid in candidate.session_ids),
            ):
                state = self._sessions.get(session_id)
                if state is None or state.departed or state.state_version != version:
                    return False
                job = state.ready_job(include_idle=self.config.include_idle_jobs)
                if job is None or job.job_id != job_id or state.in_flight is not None:
                    return False
                if self.policy_name == "fifo" and job.kind != "action":
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
            observed_at = self._advance_to(observed_at)
            if (
                candidate.wait
                or candidate.gpu_id is None
                or not math.isfinite(candidate.start_at)
                or candidate.start_at > observed_at + EPSILON
            ):
                return False
            gpu = self._gpus.get(candidate.gpu_id)
            # ``free_at`` is a planning timestamp, not proof that the child
            # worker has published the previous output.  A parent callback can
            # arrive after the predicted profile duration; dispatching on the
            # timestamp alone creates two physical leases on one GPU and
            # double-counts peak memory.  Reservation is cleared only by
            # ``complete``/``rollback`` after the worker boundary is observed.
            return (
                gpu is not None
                and gpu.available
                and gpu.free_at <= observed_at + EPSILON
                and gpu.reserved_memory_gb <= EPSILON
            )

    def reserve(self, candidate: DispatchCandidate, *, now: float | None = None) -> DispatchCandidate:
        """Atomically reserve a candidate and move its jobs in-flight."""
        observed_at = self._clock() if now is None else now
        with self._lock:
            observed_at = self._advance_to(observed_at)
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
                job = state.ready_job(include_idle=self.config.include_idle_jobs)
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
                reserved_memory_gb=candidate.profile.memory_gb,
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
            observed_at = self._advance_to(observed_at)
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
            released_memory = gpu.reserved_memory_gb
            self._gpus[candidate.gpu_id] = GpuSchedulingState(
                gpu_id=gpu.gpu_id,
                free_at=observed_at,
                memory_free_gb=gpu.memory_free_gb + released_memory,
                available=gpu.available,
                version=gpu.version + 1,
                reserved_memory_gb=0.0,
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
            observed_at = self._advance_to(observed_at)
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
            # The child may have already reported physical model completion,
            # which releases the GPU reservation before transport/publisher
            # output reaches the parent.  Do not add the same peak twice.
            released_memory = gpu.reserved_memory_gb
            self._gpus[candidate.gpu_id] = GpuSchedulingState(
                gpu_id=gpu.gpu_id,
                free_at=observed_at,
                memory_free_gb=gpu.memory_free_gb + released_memory,
                available=gpu.available,
                version=gpu.version + 1,
                reserved_memory_gb=0.0,
            )
            self._epoch += 1
            return tuple(jobs)

    def release_gpu_reservation(
        self,
        candidate: DispatchCandidate,
        *,
        completed_at: float | None = None,
    ) -> bool:
        """Release a physical GPU slot at model completion, before output drain.

        A model invocation has two independent boundaries: the child finishes
        compute, then its chunk can spend time in the parent/output transport.
        The scheduler must let another session use the GPU at the first
        boundary while retaining each selected session's ``in_flight`` job
        until :meth:`complete` commits its output credit.  Return ``True``
        only when this call consumed an outstanding reservation; duplicate
        trace/output notifications are therefore harmless.
        """
        if candidate.wait or candidate.profile is None or candidate.gpu_id is None:
            return False
        observed_at = self._clock() if completed_at is None else completed_at
        with self._lock:
            observed_at = self._advance_to(observed_at)
            gpu = self._gpus.get(candidate.gpu_id)
            if gpu is None or gpu.reserved_memory_gb <= EPSILON:
                return False
            self._gpus[candidate.gpu_id] = GpuSchedulingState(
                gpu_id=gpu.gpu_id,
                free_at=observed_at,
                memory_free_gb=gpu.memory_free_gb + gpu.reserved_memory_gb,
                available=gpu.available,
                version=gpu.version + 1,
                reserved_memory_gb=0.0,
            )
            self._epoch += 1
            return True

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
