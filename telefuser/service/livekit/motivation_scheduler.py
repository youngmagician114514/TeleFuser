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
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, Protocol

from .model_contracts import (
    ABOT_WORLD_CAPABILITIES,
    BatchCompatibilityProvider,
    ModelServingCapabilities,
)
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
from .slackserve import (
    SlackServeTier,
    choose_bmpr_profile,
    median_quality_floor,
)
from .slackserve import (
    service_credit as slackserve_service_credit,
)
from .slackserve import (
    urgency_tier as slackserve_urgency_tier,
)
from .turboserve import (
    TurboServeClusterScheduler,
    TurboServeLatencyModel,
    TurboServeRuntimeCalibration,
    TurboServeSchedulerConfig,
    TurboServeSchedulingSnapshot,
    TurboServeSessionView,
)
from .topology import WorkerEndpoint

EPSILON = 1e-9
JobKind = Literal["action", "idle"]


def _load_spread(loads: Mapping[str, float]) -> float:
    """Return the max-minus-min projected load across available GPUs.

    Motivation's search already has a per-GPU completion timeline.  Combining
    that timeline with resident-session occupancy gives the policy a cheap,
    scheduler-visible load signal without coupling it to worker internals.
    The spread is deliberately used as a soft objective: slack, quality, and
    batching still decide when their advantage is material.
    """
    if len(loads) <= 1:
        return 0.0
    values = tuple(float(value) for value in loads.values())
    return max(values) - min(values)


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


def _pareto_profile_frontier(profiles: Sequence[MotivationProfile]) -> tuple[MotivationProfile, ...]:
    """Remove execution points dominated by another point on the same GPU.

    The policy only observes latency, memory, quality, and output credit when
    scoring a profile.  A point that is no faster, no smaller, no better in
    quality, and produces no more output can therefore never win a candidate
    comparison.  Keeping the original order makes tie-breaking deterministic
    and preserves the profile table as the source of truth for non-dominated
    points.
    """

    rows = tuple(profiles)
    if len(rows) < 2:
        return rows

    def dominates(left: MotivationProfile, right: MotivationProfile) -> bool:
        left_p95 = left.p95_seconds if left.p95_seconds is not None else left.latency_seconds
        right_p95 = right.p95_seconds if right.p95_seconds is not None else right.latency_seconds
        no_worse = (
            left.latency_seconds <= right.latency_seconds
            and left_p95 <= right_p95
            and left.memory_gb <= right.memory_gb
            and left.quality >= right.quality
            and left.output_seconds >= right.output_seconds
        )
        strictly_better = (
            left.latency_seconds < right.latency_seconds
            or left_p95 < right_p95
            or left.memory_gb < right.memory_gb
            or left.quality > right.quality
            or left.output_seconds > right.output_seconds
        )
        return no_worse and strictly_better

    return tuple(
        row
        for row in rows
        if not any(other is not row and dominates(other, row) for other in rows)
    )


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

    ``pending_action`` is the head of a bounded FIFO action queue. The queue
    may retain additional action jobs in ``pending_action_queue`` so a burst of
    controls can form cross-session batches without reordering one session's
    work. It never replaces an already generated idle video. ``in_flight`` may
    coexist with the pending queue because a control update cannot cancel a
    running model invocation.
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
    pending_action_queue: deque[ActionJob] = field(default_factory=deque, repr=False)
    pending_action_slots: int = 1
    pending_idle: ActionJob | None = None
    in_flight: ActionJob | None = None
    idle_video_remaining_seconds: float = 0.0
    # A released action owns the following heartbeat window.  Idle work may
    # be prefetched only after that window closes without a newer action.
    idle_not_before: float = 0.0
    # Idle prefetch is a resume aid for a session that has produced at least
    # one action chunk; it must not run ahead of the user's first input.
    action_seen: bool = False
    # Idle output is generated ahead of time but does not consume playout
    # slack until a later action resumes the user's stream.
    idle_video_held: bool = False
    # An action may resume while an idle job is still in-flight. Preserve that
    # intent so the late idle completion cannot re-arm the hold.
    idle_video_release_requested: bool = False
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
        if (
            not isinstance(self.pending_action_slots, int)
            or isinstance(self.pending_action_slots, bool)
            or self.pending_action_slots < 1
        ):
            raise ValueError("pending_action_slots must be a positive integer")
        if len(self.pending_action_queue) > max(0, self.pending_action_slots - 1):
            raise ValueError("pending_action_queue exceeds pending_action_slots")

    @property
    def idle_video_outstanding(self) -> bool:
        """Whether an idle output is still waiting to be consumed."""
        return self.idle_video_remaining_seconds > EPSILON

    @property
    def pending_action_count(self) -> int:
        """Return the number of queued actions, excluding ``in_flight``."""
        return (1 if self.pending_action is not None else 0) + len(self.pending_action_queue)

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
            if self.idle_video_remaining_seconds > EPSILON and not self.idle_video_held:
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
        """Update controls and enqueue one ordered action job when released.

        Empty controls and non-release heartbeat updates do not create jobs.
        The return value reports whether the session's ready queue transitioned
        from empty to non-empty. A one-slot state retains the historical
        latest-wins behavior; multi-slot states append in arrival order. An
        in-flight job never occupies a pending slot.
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
        self.action_seen = True
        self.idle_video_release_requested = True
        self.idle_video_held = False
        # ``sequence`` is globally monotonic when called through the
        # scheduler. Direct state tests may omit it, so retain a local
        # monotonic fallback. The session's current state_version remains the
        # head version until a queued job becomes runnable.
        next_version = max(self.state_version + 1, int(sequence) if sequence > 0 else 0)
        job = ActionJob(
            job_id=job_id,
            session_id=self.session_id,
            kind="action",
            controls=canonical,
            created_at=now,
            state_version=next_version,
            sequence=sequence,
        )
        had_pending = self.pending_action is not None
        if self.pending_action is None:
            self.pending_action = job
            self.state_version = next_version
            return True
        if len(self.pending_action_queue) >= self.pending_action_slots - 1:
            if self.pending_action_slots == 1:
                # Preserve the historical single-slot/latest-wins behavior
                # for baselines and callers that explicitly request one slot.
                self.pending_action = job
                self.state_version = next_version
                return False
            # Preserve FIFO order and accepted work. The caller records this
            # bounded-capacity drop separately from supersession.
            return False
        self.pending_action_queue.append(job)
        return not had_pending

    def defer_idle_until(self, not_before: float) -> None:
        """Keep idle work behind the current action heartbeat window."""
        if not math.isfinite(not_before) or not_before < 0:
            raise ValueError("not_before must be finite and non-negative")
        self.idle_not_before = max(self.idle_not_before, not_before)

    def release_idle_video(self, *, now: float) -> None:
        """Start consuming a prefetched idle chunk when action resumes."""
        self.advance_to(now)
        self.idle_video_release_requested = True
        self.idle_video_held = False

    def create_idle_job(self, *, job_id: str, now: float, sequence: int = 0) -> ActionJob | None:
        """Create one idle sentinel if no action or idle output is pending."""
        self.advance_to(now)
        # Match the paper simulator's per-session head semantics: once the
        # released action job has run, the session may contribute an idle
        # sentinel while it waits for the next heartbeat.  ``submit_action``
        # still drops a not-yet-dispatched sentinel when a fresh action is
        # released, so real action work always wins within the session.
        if self.departed or self.pending_action is not None or not self.action_seen:
            return None
        if self.latest_controls:
            return None
        if now < self.idle_not_before - EPSILON:
            return None
        if self.pending_idle is not None or self.in_flight is not None:
            return None
        if self.idle_video_outstanding:
            return None
        self.idle_video_release_requested = False
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
            self.pending_action = self.pending_action_queue.popleft() if self.pending_action_queue else None
            if self.pending_action is not None:
                self.state_version = self.pending_action.state_version
        else:
            if self.pending_idle is None or self.pending_idle.job_id != job.job_id:
                raise RuntimeError(f"idle job {job.job_id} is no longer pending")
            self.pending_idle = None
        self.in_flight = job

    def rollback_dispatched(self, job: ActionJob) -> ActionJob | None:
        """Return a synchronously rejected dispatch to the ready queue.

        Dispatch rollback is deliberately narrower than completion: it does
        not advance slack, quality, or cache ownership.  A newer action may
        already occupy ``pending_action`` in an unusual reentrant adapter; in
        that case the rejected job is restored in front of the newer work.
        When a newer action filled the bounded queue while the job was
        reserved, the newest queued action is returned to the caller as a
        capacity drop so the state remains bounded and the older job can be
        retried in FIFO order.
        """
        if self.in_flight is None or self.in_flight.job_id != job.job_id:
            raise RuntimeError(f"job {job.job_id} is not in flight for session {self.session_id}")
        self.in_flight = None
        dropped: ActionJob | None = None
        if job.kind == "action":
            if self.pending_action is None:
                self.pending_action = job
                self.state_version = job.state_version
            elif self.pending_action_slots == 1:
                # Baseline policies use explicit latest-wins semantics.  The
                # rejected reservation was never executed, so retain the
                # newer pending intent and avoid manufacturing a second slot.
                return None
            else:
                if len(self.pending_action_queue) >= self.pending_action_slots - 1:
                    # The pending head is newer than ``job`` and must move
                    # behind it.  Keep the bounded FIFO invariant by dropping
                    # the newest queued item, which is the least-progressed
                    # action and the one most naturally covered by capacity
                    # accounting.
                    dropped = self.pending_action_queue.pop()
                self.pending_action_queue.appendleft(self.pending_action)
                self.pending_action = job
                self.state_version = job.state_version
        elif self.pending_action is None and self.pending_idle is None and not self.departed:
            self.pending_idle = job
        return dropped

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
            self.idle_video_held = not self.idle_video_release_requested
            self.idle_video_release_requested = False
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
        self.pending_action_queue.clear()
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
    # Node metadata is appended after the historical positional fields so
    # existing single-host callers keep their constructor semantics.
    node_id: str = "node-0"
    endpoint_address: str | None = None
    endpoint_port: int | None = None

    def __post_init__(self) -> None:
        if not self.gpu_id:
            raise ValueError("gpu_id must be non-empty")
        if not self.node_id:
            raise ValueError("node_id must be non-empty")
        if self.endpoint_port is not None and not 1 <= self.endpoint_port <= 65535:
            raise ValueError("endpoint_port must be in [1, 65535]")
        if not math.isfinite(self.free_at) or self.free_at < 0:
            raise ValueError("free_at must be finite and non-negative")
        if self.memory_free_gb != math.inf and (
            not math.isfinite(self.memory_free_gb) or self.memory_free_gb < 0
        ):
            raise ValueError("memory_free_gb must be non-negative and finite")
        if not math.isfinite(self.reserved_memory_gb) or self.reserved_memory_gb < 0:
            raise ValueError("reserved_memory_gb must be non-negative and finite")

    def endpoint(self) -> WorkerEndpoint:
        """Return the globally identifiable endpoint represented by this state."""
        return WorkerEndpoint(
            worker_id=self.gpu_id,
            node_id=self.node_id,
            gpu_ids=(self.gpu_id,),
            address=self.endpoint_address,
            port=self.endpoint_port,
        )


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
    # Softly reward candidates that reduce the projected spread between GPU
    # timelines and resident-session occupancy.  This is Motivation-only;
    # baseline policies normalize their own objectives in ``__init__``.
    lambda_load_balance: float = 0.5
    load_balance_capacity_per_worker: int = 12
    initial_slack_seconds: float = 1.0
    initial_quality: float | None = None
    include_idle_jobs: bool = True
    migration_enabled: bool = True
    allow_cross_node_migration: bool = True
    cross_node_migration_extra_seconds: float = 0.12
    policy_name: str = "motivation"
    # Motivation may retain a short ordered action burst per session. Baseline
    # policies force this back to one in ``MotivationScheduler.__init__``.
    pending_action_slots: int = 1
    # FIFO is intentionally a fixed-execution baseline.  The paper-facing
    # baseline is pinned to the normalized S4/W18 B1 point; no quality/latency
    # trade-off is made at runtime.  A caller may still override this in a
    # reduced unit-test table, but production entrypoints use this default.
    fifo_fidelity: str | None = "b1_s4_w18_rho0_bf16"
    # TurboServe is a separate policy path.  These knobs mirror the
    # source-compatible closed-loop placement controller while keeping its
    # fixed execution family independent from Motivation quality selection.
    turboserve_capacity_per_worker: int = 12
    turboserve_target_utilization: float = 0.9
    turboserve_migration_eta: float = 0.01
    turboserve_min_gain_ms: float = 40.0
    turboserve_rebalance_iteration_limit: int = 3
    turboserve_fidelity_family: str = "s4_w18_rho0_bf16"
    # SlackServe is a B1 stream scheduler. BMPR chooses among the measured B1
    # Pareto points, while service credit and re-homing follow the paper's
    # 2T/4T tiers.  The production baseline uses an explicit, mildly relaxed
    # floor so the fastest useful B1 point (currently ~Q=0.895) is eligible;
    # callers can set ``None`` to recover the paper-style median floor.
    slackserve_quality_floor: float | None = 0.895
    # Keep the paper-facing policy action-only by default, while allowing a
    # controlled SlackServe+idle baseline to reuse the same credit/BMPR path.
    slackserve_include_idle_jobs: bool = False
    slackserve_initial_slack_multiplier: float = 4.0
    slackserve_urgent_multiplier: float = 2.0
    slackserve_relaxed_multiplier: float = 4.0
    slackserve_rehoming_enabled: bool = True
    slackserve_capacity_per_worker: int = 12
    slackserve_rehoming_control_interval_seconds: float = 3.0
    slackserve_max_rehomes_per_sender: int = 2
    slackserve_max_rehomes_per_receiver: int = 1
    # Model-facing metadata is deliberately separate from policy constants.
    # The ABot default preserves the historical 12 FPS / 12-frame job
    # contract; another backend can provide its own values without changing
    # the scheduler implementation.
    model_capabilities: ModelServingCapabilities = ABOT_WORLD_CAPABILITIES

    def __post_init__(self) -> None:
        if not isinstance(self.model_capabilities, ModelServingCapabilities):
            raise TypeError("model_capabilities must be a ModelServingCapabilities instance")
        if not 1 <= self.max_batch_size <= 4:
            raise ValueError("max_batch_size must be in [1, 4]")
        if (
            not isinstance(self.pending_action_slots, int)
            or isinstance(self.pending_action_slots, bool)
            or self.pending_action_slots < 1
        ):
            raise ValueError("pending_action_slots must be a positive integer")
        for name in (
            "utility_cap_seconds",
            "initial_slack_seconds",
            "fairness_delta",
            "lambda_quality",
            "lambda_migration",
            "lambda_load_balance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.utility_cap_seconds <= 0 or self.initial_slack_seconds <= 0:
            raise ValueError("utility_cap_seconds and initial_slack_seconds must be positive")
        if (
            not isinstance(self.load_balance_capacity_per_worker, int)
            or isinstance(self.load_balance_capacity_per_worker, bool)
            or self.load_balance_capacity_per_worker < 1
        ):
            raise ValueError("load_balance_capacity_per_worker must be a positive integer")
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
        if (
            not isinstance(self.turboserve_capacity_per_worker, int)
            or isinstance(self.turboserve_capacity_per_worker, bool)
            or self.turboserve_capacity_per_worker < 1
        ):
            raise ValueError("turboserve_capacity_per_worker must be a positive integer")
        if (
            not math.isfinite(float(self.turboserve_target_utilization))
            or not 0 < self.turboserve_target_utilization <= 1
        ):
            raise ValueError("turboserve_target_utilization must be in (0, 1]")
        for name in ("turboserve_migration_eta", "turboserve_min_gain_ms"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            not isinstance(self.turboserve_rebalance_iteration_limit, int)
            or isinstance(self.turboserve_rebalance_iteration_limit, bool)
            or not 1 <= self.turboserve_rebalance_iteration_limit <= 64
        ):
            raise ValueError("turboserve_rebalance_iteration_limit must be in [1, 64]")
        if not isinstance(self.turboserve_fidelity_family, str) or not self.turboserve_fidelity_family.strip():
            raise ValueError("turboserve_fidelity_family must be a non-empty string")
        if self.slackserve_quality_floor is not None and (
            not math.isfinite(float(self.slackserve_quality_floor))
            or float(self.slackserve_quality_floor) <= 0
        ):
            raise ValueError("slackserve_quality_floor must be positive and finite")
        for name in (
            "slackserve_initial_slack_multiplier",
            "slackserve_urgent_multiplier",
            "slackserve_relaxed_multiplier",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.slackserve_initial_slack_multiplier <= 0:
            raise ValueError("slackserve_initial_slack_multiplier must be positive")
        if self.slackserve_relaxed_multiplier <= self.slackserve_urgent_multiplier:
            raise ValueError(
                "slackserve_relaxed_multiplier must exceed slackserve_urgent_multiplier"
            )
        if not isinstance(self.slackserve_rehoming_enabled, bool):
            raise ValueError("slackserve_rehoming_enabled must be a bool")
        if not isinstance(self.slackserve_include_idle_jobs, bool):
            raise ValueError("slackserve_include_idle_jobs must be a bool")
        if (
            not isinstance(self.slackserve_capacity_per_worker, int)
            or isinstance(self.slackserve_capacity_per_worker, bool)
            or self.slackserve_capacity_per_worker < 1
        ):
            raise ValueError("slackserve_capacity_per_worker must be a positive integer")
        if (
            not math.isfinite(float(self.slackserve_rehoming_control_interval_seconds))
            or self.slackserve_rehoming_control_interval_seconds <= 0
        ):
            raise ValueError("slackserve_rehoming_control_interval_seconds must be positive and finite")
        for name in (
            "slackserve_max_rehomes_per_sender",
            "slackserve_max_rehomes_per_receiver",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


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


@dataclass(frozen=True)
class PlacementDecision:
    """A cheap re-homing decision kept separate from compute selection.

    Placement deliberately carries no batch size or fidelity.  The runtime
    turns it into a migration-only ``DispatchCandidate`` and then lets the
    local compute policy choose ``(B, quality)`` after ownership commits.
    """

    session_id: str
    source_gpu: str
    target_gpu: str
    estimated_ready_at: float
    migration_seconds: float
    snapshot_epoch: int
    session_version: int
    target_gpu_version: int


@dataclass(frozen=True)
class _SlackServeAssessment:
    """One cached B1 BMPR/credit decision within a search snapshot."""

    state: SessionSchedulingState
    job: ActionJob | None
    gpu: GpuSchedulingState
    profile: MotivationProfile
    credit: float
    tier: SlackServeTier

    @property
    def priority(self) -> tuple[int, float, int, str]:
        # Only ready assessments enter a worker queue.  Resident-only
        # assessments use ``job=None`` solely for receiver tier checks.
        sequence = self.job.sequence if self.job is not None else 0
        return (int(self.tier), self.credit, sequence, self.state.session_id)


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
        batch_compatibility_provider: BatchCompatibilityProvider | None = None,
    ) -> None:
        self.profile_provider = profile_provider
        if batch_compatibility_provider is not None and not isinstance(
            batch_compatibility_provider, BatchCompatibilityProvider
        ):
            raise TypeError("batch_compatibility_provider must implement BatchCompatibilityProvider")
        self.batch_compatibility_provider = batch_compatibility_provider
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
                pending_action_slots=1,
            )
        elif self._policy.name == "turboserve":
            # TurboServe suspends idle sessions and has no adaptive quality or
            # fairness objective.  The checked TurboServe source executes one
            # SessionState at a time with LocalRoundRobinStepScheduler, so this
            # source-faithful baseline is a fixed B1 singleton path.  Its
            # migration-aware placement remains implemented below.
            self.config = replace(
                requested_config,
                max_batch_size=1,
                include_idle_jobs=False,
                lambda_quality=0.0,
                fairness_delta=0.0,
                pending_action_slots=1,
            )
        elif self._policy.name == "slackserve":
            # SlackServe schedules one sequential chunk per stream.  Its
            # BMPR and service-credit logic is policy-local; it does not use
            # Motivation's batch, utility, quality-EMA, or fairness objective.
            # Idle sentinels are disabled for the paper-facing baseline.  An
            # explicit controlled variant may enable them without entering
            # Motivation's search or changing SlackServe's B1 execution path.
            self.config = replace(
                requested_config,
                max_batch_size=1,
                include_idle_jobs=requested_config.slackserve_include_idle_jobs,
                lambda_quality=0.0,
                fairness_delta=0.0,
                pending_action_slots=1,
            )
        else:
            self.config = requested_config
        self.migration_estimator = migration_estimator or LocalMigrationEstimator()
        self._diagnostics = diagnostics or NullMotivationDiagnostics()
        self._clock = clock
        self._sessions: dict[str, SessionSchedulingState] = {}
        self._gpus: dict[str, GpuSchedulingState] = {}
        self._job_sequence = 0
        # Action-request accounting belongs to the shared coordinator rather
        # than an individual policy.  A newly released action can supersede a
        # job that is still pending, but it can never cancel an in-flight job.
        # Keeping both totals lets paper metrics exclude only genuine
        # latest-intent supersession without hiding dispatch/runtime failures.
        self._action_jobs_released = 0
        self._action_jobs_superseded_pending = 0
        self._action_jobs_dropped_capacity = 0
        self._action_jobs_completed = 0
        self._action_jobs_dropped_on_departure = 0
        # Paper re-homing limits are control-tick limits, not global lifetime
        # limits.  Counts are recorded when a migration actually starts; a
        # fresh policy search can therefore admit at most two sends per source
        # and one receive per target in one SlackServe control window.
        self._slackserve_rehome_sends: dict[tuple[int, str], int] = {}
        self._slackserve_rehome_receives: dict[tuple[int, str], int] = {}
        # Offline profile tables are immutable for the lifetime of a serving
        # runtime.  Cache their Pareto frontiers so every action-triggered
        # search does not repeat dominance checks.  Callers that replace a
        # profile provider can explicitly invalidate this cache.
        self._profile_frontier_cache: dict[tuple[str, int], tuple[MotivationProfile, ...]] = {}
        self._epoch = 0
        self._now = 0.0
        self._lock = threading.RLock()

    @property
    def model_capabilities(self) -> ModelServingCapabilities:
        """Return the backend contract carried by the scheduler config."""

        return self.config.model_capabilities

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

    def invalidate_profile_cache(self) -> None:
        """Invalidate cached profile frontiers after a profile-table update."""

        with self._lock:
            self._profile_frontier_cache.clear()

    def _estimate_migration(
        self,
        session: SessionSchedulingState,
        *,
        target_gpu: GpuSchedulingState,
        now: float,
    ) -> MigrationEstimate:
        """Estimate a move while respecting the deployment's node boundary.

        A multi-node runtime owns each session's LiveKit room on one node, so a
        cross-node candidate carries the configured network-transfer estimate.
        The distributed worker pool implements the corresponding state handoff
        through the node-agent process-NCCL data plane.
        """
        owner = self._gpus.get(session.owner_gpu)
        if (
            not self.config.allow_cross_node_migration
            and owner is not None
            and owner.node_id != target_gpu.node_id
        ):
            return MigrationEstimate(ready_at=now, required=False)
        estimate = self.migration_estimator.estimate(session, target_gpu=target_gpu.gpu_id, now=now)
        if owner is None or owner.node_id == target_gpu.node_id or not estimate.required:
            return estimate
        extra = max(0.0, float(self.config.cross_node_migration_extra_seconds))
        if extra <= 0:
            return estimate
        return replace(
            estimate,
            ready_at=estimate.ready_at + extra,
            cost_seconds=estimate.cost_seconds + extra,
            first_layer_ready_seconds=estimate.first_layer_ready_seconds + extra,
        )

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
                or previous.node_id != state.node_id
                or previous.endpoint_address != state.endpoint_address
                or previous.endpoint_port != state.endpoint_port
            ):
                version = max(previous.version + 1, version)
            self._gpus[state.gpu_id] = GpuSchedulingState(
                gpu_id=state.gpu_id,
                node_id=state.node_id,
                endpoint_address=state.endpoint_address,
                endpoint_port=state.endpoint_port,
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
                node_id=current.node_id,
                endpoint_address=current.endpoint_address,
                endpoint_port=current.endpoint_port,
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
            if initial_quality is None and self.policy_name in {"fifo", "turboserve"}:
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
            initial_slack = self.config.initial_slack_seconds
            if self.policy_name == "slackserve" and slack_seconds is None:
                # SlackServe gives the first chunk a TTFC budget of four times
                # its profiled high-quality chunk duration.  Use the measured
                # B1 point rather than a model-independent constant.
                first_chunk_profiles = tuple(
                    self.profile_provider.profiles_for(batch_size=1, gpu_id=owner_gpu)
                )
                if first_chunk_profiles:
                    highest_quality = max(
                        first_chunk_profiles,
                        key=lambda profile: (
                            profile.quality,
                            -profile.latency_seconds,
                            profile.fidelity,
                        ),
                    )
                    initial_slack = (
                        self.config.slackserve_initial_slack_multiplier
                        * highest_quality.latency_seconds
                    )
            normalized_compatibility_key = tuple(compatibility_key)
            if not normalized_compatibility_key and self.batch_compatibility_provider is not None:
                try:
                    normalized_compatibility_key = tuple(
                        self.batch_compatibility_provider.batch_compatibility_key(session_id)
                    )
                except (KeyError, LookupError):
                    # A worker may not have published model state at admission
                    # yet.  Keep the historical empty key until its first
                    # compute-complete event reports one.
                    normalized_compatibility_key = ()
            state = SessionSchedulingState(
                session_id=session_id,
                owner_gpu=owner_gpu,
                quality_update_rate=self.config.quality_ema,
                slack_seconds=(
                    initial_slack if slack_seconds is None else slack_seconds
                ),
                quality_ema=initial_quality,
                compatibility_key=normalized_compatibility_key,
                pending_action_slots=self.config.pending_action_slots,
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
            if state.pending_action is not before:
                job = state.pending_action
            else:
                job = next(
                    (queued for queued in state.pending_action_queue if queued.job_id == job_id),
                    None,
                )
            if release:
                self._action_jobs_released += 1
                if job is None:
                    self._action_jobs_dropped_capacity += 1
                elif before is not None and self.config.pending_action_slots == 1:
                    self._action_jobs_superseded_pending += 1
                # Replacement changes the session version, so a candidate that
                # captured the old job is rejected even if the global ready set
                # did not transition from empty to non-empty.
                if job is not None:
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

    def defer_idle_until(self, session_id: str, not_before: float) -> None:
        """Keep a session's idle head behind its next action window."""
        with self._lock:
            state = self._sessions[session_id]
            state.defer_idle_until(not_before)
            self._epoch += 1

    def release_idle_video(self, session_id: str, *, now: float | None = None) -> None:
        """Release one prefetched idle chunk when a new action resumes."""
        observed_at = self._clock() if now is None else now
        with self._lock:
            observed_at = self._advance_to(observed_at)
            state = self._sessions.get(session_id)
            if state is None or state.departed:
                return
            state.release_idle_video(now=observed_at)
            self._epoch += 1

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
            state = self._sessions[session_id]
            self._action_jobs_dropped_on_departure += state.pending_action_count
            state.mark_departed(now=observed_at)
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
            if self.policy_name == "slackserve" and state.owner_gpu != target_gpu:
                bucket = int(
                    math.floor(
                        observed_at
                        / self.config.slackserve_rehoming_control_interval_seconds
                    )
                )
                source_key = (bucket, state.owner_gpu)
                target_key = (bucket, target_gpu)
                self._slackserve_rehome_sends[source_key] = (
                    self._slackserve_rehome_sends.get(source_key, 0) + 1
                )
                self._slackserve_rehome_receives[target_key] = (
                    self._slackserve_rehome_receives.get(target_key, 0) + 1
                )
            state.migration_target_gpu = target_gpu
            state.migration_ready_at = ready_at
            self._epoch += 1

    def update_session_compatibility(
        self,
        session_id: str,
        compatibility_key: Iterable[object] | None = None,
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
            if compatibility_key is None:
                if self.batch_compatibility_provider is None:
                    return
                try:
                    compatibility_key = self.batch_compatibility_provider.batch_compatibility_key(session_id)
                except (KeyError, LookupError):
                    return
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

    def _find_slackserve_candidate(
        self,
        *,
        now: float | None = None,
        gpu_states: Sequence[GpuSchedulingState] | None = None,
        wait_seconds: float = 0.0,
        include_wait: bool = True,
        allow_migrations: bool = True,
        exclude_session_ids: Iterable[str] = (),
        blocked_migration_session_ids: Iterable[str] = (),
        policy_name: str = "slackserve",
    ) -> DispatchCandidate | None:
        """Select one SlackServe stream head using credit, BMPR, and re-homing.

        This path is intentionally independent from Motivation's Cartesian
        batch/quality search.  Each worker has a B1 stream queue ordered by
        service credit ``P - (R + T)``.  BMPR selects one measured Pareto
        profile per stream.  Before local dispatch, an urgent queued stream
        may be re-homed when an empty/relaxed receiver would finish it sooner
        after the measured route-ready delay.

        SlackServe's Elastic SP is not synthesized here.  It requires one
        model invocation to execute across two workers; ABot currently exposes
        only one-worker singleton or cross-session data-parallel batch calls.
        """

        observed_at = self._clock() if now is None else now
        excluded = {str(value) for value in exclude_session_ids}
        blocked = {str(value) for value in blocked_migration_session_ids}

        with self._lock:
            observed_at = self._advance_to(observed_at)
            snapshot_states = tuple(
                replace(state, pending_action_queue=deque(state.pending_action_queue))
                for state in self._sessions.values()
            )
            snapshot_by_id = {state.session_id: state for state in snapshot_states}
            all_ready = tuple(
                (snapshot_by_id[state.session_id], job)
                for state, job in self._ready_jobs()
            )
            ready = tuple(
                item for item in all_ready if item[0].session_id not in excluded
            )
            gpus = tuple(gpu_states) if gpu_states is not None else tuple(self._gpus.values())
            rehome_sends = dict(self._slackserve_rehome_sends)
            rehome_receives = dict(self._slackserve_rehome_receives)
            snapshot_epoch = self._epoch

        available_gpus = tuple(gpu for gpu in gpus if gpu.available)
        gpu_by_id = {gpu.gpu_id: gpu for gpu in available_gpus}
        enumerated = empty_batch_counts()
        compatible = empty_batch_counts()
        profiles_evaluated = empty_batch_counts()
        feasible = empty_batch_counts()
        rejected: dict[str, int] = {}

        def reject(reason: str, count: int = 1) -> None:
            rejected[reason] = rejected.get(reason, 0) + count

        def record(selected: DispatchCandidate | None) -> DispatchCandidate | None:
            result = selected
            if result is None and include_wait:
                result = self._wait_candidate(
                    now=observed_at,
                    wait_seconds=wait_seconds,
                    states=snapshot_states,
                    snapshot_epoch=snapshot_epoch,
                    policy_name=policy_name,
                )
            not_selected = dict(feasible)
            if result is not None and not result.wait:
                not_selected[1] = max(0, not_selected[1] - 1)
            summary = MotivationSearchSummary(
                observed_at=observed_at,
                snapshot_epoch=snapshot_epoch,
                ready_count=len(ready),
                ready_action_count=sum(1 for _, job in ready if job.kind == "action"),
                ready_idle_count=sum(1 for _, job in ready if job.kind == "idle"),
                pending_action_count=sum(
                    state.pending_action_count for state in snapshot_states if not state.departed
                ),
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
                selected_batch_size=result.batch_size if result is not None else 0,
                selected_wait=bool(result.wait) if result is not None else False,
                selected_gpu_id=result.gpu_id if result is not None else None,
                selected_fidelity=result.fidelity if result is not None else None,
                selected_score=result.score if result is not None else None,
                selected_migration_count=result.migration_count if result is not None else 0,
                selected_session_ids=result.session_ids if result is not None else (),
                selected_job_ids=result.job_ids if result is not None else (),
                policy_name=policy_name,
            )
            try:
                self._diagnostics.record_search(summary)
            except Exception:
                pass
            return result

        if not ready or not available_gpus:
            if not available_gpus:
                reject("gpu_unavailable", max(1, len(ready)))
            return record(None)

        enumerated[1] = len(ready) * len(available_gpus)
        profile_rows: dict[str, tuple[MotivationProfile, ...]] = {}
        unique_rows: dict[tuple[str, float, float], MotivationProfile] = {}
        for gpu in available_gpus:
            rows = tuple(
                self.profile_provider.profiles_for(batch_size=1, gpu_id=gpu.gpu_id)
            )
            profile_rows[gpu.gpu_id] = rows
            profiles_evaluated[1] += len(rows)
            for row in rows:
                unique_rows.setdefault(
                    (row.fidelity, row.latency_seconds, row.quality), row
                )
        if not unique_rows:
            reject("no_profile", len(ready))
            return record(None)
        quality_floor = self.config.slackserve_quality_floor
        if quality_floor is None:
            quality_floor = median_quality_floor(tuple(unique_rows.values()))
        rehome_bucket = int(
            math.floor(
                observed_at / self.config.slackserve_rehoming_control_interval_seconds
            )
        )

        def assess(
            state: SessionSchedulingState,
            gpu: GpuSchedulingState,
            job: ActionJob | None,
            *,
            remaining_seconds: float = 0.0,
        ) -> _SlackServeAssessment | None:
            rows = profile_rows.get(gpu.gpu_id, ())
            if not rows:
                reject("no_profile")
                return None
            try:
                profile = choose_bmpr_profile(
                    rows,
                    slack_budget_seconds=state.slack_seconds,
                    quality_floor=quality_floor,
                )
            except ValueError:
                reject("quality_floor")
                return None
            credit = slackserve_service_credit(
                playout_slack_seconds=state.slack_seconds,
                remaining_seconds=remaining_seconds,
                next_chunk_seconds=profile.latency_seconds,
            )
            tier = slackserve_urgency_tier(
                credit,
                profile.latency_seconds,
                urgent_multiplier=self.config.slackserve_urgent_multiplier,
                relaxed_multiplier=self.config.slackserve_relaxed_multiplier,
            )
            return _SlackServeAssessment(state, job, gpu, profile, credit, tier)

        def has_memory(gpu: GpuSchedulingState, profile: MotivationProfile) -> bool:
            effective_memory = gpu.memory_free_gb
            if gpu.free_at > observed_at + EPSILON and effective_memory != math.inf:
                effective_memory += gpu.reserved_memory_gb
            return profile.memory_gb <= effective_memory + EPSILON

        def make_candidate(
            assessment: _SlackServeAssessment,
            *,
            estimate: MigrationEstimate | None = None,
        ) -> DispatchCandidate | None:
            job = assessment.job
            if job is None:
                return None
            if not has_memory(assessment.gpu, assessment.profile):
                reject("memory")
                return None
            migration_count = int(estimate is not None and estimate.required)
            migration_seconds = estimate.cost_seconds if migration_count else 0.0
            ready_at = estimate.ready_at if estimate is not None else observed_at
            start_at = max(observed_at, assessment.gpu.free_at, ready_at)
            finish_at = start_at + assessment.profile.latency_seconds
            feasible[1] += 1
            return DispatchCandidate(
                session_ids=(assessment.state.session_id,),
                job_ids=(job.job_id,),
                gpu_id=assessment.gpu.gpu_id,
                fidelity=assessment.profile.fidelity,
                profile=assessment.profile,
                start_at=start_at,
                finish_at=finish_at,
                migration_count=migration_count,
                migration_seconds=migration_seconds,
                # SlackServe minimizes credit. Negation keeps the common
                # diagnostic field's larger-is-better convention readable.
                score=-assessment.credit,
                projected_slack={
                    assessment.state.session_id: (
                        assessment.state.slack_seconds
                        - max(0.0, finish_at - observed_at)
                        + assessment.profile.output_seconds
                    )
                },
                projected_quality={
                    assessment.state.session_id: assessment.profile.quality
                },
                snapshot_epoch=snapshot_epoch,
                session_versions={
                    assessment.state.session_id: assessment.state.state_version
                },
                gpu_version=assessment.gpu.version,
                policy_name=policy_name,
            )

        # Build each worker's credit-ordered stream-head queue.  A session has
        # at most one ready head in the common coordinator.
        local_queues: dict[str, list[_SlackServeAssessment]] = {
            gpu.gpu_id: [] for gpu in available_gpus
        }
        for state, job in ready:
            owner = gpu_by_id.get(state.owner_gpu)
            if owner is None:
                reject("owner_gpu_missing")
                continue
            assessment = assess(state, owner, job)
            if assessment is not None:
                compatible[1] += 1
                local_queues[owner.gpu_id].append(assessment)
        for queue in local_queues.values():
            queue.sort(key=lambda item: item.priority)

        # Paper-style re-homing runs before local dispatch.  Only queued
        # urgent work is considered, and a receiver must contain no urgent or
        # normal resident stream.  The finish-time check includes the actual
        # route-ready estimate, so a migration is never selected merely to
        # balance counts.
        migration_options: list[
            tuple[tuple[float, float, int, str, str], DispatchCandidate]
        ] = []
        if (
            self.config.migration_enabled
            and self.config.slackserve_rehoming_enabled
            and allow_migrations
        ):
            receiver_relaxed: dict[str, bool] = {}
            for receiver in available_gpus:
                relaxed_only = True
                remaining = max(0.0, receiver.free_at - observed_at)
                resident_count = 0
                for state in snapshot_states:
                    if state.departed or state.owner_gpu != receiver.gpu_id:
                        continue
                    resident_count += 1
                    resident = assess(
                        state,
                        receiver,
                        None,
                        remaining_seconds=(remaining if state.in_flight is not None else 0.0),
                    )
                    if resident is None or resident.tier is not SlackServeTier.RELAXED:
                        relaxed_only = False
                        break
                if resident_count >= self.config.slackserve_capacity_per_worker:
                    relaxed_only = False
                receiver_relaxed[receiver.gpu_id] = relaxed_only

            for source in available_gpus:
                queue = local_queues[source.gpu_id]
                if not queue:
                    continue
                if (
                    rehome_sends.get((rehome_bucket, source.gpu_id), 0)
                    >= self.config.slackserve_max_rehomes_per_sender
                ):
                    reject("rehoming_sender_tick_limit")
                    continue
                # If the worker can immediately run its sole head, migration
                # only adds transfer latency and cannot improve completion.
                if source.free_at <= observed_at + EPSILON and len(queue) <= 1:
                    continue
                source_cursor = max(observed_at, source.free_at)
                for item in queue:
                    source_finish = source_cursor + item.profile.latency_seconds
                    source_cursor = source_finish
                    if item.tier is not SlackServeTier.URGENT:
                        continue
                    if item.state.session_id in blocked:
                        continue
                    for receiver in available_gpus:
                        if (
                            receiver.gpu_id == source.gpu_id
                            or not receiver_relaxed.get(receiver.gpu_id, False)
                            or rehome_receives.get(
                                (rehome_bucket, receiver.gpu_id), 0
                            )
                            >= self.config.slackserve_max_rehomes_per_receiver
                        ):
                            continue
                        target_item = assess(item.state, receiver, item.job)
                        if target_item is None:
                            continue
                        estimate = self._estimate_migration(
                            item.state, target_gpu=receiver, now=observed_at
                        )
                        if not estimate.required:
                            continue
                        target_start = max(
                            observed_at, receiver.free_at, estimate.ready_at
                        )
                        target_finish = target_start + target_item.profile.latency_seconds
                        if target_finish >= source_finish - EPSILON:
                            reject("rehoming_no_gain")
                            continue
                        candidate = make_candidate(target_item, estimate=estimate)
                        if candidate is None:
                            continue
                        migration_options.append(
                            (
                                (
                                    item.credit,
                                    target_finish,
                                    item.job.sequence if item.job is not None else 0,
                                    item.state.session_id,
                                    receiver.gpu_id,
                                ),
                                candidate,
                            )
                        )
            if migration_options:
                migration_options.sort(key=lambda option: option[0])
                return record(migration_options[0][1])

        # Dispatch the head of every worker-local credit queue, preferring a
        # worker that is physically available first.  This mirrors independent
        # per-worker queues while still using the common global drain loop to
        # fill several idle GPUs in one scheduling turn.
        local_options: list[
            tuple[tuple[float, int, float, int, str], DispatchCandidate]
        ] = []
        for gpu in available_gpus:
            queue = local_queues[gpu.gpu_id]
            if not queue:
                continue
            item = queue[0]
            candidate = make_candidate(item)
            if candidate is None:
                continue
            local_options.append(
                (
                    (
                        candidate.start_at,
                        int(item.tier),
                        item.credit,
                        item.job.sequence if item.job is not None else 0,
                        gpu.gpu_id,
                    ),
                    candidate,
                )
            )
        if local_options:
            local_options.sort(key=lambda option: option[0])
            return record(local_options[0][1])
        return record(None)

    def _find_turboserve_candidate(
        self,
        *,
        now: float | None = None,
        gpu_states: Sequence[GpuSchedulingState] | None = None,
        wait_seconds: float = 0.0,
        include_wait: bool = True,
        allow_migrations: bool = True,
        exclude_session_ids: Iterable[str] = (),
        blocked_migration_session_ids: Iterable[str] = (),
        policy_name: str = "turboserve",
    ) -> DispatchCandidate | None:
        """Select one source-faithful TurboServe singleton chunk candidate.

        TurboServe's reference policy is a placement controller rather than a
        quality optimizer.  We therefore first run its least-load/min--max
        placement calculation over the coordinator snapshot, then expose one
        fixed-family B1 job to the shared lease machinery.  Existing owners are
        retained whenever the placement controller finds no positive
        bottleneck-reduction move; this is important for low-pressure traces,
        where moving sessions merely to make a larger batch would be harmful.
        """
        observed_at = self._clock() if now is None else now
        excluded = {str(value) for value in exclude_session_ids}
        blocked = {str(value) for value in blocked_migration_session_ids}

        with self._lock:
            observed_at = self._advance_to(observed_at)
            live_states = tuple(self._sessions.values())
            snapshot_states = tuple(
                replace(state, pending_action_queue=deque(state.pending_action_queue))
                for state in live_states
            )
            snapshot_by_id = {state.session_id: state for state in snapshot_states}
            all_ready = tuple(
                (snapshot_by_id[state.session_id], job)
                for state, job in self._ready_jobs()
                if job.kind == "action"
            )
            ready = tuple(item for item in all_ready if item[0].session_id not in excluded)
            gpus = tuple(gpu_states) if gpu_states is not None else tuple(self._gpus.values())
            snapshot_epoch = self._epoch

        available_gpus = tuple(gpu for gpu in gpus if gpu.available)
        enumerated = empty_batch_counts()
        compatible = empty_batch_counts()
        profiles_evaluated = empty_batch_counts()
        feasible = empty_batch_counts()
        rejected: dict[str, int] = {}

        def reject(reason: str, count: int = 1) -> None:
            rejected[reason] = rejected.get(reason, 0) + count

        def record_and_wait(selected: DispatchCandidate | None) -> DispatchCandidate | None:
            result = selected
            if result is None and include_wait:
                result = self._wait_candidate(
                    now=observed_at,
                    wait_seconds=wait_seconds,
                    states=snapshot_states,
                    snapshot_epoch=snapshot_epoch,
                    policy_name=policy_name,
                )
            not_selected = dict(feasible)
            if result is not None and not result.wait:
                not_selected[result.batch_size] = max(0, not_selected[result.batch_size] - 1)
            summary = MotivationSearchSummary(
                observed_at=observed_at,
                snapshot_epoch=snapshot_epoch,
                ready_count=len(ready),
                ready_action_count=sum(1 for _, job in ready if job.kind == "action"),
                ready_idle_count=sum(1 for _, job in ready if job.kind == "idle"),
                pending_action_count=sum(
                    state.pending_action_count for state in snapshot_states if not state.departed
                ),
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
                selected_batch_size=result.batch_size if result is not None else 0,
                selected_wait=bool(result.wait) if result is not None else False,
                selected_gpu_id=result.gpu_id if result is not None else None,
                selected_fidelity=result.fidelity if result is not None else None,
                selected_score=result.score if result is not None else None,
                selected_migration_count=result.migration_count if result is not None else 0,
                selected_session_ids=result.session_ids if result is not None else (),
                selected_job_ids=result.job_ids if result is not None else (),
                policy_name=policy_name,
            )
            try:
                self._diagnostics.record_search(summary)
            except Exception:
                pass
            return result

        if not ready or not available_gpus:
            if not available_gpus:
                reject("gpu_unavailable", max(1, len(ready)))
            return record_and_wait(None)

        # The source baseline has one fixed semantic execution family and one
        # session per model invocation.  The paper discusses coalesced chunks,
        # but the checked repository implementation's round-robin worker does
        # not assemble cross-session batches; keep this path B1-only.
        family = self.config.turboserve_fidelity_family.strip()
        parsed_batch, parsed_family = profile_batch_family(family)
        if parsed_batch is not None:
            family = parsed_family

        profile_cache: dict[tuple[str, int], MotivationProfile | None] = {}

        def fixed_profile(gpu_id: str, batch_size: int) -> MotivationProfile | None:
            key = (gpu_id, batch_size)
            if key in profile_cache:
                return profile_cache[key]
            requested = f"b{batch_size}_{family}"
            rows = tuple(
                row
                for row in self.profile_provider.profiles_for(
                    batch_size=batch_size,
                    gpu_id=gpu_id,
                )
                if row.fidelity == requested
            )
            profile_cache[key] = rows[0] if rows else None
            profiles_evaluated[batch_size] += len(rows)
            return profile_cache[key]

        # The shared scheduler remains the source of truth for ownership and
        # version validation. Calibrate TurboServe's analytic model from the
        # same fixed B1 profile used by physical dispatch; its generic 180-ms
        # default would otherwise undervalue measured ABot S4/W18 chunks.
        b1_latencies_ms = tuple(
            profile.latency_seconds * 1000.0
            for gpu in available_gpus
            if (profile := fixed_profile(gpu.gpu_id, 1)) is not None
        )
        base_chunk_latency_ms = (
            sum(b1_latencies_ms) / len(b1_latencies_ms)
            if b1_latencies_ms
            else 0.0
        )
        worker_order = tuple(gpu.gpu_id for gpu in available_gpus)
        active = {
            state.session_id: TurboServeSessionView(
                session_id=state.session_id,
                # TurboServe's alpha_i is user activity.  The shared action
                # adapter updates ``latest_controls`` even for non-release
                # messages, making it the exact active/idle signal while an
                # already in-flight chunk remains protected by the common
                # lease state.
                active=bool(state.latest_controls and not state.departed),
                state_size_mb=1.0,
            )
            for state in snapshot_states
        }
        placement = {
            state.session_id: state.owner_gpu
            for state in snapshot_states
            if not state.departed
        }
        route_ready_ms = max(
            0.0,
            float(getattr(self.migration_estimator, "first_layer_ready_seconds", 0.0)) * 1000.0,
        )
        latency_model = TurboServeLatencyModel()
        if base_chunk_latency_ms > 0:
            latency_model.base_chunk_latency_ms = base_chunk_latency_ms
        cluster = TurboServeClusterScheduler(
            TurboServeSchedulerConfig(
                enable_autoscaling=False,
                enable_migration=bool(self.config.migration_enabled and allow_migrations),
                min_workers=1,
                max_workers=len(worker_order),
                capacity_per_worker=self.config.turboserve_capacity_per_worker,
                target_utilization=self.config.turboserve_target_utilization,
                migration_eta=self.config.turboserve_migration_eta,
                min_gain_ms=self.config.turboserve_min_gain_ms,
                rebalance_iteration_limit=self.config.turboserve_rebalance_iteration_limit,
            ),
            latency_model=latency_model,
        )
        placement_decision = cluster.decide(
            TurboServeSchedulingSnapshot(
                time_seconds=observed_at,
                sessions=active,
                placement=placement,
                current_workers=len(worker_order),
                worker_order=worker_order,
                capacity_per_worker=self.config.turboserve_capacity_per_worker,
                runtime_calibration=TurboServeRuntimeCalibration(
                    average_route_ready_ms=route_ready_ms,
                    p50_route_ready_ms=route_ready_ms,
                    base_chunk_latency_ms=base_chunk_latency_ms,
                ),
            )
        )
        desired = placement_decision.placement

        ready_by_id = {state.session_id: (state, job) for state, job in ready}
        enumerated[1] = len(ready) * len(available_gpus)

        def make_candidate(
            session_items: Sequence[tuple[SessionSchedulingState, ActionJob]],
            gpu: GpuSchedulingState,
            estimates: Mapping[str, MigrationEstimate] | None = None,
        ) -> DispatchCandidate | None:
            if len(session_items) != 1:
                reject("turboserve_singleton_only")
                return None
            ordered = tuple(
                sorted(
                    session_items,
                    key=lambda item: (
                        item[1].sequence,
                        item[1].created_at,
                        item[1].job_id,
                    ),
                )
            )
            size = 1
            profile = fixed_profile(gpu.gpu_id, size)
            if profile is None:
                reject("no_profile")
                return None
            effective_memory = gpu.memory_free_gb
            if gpu.free_at > observed_at + EPSILON and gpu.memory_free_gb != math.inf:
                effective_memory += gpu.reserved_memory_gb
            if profile.memory_gb > effective_memory + EPSILON:
                reject("memory")
                return None
            migration_count = 0
            migration_seconds = 0.0
            max_ready_at = observed_at
            if estimates is not None:
                for state, _job in ordered:
                    estimate = estimates.get(state.session_id)
                    if estimate is None:
                        continue
                    if estimate.required:
                        migration_count += 1
                        migration_seconds += estimate.cost_seconds
                    max_ready_at = max(max_ready_at, estimate.ready_at)
            start_at = max(observed_at, gpu.free_at, max_ready_at)
            finish_at = start_at + profile.latency_seconds
            feasible[size] += 1
            return DispatchCandidate(
                session_ids=tuple(state.session_id for state, _job in ordered),
                job_ids=tuple(job.job_id for _state, job in ordered),
                gpu_id=gpu.gpu_id,
                fidelity=profile.fidelity,
                profile=profile,
                start_at=start_at,
                finish_at=finish_at,
                migration_count=migration_count,
                migration_seconds=migration_seconds,
                score=0.0,
                projected_slack={},
                projected_quality={},
                snapshot_epoch=snapshot_epoch,
                session_versions={state.session_id: state.state_version for state, _job in ordered},
                gpu_version=gpu.version,
                policy_name=policy_name,
            )

        gpu_by_id = {gpu.gpu_id: gpu for gpu in available_gpus}

        # First honor a positive min--max placement move.  Only one moved
        # session is exposed per candidate; the next search observes the new
        # ownership epoch and can dispatch the next singleton local job.
        if self.config.migration_enabled and allow_migrations:
            move_options: list[tuple[int, SessionSchedulingState, ActionJob, str, MigrationEstimate]] = []
            for session_id, (state, job) in ready_by_id.items():
                target_id = desired.get(session_id)
                if (
                    target_id is None
                    or target_id == state.owner_gpu
                    or target_id not in gpu_by_id
                    or session_id in blocked
                ):
                    continue
                estimate = self._estimate_migration(
                    state, target_gpu=gpu_by_id[target_id], now=observed_at
                )
                if not estimate.required:
                    continue
                move_options.append((job.sequence, state, job, target_id, estimate))
            if move_options:
                _sequence, moved_state, moved_job, target_id, moved_estimate = min(
                    move_options,
                    key=lambda item: (item[0], item[1].session_id),
                )
                candidate = make_candidate(
                    [(moved_state, moved_job)],
                    gpu_by_id[target_id],
                    {moved_state.session_id: moved_estimate},
                )
                if candidate is not None:
                    return record_and_wait(candidate)

        # No beneficial move: preserve residency and emulate the source
        # worker's FCFS-over-round-robin singleton execution. Select the oldest
        # local action on each GPU, then choose the earliest available GPU.
        # There is intentionally no cross-session compatibility group.
        local_options: list[tuple[tuple[float, int, int, str], DispatchCandidate]] = []
        for gpu in available_gpus:
            local = [
                (state, job)
                for state, job in ready
                if state.owner_gpu == gpu.gpu_id
            ]
            # A worker may have no local ready job while another worker does
            # (especially during session close or an in-flight migration).
            # TurboServe is a singleton placement policy; an empty local set
            # is simply not a candidate, not a scheduler error.
            if not local:
                continue
            local.sort(key=lambda item: (item[1].sequence, item[0].session_id))
            state, job = local[0]
            compatible[1] += 1
            candidate = make_candidate([(state, job)], gpu)
            if candidate is None:
                continue
            priority = (candidate.start_at, job.sequence, 0, gpu.gpu_id)
            local_options.append((priority, candidate))
        if local_options:
            local_options.sort(key=lambda item: item[0])
            return record_and_wait(local_options[0][1])
        return record_and_wait(None)

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
            snapshot_states = tuple(
                replace(state, pending_action_queue=deque(state.pending_action_queue))
                for state in live_states
            )
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
            pending_action_count=sum(
                state.pending_action_count for state in snapshot_states if not state.departed
            ),
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
        """Return signed playback utility; a negative value is unmet credit."""
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

    def plan_rehoming(
        self,
        *,
        now: float | None = None,
        blocked_session_ids: Iterable[str] = (),
    ) -> PlacementDecision | None:
        """Return one deadline-improving re-homing move for Motivation.

        Placement is deliberately separate from compute selection: this phase
        only decides *which GPU owns a session*.  It must not act as a generic
        load balancer, because a short-lived ``free_at`` imbalance is not a
        user-visible reason to move model state.  A move is therefore admitted
        only for a ready next job whose local finish would consume its current
        playout slack, and only when the target's predicted finish is earlier.

        The B1 latency used here is a placement-time estimate, not a runtime
        profile choice.  The following local phase still chooses the actual
        batch size and fidelity for the target GPU.
        """
        if self.policy_name != "motivation" or not self.config.migration_enabled:
            return None
        observed_at = self._clock() if now is None else now
        blocked = {str(value) for value in blocked_session_ids}
        with self._lock:
            observed_at = self._advance_to(observed_at)
            gpus = tuple(gpu for gpu in self._gpus.values() if gpu.available)
            if len(gpus) < 2:
                return None
            gpu_by_id = {gpu.gpu_id: gpu for gpu in gpus}
            capacity = max(1, self.config.load_balance_capacity_per_worker)
            resident_counts = {
                gpu.gpu_id: sum(
                    1
                    for state in self._sessions.values()
                    if not state.departed
                    and state.migration_target_gpu is None
                    and state.owner_gpu == gpu.gpu_id
                )
                for gpu in gpus
            }
            # A migration marks its source session with the target GPU before
            # the asynchronous transfer commits.  Reserve that target slot in
            # the placement snapshot as well; otherwise several scheduling
            # callbacks can independently observe the same free slot and send
            # more than ``capacity`` sessions to one worker.
            pending_target_counts = {
                gpu.gpu_id: sum(
                    1
                    for state in self._sessions.values()
                    if not state.departed
                    and state.migration_target_gpu == gpu.gpu_id
                )
                for gpu in gpus
            }

            # Placement needs only a neutral service-time estimate.  Do not
            # choose a quality point or a batch here; the local compute phase
            # owns that decision after the session has an owner.
            profile_cache: dict[str, MotivationProfile | None] = {}

            def placement_profile(gpu_id: str) -> MotivationProfile | None:
                if gpu_id not in profile_cache:
                    rows = tuple(
                        self.profile_provider.profiles_for(batch_size=1, gpu_id=gpu_id)
                    )
                    profile_cache[gpu_id] = (
                        min(rows, key=lambda row: (row.latency_seconds, row.fidelity))
                        if rows
                        else None
                    )
                return profile_cache[gpu_id]

            # Evaluate the next concrete job on its current owner.  A session
            # with no ready work is not a placement candidate: there is no
            # deadline that migration could improve.
            options: list[
                tuple[
                    tuple[float, float, int, str, str],
                    SessionSchedulingState,
                    GpuSchedulingState,
                    GpuSchedulingState,
                    MigrationEstimate,
                ]
            ] = []
            for state in self._sessions.values():
                if (
                    state.departed
                    or state.in_flight is not None
                    or state.migration_target_gpu is not None
                    or state.session_id in blocked
                    or not state.playback_active
                ):
                    continue
                job = state.ready_job(include_idle=self.config.include_idle_jobs)
                if job is None:
                    continue
                source = gpu_by_id.get(state.owner_gpu)
                if source is None:
                    continue
                source_profile = placement_profile(source.gpu_id)
                if source_profile is None:
                    continue
                source_finish = max(observed_at, source.free_at) + source_profile.latency_seconds
                deadline = observed_at + max(0.0, state.slack_seconds)
                if source_finish <= deadline + EPSILON:
                    continue

                for target in gpus:
                    if target.gpu_id == source.gpu_id:
                        continue
                    if (
                        resident_counts[target.gpu_id]
                        + pending_target_counts[target.gpu_id]
                        >= capacity
                    ):
                        continue
                    target_profile = placement_profile(target.gpu_id)
                    if target_profile is None:
                        continue
                    estimate = self._estimate_migration(
                        state, target_gpu=target, now=observed_at
                    )
                    if not estimate.required:
                        continue
                    target_finish = (
                        max(observed_at, target.free_at, estimate.ready_at)
                        + target_profile.latency_seconds
                    )
                    # This is the only rehoming gain test: moving must make
                    # this session's next job finish strictly earlier.  A
                    # lower aggregate GPU load by itself is not sufficient.
                    if target_finish >= source_finish - EPSILON:
                        continue
                    options.append(
                        (
                            (
                                target_finish,
                                source_finish,
                                job.sequence,
                                state.session_id,
                                target.gpu_id,
                            ),
                            state,
                            source,
                            target,
                            estimate,
                        )
                    )

            if not options:
                return None
            _, moved, source, target, estimate = min(options, key=lambda item: item[0])
            return PlacementDecision(
                session_id=moved.session_id,
                source_gpu=source.gpu_id,
                target_gpu=target.gpu_id,
                estimated_ready_at=estimate.ready_at,
                migration_seconds=estimate.cost_seconds,
                snapshot_epoch=self._epoch,
                session_version=moved.state_version,
                target_gpu_version=target.version,
            )

    def placement_candidate(self, decision: PlacementDecision) -> DispatchCandidate:
        """Convert a placement decision to the shared migration path."""
        return DispatchCandidate(
            session_ids=(decision.session_id,),
            job_ids=(),
            gpu_id=decision.target_gpu,
            fidelity=None,
            profile=None,
            start_at=decision.estimated_ready_at,
            finish_at=decision.estimated_ready_at,
            migration_count=1,
            migration_seconds=decision.migration_seconds,
            score=0.0,
            projected_slack={},
            projected_quality={},
            snapshot_epoch=decision.snapshot_epoch,
            session_versions={decision.session_id: decision.session_version},
            gpu_version=decision.target_gpu_version,
            policy_name="motivation-placement",
        )

    def find_best_local(
        self,
        gpu_id: str,
        *,
        now: float | None = None,
        include_wait: bool = False,
        exclude_session_ids: Iterable[str] = (),
    ) -> DispatchCandidate | None:
        """Select the best Motivation compute candidate on one GPU only."""
        with self._lock:
            gpu = self._gpus.get(gpu_id)
            if gpu is None:
                raise KeyError(f"unknown GPU {gpu_id!r}")
            eligible = {
                state.session_id
                for state in self._sessions.values()
                if not state.departed
                and state.owner_gpu == gpu_id
                and state.migration_target_gpu is None
            }
        eligible.difference_update(str(value) for value in exclude_session_ids)
        if not eligible:
            return None
        return self._find_best_motivation(
            now=now,
            gpu_states=(gpu,),
            include_wait=include_wait,
            allow_migrations=False,
            eligible_session_ids=eligible,
            policy_name="motivation-local",
        )

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
        eligible_session_ids: Iterable[str] | None = None,
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
        eligible = None if eligible_session_ids is None else {str(value) for value in eligible_session_ids}

        # Advance and copy the policy state while holding the lock, then do the
        # expensive Cartesian-product search without it.  Event callbacks can
        # therefore register sessions and publish completions while a large
        # ready set is being scored; ``snapshot_epoch`` makes the result fail
        # closed at reservation time if anything changed meanwhile.
        with self._lock:
            observed_at = self._advance_to(observed_at)
            live_states = tuple(self._sessions.values())
            snapshot_states = tuple(
                replace(state, pending_action_queue=deque(state.pending_action_queue))
                for state in live_states
            )
            snapshot_by_id = {state.session_id: state for state in snapshot_states}
            all_ready = tuple(
                (snapshot_by_id[state.session_id], job)
                for state, job in self._ready_jobs()
            )
            ready = tuple(
                item
                for item in all_ready
                if item[0].session_id not in excluded
                and (eligible is None or item[0].session_id in eligible)
            )
            gpus = tuple(gpu_states) if gpu_states is not None else tuple(self._gpus.values())
            snapshot_epoch = self._epoch

        active_states = tuple(state for state in snapshot_states if not state.departed)
        available_gpus = tuple(gpu for gpu in gpus if gpu.available)
        # ``free_at`` captures queued compute, while resident-session counts
        # capture a placement imbalance that is invisible once a worker has
        # briefly gone idle.  Normalize occupancy by the configured per-worker
        # capacity so the two scheduler-visible signals have comparable scale.
        occupancy_capacity = max(1, self.config.load_balance_capacity_per_worker)
        resident_counts = {
            gpu.gpu_id: sum(
                1 for state in active_states if state.owner_gpu == gpu.gpu_id
            )
            for gpu in available_gpus
        }
        timeline_loads = {
            gpu.gpu_id: max(0.0, gpu.free_at - observed_at)
            for gpu in available_gpus
        }
        base_loads = {
            gpu_id: timeline_loads[gpu_id]
            + resident_counts[gpu_id] / occupancy_capacity
            for gpu_id in timeline_loads
        }
        base_load_spread = _load_spread(base_loads)
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
            # Generate candidates inside each compatibility group.  The old
            # implementation materialized C(n, size) global combinations and
            # then discarded incompatible tuples, which became the dominant
            # control-plane cost for high-concurrency traces.  Sorting keeps
            # the old lexicographic tie order without recreating that product.
            group_combinations = [
                indexes
                for group in compatibility_groups
                if len(group) >= size
                for indexes in itertools.combinations(group, size)
            ]
            compatible_combinations[size] = tuple(sorted(group_combinations))

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
                with self._lock:
                    profiles = self._profile_frontier_cache.get(cache_key)
                if profiles is None:
                    raw_profiles = tuple(
                        self.profile_provider.profiles_for(batch_size=size, gpu_id=gpu.gpu_id)
                    )
                    profiles = _pareto_profile_frontier(raw_profiles)
                    with self._lock:
                        self._profile_frontier_cache[cache_key] = profiles
                profile_cache[cache_key] = profiles
            return profiles

        gpu_by_id = {gpu.gpu_id: gpu for gpu in gpus}
        owner_b1_latency: dict[str, float | None] = {}
        batch_b1_latency: dict[tuple[str, str], float | None] = {}

        def b1_latency_for_profile(
            gpu: GpuSchedulingState, profile: MotivationProfile
        ) -> float | None:
            """Return the measured B1 latency in the profile's family."""
            _, family = profile_batch_family(profile.fidelity)
            cache_key = (gpu.gpu_id, family)
            if cache_key not in batch_b1_latency:
                rows = profiles_for(gpu, 1)
                matching = tuple(
                    row
                    for row in rows
                    if profile_batch_family(row.fidelity)[1] == family
                )
                batch_b1_latency[cache_key] = min(
                    (row.latency_seconds for row in matching), default=None
                )
            return batch_b1_latency[cache_key]

        def owner_finish_at(state: SessionSchedulingState, ordinal: int = 1) -> float:
            """Predict the next singleton finish on a session's owner.

            This is only the migration admissibility baseline.  The actual
            candidate still uses the selected target batch/profile and the
            full joint score.  A migration must beat this local continuation
            so load-balance reward cannot justify state movement by itself.
            """
            owner_gpu = gpu_by_id.get(state.owner_gpu)
            if owner_gpu is None or not owner_gpu.available:
                # A failed/unavailable owner has no viable local
                # continuation; migration is the recovery option.
                return math.inf
            if state.owner_gpu not in owner_b1_latency:
                rows = profiles_for(owner_gpu, 1)
                owner_b1_latency[state.owner_gpu] = (
                    min((row.latency_seconds for row in rows), default=None)
                )
            latency = owner_b1_latency[state.owner_gpu]
            if latency is None:
                return math.inf
            return max(observed_at, owner_gpu.free_at) + latency * max(1, ordinal)

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
        lambda_load_balance = self.config.lambda_load_balance
        fairness_delta = self.config.fairness_delta
        active_quality_sum = sum(state.quality_ema for state in active_states)
        active_slacks = tuple(state.slack_seconds for state in active_states)
        active_playback = tuple(state.playback_active for state in active_states)
        # A local compute decision occupies only the GPUs present in this
        # search. Sessions resident on another GPU continue concurrently and
        # must not be charged this candidate's latency. A global search still
        # sees every active session; ``find_best_local`` supplies one GPU and
        # therefore obtains the separable per-worker objective used by the
        # second scheduling stage.
        objective_gpu_ids = {gpu.gpu_id for gpu in available_gpus}
        objective_mask = tuple(
            state.owner_gpu in objective_gpu_ids for state in active_states
        )
        playback_slacks = sorted(
            slack
            for slack, playback, in_scope in zip(
                active_slacks,
                active_playback,
                objective_mask,
                strict=True,
            )
            if playback and in_scope
        )
        playback_slack_prefix = [0.0]
        for slack in playback_slacks:
            playback_slack_prefix.append(playback_slack_prefix[-1] + slack)
        paused_utility = sum(
            slack if slack < utility_cap else utility_cap
            for slack, playback, in_scope in zip(
                active_slacks,
                active_playback,
                objective_mask,
                strict=True,
            )
            if not playback and in_scope
        )

        def baseline_utility(duration: float) -> float:
            """Return utility for the resources in this search scope."""
            threshold = utility_cap + duration
            split = bisect.bisect_left(playback_slacks, threshold)
            return (
                playback_slack_prefix[split]
                - split * duration
                + (len(playback_slacks) - split) * utility_cap
                + paused_utility
            )

        def load_balance_reward(
            *,
            target_gpu_id: str,
            selected_indexes: Sequence[int],
            finish_at: float,
        ) -> float:
            """Return the reduction in projected load spread for a candidate."""
            if len(available_gpus) <= 1:
                return 0.0
            projected_counts = dict(resident_counts)
            for index in selected_indexes:
                source_gpu_id = ready_states[index].owner_gpu
                if source_gpu_id == target_gpu_id:
                    continue
                if source_gpu_id in projected_counts:
                    projected_counts[source_gpu_id] = max(
                        0, projected_counts[source_gpu_id] - 1
                    )
                if target_gpu_id in projected_counts:
                    projected_counts[target_gpu_id] += 1
            projected_timeline = dict(timeline_loads)
            projected_timeline[target_gpu_id] = max(
                projected_timeline.get(target_gpu_id, 0.0),
                max(0.0, finish_at - observed_at),
            )
            projected_loads = {
                gpu_id: projected_timeline[gpu_id]
                + projected_counts[gpu_id] / occupancy_capacity
                for gpu_id in projected_timeline
            }
            return base_load_spread - _load_spread(projected_loads)

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
                    self._estimate_migration(state, target_gpu=gpu, now=observed_at)
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
                                + lambda_quality * best_quality_sum
                                + lambda_load_balance * base_load_spread,
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
                            if size > 1:
                                b1_latency = b1_latency_for_profile(gpu, profile)
                                if (
                                    b1_latency is None
                                    or profile.latency_seconds
                                    >= size * b1_latency - EPSILON
                                ):
                                    reject("batch_no_gain")
                                    continue
                            # A profile-level optimistic bound avoids walking
                            # every member combination once an earlier
                            # candidate is already strictly better.  The bound
                            # uses the earliest possible start, grants every
                            # selected member the full output increment, and
                            # ignores migration/fairness penalties.  The
                            # maximum possible load-balancing reward is added
                            # to keep pruning optimistic, so it cannot remove
                            # a true winner.  Candidate ranking is based on
                            # the same signed-slack objective for every batch
                            # size; negative slack is a deficit, not a batch
                            # admissibility condition.
                            if candidates_best_key is not None and (
                                optimistic_bound < candidates_best_key[0] - EPSILON
                            ):
                                continue
                            finish_at = start_at + profile.latency_seconds
                            # Placement is part of the same decision as
                            # batching and profile selection.  A remote
                            # candidate is admissible only if its selected
                            # batch/profile finishes at least one moved item
                            # earlier than that item's serialized continuation
                            # on its owner with the fastest measured B1 point.
                            # This is a strict gain test, not a generic
                            # load-balancing trigger.
                            remote_indexes = tuple(
                                index
                                for index in indexes
                                if gpu_estimates[index].required
                            )
                            if remote_indexes:
                                local_finishes: list[float] = []
                                by_owner: dict[str, list[int]] = {}
                                for index in remote_indexes:
                                    by_owner.setdefault(
                                        ready_states[index].owner_gpu, []
                                    ).append(index)
                                for owner_indexes in by_owner.values():
                                    owner_indexes.sort(
                                        key=lambda index: (
                                            ready_jobs[index].sequence,
                                            ready_states[index].session_id,
                                        )
                                    )
                                    local_finishes.extend(
                                        owner_finish_at(state, ordinal)
                                        for ordinal, state in enumerate(
                                            (ready_states[index] for index in owner_indexes),
                                            start=1,
                                        )
                                    )
                                # A batch migration is useful when at least
                                # one moved item finishes earlier than its
                                # serialized local continuation.  Comparing
                                # against only the first local B1 would reject
                                # valid B2/B3 gains from one source GPU.
                                if not any(
                                    finish_at < local_finish - EPSILON
                                    for local_finish in local_finishes
                                ):
                                    reject("migration_no_gain")
                                    continue
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
                                before_utility = self._utility(before, utility_cap)
                                after_utility = self._utility(after, utility_cap)
                                score += after_utility - before_utility
                            score += lambda_quality * selected_quality_sum
                            # Penalize both transfer count and predicted transfer time.
                            score -= lambda_migration * (migration_count + migration_seconds)
                            score += lambda_load_balance * load_balance_reward(
                                target_gpu_id=gpu.gpu_id,
                                selected_indexes=indexes,
                                finish_at=finish_at,
                            )
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
            pending_action_count=sum(
                state.pending_action_count for state in snapshot_states if not state.departed
            ),
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
            snapshot = dict(self._diagnostics.snapshot())
        except Exception:
            snapshot = {}
        with self._lock:
            effective = (
                self._action_jobs_released
                - self._action_jobs_superseded_pending
                - self._action_jobs_dropped_capacity
            )
            snapshot["action_lifecycle"] = {
                "released": self._action_jobs_released,
                "superseded_before_dispatch": self._action_jobs_superseded_pending,
                "dropped_pending_capacity": self._action_jobs_dropped_capacity,
                "effective": effective,
                "completed": self._action_jobs_completed,
                "dropped_on_departure": self._action_jobs_dropped_on_departure,
                "definition": (
                    "effective = released - superseded_before_dispatch; a newer action never "
                    "supersedes an action that is already in flight"
                ),
            }
            if self.policy_name == "slackserve":
                snapshot["slackserve_capabilities"] = {
                    "service_credit": True,
                    "bmpr": True,
                    "rehoming": bool(self.config.slackserve_rehoming_enabled),
                    "elastic_sp": False,
                    "elastic_sp_reason": (
                        "ABot has no single-session SP=2 execution primitive; "
                        "cross-session B2 batching is not Elastic SP"
                    ),
                }
        return snapshot

    def validate(self, candidate: DispatchCandidate, *, now: float | None = None) -> bool:
        """Check that an asynchronously searched candidate is still current."""
        observed_at = self._clock() if now is None else now
        with self._lock:
            observed_at = self._advance_to(observed_at)
            if candidate.wait:
                return candidate.snapshot_epoch == self._epoch
            if candidate.gpu_id is None:
                return False
            # Placement decisions intentionally have no profile: they only
            # validate ownership/version and enter the async migration path.
            # Compute candidates still require a measured profile.
            if candidate.profile is None and not candidate.migration_count:
                return False
            # A new ready job can change the globally optimal batch even when
            # none of the candidate's selected sessions changed.
            if candidate.snapshot_epoch != self._epoch:
                return False
            gpu = self._gpus.get(candidate.gpu_id)
            if gpu is None or not gpu.available or gpu.version != candidate.gpu_version:
                return False
            if candidate.migration_count:
                if not candidate.session_ids:
                    return False
                for session_id in candidate.session_ids:
                    state = self._sessions.get(session_id)
                    expected_version = candidate.session_versions.get(session_id)
                    if (
                        state is None
                        or state.departed
                        or state.in_flight is not None
                        or expected_version is None
                        or state.state_version != expected_version
                        or state.owner_gpu == candidate.gpu_id
                        or (
                            state.migration_target_gpu is not None
                            and state.migration_target_gpu != candidate.gpu_id
                        )
                    ):
                        return False
                # A migration-only candidate does not own a compute lease.
                # Its source/target state is fully checked above.
                if candidate.profile is None:
                    return True
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
            if self.policy_name in {"turboserve", "slackserve"} and candidate.batch_size != 1:
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
                node_id=gpu.node_id,
                endpoint_address=gpu.endpoint_address,
                endpoint_port=gpu.endpoint_port,
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
                dropped = state.rollback_dispatched(job)
                if dropped is not None:
                    self._action_jobs_dropped_capacity += 1
                state.migration_target_gpu = None
                jobs.append(job)
            gpu = self._gpus[candidate.gpu_id]
            released_memory = gpu.reserved_memory_gb
            self._gpus[candidate.gpu_id] = GpuSchedulingState(
                gpu_id=gpu.gpu_id,
                node_id=gpu.node_id,
                endpoint_address=gpu.endpoint_address,
                endpoint_port=gpu.endpoint_port,
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
                if job.kind == "action":
                    self._action_jobs_completed += 1
            gpu = self._gpus[candidate.gpu_id]
            # The child may have already reported physical model completion,
            # which releases the GPU reservation before transport/publisher
            # output reaches the parent.  Do not add the same peak twice.
            released_memory = gpu.reserved_memory_gb
            self._gpus[candidate.gpu_id] = GpuSchedulingState(
                gpu_id=gpu.gpu_id,
                node_id=gpu.node_id,
                endpoint_address=gpu.endpoint_address,
                endpoint_port=gpu.endpoint_port,
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
                node_id=gpu.node_id,
                endpoint_address=gpu.endpoint_address,
                endpoint_port=gpu.endpoint_port,
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
