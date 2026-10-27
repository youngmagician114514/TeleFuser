"""Metric-directed pool sizing and independent actor-local step batching."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Mapping, Sequence

from telefuser.pipelines.abot_world.fidelity import ABotWorldFidelity

from .motivation_scheduler import MotivationProfile
from .phase_batching import PhaseStepGroup, plan_phase_steps


@dataclass
class PhaseWorkerLoad:
    rank: int
    slots: int
    kv_budget_bytes: int
    bytes_per_frame: int
    cross_bytes: int
    timings: dict[str, dict[int, float]]
    role: str = "unassigned"
    reserved_bytes: int = 0
    outstanding_seconds: float = 0.0
    queued_jobs: int = 0
    sessions: set[str] = field(default_factory=set)
    memory_stats: dict[str, int] = field(default_factory=dict)

    def seconds(self, phase: str, window: int, rows: int) -> float:
        samples = self.timings[f"{phase}_w{window}"]
        sizes = sorted(int(size) for size in samples)
        values = {int(size): float(value) for size, value in samples.items()}
        if rows <= sizes[0]:
            return values[sizes[0]] * rows / sizes[0]
        for lower, upper in zip(sizes, sizes[1:]):
            if rows <= upper:
                return values[lower] + (rows - lower) / (upper - lower) * max(0.0, values[upper] - values[lower])
        return values[sizes[-1]] * rows / sizes[-1]

    def session_bytes(self, window: int) -> int:
        # Nonexecuting sessions retain a single canonical row. The shared-Job
        # execution arena and workspace are excluded from kv_budget_bytes.
        return window * self.bytes_per_frame + self.cross_bytes

    def observe(self, phase: str, window: int, rows: int, seconds: float, rate: float = 0.2) -> None:
        samples = self.timings.setdefault(f"{phase}_w{window}", {})
        previous = samples.get(rows, seconds)
        samples[rows] = previous + rate * (seconds - previous)


@dataclass(frozen=True)
class PhasePartition:
    denoise_ranks: tuple[int, ...]
    refresh_ranks: tuple[int, ...]
    profile: str
    capacity_jobs_per_second: float
    predicted_quality_cpr: float
    resident_sessions: int = 0


@dataclass(frozen=True)
class PhaseAdmission:
    denoise_rank: int
    refresh_rank: int
    fidelity: str
    quality: float
    predicted_completion_seconds: float
    denoise_seconds: float
    refresh_seconds: float
    reserved_bytes: int
    objective: float
    slack_seconds: float


class ABotPhasePlanner:
    """Use measured capacity for pools, then deadline/quality cost per Job.

    The pool objective is quality-weighted sustainable frame supply under the
    declared workload's request rate and residency. The per-Job objective is
    the existing integrated playback debt plus quality loss. Partitioning is
    done at startup; phase queues may relocate clean state at phase boundaries.
    """

    def __init__(self, profiles: Sequence[MotivationProfile], *, lambda_quality: float = 2.0) -> None:
        self.profiles = tuple(profile for profile in profiles if profile.batch_size == 1)
        if not self.profiles:
            raise ValueError("Phase planning requires B1 quality reference rows")
        self.lambda_quality = lambda_quality
        self.fidelities = {p.fidelity: ABotWorldFidelity.from_profile_name(p.fidelity) for p in self.profiles}
        self.transfer_seconds = {"latent": 0.0, "kv": 0.0}
        self.decode_seconds = 0.0

    def partition(
        self,
        loads: Mapping[int, PhaseWorkerLoad],
        *,
        resident_sessions: int,
        request_rate: float,
    ) -> PhasePartition:
        ranks = tuple(sorted(loads))
        if len(ranks) < 2:
            raise ValueError("Independent denoise/refresh pools require at least two GPUs")
        candidates = []
        # Small homogeneous deployments can enumerate every disjoint split.
        # At larger scale use ordered cuts, avoiding an exponential solve.
        for count in range(1, len(ranks)):
            refresh_sets = combinations(ranks, count) if len(ranks) <= 8 else (ranks[-count:],)
            for refresh in refresh_sets:
                denoise = tuple(rank for rank in ranks if rank not in refresh)
                for profile in self.profiles:
                    fidelity = self.fidelities[profile.fidelity]
                    if any(
                        sum(loads[r].kv_budget_bytes // loads[r].session_bytes(fidelity.local_attn_size) for r in pool)
                        < resident_sessions
                        for pool in (denoise, refresh)
                    ):
                        continue
                    capacities = []
                    for phase, pool, forwards in (
                        ("denoise", denoise, fidelity.denoise_steps),
                        ("refresh", refresh, 1),
                    ):
                        capacity = sum(
                            (loads[r].slots // 3)
                            / (forwards * loads[r].seconds(phase, fidelity.local_attn_size, loads[r].slots // 3 * 3))
                            for r in pool
                        )
                        capacities.append(capacity)
                    capacity = min(capacities)
                    score = profile.quality * min(1.0, capacity / max(request_rate, 1e-9))
                    decision = PhasePartition(
                        denoise, tuple(refresh), profile.fidelity, capacity, score, resident_sessions
                    )
                    candidates.append((score, profile.quality, capacity, -len(refresh), decision))
        if not candidates:
            raise RuntimeError("No D/R partition can retain this workload at any profiled quality")
        return max(candidates, key=lambda item: item[:4])[-1]

    def admit(
        self,
        state: Any,
        loads: Mapping[int, PhaseWorkerLoad],
        partition: PhasePartition,
        *,
        owners: tuple[int, int] | None = None,
        retained_bytes: int = 0,
        action_age_seconds: float | None = None,
    ) -> PhaseAdmission | None:
        best = None
        pairs = (owners,) if owners else tuple((d, r) for d in partition.denoise_ranks for r in partition.refresh_ranks)
        for d, r in pairs:
            for profile in self.profiles:
                fidelity = self.fidelities[profile.fidelity]
                window = fidelity.local_attn_size
                reserved = max(retained_bytes, loads[d].session_bytes(window), loads[r].session_bytes(window))
                if any(
                    loads[rank].reserved_bytes + reserved - retained_bytes > loads[rank].kv_budget_bytes
                    for rank in (d, r)
                ):
                    continue
                # High-quality early arrivals must leave space for the rest
                # of the declared resident cohort at the minimum window.
                # Otherwise a feasible partition could dead-end at admission.
                minimum_window = min(value.local_attn_size for value in self.fidelities.values())
                capacity_ok = True
                pools = ((partition.denoise_ranks, d), (partition.refresh_ranks, r))
                # An existing row can always advance without increasing its
                # residency. Future-admission headroom must not stop all
                # current work after a placement changes free-space packing.
                for pool, selected in pools if owners is None or reserved > retained_bytes else ():
                    resident = len(set().union(*(loads[rank].sessions for rank in pool))) + int(owners is None)
                    future_capacity = sum(
                        (
                            loads[rank].kv_budget_bytes
                            - loads[rank].reserved_bytes
                            - (reserved - retained_bytes if rank == selected else 0)
                        )
                        // loads[rank].session_bytes(minimum_window)
                        for rank in pool
                    )
                    if future_capacity < max(0, partition.resident_sessions - resident):
                        capacity_ok = False
                        break
                if not capacity_ok:
                    continue
                # Use observed amortized cost at the worker's current
                # occupancy, not a static chunk latency or a forced max batch.
                d_rows = min(loads[d].slots // 3 * 3, 3 * (loads[d].queued_jobs + 1))
                r_rows = min(loads[r].slots // 3 * 3, 3 * (loads[r].queued_jobs + 1))
                denoise = fidelity.denoise_steps * loads[d].seconds("denoise", window, d_rows) * 3 / d_rows
                refresh = loads[r].seconds("refresh", window, r_rows) * 3 / r_rows
                d_queue, r_queue = loads[d].outstanding_seconds, loads[r].outstanding_seconds
                if owners is None:
                    # Placement plans a resident cohort, including keys that
                    # are idle at arrival. Evaluate one next transition per
                    # resident key on each measured GPU timeline; otherwise
                    # empty queues place every idle arrival on the slightly
                    # fastest worker and affinity later strands the others.
                    d_cohort = min(loads[d].slots // 3, len(loads[d].sessions) + 1)
                    r_cohort = min(loads[r].slots // 3, len(loads[r].sessions) + 1)
                    d_service = fidelity.denoise_steps * loads[d].seconds("denoise", window, d_cohort * 3) / d_cohort
                    r_service = loads[r].seconds("refresh", window, r_cohort * 3) / r_cohort
                    d_queue = max(d_queue, len(loads[d].sessions) * d_service)
                    r_queue = max(r_queue, len(loads[r].sessions) * r_service)
                finish = (
                    max(d_queue + denoise + self.transfer_seconds["latent"], r_queue)
                    + refresh
                    + self.transfer_seconds["kv"]
                    + self.decode_seconds
                )
                slack = state.slack_seconds
                if action_age_seconds is not None:
                    # Pending controls have their own delivery deadline.
                    # A video surplus must not hide an aging action FIFO and
                    # select expensive quality until that FIFO overflows.
                    slack = min(slack, profile.output_seconds - action_age_seconds)
                late = max(0.0, finish - max(0.0, slack))
                debt = max(0.0, -slack)
                loss = late * (debt + 0.5 * late) / profile.output_seconds
                loss += self.lambda_quality * (1 - profile.quality) * profile.output_seconds
                choice = PhaseAdmission(
                    d, r, profile.fidelity, profile.quality, finish, denoise, refresh, reserved, loss, slack
                )
                local_finish = d_queue + denoise if owners is None else 0.0
                order = (loss, finish, local_finish, max(loads[d].reserved_bytes, loads[r].reserved_bytes), d, r)
                if best is None or order < best[0]:
                    best = (order, choice)
        return best[1] if best else None


def select_phase_batch(runtime: Any, metadata: Mapping[str, dict[str, Any]], load: PhaseWorkerLoad) -> tuple[str, ...]:
    """Replan one physical step against a feasible serial clearing schedule.

    A dependency group holds sibling latent-step rows, not a reservation for
    an entire Job. Different diffusion steps and W remain batch-compatible.
    """
    tasks = {sid: task for sid, task in runtime._requests.items() if sid in metadata}
    if not tasks:
        return ()
    groups = [
        PhaseStepGroup(
            sid,
            task.full_shape[2],
            1 if task.refreshing else len(task.timesteps) - task.step_index,
            task.window[0],
            task.latent_batch_key(),
            metadata[sid]["deadline"],
        )
        for sid, task in tasks.items()
    ]
    phase = "refresh" if next(iter(tasks.values())).refreshing else "denoise"
    return plan_phase_steps(
        groups, max_rows=runtime.max_batch_size, seconds=lambda window, rows: load.seconds(phase, window, rows)
    ).sessions
