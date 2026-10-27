"""Joint batch, fidelity and initial placement planning for ABot latent steps.

Minimize predicted playback-debt area plus quality loss using online wave
measurements. The offline table supplies Q only. Existing owners and sampling
paths stay fixed; sibling latents advance as one attention dependency group.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass, replace
from itertools import combinations_with_replacement
from math import prod
from typing import TYPE_CHECKING, Mapping, Sequence

from telefuser.pipelines.abot_world.fidelity import ABotWorldFidelity

from .motivation_scheduler import MotivationProfile, MotivationSchedulerConfig, SessionSchedulingState

if TYPE_CHECKING:
    from telefuser.pipelines.abot_world.interactive import ABotWorldInteractiveSession
    from telefuser.pipelines.abot_world.latent_step_runtime import ABotWorldLatentStepRuntime, LatentStepKey


@dataclass(frozen=True)
class LatentStepChoice:
    fidelity: ABotWorldFidelity
    quality: float
    predicted_seconds: float
    owners: tuple[int, ...]
    refresh_owners: tuple[int, ...] = ()


@dataclass(frozen=True)
class LatentStepPlan:
    session_ids: tuple[str, ...]
    admissions: Mapping[str, LatentStepChoice]
    predicted_wave_seconds: float
    objective: float


@dataclass(frozen=True)
class _Work:
    session_id: str
    owners: tuple[int, ...]
    window: int
    steps: int
    quality: float
    output_seconds: float
    compatibility: tuple
    appending: bool
    fidelity: ABotWorldFidelity | None = None
    future_jobs: int = 0
    full_steps: int = 0
    refresh_owners: tuple[int, ...] = ()

    @property
    def execution_owners(self) -> tuple[int, ...]:
        return self.refresh_owners if self.steps == 0 and self.refresh_owners else self.owners

    @property
    def phase(self) -> int:
        return 0 if self.steps == 0 else 1 if self.appending else 2


class ABotLatentStepPolicy:
    """Use a bounded incremental solve of one completion/quality objective.

    A weighted processor-sharing relaxation ranks candidates on parallel ranks;
    a feasible physical-wave rollout compares the resulting batch prefixes.
    This is an approximate planner, not an exhaustive global scheduling optimum.
    """

    def __init__(self, profiles: Sequence[MotivationProfile], config: MotivationSchedulerConfig) -> None:
        self.config = config
        self.profiles = tuple(profile for profile in profiles if profile.batch_size == 1)
        if not self.profiles:
            raise ValueError("Latent-step quality selection requires B1 reference profiles")
        self.fidelities = {
            profile.fidelity: ABotWorldFidelity.from_profile_name(profile.fidelity) for profile in self.profiles
        }
        self.step_work: dict[int, float] = {}
        self.refresh_work: dict[int, float] = {}
        self.batch_times: dict[tuple[int, bool], dict[int, float]] = {}
        self._batch_predictions: dict[tuple[int, int, bool], float] = {}
        self._wave_predictions: dict[tuple, float] = {}
        self.wave_times: OrderedDict[tuple, float] = OrderedDict()
        self.observations = 0

    def _batch_seconds(self, window: int, rows: int, *, refreshing: bool) -> float:
        """Interpolate measured batch sizes, rather than serializing batch rows."""
        key = (window, rows, refreshing)
        if key not in self._batch_predictions:
            self._batch_predictions[key] = self._interpolate_batch_seconds(window, rows, refreshing=refreshing)
        return self._batch_predictions[key]

    def _interpolate_batch_seconds(self, window: int, rows: int, *, refreshing: bool) -> float:
        samples = self.batch_times.get((window, refreshing), {})
        if not samples:
            return (self.refresh_work if refreshing else self.step_work)[window] * rows
        sizes = sorted(samples)
        if len(sizes) == 1:
            return samples[sizes[0]] * rows / sizes[0]
        # An interpolated curve cannot claim that adding rows shortens a wave.
        times = []
        for size in sizes:
            times.append(max(samples[size], times[-1] if times else 0.0))
        if rows <= sizes[0]:
            return times[0] * rows / sizes[0]
        for index in range(1, len(sizes)):
            if rows <= sizes[index]:
                fraction = (rows - sizes[index - 1]) / (sizes[index] - sizes[index - 1])
                return times[index - 1] + fraction * (times[index] - times[index - 1])
        # Beyond measured occupancy, preserve the last observed throughput;
        # a noisy local slope must not invent unmeasured batching speedups.
        return times[-1] * rows / sizes[-1]

    def _work_seconds(self, work: _Work, slots: int = 1) -> float:
        step = self._batch_seconds(work.window, slots, refreshing=False) / slots
        refresh = self._batch_seconds(work.window, slots, refreshing=True) / slots
        return step * work.steps + refresh

    def _loss(
        self, state: SessionSchedulingState, work: _Work, completion: float, *, output_offset: float = 0.0
    ) -> float:
        loss = self.config.lambda_quality * (1.0 - work.quality) * work.output_seconds
        if state.playback_active and not state.departed:
            slack = state.slack_seconds + output_offset
            late = max(0.0, completion - max(0.0, slack))
            debt = max(0.0, -slack)
            loss += late * (debt + 0.5 * late) / work.output_seconds
        return loss

    def _approximate_chain_loss(
        self, state: SessionSchedulingState, work: _Work, completion: float, slots: int
    ) -> float:
        loss = self._loss(state, work, completion)
        if work.future_jobs:
            step = self._batch_seconds(work.window, slots, refreshing=False) / slots
            refresh = self._batch_seconds(work.window, slots, refreshing=True) / slots
            interval = max(work.owners.count(rank) for rank in work.owners) * (step * work.full_steps + refresh)
            for index in range(1, work.future_jobs + 1):
                loss += self._loss(
                    state, work, completion + index * interval, output_offset=index * work.output_seconds
                )
        return loss

    @staticmethod
    def _layout_key(rows: Sequence[tuple[int, int, int]], ranks: int) -> tuple:
        counts = [Counter() for _ in range(ranks)]
        for owner, window, phase in rows:
            counts[owner][window, phase] += 1
        return tuple(tuple(sorted(count.items())) for count in counts)

    def _wave_seconds(self, wave: Sequence[_Work], ranks: int) -> float:
        rows = [(owner, work.window, work.phase) for work in wave for owner in work.execution_owners]
        key = self._layout_key(rows, ranks)
        return self._layout_seconds(key)

    def _layout_seconds(self, key: tuple) -> float:
        """Reuse one timing prediction for equal physical layouts in a solve."""
        if key in self.wave_times:
            return self.wave_times[key]
        if key in self._wave_predictions:
            return self._wave_predictions[key]
        load = []
        for entries in key:
            count = sum(count for _, count in entries)
            load.append(
                sum(
                    rows / count * self._batch_seconds(window, count, refreshing=phase == 0)
                    for (window, phase), rows in entries
                )
                if count
                else 0.0
            )
        # Measurements include NCCL and rank waiting; do not count an extra
        # bytes/bandwidth estimate on top of the same communication.
        result = max(load, default=0.0)
        self._wave_predictions[key] = result
        return result

    def _objective(
        self,
        works: Mapping[str, _Work],
        wave: Sequence[_Work],
        states: Mapping[str, SessionSchedulingState],
        ranks: int,
        slots: int,
    ) -> tuple[float, dict[str, float]]:
        duration = self._wave_seconds(wave, ranks) if wave else 0.0
        advancing = {work.session_id for work in wave}
        completion = dict.fromkeys(works, duration)
        queues = [[] for _ in range(ranks)]
        for key, work in works.items():
            steps = work.steps - int(key in advancing)
            if steps < 0:
                continue
            state = states[key]
            debt = max(0.0, -state.slack_seconds) if state.playback_active and not state.departed else 0.0
            weight = 1.0 + debt / work.output_seconds
            if work.refresh_owners:
                denoise_seconds = self._batch_seconds(work.window, slots, refreshing=False) * steps / slots
                refresh_seconds = self._batch_seconds(work.window, slots, refreshing=True) / slots
                phase_load = Counter({rank: work.owners.count(rank) * denoise_seconds for rank in set(work.owners)})
                for rank in work.refresh_owners:
                    phase_load[rank] += refresh_seconds
            else:
                seconds = (
                    self._batch_seconds(work.window, slots, refreshing=False) * steps
                    + self._batch_seconds(work.window, slots, refreshing=True)
                ) / slots
                phase_load = {rank: work.owners.count(rank) * seconds for rank in set(work.owners)}
            for rank, row_seconds in phase_load.items():
                queues[rank].append((row_seconds / weight, row_seconds, weight, key))
        for queue in queues:
            queue.sort()
            prefix = 0.0
            remaining_weight = sum(weight for _, _, weight, _ in queue)
            for normalized, seconds, weight, key in queue:
                completion[key] = max(completion[key], duration + prefix + remaining_weight * normalized)
                prefix += seconds
                remaining_weight -= weight
        loss = 0.0
        for key, work in works.items():
            loss += self._approximate_chain_loss(states[key], work, completion[key], slots)
        return loss, completion

    def _forecast_completion(
        self,
        works: Mapping[str, _Work],
        wave: Sequence[_Work],
        states: Mapping[str, SessionSchedulingState],
        ranks: int,
        slots: int,
    ) -> dict[str, float]:
        """Return each current head's completion in the causal queue rollout."""
        return {key: times[0] for key, times in self._forecast_sequences(works, wave, states, ranks, slots).items()}

    def _forecast_sequences(
        self,
        works: Mapping[str, _Work],
        wave: Sequence[_Work],
        states: Mapping[str, SessionSchedulingState],
        ranks: int,
        slots: int,
    ) -> dict[str, list[float]]:
        """Project known FIFO chains through feasible physical waves.

        Only a chain's head is executable. A queued action enters the forecast
        after its predecessor's refresh, paying for all its own denoise steps
        and refresh. Future profiles use the current candidate as a receding
        horizon estimate; they are never committed before actual dispatch.
        """
        elapsed = self._wave_seconds(wave, ranks) if wave else 0.0
        advancing = {work.session_id for work in wave}
        remaining = {key: work.steps - int(key in advancing) for key, work in works.items()}
        completion: dict[str, list[float]] = {key: [] for key in works}
        queued = {key: work.future_jobs for key, work in works.items()}
        append_step = {
            key: work.steps if work.appending and key not in advancing else None for key, work in works.items()
        }
        for key in tuple(remaining):
            if remaining[key] < 0:
                completion[key].append(elapsed)
                if queued[key]:
                    remaining[key] = works[key].full_steps
                    append_step[key] = remaining[key]
                    queued[key] -= 1
                else:
                    del remaining[key]
        weights = {
            key: 1.0
            + (
                max(0.0, -states[key].slack_seconds - len(completion[key]) * work.output_seconds) / work.output_seconds
                if states[key].playback_active and not states[key].departed
                else 0.0
            )
            for key, work in works.items()
        }
        shapes = {}
        owner_counts = {}
        refresh_counts = {}
        phase_separated = any(work.refresh_owners for work in works.values())
        for key, work in works.items():
            compatibility = work.compatibility
            # Refresh is the penultimate native compatibility field. Its
            # future value follows remaining steps, not the current phase.
            shapes[key] = compatibility[:5] + compatibility[6:] if len(compatibility) == 7 else compatibility
            owner_counts[key] = tuple(work.owners.count(rank) for rank in range(ranks))
            refresh_counts[key] = tuple((work.refresh_owners or work.owners).count(rank) for rank in range(ranks))

        while remaining:
            ordered = sorted(remaining, key=lambda key: (-weights[key] / (remaining[key] + 1), key))
            compatibility = shapes[ordered[0]], remaining[ordered[0]] == 0
            future = []
            counts = [0] * ranks
            layout = [Counter() for _ in range(ranks)]
            for key in ordered:
                if shapes[key] != compatibility[0] or (
                    not phase_separated and (remaining[key] == 0) != compatibility[1]
                ):
                    continue
                original = works[key]
                physical_counts = refresh_counts[key] if remaining[key] == 0 else owner_counts[key]
                proposed = [used + added for used, added in zip(counts, physical_counts, strict=True)]
                if max(proposed) > slots:
                    continue
                counts = proposed
                future.append(key)
                phase = 0 if remaining[key] == 0 else 1 if remaining[key] == append_step[key] else 2
                for rank, rows in enumerate(physical_counts):
                    if rows:
                        layout[rank][original.window, phase] += rows
            elapsed += self._layout_seconds(tuple(tuple(sorted(count.items())) for count in layout))
            for key in future:
                remaining[key] -= 1
                if remaining[key] < 0:
                    work = works[key]
                    completion[key].append(elapsed)
                    if queued[key]:
                        remaining[key] = works[key].full_steps
                        append_step[key] = remaining[key]
                        queued[key] -= 1
                        state = states[key]
                        slack = state.slack_seconds + len(completion[key]) * work.output_seconds
                        weights[key] = 1.0 + (
                            max(0.0, -slack) / work.output_seconds
                            if state.playback_active and not state.departed
                            else 0.0
                        )
                    else:
                        del remaining[key]
        return completion

    @staticmethod
    def _cache_layout(session: ABotWorldInteractiveSession) -> tuple[int, int, int]:
        per_token = sum(
            prod(entry[name].shape[2:]) * entry[name].element_size()
            for entry in session.self_cache
            for name in ("k", "v")
        )
        cross = sum(
            prod(entry[name].shape[1:]) * entry[name].element_size()
            for entry in session.cross_cache
            for name in ("k", "v")
        )
        return session.self_cache[0]["k"].shape[1], per_token, cross

    @classmethod
    def _cache_bytes(cls, session: ABotWorldInteractiveSession) -> int:
        capacity, per_token, cross = cls._cache_layout(session)
        return capacity * per_token + cross

    def plan(
        self,
        states: Sequence[SessionSchedulingState],
        sessions: Mapping[str, ABotWorldInteractiveSession],
        runtime: ABotWorldLatentStepRuntime,
        *,
        kv_budget_bytes: Sequence[int] | None = None,
    ) -> LatentStepPlan:
        """Choose one wave; commit new quality and placement only at dispatch.

        Placement candidates collapse equal-compute latent permutations.
        KV budgets exclude model weights, arena and workspace, reserved by the
        caller. Existing over-budget state may drain without further growth.
        """
        windows = {fidelity.local_attn_size for fidelity in self.fidelities.values()}
        if not windows.issubset(self.step_work) or not windows.issubset(self.refresh_work):
            raise RuntimeError("Measure each attention window before enabling quality decisions")
        ranks = runtime.exchange.world_size
        slots = runtime.max_batch_size
        denoise_ranks = getattr(runtime, "denoise_ranks", tuple(range(ranks)))
        refresh_ranks = getattr(runtime, "refresh_ranks", ())
        self._wave_predictions.clear()
        if kv_budget_bytes is not None and len(kv_budget_bytes) != ranks:
            raise ValueError("Provide one retained-KV budget per rank")
        layouts = tuple(
            layout
            for layout in combinations_with_replacement(denoise_ranks, 3)
            if max(layout.count(rank) for rank in range(ranks)) <= runtime.max_batch_size
            and (not refresh_ranks or len(set(layout)) == 1)
        )
        colocated_layouts = tuple(layout for layout in layouts if len(set(layout)) == 1)
        if not layouts:
            raise ValueError("Per-rank latent slots cannot fit one complete sibling group")
        by_id = {state.session_id: state for state in states}
        reference = max(self.profiles, key=lambda profile: profile.quality)
        works: dict[str, _Work] = {}
        load = [0.0] * ranks
        memory = [0] * ranks
        cache_layouts = {key: self._cache_layout(session) for key, session in sessions.items()}
        resident_memory = {}
        for key in sessions:
            owners = runtime._owners.get(key)
            capacity, per_token, cross = cache_layouts[key]
            if owners is not None:
                capacity = runtime._retained_capacities.get(key, capacity)
                refresh_owners = getattr(runtime, "_refresh_owners", {}).get(key, ())
                resident_memory[key] = [
                    (capacity * per_token + cross) * (owners.count(rank) + refresh_owners.count(rank))
                    for rank in range(ranks)
                ]
            else:
                rows = sessions[key].self_cache[0]["k"].shape[0]
                resident_memory[key] = [(capacity * per_token + cross) * rows] * ranks
            memory = [used + resident for used, resident in zip(memory, resident_memory[key], strict=True)]
        limits = tuple(max(budget, used) for budget, used in zip(kv_budget_bytes, memory)) if kv_budget_bytes else None
        for key, task in runtime._requests.items():
            profile = next(
                (
                    profile
                    for profile in self.profiles
                    if self.fidelities[profile.fidelity].local_attn_size == task.window[0]
                    and self.fidelities[profile.fidelity].denoise_steps == len(task.timesteps)
                ),
                reference,
            )
            work = _Work(
                key,
                runtime._owners[key][: task.full_shape[2]],
                task.window[0],
                len(task.timesteps) - task.step_index,
                profile.quality,
                task.full_shape[2] / 3,
                task.latent_batch_key(),
                task.step_index == 0,
                future_jobs=0 if by_id[key].departed else by_id[key].pending_action_count,
                full_steps=len(task.timesteps),
                refresh_owners=getattr(runtime, "_refresh_owners", {}).get(key, ()),
            )
            works[key] = work
            if work.refresh_owners:
                for owner in work.owners:
                    load[owner] += self._batch_seconds(work.window, slots, refreshing=False) * work.steps / slots
                for owner in work.refresh_owners:
                    load[owner] += self._batch_seconds(work.window, slots, refreshing=True) / slots
            else:
                for owner in work.owners:
                    load[owner] += self._work_seconds(work, slots)
        for state in states:
            key = state.session_id
            if key in works or state.departed or state.ready_job(include_idle=True) is None:
                continue
            session = sessions[key]
            fidelity = self.fidelities[reference.fidelity]
            seconds = (
                fidelity.denoise_steps * self._batch_seconds(fidelity.local_attn_size, slots, refreshing=False)
                + self._batch_seconds(fidelity.local_attn_size, slots, refreshing=True)
            ) / slots
            owners = runtime._owners.get(key)
            if owners is None:
                owners = min(
                    colocated_layouts or layouts,
                    key=lambda layout: max(load[rank] + layout.count(rank) * seconds for rank in range(ranks)),
                )
            refresh_owners = getattr(runtime, "_refresh_owners", {}).get(key, ())
            if refresh_ranks and not refresh_owners:
                refresh_owner = min(refresh_ranks, key=lambda rank: (load[rank], rank))
                refresh_owners = (refresh_owner,) * 3
            shape = session.first_frame_latent.shape
            compatibility = (
                (shape[1], 3, *shape[3:]),
                runtime.pipeline.torch_dtype,
                session.first_frame_latent.device,
                session.prompt_emb.shape[1:],
                runtime.pipeline.denoise_stage.dit.use_relative_rope,
                False,
                False,
            )
            work = _Work(
                key,
                owners,
                fidelity.local_attn_size,
                fidelity.denoise_steps,
                reference.quality,
                reference.output_seconds,
                compatibility,
                True,
                fidelity,
                future_jobs=max(0, state.pending_action_count - 1),
                full_steps=fidelity.denoise_steps,
                refresh_owners=refresh_owners,
            )
            works[key] = work
            if refresh_owners:
                for owner in owners:
                    load[owner] += (
                        fidelity.denoise_steps
                        * self._batch_seconds(fidelity.local_attn_size, slots, refreshing=False)
                        / slots
                    )
                for owner in refresh_owners:
                    load[owner] += self._batch_seconds(fidelity.local_attn_size, slots, refreshing=True) / slots
            else:
                for owner in owners:
                    load[owner] += seconds
        if not works:
            return LatentStepPlan((), {}, 0.0, 0.0)
        # Pending heads have the same quality choices as admitted Jobs. Do not
        # rank every new head as a slow reference-quality Job while charging
        # retained Jobs for their already-selected faster configuration.
        for key, original in works.items():
            if original.fidelity is None:
                continue
            best = None
            for profile in self.profiles:
                fidelity = self.fidelities[profile.fidelity]
                candidate = replace(
                    original,
                    window=fidelity.local_attn_size,
                    steps=fidelity.denoise_steps,
                    full_steps=fidelity.denoise_steps,
                    quality=profile.quality,
                    output_seconds=profile.output_seconds,
                    fidelity=fidelity,
                )
                # A separable completion lower bound initializes ranking.
                # The joint solve below still evaluates every feasible profile
                # with all other heads; avoid a second global search here.
                remaining = replace(candidate, steps=max(0, candidate.steps - 1))
                completion = self._wave_seconds((candidate,), ranks)
                completion += max(candidate.owners.count(rank) for rank in range(ranks)) * self._work_seconds(
                    remaining, slots
                )
                loss = self._approximate_chain_loss(by_id[key], candidate, completion, slots)
                if best is None or loss < best[0]:
                    best = (loss, candidate)
            works[key] = best[1]
        baseline, _ = self._objective(works, (), by_id, ranks, slots)
        priority = sorted(
            works,
            key=lambda key: (
                self._objective(works, (works[key],), by_id, ranks, slots)[0] - baseline,
                by_id[key].slack_seconds,
                key,
            ),
        )
        selected: list[_Work] = []
        base_works = dict(works)
        best_wave: tuple[_Work, ...] = ()
        best_loss = float("inf")
        counts = [0] * ranks
        for key in priority:
            original = works[key]
            if selected:
                shape = original.compatibility
                reference_shape = selected[0].compatibility
                if refresh_ranks:
                    shape, reference_shape = shape[:5] + shape[6:], reference_shape[:5] + reference_shape[6:]
                if shape != reference_shape:
                    continue
            profiles = self.profiles if original.fidelity is not None else (None,)
            placements = (runtime._owners[key],) if key in runtime._owners else layouts
            if original.fidelity is None:
                placements = (original.owners,)
            retained_refresh = getattr(runtime, "_refresh_owners", {}).get(key)
            refresh_placements = (
                (retained_refresh,)
                if retained_refresh is not None
                else tuple((rank,) * 3 for rank in refresh_ranks)
                if refresh_ranks
                else ((),)
            )
            feasible = []
            for owners in placements:
                for refresh_owners in refresh_placements:
                    physical_owners = refresh_owners if original.steps == 0 and refresh_ranks else owners
                    proposed_counts = [counts[rank] + physical_owners.count(rank) for rank in range(ranks)]
                    if max(proposed_counts) > runtime.max_batch_size:
                        continue
                    feasible.append((owners, proposed_counts, refresh_owners))
            # A session normally stays a local continuous-batching unit.  A
            # split placement is an elastic fallback for the case where no
            # same-rank placement satisfies the current slot/memory limits;
            # it is never selected merely because a cross-rank layout has a
            # slightly better arithmetic forecast.
            if key not in runtime._owners:
                colocated_feasible = [entry for entry in feasible if len(set(entry[0])) == 1]
                placement_groups = (colocated_feasible, feasible) if colocated_feasible else (feasible,)
            else:
                placement_groups = (feasible,)
            best = None
            for placement_group in placement_groups:
                for profile in profiles:
                    fidelity = self.fidelities[profile.fidelity] if profile is not None else None
                    configured = (
                        original
                        if profile is None
                        else replace(
                            original,
                            window=fidelity.local_attn_size,
                            steps=fidelity.denoise_steps,
                            full_steps=fidelity.denoise_steps,
                            quality=profile.quality,
                            output_seconds=profile.output_seconds,
                            fidelity=fidelity,
                        )
                    )
                    if profile is not None:
                        patch = runtime.pipeline.denoise_stage.dit.patch_size
                        shape = sessions[key].first_frame_latent.shape
                        tokens = (shape[-2] // patch[1]) * (shape[-1] // patch[2])
                        capacity, per_token, cross = cache_layouts[key]
                        configured_bytes = min(capacity, fidelity.local_attn_size * tokens) * per_token + cross
                    for owners, proposed_counts, refresh_owners in placement_group:
                        proposed_memory = (
                            [
                                used
                                - resident_memory[key][rank]
                                + configured_bytes * (owners.count(rank) + refresh_owners.count(rank))
                                for rank, used in enumerate(memory)
                            ]
                            if profile is not None
                            else memory
                        )
                        if limits is not None and any(used > limit for used, limit in zip(proposed_memory, limits)):
                            continue
                        work = (
                            configured
                            if configured.owners == owners and configured.refresh_owners == refresh_owners
                            else replace(configured, owners=owners, refresh_owners=refresh_owners)
                        )
                        works[key] = work
                        loss, completion = self._objective(works, (*selected, work), by_id, ranks, slots)
                        order = (
                            loss,
                            completion[key],
                            max(proposed_memory),
                            sum(used * used for used in proposed_memory),
                            owners,
                        )
                        if best is None or order < best[0]:
                            best = (order, work, proposed_counts, proposed_memory)
                # If a same-rank placement is feasible, do not compare it
                # against an unnecessary communication-heavy split placement.
                if best is not None:
                    break
            works[key] = original
            if best is not None:
                _, work, counts, memory = best
                works[key] = work
                selected.append(work)
                forecast = self._forecast_sequences(works, selected, by_id, ranks, slots)
                loss = sum(
                    self._loss(by_id[key], item, completed, output_offset=index * item.output_seconds)
                    for key, item in works.items()
                    for index, completed in enumerate(forecast[key])
                )
                # A single addition may imbalance the ranks before later
                # additions rebalance them. Evaluate all feasible prefixes.
                if loss < best_loss - 1e-9 or abs(loss - best_loss) <= 1e-9:
                    best_wave, best_loss = tuple(selected), loss
        selected = list(best_wave)
        works = {**base_works, **{work.session_id: work for work in selected}}
        completion = self._forecast_sequences(works, selected, by_id, ranks, slots)
        objective = sum(
            self._loss(by_id[key], item, completed, output_offset=index * item.output_seconds)
            for key, item in works.items()
            for index, completed in enumerate(completion[key])
        )
        admissions = {
            work.session_id: LatentStepChoice(
                work.fidelity, work.quality, completion[work.session_id][0], work.owners, work.refresh_owners
            )
            for work in selected
            if work.fidelity is not None
        }
        return LatentStepPlan(
            tuple(work.session_id for work in selected), admissions, self._wave_seconds(selected, ranks), objective
        )

    def observe(
        self,
        tasks: Sequence[LatentStepKey],
        windows_by_session: Mapping[str, int],
        *,
        ranks: int,
        residence_seconds: float,
    ) -> None:
        """Update from completed GPU events, using the slowest rank's wave time.

        Homogeneous windows update an occupancy-dependent batch-time curve.
        Mixed windows update only the identifiable joint layout time.
        Refresh includes output gathering and decoding and is learned separately.
        """
        if residence_seconds <= 0 or not tasks or any(key.chunk_start == 0 for key in tasks):
            return
        costs = self.refresh_work if tasks[0].refreshing else self.step_work
        windows = set(windows_by_session.values())
        rate = self.config.profile_timing_ema
        layout = self._layout_key(
            [
                (
                    task.owner,
                    windows_by_session[task.session_id],
                    0 if task.refreshing else 1 if task.step_index == 0 else 2,
                )
                for task in tasks
            ],
            ranks,
        )
        previous = self.wave_times.get(layout, residence_seconds)
        self.wave_times[layout] = previous + rate * (residence_seconds - previous)
        self.wave_times.move_to_end(layout)
        if len(self.wave_times) > 128:
            self.wave_times.popitem(last=False)
        self.observations += 1
        # A mixed D/R wave supplies one critical-path duration. It cannot
        # identify the two component costs; retain it as a joint layout sample.
        if len(windows) == 1 and len({task.refreshing for task in tasks}) == 1:
            window = next(iter(windows))
            rows = [sum(key.owner == rank for key in tasks) for rank in range(ranks)]
            measured = residence_seconds / max(rows)
            previous = costs.get(window, measured)
            costs[window] = previous + rate * (measured - previous)
            samples = self.batch_times.setdefault((window, tasks[0].refreshing), {})
            previous = samples.get(max(rows), residence_seconds)
            samples[max(rows)] = previous + rate * (residence_seconds - previous)
            self._batch_predictions.clear()
        self._wave_predictions.clear()

    def timing_snapshot(self) -> dict[str, object]:
        """Return the live timing model used by the next planning wave.

        The offline profile remains the source of quality and memory values;
        measured execution replaces only its latency prediction.  Returning a
        detached snapshot makes online adaptation observable without exposing
        mutable predictor dictionaries to a metrics or trace writer.
        """
        return {
            "denoise_seconds_per_row": dict(self.step_work),
            "refresh_seconds_per_row": dict(self.refresh_work),
            "batch_seconds": {
                f"w{window}_{'refresh' if refreshing else 'denoise'}": dict(samples)
                for (window, refreshing), samples in self.batch_times.items()
            },
            "wave_seconds": {str(key): value for key, value in self.wave_times.items()},
            "observations": self.observations,
        }
