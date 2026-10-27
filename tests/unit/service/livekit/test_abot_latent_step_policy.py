from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from telefuser.pipelines.abot_world.latent_step_runtime import LatentStepKey
from telefuser.service.livekit.abot_latent_step_policy import ABotLatentStepPolicy
from telefuser.service.livekit.motivation_scheduler import (
    MotivationProfile,
    MotivationSchedulerConfig,
    SessionSchedulingState,
)


def _profiles():
    return [
        MotivationProfile(1, "b1_s2_w6_rho0_bf16", 0.1, 0.9, 1.0),
        MotivationProfile(1, "b1_s4_w18_rho0_bf16", 1000.0, 1.0, 1.0),
    ]


def _observe(policy, window, seconds, *, refresh=False):
    tasks = tuple(LatentStepKey("sample", 3, index, 2, refresh, index) for index in range(3))
    policy.observe(tasks, {"sample": window}, ranks=4, residence_seconds=seconds)


def _policy(profiles=None):
    policy = ABotLatentStepPolicy(profiles or _profiles(), MotivationSchedulerConfig())
    for window in (6, 18):
        _observe(policy, window, 0.03 if window == 6 else 0.12)
        _observe(policy, window, 0.03, refresh=True)
    return policy


def _runtime(ranks=4, slots=8):
    return SimpleNamespace(
        exchange=SimpleNamespace(world_size=ranks),
        max_batch_size=slots,
        _requests={},
        _owners={},
        _retained_capacities={},
        pipeline=SimpleNamespace(
            torch_dtype=torch.float32,
            denoise_stage=SimpleNamespace(dit=SimpleNamespace(use_relative_rope=True, patch_size=(1, 2, 2))),
        ),
    )


def _state(slack=-10.0, name="new"):
    state = SessionSchedulingState(name, "pool", slack, 1.0, action_seen=True)
    state.submit_action(job_id=f"job-{name}", controls=("KeyW",), now=0.0, release=True)
    return state


def _session():
    return SimpleNamespace(
        first_frame_latent=torch.zeros(1, 4, 1, 4, 4),
        prompt_emb=torch.zeros(1, 4, 16),
        self_cache=[{"k": torch.zeros(1, 18, 1, 4), "v": torch.zeros(1, 18, 1, 4)}],
        cross_cache=[
            {"k": torch.zeros(1, 4, 1, 4), "v": torch.zeros(1, 4, 1, 4), "is_init": False, "sequence_length": 4}
        ],
    )


def _choice(policy, state):
    return policy.plan([state], {state.session_id: _session()}, _runtime()).admissions[state.session_id]


def test_online_measurements_change_choice_and_offline_latency_does_not():
    state = _state()
    policy = _policy()
    choice = _choice(policy, state)
    assert choice.fidelity.name == "b1_s2_w6_rho0_bf16"
    reversed_latency = [replace(profile, latency_seconds=1 / profile.latency_seconds) for profile in _profiles()]
    assert _choice(_policy(reversed_latency), state) == choice
    _observe(policy, 6, 2.0)
    assert _choice(policy, state).fidelity.name == "b1_s4_w18_rho0_bf16"


def test_parallel_ranks_use_critical_path_instead_of_serial_sum():
    policy, runtime = _policy([_profiles()[0]]), _runtime()
    states = [_state(name="a"), _state(name="b")]
    runtime._owners = {"a": (0, 0, 0), "b": (1, 1, 1)}
    decision = policy.plan(states, {state.session_id: _session() for state in states}, runtime)
    assert set(decision.session_ids) == {"a", "b"}
    assert decision.admissions["a"].predicted_seconds == pytest.approx(0.27)
    assert decision.admissions["b"].predicted_seconds == pytest.approx(0.27)


def test_sufficient_slack_preserves_reference_quality():
    state = _state(slack=20.0)
    assert _choice(_policy(), state).quality == 1.0


def test_first_placement_respects_kv_capacity_and_next_job_reuses_owners():
    policy, state, runtime, session = _policy(), _state(), _runtime(ranks=2), _session()
    row_bytes = policy._cache_bytes(session)
    decision = policy.plan([state], {"new": session}, runtime, kv_budget_bytes=(row_bytes, row_bytes * 2))
    assert decision.admissions["new"].owners == (0, 1, 1)
    runtime._owners["new"] = (0, 1, 1)
    assert policy.plan([state], {"new": session}, runtime).admissions["new"].owners == (0, 1, 1)


def test_unplaced_empty_kv_shell_consumes_no_replica_budget_and_charges_selected_window():
    policy, runtime, session = _policy(), _runtime(), _session()
    session.first_frame_latent = torch.zeros(1, 4, 1, 2, 2)
    for cache in (session.self_cache, session.cross_cache):
        for entry in cache:
            entry["k"], entry["v"] = entry["k"][:0], entry["v"][:0]
    # One W6 row plus the text cache fits; a W18 row cannot fit this budget.
    row_budget = (6 + 4) * 2 * 4 * 4
    result = policy.plan([_state()], {"new": session}, runtime, kv_budget_bytes=(row_budget,) * 4)
    choice = result.admissions["new"]
    assert choice.fidelity.name == "b1_s2_w6_rho0_bf16"
    assert len(set(choice.owners)) == 3


def test_negative_slack_remains_work_conserving_and_obeys_slots():
    states = [_state(slack=-1000, name=str(index)) for index in range(16)]
    runtime = _runtime(slots=2)
    decision = _policy().plan(states, {state.session_id: _session() for state in states}, runtime)
    assert decision.session_ids
    rows = [0] * 4
    for choice in decision.admissions.values():
        assert len(choice.owners) == 3
        for owner in choice.owners:
            rows[owner] += 1
    assert max(rows) <= 2


def test_first_placement_avoids_a_rank_with_more_remaining_work():
    policy, runtime = _policy(), _runtime()
    busy, arriving = _state(slack=20, name="busy"), _state()
    busy.mark_dispatched(busy.ready_job(include_idle=False))
    runtime._owners["busy"] = (0, 0, 0)
    runtime._requests["busy"] = SimpleNamespace(
        session_id="busy",
        window=(18, 6),
        timesteps=(0, 1, 2, 3),
        step_index=0,
        full_shape=(1, 4, 3, 4, 4),
        latent_batch_key=lambda: ("other-shape",),
    )
    decision = policy.plan([busy, arriving], {"busy": _session(), "new": _session()}, runtime)
    assert "new" in decision.admissions
    assert 0 not in decision.admissions["new"].owners
    assert runtime._owners["busy"] == (0, 0, 0)


def test_batch_growth_can_cross_a_temporarily_imbalanced_prefix():
    runtime = _runtime()
    policy = _policy()
    tasks = tuple(LatentStepKey(str(index // 3), 3, index % 3, 1, False, index % 4) for index in range(30))
    policy.observe(tasks, {task.session_id: 18 for task in tasks}, ranks=4, residence_seconds=0.5)
    states = [_state(slack=-10, name=str(index)) for index in range(16)]
    sessions = {state.session_id: _session() for state in states}
    for index, state in enumerate(states):
        state.mark_dispatched(state.ready_job(include_idle=False))
        runtime._owners[state.session_id] = tuple((index * 3 + latent) % 4 for latent in range(3))
        runtime._requests[state.session_id] = SimpleNamespace(
            session_id=state.session_id,
            window=(18, 6),
            timesteps=(0, 1, 2, 3),
            step_index=1,
            full_shape=(1, 4, 3, 4, 4),
            latent_batch_key=lambda: ("same",),
        )
    decision = policy.plan(states, sessions, runtime)
    assert len(decision.session_ids) >= 8
    assert not decision.admissions


def test_refresh_and_denoising_update_separate_online_costs():
    policy = _policy()
    before = dict(policy.step_work)
    _observe(policy, 6, 0.3, refresh=True)
    assert policy.step_work == before
    assert policy.refresh_work[6] > policy.refresh_work[18]


def test_timing_snapshot_exposes_live_predictor_without_aliasing_state():
    policy = _policy()
    snapshot = policy.timing_snapshot()
    assert snapshot["denoise_seconds_per_row"] == policy.step_work
    assert snapshot["refresh_seconds_per_row"] == policy.refresh_work
    assert snapshot["observations"] == policy.observations
    snapshot["denoise_seconds_per_row"][6] = 999.0
    assert policy.step_work[6] != 999.0


def test_mixed_window_measurements_and_startup_exclusion():
    policy = _policy()
    tasks = (LatentStepKey("a", 3, 0, 1, False, 0), LatentStepKey("b", 3, 0, 2, False, 0))
    before = dict(policy.step_work)
    policy.observe(tasks, {"a": 6, "b": 18}, ranks=4, residence_seconds=0.12)
    # A single joint duration cannot identify two independent window costs.
    assert policy.step_work == before
    assert 0.12 in policy.wave_times.values()
    policy.observe([replace(tasks[0], chunk_start=0)], {"a": 6}, ranks=4, residence_seconds=100.0)
    assert policy.step_work == before


def test_deeply_overdue_new_job_is_not_starved_by_short_inflight_work():
    policy, runtime = _policy(), _runtime(ranks=1, slots=3)
    busy, waiting = _state(slack=-1, name="busy"), _state(slack=-100, name="waiting")
    busy.mark_dispatched(busy.ready_job(include_idle=False))
    runtime._owners = {"busy": (0, 0, 0), "waiting": (0, 0, 0)}
    session = _session()
    compatibility = ((4, 3, 4, 4), torch.float32, torch.device("cpu"), (4, 16), True, False, False)
    runtime._requests["busy"] = SimpleNamespace(
        window=(6, 6),
        timesteps=(0, 1),
        step_index=1,
        full_shape=(1, 4, 3, 4, 4),
        latent_batch_key=lambda: compatibility,
    )
    decision = policy.plan([busy, waiting], {"busy": session, "waiting": _session()}, runtime)
    assert decision.session_ids == ("waiting",)
    assert "waiting" in decision.admissions


def test_measured_batch_speedup_is_used_when_filling_parallel_ranks():
    policy = _policy([_profiles()[0]])
    tasks = tuple(LatentStepKey(str(index // 3), 3, index % 3, 1, False, index % 4) for index in range(30))
    # Strong amortization in both phases makes filling slots beneficial even
    # after accounting for the earlier completion of a smaller first wave.
    policy.observe(tasks, {task.session_id: 6 for task in tasks}, ranks=4, residence_seconds=0.04)
    policy.observe(
        tuple(replace(task, refreshing=True) for task in tasks),
        {task.session_id: 6 for task in tasks},
        ranks=4,
        residence_seconds=0.04,
    )
    states = [_state(name=str(index)) for index in range(16)]
    runtime = _runtime()
    runtime._owners = {
        state.session_id: tuple((index * 3 + latent) % 4 for latent in range(3)) for index, state in enumerate(states)
    }
    decision = policy.plan(states, {state.session_id: _session() for state in states}, runtime)
    assert len(decision.session_ids) >= 8
    assert decision.predicted_wave_seconds <= 0.04 + 1e-9


def test_reference_quality_is_not_forced_down_by_an_unrelated_mixed_observation():
    policy = _policy()
    state = _state(slack=20)
    before = _choice(policy, state)
    tasks = (LatentStepKey("a", 3, 0, 1, False, 0), LatentStepKey("b", 3, 0, 2, False, 0))
    policy.observe(tasks, {"a": 6, "b": 18}, ranks=4, residence_seconds=100)
    assert _choice(policy, state) == before


def test_batch_growth_uses_measured_benefit_instead_of_forcing_maximum_size():
    policy = _policy([_profiles()[0]])
    tasks = tuple(LatentStepKey(str(index // 3), 3, index % 3, 1, False, index % 4) for index in range(30))
    policy.observe(tasks, {task.session_id: 6 for task in tasks}, ranks=4, residence_seconds=10)
    states = [_state(name=str(index)) for index in range(16)]
    decision = policy.plan(states, {state.session_id: _session() for state in states}, _runtime())
    assert 0 < len(decision.session_ids) < 8


def test_unmeasured_occupancy_does_not_extrapolate_a_noisy_launch_discount():
    policy = _policy()
    tasks = tuple(LatentStepKey(str(index // 3), 3, index % 3, 1, False, index % 4) for index in range(12))
    policy.observe(tasks, {task.session_id: 6 for task in tasks}, ranks=4, residence_seconds=0.05)
    assert policy._batch_seconds(6, 8, refreshing=False) >= 0.05 * 8 / 3


def test_completion_forecast_pays_for_slot_and_precision_constrained_waves():
    from telefuser.service.livekit.abot_latent_step_policy import _Work

    policy = _policy()
    states = {key: _state(name=key) for key in ("d", "r")}
    works = {key: _Work(key, (0, 0, 0), 6, steps, 0.9, 1, ("shape",), False) for key, steps in (("d", 1), ("r", 0))}
    completion = policy._forecast_completion(works, (), states, ranks=1, slots=3)
    assert completion["r"] == pytest.approx(0.09)
    assert completion["d"] == pytest.approx(0.27)


def test_known_fifo_successors_pay_for_complete_denoise_and_refresh_after_their_predecessors():
    from telefuser.service.livekit.abot_latent_step_policy import _Work

    policy = _policy()
    state = _state(name="a")
    work = _Work("a", (0, 0, 0), 6, 1, 0.9, 1, ("shape",), False, future_jobs=2, full_steps=2)
    forecast = policy._forecast_sequences({"a": work}, (), {"a": state}, ranks=1, slots=3)
    # Every physical wave costs 0.09 seconds: current D+R, then D+D+R twice.
    assert forecast["a"] == pytest.approx([0.18, 0.45, 0.72])
    assert policy._forecast_completion({"a": work}, (), {"a": state}, ranks=1, slots=3)["a"] == pytest.approx(0.18)
    # Completing a refresh in the first wave must expose, not skip, its successor.
    refreshing = replace(work, steps=0)
    forecast = policy._forecast_sequences({"a": refreshing}, (refreshing,), {"a": state}, ranks=1, slots=3)
    assert forecast["a"] == pytest.approx([0.09, 0.36, 0.63])


def test_fifo_forecast_uses_parallel_ranks_without_exposing_two_heads_of_the_same_session():
    from telefuser.service.livekit.abot_latent_step_policy import _Work

    policy = _policy()
    states = {key: _state(name=key) for key in ("a", "b")}
    works = {
        key: _Work(key, (rank, rank, rank), 6, 2, 0.9, 1, ("shape",), True, future_jobs=3, full_steps=2)
        for rank, key in enumerate(states)
    }
    completion = policy._forecast_sequences(works, (), states, ranks=2, slots=3)
    assert completion["a"] == pytest.approx([0.27, 0.54, 0.81, 1.08])
    assert completion["b"] == pytest.approx(completion["a"])


def test_planning_accounts_for_known_fifo_work_without_dispatching_or_mutating_successors():
    state = _state()
    state.pending_action_slots = 4
    for sequence in range(3):
        state.submit_action(job_id=f"queued-{sequence}", controls=("KeyD",), now=0, release=True)
    ids = [state.pending_action.job_id, *(job.job_id for job in state.pending_action_queue)]
    policy = _policy([_profiles()[0]])
    decision = policy.plan([state], {state.session_id: _session()}, _runtime())
    assert decision.session_ids == (state.session_id,)
    assert set(decision.admissions) == {state.session_id}
    assert state.in_flight is None
    assert ids == [state.pending_action.job_id, *(job.job_id for job in state.pending_action_queue)]


def test_quality_selection_requires_measured_windows():
    policy = ABotLatentStepPolicy(_profiles(), MotivationSchedulerConfig())
    state = _state()
    with pytest.raises(RuntimeError, match="Measure each attention window"):
        policy.plan([state], {"new": _session()}, _runtime())


def test_phase_pool_forecast_overlaps_refresh_with_another_sessions_denoise():
    from telefuser.service.livekit.abot_latent_step_policy import _Work

    policy = _policy()
    states = {key: _state(name=key) for key in ("d", "r")}
    works = {
        key: _Work(key, (0, 0, 0), 6, steps, 0.9, 1, ("shape",), False, refresh_owners=(1, 1, 1))
        for key, steps in (("d", 1), ("r", 0))
    }
    forecast = policy._forecast_completion(works, tuple(works.values()), states, ranks=2, slots=3)
    assert forecast["r"] == pytest.approx(0.09)
    assert forecast["d"] == pytest.approx(0.18)


def test_phase_policy_admits_denoise_and_refresh_in_one_parallel_dispatch():
    policy, runtime = _policy([_profiles()[0]]), _runtime(ranks=2, slots=3)
    runtime.denoise_ranks, runtime.refresh_ranks = (0,), (1,)
    runtime._owners = {"refreshing": (0, 0, 0)}
    runtime._refresh_owners = {"refreshing": (1, 1, 1)}
    refreshing, arriving = _state(name="refreshing"), _state(name="arriving")
    refreshing.mark_dispatched(refreshing.ready_job(include_idle=False))
    session = _session()
    compatibility = ((4, 3, 4, 4), torch.float32, torch.device("cpu"), (4, 16), True, True, False)
    runtime._requests["refreshing"] = SimpleNamespace(
        window=(6, 6),
        timesteps=(0, 1),
        step_index=2,
        full_shape=(1, 4, 3, 4, 4),
        latent_batch_key=lambda: compatibility,
    )
    result = policy.plan([refreshing, arriving], {"refreshing": session, "arriving": _session()}, runtime)
    assert set(result.session_ids) == {"refreshing", "arriving"}
    assert result.admissions["arriving"].owners == (0, 0, 0)
    assert result.admissions["arriving"].refresh_owners == (1, 1, 1)


def test_mixed_phase_wave_does_not_attribute_one_duration_to_two_cost_curves():
    policy = _policy()
    before = dict(policy.step_work), dict(policy.refresh_work)
    tasks = (LatentStepKey("a", 3, 0, 1, False, 0), LatentStepKey("b", 3, 0, 4, True, 1))
    policy.observe(tasks, {"a": 6, "b": 6}, ranks=2, residence_seconds=0.5)
    assert (policy.step_work, policy.refresh_work) == before
    assert 0.5 in policy.wave_times.values()


def test_refresh_replica_memory_is_charged_to_its_own_gpu():
    policy, runtime = _policy([_profiles()[0]]), _runtime(ranks=2, slots=3)
    runtime.denoise_ranks, runtime.refresh_ranks = (0,), (1,)
    runtime._refresh_owners = {}
    session = _session()
    for cache in (session.self_cache, session.cross_cache):
        for entry in cache:
            entry["k"], entry["v"] = entry["k"][:0], entry["v"][:0]
    row_bytes = policy._cache_bytes(session)
    result = policy.plan([_state()], {"new": session}, runtime, kv_budget_bytes=(row_bytes * 3, 1))
    assert result.session_ids == ()


def test_refresh_placement_considers_another_gpu_when_first_choice_lacks_memory():
    policy, runtime = _policy([_profiles()[0]]), _runtime(ranks=3, slots=3)
    runtime.denoise_ranks, runtime.refresh_ranks = (0,), (1, 2)
    runtime._refresh_owners = {}
    session = _session()
    for cache in (session.self_cache, session.cross_cache):
        for entry in cache:
            entry["k"], entry["v"] = entry["k"][:0], entry["v"][:0]
    row_bytes = policy._cache_bytes(session)
    result = policy.plan([_state()], {"new": session}, runtime, kv_budget_bytes=(row_bytes * 3, 1, row_bytes * 3))
    assert result.admissions["new"].refresh_owners == (2, 2, 2)
