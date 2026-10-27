"""Replay an existing interaction trace through the whole-latent-step executor.

This direct pipeline experiment supports fixed S4/W18 or per-Job quality
selection with mixed-window latent-step execution. Input cadence, lifecycle,
bounded FIFO and held idle semantics are retained. A CPU Gloo group carries replay commands;
NCCL is reserved for model data and does not synchronize clocks per step.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tools.validation.augment_abot_cpr_quality import collect_producer_metrics
from tools.validation.benchmark_abot_livekit_burst import Scenario, load_scenario
from tools.validation.replay_abot_livekit_lifecycle_trace import load_explicit_lifecycle_trace

if TYPE_CHECKING:
    from telefuser.pipelines.abot_world.continuous import ABotWorldCompletedChunk


@dataclass(frozen=True)
class ReplayEvent:
    at: float
    kind: str
    session_id: str
    controls: tuple[str, ...] = ()


def build_events(scenario: Scenario, window: float) -> tuple[list[ReplayEvent], dict[str, Any]]:
    """Project the client's seeded heartbeat stream onto shared release phases."""
    trace = load_explicit_lifecycle_trace(scenario)
    control = scenario.session.control
    if window <= 0 or control.idle_probability:
        raise ValueError("Use a positive collection window and lifecycle-controlled active/idle intervals")
    lifecycles = defaultdict(list)
    for event in trace.events:
        lifecycles[event.trace_session_id].append(event)
    events = []
    raw_count = action_count = 0
    for index, (session_id, lifecycle) in enumerate(lifecycles.items()):
        rng = random.Random(scenario.seed + index)
        arrival, departure = lifecycle[0], lifecycle[-1]
        if arrival.event != "session_arrival" or departure.event != "session_departure":
            raise ValueError("Each trace identity must have one arrival and departure")
        events.extend(
            [
                ReplayEvent(arrival.offset_seconds, "arrival", session_id),
                ReplayEvent(departure.offset_seconds, "departure", session_id),
            ]
        )
        enabled = bool(arrival.input_enabled)
        next_heartbeat = arrival.offset_seconds
        changes = iter(lifecycle[1:-1])
        change = next(changes, None)
        sent = False
        phases = {}
        next_release = -math.inf
        while next_heartbeat < departure.offset_seconds or change is not None:
            if change is not None and change.offset_seconds <= next_heartbeat:
                at = change.offset_seconds
                enabled = bool(change.input_enabled)
                controls = rng.choice(control.action_states) if enabled else ()
                change = next(changes, None)
            else:
                at = next_heartbeat
                if at >= departure.offset_seconds:
                    break
                if enabled:
                    if sent:
                        rng.random()  # The client draws even with idle_probability=0.
                    controls = rng.choice(control.action_states)
                    sent = True
                else:
                    controls = ()
                next_heartbeat += max(
                    0.01, control.interval_seconds + rng.uniform(-control.jitter_seconds, control.jitter_seconds)
                )
            events.append(ReplayEvent(at, "control", session_id, tuple(controls)))
            raw_count += 1
            boundary = (math.floor(at / window) + 1) * window
            if controls:
                # Ordinary lifecycle clients have no explicit release metadata:
                # the service first applies a per-session minimum interval,
                # then synchronizes eligible messages to a shared phase.
                if at >= next_release - 1e-9:
                    next_release = at + window
                    phases[boundary] = tuple(controls)
                elif boundary in phases:
                    phases[boundary] = tuple(controls)
            else:
                phases.pop(boundary, None)
                next_release = -math.inf
        for boundary, controls in phases.items():
            if boundary < departure.offset_seconds:
                events.append(ReplayEvent(boundary, "release", session_id, controls))
                action_count += 1
    priority = {"departure": 0, "arrival": 1, "release": 2, "control": 3}
    events.sort(key=lambda event: (event.at, priority[event.kind], event.session_id))
    return events, {
        "duration_seconds": trace.duration_seconds,
        "sessions": len(lifecycles),
        "raw_control_messages": raw_count,
        "phase_action_releases": action_count,
        "collection_window_seconds": window,
        "control_interval_seconds": control.interval_seconds,
        "control_jitter_seconds": control.jitter_seconds,
        "phase_origin_seconds": 0.0,
        "minimum_release_interval_seconds": window,
        "input_source": "seeded client controls at declared lifecycle times; excludes network timing jitter",
    }


def replay(args: argparse.Namespace, scenario: Scenario, events: list[ReplayEvent], plan: dict[str, Any]) -> None:
    import torch
    import torch.distributed as dist
    from PIL import Image

    from examples.abot_world._loader import get_pipeline
    from telefuser.core.config import AttentionConfig, AttnImplType, WeightOffloadType
    from telefuser.pipelines.abot_world.interactive import ABotWorldInteractivePipeline
    from telefuser.service.livekit.abot_latent_step_policy import ABotLatentStepPolicy
    from telefuser.service.livekit.motivation_scheduler import (
        MotivationSchedulerConfig,
        SessionSchedulingState,
        load_motivation_profiles_csv,
    )

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    control_group = dist.new_group(backend="gloo", timeout=timedelta(minutes=5))
    clock_value = torch.zeros(1, dtype=torch.float64, device="cpu")
    pipeline = driver = policy = None
    root = rank == 0
    args.output.mkdir(parents=True, exist_ok=True)
    trace_path = args.output / "dispatch-trace.jsonl"
    dispatch_file = trace_path.open("w") if root else None
    physical_file = (args.output / "latent-step-dispatch.jsonl").open("w") if root else None
    timing_file = (args.output / "online-timing.jsonl").open("w") if root else None
    pending_timings = []
    timing_sequence = 0
    calibration = None
    kv_budget_bytes = None
    planner_calls = 0
    planner_cpu_seconds = 0.0
    planner_cpu_max_seconds = 0.0
    states, sessions, inflight = {}, {}, {}
    counts = Counter()
    batch_histogram = Counter()
    source_lag = []
    chosen_profiles, job_qualities = {}, {}
    profile_counts = Counter()
    qualities = {"b1_s4_w18_rho0_bf16": 1.0}
    if args.profile is not None:
        table = load_motivation_profiles_csv(args.profile)
        profiles = table.profiles_for(batch_size=1, gpu_id="latent-pool")
        qualities = {profile.fidelity: profile.quality for profile in profiles}
        policy = ABotLatentStepPolicy(profiles, MotivationSchedulerConfig(lambda_quality=args.lambda_quality))
    session_numbers = {event.session_id: index for index, event in enumerate(e for e in events if e.kind == "arrival")}
    sequence = event_index = 0
    started = time.monotonic()

    def shared_time() -> float:
        if root:
            clock_value[0] = time.monotonic() - started
        dist.broadcast(clock_value, src=0, group=control_group)
        return float(clock_value[0])

    def update_timings(*, keep_latest: bool) -> None:
        """Read completed events at existing chunk boundaries, with no CUDA wait."""
        count = len(pending_timings) - int(keep_latest)
        if count <= 0:
            return
        records = pending_timings[:count]
        del pending_timings[:count]
        durations = torch.tensor([start.elapsed_time(end) / 1000 for _, start, end, _, _ in records])
        dist.all_reduce(durations, op=dist.ReduceOp.MAX, group=control_group)
        for (wave, _, _, tasks, windows), seconds in zip(records, durations.tolist(), strict=True):
            policy.observe(tasks, windows, ranks=dist.get_world_size(), residence_seconds=seconds)
            if root:
                online_timing = policy.timing_snapshot()
                timing_file.write(
                    json.dumps(
                        {
                            "wave": wave,
                            "slowest_rank_seconds": seconds,
                            "per_rank_rows": dict(Counter(task.owner for task in tasks)),
                            "windows": windows,
                            "refreshing": tasks[0].refreshing,
                            "startup_chunk": any(task.chunk_start == 0 for task in tasks),
                            "online_timing": online_timing,
                        }
                    )
                    + "\n"
                )

    def step_runtime(session_ids: tuple[str, ...] | None = None) -> list[ABotWorldCompletedChunk]:
        nonlocal timing_sequence
        if policy is None:
            return driver.step(session_ids)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        outputs = driver.step(session_ids)
        end.record()
        pending_timings.append((timing_sequence, start, end, driver.last_tasks, dict(driver.last_windows)))
        timing_sequence += 1
        if outputs:
            # The existing output count.item() has completed every prior wave
            # on this stream. Leave the newest event for the next boundary.
            update_timings(keep_latest=True)
        return outputs

    def write_summary(status: str, elapsed: float) -> None:
        if not root:
            return
        dispatch_file.flush()
        physical_file.flush()
        producer = collect_producer_metrics(
            trace_path,
            qualities,
            released_action_jobs=counts["actions_released"],
            superseded_action_jobs=0,
            effective_action_jobs=counts["actions_released"] - counts["actions_dropped_pending_capacity"],
        )
        result = {
            "status": status,
            "experiment": "direct_pipeline_latent_step_trace",
            "trace": str(args.trace.resolve()),
            "plan": plan,
            "elapsed_seconds": elapsed,
            "execution": "whole_latent_step",
            "generation_config": "per_job_s_w" if policy is not None else "s4_w18_rho0_bf16",
            "quality_basis": "B-invariant normalized profile Q; numerical parity is validated separately",
            "profile_path": str(args.profile) if args.profile is not None else None,
            "completed_profile_counts": dict(profile_counts),
            "online_timing": policy.timing_snapshot() if policy is not None else None,
            "planner_objective": (
                "sum integrated playback debt / output_seconds + lambda_quality * (1-Q) * output_seconds"
                if policy is not None
                else None
            ),
            "planner_cpu_calls": planner_calls,
            "planner_cpu_seconds": planner_cpu_seconds,
            "planner_cpu_max_seconds": planner_cpu_max_seconds,
            "timing_source": "slowest-rank CUDA events; completion-boundary EWMA" if policy is not None else None,
            "calibration": calibration,
            "attention": args.attention,
            "backend": args.backend,
            "session_setup_weight_policy": "T5 and image encoder CPU offload; DiT and TAeW decoder resident",
            "world_size": dist.get_world_size(),
            "phase_pools": {
                "denoise_ranks": list(driver.denoise_ranks),
                "refresh_ranks": list(driver.refresh_ranks),
                "handoff_latent_bytes": driver.phase_transfer.logical_latent_bytes if driver.phase_transfer else 0,
                "handoff_clean_kv_bytes": driver.phase_transfer.logical_kv_bytes if driver.phase_transfer else 0,
                "execution": "parallel phase batches with a world-wave completion boundary",
            }
            if driver is not None
            else None,
            "latent_slots_per_rank": args.latent_slots,
            "pending_action_slots": args.pending_slots,
            "idle_enabled": True,
            "counts": dict(counts),
            "per_rank_physical_batch_histogram": dict(sorted(batch_histogram.items())),
            "event_lag_max_seconds": max(source_lag, default=0.0),
            "producer": producer,
            "comparison_scope": (
                "direct executor replay; optional per-Job quality policy; no LiveKit or original placement search"
            ),
        }
        temporary = args.output / "result.json.tmp"
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.output / "result.json")

    try:
        pipeline = get_pipeline(
            args.model_root, device_id=rank, pipeline_class=ABotWorldInteractivePipeline, latent_parallel=True
        )
        pipeline.preload_models()
        # T5 and the image encoder are only used at session arrival. Preserve
        # GPU space for retained task KV; keep DiT and the video decoder resident.
        for stage in (pipeline.vae_stage, pipeline.text_encoding_stage):
            stage.model_runtime_config = replace(
                stage.model_runtime_config,
                offload_config=replace(
                    stage.model_runtime_config.offload_config, offload_type=WeightOffloadType.MODEL_CPU_OFFLOAD
                ),
            )
            stage.offload_models()
            stage.onload_models_flag = False
        pipeline.denoise_stage.configure_cuda_graph(False)
        backend = {"sdpa": AttnImplType.TORCH_SDPA, "sage-sm90": AttnImplType.SAGE_ATTN_2_8_8_SM90}[args.attention]
        attention = AttentionConfig.dense_attention(backend)
        pipeline.denoise_stage.model_runtime_config.attention_config = attention
        pipeline.denoise_stage.dit.set_attention_config(attention)
        driver = pipeline.create_continuous_runtime(
            max_batch_size=args.latent_slots,
            granularity="latent_step",
            backend=args.backend,
            refresh_ranks=args.refresh_ranks,
        )
        with Image.open(scenario.session.image_path) as source:
            image = source.convert("RGB")
        if policy is not None:
            from telefuser.pipelines.abot_world.fidelity import ABotWorldFidelity

            calibration_started = time.monotonic()
            # Both pools retain per-session state. Bound startup probes by
            # the smaller pool so calibration cannot allocate an entire
            # world-sized cohort on one refresh GPU before memory budgeting.
            calibration_ranks = (
                min(len(driver.denoise_ranks), len(driver.refresh_ranks))
                if driver.refresh_ranks
                else dist.get_world_size()
            )
            sample_count = max(1, calibration_ranks * args.latent_slots // 3)
            windows = sorted({fidelity.local_attn_size for fidelity in policy.fidelities.values()}, reverse=True)
            prefill = ABotWorldFidelity.from_profile_name(f"b1_s4_w{min(windows)}_rho0_bf16")
            samples = []
            for index in range(sample_count):
                sample = driver.create_session(
                    image,
                    scenario.session.prompt,
                    seed=scenario.seed + 10_000 + index,
                    session_id=f"timing-warmup-{index}",
                )
                # Retain each state immediately: initializing a large pool must
                # not leave canonical full KV replicas on every GPU first.
                driver.submit(sample, {"W": True}, fidelity=prefill)
                samples.append(sample)
            while driver.has_pending:
                step_runtime()
            # Measure full occupancy while all retained rows still share the
            # arena. Grow W in place instead of holding a detached W18 pool
            # beside each smaller partial-batch probe.
            for window in reversed(windows):
                fidelity = ABotWorldFidelity.from_profile_name(f"b1_s4_w{window}_rho0_bf16")
                for _ in range(math.ceil(window / 3) + 1):
                    for sample in samples:
                        driver.submit(sample, {"W": True}, fidelity=fidelity)
                    while driver.has_pending:
                        step_runtime()
            probe_counts = sorted({1, max(1, sample_count // 4), max(1, sample_count // 2)} - {sample_count})
            if probe_counts:
                for sample in samples[max(probe_counts) :]:
                    driver.release_session(sample.session_id, materialize=False)
                samples = samples[: max(probe_counts)]
            for window in windows:
                fidelity = ABotWorldFidelity.from_profile_name(f"b1_s4_w{window}_rho0_bf16")
                # Learn launch amortization at more than one occupancy. These
                # probes run on already-warm KV and stay outside trace time.
                for count in probe_counts:
                    for _ in range(2):
                        for sample in samples[:count]:
                            driver.submit(sample, {"W": True}, fidelity=fidelity)
                        while driver.has_pending:
                            step_runtime()
            # One synchronization is outside the replay, never in dispatch.
            torch.cuda.synchronize()
            update_timings(keep_latest=False)
            calibration = {
                "elapsed_seconds": time.monotonic() - calibration_started,
                "sessions": sample_count,
                "partial_probe_sessions": len(samples),
                "windows": windows,
                **policy.timing_snapshot(),
                "excluded_from_trace_metrics": True,
            }
            for sample in samples:
                driver.release_session(sample.session_id, materialize=False)
            # Reserve the resident model/arena and the measured transient peak.
            # Retained/canonical KV is the only memory placed by the planner.
            free, _ = torch.cuda.mem_get_info()
            allocated = torch.cuda.memory_allocated()
            reclaimable = torch.cuda.memory_reserved() - allocated
            workspace = max(0, torch.cuda.max_memory_allocated() - allocated)
            budgets = torch.zeros(dist.get_world_size(), dtype=torch.int64)
            budgets[rank] = max(0, free + reclaimable - workspace)
            dist.all_reduce(budgets, op=dist.ReduceOp.SUM, group=control_group)
            kv_budget_bytes = tuple(budgets.tolist())
            calibration["kv_budget_bytes"] = kv_budget_bytes
            # Keep the future arena in the budget, but release calibration-only
            # storage before creating the trace's canonical session states.
            driver.close(materialize=False)
            # Calibration has ended and no trace work is active. Return its
            # large allocator segments once; dispatch never clears the cache.
            torch.cuda.empty_cache()
            driver = pipeline.create_continuous_runtime(
                max_batch_size=args.latent_slots,
                granularity="latent_step",
                backend=args.backend,
                refresh_ranks=args.refresh_ranks,
            )
            calibration["scratch_state_released_before_trace"] = True
            if root:
                (args.output / "online-calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
                print(f"Online timing initialized: {calibration}", flush=True)
        if root:
            print("Models ready; starting interaction trace", flush=True)
        # Calibration must not prewarm the benchmark's initial conditioning.
        # The first real session pays the cold encode; subsequent exact-input
        # reuse is the same bounded pipeline optimization used in serving.
        pipeline._conditioning_cache.clear()
        dist.barrier(group=control_group)
        started = time.monotonic()
        next_report = 30.0
        duration = min(plan["duration_seconds"], args.duration or plan["duration_seconds"])
        while True:
            now = shared_time()
            while event_index < len(events) and events[event_index].at <= min(now, duration):
                event = events[event_index]
                event_index += 1
                session_id = event.session_id
                source_lag.append(max(0.0, now - event.at))
                if event.kind == "arrival":
                    sessions[session_id] = driver.create_session(
                        image,
                        scenario.session.prompt,
                        seed=scenario.seed + session_numbers[session_id],
                        session_id=session_id,
                    )
                    states[session_id] = SessionSchedulingState(
                        session_id=session_id,
                        owner_gpu="latent-pool",
                        slack_seconds=0.0,
                        quality_ema=1.0,
                        pending_action_slots=args.pending_slots,
                        last_updated_at=now,
                    )
                    counts["sessions_ready"] += 1
                elif event.kind == "departure":
                    state = states[session_id]
                    counts["actions_dropped_on_departure"] += state.pending_action_count
                    state.mark_departed(now=now)
                elif not states[session_id].departed:
                    state = states[session_id]
                    sequence += 1
                    before = state.pending_action_count
                    state.submit_action(
                        job_id=f"job-{sequence}",
                        controls=event.controls,
                        now=now,
                        release=event.kind == "release",
                        sequence=sequence,
                    )
                    if event.kind == "release":
                        state.defer_idle_until(now + args.collection_window)
                        state.release_idle_video(now=now)
                        counts["actions_released"] += 1
                        if before == args.pending_slots:
                            counts["actions_dropped_pending_capacity"] += 1
            if now >= duration:
                for state in states.values():
                    if not state.departed:
                        counts["actions_dropped_on_departure"] += state.pending_action_count
                        state.mark_departed(now=now)
            # Session creation can use the GPU; resample once before dispatch.
            now = shared_time()
            for state in states.values():
                state.advance_to(now)
            for session_id, state in sorted(states.items(), key=lambda item: item[1].slack_seconds):
                if state.departed and state.in_flight is None:
                    if sessions[session_id].latent_runtime_owner is driver:
                        driver.release_session(session_id, materialize=False)
                    else:
                        pipeline.close_interactive_session(sessions[session_id])
                    del states[session_id], sessions[session_id]
                    counts["sessions_closed"] += 1
                    continue
                if state.departed or state.in_flight is not None:
                    continue
                state.create_idle_job(job_id=f"idle-{sequence}-{session_id}", now=now)
                if policy is not None:
                    continue
                job = state.ready_job(include_idle=True)
                if job is None:
                    continue
                driver.submit(
                    sessions[session_id],
                    {key.removeprefix("Key"): True for key in job.controls},
                )
                chosen_profiles[session_id] = "b1_s4_w18_rho0_bf16"
                job_qualities[session_id] = 1.0
                state.mark_dispatched(job)
                inflight[session_id] = now
            selected_ids = None
            planning_seconds = 0.0
            if policy is not None:
                planning_started = time.monotonic()
                decision = policy.plan(list(states.values()), sessions, driver, kv_budget_bytes=kv_budget_bytes)
                planning_seconds = time.monotonic() - planning_started
                planner_calls += 1
                planner_cpu_seconds += planning_seconds
                planner_cpu_max_seconds = max(planner_cpu_max_seconds, planning_seconds)
                selected_ids = decision.session_ids
                for session_id, choice in decision.admissions.items():
                    state = states[session_id]
                    job = state.ready_job(include_idle=True)
                    driver.submit(
                        sessions[session_id],
                        {key.removeprefix("Key"): True for key in job.controls},
                        fidelity=choice.fidelity,
                        owners=choice.owners,
                        refresh_owners=choice.refresh_owners or None,
                    )
                    chosen_profiles[session_id] = choice.fidelity.name
                    job_qualities[session_id] = choice.quality
                    state.mark_dispatched(job)
                    inflight[session_id] = now
            if not driver.has_pending or selected_ids == ():
                if now >= duration and not states:
                    break
                if root:
                    time.sleep(0.01)
                continue
            completed = step_runtime(selected_ids)
            finished_at = shared_time()
            if root:
                per_rank = Counter(key.owner for key in driver.last_tasks)
                for owner in range(dist.get_world_size()):
                    batch_histogram[per_rank[owner]] += 1
                physical_file.write(
                    json.dumps(
                        {
                            "elapsed_seconds": now,
                            "unit": "whole_latent_step",
                            "planning_seconds": planning_seconds,
                            "predicted_wave_seconds": decision.predicted_wave_seconds if policy is not None else None,
                            "planner_objective": decision.objective if policy is not None else None,
                            "owners": {key: driver._owners[key] for key in driver.last_windows},
                            "refresh_owners": {key: driver._refresh_owners[key] for key in driver.last_windows}
                            if driver.refresh_ranks
                            else {},
                            "phase_items": dict(
                                Counter("refresh" if key.refreshing else "denoise" for key in driver.last_tasks)
                            ),
                            "phase_handoff_latent_bytes_cumulative": driver.phase_transfer.logical_latent_bytes
                            if driver.phase_transfer
                            else 0,
                            "phase_handoff_kv_bytes_cumulative": driver.phase_transfer.logical_kv_bytes
                            if driver.phase_transfer
                            else 0,
                            "admissions": {
                                key: {
                                    "fidelity": choice.fidelity.name,
                                    "quality": choice.quality,
                                    "predicted_completion_seconds": choice.predicted_seconds,
                                }
                                for key, choice in decision.admissions.items()
                            }
                            if policy is not None
                            else {},
                            "per_rank_batch": dict(per_rank),
                            "step_items": dict(Counter(key.step_index for key in driver.last_tasks)),
                            "windows": driver.last_windows,
                            "sessions": len({key.session_id for key in driver.last_tasks}),
                            "pending_actions": sum(state.pending_action_count for state in states.values()),
                            "local_kv_repack_bytes_cumulative": driver.arena.cache_copy_bytes,
                        }
                    )
                    + "\n"
                )
            if completed:
                completed_at = finished_at
                for output in completed:
                    state = states[output.session_id]
                    model_started = inflight.pop(output.session_id)
                    frames = int(output.decoded.shape[2]) if root else 12
                    quality = job_qualities.pop(output.session_id)
                    fidelity = chosen_profiles.pop(output.session_id)
                    job = state.complete_job(completed_at=completed_at, output_seconds=frames / 12, quality=quality)
                    profile_counts[fidelity] += 1
                    counts[f"{job.kind}_jobs_completed"] += 1
                    counts["generated_frames"] += frames
                    if root:
                        dispatch_file.write(
                            json.dumps(
                                {
                                    "event_type": "model_dispatch",
                                    "outcome": "ok",
                                    "batch_size": 1,
                                    "physical_batch_unit": "see latent-step-dispatch.jsonl",
                                    "model_completed_monotonic_seconds": completed_at,
                                    "model_duration_seconds": completed_at - model_started,
                                    "sessions": [
                                        {
                                            "session_id": output.session_id,
                                            "frames": frames,
                                            "fps": 12,
                                            "motivation_kind": job.kind,
                                            "fidelity": fidelity,
                                            "queue_wait_seconds": max(0.0, model_started - job.created_at),
                                            "output_gate_enabled": 0,
                                        }
                                    ],
                                }
                            )
                            + "\n"
                        )
            if now >= next_report:
                write_summary("running", now)
                if root:
                    print(f"trace={now:.1f}/{duration:.0f}s counts={dict(counts)}", flush=True)
                next_report = now + 30
        if policy is not None:
            torch.cuda.synchronize()
            update_timings(keep_latest=False)
        write_summary("complete", shared_time())
    except BaseException:
        write_summary("failed", time.monotonic() - started)
        raise
    finally:
        if dispatch_file is not None:
            dispatch_file.close()
            physical_file.close()
            timing_file.close()
        if driver is not None and not driver.has_pending:
            driver.close(materialize=False)
        if pipeline is not None:
            pipeline.close()
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collection-window", type=float, default=0.5)
    parser.add_argument("--pending-slots", type=int, default=4)
    parser.add_argument("--latent-slots", type=int, default=8)
    parser.add_argument("--backend", choices=("eager", "cuda_graph"), default="eager")
    parser.add_argument("--attention", choices=("sdpa", "sage-sm90"), default="sdpa")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--profile", type=Path, help="Enable per-Job S/W quality selection from this profile CSV")
    parser.add_argument("--lambda-quality", type=float, default=2.0)
    parser.add_argument(
        "--refresh-ranks",
        type=int,
        nargs="+",
        default=(),
        help="Reserve these logical GPU ranks for refresh; remaining ranks run denoise",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.pending_slots < 2 or args.latent_slots < 1 or (args.duration is not None and args.duration <= 0):
        parser.error("Use at least two ordered pending slots, positive latent slots and duration")
    scenario = load_scenario(args.trace)
    events, plan = build_events(scenario, args.collection_window)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
    else:
        replay(args, scenario, events, plan)


if __name__ == "__main__":
    main()
