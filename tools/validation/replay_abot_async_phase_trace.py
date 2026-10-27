"""Replay unchanged interaction inputs through independent D/R subclusters.

The coordinator chooses metric-directed pool sizes, joint session placement
and per-Job quality from measured phase curves. Workers independently batch
causal latent-step heads. No serving iteration performs a world collective.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from collections import Counter
from dataclasses import asdict, replace
from datetime import timedelta
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

from tools.validation.augment_abot_cpr_quality import collect_producer_metrics
from tools.validation.benchmark_abot_livekit_burst import load_scenario
from tools.validation.replay_abot_latent_step_trace import build_events


class PhaseReplayCoordinator:
    """Root-side logical sessions; model and communication never run here."""

    def __init__(
        self, args: Any, scenario: Any, events: list, plan: dict, server: Any, loads: dict, planner: Any, partition: Any
    ) -> None:
        from telefuser.service.livekit.phase_channels import PairHandoffQueue
        from telefuser.service.livekit.phase_pool import SharedPhaseQueue

        self.args, self.scenario, self.events, self.plan = args, scenario, events, plan
        self.server, self.loads, self.planner, self.partition = server, loads, planner, partition
        self.handoffs = PairHandoffQueue(server.send)
        self.states, self.owners, self.reserved, self.active = {}, {}, {}, {}
        self.ready, self.creating, self.closing = set(), set(), {}
        self.admission, self.completed_parts = {}, {}
        self.pending_outputs = Counter()
        self.denoise_queue = SharedPhaseQueue("denoise", partition.denoise_ranks)
        self.refresh_queue = SharedPhaseQueue("refresh", partition.refresh_ranks)
        self.decode_owners, self.state_reservations, self.pending_state = {}, {}, {}
        self.copy_reservations = {}
        self.evicting = set()
        self.transfer_rates = {}
        self.chunk_starts = Counter()
        self.counts, self.profile_counts, self.batch_histogram = Counter(), Counter(), Counter()
        self.transfer_bytes, self.transfer_seconds = Counter(), Counter()
        self.session_numbers = {event.session_id: i for i, event in enumerate(e for e in events if e.kind == "arrival")}
        self.qualities = {profile.fidelity: profile.quality for profile in planner.profiles}
        self.started = time.monotonic()
        self.max_event_lag = 0.0
        self.max_planner_seconds = 0.0
        self.source_index = self.sequence = 0
        self.output = args.output
        self.dispatch = (self.output / "dispatch-trace.jsonl").open("w")
        self.physical = (self.output / "phase-events.jsonl").open("w")

    def now(self) -> float:
        return time.monotonic() - self.started

    def _ingress(self, now: float, duration: float) -> None:
        from telefuser.service.livekit.motivation_scheduler import SessionSchedulingState

        while self.source_index < len(self.events) and self.events[self.source_index].at <= min(now, duration):
            event = self.events[self.source_index]
            self.source_index += 1
            self.max_event_lag = max(self.max_event_lag, now - event.at)
            sid = event.session_id
            if event.kind == "arrival":
                self.states[sid] = SessionSchedulingState(
                    session_id=sid,
                    owner_gpu="phase-pools",
                    slack_seconds=0.0,
                    quality_ema=1.0,
                    pending_action_slots=self.args.pending_slots,
                    last_updated_at=now,
                )
            elif event.kind == "departure":
                self.counts["actions_dropped_on_departure"] += self.states[sid].pending_action_count
                self.states[sid].mark_departed(now=now)
            elif not self.states[sid].departed:
                state = self.states[sid]
                self.sequence += 1
                before = state.pending_action_count
                state.submit_action(
                    job_id=f"job-{self.sequence}",
                    controls=event.controls,
                    now=now,
                    release=event.kind == "release",
                    sequence=self.sequence,
                )
                if event.kind == "release":
                    state.defer_idle_until(now + self.args.collection_window)
                    state.release_idle_video(now=now)
                    self.counts["actions_released"] += 1
                    if before == self.args.pending_slots:
                        self.counts["actions_dropped_pending_capacity"] += 1
        if now >= duration:
            for state in self.states.values():
                if not state.departed:
                    self.counts["actions_dropped_on_departure"] += state.pending_action_count
                    state.mark_departed(now=now)

    def _reserve(self, sid: str, choice: Any) -> None:
        self.owners[sid] = (choice.denoise_rank, choice.refresh_rank)
        self.reserved[sid] = choice.reserved_bytes
        for rank in self.owners[sid]:
            self._reserve_rank(sid, rank, choice.reserved_bytes)

    def _reserve_rank(self, sid: str, rank: int, amount: int) -> None:
        old = self.state_reservations.get((sid, rank), 0)
        self.state_reservations[sid, rank] = max(old, amount)
        self.loads[rank].reserved_bytes += max(0, amount - old)
        self.loads[rank].sessions.add(sid)

    def _state_cost(self, task: Any, rank: int, phase: str) -> Any:
        from telefuser.service.livekit.phase_pool import PhaseStateCost, state_copy_seconds

        sid = task.session_id
        if (sid, rank) in self.evicting:
            return None
        source = self.owners[sid][0 if phase == "denoise" else 1]
        amount = self.reserved[sid]
        extra = max(0, amount - self.state_reservations.get((sid, rank), 0))
        # Paged receives become the installed state without a second dense
        # allocation. Legacy dense destinations still need both copies.
        if rank != source and self.chunk_starts[sid] and not self.loads[rank].memory_stats.get("paged_state"):
            extra += amount
        if self.loads[rank].reserved_bytes + extra > self.loads[rank].kv_budget_bytes:
            return None
        # Only clean history moves on a cold worker. Initial sessions have
        # no history; subsequent warm hits retain the incremental fast path.
        payload = 0 if not self.chunk_starts[sid] else amount
        seconds = state_copy_seconds(
            source=source, destination=rank, byte_count=payload, observations=self.transfer_rates
        )
        return PhaseStateCost(seconds, extra)

    def _reserve_phase_state(self, task: Any, rank: int, phase: str) -> None:
        sid = task.session_id
        self._reserve_rank(sid, rank, self.reserved[sid])
        source = self.owners[sid][0 if phase == "denoise" else 1]
        if rank != source and self.chunk_starts[sid] and not self.loads[rank].memory_stats.get("paged_state"):
            self.copy_reservations[sid, rank] = self.reserved[sid]
            self.loads[rank].reserved_bytes += self.reserved[sid]

    def _move_state(self, sid: str, source: int, target: int, kind: str, continuation: Any) -> None:
        active = self.active[sid]
        self.pending_state[active["job_id"], kind] = source, target, continuation
        self.server.send(
            source,
            {
                "command": "export_state",
                "metadata": {
                    "session_id": sid,
                    "job_id": active["job_id"],
                    "state_kind": kind,
                    "target_rank": target,
                    "seed": self.scenario.seed + self.session_numbers[sid],
                },
            },
        )

    def _evict_state(self, sid: str, rank: int) -> None:
        self.evicting.add((sid, rank))
        self.server.send(rank, {"command": "evict", "session_id": sid})

    def _send_denoise(self, sid: str, rank: int) -> None:
        active = self.active[sid]
        choice, job = active["choice"], active["job"]
        self.owners[sid] = rank, self.owners[sid][1]
        active["choice"] = replace(choice, denoise_rank=rank)
        self.server.send(
            rank,
            {
                "command": "job",
                "metadata": {
                    "session_id": sid,
                    "job_id": job.job_id,
                    "fidelity": choice.fidelity,
                    "quality": choice.quality,
                    "denoise_rank": rank,
                    "refresh_rank": self.owners[sid][1],
                    "decode_rank": self.decode_owners[sid],
                    "actions": {key.removeprefix("Key"): True for key in job.controls},
                    "slack_seconds": active["deadline"] - self.now(),
                    "seed": self.scenario.seed + self.session_numbers[sid],
                },
            },
        )

    def _send_refresh(self, sid: str, record: dict, rank: int) -> None:
        self.owners[sid] = self.owners[sid][0], rank
        record["destination"] = rank
        record["metadata"]["refresh_rank"] = rank
        record["metadata"]["remaining_slack_seconds"] = self.active[sid]["deadline"] - self.now()
        self.handoffs.submit(record)

    def _dispatch_pools(self, now: float) -> None:
        for phase, pool in (("denoise", self.denoise_queue), ("refresh", self.refresh_queue)):
            choices = pool.dispatch(
                self.loads,
                now=now,
                state_cost=lambda task, rank: self._state_cost(task, rank, phase),
                reserve=lambda task, rank: self._reserve_phase_state(task, rank, phase),
            )
            for dispatch in choices:
                for task in dispatch.tasks:
                    sid, rank = task.session_id, dispatch.rank
                    if phase == "denoise":
                        choice, job = task.payload
                        state = self.states[sid]
                        state.mark_dispatched(job)
                        self.active[sid] = {
                            "started": now,
                            "deadline": task.deadline,
                            "choice": choice,
                            "job_id": job.job_id,
                            "job": job,
                            "denoise_seconds": dispatch.predicted_seconds / len(dispatch.tasks),
                        }
                        self.completed_parts[job.job_id] = {"active": self.active[sid]}
                        self.pending_outputs[sid] += 1
                        source = self.owners[sid][0]
                        if rank != source:
                            self._move_state(sid, source, rank, phase, None)
                        else:
                            self._send_denoise(sid, rank)
                    else:
                        self.active[sid]["refresh_seconds"] = dispatch.predicted_seconds / len(dispatch.tasks)
                        source = self.owners[sid][1]
                        if rank != source and self.chunk_starts[sid]:
                            self._move_state(sid, source, rank, phase, task.payload)
                        else:
                            if rank != source:
                                self._evict_state(sid, source)
                            self._send_refresh(sid, task.payload, rank)

    def _advance(self, now: float) -> None:
        from telefuser.service.livekit.phase_pool import PhasePoolTask

        for sid, state in list(self.states.items()):
            state.advance_to(now)
            if state.departed:
                self.denoise_queue.cancel(sid)
            if sid in self.closing:
                continue
            if (
                state.departed
                and state.in_flight is None
                and sid not in self.creating
                and not self.pending_outputs[sid]
            ):
                if sid not in self.owners:
                    del self.states[sid]
                    continue
                self.closing[sid] = {rank for (key, rank) in self.state_reservations if key == sid} | {
                    self.decode_owners[sid]
                }
                for rank in self.closing[sid]:
                    self.server.send(rank, {"command": "close", "session_id": sid})
                continue
            if state.departed or state.in_flight is not None or sid in self.creating:
                continue
            state.create_idle_job(job_id=f"idle-{self.sequence}-{sid}", now=now)
            if sid in self.owners and sid not in self.ready:
                continue
            job = state.ready_job(include_idle=True)
            if sid in self.owners and job is None:
                continue
            # Admission selects quality against the warm state owners. The
            # shared phase queue then compares workers including copy costs.
            started = time.monotonic()
            choice = self.planner.admit(
                state,
                self.loads,
                self.partition,
                owners=self.owners.get(sid),
                retained_bytes=self.reserved.get(sid, 0),
                action_age_seconds=max(0.0, now - job.created_at) if job is not None and job.kind == "action" else None,
            )
            self.max_planner_seconds = max(self.max_planner_seconds, time.monotonic() - started)
            if choice is None:
                self.counts["capacity_wait_iterations"] += 1
                continue
            if sid not in self.owners:
                # Creation installs conditioning, not a chosen Job's KV.
                # Reserve the minimum resident row until real dispatch fixes
                # S/W, then grow to that actual allocation's high-water mark.
                minimum_window = min(f.local_attn_size for f in self.planner.fidelities.values())
                initial_bytes = max(
                    self.loads[rank].session_bytes(minimum_window)
                    for rank in (choice.denoise_rank, choice.refresh_rank)
                )
                self._reserve(sid, replace(choice, reserved_bytes=initial_bytes))
                self.decode_owners[sid] = choice.denoise_rank
                self.creating.add(sid)
                self.server.send(
                    choice.denoise_rank,
                    {
                        "command": "create",
                        "session_id": sid,
                        "image_path": self.scenario.session.image_path,
                        "prompt": self.scenario.session.prompt,
                        "seed": self.scenario.seed + self.session_numbers[sid],
                    },
                )
                continue
            if job is None:
                continue
            self._reserve(sid, choice)
            # Keep the causal head in the shared pool until an actor has
            # credit. Quality is refreshed while queued, not frozen forever.
            self.denoise_queue.cancel(sid)
            fidelity = self.planner.fidelities[choice.fidelity]
            self.denoise_queue.submit(
                PhasePoolTask(
                    sid,
                    job.job_id,
                    fidelity.local_attn_size,
                    fidelity.denoise_steps,
                    now + choice.slack_seconds,
                    payload=(choice, job),
                )
            )
        self._dispatch_pools(now)

    def _event(self, record: dict[str, Any]) -> None:
        event, rank = record["event"], record["rank"]
        now = self.now()
        self.physical.write(json.dumps(dict(record, elapsed_seconds=now)) + "\n")
        if event == "error":
            raise RuntimeError(f"GPU worker {rank} failed:\n{record['message']}")
        if event == "session_ready":
            self.creating.remove(record["session_id"])
            self.ready.add(record["session_id"])
            self.counts["sessions_ready"] += 1
        elif event == "handoff_ready":
            sid = record["metadata"]["session_id"]
            phase = record["phase"]
            if phase in {"latent", "kv"}:
                kind = "denoise" if phase == "latent" else "refresh"
                seconds = self.active[sid][f"{kind}_seconds"]
                self.loads[rank].outstanding_seconds = max(0.0, self.loads[rank].outstanding_seconds - seconds)
                self.loads[rank].queued_jobs -= 1
                (self.denoise_queue if phase == "latent" else self.refresh_queue).complete(sid, rank)
            if phase == "latent":
                from telefuser.service.livekit.phase_pool import PhasePoolTask

                fidelity = self.planner.fidelities[record["metadata"]["fidelity"]]
                self.refresh_queue.submit(
                    PhasePoolTask(
                        sid,
                        record["metadata"]["job_id"],
                        fidelity.local_attn_size,
                        1,
                        self.active[sid]["deadline"],
                        payload=record,
                    )
                )
            else:
                self.handoffs.submit(record)
        elif event == "state_ready":
            sid, kind = record["session_id"], record["kind"]
            source, target, continuation = self.pending_state.pop((record["job_id"], kind))
            self.loads[target].reserved_bytes -= self.copy_reservations.pop((sid, target), 0)
            self._evict_state(sid, source)
            self.counts[f"{kind}_owner_changes"] += 1
            if kind == "denoise":
                self._send_denoise(sid, target)
            else:
                self._send_refresh(sid, continuation, target)
        elif event == "state_evicted":
            sid = record["session_id"]
            self.evicting.discard((sid, rank))
            self.loads[rank].reserved_bytes -= self.state_reservations.pop((sid, rank), 0)
            self.loads[rank].sessions.discard(sid)
        elif event == "transfer_complete":
            self.handoffs.complete(record["transfer_id"], rank)
            if rank == record["destination"]:
                phase, seconds = record["phase"], record["total_duration_ms"] / 1000
                if phase in self.planner.transfer_seconds:
                    previous = self.planner.transfer_seconds[phase]
                    self.planner.transfer_seconds[phase] = previous + 0.2 * (seconds - previous)
                pair = record["source"], record["destination"]
                amount, elapsed = self.transfer_rates.get(pair, (0, 0.0))
                self.transfer_rates[pair] = amount + record["total_bytes"], elapsed + seconds
                self.transfer_bytes[phase] += record["total_bytes"]
                self.transfer_seconds[phase] += seconds
                self.counts[f"{phase}_transfers"] += 1
        elif event == "phase_batch":
            if len(record["windows"]) == 1:
                self.loads[rank].observe(record["phase"], record["windows"][0], record["rows"], record["seconds"])
            self.batch_histogram[f"{rank}:{record['phase']}:rows{record['rows']}"] += 1
        elif event == "vae_batch":
            seconds = record["seconds"] / record["batch_size"]
            self.planner.decode_seconds += 0.2 * (seconds - self.planner.decode_seconds)
            self.counts["vae_batches"] += 1
        elif event in {"decoded", "kv_committed"}:
            parts = self.completed_parts[record["job_id"]]
            parts[event] = dict(record, at=now)
            if event == "decoded":
                sid = record["session_id"]
                active = parts["active"]
                frames = record["frames"]
                self.states[sid].record_generated_output(
                    active["job"], completed_at=now, output_seconds=frames / 12, quality=active["choice"].quality
                )
                self.pending_outputs[sid] -= 1
                self.profile_counts[active["choice"].fidelity] += 1
                self.counts[f"{active['job'].kind}_jobs_completed"] += 1
                self.counts["generated_frames"] += frames
            if event == "kv_committed":
                # Clean state is the causal execution boundary. RGB remains
                # separately tracked and cannot hold the next DiT Job idle.
                sid = record["session_id"]
                if record.get("refresh_state_bytes") is not None:
                    # The matching clean-state completion is a safe boundary
                    # to return unused/shrunk physical-state reservations.
                    # No later Job for this session has been admitted yet.
                    sizes = (record["state_bytes"], record["refresh_state_bytes"])
                    for owner, amount in zip(self.owners[sid], sizes, strict=True):
                        old = self.state_reservations[sid, owner]
                        self.state_reservations[sid, owner] = amount
                        self.loads[owner].reserved_bytes += amount - old
                    self.reserved[sid] = max(sizes)
                del self.active[sid]
                self.states[sid].in_flight = None
                self.chunk_starts[sid] += 3
            if "decoded" in parts and "kv_committed" in parts:
                self._complete(record["session_id"], parts)
        elif event == "session_closed":
            sid = record["session_id"]
            self.closing[sid].remove(rank)
            self.loads[rank].sessions.discard(sid)
            self.loads[rank].reserved_bytes -= self.state_reservations.pop((sid, rank), 0)
            if not self.closing[sid]:
                for mapping in (self.closing, self.states, self.owners, self.reserved, self.chunk_starts):
                    del mapping[sid]
                del self.pending_outputs[sid]
                del self.decode_owners[sid]
                self.ready.discard(sid)
                self.counts["sessions_closed"] += 1

    def _complete(self, sid: str, parts: dict[str, Any]) -> None:
        active = parts["active"]
        choice = active["choice"]
        # Record both boundaries, but credit playout at RGB arrival rather
        # than delaying available video behind the clean-state commit.
        frames = parts["decoded"]["frames"]
        job = active["job"]
        decoded_at = parts["decoded"]["at"]
        self.dispatch.write(
            json.dumps(
                {
                    "event_type": "model_dispatch",
                    "outcome": "ok",
                    "batch_size": 1,
                    "physical_batch_unit": "see phase-events.jsonl",
                    "model_completed_monotonic_seconds": decoded_at,
                    "model_duration_seconds": decoded_at - active["started"],
                    "clean_state_committed_seconds": parts["kv_committed"]["at"],
                    "sessions": [
                        {
                            "session_id": sid,
                            "frames": frames,
                            "fps": 12,
                            "motivation_kind": job.kind,
                            "fidelity": choice.fidelity,
                            "queue_wait_seconds": max(0.0, active["started"] - job.created_at),
                            "output_gate_enabled": 0,
                        }
                    ],
                }
            )
            + "\n"
        )
        del self.completed_parts[active["job_id"]]

    def summary(self, status: str) -> None:
        self.dispatch.flush()
        self.physical.flush()
        producer = collect_producer_metrics(
            self.output / "dispatch-trace.jsonl",
            self.qualities,
            released_action_jobs=self.counts["actions_released"],
            superseded_action_jobs=0,
            effective_action_jobs=self.counts["actions_released"] - self.counts["actions_dropped_pending_capacity"],
        )
        value = {
            "status": status,
            "experiment": "independent_metric_directed_phase_pools",
            "elapsed_seconds": self.now(),
            "input_plan": self.plan,
            "partition": asdict(self.partition),
            "counts": dict(self.counts),
            "producer": producer,
            "profile_job_counts": dict(self.profile_counts),
            "per_rank_physical_batch_histogram": dict(self.batch_histogram),
            "phase_transfer_bytes": dict(self.transfer_bytes),
            "phase_transfer_seconds": dict(self.transfer_seconds),
            "online_transfer_seconds": self.planner.transfer_seconds,
            "online_decode_seconds": self.planner.decode_seconds,
            "event_lag_max_seconds": self.max_event_lag,
            "planner_cpu_max_seconds": self.max_planner_seconds,
            "world_size": len(self.loads),
            "attention": self.args.attention,
            "backend": self.args.backend,
            "collection_window_seconds": self.args.collection_window,
            "pending_action_slots": self.args.pending_slots,
            "latent_slots_per_rank": self.args.latent_slots,
            "idle_enabled": True,
            "measurement_boundary": "actual TAeW RGB completion; client/output conversion excluded",
            "execution": "independent GPU loops; async P2P incremental SST; no serving world-wave barrier",
            "partition_lifecycle": "logical D/R pools; shared stage queues; costed boundary state placement",
            "pool_queues": {
                "denoise_pending": len(self.denoise_queue.pending),
                "refresh_pending": len(self.refresh_queue.pending),
                "denoise_active": len(self.denoise_queue.active),
                "refresh_active": len(self.refresh_queue.active),
            },
        }
        temporary = self.output / "result.json.tmp"
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(self.output / "result.json")

    def run(self) -> None:
        duration = min(self.plan["duration_seconds"], self.args.duration or self.plan["duration_seconds"])
        self.started = time.monotonic()
        next_report = 30.0
        try:
            while True:
                now = self.now()
                self._ingress(now, duration)
                # Drain only a bounded batch; neither constant timings nor
                # sustained ingress can starve causal admission and controls.
                for _ in range(256):
                    try:
                        self._event(self.server.events.get_nowait())
                    except queue.Empty:
                        break
                self._advance(self.now())
                if now >= duration and not self.states:
                    break
                if now >= next_report:
                    self.summary("running")
                    print(f"trace={now:.1f}/{duration:.0f}s counts={dict(self.counts)}", flush=True)
                    next_report = now + 30
                try:
                    self._event(self.server.events.get(timeout=0.005))
                except queue.Empty:
                    pass
            self.summary("complete")
        except BaseException:
            self.summary("failed")
            raise
        finally:
            for rank in self.loads:
                self.server.send(rank, {"command": "stop"})
            self.dispatch.close()
            self.physical.close()


def replay(args: Any, scenario: Any, events: list, plan: dict) -> None:
    import torch
    import torch.distributed as dist
    from PIL import Image

    from examples.abot_world._loader import get_pipeline
    from telefuser.core.config import AttentionConfig, AttnImplType, WeightOffloadType
    from telefuser.pipelines.abot_world.interactive import ABotWorldInteractivePipeline
    from telefuser.service.livekit.abot_phase_policy import ABotPhasePlanner
    from telefuser.service.livekit.abot_phase_runtime import ABotPhaseWorker, calibrate_phase_worker
    from telefuser.service.livekit.motivation_scheduler import load_motivation_profiles_csv
    from telefuser.service.livekit.phase_channels import PhaseControlServer
    from telefuser.service.livekit.phase_transport import PhasePeerTransport

    local_rank, rank = int(os.environ["LOCAL_RANK"]), int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    control = dist.new_group(backend="gloo", timeout=timedelta(minutes=10))
    # Initialize the NCCL communicator collectively once. Subsequent point-
    # to-point operations need no all-rank participation, including first use.
    dist.all_reduce(torch.ones(1, device=f"cuda:{local_rank}"))
    transport = PhasePeerTransport.bootstrap(torch.device(f"cuda:{local_rank}"))
    server = PhaseControlServer("0.0.0.0", os.urandom(32)) if rank == 0 else None
    endpoint = [(os.environ["MASTER_ADDR"], server.listener.address[1], server.listener._authkey)] if server else [None]
    dist.broadcast_object_list(endpoint, src=0, group=control)
    host, port, authkey = endpoint[0]
    accept_thread = threading.Thread(target=server.accept, args=(dist.get_world_size(),)) if server else None
    if accept_thread:
        accept_thread.start()
    connector = Client((host, port), authkey=authkey)
    connector.send({"rank": rank})
    if server:
        accept_thread.join()
    pipeline = get_pipeline(args.model_root, device_id=local_rank, pipeline_class=ABotWorldInteractivePipeline)
    pipeline.preload_models()
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
    profiles = load_motivation_profiles_csv(args.profile).profiles_for(batch_size=1, gpu_id="phase-pools")
    planner = ABotPhasePlanner(profiles, lambda_quality=args.lambda_quality)
    windows = sorted({f.local_attn_size for f in planner.fidelities.values()})
    with Image.open(scenario.session.image_path) as source:
        image = source.convert("RGB")
    load = calibrate_phase_worker(
        pipeline,
        image,
        scenario.session.prompt,
        rank=rank,
        slots=args.latent_slots,
        windows=windows,
        backend=args.backend,
    )
    connector.send({"event": "calibration", "load": asdict(load) | {"sessions": []}})
    actor = ABotPhaseWorker(pipeline, connector, rank=rank, load=load, backend=args.backend, transport=transport)
    if server:
        thread = threading.Thread(target=actor.run, name="rank0-gpu-actor", daemon=True)
        thread.start()
        from telefuser.service.livekit.abot_phase_policy import PhaseWorkerLoad

        loads = {}
        while len(loads) < dist.get_world_size():
            record = server.events.get(timeout=600)
            if record["event"] != "calibration":
                raise RuntimeError(f"Unexpected startup event {record}")
            value = record["load"]
            value["sessions"] = set(value["sessions"])
            loads[record["rank"]] = PhaseWorkerLoad(**value)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "worker-calibration.json").write_text(
            json.dumps({r: asdict(worker_load) | {"sessions": []} for r, worker_load in loads.items()}, indent=2) + "\n"
        )
        print(
            "Worker memory capacities: "
            + str(
                {
                    r: {
                        "budget_GiB": worker_load.kv_budget_bytes / 2**30,
                        "canonical_w6_GiB": worker_load.session_bytes(6) / 2**30,
                    }
                    for r, worker_load in loads.items()
                }
            ),
            flush=True,
        )
        partition = planner.partition(
            loads,
            resident_sessions=plan["sessions"],
            request_rate=plan["phase_action_releases"] / plan["duration_seconds"],
        )
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "online-calibration.json").write_text(
            json.dumps(
                {
                    "loads": {r: asdict(worker_load) | {"sessions": []} for r, worker_load in loads.items()},
                    "partition": asdict(partition),
                    "excluded_from_trace_metrics": True,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Metric-directed partition: {partition}", flush=True)
        for worker_rank in loads:
            role = "denoise" if worker_rank in partition.denoise_ranks else "refresh"
            loads[worker_rank].role = role
            server.send(worker_rank, {"command": "configure", "role": role, "pool_scheduling": True})
        PhaseReplayCoordinator(args, scenario, events, plan, server, loads, planner, partition).run()
        stopped = set()
        while len(stopped) < len(loads):
            record = server.events.get(timeout=120)
            if record["event"] == "error":
                raise RuntimeError(record["message"])
            if record["event"] == "stopped":
                stopped.add(record["rank"])
        thread.join(timeout=10)
        server.close()
    else:
        actor.run()
    pipeline.close()
    transport.close()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collection-window", type=float, default=0.5)
    parser.add_argument("--pending-slots", type=int, default=4)
    parser.add_argument("--latent-slots", type=int, default=12)
    parser.add_argument("--backend", choices=("eager", "cuda_graph"), default="eager")
    parser.add_argument("--attention", choices=("sdpa", "sage-sm90"), default="sdpa")
    parser.add_argument("--lambda-quality", type=float, default=2.0)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.latent_slots < 3 or args.pending_slots < 2 or args.collection_window <= 0:
        parser.error("Use >=3 latent slots, >=2 FIFO pending slots and a positive input window")
    if args.duration is not None and args.duration <= 0:
        parser.error("Duration must be positive")
    scenario = load_scenario(args.trace)
    events, plan = build_events(scenario, args.collection_window)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
    else:
        replay(args, scenario, events, plan)


if __name__ == "__main__":
    main()
