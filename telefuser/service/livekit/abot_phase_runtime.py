"""Independent ABot denoise/refresh actors, local batching and incremental SST.

Each process owns one GPU and a local latent-step executor. Control commands
never call a world collective. NCCL handoffs run on communication threads;
TAeW runs on its own batching thread and stream. The actor only handles its
own causal heads, not a synchronized list of all workers' tasks.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from multiprocessing.connection import Connection
from typing import Any

import torch

from telefuser.pipelines.abot_world.async_decode import _AsyncVAEDecodeQueue
from telefuser.pipelines.abot_world.continuous import ABotWorldDecodedBatch
from telefuser.pipelines.abot_world.fidelity import ABotWorldFidelity

from .abot_phase_policy import PhaseWorkerLoad, select_phase_batch
from .nccl_transfer import (
    PersistentNCCLTransferLane,
    PersistentTensorPagePool,
    TensorTransferGroup,
    flatten_tensor_tree,
    rebuild_tensor_tree,
    transfer_tensor_leaves_nccl_streamed,
)
from .phase_transport import PhasePeerTransport


def _discard_decode(latents: torch.Tensor, session: Any) -> Future:
    result = Future()
    result.set_result(ABotWorldDecodedBatch(latents.new_empty((1, 3, 12, 0, 0)), 12))
    return result


@torch.inference_mode()
def calibrate_phase_worker(
    pipeline: Any,
    image: Any,
    prompt: str,
    *,
    rank: int,
    slots: int,
    windows: list[int],
    backend: str,
) -> PhaseWorkerLoad:
    """Probe independent phase curves at warm KV occupancy outside trace time."""
    runtime = pipeline.create_continuous_runtime(
        max_batch_size=slots,
        granularity="latent_step",
        backend=backend,
        local_execution=True,
        decode_submitter=_discard_decode,
    )
    samples = [
        runtime.create_session(image, prompt, seed=17000 + i, session_id=f"probe-{i}") for i in range(slots // 3)
    ]
    timings: dict[str, dict[int, float]] = {}
    device = pipeline.device
    torch.cuda.reset_peak_memory_stats(device)
    bytes_per_frame = cross_bytes = 0
    previous_window = None
    for window in sorted(windows):
        fidelity = ABotWorldFidelity.from_profile_name(f"b1_s4_w{window}_rho0_bf16")
        if previous_window is not None and len(samples) > 1:
            # Measure arena growth while another cohort is still live, rather
            # than assuming every session parks before all W changes at once.
            smaller = ABotWorldFidelity.from_profile_name(f"b1_s4_w{previous_window}_rho0_bf16")
            for index, sample in enumerate(samples):
                selected = fidelity if index == 0 else smaller
                runtime.ensure_local_capacity(sample.session_id, selected)
                runtime.submit(sample, {"W": True}, fidelity=selected)
            runtime.step(tuple(sample.session_id for sample in samples[1:]))
            runtime.step((samples[0].session_id,))
            while runtime.has_pending:
                for output in runtime.step():
                    runtime.park_session(output.session_id)
        for _ in range((window + 2) // 3 + 1):
            for sample in samples:
                runtime.ensure_local_capacity(sample.session_id, fidelity)
                runtime.submit(sample, {"W": True}, fidelity=fidelity)
            while runtime.has_pending:
                runtime.step()
            for sample in samples:
                runtime.park_session(sample.session_id)
        for count in sorted({1, max(1, len(samples) // 2), len(samples)}):
            observations: dict[str, list[float]] = {"denoise": [], "refresh": []}
            for _ in range(2):
                for sample in samples[:count]:
                    runtime.ensure_local_capacity(sample.session_id, fidelity)
                    runtime.submit(sample, {"W": True}, fidelity=fidelity)
                while runtime.has_pending:
                    phase = "refresh" if next(iter(runtime._requests.values())).refreshing else "denoise"
                    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    start.record()
                    runtime.step()
                    end.record()
                    end.synchronize()  # Calibration only; serving uses event.query().
                    observations[phase].append(start.elapsed_time(end) / 1000)
                for sample in samples[:count]:
                    runtime.park_session(sample.session_id)
            for phase, values in observations.items():
                timings.setdefault(f"{phase}_w{window}", {})[count * 3] = sum(values) / len(values)
        state = samples[0]
        shape = samples[0].first_frame_latent.shape
        patch = pipeline.denoise_stage.dit.patch_size
        tokens = (shape[-2] // patch[1]) * (shape[-1] // patch[2])
        bytes_per_frame = sum(
            entry[name][:, :tokens].numel() * entry[name].element_size()
            for entry in state.self_cache
            for name in ("k", "v")
        )
        cross_bytes = sum(
            entry[name].numel() * entry[name].element_size() for entry in state.cross_cache for name in ("k", "v")
        )
        paged = runtime._paged_states.get(state.session_id)
        if paged is not None:
            page = runtime.kv_pages.page(paged.table[0])
            bytes_per_frame = page.numel() * page.element_size()
            page = runtime.kv_pages.page(paged.cross_page)
            cross_bytes = page.numel() * page.element_size()
        previous_window = window
    allocated_with_state = torch.cuda.memory_allocated(device)
    workspace = max(0, torch.cuda.max_memory_allocated(device) - allocated_with_state)
    for sample in samples:
        runtime.release_session(sample.session_id, materialize=False)
    runtime.kv_pages.trim_free()
    free, _ = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reclaimable = torch.cuda.memory_reserved(device) - allocated
    # Keep the arena and peak workspace out of the resident-state budget.
    # Two simultaneous handoff buffers are real allocations too.
    transfer_reserve = max(
        (slots // 3 + 2) * (3 * bytes_per_frame + cross_bytes),
        2 * (max(windows) * bytes_per_frame + cross_bytes),
    )
    page_cache_reserve = PersistentTensorPagePool().max_cached_bytes
    budget = max(0, free + reclaimable - workspace - transfer_reserve - page_cache_reserve)
    memory_stats = {
        "free_after_session_release": free,
        "allocated_after_session_release": allocated,
        "reclaimable_after_session_release": reclaimable,
        "workspace_peak_delta": workspace,
        "allocated_with_canonical_states": allocated_with_state,
        "transfer_reserve": transfer_reserve,
        "page_cache_reserve": page_cache_reserve,
        "kv_budget": budget,
        "latent_query_slots": slots,
        "active_history_rows": slots // 3,
        "paged_state": 1,
    }
    runtime.close(materialize=False)
    del runtime, samples
    torch.cuda.empty_cache()
    pipeline._conditioning_cache.clear()
    return PhaseWorkerLoad(rank, slots, budget, bytes_per_frame, cross_bytes, timings, memory_stats=memory_stats)


class ABotPhaseWorker:
    """A work-conserving local executor with asynchronous phase handoffs."""

    def __init__(
        self,
        pipeline: Any,
        connection: Connection,
        *,
        rank: int,
        load: PhaseWorkerLoad,
        backend: str = "eager",
        transport: PhasePeerTransport | None = None,
    ) -> None:
        self.pipeline, self.connection, self.rank, self.load = pipeline, connection, rank, load
        self.device = torch.device(pipeline.device)
        self.commands: queue.Queue[dict[str, Any]] = queue.Queue()
        self._send_lock = threading.Lock()
        self._submission_lock = threading.Lock()
        self._transfers = ThreadPoolExecutor(max_workers=4, thread_name_prefix="phase-transfer")
        self._pages = PersistentTensorPagePool()
        self._lane = PersistentNCCLTransferLane(self.device)
        self._transport = transport
        self._exports: dict[str, dict[tuple, torch.Tensor]] = {}
        self._export_events: dict[str, torch.cuda.Event] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._waiting: deque[dict[str, Any]] = deque()
        self._sessions: dict[str, Any] = {}
        self._pending_events: deque[tuple[Any, Any, dict[str, Any]]] = deque()
        self._stopping = threading.Event()
        self.pool_scheduling = False
        self._decode_states: dict[str, Any] = {}
        self._decode_next: dict[str, int] = {}
        self._decode_waiting: dict[str, dict[int, tuple[dict, torch.Tensor]]] = {}
        self.decode = _AsyncVAEDecodeQueue(
            pipeline.taew_decode_stage,
            max_batch_size=max(1, load.slots // 3),
            on_batch=lambda record: self.emit("vae_batch", **record),
        )
        self.runtime = pipeline.create_continuous_runtime(
            max_batch_size=load.slots,
            granularity="latent_step",
            backend=backend,
            local_execution=True,
            decode_submitter=lambda latents, session: (
                _discard_decode(latents, session)
                if self.pool_scheduling
                else self.decode.submit(latents, session.session_id, session.taew_decode_state)
            ),
        )
        self.runtime.kv_pages.max_bytes = load.kv_budget_bytes

    def emit(self, event: str, **record: Any) -> None:
        with self._send_lock:
            self.connection.send(dict(record, event=event))

    def _read(self) -> None:
        try:
            while not self._stopping.is_set():
                command = self.connection.recv()
                command["received_at"] = time.monotonic()
                if command["command"] in {"send", "receive"}:
                    self._transfers.submit(self._transfer, command)
                else:
                    self.commands.put(command)
                    if command["command"] == "stop":
                        return
        except BaseException:
            self.commands.put({"command": "error", "message": traceback.format_exc()})

    @torch.inference_mode()
    def _transfer(self, command: dict[str, Any]) -> None:
        try:
            torch.cuda.set_device(self.device)
            send = command["command"] == "send"
            peer = command["destination"] if send else command["source"]
            transfer_id = command["transfer_id"]
            if send:
                leaves = self._exports[transfer_id]
                slot = None
            else:
                lane = self._transport.lane if self._transport is not None else self._lane
                # Allocate on the consumer stream. A fresh allocation on the
                # default stream may reuse a block whose earlier model kernels
                # still read it; NCCL must not overwrite it from another stream.
                paged_state = command["phase"] == "state" and any(
                    entry["path"][0] == "paged_self" for entry in command["manifest"]
                )
                with torch.cuda.stream(lane.stream_for(peer)):
                    ordinary = [
                        entry
                        for entry in command["manifest"]
                        if not paged_state or entry["path"][0] not in {"paged_self", "paged_cross"}
                    ]
                    leaves, slot = self._pages.acquire(ordinary, self.device)
                    if paged_state:
                        for entry in command["manifest"]:
                            if entry["path"][0] not in {"paged_self", "paged_cross"}:
                                continue
                            page_id = self.runtime.kv_pages.acquire(
                                tuple(entry["shape"]), dtype=getattr(torch, entry["dtype"]), device=self.device
                            )
                            leaves[tuple(entry["path"])] = self.runtime.kv_pages.page(page_id)
            groups = (TensorTransferGroup("phase_payload", tuple(tuple(e["path"]) for e in command["manifest"])),)
            ready = self._export_events[transfer_id] if send else None
            if self._transport is not None:
                report = self._transport.transfer(leaves, peer=peer, send=send, groups=groups, ready=ready)
            else:
                report = transfer_tensor_leaves_nccl_streamed(
                    leaves,
                    peer_rank=peer,
                    send=send,
                    groups=groups,
                    stream=self._lane.stream_for(peer),
                    submission_lock=self._submission_lock,
                    source_ready_event=ready,
                )
            if send:
                del self._exports[transfer_id]
                del self._export_events[transfer_id]
            else:
                payload = rebuild_tensor_tree(command["skeleton"], leaves)
                if command["phase"] == "latent":
                    command["metadata"]["remaining_slack_seconds"] -= time.monotonic() - command["received_at"]
                self.commands.put(dict(command, command="import", payload=payload, page_slot=slot))
            self.emit(
                "transfer_complete",
                transfer_id=transfer_id,
                phase=command["phase"],
                source=command["source"],
                destination=command["destination"],
                **report.as_dict(),
            )
        except BaseException:
            self.commands.put({"command": "error", "message": traceback.format_exc()})

    def _export(self, payload: dict[str, Any], metadata: dict[str, Any], phase: str) -> None:
        skeleton, manifest, leaves = flatten_tensor_tree(payload)
        transfer_id = f"{metadata['job_id']}:{phase}"
        if phase == "state":
            transfer_id += f":{metadata['state_kind']}"
        self._exports[transfer_id] = leaves
        ready = torch.cuda.Event()
        ready.record()
        self._export_events[transfer_id] = ready
        self.emit(
            "handoff_ready",
            transfer_id=transfer_id,
            phase=phase,
            source=self.rank,
            destination=metadata[
                {"latent": "refresh_rank", "kv": "denoise_rank", "decode": "decode_rank", "state": "target_rank"}[phase]
            ],
            skeleton=skeleton,
            manifest=manifest,
            metadata=metadata,
        )

    def _queue_decode(self, metadata: dict[str, Any], latents: torch.Tensor) -> None:
        sid, chunk = metadata["session_id"], metadata["chunk_start"]
        waiting = self._decode_waiting.setdefault(sid, {})
        if chunk < self._decode_next[sid] or chunk in waiting:
            raise RuntimeError("A duplicate or stale chunk reached its decoder owner")
        # A later chunk from another D worker may arrive first. Keep only
        # latent tensors here, with the session's decoder on its own owner.
        waiting[chunk] = metadata, latents.clone() if chunk != self._decode_next[sid] else latents
        cursor = self._decode_next[sid]
        while cursor in waiting:
            item, value = waiting.pop(cursor)
            future = self.decode.submit(value, sid, self._decode_states[sid])
            future.add_done_callback(lambda result, m=item: self._decoded(m, result))
            cursor += value.shape[2]
        self._decode_next[sid] = cursor

    def _decoded(self, metadata: dict[str, Any], future: Future) -> None:
        try:
            output = future.result()
            self.emit(
                "decoded",
                session_id=metadata["session_id"],
                job_id=metadata["job_id"],
                frames=output.frames_per_session,
            )
        except BaseException:
            self.commands.put({"command": "error", "message": traceback.format_exc()})

    @torch.inference_mode()
    def _command(self, command: dict[str, Any]) -> bool:
        kind = command["command"]
        if kind == "stop":
            return False
        if kind == "error":
            raise RuntimeError(command["message"])
        if kind == "configure":
            self.load.role = command["role"]
            self.pool_scheduling = command.get("pool_scheduling", False)
        elif kind == "create":
            from PIL import Image

            with Image.open(command["image_path"]) as source:
                session = self.runtime.create_session(
                    source.convert("RGB"), command["prompt"], seed=command["seed"], session_id=command["session_id"]
                )
            self._sessions[session.session_id] = session
            if self.pool_scheduling:
                self._decode_states[session.session_id] = session.taew_decode_state
                self._decode_next[session.session_id] = 0
                session.taew_decode_state = None
            self.emit("session_ready", session_id=session.session_id)
        elif kind == "job":
            metadata = dict(command["metadata"])
            # Deadlines are durations on the wire; node clocks need not agree.
            metadata["deadline"] = time.monotonic() + metadata.pop("slack_seconds")
            self._waiting.append({"phase": "denoise", "metadata": metadata})
        elif kind == "import":
            metadata, payload = dict(command["metadata"]), command["payload"]
            session_id = metadata["session_id"]
            if command["phase"] == "state":
                session = self.runtime.import_canonical_state(
                    session_id, payload, seed=metadata["seed"], adopt_pages=True
                )
                self._sessions[session_id] = session
            elif command["phase"] == "decode":
                self._queue_decode(metadata, payload["latents"])
            elif command["phase"] == "latent":
                # Carry remaining deadline, not a remote monotonic timestamp.
                metadata["deadline"] = time.monotonic() + metadata.pop("remaining_slack_seconds")
                self._waiting.append(dict(command, metadata=metadata))
                return True
            else:
                self.runtime.commit_external_refresh(session_id, payload["kv"])
                self.emit(
                    "kv_committed",
                    session_id=session_id,
                    job_id=metadata["job_id"],
                    state_bytes=self.runtime._paged_states[session_id].resident_bytes(),
                    refresh_state_bytes=metadata.get("refresh_state_bytes"),
                )
            # submit_refresh clones and commit copies on the model stream.
            # Reuse receive pages only once those reads complete, without
            # synchronizing compute or holding the communication lane.
            fence = torch.cuda.Event()
            fence.record()
            completion = {"page_slot": command["page_slot"]}
            if command["phase"] == "state":
                completion["state_ready"] = dict(
                    session_id=session_id, job_id=metadata["job_id"], kind=metadata["state_kind"]
                )
            self._pending_events.append((None, fence, completion))
        elif kind == "export_state":
            metadata = command["metadata"]
            self._export(self.runtime.export_canonical_state(metadata["session_id"]), metadata, "state")
        elif kind in {"close", "evict"}:
            session_id = command["session_id"]
            if session_id in self._sessions:
                self.runtime.release_session(session_id, materialize=False)
                del self._sessions[session_id]
            if kind == "close" and session_id in self._decode_states:
                self.pipeline.taew_decode_stage.clear_decode_state(self._decode_states.pop(session_id))
                self._decode_next.pop(session_id)
                self._decode_waiting.pop(session_id, None)
            self.emit("session_closed" if kind == "close" else "state_evicted", session_id=session_id)
        else:
            raise ValueError(f"Unknown phase command {kind}")
        return True

    @torch.inference_mode()
    def _activate(self) -> None:
        while self._waiting and len(self._metadata) < self.load.slots // 3:
            command = self._waiting.popleft()
            metadata = command["metadata"]
            sid = metadata["session_id"]
            fidelity = ABotWorldFidelity.from_profile_name(metadata["fidelity"])
            if command["phase"] == "latent":
                payload = command["payload"]
                if sid not in self._sessions:
                    self._sessions[sid] = self.runtime.create_conditioned_session(
                        sid,
                        payload["first_frame"],
                        payload["prompt"],
                        seed=metadata["seed"],
                        initialize_decode=not self.pool_scheduling,
                    )
                self.runtime.ensure_local_capacity(sid, fidelity)
                self.runtime.submit_refresh(
                    self._sessions[sid],
                    payload["latents"],
                    metadata["actions"],
                    fidelity=fidelity,
                    chunk_start=metadata["chunk_start"],
                )
                fence = torch.cuda.Event()
                fence.record()
                self._pending_events.append((None, fence, {"page_slot": command["page_slot"]}))
            else:
                self.runtime.ensure_local_capacity(sid, fidelity)
                self.runtime.submit(self._sessions[sid], metadata["actions"], fidelity=fidelity)
                metadata["chunk_start"] = self.runtime._requests[sid].current_start
            self._metadata[sid] = metadata

    def _poll_events(self) -> None:
        while self._pending_events and self._pending_events[0][1].query():
            start, end, record = self._pending_events.popleft()
            if start is None:
                self._pages.release(record["page_slot"])
                if "state_ready" in record:
                    # Release the parent's transient-memory reservation only
                    # after the payload's asynchronous reads have completed.
                    self.emit("state_ready", **record["state_ready"])
                continue
            seconds = start.elapsed_time(end) / 1000
            if len(record["windows"]) == 1:
                self.load.observe(record["phase"], record["windows"][0], record["rows"], seconds)
            self.emit("phase_batch", **record, seconds=seconds)

    @torch.inference_mode()
    def run(self) -> None:
        torch.cuda.set_device(self.device)
        threading.Thread(target=self._read, name="phase-command-reader", daemon=True).start()
        try:
            running = True
            while running:
                # A bounded drain prevents sustained ingress from starving DiT.
                for _ in range(64):
                    try:
                        command = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    running = self._command(command)
                    if not running:
                        break
                self._poll_events()
                self._activate()
                ready_steps = self.runtime.ready_latent_steps()
                # A single outstanding physical invocation is an execution
                # credit. Keep processing IPC while it runs, then reconsider
                # arrivals before enqueueing the next step on this stream.
                compute_pending = any(start is not None for start, _, _ in self._pending_events)
                selected = (
                    select_phase_batch(self.runtime, self._metadata, self.load)
                    if running and not compute_pending
                    else ()
                )
                if not selected:
                    if running:
                        try:
                            running = self._command(self.commands.get(timeout=0.002))
                        except queue.Empty:
                            pass
                    continue
                tasks = [self.runtime._requests[sid] for sid in selected]
                phase = "refresh" if tasks[0].refreshing else "denoise"
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                outputs = self.runtime.step_latent_tasks([key for key in ready_steps if key.session_id in selected])
                if phase == "denoise":
                    for sid in selected:
                        if not self.runtime._requests[sid].refreshing:
                            continue
                        metadata = self._metadata.pop(sid)
                        metadata["remaining_slack_seconds"] = metadata["deadline"] - time.monotonic()
                        session = self._sessions[sid]
                        payload = {"latents": self.runtime.export_denoised(sid)}
                        if self.pool_scheduling:
                            if metadata["decode_rank"] == self.rank:
                                self._queue_decode(metadata, payload["latents"])
                            else:
                                self._export(payload, metadata, "decode")
                        if session.next_latent_frame == 0:
                            payload.update(first_frame=session.first_frame_latent, prompt=session.prompt_emb)
                        self._export(payload, metadata, "latent")
                        self.runtime.park_session(sid)
                else:
                    for output in outputs:
                        metadata = self._metadata.pop(output.session_id)
                        metadata["refresh_state_bytes"] = self.runtime._paged_states[output.session_id].resident_bytes()
                        self._export({"kv": self.runtime.export_clean_delta(output.session_id)}, metadata, "kv")
                        self.runtime.park_session(output.session_id)
                        if not self.pool_scheduling:
                            output.decode_future.add_done_callback(lambda future, m=metadata: self._decoded(m, future))
                end.record()
                self._pending_events.append(
                    (
                        start,
                        end,
                        {
                            "phase": phase,
                            "rows": self.runtime.last_batch_size,
                            "sessions": list(selected),
                            "windows": sorted({task.window[0] for task in tasks}),
                            "step_indices": [key.step_index for key in self.runtime.last_local_tasks],
                            "ready_latent_rows": len(ready_steps),
                            "ready_dependency_groups": len({key.session_id for key in ready_steps}),
                            "includes": "DiT, sampler, cache binding, payload packing and canonical parking",
                            "allocated_bytes": (
                                torch.cuda.memory_allocated(self.device) if self.device.type == "cuda" else 0
                            ),
                            "kv_pages": self.runtime.kv_pages.snapshot(),
                        },
                    )
                )
            self._transfers.shutdown(wait=True)
            self.decode.close()
            torch.cuda.synchronize(self.device)
            self._poll_events()
            self.runtime.close(materialize=False)
            self.emit("stopped", page_pool=self._pages.snapshot())
        except BaseException:
            self.emit(
                "error",
                message=traceback.format_exc(),
                kv_pages=self.runtime.kv_pages.snapshot(),
                resident_sessions=len(self._sessions),
                active_jobs=len(self._metadata),
                outstanding_exports=len(self._exports),
                queued_commands=self.commands.qsize(),
            )
            raise
        finally:
            self._stopping.set()
