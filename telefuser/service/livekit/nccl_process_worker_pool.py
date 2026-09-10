"""Process-isolated ABot model workers with NCCL state migration.

The parent retains LiveKit transport ownership. Child processes retain model
state, so a committed migration changes only the model route, not the room.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import socket
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, TypeVar

import torch
import torch.distributed as dist

from telefuser.service.core.stream_pipeline_service import STREAM_MODE_BIDIRECTIONAL
from telefuser.service.security.security_validator import SecurityLevel
from telefuser.utils.logging import logger

from .dispatch_trace import DispatchTraceWriter as _DispatchTraceWriter
from .migration_diagnostics import MigrationDiagnostics, classify_migration_error
from .nccl_transfer import (
    LayerTransferProgress,
    TensorTransferGroup,
    allocate_tensor_tree_leaves,
    build_layer_transfer_groups,
    transfer_tensor_leaves_nccl_streamed,
)
from .pipeline_adapter import LiveKitPipelineAdapter
from .process_worker_pool import (
    ProcessLiveKitWorkerPool,
    ProcessWorkerSpec,
    _close_queue,
    _current_cuda_device_for_trace,
    _install_process_dispatch_trace_callback,
)
from .session_registry import SessionRecord
from .token_service import LiveKitTokenService
from .turboserve import TurboServeOwnership, TurboServeOwnershipTable
from .worker import LiveKitWorker

# The parent retains at most one decoded payload waiting for the LiveKit
# transport. The transport acknowledges a payload as soon as it dequeues it,
# which permits exactly one next payload to be prefetched while the current
# one is paced onto WebRTC. Consequently, at most two fully-materialized
# payloads per session live outside ABot's own bounded/latest output queue:
# one being published and one in this parent queue.
_MODEL_OUTPUT_PARENT_QUEUE_SIZE = 1
_VIDEO_OUTPUT_TYPES = frozenset({"preview", "chunk"})
_TERMINAL_OUTPUT_TYPES = frozenset({"error", "done"})
# Process-group creation happens only after every worker has loaded the model,
# so it needs a dedicated budget rather than the normal 15-second IPC timeout.
# Keep the parent request longer than the child process-group timeout so a
# child can return a useful failure instead of being torn down mid-initialization.
_NCCL_INIT_GROUP_TIMEOUT_SECONDS = 180.0
_NCCL_INIT_PARENT_TIMEOUT_SECONDS = 210.0
_MIGRATION_COMMAND_TIMEOUT_SECONDS = 300.0
_MODEL_SESSION_READY_TIMEOUT_SECONDS = 120.0

_MigrationResult = TypeVar("_MigrationResult")


def _validate_nccl_source_leaves(leaves: dict[tuple[Any, ...], torch.Tensor]) -> None:
    """Reject a mixed/CPU source tree before a peer is asked to receive it."""

    non_cuda_paths = [path for path, tensor in leaves.items() if tensor.device.type != "cuda"]
    if not non_cuda_paths:
        return
    preview = ", ".join(repr(path) for path in non_cuda_paths[:4])
    suffix = "..." if len(non_cuda_paths) > 4 else ""
    raise RuntimeError(
        "NCCL migration source tensors must reside on CUDA; "
        f"found CPU leaves at {preview}{suffix}"
    )


@dataclass(frozen=True)
class _ModelOutput:
    """One child-model payload plus the worker that owns its output credit."""

    worker_id: str
    payload: dict[str, Any]


async def _pump_model_outputs(
    adapter: Any,
    service: Any,
    *,
    worker_id: str,
    session_id: str,
    credits: asyncio.BoundedSemaphore,
    events: Any,
) -> None:
    """Pull only after a parent credit, so IPC cannot outrun WebRTC playback."""
    chunks = adapter.pull_chunks(session_id)
    iterator = chunks.__aiter__()
    credit_held = False
    try:
        while True:
            await credits.acquire()
            credit_held = True
            try:
                payload = await iterator.__anext__()
            except StopAsyncIteration:
                credits.release()
                credit_held = False
                # An inactive model session naturally ends its generator. The
                # parent transport must see this EOF and release its route;
                # otherwise it waits forever in ``pull_model_chunks``.
                events.put(
                    {
                        "type": "model_output_eos",
                        "worker_id": worker_id,
                        "session_id": session_id,
                    }
                )
                return
            except asyncio.CancelledError:
                credits.release()
                credit_held = False
                raise
            except Exception:
                credits.release()
                credit_held = False
                raise
            # No await separates dequeue from IPC submission. If migration
            # pauses this task, a payload is either sent exactly once or has
            # not been removed from the ABot generator.
            events.put(
                {
                    "type": "model_output",
                    "worker_id": worker_id,
                    "session_id": session_id,
                    "payload": payload,
                    "runtime_metrics": adapter.runtime_metrics() or {},
                    "session_runtime_metrics": service.runtime_metrics(session_id),
                }
            )
            # The credit now belongs to the parent queue and transport path.
            credit_held = False
    finally:
        if credit_held:
            with contextlib.suppress(ValueError):
                credits.release()
        aclose = getattr(chunks, "aclose", None)
        if callable(aclose):
            with contextlib.suppress(Exception):
                await aclose()


class _ProcessPipelineAdapter:
    stream_mode = STREAM_MODE_BIDIRECTIONAL
    # The concrete ABot pipeline lives in the child model process. Its native
    # batched path supports per-session cursors under Relative-RoPE, so expose
    # that immutable capability to the parent-side scheduler explicitly.
    uses_relative_rope = True

    def __init__(self, pool: "NCCLProcessLiveKitWorkerPool", initial_worker_id: str) -> None:
        self._pool = pool
        self._initial_worker_id = initial_worker_id

    def create_session(self, config: dict) -> str:
        session_id = str(config["session_id"])
        self._pool.create_model_session(self._initial_worker_id, session_id, config)
        return session_id

    async def wait_session_ready(self, session_id: str) -> None:
        """Wait until the child has created the model-owned session."""
        await self._pool.wait_model_session_ready(
            session_id,
            timeout=_MODEL_SESSION_READY_TIMEOUT_SECONDS,
        )

    def push_chunk(self, session_id: str, chunk: dict) -> None:
        self._pool.push_model_chunk(session_id, chunk)

    def push_batch(self, items: list[tuple[str, dict]]) -> None:
        self._pool.push_model_batch(items)

    async def pull_chunks(self, session_id: str):
        async for chunk in self._pool.pull_model_chunks(session_id):
            yield chunk

    def enable_publisher_frame_tracking(self, session_id: str) -> bool:
        return self._pool.enable_publisher_frame_tracking(session_id)

    def report_publisher_frame_progress(
        self, session_id: str, *, event: str, frames_delta: int, sequence: int, observed_monotonic_seconds: float
    ) -> bool:
        del sequence
        return self._pool.report_publisher_frame_progress(
            session_id,
            event=event,
            frames_delta=frames_delta,
            observed_monotonic_seconds=observed_monotonic_seconds,
        )

    def close_session(self, session_id: str) -> None:
        self._pool.close_model_session(session_id)


class _ParentTransportSink:
    def __init__(self, pool: "NCCLProcessLiveKitWorkerPool") -> None:
        self.pool = pool

    def on_worker_status(self, worker_id: str, status: str) -> None:
        del worker_id, status

    def on_worker_capacity(self, worker_id: str, capacity: int, profile: dict[str, object] | None = None) -> None:
        self.pool._event_sink.on_worker_capacity(worker_id, capacity, profile)

    def on_session_status(self, session_id: str, status: str, error: str | None = None) -> None:
        self.pool._event_sink.on_session_status(session_id, status, error)

    def on_pipeline_session(self, session_id: str, pipeline_session_id: str) -> None:
        self.pool._event_sink.on_pipeline_session(session_id, pipeline_session_id)

    def on_session_finished(self, worker_id: str, session_id: str, error: str | None = None) -> None:
        self.pool._transport_finished(session_id)
        self.pool._event_sink.on_session_finished(worker_id, session_id, error)

    def on_control_received(self, worker_id: str, session_id: str) -> None:
        callback = getattr(self.pool._event_sink, "on_control_received", None)
        if callable(callback):
            callback(worker_id, session_id)

    def on_control_message(self, worker_id: str, session_id: str, chunk: dict) -> bool:
        callback = getattr(self.pool._event_sink, "on_control_message", None)
        return bool(callback(worker_id, session_id, chunk)) if callable(callback) else False

    def on_chunk_published(
        self, worker_id: str, session_id: str, frames: int, first_frame_at: float | None = None
    ) -> None:
        callback = getattr(self.pool._event_sink, "on_chunk_published", None)
        if callable(callback):
            callback(worker_id, session_id, frames, first_frame_at)


class NCCLProcessLiveKitWorkerPool(ProcessLiveKitWorkerPool):
    """TurboServe-compatible parent transport / GPU model-process pool."""

    progressive_migration_supported = True

    def __init__(self, specs: list[ProcessWorkerSpec], **kwargs: Any) -> None:
        super().__init__(specs, **kwargs)
        self._worker_target = _nccl_model_worker_main
        self._ownership = TurboServeOwnershipTable()
        self._model_outputs: dict[str, asyncio.Queue[_ModelOutput | None]] = {}
        self._model_output_inflight: set[str] = set()
        self._model_output_drained: dict[str, asyncio.Event] = {}
        self._model_output_inflight_owner: dict[str, str] = {}
        self._publisher_progress_sequences: dict[str, int] = {}
        self._publisher_frame_tracking: dict[str, bool] = {}
        self._model_output_dropped: dict[str, int] = {}
        self._transport_workers: dict[str, LiveKitWorker] = {}
        self._transport_tasks: dict[str, asyncio.Task[None]] = {}
        self._migrating_controls: dict[str, list[dict]] = {}
        self._provisional_migration_controls: dict[str, list[dict]] = {}
        self._provisional_model_events: dict[str, list[dict[str, Any]]] = {}
        self._migration_ready_waiters: dict[str, asyncio.Future[None]] = {}
        self._model_session_ready: dict[str, concurrent.futures.Future[str]] = {}
        # A LiveKit room can depart while its model state is being transferred.
        # Defer the route teardown until the migration transaction has either
        # committed the target or restored the source; closing the source in
        # the middle of a P2P copy can invalidate NCCL buffers on both peers.
        self._active_migrations: set[str] = set()
        self._deferred_model_closes: set[str] = set()
        # Worker snapshots include scalar timings/counters plus the bounded
        # scheduler mode string (``batched`` or ``round_robin``).
        self._worker_runtime_metrics: dict[str, dict[str, float | int | str]] = {}
        self._session_runtime_metrics: dict[str, dict[str, float | int | str]] = {}
        self._migration_total_ms: list[float] = []
        self._migration_first_layer_ms: list[float] = []
        self._migration_transfer_complete_ms: list[float] = []
        self._migration_drain_ms: list[float] = []
        self._migration_background_cleanup_ms: list[float] = []
        self._migration_deferred_publisher_frames = 0
        self._migration_cleanup_failures = 0
        self._migration_diagnostics = MigrationDiagnostics()
        self._nccl_ranks: dict[str, int] = {}
        self._nccl_warmup_ms = 0.0
        self._migration_lock = asyncio.Lock()
        self._initializing_workers = False
        # ``ProcessLiveKitWorkerPool`` owns the parent JSONL writer for both
        # isolated-worker modes. Recreating it here would reject the path it
        # has just created, preventing a traced process-NCCL run from starting.

    def _handle_unexpected_exit(
        self,
        worker_id: str,
        exitcode: int | None,
        *,
        expected: bool = False,
    ) -> None:
        """Attach migration context before the base pool tears down a dead worker."""

        # The suite terminates the server process group with SIGTERM after it
        # has persisted metrics.  That signal reaches children at the same
        # time as the parent's cleanup coroutine, so the monitor can observe
        # ``-15`` before ``_stopping_workers`` is populated.  Treat it as a
        # normal teardown and keep it out of migration-failure diagnostics.
        expected = expected or exitcode == -15 or (exitcode == 0 and self._closing)
        diagnostics = getattr(self, "_migration_diagnostics", None)
        if expected:
            logger.info(f"NCCL worker exited during normal shutdown: worker={worker_id} exit_code={exitcode}")
        else:
            if isinstance(diagnostics, MigrationDiagnostics):
                diagnostics.record_worker_exit(worker_id, exitcode)
            logger.error(
                f"NCCL worker exited unexpectedly: worker={worker_id} exit_code={exitcode} "
                "active_migrations="
                f"{diagnostics.snapshot().get('active', 0) if isinstance(diagnostics, MigrationDiagnostics) else 0}"
            )
        error = None if expected else RuntimeError(f"Worker process exited unexpectedly with code {exitcode}")
        readiness_error = error or RuntimeError(f"Worker process shut down before model readiness (code {exitcode})")
        for session_id, ready in tuple(self._model_session_ready.items()):
            if self._session_workers.get(session_id) == worker_id and not ready.done():
                ready.set_exception(readiness_error)
        super()._handle_unexpected_exit(worker_id, exitcode, expected=expected)

    async def start(self, *, skip_validation: bool = False) -> None:
        # ``ProcessLiveKitWorkerPool.start`` calls this class's ``scale_to``
        # once per replica. Defer communicator construction until all initial
        # workers have completed their sequential checkpoint load.
        self._initializing_workers = True
        try:
            await super().start(skip_validation=skip_validation)
        finally:
            self._initializing_workers = False
        if len(self._active_workers) > 1 and not self._nccl_ranks:
            await self._init_nccl()

    async def scale_to(self, target_workers: int) -> int:
        """Rebuild the static NCCL communicator around a new replica set."""
        async with self._migration_lock:
            if len(self._active_workers) == target_workers:
                return target_workers
            if self._nccl_ranks:
                await asyncio.gather(
                    *(self._request(worker_id, "nccl_destroy") for worker_id in self._nccl_ranks),
                    return_exceptions=True,
                )
                self._nccl_ranks.clear()
            actual = await super().scale_to(target_workers)
            if actual > 1 and not self._initializing_workers:
                await self._init_nccl()
            return actual

    def start_session(self, record: SessionRecord) -> None:
        if record.worker_id is None or record.worker_id not in self._active_workers:
            raise RuntimeError("Model worker is not active")
        runner = LiveKitWorker(
            worker_id=record.worker_id,
            config=self._config,
            pipeline_file=self._pipeline_file,
            token_service=LiveKitTokenService(
                api_key=self._config.livekit_api_key,
                api_secret=self._config.livekit_api_secret,
                token_ttl=self._config.token_ttl,
            ),
            event_sink=_ParentTransportSink(self),
            pipeline_adapter=_ProcessPipelineAdapter(self, record.worker_id),
        )
        task = asyncio.create_task(runner.run_session(record), name=f"livekit-transport-{record.session_id}")
        self._transport_workers[record.session_id] = runner
        self._transport_tasks[record.session_id] = task
        task.add_done_callback(lambda done, sid=record.session_id: self._transport_task_done(sid, done))

    def dispatch_batch(self, lease: Any, payloads: list[tuple[str, dict]]) -> None:
        """Send one policy-selected batch through the parent transport routes."""
        expected_worker_id = getattr(getattr(lease, "candidate", None), "gpu_id", None)
        job_ids = [str(job.job_id) for job in getattr(lease, "jobs", ())]
        session_ids = [str(session_id) for session_id, _ in payloads]
        grouped: dict[str, list[tuple[str, dict]]] = {}
        for session_id, chunk in payloads:
            runner = self._transport_workers.get(session_id)
            if runner is None or runner.pipeline_session_id is None:
                raise RuntimeError(f"Session {session_id!r} has no active model transport")
            worker_id = self._pipeline_routes.get(runner.pipeline_session_id)
            if worker_id is None:
                raise RuntimeError(f"Session {session_id!r} has no model route")
            if expected_worker_id is not None and worker_id != expected_worker_id:
                raise RuntimeError(
                    f"Motivation owner mismatch for {session_id!r}: "
                    f"candidate={expected_worker_id!r} actual={worker_id!r}"
                )
            grouped.setdefault(worker_id, []).append((runner.pipeline_session_id, dict(chunk)))
        for worker_id, items in grouped.items():
            self._send(
                worker_id,
                {
                    "type": "model_push_batch",
                    "items": items,
                    "motivation_job_ids": job_ids,
                    "motivation_session_ids": session_ids,
                },
            )

    def dispatch_owner(self, session_id: str) -> str | None:
        """Resolve a LiveKit transport session to its physical model route."""
        runner = self._transport_workers.get(session_id)
        if runner is None or runner.pipeline_session_id is None:
            return None
        return self._pipeline_routes.get(runner.pipeline_session_id)

    async def stop_session(self, session_id: str) -> None:
        runner = self._transport_workers.get(session_id)
        task = self._transport_tasks.get(session_id)
        if runner is not None:
            await runner.stop_session(session_id)
        if task is not None:
            with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(task), timeout=15.0)

    def create_model_session(self, worker_id: str, session_id: str, config: dict) -> None:
        if session_id in self._model_session_ready:
            raise RuntimeError(f"Model session {session_id!r} is already being created")
        output: asyncio.Queue[_ModelOutput | None] = asyncio.Queue(maxsize=_MODEL_OUTPUT_PARENT_QUEUE_SIZE)
        drained = asyncio.Event()
        drained.set()
        self._model_outputs[session_id] = output
        self._model_output_drained[session_id] = drained
        self._model_output_dropped[session_id] = 0
        self._pipeline_routes[session_id] = worker_id
        self._publisher_progress_sequences[session_id] = -1
        self._publisher_frame_tracking[session_id] = True
        self._session_workers[session_id] = worker_id
        self._ownership.register(session_id, worker_id)
        # A concurrent future is loop-independent because session creation is
        # submitted synchronously while readiness is awaited by the transport
        # task. This also keeps direct pool tests usable outside asyncio.run().
        ready: concurrent.futures.Future[str] = concurrent.futures.Future()
        self._model_session_ready[session_id] = ready
        try:
            self._send(
                worker_id,
                {
                    "type": "model_create",
                    "session_id": session_id,
                    "config": dict(config),
                    "model_output_credit_window": _MODEL_OUTPUT_PARENT_QUEUE_SIZE,
                },
            )
        except Exception:
            self._model_session_ready.pop(session_id, None)
            self._model_outputs.pop(session_id, None)
            self._model_output_drained.pop(session_id, None)
            self._model_output_dropped.pop(session_id, None)
            self._pipeline_routes.pop(session_id, None)
            self._session_workers.pop(session_id, None)
            self._ownership.release(session_id)
            ready.cancel()
            raise

    async def wait_model_session_ready(self, session_id: str, *, timeout: float) -> None:
        """Wait for a child model-create ACK, failing closed on timeout."""
        ready = self._model_session_ready.get(session_id)
        if ready is None:
            raise RuntimeError(f"Model session {session_id!r} has no pending readiness ACK")
        try:
            await asyncio.wait_for(asyncio.wrap_future(ready), timeout=timeout)
        finally:
            if ready.done():
                self._model_session_ready.pop(session_id, None)

    def push_model_chunk(self, session_id: str, chunk: dict) -> None:
        if session_id in self._provisional_migration_controls:
            self._provisional_migration_controls[session_id].append(dict(chunk))
            worker_id = self._pipeline_routes.get(session_id)
            if worker_id is not None:
                self._send(
                    worker_id,
                    {"type": "model_push", "session_id": session_id, "chunk": dict(chunk)},
                )
            return
        if session_id in self._migrating_controls:
            self._migrating_controls[session_id].append(dict(chunk))
            return
        worker_id = self._pipeline_routes.get(session_id)
        if worker_id is None:
            # A controller can publish a final stop/reset after the transport
            # has already torn down the model route. The session is gone, so
            # there is no child command to deliver; late control must not turn
            # normal teardown into an uncaught KeyError in LiveKit's callback.
            return
        self._send(
            worker_id,
            {"type": "model_push", "session_id": session_id, "chunk": dict(chunk)},
        )

    def push_model_batch(self, items: list[tuple[str, dict]]) -> None:
        """Forward a batch to one child command so its service sees one update turn."""
        grouped: dict[str, list[tuple[str, dict]]] = {}
        for session_id, chunk in items:
            if session_id in self._provisional_migration_controls:
                self._provisional_migration_controls[session_id].append(dict(chunk))
                worker_id = self._pipeline_routes.get(session_id)
                if worker_id is not None:
                    grouped.setdefault(worker_id, []).append((session_id, dict(chunk)))
                continue
            if session_id in self._migrating_controls:
                self._migrating_controls[session_id].append(dict(chunk))
                continue
            worker_id = self._pipeline_routes.get(session_id)
            if worker_id is None:
                # The session may have completed while a batch was being
                # assembled. Drop that stale member and let remaining current
                # routes proceed; the owning policy lease is invalidated by
                # the session-finished callback.
                continue
            grouped.setdefault(worker_id, []).append((session_id, dict(chunk)))
        for worker_id, grouped_items in grouped.items():
            self._send(worker_id, {"type": "model_push_batch", "items": grouped_items})

    def enable_publisher_frame_tracking(self, session_id: str) -> bool:
        return bool(self._publisher_frame_tracking.get(session_id, False))

    def report_publisher_frame_progress(
        self, session_id: str, *, event: str, frames_delta: int, observed_monotonic_seconds: float
    ) -> bool:
        worker_id = self._model_output_inflight_owner.get(session_id) or self._pipeline_routes.get(session_id)
        return self._send_publisher_frame_progress(
            session_id,
            worker_id=worker_id,
            event=event,
            frames_delta=frames_delta,
            observed_monotonic_seconds=observed_monotonic_seconds,
        )

    def _send_publisher_frame_progress(
        self,
        session_id: str,
        *,
        worker_id: str | None,
        event: str,
        frames_delta: int,
        observed_monotonic_seconds: float,
    ) -> bool:
        if (
            worker_id is None
            or worker_id not in self._active_workers
            or not self._publisher_frame_tracking.get(session_id, False)
        ):
            return False
        sequence = int(self._publisher_progress_sequences.get(session_id, -1)) + 1
        self._publisher_progress_sequences[session_id] = sequence
        try:
            self._send(
                worker_id,
                {
                    "type": "model_publisher_frame_progress",
                    "session_id": session_id,
                    "event": event,
                    "frames_delta": int(frames_delta),
                    "sequence": sequence,
                    "observed_monotonic_seconds": float(observed_monotonic_seconds),
                },
            )
        except Exception:
            return False
        return True

    def close_model_session(self, session_id: str) -> None:
        if session_id in self._active_migrations:
            self._deferred_model_closes.add(session_id)
            return
        self._close_model_session_now(session_id)

    def _close_model_session_now(self, session_id: str) -> None:
        """Release a model route after any in-flight migration has settled."""
        ready = self._model_session_ready.pop(session_id, None)
        if ready is not None and not ready.done():
            ready.set_exception(RuntimeError(f"Model session {session_id!r} was closed before readiness"))
        worker_id = self._pipeline_routes.pop(session_id, None)
        self._session_workers.pop(session_id, None)
        self._ownership.release(session_id)
        self._migrating_controls.pop(session_id, None)
        self._provisional_migration_controls.pop(session_id, None)
        self._provisional_model_events.pop(session_id, None)
        self._session_runtime_metrics.pop(session_id, None)
        self._publisher_progress_sequences.pop(session_id, None)
        self._publisher_frame_tracking.pop(session_id, None)
        if worker_id in self._active_workers:
            self._send(worker_id, {"type": "model_close", "session_id": session_id})
        self._close_model_output(session_id)

    async def pull_model_chunks(self, session_id: str):
        output = self._model_outputs.get(session_id)
        if output is None:
            return
        while True:
            item = await output.get()
            if item is None:
                return
            # This is deliberately before ``yield``: it allows one queued
            # prefetch while the transport paces the just-dequeued payload.
            self._model_output_inflight.add(session_id)
            self._model_output_inflight_owner[session_id] = item.worker_id
            self._update_model_output_drained(session_id)
            self._ack_model_output(session_id, item)
            try:
                yield item.payload
            finally:
                self._model_output_inflight.discard(session_id)
                self._model_output_inflight_owner.pop(session_id, None)
                self._update_model_output_drained(session_id)

    def _close_model_output(self, session_id: str) -> None:
        output = self._model_outputs.pop(session_id, None)
        self._model_output_inflight.discard(session_id)
        self._model_output_inflight_owner.pop(session_id, None)
        drained = self._model_output_drained.pop(session_id, None)
        self._model_output_dropped.pop(session_id, None)
        if isinstance(output, asyncio.Queue):
            while True:
                try:
                    output.get_nowait()
                except asyncio.QueueEmpty:
                    break
            with contextlib.suppress(asyncio.QueueFull):
                output.put_nowait(None)
        elif output is not None:
            # Compatibility for simple queue doubles used by isolated tests.
            output.put_nowait(None)
        if drained is not None:
            drained.set()

    def _enqueue_model_output(self, session_id: str, item: _ModelOutput) -> None:
        output = self._model_outputs.get(session_id)
        if output is None:
            return
        if not isinstance(output, asyncio.Queue):
            # Keep legacy light-weight test doubles usable while production
            # always takes the bounded branch below.
            output.put_nowait(item.payload)
            return
        try:
            output.put_nowait(item)
        except asyncio.QueueFull:
            queued = output.get_nowait()
            if queued is None:
                output.put_nowait(None)
                self._record_dropped_model_output(session_id, item)
            elif self._should_replace_queued_output(queued, item):
                output.put_nowait(item)
                self._record_dropped_model_output(session_id, queued)
            else:
                output.put_nowait(queued)
                self._record_dropped_model_output(session_id, item)
        self._update_model_output_drained(session_id)

    @staticmethod
    def _should_replace_queued_output(queued: _ModelOutput, incoming: _ModelOutput) -> bool:
        queued_type = str(queued.payload.get("type", ""))
        incoming_type = str(incoming.payload.get("type", ""))
        if queued_type in _TERMINAL_OUTPUT_TYPES:
            return False
        if incoming_type in _TERMINAL_OUTPUT_TYPES:
            return True
        # Preserve an initial preview until it reaches the transport. Later
        # generated chunks are latest-wins, matching ABot's own queue.
        if queued_type == "preview":
            return False
        if incoming_type == "preview":
            return queued_type in _VIDEO_OUTPUT_TYPES
        return queued_type == "chunk" and incoming_type == "chunk"

    def _record_dropped_model_output(self, session_id: str, item: _ModelOutput, *, acknowledge: bool = True) -> None:
        dropped = getattr(self, "_model_output_dropped", None)
        if isinstance(dropped, dict):
            dropped[session_id] = int(dropped.get(session_id, 0)) + 1
        payload = item.payload
        frames = payload.get("frames")
        if payload.get("type") == "chunk" and isinstance(frames, list) and frames:
            self._send_publisher_frame_progress(
                session_id,
                worker_id=item.worker_id,
                event="dropped",
                frames_delta=-len(frames),
                observed_monotonic_seconds=time.monotonic(),
            )
        if acknowledge:
            self._ack_model_output(session_id, item)

    def _ack_model_output(self, session_id: str, item: _ModelOutput) -> None:
        # Never wait for a child response from the parent event loop. The
        # command only releases that child session's bounded semaphore.
        try:
            if item.worker_id in self._active_workers:
                self._send(
                    item.worker_id,
                    {"type": "model_output_credit", "session_id": session_id},
                )
        except Exception:
            # The owning child can disappear during close/migration; there is
            # then no useful credit to return.
            return

    def _update_model_output_drained(self, session_id: str) -> None:
        drained = self._model_output_drained.get(session_id)
        output = self._model_outputs.get(session_id)
        if drained is None or output is None or not isinstance(output, asyncio.Queue):
            return
        if session_id not in self._model_output_inflight and output.empty():
            drained.set()
        else:
            drained.clear()

    async def _wait_for_model_output_drain(self, session_id: str, *, timeout: float) -> None:
        self._update_model_output_drained(session_id)
        drained = self._model_output_drained.get(session_id)
        if drained is not None:
            await asyncio.wait_for(drained.wait(), timeout=timeout)

    @staticmethod
    def _source_model_output_drain_complete(
        status: object,
        *,
        require_output_queue: bool | None = True,
        require_publisher: bool = True,
    ) -> bool:
        """Return whether source model state is quiescent for migration.

        Publisher-owned frames are transport data, not model state. Progressive
        SST may therefore start once the child service queue is empty; those
        already-materialized frames continue draining through the parent while
        the target receives state. Callers that are closing a route can retain
        the stricter publisher check explicitly.
        """
        if not isinstance(status, dict):
            return False
        if require_output_queue is None:
            # The child explicitly marks latest-mode output as discardable.
            # Missing metadata is treated conservatively for older workers.
            require_output_queue = not bool(status.get("output_queue_discardable", False))
        if status.get("in_flight", True):
            return False
        if require_output_queue and not status.get("output_queue_empty", False):
            return False
        if require_publisher and int(status.get("publisher_unsubmitted_frames", 1)) != 0:
            return False
        return True

    async def _drain_model_outputs_for_migration(
        self,
        session_id: str,
        *,
        source_worker_id: str,
        timeout: float,
    ) -> dict[str, Any]:
        """Quiesce model output before SST while publisher frames drain in background.

        The child service queue is the model-state boundary. Once it is empty
        and no model output is in flight, no future source computation can
        mutate the snapshot. Parent-transport/publisher frames may still be
        outstanding (bounded by the output credit window); they remain on the
        existing LiveKit route while the target imports state.
        """
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out draining model output before NCCL migration")
            status_event = await self._request(
                source_worker_id,
                "model_output_drain_status",
                session_id=session_id,
                timeout=remaining,
            )
            status = status_event.get("result")
            if self._source_model_output_drain_complete(
                status,
                require_output_queue=None,
                require_publisher=False,
            ):
                return dict(status) if isinstance(status, dict) else {}
            await asyncio.sleep(min(0.005, max(0.001, deadline - time.monotonic())))

    async def _run_migration_phase(
        self,
        transfer_id: str,
        phase: str,
        operation: Callable[[], Awaitable[_MigrationResult]],
    ) -> _MigrationResult:
        """Run one migration phase while retaining bounded timing telemetry."""

        diagnostics = getattr(self, "_migration_diagnostics", None)
        started = time.monotonic()
        if isinstance(diagnostics, MigrationDiagnostics):
            diagnostics.phase_started(transfer_id, phase)
        try:
            value = await operation()
        except BaseException as exc:
            if isinstance(diagnostics, MigrationDiagnostics):
                diagnostics.phase_finished(
                    transfer_id,
                    phase,
                    success=False,
                    duration_seconds=time.monotonic() - started,
                    error=exc,
                )
            raise
        else:
            if isinstance(diagnostics, MigrationDiagnostics):
                diagnostics.phase_finished(
                    transfer_id,
                    phase,
                    success=True,
                    duration_seconds=time.monotonic() - started,
                )
            return value

    async def migrate_session(
        self,
        pipeline_session_id: str,
        target_worker_id: str,
        *,
        on_compute_ready: Callable[[], None] | None = None,
    ) -> TurboServeOwnership:
        async with self._migration_lock:
            source_worker_id = self._pipeline_routes[pipeline_session_id]
            if source_worker_id == target_worker_id:
                return self._ownership.owner(pipeline_session_id)
            diagnostics = getattr(self, "_migration_diagnostics", None)
            if source_worker_id not in self._nccl_ranks or target_worker_id not in self._nccl_ranks:
                error = RuntimeError("NCCL migration requires initialized source and target workers")
                if isinstance(diagnostics, MigrationDiagnostics):
                    diagnostics.reject(
                        source_worker_id=source_worker_id,
                        target_worker_id=target_worker_id,
                        reason=error,
                    )
                raise error
            try:
                token = self._ownership.prepare_migration(pipeline_session_id, source_worker_id, target_worker_id)
            except Exception as exc:
                if isinstance(diagnostics, MigrationDiagnostics):
                    diagnostics.reject(
                        source_worker_id=source_worker_id,
                        target_worker_id=target_worker_id,
                        reason=exc,
                    )
                raise
            if isinstance(diagnostics, MigrationDiagnostics):
                diagnostics.begin(
                    token.token_id,
                    source_worker_id=source_worker_id,
                    target_worker_id=target_worker_id,
                )
            logger.info(
                f"NCCL migration started: transfer={token.token_id} "
                f"source={source_worker_id} target={target_worker_id}"
            )
            self._active_migrations.add(pipeline_session_id)
            self._migrating_controls[pipeline_session_id] = []
            started = time.monotonic()
            source_output_paused = False
            completed = False

            async def rollback_migration() -> None:
                """Best-effort cleanup for cancellation or transport failure."""

                with contextlib.suppress(BaseException):
                    await self._request(
                        target_worker_id,
                        "nccl_discard",
                        transfer_id=token.token_id,
                        session_id=pipeline_session_id,
                    )
                with contextlib.suppress(BaseException):
                    await self._request(
                        source_worker_id,
                        "nccl_abort_source",
                        session_id=pipeline_session_id,
                        transfer_id=token.token_id,
                    )
                if source_output_paused:
                    with contextlib.suppress(BaseException):
                        await self._request(
                            source_worker_id,
                            "model_output_resume",
                            session_id=pipeline_session_id,
                        )
                with contextlib.suppress(BaseException):
                    self._ownership.abort_migration(token)
                self._pipeline_routes[pipeline_session_id] = source_worker_id
                self._session_workers[pipeline_session_id] = source_worker_id
                pending_controls = self._migrating_controls.pop(pipeline_session_id, [])
                pending_controls.extend(
                    self._provisional_migration_controls.pop(pipeline_session_id, [])
                )
                staged_events = self._provisional_model_events.pop(pipeline_session_id, [])
                for event in staged_events:
                    if event.get("type") == "model_output":
                        self._record_dropped_model_output(
                            pipeline_session_id,
                            _ModelOutput(
                                worker_id=str(event.get("worker_id", target_worker_id)),
                                payload=dict(event.get("payload", {})),
                            ),
                            acknowledge=False,
                        )
                for chunk in pending_controls:
                    with contextlib.suppress(BaseException):
                        self._send(
                            source_worker_id,
                            {"type": "model_push", "session_id": pipeline_session_id, "chunk": chunk},
                        )

            try:
                # ABot marks only this session as migrating and waits for its
                # own boundary. Other sessions on both workers keep running
                # while the state transfer is prepared.
                drain_started = time.monotonic()
                drain_status = await self._run_migration_phase(
                    token.token_id,
                    "drain",
                    lambda: self._drain_model_outputs_for_migration(
                        pipeline_session_id,
                        source_worker_id=source_worker_id,
                        timeout=_MIGRATION_COMMAND_TIMEOUT_SECONDS,
                    ),
                )
                self._migration_drain_ms.append((time.monotonic() - drain_started) * 1000.0)
                if isinstance(drain_status, dict):
                    self._migration_deferred_publisher_frames += max(
                        0, int(drain_status.get("publisher_unsubmitted_frames", 0))
                    )

                async def pause_and_verify() -> None:
                    nonlocal source_output_paused
                    await self._request(
                        source_worker_id,
                        "model_output_pause",
                        session_id=pipeline_session_id,
                        timeout=_MIGRATION_COMMAND_TIMEOUT_SECONDS,
                    )
                    # The pause command may have taken effect even when the
                    # following status barrier fails. Mark it immediately so
                    # the failure path always attempts to resume the source.
                    source_output_paused = True
                    paused_status_event = await self._request(
                        source_worker_id,
                        "model_output_drain_status",
                        session_id=pipeline_session_id,
                        timeout=_MIGRATION_COMMAND_TIMEOUT_SECONDS,
                    )
                    if not self._source_model_output_drain_complete(
                        paused_status_event.get("result"),
                        require_output_queue=None,
                        require_publisher=False,
                    ):
                        raise RuntimeError("Source output changed while preparing NCCL migration")

                await self._run_migration_phase(token.token_id, "pause", pause_and_verify)

                exported = await self._run_migration_phase(
                    token.token_id,
                    "export",
                    lambda: self._request(
                        source_worker_id,
                        "nccl_export",
                        session_id=pipeline_session_id,
                        transfer_id=token.token_id,
                        timeout=_MIGRATION_COMMAND_TIMEOUT_SECONDS,
                    ),
                )
                metadata = dict(exported["result"])
                state_bytes = metadata.get("state_bytes", 0)
                if isinstance(diagnostics, MigrationDiagnostics):
                    diagnostics.set_state_bytes(token.token_id, state_bytes)

                await self._run_migration_phase(
                    token.token_id,
                    "prepare_recv",
                    lambda: self._request(
                        target_worker_id,
                        "nccl_prepare_recv",
                        transfer_id=token.token_id,
                        metadata=metadata,
                        source_rank=self._nccl_ranks[source_worker_id],
                        owner_worker_id=target_worker_id,
                        ownership_epoch=token.source_epoch + 1,
                        model_output_credit_window=_MODEL_OUTPUT_PARENT_QUEUE_SIZE,
                        timeout=_MIGRATION_COMMAND_TIMEOUT_SECONDS,
                    ),
                )
                ready_waiter = asyncio.get_running_loop().create_future()
                self._migration_ready_waiters[token.token_id] = ready_waiter

                async def transfer_both() -> list[dict[str, Any]]:
                    return await asyncio.gather(
                        self._request(
                            source_worker_id,
                            "nccl_send",
                            transfer_id=token.token_id,
                            target_rank=self._nccl_ranks[target_worker_id],
                            timeout=_MIGRATION_COMMAND_TIMEOUT_SECONDS,
                        ),
                        self._request(
                            target_worker_id,
                            "nccl_recv",
                            transfer_id=token.token_id,
                            source_rank=self._nccl_ranks[source_worker_id],
                            timeout=_MIGRATION_COMMAND_TIMEOUT_SECONDS,
                        ),
                    )

                transfer_task = asyncio.create_task(
                    self._run_migration_phase(token.token_id, "transfer", transfer_both),
                    name=f"sst-transfer-{token.token_id}",
                )
                first_done, _ = await asyncio.wait(
                    (ready_waiter, transfer_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if transfer_task in first_done and not ready_waiter.done():
                    # Compatibility path for a transport that only reports its
                    # final result. A real process-NCCL child emits the early
                    # readiness event immediately after target import.
                    await transfer_task
                    ready_waiter.set_result(None)
                await ready_waiter

                # The target session contains receive buffers for every cache
                # layer, while the child deliberately allocates those buffers
                # before the ordered P2P copy starts.  Publishing the route at
                # first-layer readiness lets a batched invocation observe an
                # as-yet-unreceived scalar cursor (Relative-RoPE permits
                # different *valid* global cursors, so that race is otherwise
                # indistinguishable from a legal batch).  Keep the source
                # route authoritative until the complete receive has settled;
                # this is a small handoff cost compared with the migration
                # drain and makes the ownership transaction memory-safe.
                transfer_events = await transfer_task
                if isinstance(diagnostics, MigrationDiagnostics) and isinstance(transfer_events, list):
                    for event in reversed(transfer_events):
                        report = event.get("result") if isinstance(event, dict) else None
                        if isinstance(report, dict) and isinstance(report.get("groups"), list):
                            diagnostics.set_transport_report(token.token_id, report)
                            progress_report = report.get("progress")
                            if isinstance(progress_report, dict):
                                first_layer_ms = progress_report.get("first_layer_ready_ms")
                                transfer_complete_ms = progress_report.get("transfer_complete_ms")
                                if isinstance(first_layer_ms, int | float) and float(first_layer_ms) >= 0:
                                    self._migration_first_layer_ms.append(float(first_layer_ms))
                                if isinstance(transfer_complete_ms, int | float) and float(transfer_complete_ms) >= 0:
                                    self._migration_transfer_complete_ms.append(float(transfer_complete_ms))
                            break

                async def publish_compute_route() -> None:
                    self._pipeline_routes[pipeline_session_id] = target_worker_id
                    self._session_workers[pipeline_session_id] = target_worker_id
                    pending_controls = self._migrating_controls.pop(pipeline_session_id, [])
                    self._provisional_migration_controls[pipeline_session_id] = list(pending_controls)
                    self._provisional_model_events[pipeline_session_id] = []
                    for chunk in pending_controls:
                        self._send(
                            target_worker_id,
                            {"type": "model_push", "session_id": pipeline_session_id, "chunk": chunk},
                        )
                    if on_compute_ready is not None:
                        on_compute_ready()

                await self._run_migration_phase(
                    token.token_id,
                    "compute_ready",
                    publish_compute_route,
                )
                async def commit_route() -> TurboServeOwnership:
                    ownership = self._ownership.commit_migration(token)
                    self._pipeline_routes[pipeline_session_id] = target_worker_id
                    self._session_workers[pipeline_session_id] = target_worker_id
                    return ownership

                ownership = await self._run_migration_phase(token.token_id, "route_commit", commit_route)
                # Ownership is the transaction commit point. Source cleanup is
                # deliberately ordered afterwards: if that worker disappears,
                # the imported target remains authoritative instead of trying
                # to roll back to state that may already have been deleted.
                completed = True
                self._provisional_migration_controls.pop(pipeline_session_id, None)
                cleanup_started = time.monotonic()
                try:
                    await self._run_migration_phase(
                        token.token_id,
                        "commit_source",
                        lambda: self._request(
                            source_worker_id,
                            "nccl_commit_source",
                            session_id=pipeline_session_id,
                            timeout=_MIGRATION_COMMAND_TIMEOUT_SECONDS,
                        ),
                    )
                except Exception as exc:
                    self._migration_cleanup_failures += 1
                    logger.warning(
                        f"NCCL source cleanup failed after committed migration: "
                        f"session={pipeline_session_id} source={source_worker_id} error={exc}"
                    )
                finally:
                    self._migration_background_cleanup_ms.append((time.monotonic() - cleanup_started) * 1000.0)
                staged_events = self._provisional_model_events.pop(pipeline_session_id, [])
                for event in staged_events:
                    self._dispatch_event(event)
                total_ms = (time.monotonic() - started) * 1000.0
                self._migration_total_ms.append(total_ms)
                if isinstance(diagnostics, MigrationDiagnostics):
                    diagnostics.finish(token.token_id, outcome="success")
                logger.info(
                    f"NCCL migration completed: transfer={token.token_id} "
                    f"source={source_worker_id} target={target_worker_id} duration_ms={total_ms:.3f}"
                )
                return ownership
            except asyncio.CancelledError:
                if not completed:
                    await rollback_migration()
                if isinstance(diagnostics, MigrationDiagnostics):
                    diagnostics.finish(token.token_id, outcome="aborted", error="migration cancelled")
                raise
            except Exception as exc:
                if not completed:
                    await rollback_migration()
                if isinstance(diagnostics, MigrationDiagnostics):
                    diagnostics.finish(token.token_id, outcome="failure", error=exc)
                logger.warning(
                    f"NCCL migration failed: transfer={token.token_id} "
                    f"source={source_worker_id} target={target_worker_id} "
                    f"error_kind={classify_migration_error(exc)} error={exc}"
                )
                raise
            finally:
                self._migration_ready_waiters.pop(token.token_id, None)
                # A normal success path has already finished telemetry.  This
                # guard covers an unexpected cancellation/error during cleanup
                # without allowing a stale active transfer in the snapshot.
                if not completed and isinstance(diagnostics, MigrationDiagnostics):
                    diagnostics.finish(token.token_id, outcome="aborted", error="migration interrupted")
                self._active_migrations.discard(pipeline_session_id)
                if pipeline_session_id in self._deferred_model_closes:
                    self._deferred_model_closes.discard(pipeline_session_id)
                    with contextlib.suppress(Exception):
                        self._close_model_session_now(pipeline_session_id)

    def turboserve_snapshot(self) -> dict[str, object]:
        snapshot = super().turboserve_snapshot()
        migration_total_ms = tuple(getattr(self, "_migration_total_ms", ()))
        migration_first_layer_ms = tuple(getattr(self, "_migration_first_layer_ms", ()))
        migration_transfer_complete_ms = tuple(getattr(self, "_migration_transfer_complete_ms", ()))
        migration_drain_ms = tuple(getattr(self, "_migration_drain_ms", ()))
        migration_background_cleanup_ms = tuple(getattr(self, "_migration_background_cleanup_ms", ()))
        snapshot.update(
            {
                "migration_supported": bool(self._nccl_ranks),
                "migration_backend": "process_nccl" if self._nccl_ranks else None,
                "nccl_ranks": dict(self._nccl_ranks),
                "nccl_pair_warmup_ms": getattr(self, "_nccl_warmup_ms", 0.0),
                "migration_cleanup_failures": getattr(self, "_migration_cleanup_failures", 0),
                "worker_runtime_metrics": {
                    worker_id: dict(self._worker_runtime_metrics.get(worker_id, {})) for worker_id in self._specs
                },
                "session_runtime_metrics": dict(self._session_runtime_metrics),
                "migration_calibration": {
                    # ``average_total_ms`` is retained for diagnostics only;
                    # placement uses first-layer readiness below and never
                    # learns the publisher/source-drain tail as blocking cost.
                    "average_total_ms": sum(migration_total_ms) / len(migration_total_ms)
                    if migration_total_ms
                    else 0.0,
                    "average_first_layer_ready_ms": (
                        sum(migration_first_layer_ms) / len(migration_first_layer_ms)
                        if migration_first_layer_ms
                        else 0.0
                    ),
                    "average_transfer_complete_ms": (
                        sum(migration_transfer_complete_ms) / len(migration_transfer_complete_ms)
                        if migration_transfer_complete_ms
                        else 0.0
                    ),
                    "average_blocking_drain_ms": (
                        sum(migration_drain_ms) / len(migration_drain_ms)
                        if migration_drain_ms
                        else 0.0
                    ),
                    "average_background_cleanup_ms": (
                        sum(migration_background_cleanup_ms) / len(migration_background_cleanup_ms)
                        if migration_background_cleanup_ms
                        else 0.0
                    ),
                    "deferred_publisher_frames": int(
                        getattr(self, "_migration_deferred_publisher_frames", 0)
                    ),
                },
                "migration_diagnostics": (
                    self._migration_diagnostics.snapshot()
                    if isinstance(getattr(self, "_migration_diagnostics", None), MigrationDiagnostics)
                    else {}
                ),
                "model_output_flow_control": self._model_output_flow_snapshot(),
            }
        )
        return snapshot

    def _model_output_flow_snapshot(self) -> dict[str, object]:
        outputs = getattr(self, "_model_outputs", {})
        inflight = getattr(self, "_model_output_inflight", set())
        backlog: dict[str, int] = {}
        for session_id, output in outputs.items():
            qsize = getattr(output, "qsize", None)
            if callable(qsize):
                queued = int(qsize())
            else:
                queued = len(getattr(output, "items", ()))
            backlog[session_id] = queued + int(session_id in inflight)
        return {
            "parent_queue_capacity": _MODEL_OUTPUT_PARENT_QUEUE_SIZE,
            "ack_on_dequeue": True,
            "max_materialized_payloads_per_session": _MODEL_OUTPUT_PARENT_QUEUE_SIZE + 1,
            "backlog": backlog,
            "dropped_payloads": dict(getattr(self, "_model_output_dropped", {})),
        }

    async def aclose(self) -> None:
        for session_id in tuple(self._transport_workers):
            with contextlib.suppress(Exception):
                await self.stop_session(session_id)
        if self._nccl_ranks:
            await asyncio.gather(
                *(self._request(worker_id, "nccl_destroy") for worker_id in self._nccl_ranks),
                return_exceptions=True,
            )
            self._nccl_ranks.clear()
        # The base process pool owns and closes the shared parent-side trace
        # writer for both isolated-worker transports.
        await super().aclose()

    async def _init_nccl(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        workers = sorted(self._active_workers)
        results = await asyncio.gather(
            *(
                self._request(
                    worker_id,
                    "nccl_init",
                    rank=rank,
                    world_size=len(workers),
                    init_method=f"tcp://127.0.0.1:{port}",
                    timeout=_NCCL_INIT_PARENT_TIMEOUT_SECONDS,
                )
                for rank, worker_id in enumerate(workers)
            ),
            return_exceptions=True,
        )
        failures = [
            f"{worker_id}: {result}"
            for worker_id, result in zip(workers, results, strict=True)
            if isinstance(result, BaseException)
        ]
        if failures:
            raise RuntimeError(f"NCCL process-group initialization failed: {'; '.join(failures)}")
        ranks = {worker_id: rank for rank, worker_id in enumerate(workers)}
        warmup_started = time.monotonic()
        collective_results = await asyncio.gather(
            *(
                self._request(
                    worker_id,
                    "nccl_warmup_collective",
                    timeout=_NCCL_INIT_PARENT_TIMEOUT_SECONDS,
                )
                for worker_id in workers
            ),
            return_exceptions=True,
        )
        collective_failures = [
            f"{worker_id}: {result}"
            for worker_id, result in zip(workers, collective_results, strict=True)
            if isinstance(result, BaseException)
        ]
        if collective_failures:
            raise RuntimeError(f"NCCL collective warmup failed: {'; '.join(collective_failures)}")
        for source_index, source_worker_id in enumerate(workers):
            for target_worker_id in workers[source_index + 1 :]:
                pair_results = await asyncio.gather(
                    self._request(
                        source_worker_id,
                        "nccl_warmup_peer",
                        peer_rank=ranks[target_worker_id],
                        send_first=True,
                        timeout=_NCCL_INIT_PARENT_TIMEOUT_SECONDS,
                    ),
                    self._request(
                        target_worker_id,
                        "nccl_warmup_peer",
                        peer_rank=ranks[source_worker_id],
                        send_first=False,
                        timeout=_NCCL_INIT_PARENT_TIMEOUT_SECONDS,
                    ),
                    return_exceptions=True,
                )
                pair_failures = [str(result) for result in pair_results if isinstance(result, BaseException)]
                if pair_failures:
                    raise RuntimeError(
                        f"NCCL peer warmup failed for {source_worker_id}/{target_worker_id}: "
                        f"{'; '.join(pair_failures)}"
                    )
        self._nccl_warmup_ms = (time.monotonic() - warmup_started) * 1000.0
        self._nccl_ranks = ranks

    def _transport_finished(self, session_id: str) -> None:
        self.close_model_session(session_id)

    def _transport_task_done(self, session_id: str, task: asyncio.Task[None]) -> None:
        self._transport_tasks.pop(session_id, None)
        self._transport_workers.pop(session_id, None)
        if not task.cancelled() and task.exception() is not None:
            logger.warning("LiveKit transport failed: session=%s error=%s", session_id, task.exception())

    def _dispatch_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type in {"model_session_ready", "model_session_failed"}:
            session_id = str(event.get("session_id", ""))
            ready = self._model_session_ready.get(session_id)
            expected_worker = self._session_workers.get(session_id)
            reported_worker = str(event.get("worker_id", ""))
            if ready is not None and not ready.done():
                if expected_worker != reported_worker:
                    ready.set_exception(
                        RuntimeError(
                            f"Model session {session_id!r} readiness reported by unexpected worker "
                            f"{reported_worker!r}; expected {expected_worker!r}"
                        )
                    )
                elif event_type == "model_session_ready":
                    ready.set_result(session_id)
                else:
                    ready.set_exception(RuntimeError(str(event.get("error", "model session creation failed"))))
            return
        if event.get("type") == "nccl_first_layer_ready":
            waiter = self._migration_ready_waiters.get(str(event.get("transfer_id", "")))
            if waiter is not None and not waiter.done():
                waiter.set_result(None)
            return
        if event.get("type") in {"model_output", "model_output_eos"}:
            session_id = str(event.get("session_id", ""))
            provisional_events = getattr(self, "_provisional_model_events", None)
            staged = provisional_events.get(session_id) if isinstance(provisional_events, dict) else None
            if staged is not None:
                # The target may finish speculative/progressive compute before
                # the residual copy transaction commits. Retain only target
                # events until source cleanup and ownership publication are
                # durable. A source event can still be delivered after the
                # route flips because publisher draining is intentionally
                # asynchronous; enqueue it on the existing parent queue.
                target_worker = self._pipeline_routes.get(session_id)
                if target_worker is None or str(event.get("worker_id")) == str(target_worker):
                    staged.append(dict(event))
                    return
        if event.get("type") == "worker_start_failed":
            # Startup failures arrive before the base monitor can invoke the
            # unexpected-exit hook. Capture the child traceback/error here;
            # once a worker is active, the monitor remains the single source
            # of process-exit accounting to avoid double counting.
            worker_id = str(event.get("worker_id", "unknown"))
            diagnostics = getattr(self, "_migration_diagnostics", None)
            startup_workers = getattr(self, "_startup", {})
            active_workers = getattr(self, "_active_workers", set())
            if (
                isinstance(diagnostics, MigrationDiagnostics)
                and isinstance(startup_workers, dict)
                and worker_id in startup_workers
                and worker_id not in active_workers
            ):
                diagnostics.record_worker_exit(
                    worker_id,
                    event.get("exitcode"),
                    error=event.get("error"),
                )
            super()._dispatch_event(event)
            return
        if event.get("type") == "model_publisher_frame_tracking":
            session_id = str(event.get("session_id", ""))
            if session_id in self._publisher_frame_tracking:
                self._publisher_frame_tracking[session_id] = bool(event.get("enabled", False))
            return
        if event.get("type") == "model_batch_failed":
            callback = getattr(self._event_sink, "on_motivation_dispatch_failed", None)
            if callable(callback):
                callback(
                    job_ids=tuple(str(value) for value in event.get("job_ids", ())),
                    session_ids=tuple(str(value) for value in event.get("session_ids", ())),
                    error=str(event.get("error", "physical batch dispatch failed")),
                )
            return
        if event.get("type") == "model_output_eos":
            # ``close_model_session`` installs the parent queue sentinel and
            # asks the child to release retained state. It is deliberately
            # idempotent because the transport finally block also calls back.
            session_id = str(event["session_id"])
            current_worker = self._pipeline_routes.get(session_id)
            if current_worker is not None and str(event.get("worker_id")) != str(current_worker):
                # A source EOS can arrive after progressive routing has moved
                # the session. It must not tear down the target model route.
                return
            self.close_model_session(session_id)
            return
        if event.get("type") == "model_output":
            metrics = event.get("runtime_metrics")
            if isinstance(metrics, dict):
                self._worker_runtime_metrics[event["worker_id"]] = {
                    key: value for key, value in metrics.items() if isinstance(value, int | float | str)
                }
            session_metrics = event.get("session_runtime_metrics")
            if isinstance(session_metrics, dict):
                self._session_runtime_metrics[event["session_id"]] = {
                    key: value
                    for key, value in session_metrics.items()
                    if isinstance(value, int | float | str)
                }
            # Materialize the bounded parent output before notifying the
            # Motivation bridge.  The callback may immediately release the
            # scheduler lease and enqueue the next job; publishing the event
            # first otherwise lets that next job overtake this payload and
            # makes the parent queue look empty during a real handoff.
            self._enqueue_model_output(
                event["session_id"],
                _ModelOutput(worker_id=event["worker_id"], payload=event["payload"]),
            )
            output_callback = getattr(self._event_sink, "on_model_output", None)
            if callable(output_callback):
                output_callback(
                    event["worker_id"],
                    event["session_id"],
                    event["payload"],
                    runtime_metrics=metrics if isinstance(metrics, dict) else None,
                    session_runtime_metrics=session_metrics if isinstance(session_metrics, dict) else None,
                )
            return
        super()._dispatch_event(event)


def _nccl_model_worker_main(
    spec: ProcessWorkerSpec,
    config_values: dict[str, Any],
    pipeline_file: str,
    skip_validation: bool,
    security_name: str | None,
    commands: Any,
    events: Any,
) -> None:
    try:
        asyncio.run(
            _run_nccl_model_worker(spec, config_values, pipeline_file, skip_validation, security_name, commands, events)
        )
    except BaseException as exc:
        # Match the generic process worker's startup-failure signal. The
        # parent consumes this only while the worker is still in startup;
        # active-worker exits are accounted for by its process monitor.
        with contextlib.suppress(Exception):
            events.put(
                {
                    "type": "worker_start_failed",
                    "worker_id": spec.worker_id,
                    "error": repr(exc),
                }
            )
        raise
    finally:
        _close_queue(commands, join=False)
        _close_queue(events)


async def _run_nccl_model_worker(
    spec: ProcessWorkerSpec,
    config_values: dict[str, Any],
    pipeline_file: str,
    skip_validation: bool,
    security_name: str | None,
    commands: Any,
    events: Any,
) -> None:
    if not spec.gpu_ids:
        raise RuntimeError("process-nccl requires one CUDA GPU per worker")
    torch.cuda.set_device(int(spec.gpu_ids[0]))
    adapter = LiveKitPipelineAdapter(security_level=SecurityLevel[security_name] if security_name else None)
    adapter.start(pipeline_file, skip_validation=skip_validation, gpu_num=1, gpu_ids=spec.gpu_ids)
    if adapter.stream_mode != STREAM_MODE_BIDIRECTIONAL:
        raise RuntimeError("process-nccl requires a bidirectional pipeline")
    # Honour the operator ceiling in process-NCCL too. Previously this path
    # always auto-sized and then overwrote --max-sessions-per-worker at the
    # parent scheduler, which can violate a measured per-session FPS SLO.
    from .config import LiveKitServeConfig

    config = LiveKitServeConfig(**config_values)
    profile = adapter.configure_session_capacity(config.session_capacity_limit())
    events.put(
        {
            "type": "worker_capacity",
            "worker_id": spec.worker_id,
            "capacity": int((profile or {}).get("effective_capacity", 1)),
            "profile": profile,
        }
    )
    events.put({"type": "worker_status", "worker_id": spec.worker_id, "status": "idle"})
    events.put({"type": "worker_ready", "worker_id": spec.worker_id})
    service = adapter.stream_service.service
    # Use the same callback installer as the generic process transport.  The
    # parent Motivation scheduler and the optional JSONL writer therefore see
    # one identical compute-boundary event in both modes.
    _install_process_dispatch_trace_callback(
        service=service,
        config=config,
        spec=spec,
        events=events,
        logical_cuda_device=_current_cuda_device_for_trace(spec),
    )
    outputs: dict[str, asyncio.Task[None]] = {}
    output_credits: dict[str, asyncio.BoundedSemaphore] = {}
    outgoing: dict[
        str,
        tuple[dict[tuple[Any, ...], torch.Tensor], tuple[TensorTransferGroup, ...]],
    ] = {}
    incoming: dict[str, tuple[dict[str, Any], dict[tuple[Any, ...], torch.Tensor], str, int, int]] = {}
    transfer_tasks: set[asyncio.Task[None]] = set()
    transfer_stream = torch.cuda.Stream(device=torch.device("cuda", int(spec.gpu_ids[0])))

    def warmup_nccl_collective() -> None:
        """Initialize communicator-wide NCCL channels before pair-only P2P."""

        torch.cuda.set_device(int(spec.gpu_ids[0]))
        with torch.cuda.stream(transfer_stream):
            tensor = torch.ones(1, dtype=torch.float32, device=transfer_stream.device)
            dist.all_reduce(tensor)
            transfer_stream.synchronize()

    def warmup_nccl_peer(peer_rank: int, *, send_first: bool) -> None:
        """Eagerly establish one pair's lazy NCCL P2P transport channels."""

        torch.cuda.set_device(int(spec.gpu_ids[0]))
        with torch.cuda.stream(transfer_stream):
            send_tensor = torch.zeros(1, dtype=torch.uint8, device=transfer_stream.device)
            recv_tensor = torch.empty_like(send_tensor)
            send_op = dist.P2POp(dist.isend, send_tensor, peer_rank)
            recv_op = dist.P2POp(dist.irecv, recv_tensor, peer_rank)
            requests = dist.batch_isend_irecv(
                [send_op, recv_op] if send_first else [recv_op, send_op]
            )
            for request in requests:
                request.wait()
            transfer_stream.synchronize()

    def transfer_on_worker_device(
        leaves: dict[tuple[Any, ...], torch.Tensor],
        groups: tuple[TensorTransferGroup, ...],
        *,
        peer_rank: int,
        send: bool,
        progress: LayerTransferProgress | None = None,
    ) -> dict[str, Any]:
        """Run NCCL transfer on a worker thread with the child CUDA device selected."""
        torch.cuda.set_device(int(spec.gpu_ids[0]))

        def group_complete(group: TensorTransferGroup, event: torch.cuda.Event | None) -> None:
            if progress is not None and group.layer_index is not None:
                progress.mark_layer_ready(group.layer_index, event)

        return transfer_tensor_leaves_nccl_streamed(
            leaves,
            peer_rank=peer_rank,
            send=send,
            groups=groups,
            stream=transfer_stream,
            on_group_complete=group_complete if progress is not None else None,
        ).as_dict()

    def import_session_on_worker_device(
        metadata: dict[str, Any],
        leaves: dict[tuple[Any, ...], torch.Tensor],
        *,
        owner_worker_id: str,
        ownership_epoch: int,
        migration_layer_readiness: LayerTransferProgress | None = None,
    ) -> str:
        """Restore received session tensors on a thread bound to this worker GPU."""
        torch.cuda.set_device(int(spec.gpu_ids[0]))
        return service.import_migration_nccl(
            metadata,
            leaves,
            owner_worker_id=owner_worker_id,
            ownership_epoch=ownership_epoch,
            migration_layer_readiness=migration_layer_readiness,
        )

    def start_pump(session_id: str, *, credit_window: int = _MODEL_OUTPUT_PARENT_QUEUE_SIZE) -> None:
        credits = output_credits.get(session_id)
        if credits is None:
            credits = asyncio.BoundedSemaphore(max(1, int(credit_window)))
            output_credits[session_id] = credits
        if session_id not in outputs:
            outputs[session_id] = asyncio.create_task(
                _pump_model_outputs(
                    adapter,
                    service,
                    worker_id=spec.worker_id,
                    session_id=session_id,
                    credits=credits,
                    events=events,
                ),
                name=f"model-output-{session_id}",
            )

    async def stop_pump(session_id: str, *, drop_credit_state: bool = False) -> None:
        task = outputs.pop(session_id, None)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if drop_credit_state:
            output_credits.pop(session_id, None)

    async def result(request_id: str | None, value: Any = True, error: Exception | None = None) -> None:
        if request_id is not None:
            events.put(
                {
                    "type": "command_result",
                    "worker_id": spec.worker_id,
                    "request_id": request_id,
                    "result": value,
                    "error": repr(error) if error else None,
                }
            )

    async def run_nccl_send(command: dict[str, Any]) -> None:
        """Complete one send without monopolizing the child command loop."""
        request_id = command.get("request_id")
        try:
            transfer_id = str(command["transfer_id"])
            leaves, groups = outgoing.pop(transfer_id)
            report = await asyncio.to_thread(
                transfer_on_worker_device,
                leaves,
                groups,
                peer_rank=int(command["target_rank"]),
                send=True,
            )
            await result(request_id, report)
        except Exception as exc:
            await result(request_id, error=exc)

    async def run_nccl_recv(command: dict[str, Any]) -> None:
        """Receive and install one session while unrelated commands are served."""
        request_id = command.get("request_id")
        transfer_task: asyncio.Task[dict[str, Any]] | None = None
        progress: LayerTransferProgress | None = None
        try:
            transfer_id = str(command["transfer_id"])
            metadata, leaves, owner, epoch, credit_window = incoming.pop(transfer_id)
            groups = build_layer_transfer_groups(metadata["tensor_manifest"])
            layer_count = max(
                (group.layer_index for group in groups if group.layer_index is not None),
                default=-1,
            ) + 1
            if layer_count < 1:
                raise RuntimeError("NCCL migration manifest has no layer-grouped cache state")
            progress = LayerTransferProgress(layer_count)

            def receive_transfer() -> dict[str, Any]:
                try:
                    return transfer_on_worker_device(
                        leaves,
                        groups,
                        peer_rank=int(command["source_rank"]),
                        send=False,
                        progress=progress,
                    )
                except BaseException as exc:
                    progress.mark_failed(exc)
                    raise

            transfer_task = asyncio.create_task(
                asyncio.to_thread(receive_transfer),
                name=f"nccl-recv-copy-{transfer_id}",
            )
            await asyncio.to_thread(progress.first_layer_ready.wait)
            if transfer_task.done():
                # Propagate a failure that woke the readiness event before a
                # partially initialized target session can become visible.
                await transfer_task
            session_id = await asyncio.to_thread(
                import_session_on_worker_device,
                metadata,
                leaves,
                owner_worker_id=owner,
                ownership_epoch=epoch,
                migration_layer_readiness=progress,
            )
            events.put(
                {
                    "type": "nccl_first_layer_ready",
                    "worker_id": spec.worker_id,
                    "transfer_id": transfer_id,
                    "session_id": session_id,
                    "progress": progress.snapshot(),
                }
            )
            report = await transfer_task
            progress.mark_complete()
            report["progress"] = progress.snapshot()
            try:
                publisher_tracking_enabled = bool(adapter.enable_publisher_frame_tracking(session_id))
            except Exception:
                publisher_tracking_enabled = False
            events.put(
                {
                    "type": "model_publisher_frame_tracking",
                    "worker_id": spec.worker_id,
                    "session_id": session_id,
                    "enabled": publisher_tracking_enabled,
                }
            )
            start_pump(session_id, credit_window=credit_window)
            await result(request_id, report)
        except Exception as exc:
            if progress is not None:
                progress.mark_failed(exc)
            if transfer_task is not None and not transfer_task.done():
                await asyncio.gather(transfer_task, return_exceptions=True)
            await result(request_id, error=exc)

    try:
        while True:
            command = await asyncio.to_thread(commands.get)
            request_id, kind = command.get("request_id"), command["type"]
            try:
                if kind == "model_create":
                    try:
                        session_id = adapter.create_session(command["config"])
                    except Exception as exc:
                        events.put(
                            {
                                "type": "model_session_failed",
                                "worker_id": spec.worker_id,
                                "session_id": str(command.get("session_id", "")),
                                "error": repr(exc),
                            }
                        )
                        raise
                    try:
                        publisher_tracking_enabled = bool(adapter.enable_publisher_frame_tracking(session_id))
                    except Exception:
                        publisher_tracking_enabled = False
                    events.put(
                        {
                            "type": "model_publisher_frame_tracking",
                            "worker_id": spec.worker_id,
                            "session_id": session_id,
                            "enabled": publisher_tracking_enabled,
                        }
                    )
                    # ACK the model-owned session before starting its output
                    # pump. The parent can now publish ``pipeline_session`` and
                    # register Motivation state before the first payload event.
                    events.put(
                        {
                            "type": "model_session_ready",
                            "worker_id": spec.worker_id,
                            "session_id": session_id,
                        }
                    )
                    start_pump(
                        session_id,
                        credit_window=int(command.get("model_output_credit_window", _MODEL_OUTPUT_PARENT_QUEUE_SIZE)),
                    )
                elif kind == "model_push":
                    adapter.push_chunk(command["session_id"], command["chunk"])
                elif kind == "model_push_batch":
                    try:
                        adapter.push_batch(
                            [(str(session_id), dict(chunk)) for session_id, chunk in command["items"]]
                        )
                    except Exception as exc:
                        if command.get("motivation_job_ids"):
                            events.put(
                                {
                                    "type": "model_batch_failed",
                                    "worker_id": spec.worker_id,
                                    "job_ids": [str(value) for value in command.get("motivation_job_ids", ())],
                                    "session_ids": [
                                        str(value) for value in command.get("motivation_session_ids", ())
                                    ],
                                    "error": repr(exc),
                                }
                            )
                        raise
                elif kind == "model_publisher_frame_progress":
                    adapter.report_publisher_frame_progress(
                        command["session_id"],
                        event=str(command["event"]),
                        frames_delta=int(command["frames_delta"]),
                        sequence=int(command["sequence"]),
                        observed_monotonic_seconds=float(command["observed_monotonic_seconds"]),
                    )
                elif kind == "model_close":
                    await stop_pump(command["session_id"], drop_credit_state=True)
                    adapter.close_session(command["session_id"])
                elif kind == "model_output_credit":
                    credits = output_credits.get(command["session_id"])
                    if credits is not None:
                        with contextlib.suppress(ValueError):
                            credits.release()
                elif kind == "model_output_pause":
                    await stop_pump(command["session_id"])
                    # The model state boundary is the completed compute, not
                    # the real-time consumer.  In latest delivery mode any
                    # payload that is still in the child-local queue is stale
                    # once the source snapshot has been taken; clear it so a
                    # slow LiveKit publisher cannot hold the migration open.
                    discard_pending = getattr(service, "discard_pending_migration_outputs", None)
                    if callable(discard_pending):
                        await asyncio.to_thread(discard_pending, command["session_id"])
                elif kind == "model_output_drain_status":
                    migration_drain_status = getattr(service, "migration_drain_status", None)
                    if not callable(migration_drain_status):
                        raise RuntimeError("Pipeline service does not expose migration drain status")
                    status = await asyncio.to_thread(migration_drain_status, command["session_id"])
                    await result(request_id, status)
                    continue

                elif kind == "model_output_resume":
                    has_session = getattr(service, "has_session", None)
                    if not callable(has_session) or has_session(command["session_id"]):
                        start_pump(command["session_id"])
                elif kind == "nccl_init":
                    device_id = torch.device("cuda", torch.cuda.current_device())
                    await asyncio.to_thread(
                        dist.init_process_group,
                        "nccl",
                        init_method=command["init_method"],
                        rank=command["rank"],
                        world_size=command["world_size"],
                        timeout=timedelta(seconds=_NCCL_INIT_GROUP_TIMEOUT_SECONDS),
                        device_id=device_id,
                    )
                elif kind == "nccl_warmup_peer":
                    await asyncio.to_thread(
                        warmup_nccl_peer,
                        int(command["peer_rank"]),
                        send_first=bool(command["send_first"]),
                    )
                elif kind == "nccl_warmup_collective":
                    await asyncio.to_thread(warmup_nccl_collective)
                elif kind == "scheduler_pause":
                    await asyncio.to_thread(service.pause_scheduler)
                elif kind == "scheduler_resume":
                    service.resume_scheduler()
                elif kind == "nccl_export":
                    metadata = await asyncio.to_thread(service.prepare_migration_nccl_metadata, command["session_id"])
                    tensor_leaves = metadata.pop("_nccl_tensor_leaves")
                    _validate_nccl_source_leaves(tensor_leaves)
                    outgoing[command["transfer_id"]] = (
                        tensor_leaves,
                        build_layer_transfer_groups(metadata["tensor_manifest"]),
                    )
                    await result(request_id, metadata)
                    continue
                elif kind == "nccl_prepare_recv":
                    metadata = command["metadata"]
                    leaves = allocate_tensor_tree_leaves(
                        metadata["tensor_manifest"], torch.device(f"cuda:{spec.gpu_ids[0]}")
                    )
                    incoming[command["transfer_id"]] = (
                        metadata,
                        leaves,
                        command["owner_worker_id"],
                        command["ownership_epoch"],
                        int(command.get("model_output_credit_window", _MODEL_OUTPUT_PARENT_QUEUE_SIZE)),
                    )
                elif kind == "nccl_send":
                    # The parent still waits for this request result before
                    # committing ownership, but unrelated commands can run
                    # while the point-to-point copy is in flight.
                    task = asyncio.create_task(run_nccl_send(command), name=f"nccl-send-{command['transfer_id']}")
                    transfer_tasks.add(task)
                    task.add_done_callback(transfer_tasks.discard)
                    continue
                elif kind == "nccl_recv":
                    task = asyncio.create_task(run_nccl_recv(command), name=f"nccl-recv-{command['transfer_id']}")
                    transfer_tasks.add(task)
                    task.add_done_callback(transfer_tasks.discard)
                    continue
                elif kind == "nccl_commit_source":
                    await stop_pump(command["session_id"], drop_credit_state=True)
                    await asyncio.to_thread(service.commit_migration, command["session_id"])
                elif kind == "nccl_abort_source":
                    service.abort_migration(command["session_id"])
                    outgoing.pop(command.get("transfer_id", ""), None)
                elif kind == "nccl_discard":
                    incoming.pop(command["transfer_id"], None)
                    await stop_pump(command["session_id"], drop_credit_state=True)
                    if service.has_session(command["session_id"]):
                        service.close_session(command["session_id"])
                elif kind == "nccl_destroy":
                    if dist.is_initialized():
                        dist.destroy_process_group()
                elif kind == "shutdown":
                    break
                else:
                    raise ValueError(f"Unknown process-nccl command {kind!r}")
                await result(request_id)
            except Exception as exc:
                await result(request_id, error=exc)
    finally:
        for task in tuple(transfer_tasks):
            task.cancel()
        if transfer_tasks:
            await asyncio.gather(*transfer_tasks, return_exceptions=True)
        for session_id in tuple(outputs):
            await stop_pump(session_id, drop_credit_state=True)
        if dist.is_initialized():
            with contextlib.suppress(Exception):
                dist.destroy_process_group()
        await adapter.aclose()
