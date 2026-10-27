"""Service-owned asynchronous, cross-session TAeW decode batching."""

from __future__ import annotations

import concurrent.futures
import contextlib
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable

import torch

from .continuous import ABotWorldDecodedBatch


@dataclass
class _AsyncVAERequest:
    """One session row waiting for the serialized TAeW worker."""

    session_id: str
    latents: torch.Tensor
    state: Any
    future: concurrent.futures.Future[ABotWorldDecodedBatch]
    device_event: Any | None = None


class _AsyncVAEDecodeQueue:
    """Batch TAeW rows without concurrent mutation of streaming decoder state.

    TAeW's temporal state is mutable per session, so independent decode calls
    must be serialized.  The queue still exposes a batched physical call: rows
    from different sessions are coalesced for a short rendezvous window, while
    duplicate rows for one session remain ordered in the deque.  Latent rows
    are cloned before returning to the DiT caller because the distributed
    exchange arena may reuse its storage for the next wave.
    """

    def __init__(
        self,
        stage: Any,
        *,
        max_batch_size: int,
        batch_wait_seconds: float = 0.002,
        on_batch: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._stage = stage
        self._max_batch_size = max(1, int(max_batch_size))
        self._batch_wait_seconds = max(0.0, float(batch_wait_seconds))
        self._condition = threading.Condition()
        self._pending: deque[_AsyncVAERequest] = deque()
        self._stopping = False
        self._on_batch = on_batch
        self._stream: torch.cuda.Stream | None = None
        self._thread = threading.Thread(target=self._run, name="abot-taew-decode", daemon=True)
        self._thread.start()

    @staticmethod
    def _batch_key(latents: torch.Tensor) -> tuple[Any, ...]:
        return (tuple(latents.shape[1:]), str(latents.dtype), str(latents.device))

    def submit(
        self, latents: torch.Tensor, session_id: str, state: Any
    ) -> concurrent.futures.Future[ABotWorldDecodedBatch]:
        future: concurrent.futures.Future[ABotWorldDecodedBatch] = concurrent.futures.Future()
        if latents.ndim != 5:
            future.set_exception(ValueError(f"TAeW async decode expects BCTHW latents, got {tuple(latents.shape)}"))
            return future
        # The model exchange owns the source tensor.  Clone synchronously on
        # its current stream so the next DiT wave can safely reuse the source.
        owned = latents.detach().contiguous().clone()
        event = None
        if owned.device.type == "cuda":
            event = torch.cuda.Event()
            torch.cuda.current_stream(owned.device).record_event(event)
        request = _AsyncVAERequest(session_id, owned, state, future, event)
        with self._condition:
            if self._stopping:
                future.set_exception(RuntimeError("asynchronous TAeW decode queue is stopped"))
                return future
            self._pending.append(request)
            self._condition.notify()
        return future

    def _take_batch(self) -> list[_AsyncVAERequest]:
        first = self._pending.popleft()
        selected = [first]
        retained: deque[_AsyncVAERequest] = deque()
        key = self._batch_key(first.latents)
        selected_sessions = {first.session_id}
        while self._pending:
            request = self._pending.popleft()
            if (
                len(selected) < self._max_batch_size
                and request.session_id not in selected_sessions
                and self._batch_key(request.latents) == key
            ):
                selected.append(request)
                selected_sessions.add(request.session_id)
            else:
                retained.append(request)
        self._pending = retained
        return selected

    @staticmethod
    def _wait_for_events(requests: Sequence[_AsyncVAERequest]) -> None:
        cuda_requests = [request for request in requests if request.device_event is not None]
        if not cuda_requests:
            return
        device = cuda_requests[0].latents.device
        with torch.cuda.device(device):
            stream = torch.cuda.current_stream(device)
            for request in cuda_requests:
                stream.wait_event(request.device_event)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._stopping:
                    self._condition.wait()
                if self._stopping and not self._pending:
                    return
                if self._batch_wait_seconds:
                    deadline = time.monotonic() + self._batch_wait_seconds
                    while len(self._pending) < self._max_batch_size and not self._stopping:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        self._condition.wait(timeout=remaining)
                if not self._pending:
                    # close() can cancel the rendezvous cohort while this
                    # thread releases the condition inside wait().
                    continue
                requests = self._take_batch()
            try:
                device = requests[0].latents.device
                if device.type == "cuda" and self._stream is None:
                    self._stream = torch.cuda.Stream(device=device)
                context = torch.cuda.stream(self._stream) if self._stream is not None else contextlib.nullcontext()
                started = time.monotonic()
                with context, torch.inference_mode():
                    self._wait_for_events(requests)
                    latents = torch.cat([request.latents for request in requests], dim=0)
                    states = [request.state for request in requests]
                    decoded = self._stage.decode_chunks(latents, states)
                # Complete the Future at the actual decoded-frame boundary.
                # Only this decode worker waits; the DiT actor uses another
                # stream and continues dispatching independent work.
                if self._stream is not None:
                    self._stream.synchronize()
                frames = int(decoded.shape[2])
                if self._on_batch is not None:
                    self._on_batch(
                        {
                            "batch_size": len(requests),
                            "seconds": time.monotonic() - started,
                            "frames_per_session": frames,
                        }
                    )
                for row, request in enumerate(requests):
                    request.future.set_result(ABotWorldDecodedBatch(decoded[row : row + 1], frames))
            except BaseException as exc:
                for request in requests:
                    request.future.set_exception(exc)

    def close(self) -> None:
        with self._condition:
            if self._stopping:
                return
            self._stopping = True
            pending = list(self._pending)
            self._pending.clear()
            for request in pending:
                request.future.set_exception(RuntimeError("asynchronous TAeW decode queue stopped"))
            self._condition.notify_all()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=10.0)
