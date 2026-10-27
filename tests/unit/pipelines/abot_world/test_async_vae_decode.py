from __future__ import annotations

import threading

import pytest
import torch

from telefuser.pipelines.abot_world.service import _AsyncVAEDecodeQueue


class _FakeTAEW:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def decode_chunks(self, latents: torch.Tensor, states: list[tuple[str, int]]) -> torch.Tensor:
        self.calls.append(tuple(session_id for session_id, _ in states))
        # Preserve the decoder contract: BCTHW output with one frame.
        return latents[:, :1, :1].clone()


def test_async_decode_batches_distinct_sessions_and_serializes_duplicate_state() -> None:
    stage = _FakeTAEW()
    queue = _AsyncVAEDecodeQueue(stage, max_batch_size=2, batch_wait_seconds=0.01)
    try:
        first = queue.submit(torch.zeros(1, 2, 1, 2, 2), "a", ("a", 0))
        second = queue.submit(torch.ones(1, 2, 1, 2, 2), "b", ("b", 0))
        # A duplicate session is retained for the next physical call, even
        # though it was submitted before the first call finished.
        third = queue.submit(torch.full((1, 2, 1, 2, 2), 2.0), "a", ("a", 1))

        assert first.result(timeout=1.0).frames_per_session == 1
        assert second.result(timeout=1.0).frames_per_session == 1
        assert third.result(timeout=1.0).frames_per_session == 1
        assert stage.calls[0] == ("a", "b")
        assert stage.calls[1] == ("a",)
    finally:
        queue.close()


def test_close_during_batch_rendezvous_cancels_pending_decode_without_thread_failure(monkeypatch) -> None:
    stage = _FakeTAEW()
    queue = _AsyncVAEDecodeQueue(stage, max_batch_size=2, batch_wait_seconds=5)
    waiting = threading.Event()
    original_wait = queue._condition.wait

    def wait(timeout=None):
        if timeout is not None:
            waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(queue._condition, "wait", wait)
    future = queue.submit(torch.zeros(1, 2, 1, 2, 2), "a", ("a", 0))
    try:
        assert waiting.wait(timeout=1)
    finally:
        queue.close()
    with pytest.raises(RuntimeError, match="stopped"):
        future.result(timeout=1)
    assert not queue._thread.is_alive() and not stage.calls
