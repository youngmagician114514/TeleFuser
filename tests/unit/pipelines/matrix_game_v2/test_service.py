import asyncio

import pytest

from telefuser.pipelines.matrix_game_v2.action import MatrixGameAction
from telefuser.pipelines.matrix_game_v2.backend import MatrixGameStep
from telefuser.pipelines.matrix_game_v2.service import MatrixGameV2Service
from telefuser.service.livekit.model_contracts import ModelServingCapabilities


class FakeMatrixBackend:
    capabilities = ModelServingCapabilities(
        model_name="matrix-game-test",
        output_fps=25.0,
        frames_per_job=12,
        supports_cross_session_batching=False,
        supports_session_state_transfer=False,
    )
    max_sessions = 2

    def __init__(self) -> None:
        self.actions: list[MatrixGameAction] = []
        self.closed: list[str] = []
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def create_session(self, session_id: str, config: dict):
        return session_id

    def step(self, session: str, action: MatrixGameAction) -> MatrixGameStep:
        self.actions.append(action)
        return MatrixGameStep(frames=[(session, action.keyboard)], index=len(self.actions))

    def close_session(self, session: str) -> None:
        self.closed.append(session)


async def _collect_one(service: MatrixGameV2Service, session_id: str) -> dict:
    async for chunk in service.pull_chunks(session_id):
        return chunk
    raise AssertionError("session closed before producing a chunk")


def test_matrix_service_lifecycle_and_capacity() -> None:
    backend = FakeMatrixBackend()
    service = MatrixGameV2Service(backend=backend)
    service.start()
    try:
        assert service.configure_session_capacity(2)["effective_capacity"] == 2
        service.create_session({"session_id": "a"})
        service.create_session({"session_id": "b"})
        with pytest.raises(RuntimeError, match="capacity exhausted"):
            service.create_session({"session_id": "c"})
        service.push_chunk("a", {"controls": "ArrowUp"})
        chunk = asyncio.run(_collect_one(service, "a"))
        assert chunk["type"] == "chunk"
        assert chunk["fps"] == 25.0
        assert backend.actions[0].keyboard == ("w",)
    finally:
        service.stop()
    assert backend.closed == ["a", "b"]


def test_matrix_service_rejects_cross_session_batch() -> None:
    backend = FakeMatrixBackend()
    service = MatrixGameV2Service(backend=backend)
    service.start()
    try:
        service.create_session({"session_id": "a"})
        service.create_session({"session_id": "b"})
        with pytest.raises(RuntimeError, match="cross-session batching"):
            service.push_batch([("a", {}), ("b", {})])
    finally:
        service.stop()
