from __future__ import annotations

from types import SimpleNamespace

from telefuser.service.livekit.pipeline_adapter import LiveKitPipelineAdapter


class _SessionMetricsService:
    def __init__(self) -> None:
        self.calls: list[str | None] = []

    def runtime_metrics(self, session_id: str | None = None) -> dict[str, object]:
        self.calls.append(session_id)
        return {"session_id": session_id or "aggregate"}


class _LegacyMetricsService:
    def __init__(self) -> None:
        self.calls = 0

    def runtime_metrics(self) -> dict[str, object]:
        self.calls += 1
        return {"scheduler_mode": "legacy"}


def _adapter(service: object) -> LiveKitPipelineAdapter:
    adapter = object.__new__(LiveKitPipelineAdapter)
    adapter.stream_service = SimpleNamespace(service=service)
    return adapter


def test_runtime_metrics_forwards_optional_session_id() -> None:
    service = _SessionMetricsService()
    adapter = _adapter(service)

    assert adapter.runtime_metrics() == {"session_id": "aggregate"}
    assert adapter.runtime_metrics("pipeline-session-1") == {"session_id": "pipeline-session-1"}
    assert service.calls == [None, "pipeline-session-1"]


def test_runtime_metrics_falls_back_to_legacy_no_argument_service() -> None:
    service = _LegacyMetricsService()
    adapter = _adapter(service)

    assert adapter.runtime_metrics("pipeline-session-1") == {"scheduler_mode": "legacy"}
    assert service.calls == 1


def test_runtime_metrics_returns_none_when_service_is_unavailable() -> None:
    adapter = _adapter(object())

    assert adapter.runtime_metrics("pipeline-session-1") is None
