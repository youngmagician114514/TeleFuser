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


class _BatchService:
    def __init__(self) -> None:
        self.calls: list[list[tuple[str, dict]]] = []

    def push_batch(self, items: list[tuple[str, dict]]) -> None:
        self.calls.append(items)


def test_push_batch_delegates_to_nested_pipeline_service() -> None:
    service = _BatchService()
    adapter = _adapter(service)
    items = [("pipeline-session-1", {"type": "control_state", "controls": ["W"]})]

    adapter.push_batch(items)

    assert service.calls == [items]


def test_denoise_stage_is_forwarded_from_nested_abot_service() -> None:
    denoise_stage = object()
    nested = SimpleNamespace(pipeline=SimpleNamespace(denoise_stage=denoise_stage))
    adapter = _adapter(nested)

    assert adapter.denoise_stage is denoise_stage


def test_uses_relative_rope_is_forwarded_from_nested_abot_service() -> None:
    dit = SimpleNamespace(use_relative_rope=True)
    nested = SimpleNamespace(pipeline=SimpleNamespace(denoise_stage=SimpleNamespace(dit=dit)))
    adapter = _adapter(nested)

    assert adapter.uses_relative_rope is True
