"""Normalize bounded model facts forwarded through the LiveKit serving path."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

Scalar = bool | int | float | str

_OBSERVABILITY_KEYS = (
    "type",
    "index",
    "stage",
    "fps",
    "scheduler",
    "scheduler_metrics",
    "runtime_metrics",
    "session_runtime_metrics",
    "worker_runtime_metrics",
    "measurement",
    "stream_progress",
    "pipeline_residence_seconds",
    "applied_control_latency_seconds",
    "chunk_elapsed_seconds",
    "control_to_chunk_seconds",
)
_SCHEDULER_KEYS = (
    "batch_size",
    "compute_seconds",
    "queue_wait_seconds",
    "input_prepare_seconds",
    "cache_collate_seconds",
    "denoise_seconds",
    "cache_scatter_seconds",
    "vae_encode_seconds",
    "vae_decode_seconds",
    "postprocess_seconds",
    "taew_decode_items",
    "taew_decode_batch_size",
    "taew_decode_invocations",
    "taew_decode_mode",
)
_STAGE_NAMES = {
    "input_prepare": "input_prepare_seconds",
    "cache_collate": "cache_collate_seconds",
    "encode": "vae_encode_seconds",
    "encode_actor": "vae_encode_seconds",
    "encode_worker": "vae_encode_seconds",
    "denoise": "denoise_seconds",
    "denoise_actor": "denoise_seconds",
    "denoise_worker": "denoise_seconds",
    "dit": "denoise_seconds",
    "cache_scatter": "cache_scatter_seconds",
    "decode": "vae_decode_seconds",
    "decode_actor": "vae_decode_seconds",
    "decode_worker": "vae_decode_seconds",
    "vae_encode": "vae_encode_seconds",
    "vae_decode": "vae_decode_seconds",
    "postprocess": "postprocess_seconds",
    "tensor_to_frames": "postprocess_seconds",
}


def _as_mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def bounded_scalars(value: object) -> dict[str, Scalar]:
    """Keep only small scalar facts suitable for a bounded runtime snapshot."""
    result: dict[str, Scalar] = {}
    for key, item in _as_mapping(value).items():
        if not isinstance(key, str) or len(key) > 80:
            continue
        if isinstance(item, str):
            if len(item) > 160:
                continue
        elif isinstance(item, float) and not math.isfinite(item):
            continue
        elif not isinstance(item, (bool, int, float)):
            continue
        result[key] = item
    return result


def merge_scalar_maps(*values: object) -> dict[str, Scalar]:
    result: dict[str, Scalar] = {}
    for value in values:
        result.update(bounded_scalars(value))
    return result


def build_observability_payload(chunk: Mapping[str, Any], frame_count: int) -> dict[str, Any]:
    """Forward bounded metadata without copying video/audio payloads."""
    nested = _as_mapping(chunk.get("data"))
    sources = (chunk, nested)
    payload: dict[str, Any] = {}
    for key in _OBSERVABILITY_KEYS:
        for source in sources:
            if key not in source:
                continue
            value = source[key]
            if key in {
                "scheduler",
                "scheduler_metrics",
                "runtime_metrics",
                "session_runtime_metrics",
                "worker_runtime_metrics",
            }:
                payload[key] = bounded_scalars(value)
            elif key in {"measurement", "stream_progress"}:
                payload[key] = _bounded_metadata(value)
            elif isinstance(value, (bool, int, float, str)):
                payload[key] = value
            break
    payload["type"] = str(chunk.get("type", payload.get("type", "")))
    payload["frame_count"] = max(0, int(frame_count))
    return payload


def _bounded_metadata(value: object, *, depth: int = 0) -> object:
    """Copy small nested metadata while dropping tensors, media, and deep objects."""
    if isinstance(value, (bool, int, str)):
        return value if not isinstance(value, str) or len(value) <= 160 else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if depth > 2:
        return None
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 64:
                break
            if not isinstance(key, str) or len(key) > 80:
                continue
            bounded = _bounded_metadata(item, depth=depth + 1)
            if bounded is not None:
                result[key] = bounded
        return result
    return None


@dataclass(frozen=True)
class ServingOutputFacts:
    """Normalized facts from either a media chunk or a status message."""

    kind: str
    scheduler: dict[str, Any]
    measurement: dict[str, Any]
    worker_metrics: dict[str, Scalar]
    session_metrics: dict[str, Scalar]
    frame_count: int
    fps: float | int | str | None

    @property
    def has_metrics(self) -> bool:
        return bool(self.scheduler or self.measurement or self.worker_metrics or self.session_metrics)


def _stage_metric_name(key: str) -> str | None:
    normalized = key.strip().lower()
    if normalized in _STAGE_NAMES.values():
        return normalized
    if normalized.endswith("_seconds"):
        normalized = normalized[:-8]
    for suffix in ("_actor", "_worker", "_queue", "_wait"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    return _STAGE_NAMES.get(normalized)


def extract_serving_output_facts(
    payload: object,
    *,
    runtime_metrics: object = None,
    session_runtime_metrics: object = None,
) -> ServingOutputFacts:
    """Normalize model-specific and generic stream-service output metadata."""
    message = _as_mapping(payload)
    nested = _as_mapping(message.get("data"))
    sources = (message, nested)
    scheduler: dict[str, Any] = {}
    measurement: dict[str, Any] = {}
    for source in sources:
        for key in ("scheduler", "scheduler_metrics"):
            scheduler.update(_as_mapping(source.get(key)))
        measurement.update(_as_mapping(source.get("measurement")))
    for source in (measurement, scheduler, message, nested):
        for key in _SCHEDULER_KEYS:
            if key not in scheduler and key in source:
                scheduler[key] = source[key]
    phases = _as_mapping(measurement.get("phases"))
    for key, value in phases.items():
        metric_name = _stage_metric_name(str(key))
        if metric_name is None or metric_name in scheduler:
            continue
        if isinstance(value, Mapping):
            value = value.get("seconds")
        scheduler[metric_name] = value

    raw_frames: object = None
    for source in sources:
        if "frame_count" in source:
            candidate = source["frame_count"]
            if raw_frames is None:
                raw_frames = candidate
            try:
                if int(candidate) > 0:
                    break
            except (TypeError, ValueError):
                pass
        if "frames" in source and isinstance(source["frames"], (list, tuple)):
            raw_frames = len(source["frames"])
            break
    if "frames" in measurement:
        try:
            current_frames = int(raw_frames or 0)
        except (TypeError, ValueError):
            current_frames = 0
        if current_frames <= 0:
            raw_frames = measurement["frames"]
    try:
        frame_count = max(0, int(raw_frames or 0))
    except (TypeError, ValueError):
        frame_count = 0

    fps: float | int | str | None = None
    for source in sources:
        if source.get("fps") is not None:
            fps = source["fps"]
            break
        progress = _as_mapping(source.get("stream_progress"))
        if progress.get("fps") is not None:
            fps = progress["fps"]
            break
    worker_metrics = merge_scalar_maps(runtime_metrics, message.get("worker_runtime_metrics"))
    session_metrics = merge_scalar_maps(
        session_runtime_metrics,
        message.get("session_runtime_metrics"),
        message.get("runtime_metrics"),
        message.get("scheduler_metrics"),
    )
    return ServingOutputFacts(
        kind=str(message.get("type", "")),
        scheduler=scheduler,
        measurement=measurement,
        worker_metrics=worker_metrics,
        session_metrics=session_metrics,
        frame_count=frame_count,
        fps=fps,
    )
