from __future__ import annotations

from telefuser.service.livekit.metric_facts import (
    build_observability_payload,
    extract_serving_output_facts,
)


def test_extracts_lingbot_style_status_measurements() -> None:
    facts = extract_serving_output_facts(
        {
            "type": "status",
            "stage": "chunk_sent",
            "fps": 12,
            "measurement": {
                "frames": 12,
                "compute_seconds": 0.7,
                "phases": {
                    "encode_actor_seconds": 0.1,
                    "denoise_worker_seconds": 0.5,
                    "decode_worker_seconds": 0.1,
                },
            },
            "runtime_metrics": {"first_chunk_seconds": 1.2},
            "scheduler_metrics": {"first_output_latency_seconds": 0.8},
        }
    )

    assert facts.kind == "status"
    assert facts.frame_count == 12
    assert facts.fps == 12
    assert facts.scheduler["compute_seconds"] == 0.7
    assert facts.scheduler["vae_encode_seconds"] == 0.1
    assert facts.scheduler["denoise_seconds"] == 0.5
    assert facts.scheduler["vae_decode_seconds"] == 0.1
    assert facts.session_metrics["first_chunk_seconds"] == 1.2
    assert facts.has_metrics


def test_payload_forwards_metadata_without_video_frames() -> None:
    payload = build_observability_payload(
        {
            "type": "status",
            "frames": [object()],
            "data": {"measurement": {"compute_seconds": 0.3}},
            "runtime_metrics": {"active": 1},
        },
        frame_count=1,
    )

    assert payload["measurement"] == {"compute_seconds": 0.3}
    assert payload["runtime_metrics"] == {"active": 1}
    assert payload["frame_count"] == 1
    assert "frames" not in payload


def test_status_payload_uses_measurement_frame_count_without_media() -> None:
    payload = build_observability_payload(
        {
            "type": "status",
            "measurement": {"frames": 12},
        },
        frame_count=0,
    )
    facts = extract_serving_output_facts(payload)

    assert facts.frame_count == 12


def test_payload_preserves_nested_phase_seconds() -> None:
    payload = build_observability_payload(
        {"type": "status", "measurement": {"phases": {"encode": {"seconds": 0.1}}}},
        frame_count=0,
    )
    facts = extract_serving_output_facts(payload)

    assert facts.scheduler["vae_encode_seconds"] == 0.1
