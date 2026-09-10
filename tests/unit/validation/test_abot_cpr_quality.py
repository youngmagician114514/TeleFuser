from __future__ import annotations

import json
from pathlib import Path

from tools.validation.augment_abot_cpr_quality import (
    augment_result,
    collect_dispatch_quality,
    collect_producer_metrics,
    count_released_action_jobs,
    load_profile_quality,
)


def _write_profile(path: Path) -> None:
    path.write_text(
        "B,S,W,rho,precision,latency_ms,FPS,memory_GB,Q_action,Q_temporal,Q_visual,Q_world,config\n"
        "1,2,6,0,bf16,1,12,1,0.6,0.6,0.6,0.60,b1_s2_w6_rho0_bf16\n"
        "2,2,6,0,bf16,1,12,1,0.62,0.62,0.62,0.62,b2_s2_w6_rho0_bf16\n"
        "4,2,6,0,bf16,1,12,1,0.65,0.65,0.65,0.65,b4_s2_w6_rho0_bf16\n",
        encoding="utf-8",
    )


def test_quality_join_interpolates_b3_and_adjusts_cpr(tmp_path: Path) -> None:
    profile = tmp_path / "profile.csv"
    trace = tmp_path / "dispatch.jsonl"
    _write_profile(profile)
    trace.write_text(
        '{"event_type":"trace_metadata"}\n'
        '{"event_type":"model_dispatch","outcome":"ok","batch_size":1,"sessions":['
        '{"session_id":"server-1","fidelity":"b1_s2_w6_rho0_bf16","frames":12}]}'
        "\n",
        encoding="utf-8",
    )
    quality = load_profile_quality(profile)
    assert quality["b3_s2_w6_rho0_bf16"] == 0.62
    joined, diagnostics = collect_dispatch_quality(trace, quality)
    assert diagnostics == {"missing_fidelity_records": 0, "missing_quality_records": 0}
    result = {
        "sessions": [
            {
                "server_session_id": "server-1",
                "cpr": {
                    "playback_started": True,
                    "engaged_seconds": 10.0,
                    "playable_seconds": 8.0,
                    "stall_seconds": 2.0,
                },
            }
        ],
        "phase_results": [{"phase": "test", "summary": {"cpr_proxy": {}}}],
    }
    augment_result(
        result,
        dispatch_quality=joined,
        profile_quality=quality,
        dispatch_diagnostics=diagnostics,
        profile_path=profile,
    )
    assert result["quality_cpr"]["q_world_frame_weighted"] == 0.6
    assert result["phase_results"][0]["summary"]["quality_cpr"]["quality_adjusted_cpr_proxy"] == 0.738462


def test_producer_metrics_credit_all_action_and_idle_frames_without_transport(tmp_path: Path) -> None:
    profile = tmp_path / "profile.csv"
    trace = tmp_path / "dispatch.jsonl"
    _write_profile(profile)
    rows = [
        {
            "event_type": "model_dispatch",
            "outcome": "ok",
            "batch_size": 1,
            "model_completed_monotonic_seconds": 10.0,
            "model_duration_seconds": 0.4,
            "sessions": [
                {
                    "session_id": "server-1",
                    "motivation_kind": "action",
                    "fidelity": "b1_s2_w6_rho0_bf16",
                    "frames": 12,
                    "fps": 12,
                    "queue_wait_seconds": 0.1,
                    "output_gate_enabled": 0,
                }
            ],
        },
        {
            "event_type": "model_dispatch",
            "outcome": "ok",
            "batch_size": 2,
            "model_completed_monotonic_seconds": 10.5,
            "model_duration_seconds": 1.1,
            "sessions": [
                {
                    "session_id": "server-1",
                    "motivation_kind": "idle",
                    "fidelity": "b2_s2_w6_rho0_bf16",
                    "frames": 12,
                    "fps": 12,
                    "queue_wait_seconds": 0.1,
                    "output_gate_enabled": 0,
                }
            ],
        },
        {
            "event_type": "model_dispatch",
            "outcome": "ok",
            "batch_size": 4,
            "model_completed_monotonic_seconds": 13.0,
            "model_duration_seconds": 0.7,
            "sessions": [
                {
                    "session_id": "server-1",
                    "motivation_kind": "action",
                    "fidelity": "b4_s2_w6_rho0_bf16",
                    "frames": 12,
                    "fps": 12,
                    "queue_wait_seconds": 0.1,
                    "output_gate_enabled": 0,
                }
            ],
        },
    ]
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    metrics = collect_producer_metrics(
        trace,
        load_profile_quality(profile),
        released_action_jobs=3,
    )

    assert metrics["schema_version"] == "abot_producer_metrics_v2"
    assert metrics["jobs_completed"] == 3
    assert metrics["released_action_jobs"] == 3
    assert metrics["action_jobs_completed"] == 2
    assert metrics["action_jobs_on_time"] == 2
    assert metrics["action_job_completion_ratio"] == 0.666667
    assert metrics["idle_jobs_completed"] == 1
    assert metrics["generated_frames"] == 36
    assert metrics["producer_cpr"] == 0.75
    assert metrics["producer_slo_attainment"] == 0.666667
    assert metrics["completed_job_deadline_attainment"] == 0.666667
    assert metrics["producer_fps_per_engaged_session"] == 9.0
    assert metrics["normalized_quality"] == 0.958974
    assert metrics["quality_adjusted_cpr"] == 0.719231
    assert metrics["first_action_job_latency_p95_seconds"] == 0.5
    assert metrics["output_gate_enabled_values"] == [0]


def test_released_action_count_uses_trace_heartbeat_contract(tmp_path: Path) -> None:
    trace = tmp_path / "actions.jsonl"
    rows = [
        {"kind": "action_update", "data": {"controls": ["W"], "heartbeat": False, "reason": "first_nonempty_input"}},
        {"kind": "action_update", "data": {"controls": ["W", "A"], "heartbeat": False, "reason": "state_change"}},
        {"kind": "action_update", "data": {"controls": ["W", "A"], "heartbeat": True, "reason": "one_second_heartbeat"}},
        {"kind": "action_update", "data": {"controls": [], "heartbeat": True, "reason": "one_second_heartbeat"}},
        {"kind": "user_inactive", "data": {}},
        {"kind": "action_update", "data": {"controls": ["D"], "heartbeat": False, "reason": "resume_first_nonempty_input"}},
    ]
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    assert count_released_action_jobs(trace) == 3
