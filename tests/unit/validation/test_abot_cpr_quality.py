from __future__ import annotations

from pathlib import Path

from tools.validation.augment_abot_cpr_quality import (
    augment_result,
    collect_dispatch_quality,
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
