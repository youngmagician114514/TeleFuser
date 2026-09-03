from __future__ import annotations

import json

import pytest

from telefuser.service.livekit.migration_diagnostics import (
    MigrationDiagnostics,
    classify_migration_error,
)


def test_migration_diagnostics_tracks_phases_and_bounded_records() -> None:
    clock_value = [0.0]

    def clock() -> float:
        return clock_value[0]

    diagnostics = MigrationDiagnostics(recent_limit=1, clock=clock)
    diagnostics.begin("transfer-1", source_worker_id="worker-0", target_worker_id="worker-1")
    diagnostics.phase_started("transfer-1", "export")
    clock_value[0] = 0.125
    diagnostics.phase_finished("transfer-1", "export", success=True)
    diagnostics.phase_started("transfer-1", "transfer")
    clock_value[0] = 0.5
    diagnostics.phase_finished("transfer-1", "transfer", success=False, error="NCCL timeout")
    diagnostics.finish("transfer-1", outcome="failure", error="Worker process worker-1 is not alive")

    snapshot = diagnostics.snapshot()
    assert snapshot["attempts_total"] == 1
    assert snapshot["success_total"] == 0
    assert snapshot["failure_total"] == 1
    assert snapshot["active"] == 0
    assert snapshot["phase_timings"]["export"] == {
        "count": 1,
        "success": 1,
        "failures": 0,
        "total_ms": 125.0,
        "average_ms": 125.0,
    }
    assert snapshot["phase_timings"]["transfer"]["failures"] == 1
    assert snapshot["error_counts"] == {"worker_unavailable": 1}
    assert snapshot["phase_error_counts"] == {"timeout": 1}
    assert snapshot["last_failure"]["failed_phase"] == "transfer"
    assert snapshot["recent"][-1]["error"] == "Worker process worker-1 is not alive"
    json.dumps(snapshot)


def test_migration_diagnostics_records_exit_context_and_rejections() -> None:
    diagnostics = MigrationDiagnostics()
    diagnostics.begin("transfer-1", source_worker_id="worker-0", target_worker_id="worker-1")
    diagnostics.phase_started("transfer-1", "drain")
    diagnostics.record_worker_exit("worker-1", 137)
    diagnostics.reject(
        source_worker_id="worker-0",
        target_worker_id="worker-1",
        reason="NCCL process group is not initialized",
    )

    snapshot = diagnostics.snapshot()
    assert snapshot["worker_exits_total"] == 1
    assert snapshot["worker_exits_by_code"] == {"137": 1}
    assert snapshot["last_worker_exit"]["active_transfers"][0]["phase"] == "drain"
    assert snapshot["rejected_total"] == 1
    assert snapshot["error_counts"]["nccl"] == 1
    assert snapshot["active"] == 1


def test_migration_diagnostics_retains_bounded_layer_transfer_report() -> None:
    diagnostics = MigrationDiagnostics()
    diagnostics.begin("transfer-1", source_worker_id="worker-0", target_worker_id="worker-1")
    diagnostics.set_transport_report(
        "transfer-1",
        {
            "total_bytes": 300,
            "total_duration_ms": 12.5,
            "groups": [
                {"name": "preamble", "layer_index": None, "bytes": 100, "duration_ms": 2.0},
                {"name": "layer_00", "layer_index": 0, "bytes": 200, "duration_ms": 10.5},
            ],
            "progress": {
                "layer_count": 30,
                "ready_layers": 30,
                "complete": True,
                "failed": False,
                "first_layer_ready_ms": 3.25,
                "transfer_complete_ms": 12.5,
                "first_compute_residual_wait_ms": 0.75,
                "host_wait_ms": 1.0,
                "wait_calls": 4,
                "complete_host_wait_ms": 0.25,
                "complete_wait_calls": 1,
            },
        },
    )
    diagnostics.finish("transfer-1", outcome="success")

    report = diagnostics.snapshot()["last"]["transport_report"]
    assert report["total_bytes"] == 300
    assert report["groups"][1] == {
        "name": "layer_00",
        "layer_index": 0,
        "bytes": 200,
        "duration_ms": 10.5,
    }
    assert report["progress"]["first_layer_ready_ms"] == 3.25
    assert report["progress"]["transfer_complete_ms"] == 12.5
    assert report["progress"]["wait_calls"] == 4
    assert report["progress"]["complete_host_wait_ms"] == 0.25


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ("Worker process worker-2 is not alive", "worker_unavailable"),
        ("Timed out waiting for NCCL", "timeout"),
        ("CUDA out of memory", "oom"),
        ("migration cancelled", "cancelled"),
        ("NCCL process group failed", "nccl"),
        ("ownership epoch mismatch", "ownership"),
    ],
)
def test_classify_migration_error_is_stable(error: str, expected: str) -> None:
    assert classify_migration_error(error) == expected
