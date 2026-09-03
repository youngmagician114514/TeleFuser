from __future__ import annotations

import pytest

from telefuser.service.livekit.motivation_batch_gate import (
    MotivationBatchGate,
    geometric_batch_wait_seconds,
)
from telefuser.service.livekit.motivation_scheduler import (
    MotivationProfile,
    StaticMotivationProfileTable,
)


def _table() -> StaticMotivationProfileTable:
    return StaticMotivationProfileTable(
        (
            MotivationProfile(1, "high", 0.70, 0.68, 20.0),
            MotivationProfile(2, "high", 1.00, 0.68, 30.0),
            MotivationProfile(1, "low", 0.40, 0.60, 20.0),
            MotivationProfile(2, "low", 0.75, 0.60, 30.0),
        )
    )


def test_geometric_window_matches_serial_b1_budget() -> None:
    assert geometric_batch_wait_seconds(0.70, 1.00) == pytest.approx(0.40)
    assert geometric_batch_wait_seconds(0.40, 0.75) == pytest.approx(0.05)
    assert geometric_batch_wait_seconds(0.40, 0.90) == pytest.approx(0.0)


def test_gate_matches_b1_and_b2_by_fidelity() -> None:
    gate = MotivationBatchGate(_table())

    high = gate.window(gpu_id="gpu-0", fidelity="high")
    low = gate.window(gpu_id="gpu-0", fidelity="low")

    assert high.fidelity == "high"
    assert high.b1_latency_seconds == pytest.approx(0.70)
    assert high.b2_latency_seconds == pytest.approx(1.00)
    assert high.wait_seconds == pytest.approx(0.40)
    assert low.wait_seconds == pytest.approx(0.05)


def test_gate_pairs_batch_prefixed_offline_fidelities() -> None:
    table = StaticMotivationProfileTable(
        (
            MotivationProfile(1, "b1_s4_w18_rho0_bf16", 0.4102, 0.68, 28.13),
            MotivationProfile(2, "b2_s4_w18_rho0_bf16", 0.7386, 0.68, 37.12),
        )
    )
    gate = MotivationBatchGate(table)

    by_family = gate.window(gpu_id="gpu-0")
    from_b1_candidate = gate.window(gpu_id="gpu-0", fidelity="b1_s4_w18_rho0_bf16")

    assert by_family.fidelity == "b1_s4_w18_rho0_bf16"
    assert by_family.wait_seconds == pytest.approx(2 * 0.4102 - 0.7386)
    assert from_b1_candidate.b2_latency_seconds == pytest.approx(0.7386)
    assert from_b1_candidate.wait_seconds == pytest.approx(2 * 0.4102 - 0.7386)


def test_gate_uses_largest_common_window_without_selected_fidelity() -> None:
    gate = MotivationBatchGate(_table())

    window = gate.window(gpu_id="gpu-0")

    assert window.fidelity == "high"
    assert window.wait_seconds == pytest.approx(0.40)
    assert gate.wait_seconds(gpu_id="gpu-0") == pytest.approx(0.40)


def test_missing_b2_or_fidelity_disables_profile_wait() -> None:
    b1_only = StaticMotivationProfileTable((MotivationProfile(1, "high", 0.70, 0.68, 20.0),))
    gate = MotivationBatchGate(b1_only)

    missing_b2 = gate.window(gpu_id="gpu-0")
    missing_fidelity = gate.window(gpu_id="gpu-0", fidelity="low")

    assert missing_b2.fidelity is None
    assert missing_b2.wait_seconds == 0.0
    assert missing_fidelity.wait_seconds == 0.0


def test_gate_cap_is_applied_after_geometric_calculation() -> None:
    gate = MotivationBatchGate(_table(), max_wait_seconds=0.10)

    window = gate.window(gpu_id="gpu-0", fidelity="high")

    assert window.wait_seconds == pytest.approx(0.10)


@pytest.mark.parametrize(
    "b1,b2",
    [(0.0, 1.0), (float("nan"), 1.0), (1.0, 0.0), (1.0, float("inf"))],
)
def test_geometric_window_rejects_invalid_latencies(b1: float, b2: float) -> None:
    with pytest.raises(ValueError, match="latency"):
        geometric_batch_wait_seconds(b1, b2)


def test_gate_rejects_invalid_cap_and_gpu_id() -> None:
    with pytest.raises(ValueError, match="max_wait_seconds"):
        MotivationBatchGate(_table(), max_wait_seconds=-1.0)

    gate = MotivationBatchGate(_table())
    with pytest.raises(ValueError, match="gpu_id"):
        gate.window(gpu_id="")
