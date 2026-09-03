from __future__ import annotations

import json

import pytest

from telefuser.service.livekit.migration_hysteresis import MigrationCooldownPolicy


def test_commit_starts_per_session_cooldown_and_expires() -> None:
    clock_value = [0.0]

    policy = MigrationCooldownPolicy(cooldown_seconds=60.0, clock=lambda: clock_value[0])

    assert policy.admit("session-a", "gpu-0", "gpu-1").allowed
    policy.record_commit("session-a", "gpu-0", "gpu-1", committed_at=10.0)

    clock_value[0] = 20.0
    blocked = policy.admit("session-a", "gpu-1", "gpu-0")
    assert not blocked.allowed
    assert blocked.reason == "cooldown"
    assert blocked.retry_at == pytest.approx(70.0)
    assert blocked.cooldown_remaining_seconds == pytest.approx(50.0)

    clock_value[0] = 70.0
    assert policy.admit("session-a", "gpu-1", "gpu-0").allowed

    snapshot = policy.snapshot()
    assert snapshot["tracked_sessions"] == 1
    assert snapshot["active_cooldowns"] == 0
    assert snapshot["commits_total"] == 1
    assert snapshot["suppressed_total"] == 1
    assert snapshot["sessions"][0]["commit_count"] == 1
    json.dumps(snapshot)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cooldown_seconds": 0.0},
        {"cooldown_seconds": 60.0, "enabled": False},
    ],
)
def test_zero_or_disabled_policy_is_an_ablation(kwargs: dict[str, object]) -> None:
    policy = MigrationCooldownPolicy(**kwargs)
    policy.record_commit("session-a", "gpu-0", "gpu-1", committed_at=1.0)

    admission = policy.admit("session-a", "gpu-1", "gpu-0", now=1.1)

    assert admission.allowed
    assert admission.reason == "disabled"
    assert policy.snapshot(now=1.1)["active_cooldowns"] == 0


def test_optional_gain_margin_and_emergency_bypass() -> None:
    policy = MigrationCooldownPolicy(cooldown_seconds=60.0, min_gain_seconds=0.5)

    insufficient = policy.admit("session-a", "gpu-0", "gpu-1", expected_gain_seconds=0.2)
    assert not insufficient.allowed
    assert insufficient.reason == "insufficient_gain"

    assert policy.admit("session-a", "gpu-0", "gpu-1", expected_gain_seconds=0.5).allowed
    policy.record_commit("session-a", "gpu-0", "gpu-1", committed_at=10.0)

    bypass = policy.admit(
        "session-a",
        "gpu-1",
        "gpu-0",
        expected_gain_seconds=0.1,
        now=11.0,
        emergency=True,
    )
    assert bypass.allowed
    assert bypass.reason == "emergency"
    assert policy.snapshot()["emergency_bypasses_total"] == 1


def test_same_owner_and_forget_are_explicit() -> None:
    policy = MigrationCooldownPolicy()

    same_owner = policy.admit("session-a", "gpu-0", "gpu-0")
    assert not same_owner.allowed
    assert same_owner.reason == "same_owner"

    policy.record_commit("session-a", "gpu-0", "gpu-1", committed_at=2.0)
    assert policy.cooldown_until("session-a") == pytest.approx(62.0)
    policy.forget("session-a")
    assert policy.cooldown_until("session-a") is None
    assert policy.snapshot()["tracked_sessions"] == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"cooldown_seconds": -1.0}, "cooldown_seconds"),
        ({"min_gain_seconds": float("nan")}, "min_gain_seconds"),
        ({"recent_limit": -1}, "recent_limit"),
    ],
)
def test_constructor_rejects_invalid_policy_values(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        MigrationCooldownPolicy(**kwargs)


def test_recent_events_are_bounded() -> None:
    policy = MigrationCooldownPolicy(cooldown_seconds=0.0, recent_limit=2)
    for index in range(4):
        policy.admit(f"session-{index}", "gpu-0", "gpu-1", now=float(index))

    recent = policy.snapshot(now=4.0)["recent"]
    assert len(recent) == 2
    assert [entry["session_id"] for entry in recent] == ["session-2", "session-3"]
