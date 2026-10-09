from __future__ import annotations

import pytest

from telefuser.service.livekit.model_contracts import (
    ABOT_WORLD_CAPABILITIES,
    ModelServingCapabilities,
)
from telefuser.service.livekit.motivation_scheduler import (
    GpuSchedulingState,
    MotivationProfile,
    MotivationScheduler,
    MotivationSchedulerConfig,
    StaticMotivationProfileTable,
)


def _profiles() -> StaticMotivationProfileTable:
    return StaticMotivationProfileTable(
        [
            MotivationProfile(
                batch_size=1,
                fidelity="b1",
                latency_seconds=0.1,
                quality=1.0,
                memory_gb=1.0,
            )
        ]
    )


def test_abot_capabilities_preserve_one_second_job_contract() -> None:
    assert ABOT_WORLD_CAPABILITIES.output_fps == 12.0
    assert ABOT_WORLD_CAPABILITIES.frames_per_job == 12
    assert ABOT_WORLD_CAPABILITIES.job_duration_seconds == pytest.approx(1.0)


def test_scheduler_accepts_backend_capabilities_without_changing_policy_defaults() -> None:
    capabilities = ModelServingCapabilities(
        model_name="matrix-game-2",
        output_fps=25.0,
        frames_per_job=12,
    )
    scheduler = MotivationScheduler(
        _profiles(),
        config=MotivationSchedulerConfig(model_capabilities=capabilities),
    )
    scheduler.add_gpu(GpuSchedulingState("gpu-0", memory_free_gb=8.0))
    assert scheduler.model_capabilities is capabilities
    scheduler.register_session("s1", owner_gpu="gpu-0")
    assert scheduler.session("s1").compatibility_key == ()


def test_scheduler_can_pull_a_backend_batch_key_at_admission() -> None:
    class Provider:
        def batch_compatibility_key(self, session_id: str, *, fidelity: str | None = None):
            del fidelity
            return ("matrix", session_id)

    scheduler = MotivationScheduler(_profiles(), batch_compatibility_provider=Provider())
    scheduler.add_gpu(GpuSchedulingState("gpu-0", memory_free_gb=8.0))
    state = scheduler.register_session("s1", owner_gpu="gpu-0")
    assert state.compatibility_key == ("matrix", "s1")


def test_capabilities_reject_invalid_output_contract() -> None:
    with pytest.raises(ValueError):
        ModelServingCapabilities(model_name="bad", output_fps=0.0, frames_per_job=12)
    with pytest.raises(ValueError):
        ModelServingCapabilities(model_name="bad", output_fps=12.0, frames_per_job=0)

