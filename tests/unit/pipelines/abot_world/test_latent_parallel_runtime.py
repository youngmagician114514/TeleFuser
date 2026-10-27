"""CPU distributed coverage of the actual ABot frame-sharded execution path."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    from telefuser.core.config import ModelRuntimeConfig, ParallelConfig
    from telefuser.core.module_manager import ModuleManager
    from telefuser.distributed.latent_parallel import LatentParallelRuntime
    from telefuser.models.abot_world_dit import ABotWorldDiT
    from telefuser.pipelines.abot_world.denoising import ABotWorldDenoisingStage
except ImportError as exc:
    pytest.skip(f"ABot optional dependencies unavailable: {exc}", allow_module_level=True)


def _stage() -> ABotWorldDenoisingStage:
    torch.manual_seed(71)
    model = ABotWorldDiT(
        text_len=4,
        in_dim=4,
        out_dim=4,
        dim=24,
        ffn_dim=48,
        freq_dim=8,
        text_dim=16,
        num_heads=3,
        num_layers=2,
        local_attn_size=6,
        sink_size=1,
        downscale_factor_control_adapter=2,
    ).eval()
    manager = ModuleManager(device="cpu", torch_dtype=torch.float32)
    manager.add_module(model, "abot_world_dit")
    return ABotWorldDenoisingStage(
        "latent-test", manager, ModelRuntimeConfig(device_type="cpu", torch_dtype=torch.float32)
    )


def _check_collectives(runtime: LatentParallelRuntime) -> None:
    rank, world = runtime.rank, runtime.world_size
    # Batch size > 1 detects confusion between rank-major and batch-major layouts.
    for batch in (2, 1, 2):
        local = torch.arange(batch * 3 * 2 * 2, dtype=torch.float32).view(batch, 3, 1, 2, 2) + rank * 100
        expected = torch.cat([local - rank * 100 + peer * 100 for peer in range(world)], dim=2)
        output = runtime.gather_frames(local)
        torch.testing.assert_close(output, expected)
        runtime.gather_frames(local + 1)
        torch.testing.assert_close(output, expected)  # Earlier chunks must not alias reusable storage.

        key = torch.arange(batch * 4 * 3 * 8, dtype=torch.float32).view(batch, 4, 3, 8) + rank * 1000
        full_key, full_value = runtime.gather_current_kv(key, key + 100)
        expected_key = torch.cat([key - rank * 1000 + peer * 1000 for peer in range(world)], dim=1)
        torch.testing.assert_close(full_key, expected_key)
        torch.testing.assert_close(full_value, expected_key + 100)
        pointer = full_key.data_ptr()
        repeated_key, _ = runtime.gather_current_kv(key, key + 100)
        assert repeated_key.data_ptr() == pointer
    assert len(runtime._buffers) == 5  # Shape changes replace storage instead of retaining every profile.


def _check_forward(stage: ABotWorldDenoisingStage, runtime: LatentParallelRuntime) -> None:
    model = stage.dit
    # Cover bounded-sink, non-sink and absolute-RoPE variants through eviction.
    for sink, relative in ((1, True), (0, True), (1, False)):
        model.set_causal_attention_window(6, sink)
        model.use_relative_rope = relative
        for block in model.blocks:
            block.self_attn.use_relative_rope = relative
        ref_self, ref_cross = stage._new_cache(2, 4, 4)
        sp_self, sp_cross = stage._new_cache(2, 4, 4)
        for start in (0, 0, 3, 6, 9):
            x = torch.randn(2, 4, 3, 4, 4)
            timestep = torch.full((2, 3), 500.0)
            context = torch.randn(2, 4, 16)
            action = torch.randn(2, 32, 3, 8, 8)
            reference = model(x, timestep, context, action, ref_self, ref_cross, start * 4)
            local = model.forward_latent_parallel(
                runtime.frame_slice(x),
                timestep[:, runtime.rank : runtime.rank + 1],
                context,
                runtime.frame_slice(action),
                sp_self,
                sp_cross,
                start * 4,
            )
            actual = runtime.gather_frames(local)
            torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-5)
            for ref_layer, sp_layer in zip(ref_self, sp_self, strict=True):
                for name in ("k", "v", "global_end_index", "local_end_index"):
                    torch.testing.assert_close(ref_layer[name], sp_layer[name], rtol=2e-5, atol=2e-5)


def _check_sampler(stage: ABotWorldDenoisingStage, runtime: LatentParallelRuntime) -> None:
    model = stage.dit
    model.use_relative_rope = True
    model.set_causal_attention_window(6, 1)
    for block in model.blocks:
        block.self_attn.use_relative_rope = True
    for positions, chunks in (
        (None, ((0, 1), (1, 3), (4, 3), (7, 3))),
        ((0, 2), ((0, 1), (1, 3), (4, 3), (7, 3))),
        (None, ((0, 3), (3, 3), (6, 3))),
    ):
        ref_self, ref_cross = stage._new_cache(2, 4, 4)
        sp_self, sp_cross = stage._new_cache(2, 4, 4)
        ref_generators = [torch.Generator().manual_seed(seed) for seed in (43, 87)]
        sp_generators = [torch.Generator().manual_seed(seed) for seed in (43, 87)]
        context = torch.randn(2, 4, 16)
        for start, frames in chunks:
            noise = torch.randn(2, 4, frames, 4, 4)
            action = torch.randn(2, 32, frames, 8, 8)
            first = torch.randn(2, 4, 1, 4, 4) if start == 0 else None
            stage._latent_parallel_runtime = None
            reference = stage._denoise_block(
                noise,
                context,
                action,
                first,
                ref_self,
                ref_cross,
                start,
                ref_generators,
                stage._scheduler(),
                timestep_positions=positions,
            )
            stage._latent_parallel_runtime = runtime
            actual = stage._denoise_block(
                noise,
                context,
                action,
                first,
                sp_self,
                sp_cross,
                start,
                sp_generators,
                stage._scheduler(),
                timestep_positions=positions,
            )
            torch.testing.assert_close(actual, reference, rtol=5e-4, atol=1e-4)
            for ref_rng, sp_rng in zip(ref_generators, sp_generators, strict=True):
                assert torch.equal(ref_rng.get_state(), sp_rng.get_state())
            for ref_layer, sp_layer in zip(ref_self, sp_self, strict=True):
                for name in ("k", "v", "global_end_index", "local_end_index"):
                    torch.testing.assert_close(ref_layer[name], sp_layer[name], rtol=5e-4, atol=1e-4)


def _distributed_check(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=3, timeout=timedelta(seconds=60))
    try:
        runtime = LatentParallelRuntime()
        _check_collectives(runtime)
        stage = _stage()
        stage.dit.set_latent_parallel_runtime(runtime)
        with torch.inference_mode():
            _check_forward(stage, runtime)
            _check_sampler(stage, runtime)
            noise = torch.randn(2, 4, 10, 4, 4)
            context = torch.randn(2, 4, 16)
            action = torch.randn(2, 32, 10, 8, 8)
            first = torch.randn(2, 4, 1, 4, 4)
            stage._latent_parallel_runtime = None
            reference = stage.process(noise, context, action, first, 43)
            stage._latent_parallel_runtime = runtime
            actual = stage.process(noise, context, action, first, 43)
            assert actual.shape == noise.shape
            torch.testing.assert_close(actual, reference, rtol=5e-4, atol=1e-4)
            from .test_step_batching import _exercise_interactive_driver, _exercise_late_admission

            _exercise_late_admission(stage)
            _exercise_interactive_driver(stage)
        runtime.clear_buffers()
        assert not runtime._buffers
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU Gloo backend unavailable")
def test_three_rank_forward_sampler_rng_cache_and_persistent_buffers(tmp_path: Path) -> None:
    mp.spawn(_distributed_check, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=3, join=True)


def test_latent_parallel_config_has_valid_parallel_degrees() -> None:
    config = ParallelConfig(device_ids=[0, 1, 2], sp_ulysses_degree=3)
    config.validate()
    assert config.world_size == 3
