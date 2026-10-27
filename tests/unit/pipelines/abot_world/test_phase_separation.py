"""CPU distributed checks of the real denoise/refresh state path."""

from __future__ import annotations

from concurrent.futures import Future
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from telefuser.pipelines.abot_world.continuous import ABotWorldDecodedBatch
from telefuser.pipelines.abot_world.fidelity import ABotWorldFidelity

from .test_latent_parallel_runtime import _stage
from .test_latent_step_runtime import _pipelines


@torch.inference_mode()
def _phase_worker(rank: int, rendezvous: str, async_decode: bool) -> None:
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        torch.manual_seed(19)
        native, pipeline = _pipelines(_stage(), count=2)
        references = list(native._interactive_sessions.values())
        sessions = list(pipeline._interactive_sessions.values())

        def submit_decode(latents, session):
            result = Future()
            decoded = pipeline.taew_decode_stage.decode_chunks(latents, [session.taew_decode_state])
            result.set_result(ABotWorldDecodedBatch(decoded, int(decoded.shape[2])))
            return result

        runtime = pipeline.create_continuous_runtime(
            max_batch_size=6,
            granularity="latent_step",
            refresh_ranks=(1,),
            decode_submitter=submit_decode if async_decode else None,
        )
        profiles = ("b1_s4_w18_rho0_bf16",) + ("b1_s2_w6_rho0_bf16",) * 3 + ("b1_s4_w18_rho0_bf16",)
        for profile in profiles:
            fidelity = ABotWorldFidelity.from_profile_name(profile)
            expected = native.generate_next_blocks_raw(references, [{"W": True}, {"D": True}], fidelity=fidelity)
            runtime.submit(sessions[0], {"W": True}, fidelity=fidelity)
            for _ in range(fidelity.denoise_steps):
                assert runtime.step((sessions[0].session_id,)) == []
            runtime.submit(sessions[1], {"D": True}, fidelity=fidelity)
            outputs = runtime.step(tuple(session.session_id for session in sessions))
            # A's refresh and B's denoise are physically placed on disjoint
            # GPUs in one dispatch, rather than serialized phase waves.
            assert {key.refreshing for key in runtime.last_tasks} == {False, True}
            assert all(key.owner == int(key.refreshing) for key in runtime.last_tasks)
            while runtime.has_pending:
                outputs.extend(runtime.step())
            if rank == 0:
                actual = {
                    output.session_id: output.decode_future.result().decoded if async_decode else output.decoded
                    for output in outputs
                }
                for index, session in enumerate(sessions):
                    torch.testing.assert_close(
                        actual[session.session_id], expected.decoded[index : index + 1], rtol=6e-4, atol=2e-4
                    )
            for reference, session in zip(references, sessions, strict=True):
                assert torch.equal(reference.generator.get_state(), session.generator.get_state())
        # Returning state checks every clean KV layer, not just video output.
        for reference, session in zip(references, sessions, strict=True):
            runtime.release_session(session.session_id)
            valid = int(reference.self_cache[0]["local_end_index"].item())
            for actual, expected in zip(session.self_cache, reference.self_cache, strict=True):
                for name in ("k", "v"):
                    torch.testing.assert_close(actual[name][:, :valid], expected[name][:, :valid], rtol=6e-4, atol=2e-4)
        # Only final latent payloads leave D, only current clean tails leave R.
        chunks = len(profiles) * len(sessions)
        latent_bytes = sessions[0].first_frame_latent.numel() * 3 * sessions[0].first_frame_latent.element_size()
        reference = references[0].self_cache[0]["k"]
        kv_bytes = (
            len(references[0].self_cache) * 2 * 12 * reference.shape[2] * reference.shape[3] * reference.element_size()
        )
        assert runtime.phase_transfer.logical_latent_bytes == chunks * latent_bytes
        assert runtime.phase_transfer.logical_kv_bytes == chunks * kv_bytes
        if rank == 0:
            assert runtime.phase_transfer.latent_bytes == chunks * latent_bytes
        else:
            assert runtime.phase_transfer.kv_bytes == chunks * kv_bytes
        assert runtime.logical_kv_send_bytes == 0  # No per-layer attention exchange.
        runtime.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("async_decode", (False, True))
def test_refresh_pool_matches_native_across_mixed_phases_and_window_changes(tmp_path, async_decode) -> None:
    mp.spawn(_phase_worker, args=(f"file://{tmp_path / 'phase-init'}", async_decode), nprocs=2, join=True)


def test_invalid_phase_partition_fails_before_state_admission() -> None:
    _, pipeline = _pipelines(_stage(), count=0)
    with pytest.raises(ValueError, match="refresh_ranks"):
        pipeline.create_continuous_runtime(max_batch_size=3, granularity="latent_step", refresh_ranks=(0,))
