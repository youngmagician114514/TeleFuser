"""Independent local actors preserve native rollout and causal clean state."""

import json
import pickle

import pytest
import torch

from telefuser.pipelines.abot_world.fidelity import ABotWorldFidelity

from .test_latent_parallel_runtime import _stage
from .test_latent_step_runtime import _pipelines


@torch.inference_mode()
@pytest.mark.parametrize("compact", (False, True))
def test_independent_phase_handoff_matches_native_with_rolling_quality_windows(compact):
    torch.set_num_threads(1)
    native, denoise = _pipelines(_stage(), count=2)
    _, refresh = _pipelines(denoise.denoise_stage, count=0)
    references = list(native._interactive_sessions.values())
    d_sessions = list(denoise._interactive_sessions.values())
    refresh.taew_decode_stage.create_decode_state = object
    refresh.taew_decode_stage.warmup_first_frame = lambda *args: None
    d_runtime = denoise.create_continuous_runtime(max_batch_size=6, granularity="latent_step", local_execution=True)
    d_runtime._sessions.update(denoise._interactive_sessions)
    r_runtime = refresh.create_continuous_runtime(max_batch_size=6, granularity="latent_step", local_execution=True)
    r_sessions = [
        r_runtime.create_conditioned_session(d.session_id, d.first_frame_latent, d.prompt_emb, seed=17 + i)
        for i, d in enumerate(d_sessions)
    ]
    for profile in ("b1_s4_w18_rho0_bf16",) + ("b1_s2_w6_rho0_bf16",) * 4 + ("b1_s4_w18_rho0_bf16",):
        fidelity = ABotWorldFidelity.from_profile_name(profile)
        expected = native.generate_next_blocks_raw(references, [{"W": True}, {"D": True}], fidelity=fidelity)
        for d, r, action in zip(d_sessions, r_sessions, ({"W": True}, {"D": True})):
            if compact:
                d_runtime.ensure_local_capacity(d.session_id, fidelity)
                r_runtime.ensure_local_capacity(r.session_id, fidelity)
            d_runtime.submit(d, action, fidelity=fidelity)
            for _ in range(fidelity.denoise_steps):
                assert not d_runtime.step((d.session_id,))
            with pytest.raises(ValueError, match="unfinished"):
                d_runtime.submit(d, action, fidelity=fidelity)
            latents = d_runtime.export_denoised(d.session_id)
            if compact:
                d_runtime.park_session(d.session_id)
                assert not any(sid == d.session_id for sid, _ in d_runtime._local)
                assert d.self_cache[0]["k"].shape[0] == 0
                assert d_runtime._paged_states[d.session_id].table
            r_runtime.submit_refresh(r, latents, action, fidelity=fidelity, chunk_start=d.next_latent_frame)
        # D has independently admitted and finished B while R held A ready.
        outputs = r_runtime.step(tuple(r.session_id for r in r_sessions))
        for index, output in enumerate(outputs):
            torch.testing.assert_close(output.decoded, expected.decoded[index : index + 1], rtol=6e-4, atol=2e-4)
            d_runtime.commit_external_refresh(output.session_id, r_runtime.export_clean_delta(output.session_id))
            if compact:
                r_runtime.park_session(output.session_id)
        for d, reference in zip(d_sessions, references):
            assert d.next_latent_frame == reference.next_latent_frame
            assert torch.equal(d.generator.get_state(), reference.generator.get_state())
    for d, reference in zip(d_sessions, references):
        d_runtime.release_session(d.session_id)
        valid = int(reference.self_cache[0]["local_end_index"].item())
        for actual, expected in zip(d.self_cache, reference.self_cache):
            for field in ("k", "v"):
                torch.testing.assert_close(actual[field][:, :valid], expected[field][:, :valid], rtol=6e-4, atol=2e-4)
    r_runtime.close()


@torch.inference_mode()
@pytest.mark.parametrize("move_before_first_chunk", (False, True))
def test_canonical_sst_moves_d_and_r_workers_without_changing_rollout_or_rng(move_before_first_chunk):
    torch.set_num_threads(1)
    native, d_pipeline = _pipelines(_stage(), count=1)
    other_pipelines = [_pipelines(d_pipeline.denoise_stage, count=0)[1] for _ in range(3)]
    pipelines = [d_pipeline, *other_pipelines]
    runtimes = [
        pipeline.create_continuous_runtime(max_batch_size=3, granularity="latent_step", local_execution=True)
        for pipeline in pipelines
    ]
    d_runtime, other_d, r_runtime, other_r = runtimes
    d_session = d_pipeline._interactive_sessions["session-0"]
    reference = native._interactive_sessions["session-0"]
    d_runtime._sessions[d_session.session_id] = d_session
    sid = d_session.session_id
    r_session = r_runtime.create_conditioned_session(
        sid, d_session.first_frame_latent, d_session.prompt_emb, seed=17, initialize_decode=False
    )
    if move_before_first_chunk:
        # Runtime-created sessions have metadata but zero allocated KV rows.
        for attribute in ("self_cache", "cross_cache"):
            for entry in getattr(d_session, attribute):
                for field in ("k", "v"):
                    entry[field] = entry[field].new_empty((0, *entry[field].shape[1:]))
        payload = d_runtime.export_canonical_state(sid)
        assert isinstance(payload["generator_state"], str)
        assert payload["capacity_tokens"] == 0
        d_session = other_d.import_canonical_state(sid, payload, seed=999)
        assert d_session.self_cache[0]["k"].numel() == 0
        d_runtime.release_session(sid, materialize=False)
        d_runtime, other_d = other_d, d_runtime
    for index, window in enumerate((6, 6, 18, 6, 6, 18)):
        if index:
            for source, target in ((d_runtime, other_d), (r_runtime, other_r)):
                payload = source.export_canonical_state(sid)
                assert isinstance(payload["generator_state"], str)
                target.import_canonical_state(sid, payload, seed=999)
                assert target._sessions[sid].taew_decode_state is None
                for page_id, exported in zip(target._paged_states[sid].table, payload["paged_self"], strict=True):
                    assert target.kv_pages.page(page_id).data_ptr() != exported.data_ptr()
                source.release_session(sid, materialize=False)
            d_runtime, other_d = other_d, d_runtime
            r_runtime, other_r = other_r, r_runtime
            d_session, r_session = d_runtime._sessions[sid], r_runtime._sessions[sid]
        fidelity = ABotWorldFidelity.from_profile_name(f"b1_s4_w{window}_rho0_bf16")
        actions = {"W" if index % 2 else "D": True}
        expected = native.generate_next_blocks_raw([reference], [actions], fidelity=fidelity)
        d_runtime.ensure_local_capacity(sid, fidelity)
        d_runtime.submit(d_session, actions, fidelity=fidelity)
        with pytest.raises(RuntimeError, match="idle parked"):
            d_runtime.export_canonical_state(sid)
        for _ in range(fidelity.denoise_steps):
            d_runtime.step((sid,))
        latent = d_runtime.export_denoised(sid)
        d_runtime.park_session(sid)
        r_runtime.ensure_local_capacity(sid, fidelity)
        r_runtime.submit_refresh(r_session, latent, actions, fidelity=fidelity, chunk_start=index * 3)
        output = r_runtime.step((sid,))[0]
        torch.testing.assert_close(output.decoded, expected.decoded, rtol=6e-4, atol=2e-4)
        d_runtime.commit_external_refresh(sid, r_runtime.export_clean_delta(sid))
        r_runtime.park_session(sid)
        assert d_session.next_latent_frame == r_session.next_latent_frame == reference.next_latent_frame
        assert torch.equal(d_session.generator.get_state(), reference.generator.get_state())
        valid = d_runtime._cursors[sid][1]
        for layer, (received, expected_cache) in enumerate(zip(d_session.self_cache, reference.self_cache)):
            for field in ("k", "v"):
                actual = torch.cat(d_runtime._paged_states[sid].layer_pages(layer, ("k", "v").index(field)))
                torch.testing.assert_close(actual.unsqueeze(0), expected_cache[field][:, :valid], rtol=6e-4, atol=2e-4)
    for runtime in runtimes:
        runtime.close(materialize=False)


@torch.inference_mode()
def test_cuda_sst_rng_metadata_preserves_noise_across_devices():
    from telefuser.service.livekit.nccl_transfer import flatten_tensor_tree, rebuild_tensor_tree

    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA devices")
    _, source = _pipelines(_stage(), count=1)
    _, target = _pipelines(source.denoise_stage, count=0)
    runtimes = [
        pipeline.create_continuous_runtime(max_batch_size=3, granularity="latent_step", local_execution=True)
        for pipeline in (source, target)
    ]
    source.device, target.device = torch.device("cuda:0"), torch.device("cuda:1")
    sid = "session-0"
    session = source._interactive_sessions[sid]
    session.first_frame_latent = session.first_frame_latent.to(source.device)
    session.prompt_emb = session.prompt_emb.to(source.device)
    session.generator = torch.Generator(device=source.device).manual_seed(17)
    for attribute in ("self_cache", "cross_cache"):
        for entry in getattr(session, attribute):
            for key, value in entry.items():
                if isinstance(value, torch.Tensor):
                    entry[key] = value.to(source.device)
    runtimes[0]._sessions[sid] = session
    for _ in range(4):
        torch.randn(1, 4, 3, 4, 4, device=source.device, generator=session.generator)
        payload = runtimes[0].export_canonical_state(sid)
        skeleton, manifest, leaves = flatten_tensor_tree(payload)
        assert all(entry["path"] != ["generator_state"] for entry in manifest)
        assert json.loads(json.dumps(skeleton)) == skeleton
        # TCP serializes metadata; the tensor lane places GPU leaves on the
        # destination device. RNG must survive without an H2D/D2H round trip.
        received = rebuild_tensor_tree(
            pickle.loads(pickle.dumps(skeleton)), {path: value.to(target.device) for path, value in leaves.items()}
        )
        installed = runtimes[1].import_canonical_state(sid, received, seed=999)
        assert torch.equal(installed.generator.get_state(), session.generator.get_state())
        expected = torch.randn(1, 4, 3, 4, 4, device=source.device, generator=session.generator)
        actual = torch.randn(1, 4, 3, 4, 4, device=target.device, generator=installed.generator)
        torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0, rtol=0)
    for runtime in runtimes:
        runtime.close(materialize=False)
