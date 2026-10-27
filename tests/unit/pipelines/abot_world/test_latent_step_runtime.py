"""Whole-latent task routing, real DiT parity, and persistent state ownership."""

from __future__ import annotations

from collections import OrderedDict
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from telefuser.pipelines.abot_world.fidelity import ABotWorldFidelity
from telefuser.pipelines.abot_world.interactive import ABotWorldInteractivePipeline, ABotWorldInteractiveSession

from .test_latent_parallel_runtime import _stage


@torch.inference_mode()
def test_mixed_prompt_initialization_preserves_retained_rows():
    from telefuser.models.abot_world_dit import CausalWanCrossAttention

    torch.set_num_threads(1)
    model = CausalWanCrossAttention(16, 4)
    x, context = torch.randn(2, 2, 16), torch.randn(2, 4, 16)
    cache = {"k": torch.zeros(1, 4, 4, 4), "v": torch.zeros(1, 4, 4, 4), "is_init": False, "sequence_length": 0}
    expected = model(x[:1], context[:1], cache)
    saved = {name: cache[name].clone() for name in ("k", "v")}
    mixed = {name: torch.cat([saved[name], torch.zeros_like(saved[name])]) for name in ("k", "v")}
    mixed.update(is_init=False, sequence_length=4, initialize_rows=((1, 2),))
    projected_rows = []
    hook = model.k.register_forward_pre_hook(lambda module, args: projected_rows.append(args[0].shape[0]))
    context[0].normal_()  # A retained row must use cached KV, even if its input changes.
    actual = model(x, context, mixed)
    hook.remove()
    torch.testing.assert_close(actual[:1], expected)
    assert projected_rows == [1]
    for name in ("k", "v"):
        assert torch.equal(mixed[name][:1], saved[name])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_first", [False, True])
@torch.inference_mode()
def test_batched_sampler_is_bitwise_equal_to_native_scalar_sampling(device, dtype, with_first):
    from telefuser.pipelines.abot_world.denoising import ABotWorldDenoisingStage
    from telefuser.pipelines.abot_world.latent_step_runtime import ABotWorldLatentStepRuntime, _LatentState

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    class Task(SimpleNamespace):
        @property
        def refreshing(self):
            return self.step_index == len(self.timesteps)

    scheduler = ABotWorldDenoisingStage._scheduler()
    wave, references, old_indices = [], {}, {}
    for name, positions, step in (("a", (0, 3), 0), ("b", (0, 1, 2, 3), 1), ("c", (0, 3), 1)):
        indices = torch.tensor([0, 250, 500, 750])[list(positions)]
        task = Task(
            session_id=name,
            step_index=step,
            timesteps=scheduler.timesteps[indices].to(device),
            sampling_sigmas=tuple(scheduler.sigmas[index] for index in indices),
            scheduler=scheduler,
            full_shape=(1, 4, 3, 4, 4),
            latent=torch.empty(0, device=device),
            generator=torch.Generator(device=device).manual_seed(ord(name)),
            first_frame=torch.randn(1, 4, 1, 4, 4, device=device, dtype=dtype) if with_first else None,
        )
        wave.append(task)
        references[name] = torch.Generator(device=device).manual_seed(ord(name))
        old_indices[name] = step
    local = [
        _LatentState(task.session_id, index, [], [], latent=torch.randn(1, 4, 1, 4, 4, device=device, dtype=dtype))
        for task in wave
        for index in range(3)
    ]
    prediction = torch.randn(9, 4, 1, 4, 4, device=device, dtype=dtype)
    timestep = torch.cat(
        [
            wave[row // 3].timesteps[old_indices[state.session_id] : old_indices[state.session_id] + 1].view(1, 1)
            for row, state in enumerate(local)
        ]
    )
    if with_first:
        for row, state in enumerate(local):
            if state.latent_index == 0:
                timestep[row] = 0
    noise = {
        task.session_id: torch.randn(task.full_shape, generator=references[task.session_id], dtype=dtype, device=device)
        for task in wave
        if task.step_index + 1 < len(task.timesteps)
    }
    expected = []
    for row, state in enumerate(local):
        task = wave[row // 3]
        value = ABotWorldDenoisingStage._x0_prediction(
            prediction[row : row + 1], state.latent, timestep[row : row + 1], scheduler
        )
        if task.session_id in noise:
            value = scheduler.add_noise(
                value,
                noise[task.session_id][:, :, state.latent_index : state.latent_index + 1],
                task.timesteps[task.step_index + 1],
            )
        if task.first_frame is not None and state.latent_index == 0:
            value.copy_(task.first_frame)
        expected.append(value)
    driver = ABotWorldLatentStepRuntime.__new__(ABotWorldLatentStepRuntime)
    driver.pipeline = SimpleNamespace(
        denoise_stage=SimpleNamespace(_x0_prediction=ABotWorldDenoisingStage._x0_prediction)
    )
    driver._requests = {task.session_id: task for task in wave}
    driver._sample(wave, local, prediction, timestep)
    torch.testing.assert_close(torch.cat([state.latent for state in local]), torch.cat(expected), rtol=0, atol=0)
    for task in wave:
        assert torch.equal(task.generator.get_state(), references[task.session_id].get_state())


@torch.inference_mode()
def test_arena_repack_copies_valid_history_instead_of_full_capacity():
    from telefuser.pipelines.abot_world.latent_step_execution import LatentStepArena

    def row(valid):
        return SimpleNamespace(
            valid_tokens=valid,
            self_cache=[{"k": torch.randn(1, 18, 1, 4), "v": torch.randn(1, 18, 1, 4)}],
            cross_cache=[
                {"k": torch.randn(1, 12, 1, 4), "v": torch.randn(1, 12, 1, 4), "is_init": True, "sequence_length": 4}
            ],
        )

    a, b = row(6), row(3)
    expected = a.self_cache[0]["k"][:, :6].clone()
    arena = LatentStepArena(8)
    arena.bind([a])
    # Two K/V fields; cat and scatter each copy only the valid self/text prefix.
    assert arena.cache_copy_bytes == 2 * 2 * (6 + 4) * 4 * 4
    arena.bind([b])
    torch.testing.assert_close(a.self_cache[0]["k"][:, :6], expected)
    assert a.self_cache[0]["k"].shape[1] == 18


@torch.inference_mode()
def test_detached_siblings_share_one_history_and_rebind_independent_write_rows():
    from telefuser.pipelines.abot_world.latent_step_execution import LatentStepArena
    from telefuser.pipelines.abot_world.latent_step_runtime import _LatentState

    group = object()
    siblings = [
        _LatentState(
            "session",
            index,
            [{"k": torch.ones(1, 18, 1, 4), "v": torch.full((1, 18, 1, 4), 2.0)}],
            [dict(k=torch.full((1, 4, 1, 4), 3.0), v=torch.ones(1, 4, 1, 4), is_init=True, sequence_length=4)],
            valid_tokens=6,
            capacity_tokens=18,
            cache_group=group,
        )
        for index in range(3)
    ]
    arena = LatentStepArena(6)
    arena.bind(siblings)
    before = arena.cache_copy_bytes
    arena.detach()
    assert arena.cache_copy_bytes - before == 2 * (6 + 4) * 4 * 4
    assert len({state.self_cache[0]["k"].data_ptr() for state in siblings}) == 1
    arena._storage["self_cache"][0]["k"].zero_()
    for state in siblings:
        assert state.self_cache[0]["k"][:, :6].eq(1).all()
    arena.bind(siblings)
    assert len({state.self_cache[0]["k"].data_ptr() for state in siblings}) == 3
    siblings[0].self_cache[0]["k"].zero_()
    for state in siblings[1:]:
        assert state.self_cache[0]["k"][:, :6].eq(1).all()


@torch.inference_mode()
def test_arena_retains_unmoved_rows_when_a_batch_member_changes():
    from telefuser.pipelines.abot_world.latent_step_execution import LatentStepArena
    from telefuser.pipelines.abot_world.latent_step_runtime import _LatentState

    def row():
        return _LatentState(
            session_id="session",
            latent_index=0,
            valid_tokens=6,
            self_cache=[{"k": torch.randn(1, 18, 1, 4), "v": torch.randn(1, 18, 1, 4)}],
            cross_cache=[
                {"k": torch.randn(1, 4, 1, 4), "v": torch.randn(1, 4, 1, 4), "is_init": True, "sequence_length": 4}
            ],
        )

    a, b, c = row(), row(), row()
    arena = LatentStepArena(8)
    arena.bind([a, b])
    pointer = a.self_cache[0]["k"].data_ptr()
    retained = a.self_cache[0]["k"][:, :6].clone()
    departed = b.self_cache[0]["k"][:, :6].clone()
    generation = arena.generation
    copies = arena.cache_copy_bytes
    a.capacity_tokens = b.capacity_tokens = 6
    arena.bind([a, b])
    assert a.self_cache[0]["k"].data_ptr() == pointer
    assert a.self_cache[0]["k"].shape[1] == 18
    assert arena.generation == generation and arena.cache_copy_bytes == copies
    before = arena.cache_copy_bytes
    arena.bind([a, c])
    assert a.self_cache[0]["k"].data_ptr() == pointer
    torch.testing.assert_close(a.self_cache[0]["k"][:, :6], retained)
    torch.testing.assert_close(b.self_cache[0]["k"][:, :6], departed)
    # Preserve the departing row once, stage and bind only the incoming row.
    assert arena.cache_copy_bytes - before == 3 * 2 * (6 + 4) * 4 * 4


def _pipelines(stage, count=3):
    stage.dit.set_causal_attention_window(18, 6)
    pairs = [ABotWorldInteractivePipeline(device="cpu", torch_dtype=torch.float32) for _ in range(2)]
    for pipeline in pairs:
        pipeline.config = SimpleNamespace(height=8, width=8)
        pipeline.denoise_stage = stage
        pipeline.taew_decode_stage = SimpleNamespace(
            decode_chunks=lambda latents, states: latents[:, :3].repeat_interleave(4, dim=2),
            last_decode_metrics=lambda: {},
            clear_decode_state=lambda state: None,
        )
    for row in range(count):
        image = torch.randn(1, 4, 1, 4, 4)
        prompt = torch.randn(1, 4, 16)
        for pipeline in pairs:
            self_cache, cross_cache = stage._new_cache(1, 4, 4)
            session = ABotWorldInteractiveSession(
                session_id=f"session-{row}",
                first_frame_latent=image.clone(),
                prompt_emb=prompt.clone(),
                self_cache=self_cache,
                cross_cache=cross_cache,
                scheduler=stage._scheduler(),
                generator=torch.Generator().manual_seed(17 + row),
                taew_decode_state=object(),
            )
            pipeline._interactive_sessions[session.session_id] = session
    return pairs


@torch.inference_mode()
def test_explicit_latent_step_heads_reject_stale_or_missing_dependencies_and_mix_steps():
    torch.set_num_threads(1)
    _, pipeline = _pipelines(_stage(), count=2)
    runtime = pipeline.create_continuous_runtime(max_batch_size=6, granularity="latent_step", local_execution=True)
    sessions = list(pipeline._interactive_sessions.values())
    runtime.submit(sessions[0], {"W": True})
    first = runtime.ready_latent_steps()
    assert len(first) == 3 and {key.step_index for key in first} == {0}
    with pytest.raises(ValueError, match="dependencies"):
        runtime.step_latent_tasks(first[:1])
    runtime.step_latent_tasks(first)
    with pytest.raises(ValueError, match="dependencies"):
        runtime.step_latent_tasks(first)
    runtime.submit(sessions[1], {"D": True})
    mixed = runtime.ready_latent_steps()
    assert {key.step_index for key in mixed} == {0, 1}
    runtime.step_latent_tasks(mixed)
    assert runtime.last_batch_size == 6
    assert {key.step_index for key in runtime.last_local_tasks} == {0, 1}
    while runtime.has_pending:
        keys = runtime.ready_latent_steps()
        runtime.step_latent_tasks([key for key in keys if key.refreshing == keys[0].refreshing])
    runtime.close(materialize=False)


@torch.inference_mode()
def _exercise_runtime_created_session():
    from PIL import Image

    torch.set_num_threads(1)
    native, pipeline = _pipelines(_stage(), count=0)
    first, prompt = torch.randn(4, 1, 4, 4), torch.randn(1, 4, 16)
    for target in (native, pipeline):
        target.preload_models = lambda: None
        target.preprocess_image = lambda *args: torch.zeros(3, 8, 8)
        target.vae_stage = SimpleNamespace(process=lambda *args, **kwargs: (first.clone(), None))
        target.text_encoding_stage = SimpleNamespace(process=lambda *args: [prompt.clone()])
        target.taew_decode_stage.create_decode_state = object
        target.taew_decode_stage.warmup_first_frame = lambda *args: None
    image = Image.new("RGB", (8, 8))
    reference = native.create_interactive_session(image, "world", seed=7)
    driver = pipeline.create_continuous_runtime(max_batch_size=3, granularity="latent_step")
    session = driver.create_session(image, "world", seed=7)
    assert session.self_cache[0]["k"].numel() == session.cross_cache[0]["k"].numel() == 0
    assert not driver._owners and not driver._local
    with pytest.raises(RuntimeError, match="owned by a latent-step runtime"):
        pipeline.generate_next_blocks_raw([session], [{}])
    for profile in ("b1_s2_w6_rho0_bf16", "b1_s4_w18_rho0_bf16", "b1_s2_w6_rho0_bf16"):
        fidelity = ABotWorldFidelity.from_profile_name(profile)
        expected = native.generate_next_blocks_raw([reference], [{"W": True}], fidelity=fidelity)
        driver.submit(session, {"W": True}, fidelity=fidelity)
        outputs = []
        while driver.has_pending:
            outputs.extend(driver.step())
        if driver.is_output_rank:
            torch.testing.assert_close(outputs[0].decoded, expected.decoded, rtol=6e-4, atol=2e-4)
        assert torch.equal(session.generator.get_state(), reference.generator.get_state())
    driver.release_session(session.session_id)
    for actual, expected in zip(session.self_cache, reference.self_cache, strict=True):
        assert actual["global_end_index"].item() == expected["global_end_index"].item()
        assert actual["local_end_index"].item() == expected["local_end_index"].item()
    # An admitted lifecycle can also leave before ever publishing a Job.
    abandoned = driver.create_session(image, "world", session_id="no-input")
    driver.release_session(abandoned.session_id, materialize=False)
    assert abandoned.closed
    empty = driver.create_session(image, "world", session_id="return-unused")
    driver.release_session(empty.session_id)
    assert empty.latent_runtime_owner is None and empty.self_cache[0]["k"].shape[0] == 1
    assert torch.count_nonzero(empty.self_cache[0]["k"]) == 0
    driver.close()


def test_runtime_created_session_defers_kv_until_dispatch_and_matches_native_generation():
    _exercise_runtime_created_session()


@pytest.mark.parametrize("park", (False, True))
@torch.inference_mode()
def test_local_job_shared_kv_matches_native_with_interleaved_queries_and_quality_changes(park):
    torch.set_num_threads(1)
    native, pipeline = _pipelines(_stage(), count=4)
    references = list(native._interactive_sessions.values())
    sessions = list(pipeline._interactive_sessions.values())
    driver = pipeline.create_continuous_runtime(max_batch_size=12, granularity="latent_step", local_execution=True)
    # Stable slots can interleave siblings after cohort churn. Deliberately
    # exercise that layout rather than assuming chunk-major physical rows.
    driver._stable_row_order = lambda: tuple(
        job * 3 + frame for frame in range(3) for job in range(len(driver.last_tasks) // 3)
    )
    for cycle in range(7):
        windows = (18, 6, 12, 18) if cycle % 2 == 0 else (6, 18, 6, 12)
        expected = {}
        for index, (session, reference, window) in enumerate(zip(sessions, references, windows, strict=True)):
            fidelity = ABotWorldFidelity.from_profile_name(f"b1_s{2 if index % 2 else 4}_w{window}_rho0_bf16")
            if park and cycle:
                driver.ensure_local_capacity(session.session_id, fidelity)
            actions = {"W" if index % 2 else "D": True}
            expected[session.session_id] = native.generate_next_blocks_raw(
                [reference], [actions], fidelity=fidelity
            ).decoded
            driver.submit(session, actions, fidelity=fidelity)
        driver.step((sessions[0].session_id,))
        complete = {}
        while driver.has_pending:
            for output in driver.step():
                complete[output.session_id] = output.decoded
            for rows in driver.arena._groups:
                states = [driver.arena.cohort[row] for row in rows]
                assert len({id(state.self_cache[0]["paged_state"]) for state in states}) == 1
            # No all-layer dense execution arena remains. Each session table
            # references its own physical pages; inputs retain 12 query slots.
            assert not driver.arena._storage
            assert len(driver.arena.self_cache[0]["paged_jobs"]) <= 4
        for session, reference in zip(sessions, references, strict=True):
            torch.testing.assert_close(complete[session.session_id], expected[session.session_id], rtol=6e-4, atol=2e-4)
            assert torch.equal(session.generator.get_state(), reference.generator.get_state())
            if park:
                driver.park_session(session.session_id)
    driver.close(materialize=False)


@torch.inference_mode()
def test_relative_attention_batches_different_chunk_positions_and_append_phases(monkeypatch):
    from telefuser.models import abot_world_dit

    torch.set_num_threads(1)
    stage = _stage()
    native, pipeline = _pipelines(stage)
    references = list(native._interactive_sessions.values())
    sessions = list(pipeline._interactive_sessions.values())
    driver = pipeline.create_continuous_runtime(max_batch_size=9, granularity="latent_step")
    fidelity = ABotWorldFidelity.from_profile_name("b1_s2_w6_rho0_bf16")
    for session, reference, chunks in zip(sessions, references, (4, 6, 8), strict=True):
        for _ in range(chunks):
            expected = native.generate_next_blocks_raw([reference], [{"W": True}], fidelity=fidelity)
            driver.submit(session, {"W": True}, fidelity=fidelity)
            while driver.has_pending:
                outputs = driver.step()
            torch.testing.assert_close(outputs[0].decoded, expected.decoded, rtol=6e-4, atol=2e-4)
    expected = {}
    for session, reference in zip(sessions, references, strict=True):
        expected[session.session_id] = native.generate_next_blocks_raw([reference], [{}], fidelity=fidelity).decoded
        driver.submit(session, {}, fidelity=fidelity)
    driver.step((sessions[0].session_id,))
    assert len({task.current_start for task in driver._requests.values()}) == 3
    assert len({task.step_index for task in driver._requests.values()}) == 2
    attention = abot_world_dit.attention_fn
    self_attention_batches = []

    def record_attention(query, key, value, **kwargs):
        if key.shape[1] == 6 * 4:
            self_attention_batches.append(query.shape[0])
        return attention(query, key, value, **kwargs)

    monkeypatch.setattr(abot_world_dit, "attention_fn", record_attention)
    driver.step(tuple(session.session_id for session in sessions))
    assert self_attention_batches == [9] * stage.dit.num_layers
    completed = {}
    while driver.has_pending:
        completed.update((output.session_id, output.decoded) for output in driver.step())
    for session, reference in zip(sessions, references, strict=True):
        torch.testing.assert_close(completed[session.session_id], expected[session.session_id], rtol=6e-4, atol=2e-4)
        assert torch.equal(session.generator.get_state(), reference.generator.get_state())
    driver.close()


@torch.inference_mode()
def _exercise_runtime(world, placements=None):
    torch.set_num_threads(1)
    stage = _stage()
    native, pipeline = _pipelines(stage)
    references = list(native._interactive_sessions.values())
    sessions = list(pipeline._interactive_sessions.values())
    driver = pipeline.create_continuous_runtime(
        max_batch_size=max(6 if placements else 3, (9 + world - 1) // world), granularity="latent_step"
    )
    # Replays/captures must be selected uniformly even when some ranks have no
    # task rows or allocate their KV arena later. Simulate cache decisions on
    # CPU while still executing the actual eager model and collectives.
    cache, decisions, generation = OrderedDict(), [], None
    eager_run = driver.executor.run

    def record_run(*args, **kwargs):
        nonlocal generation
        if kwargs["generation"] != generation:
            generation = kwargs["generation"]
            cache.clear()
        key = kwargs["key"]
        decisions.append(key in cache)
        cache.pop(key, None)
        cache[key] = None
        if len(cache) > driver.executor.max_graphs:
            cache.popitem(last=False)
        return eager_run(*args, **kwargs)

    driver.executor.run = record_run
    saw_mixed_steps = saw_cross_session_batch = saw_local_sibling_route = False
    for chunk in range(7):
        reference = native.generate_next_blocks_raw(references, [{"W": True}, {"D": True}, {}])
        admitted, iteration, outputs = 0, 0, {}
        while admitted < 3 or driver.has_pending:
            # Startup rows now join retained rows immediately. Admit while the
            # oldest group is still denoising, including on an eight-rank pool.
            if admitted < 3 and iteration >= admitted:
                driver.submit(
                    sessions[admitted],
                    ({"W": True}, {"D": True}, {})[admitted],
                    owners=placements[admitted] if placements else None,
                )
                admitted += 1
            for output in driver.step():
                outputs[output.session_id] = output
            assert all(key.latent_index in {0, 1, 2} for key in driver.last_tasks)
            assert driver.last_batch_size <= driver.max_batch_size
            saw_mixed_steps |= len({key.step_index for key in driver.last_tasks}) > 1
            saw_cross_session_batch |= len({key.session_id for key in driver.last_local_tasks}) > 1
            saw_local_sibling_route |= driver.exchange.local_sibling_only
            iteration += 1
        if driver.is_output_rank:
            actual = torch.cat([outputs[session.session_id].decoded for session in sessions])
            torch.testing.assert_close(actual, reference.decoded, rtol=6e-4, atol=2e-4)
        for session, baseline in zip(sessions, references, strict=True):
            assert session.next_latent_frame == baseline.next_latent_frame
            assert session.emitted_frames == baseline.emitted_frames
            assert torch.equal(session.generator.get_state(), baseline.generator.get_state())
            assert session.self_cache[0]["k"].shape[0] == 0  # No redundant canonical GPU replica.
        if chunk == 0:
            with pytest.raises(RuntimeError, match="owned by a latent-step runtime"):
                pipeline.generate_next_blocks_raw([sessions[0]], [{}])
            if placements and world > 1:
                before = sessions[0].generator.get_state()
                with pytest.raises(ValueError, match="owners cannot change"):
                    driver.submit(sessions[0], owners=tuple((owner + 1) % world for owner in placements[0]))
                assert torch.equal(before, sessions[0].generator.get_state())
    assert saw_mixed_steps
    if world > 1:
        # The default placement intentionally keeps all siblings of one
        # session on one rank.  Small pools may still batch different sessions
        # locally, but with an eight-rank pool there is one session per rank;
        # the invariant we need to test is that the local K/V route is selected
        # consistently instead of forcing an all-rank collective.
        saw = torch.tensor(int(saw_local_sibling_route if placements is None else saw_cross_session_batch))
        dist.all_reduce(saw, op=dist.ReduceOp.MAX)
        assert saw.item()
        all_decisions = [None] * world
        dist.all_gather_object(all_decisions, decisions)
        assert all(trace == decisions for trace in all_decisions)
    for session, baseline in zip(sessions, references, strict=True):
        driver.release_session(session.session_id)
        for actual_cache, expected_cache in (
            (session.self_cache, baseline.self_cache),
            (session.cross_cache, baseline.cross_cache),
        ):
            for actual, expected in zip(actual_cache, expected_cache, strict=True):
                for name, value in expected.items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(actual[name], value, rtol=6e-4, atol=2e-4)
                    else:
                        assert actual[name] == value
    driver.close()
    # Materialized sessions can return to the unchanged whole-chunk pipeline.
    actual = pipeline.generate_next_blocks_raw(sessions, [{}] * 3)
    expected = native.generate_next_blocks_raw(references, [{}] * 3)
    torch.testing.assert_close(actual.decoded, expected.decoded, rtol=6e-4, atol=2e-4)


def test_whole_latent_step_runtime_single_rank():
    _exercise_runtime(1)


@torch.inference_mode()
def test_releasing_one_session_preserves_an_unselected_survivors_history():
    torch.set_num_threads(1)
    native, pipeline = _pipelines(_stage())
    references = list(native._interactive_sessions.values())
    sessions = list(pipeline._interactive_sessions.values())
    native.generate_next_blocks_raw(references, [{"W": True}, {"D": True}, {}])
    driver = pipeline.create_continuous_runtime(max_batch_size=9, granularity="latent_step")
    for session, controls in zip(sessions, ({"W": True}, {"D": True}, {}), strict=True):
        driver.submit(session, controls)
    while driver.has_pending:
        driver.step()
    driver.release_session(sessions[2].session_id, materialize=False)
    # Reuse the survivor's old arena positions while that survivor is not
    # selected. Its history must remain tracked and be preserved before reuse.
    for _ in range(2):
        native.generate_next_blocks_raw([references[1]], [{"S": True}])
        driver.submit(sessions[1], {"S": True})
        while driver.has_pending:
            driver.step((sessions[1].session_id,))
    expected = native.generate_next_blocks_raw([references[0]], [{"A": True}])
    driver.submit(sessions[0], {"A": True})
    outputs = []
    while driver.has_pending:
        outputs.extend(driver.step((sessions[0].session_id,)))
    torch.testing.assert_close(outputs[0].decoded, expected.decoded, rtol=6e-4, atol=2e-4)
    driver.close()


def _distributed_custom_owners(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=120))
    try:
        _exercise_runtime(4, placements=((0, 0, 0), (1, 2, 3), (3, 3, 3)))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU Gloo backend unavailable")
def test_custom_colocated_and_split_owners_preserve_four_rank_parity(tmp_path):
    mp.spawn(_distributed_custom_owners, args=(f"file://{tmp_path / 'owners'}",), nprocs=4, join=True)


@torch.inference_mode()
def test_explicit_waves_validate_capacity_and_do_not_advance_unselected_jobs():
    torch.set_num_threads(1)
    _, pipeline = _pipelines(_stage(), count=2)
    sessions = list(pipeline._interactive_sessions.values())
    driver = pipeline.create_continuous_runtime(max_batch_size=3, granularity="latent_step")
    for session in sessions:
        driver.submit(session)
    before = [session.generator.get_state() for session in sessions]
    with pytest.raises(ValueError, match="distinct complete"):
        driver.step((sessions[0].session_id,) * 2)
    with pytest.raises(ValueError, match="slot capacity"):
        driver.step(tuple(session.session_id for session in sessions))
    assert all(torch.equal(value, session.generator.get_state()) for value, session in zip(before, sessions))
    driver.step((sessions[1].session_id,))
    assert driver._requests[sessions[0].session_id].step_index == 0
    assert driver._requests[sessions[1].session_id].step_index == 1
    while driver.has_pending:
        driver.step()
    driver.close()


@torch.inference_mode()
def _exercise_mixed_windows(world):
    torch.set_num_threads(1)
    native, pipeline = _pipelines(_stage(), count=3)
    references = list(native._interactive_sessions.values())
    sessions = list(pipeline._interactive_sessions.values())
    driver = pipeline.create_continuous_runtime(
        max_batch_size=max(3, (9 + world - 1) // world), granularity="latent_step"
    )
    saw_mixed_windows = saw_mixed_steps = False
    for chunk in range(8):
        # Shrink, evict, and expand without resurrecting discarded history.
        names = ("s4_w18", "s2_w6", "s3_w12") if chunk < 4 else ("s2_w6", "s4_w18", "s4_w6")
        fidelities = [ABotWorldFidelity.from_profile_name(f"b1_{name}_rho0_bf16") for name in names]
        expected = []
        for baseline, fidelity in zip(references, fidelities):
            expected.append(native.generate_next_blocks_raw([baseline], [{"W": True}], fidelity=fidelity).decoded)
        driver.submit(sessions[0], {"W": True}, fidelity=fidelities[0])
        outputs, iteration = {}, 0
        while driver.has_pending or len(outputs) < 3:
            if iteration == 1:
                for row in (1, 2):
                    driver.submit(sessions[row], {"W": True}, fidelity=fidelities[row])
            for output in driver.step():
                outputs[output.session_id] = output
            if chunk == 0 and iteration == 1:
                # New prompt KV must not create a separate startup bucket.
                assert {key.session_id for key in driver.last_tasks} == {session.session_id for session in sessions}
            saw_mixed_windows |= len(set(driver.last_windows.values())) > 1
            saw_mixed_steps |= len({key.step_index for key in driver.last_tasks}) > 1
            iteration += 1
        if driver.is_output_rank:
            for row, session in enumerate(sessions):
                torch.testing.assert_close(
                    outputs[session.session_id].decoded,
                    expected[row],
                    rtol=8e-4,
                    atol=3e-4,
                    msg=lambda message: f"chunk={chunk} row={row} profile={names[row]}: {message}",
                )
        for session, baseline in zip(sessions, references):
            assert torch.equal(session.generator.get_state(), baseline.generator.get_state())
            assert session.next_latent_frame == baseline.next_latent_frame
    assert saw_mixed_windows and saw_mixed_steps
    driver.close()
    for session, baseline in zip(sessions, references):
        valid = int(baseline.self_cache[0]["local_end_index"].item())
        for actual, expected in zip(session.self_cache, baseline.self_cache):
            torch.testing.assert_close(actual["k"][:, :valid], expected["k"][:, :valid], rtol=8e-4, atol=3e-4)
            torch.testing.assert_close(actual["v"][:, :valid], expected["v"][:, :valid], rtol=8e-4, atol=3e-4)


def test_mixed_quality_steps_windows_and_window_changes_match_independent_execution():
    _exercise_mixed_windows(1)


def _distributed_mixed_windows(rank, world, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world, timeout=timedelta(seconds=120))
    try:
        _exercise_mixed_windows(world)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU Gloo backend unavailable")
def test_distributed_mixed_quality_windows_match_independent_execution(tmp_path):
    mp.spawn(_distributed_mixed_windows, args=(4, f"file://{tmp_path / 'mixed'}"), nprocs=4, join=True)


@torch.inference_mode()
def _exercise_joint_planner(world):
    from telefuser.pipelines.abot_world.latent_step_runtime import LatentStepKey
    from telefuser.service.livekit.abot_latent_step_policy import ABotLatentStepPolicy
    from telefuser.service.livekit.motivation_scheduler import (
        MotivationProfile,
        MotivationSchedulerConfig,
        SessionSchedulingState,
    )

    torch.set_num_threads(1)
    native, pipeline = _pipelines(_stage(), count=3)
    references = native._interactive_sessions
    sessions = pipeline._interactive_sessions
    driver = pipeline.create_continuous_runtime(max_batch_size=9, granularity="latent_step")
    profiles = [
        MotivationProfile(1, f"b1_s{steps}_w{window}_rho0_bf16", 1000, quality, 1)
        for steps, window, quality in ((2, 6, 0.9), (3, 12, 0.95), (4, 18, 1.0))
    ]
    policy = ABotLatentStepPolicy(profiles, MotivationSchedulerConfig())
    for window, seconds in ((6, 0.02), (12, 0.03), (18, 0.05)):
        for refreshing in (False, True):
            tasks = [LatentStepKey("warmup", 3, index, 1, refreshing, index % world) for index in range(3)]
            policy.observe(
                tasks,
                {"warmup": window},
                ranks=world,
                residence_seconds=seconds * max(sum(key.owner == rank for key in tasks) for rank in range(world)),
            )
    states = {
        key: SessionSchedulingState(key, "pool", -100.0 if index == 2 else 20.0, 1.0)
        for index, key in enumerate(sessions)
    }
    placements, decisions = {}, []
    for chunk in range(7):
        expected, qualities = {}, {}
        for state in states.values():
            state.submit_action(job_id=f"{state.session_id}-{chunk}", controls=("KeyW",), now=0.0, release=True)
        iterations = 0
        while driver.has_pending or any(state.ready_job(include_idle=False) is not None for state in states.values()):
            plan = policy.plan(list(states.values()), sessions, driver)
            assert plan.session_ids
            decisions.append(
                (
                    plan.session_ids,
                    [(key, choice.owners, choice.fidelity.name) for key, choice in plan.admissions.items()],
                )
            )
            for key, choice in plan.admissions.items():
                expected[key] = native.generate_next_blocks_raw(
                    [references[key]],
                    [{"W": True}],
                    fidelity=choice.fidelity,
                ).decoded
                before = sessions[key].generator.get_state()
                driver.submit(sessions[key], {"W": True}, fidelity=choice.fidelity, owners=choice.owners)
                assert not torch.equal(before, sessions[key].generator.get_state())
                assert placements.setdefault(key, choice.owners) == choice.owners
                qualities[key] = choice.quality
                states[key].mark_dispatched(states[key].ready_job(include_idle=False))
            for output in driver.step(plan.session_ids):
                if driver.is_output_rank:
                    torch.testing.assert_close(output.decoded, expected[output.session_id], rtol=8e-4, atol=3e-4)
                states[output.session_id].complete_job(
                    completed_at=0.0,
                    output_seconds=1.0,
                    quality=qualities[output.session_id],
                )
            iterations += 1
            assert iterations < 30
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, decisions)
        assert all(value == decisions for value in gathered)
    driver.close()
    for key, session in sessions.items():
        assert torch.equal(session.generator.get_state(), references[key].generator.get_state())
        valid = int(references[key].self_cache[0]["local_end_index"].item())
        for actual, expected in zip(session.self_cache, references[key].self_cache):
            torch.testing.assert_close(actual["k"][:, :valid], expected["k"][:, :valid], rtol=8e-4, atol=3e-4)


def test_joint_planner_runs_real_dit_without_changing_sampling_results():
    _exercise_joint_planner(1)


def _distributed_joint_planner(rank, world, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world, timeout=timedelta(seconds=120))
    try:
        _exercise_joint_planner(world)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU Gloo backend unavailable")
def test_four_rank_joint_planner_uses_identical_decisions_and_preserves_parity(tmp_path):
    mp.spawn(_distributed_joint_planner, args=(4, f"file://{tmp_path / 'planned'}"), nprocs=4, join=True)


@torch.inference_mode()
def _exercise_independent_release(world):
    torch.set_num_threads(1)
    native, pipeline = _pipelines(_stage(), count=2)
    references = list(native._interactive_sessions.values())
    sessions = list(pipeline._interactive_sessions.values())
    driver = pipeline.create_continuous_runtime(
        max_batch_size=max(3, (6 + world - 1) // world), granularity="latent_step"
    )
    expected = native.generate_next_blocks_raw(references, [{"W": True}, {"D": True}])
    driver.submit(sessions[0], {"W": True})
    driver.step()
    driver.step()
    driver.submit(sessions[1], {"D": True})
    with pytest.raises(RuntimeError, match="this session"):
        driver.release_session(sessions[1].session_id)
    outputs = []
    while not outputs:
        outputs = driver.step()
    assert [output.session_id for output in outputs] == [sessions[0].session_id]
    assert driver.has_pending
    with pytest.raises(RuntimeError, match="before closing"):
        driver.close()
    assert sessions[0].latent_runtime_owner is driver
    if driver.is_output_rank:
        torch.testing.assert_close(outputs[0].decoded, expected.decoded[:1], rtol=6e-4, atol=2e-4)
    remaining = [state for key, state in driver._local.items() if key[0] == sessions[1].session_id]
    pointers = [entry["k"].data_ptr() for state in remaining for entry in state.self_cache]
    driver.release_session(sessions[0].session_id)
    assert driver.has_pending
    assert pointers == [entry["k"].data_ptr() for state in remaining for entry in state.self_cache]
    assert sessions[0].latent_runtime_owner is None
    assert sessions[1].latent_runtime_owner is driver
    for actual, baseline in zip(sessions[0].self_cache, references[0].self_cache, strict=True):
        torch.testing.assert_close(actual["k"], baseline["k"], rtol=6e-4, atol=2e-4)
        torch.testing.assert_close(actual["v"], baseline["v"], rtol=6e-4, atol=2e-4)

    # A new chunk can re-enter while the other session is still executing.
    expected_next = native.generate_next_blocks_raw([references[0]], [{}])
    driver.submit(sessions[0], {})
    outputs = {}
    while driver.has_pending:
        outputs.update((output.session_id, output) for output in driver.step())
    if driver.is_output_rank:
        torch.testing.assert_close(outputs[sessions[0].session_id].decoded, expected_next.decoded, rtol=6e-4, atol=2e-4)
        torch.testing.assert_close(outputs[sessions[1].session_id].decoded, expected.decoded[1:], rtol=6e-4, atol=2e-4)
    driver.close()
    for session, baseline in zip(sessions, references, strict=True):
        assert session.next_latent_frame == baseline.next_latent_frame
        assert session.emitted_frames == baseline.emitted_frames
        assert torch.equal(session.generator.get_state(), baseline.generator.get_state())
        for actual, baseline_cache in zip(session.self_cache, baseline.self_cache, strict=True):
            torch.testing.assert_close(actual["k"], baseline_cache["k"], rtol=6e-4, atol=2e-4)
            torch.testing.assert_close(actual["v"], baseline_cache["v"], rtol=6e-4, atol=2e-4)


def test_completed_session_can_leave_and_reenter_without_draining_other_work():
    _exercise_independent_release(1)


def _distributed_check(rank, world, rendezvous, independent_release=False, runtime_created=False):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world, timeout=timedelta(seconds=120))
    try:
        if runtime_created:
            _exercise_runtime_created_session()
        elif independent_release:
            _exercise_independent_release(world)
        else:
            _exercise_runtime(world)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU Gloo backend unavailable")
def test_whole_latent_step_runtime_arbitrary_rank_count(tmp_path: Path, world: int):
    mp.spawn(_distributed_check, args=(world, f"file://{tmp_path / 'rendezvous'}"), nprocs=world, join=True)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU Gloo backend unavailable")
def test_distributed_session_release_does_not_drain_other_work(tmp_path: Path):
    mp.spawn(_distributed_check, args=(4, f"file://{tmp_path / 'rendezvous'}", True), nprocs=4, join=True)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU Gloo backend unavailable")
def test_distributed_runtime_created_session_matches_native_generation(tmp_path: Path):
    mp.spawn(_distributed_check, args=(4, f"file://{tmp_path / 'rendezvous'}", False, True), nprocs=4, join=True)


def test_slot_capacity_and_backend_validation_do_not_advance_session_rng():
    _, pipeline = _pipelines(_stage(), count=1)
    session = pipeline._interactive_sessions["session-0"]
    driver = pipeline.create_continuous_runtime(max_batch_size=2, granularity="latent_step")
    before = session.generator.get_state()
    with pytest.raises(ValueError, match="slots cannot fit"):
        driver.submit(session)
    assert torch.equal(before, session.generator.get_state())
    with pytest.raises(ValueError, match="backend requires"):
        pipeline.create_continuous_runtime(max_batch_size=3, backend="cuda_graph")


@torch.inference_mode()
def test_discarded_session_closes_without_materializing_its_kv(monkeypatch):
    from telefuser.pipelines.abot_world import latent_step_execution

    torch.set_num_threads(1)
    _, pipeline = _pipelines(_stage(), count=1)
    session = pipeline._interactive_sessions["session-0"]
    driver = pipeline.create_continuous_runtime(max_batch_size=3, granularity="latent_step")
    driver.submit(session)
    while driver.has_pending:
        driver.step()

    def reject_copy(_cache):
        raise AssertionError("Discarding state must not allocate another full KV copy")

    monkeypatch.setattr(latent_step_execution, "clone_cache", reject_copy)
    driver.release_session(session.session_id, materialize=False)
    assert session.closed
    assert session.latent_runtime_owner is None
    driver.close(materialize=False)


@torch.inference_mode()
def test_short_initial_chunk_primes_future_sibling_latents():
    torch.set_num_threads(1)
    stage = _stage()
    native, pipeline = _pipelines(stage, count=1)
    reference_session = native._interactive_sessions["session-0"]
    session = pipeline._interactive_sessions["session-0"]
    driver = pipeline.create_continuous_runtime(max_batch_size=3, granularity="latent_step")
    for frames in (1, 3, 2, 3):
        expected = native.generate_next_blocks_raw([reference_session], [{"W": True}], control_latent_frames=frames)
        driver.submit(session, {"W": True}, control_latent_frames=frames)
        outputs = []
        while driver.has_pending:
            outputs.extend(driver.step())
        torch.testing.assert_close(outputs[0].decoded, expected.decoded, rtol=6e-4, atol=2e-4)
        assert torch.equal(session.generator.get_state(), reference_session.generator.get_state())
    driver.release_session(session.session_id)
    for actual, expected in zip(session.self_cache, reference_session.self_cache, strict=True):
        torch.testing.assert_close(actual["k"], expected["k"], rtol=6e-4, atol=2e-4)
        torch.testing.assert_close(actual["v"], expected["v"], rtol=6e-4, atol=2e-4)
    driver.close()
