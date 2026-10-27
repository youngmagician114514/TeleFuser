"""Compare step-resumable execution with the native causal sampler."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from telefuser.pipelines.abot_world.interactive import ABotWorldInteractivePipeline, ABotWorldInteractiveSession
from telefuser.pipelines.abot_world.step_batching import ABotWorldStepBatcher

from .test_latent_parallel_runtime import _stage


def _clone_cache(cache):
    return [
        {name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in entry.items()}
        for entry in cache
    ]


def _prepared_task(stage, batcher, name, *, start=0, frames=3, positions=None):
    context = torch.randn(1, 4, 16)
    generator = torch.Generator().manual_seed(74 if name == "first" else 17)
    self_cache, cross_cache = stage._new_cache(1, 4, 4)
    for offset in range(0, start, 3):
        stage._denoise_block(
            torch.randn(1, 4, 3, 4, 4),
            context,
            torch.randn(1, 32, 3, 8, 8),
            None,
            self_cache,
            cross_cache,
            offset,
            generator,
            stage._scheduler(),
        )
    reference_self, reference_cross = _clone_cache(self_cache), _clone_cache(cross_cache)
    reference_rng = torch.Generator().set_state(generator.get_state())
    noise = torch.randn(1, 4, frames, 4, 4)
    action = torch.randn(1, 32, frames, 8, 8)
    first = torch.randn(1, 4, 1, 4, 4) if start == 0 else None
    reference = stage._denoise_block(
        noise.clone(),
        context,
        action,
        first,
        reference_self,
        reference_cross,
        start,
        reference_rng,
        stage._scheduler(),
        timestep_positions=positions,
    )
    task = batcher.begin(
        name,
        noise,
        context,
        action,
        first,
        self_cache,
        cross_cache,
        start,
        generator,
        stage._scheduler(),
        timestep_positions=positions,
    )
    return task, (reference, reference_self, reference_cross, reference_rng)


def _assert_result(task, expected):
    output, self_cache, cross_cache, generator = expected
    assert task.complete
    torch.testing.assert_close(task.latent, output, rtol=5e-4, atol=1e-4)
    assert torch.equal(task.generator.get_state(), generator.get_state())
    for actual_cache, reference_cache in ((task.self_cache, self_cache), (task.cross_cache, cross_cache)):
        for actual, reference in zip(actual_cache, reference_cache, strict=True):
            for name, value in reference.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(actual[name], value, rtol=5e-4, atol=1e-4)
                else:
                    assert actual[name] == value


def _exercise_late_admission(stage):
    batcher = ABotWorldStepBatcher(stage, max_batch_size=2)
    first, ref_first = _prepared_task(stage, batcher, "first", start=3)
    second, ref_second = _prepared_task(stage, batcher, "second", start=9, positions=(0, 2))
    batcher.submit(first)
    assert not batcher.step()
    assert not batcher.step()
    assert first.step_index == 2
    batcher.submit(second)
    completed, batches = [], []
    while batcher.has_pending:
        completed.extend(batcher.step())
        batches.append(batcher.last_step_indices)
    assert (2, 1) in batches  # Different diffusion steps share a physical forward.
    assert set(completed) == {first, second}
    _assert_result(first, ref_first)
    _assert_result(second, ref_second)


@torch.inference_mode()
def test_late_admission_mixed_steps_profiles_and_global_positions():
    torch.set_num_threads(1)
    _exercise_late_admission(_stage())


@pytest.mark.parametrize("frames", [1, 3])
@torch.inference_mode()
def test_stable_cohort_reuses_kv_and_preserves_conditioning(frames):
    torch.set_num_threads(1)
    stage = _stage()
    batcher = ABotWorldStepBatcher(stage, max_batch_size=2)
    tasks, expected = zip(*[_prepared_task(stage, batcher, name, frames=frames) for name in ("first", "second")])
    for task in tasks:
        batcher.submit(task)
    pointer = None
    completed = []
    while batcher.has_pending:
        completed.extend(batcher.step())
        if batcher.has_pending:
            current_pointer = batcher._self_cache[0]["k"].data_ptr()
            if pointer is not None:
                assert current_pointer == pointer
            pointer = current_pointer
    assert len(completed) == 2
    for task, reference in zip(tasks, expected, strict=True):
        _assert_result(task, reference)


@torch.inference_mode()
def test_step_batcher_rejects_duplicate_session_work():
    stage = _stage()
    batcher = ABotWorldStepBatcher(stage, max_batch_size=1)
    task, _ = _prepared_task(stage, batcher, "first")
    batcher.submit(task)
    with pytest.raises(ValueError, match="already has pending"):
        batcher.submit(task)


def _exercise_interactive_driver(stage):
    stage.dit.set_causal_attention_window(18, 6)
    pipelines = [ABotWorldInteractivePipeline(device="cpu", torch_dtype=torch.float32) for _ in range(2)]
    first_frames = [torch.randn(1, 4, 1, 4, 4) for _ in range(2)]
    prompts = [torch.randn(1, 4, 16) for _ in range(2)]
    for pipeline in pipelines:
        pipeline.config = SimpleNamespace(height=8, width=8)
        pipeline.denoise_stage = stage
        pipeline.taew_decode_stage = SimpleNamespace(
            decode_chunks=lambda latents, states: latents[:, :3].repeat_interleave(4, dim=2),
            last_decode_metrics=lambda: {},
        )
        for row, name in enumerate(("first", "second")):
            self_cache, cross_cache = stage._new_cache(1, 4, 4)
            pipeline._interactive_sessions[name] = ABotWorldInteractiveSession(
                session_id=name,
                prompt_emb=prompts[row].clone(),
                first_frame_latent=first_frames[row].clone(),
                self_cache=self_cache,
                cross_cache=cross_cache,
                scheduler=stage._scheduler(),
                generator=torch.Generator().manual_seed(row + 21),
                taew_decode_state=object(),
            )
    native, continuous = pipelines
    reference_sessions = list(native._interactive_sessions.values())
    sessions = list(continuous._interactive_sessions.values())
    driver = continuous.create_continuous_runtime(max_batch_size=2)
    for _ in range(2):
        reference = native.generate_next_blocks_raw(reference_sessions, [{"W": True}, {"D": True}])
        driver.submit(sessions[0], {"W": True})
        assert not driver.step()
        assert not driver.step()
        with pytest.raises(RuntimeError, match="unfinished continuous-batching"):
            continuous.snapshot_interactive_session(sessions[0])
        driver.submit(sessions[1], {"D": True})
        outputs = {}
        while driver.has_pending:
            for output in driver.step():
                outputs[output.session_id] = output
        actual = torch.cat([outputs[session.session_id].decoded for session in sessions])
        torch.testing.assert_close(actual, reference.decoded, rtol=5e-4, atol=1e-4)
        for actual_session, ref_session in zip(sessions, reference_sessions, strict=True):
            assert actual_session.next_latent_frame == ref_session.next_latent_frame
            assert actual_session.emitted_frames == ref_session.emitted_frames
            assert actual_session.step_task_active is False
            assert torch.equal(actual_session.generator.get_state(), ref_session.generator.get_state())


@torch.inference_mode()
def test_interactive_continuous_driver_matches_chunk_path_across_two_chunks():
    torch.set_num_threads(1)
    _exercise_interactive_driver(_stage())
