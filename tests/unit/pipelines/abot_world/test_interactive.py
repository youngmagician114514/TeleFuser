from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from telefuser.models.wan22_video_vae import Wan22VideoVAEStreamingDecodeState
from telefuser.pipelines.abot_world.interactive import (
    ABotWorldInteractivePipeline,
    ABotWorldInteractiveSession,
    ABotWorldSessionLifecycle,
)


def _session(session_id: str) -> ABotWorldInteractiveSession:
    return ABotWorldInteractiveSession(
        session_id=session_id,
        prompt_emb=torch.ones(1),
        first_frame_latent=torch.ones(1),
        self_cache=[{"k": torch.ones(1)}],
        cross_cache=[{"k": torch.ones(1)}],
        scheduler=object(),
        generator=torch.Generator(device="cpu"),
        vae_decode_state=Wan22VideoVAEStreamingDecodeState(feat_cache=[torch.ones(1)]),
    )


@torch.inference_mode()
def test_repeated_conditioning_reuses_encoders_but_keeps_session_state_independent():
    pipeline = ABotWorldInteractivePipeline(device="cpu", torch_dtype=torch.float32)
    pipeline.config = SimpleNamespace(height=8, width=8)
    pipeline.preload_models = lambda: None
    pipeline.preprocess_image = lambda image, height, width: torch.full((3, height, width), image.getpixel((0, 0))[0])
    images, prompts = [], []

    def encode_image(method, pixels, *args, **kwargs):
        images.append(pixels.clone())
        return torch.full((4, 1, 4, 4), float(pixels[0, 0, 0])), None

    def encode_text(text):
        prompts.extend(text)
        return [torch.full((1, 4, 16), float(len(text[0])))]

    pipeline.vae_stage = SimpleNamespace(process=encode_image)
    pipeline.text_encoding_stage = SimpleNamespace(process=encode_text)
    pipeline.denoise_stage = SimpleNamespace(
        latent_parallel_runtime=None,
        _new_cache=lambda *args: ([{"k": torch.zeros(1, 18, 1, 4)}], [{"k": torch.zeros(1, 4, 1, 4)}]),
        _scheduler=object,
    )
    pipeline.taew_decode_stage = SimpleNamespace(create_decode_state=object, warmup_first_frame=lambda *args: None)
    image = Image.new("RGB", (8, 8), (7, 0, 0))
    first = pipeline.create_interactive_session(image, "world", seed=1)
    second = pipeline.create_interactive_session(image, "world", seed=2)
    assert len(images) == len(prompts) == 1
    assert first.generator.initial_seed() == 1 and second.generator.initial_seed() == 2
    assert first.scheduler is not second.scheduler
    assert first.taew_decode_state is not second.taew_decode_state
    assert first.self_cache[0]["k"].data_ptr() != second.self_cache[0]["k"].data_ptr()
    first.prompt_emb.zero_()
    first.first_frame_latent.zero_()
    third = pipeline.create_interactive_session(image, "world")
    torch.testing.assert_close(third.prompt_emb, second.prompt_emb)
    torch.testing.assert_close(third.first_frame_latent, second.first_frame_latent)
    image.putpixel((0, 0), (9, 0, 0))
    pipeline.create_interactive_session(image, "world")
    assert len(images) == 2 and len(prompts) == 1
    pipeline.create_interactive_session(image, "different")
    assert len(images) == len(prompts) == 2


def test_window_shrink_retains_valid_tail_in_partially_filled_cache() -> None:
    key = torch.arange(18, dtype=torch.float32).reshape(1, 18, 1, 1)
    key[:, 9:] = -100
    cache = [{"k": key, "v": key + 1, "local_end_index": torch.tensor(9)}]
    ABotWorldInteractivePipeline._resize_self_cache(
        cache,
        frame_tokens=1,
        old_window=18,
        old_sink=6,
        new_window=6,
        new_sink=2,
    )
    torch.testing.assert_close(cache[0]["k"].flatten(), torch.tensor([0.0, 1.0, 5.0, 6.0, 7.0, 8.0]))
    torch.testing.assert_close(cache[0]["v"].flatten(), torch.tensor([1.0, 2.0, 6.0, 7.0, 8.0, 9.0]))
    assert cache[0]["local_end_index"].item() == 6


def test_close_interactive_session_clears_only_target_state() -> None:
    pipeline = ABotWorldInteractivePipeline(device="cpu")
    pipeline.vae_stage = SimpleNamespace(vae=SimpleNamespace(_feat_cache=[torch.ones(1)], _feat_idx=[3]))
    first = _session("first")
    second = _session("second")
    pipeline._interactive_sessions = {"first": first, "second": second}

    pipeline.close_interactive_session(first)

    assert first.closed
    assert first.lifecycle == ABotWorldSessionLifecycle.CLOSED
    assert first.self_cache == []
    assert first.cross_cache == []
    assert first.vae_decode_state.feat_cache == []
    assert second.self_cache
    assert second.cross_cache
    assert second.vae_decode_state.feat_cache
    assert pipeline._interactive_sessions == {"second": second}
    # Session cleanup must not mutate the legacy model-owned cache.
    assert pipeline.vae_stage.vae._feat_idx == [3]


def test_cache_collation_and_scatter_preserve_session_isolation() -> None:
    first = _session("first")
    second = _session("second")
    first.self_cache = [
        {
            "k": torch.tensor([[[[1.0]]]]),
            "v": torch.tensor([[[[2.0]]]]),
            "global_end_index": torch.tensor([3]),
            "local_end_index": torch.tensor([3]),
        }
    ]
    second.self_cache = [
        {
            "k": torch.tensor([[[[4.0]]]]),
            "v": torch.tensor([[[[5.0]]]]),
            "global_end_index": torch.tensor([12]),
            "local_end_index": torch.tensor([3]),
        }
    ]

    collated = ABotWorldInteractivePipeline._collate_caches([first, second], "self_cache")
    assert collated[0]["k"].shape[0] == 2
    collated[0]["k"].add_(10)
    ABotWorldInteractivePipeline._scatter_caches([first, second], "self_cache", collated)

    assert first.self_cache[0]["k"].item() == 11
    assert second.self_cache[0]["k"].item() == 14
    first.self_cache[0]["k"].zero_()
    assert second.self_cache[0]["k"].item() == 14


class _DecodeMetricsStage:
    def __init__(self) -> None:
        self.states: list[object] = []

    def decode_chunks(self, latents: torch.Tensor, states: list[object]) -> torch.Tensor:
        self.states = list(states)
        return latents

    @staticmethod
    def last_decode_metrics() -> dict[str, int]:
        return {
            "taew_decode_items": 2,
            "taew_decode_batch_size": 1,
            "taew_decode_invocations": 2,
            "taew_decode_mode": 2,
        }


def test_generate_next_blocks_surfaces_effective_taew_decode_metrics() -> None:
    pipeline = ABotWorldInteractivePipeline(device="cpu", torch_dtype=torch.float32)
    pipeline.config = SimpleNamespace(height=32, width=32)

    def denoise(noise: torch.Tensor, *_: object) -> torch.Tensor:
        return noise

    pipeline.denoise_stage = SimpleNamespace(
        dit=SimpleNamespace(use_relative_rope=True),
        _denoise_block=denoise,
    )
    decode_stage = _DecodeMetricsStage()
    pipeline.taew_decode_stage = decode_stage
    pipeline.tensor2video = lambda decoded: [object() for _ in range(decoded.shape[1])]

    def make_session(session_id: str) -> ABotWorldInteractiveSession:
        return ABotWorldInteractiveSession(
            session_id=session_id,
            prompt_emb=torch.ones(1, 1),
            first_frame_latent=torch.ones(1, 1, 1, 1, 1),
            self_cache=[],
            cross_cache=[],
            scheduler=object(),
            generator=torch.Generator(device="cpu").manual_seed(3),
            taew_decode_state=object(),
        )

    first = make_session("first")
    second = make_session("second")
    decoder_tail_waits: list[bool] = []
    readiness = SimpleNamespace(complete=False)

    def wait_complete() -> None:
        decoder_tail_waits.append(True)
        readiness.complete = True

    readiness.wait_complete = wait_complete
    first.migration_layer_readiness = readiness
    pipeline._interactive_sessions = {"first": first, "second": second}

    output = pipeline.generate_next_blocks([first, second], [{"W": True}, {"D": True}])

    assert [len(frames) for frames in output] == [3, 3]
    assert decode_stage.states == [first.taew_decode_state, second.taew_decode_state]
    assert pipeline.last_stage_metrics()["batch_size"] == 2
    assert pipeline.last_stage_metrics()["taew_decode_items"] == 2
    assert pipeline.last_stage_metrics()["taew_decode_batch_size"] == 1
    assert pipeline.last_stage_metrics()["taew_decode_invocations"] == 2
    assert pipeline.last_stage_metrics()["taew_decode_mode"] == 2
    assert decoder_tail_waits == [True]


@pytest.mark.parametrize("output_rank", [True, False])
@pytest.mark.parametrize("raw_output", [True, False])
def test_latent_parallel_output_owner_and_replica_counters(output_rank: bool, raw_output: bool) -> None:
    pipeline = ABotWorldInteractivePipeline(device="cpu", torch_dtype=torch.float32)
    pipeline.config = SimpleNamespace(height=32, width=32)
    broadcasts: list[int] = []

    def broadcast(value: torch.Tensor) -> None:
        broadcasts.append(value.item())
        value.fill_(3)  # The output owner's decoded frame count.

    pipeline.denoise_stage = SimpleNamespace(
        dit=SimpleNamespace(use_relative_rope=True),
        latent_parallel_runtime=SimpleNamespace(world_size=3, is_output_rank=output_rank, broadcast_=broadcast),
        _denoise_block=lambda noise, *args: noise,
    )
    decode_stage = _DecodeMetricsStage()
    pipeline.taew_decode_stage = decode_stage
    pipeline.tensor2video = lambda decoded: [object() for _ in range(decoded.shape[1])]
    sessions = [
        ABotWorldInteractiveSession(
            session_id=name,
            prompt_emb=torch.ones(1, 1),
            first_frame_latent=torch.ones(1, 1, 1, 1, 1),
            self_cache=[],
            cross_cache=[],
            scheduler=object(),
            generator=torch.Generator().manual_seed(3),
            taew_decode_state=object(),
        )
        for name in ("first", "second")
    ]
    pipeline._interactive_sessions = {session.session_id: session for session in sessions}
    generate = pipeline.generate_next_blocks_raw if raw_output else pipeline.generate_next_blocks
    result = generate(sessions, [{"W": True}, {"D": True}])

    assert [session.next_latent_frame for session in sessions] == [3, 3]
    assert [session.emitted_frames for session in sessions] == [3, 3]
    assert broadcasts == [3 if output_rank else 0]
    assert len(decode_stage.states) == (2 if output_rank else 0)
    if raw_output:
        assert result.frames_per_session == (3 if output_rank else 0)
        assert result.decoded.shape[2] == (3 if output_rank else 0)
    else:
        assert [len(frames) for frames in result] == ([3, 3] if output_rank else [0, 0])
