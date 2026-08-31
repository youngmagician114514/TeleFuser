from __future__ import annotations

import pytest
import torch

from telefuser.pipelines.abot_world.fidelity import ABotWorldFidelity
from telefuser.pipelines.abot_world.interactive import ABotWorldInteractivePipeline


@pytest.mark.parametrize(
    ("name", "positions", "window", "sink"),
    [
        ("b1_s4_w18_rho0_bf16", (0, 1, 2, 3), 18, 6),
        ("b2_s3_w12_rho0_bf16", (0, 2, 3), 12, 4),
        ("b4_s2_w6_rho0_bf16", (0, 3), 6, 2),
    ],
)
def test_profile_name_maps_to_runtime_fidelity(name: str, positions: tuple[int, ...], window: int, sink: int) -> None:
    fidelity = ABotWorldFidelity.from_profile_name(name)

    assert fidelity.denoise_step_positions == positions
    assert fidelity.local_attn_size == window
    assert fidelity.sink_size == sink


def test_profile_parser_rejects_unsupported_attention_or_precision() -> None:
    with pytest.raises(ValueError, match="dense rho=0 BF16"):
        ABotWorldFidelity.from_profile_name("b1_s4_w18_rho0.5_bf16")
    with pytest.raises(ValueError, match="dense rho=0 BF16"):
        ABotWorldFidelity.from_profile_name("b1_s4_w18_rho0_fp16")


def test_resize_self_cache_preserves_sink_and_newest_tail() -> None:
    cache = [
        {
            "k": torch.arange(18, dtype=torch.float32).view(1, 18, 1, 1),
            "v": torch.arange(18, dtype=torch.float32).view(1, 18, 1, 1) + 100,
            "local_end_index": torch.tensor([18]),
            "global_end_index": torch.tensor([18]),
        }
    ]

    ABotWorldInteractivePipeline._resize_self_cache(
        cache, frame_tokens=1, old_window=18, old_sink=6, new_window=6, new_sink=2
    )

    assert cache[0]["k"].shape[1] == 6
    assert cache[0]["k"][0, :, 0, 0].tolist() == [0, 1, 14, 15, 16, 17]
    assert cache[0]["v"][0, :, 0, 0].tolist() == [100, 101, 114, 115, 116, 117]
    assert cache[0]["local_end_index"].item() == 6
    assert cache[0]["global_end_index"].item() == 18
