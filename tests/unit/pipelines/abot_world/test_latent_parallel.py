import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_MODULE_PATH = Path(__file__).parents[4] / "telefuser" / "pipelines" / "abot_world" / "latent_parallel.py"
_SPEC = importlib.util.spec_from_file_location("telefuser_abot_latent_parallel_test_module", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
LatentParallelCommunicator = _MODULE.LatentParallelCommunicator
LatentParallelConfig = _MODULE.LatentParallelConfig
latent_parallel_projection = _MODULE.latent_parallel_projection


def test_three_way_shape_and_communication_contract():
    config = LatentParallelConfig()
    config.validate()

    assert config.frames_per_rank == 1
    assert config.tokens_per_rank == 390
    per_layer = config.communication_bytes_per_layer()
    assert per_layer["current_kv_peer_bytes"] == 4_792_320 * 2
    assert per_layer["output_peer_bytes"] == 2_396_160 * 2
    assert per_layer["total_peer_bytes"] == 14_376_960

    projection = latent_parallel_projection(config, layers=30, steps=4)
    assert projection["peer_bytes_per_chunk"] == 1_725_235_200
    assert projection["peer_gb_per_chunk"] == pytest.approx(1.7252352)


def test_frame_aligned_sharding_is_contiguous_without_distributed_runtime():
    config = LatentParallelConfig()
    communicator = LatentParallelCommunicator(config, rank=1)
    tokens = torch.arange(3 * 1170 * 2, dtype=torch.float32).reshape(3, 1170, 2)

    local = communicator.shard_tokens(tokens)

    assert local.shape == (3, 390, 2)
    torch.testing.assert_close(local, tokens[:, 390:780])


def test_invalid_shape_contracts_fail_early():
    with pytest.raises(ValueError, match="divisible"):
        LatentParallelConfig(world_size=2).validate()

    communicator = LatentParallelCommunicator(LatentParallelConfig(), rank=0)
    with pytest.raises(ValueError, match="expected 1170 tokens"):
        communicator.shard_tokens(torch.empty(1, 1169, 3072))
