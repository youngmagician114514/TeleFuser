from __future__ import annotations

import pytest
import torch

from telefuser.service.livekit.config import LiveKitServeConfig
from telefuser.service.livekit.nccl_transfer import (
    LayerTransferProgress,
    TensorTransferGroup,
    build_layer_transfer_groups,
    flatten_tensor_tree,
    rebuild_tensor_tree,
    transfer_tensor_leaves_nccl_streamed,
)


def test_tensor_manifest_round_trip_preserves_nested_structure() -> None:
    source = {
        "cache": [{"k": torch.ones((1, 2)), "cursor": 3}],
        "latent": torch.zeros((1, 3)),
        "flags": (True, None),
    }
    skeleton, manifest, leaves = flatten_tensor_tree(source)

    restored = rebuild_tensor_tree(skeleton, leaves)

    assert len(manifest) == 2
    assert restored["cache"][0]["cursor"] == 3
    assert restored["flags"] == (True, None)
    assert torch.equal(restored["cache"][0]["k"], source["cache"][0]["k"])


def test_process_nccl_requires_fixed_two_gpu_group() -> None:
    config = LiveKitServeConfig(
        worker_mode="process-nccl",
        num_workers=2,
        worker_gpu_map="0;1",
    )

    assert config.worker_mode == "process-nccl"

    autoscaling = LiveKitServeConfig(
        worker_mode="process-nccl",
        num_workers=2,
        worker_gpu_map="0;1",
        autoscaling_enabled=True,
        queue_size=1,
    )
    assert autoscaling.autoscaling_enabled is True


def test_layer_transfer_groups_prioritize_preamble_then_cache_layers() -> None:
    source = {
        "prompt_emb": torch.zeros((1, 2)),
        "self_cache": [
            {"k": torch.zeros((1, 3)), "v": torch.zeros((1, 3))},
            {"k": torch.zeros((1, 4)), "v": torch.zeros((1, 4))},
        ],
        "cross_cache": [
            {"k": torch.zeros((1, 5)), "v": torch.zeros((1, 5))},
            {"k": torch.zeros((1, 6)), "v": torch.zeros((1, 6))},
        ],
        "vae_feat_cache": [torch.zeros((1,))],
        "taew_decode_state": {"history": torch.zeros((1,))},
    }
    _, manifest, leaves = flatten_tensor_tree(source)

    groups = build_layer_transfer_groups(manifest)

    assert [group.name for group in groups] == ["preamble", "layer_00", "layer_01", "decoder_tail"]
    assert groups[0].layer_index is None
    assert groups[1].layer_index == 0
    assert set(groups[1].paths) == {
        ("self_cache", 0, "k"),
        ("self_cache", 0, "v"),
        ("cross_cache", 0, "k"),
        ("cross_cache", 0, "v"),
    }
    assert set(groups[-1].paths) == {
        ("vae_feat_cache", 0),
        ("taew_decode_state", "history"),
    }
    assert {path for group in groups for path in group.paths} == set(leaves)


def test_streamed_transfer_rejects_duplicate_or_missing_paths(monkeypatch) -> None:
    path = ("self_cache", 0, "k")
    leaves = {path: torch.zeros((1, 3))}
    duplicate = TensorTransferGroup("layer_00", (path, path), layer_index=0)
    monkeypatch.setattr("telefuser.service.livekit.nccl_transfer.dist.is_available", lambda: True)
    monkeypatch.setattr("telefuser.service.livekit.nccl_transfer.dist.is_initialized", lambda: True)

    with pytest.raises(ValueError, match="exactly once"):
        transfer_tensor_leaves_nccl_streamed(
            leaves,
            peer_rank=1,
            send=True,
            groups=(duplicate,),
        )


def test_streamed_transfer_completes_groups_in_manifest_order(monkeypatch) -> None:
    source = {
        "prompt_emb": torch.zeros((1, 2)),
        "self_cache": [{"k": torch.zeros((1, 3)), "v": torch.zeros((1, 3))}],
        "cross_cache": [{"k": torch.zeros((1, 4)), "v": torch.zeros((1, 4))}],
    }
    _, manifest, leaves = flatten_tensor_tree(source)
    groups = build_layer_transfer_groups(manifest)
    submitted_sizes = []
    completed = []

    class _Request:
        def wait(self) -> None:
            return None

    monkeypatch.setattr("telefuser.service.livekit.nccl_transfer.dist.is_available", lambda: True)
    monkeypatch.setattr("telefuser.service.livekit.nccl_transfer.dist.is_initialized", lambda: True)
    monkeypatch.setattr(
        "telefuser.service.livekit.nccl_transfer.dist.P2POp",
        lambda operation, tensor, peer_rank: (operation, tensor, peer_rank),
    )

    def batch(ops):
        submitted_sizes.append(len(ops))
        return [_Request() for _ in ops]

    monkeypatch.setattr("telefuser.service.livekit.nccl_transfer.dist.batch_isend_irecv", batch)

    report = transfer_tensor_leaves_nccl_streamed(
        leaves,
        peer_rank=1,
        send=True,
        groups=groups,
        on_group_complete=lambda group, event: completed.append((group.name, event)),
    )

    assert submitted_sizes == [1, 4]
    assert completed == [("preamble", None), ("layer_00", None)]
    assert report.total_bytes == sum(tensor.numel() * tensor.element_size() for tensor in leaves.values())
    assert [group.name for group in report.groups] == ["preamble", "layer_00"]


def test_layer_transfer_progress_tracks_first_ready_and_completion() -> None:
    now = [10.0]
    progress = LayerTransferProgress(2, clock=lambda: now[0])

    now[0] = 10.02
    progress.mark_layer_ready(0, None)
    now[0] = 10.03
    assert progress.wait_layer(0) == pytest.approx(0.0)
    now[0] = 10.05
    progress.mark_layer_ready(1, None)
    progress.mark_complete()
    assert progress.wait_complete() == pytest.approx(0.0)

    snapshot = progress.snapshot()
    assert snapshot["ready_layers"] == 2
    assert snapshot["complete"] is True
    assert snapshot["first_layer_ready_ms"] == pytest.approx(20.0)
    assert snapshot["transfer_complete_ms"] == pytest.approx(50.0)
    assert snapshot["wait_calls"] == 1
    assert snapshot["complete_wait_calls"] == 1


def test_layer_transfer_progress_propagates_transfer_failure() -> None:
    progress = LayerTransferProgress(1)
    progress.mark_failed(RuntimeError("NCCL failed"))

    with pytest.raises(RuntimeError, match="layer transfer failed"):
        progress.wait_layer(0)
    with pytest.raises(RuntimeError, match="decoder state"):
        progress.wait_complete()
    assert progress.snapshot()["failed"] is True
