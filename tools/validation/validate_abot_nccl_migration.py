"""Validate one ABot session migration across two GPU processes with NCCL.

This is intentionally model-service level: each rank owns an independent ABot
replica, rank 0 generates one causal chunk, and rank 1 imports the session as
soon as layer 0 arrives while the residual cache continues over a dedicated
CUDA/NCCL stream.  The target then runs the next causal chunk with per-layer
readiness waits.  No CPU tensor snapshot is used for model state.
"""

from __future__ import annotations

import argparse
import importlib.util
import socket
import threading
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from telefuser.pipelines.abot_world.interactive import ABotWorldInteractivePipeline
from telefuser.pipelines.abot_world.service import ABotWorldLiveKitService
from telefuser.service.livekit.nccl_transfer import (
    LayerTransferProgress,
    allocate_tensor_tree_leaves,
    build_layer_transfer_groups,
    transfer_tensor_leaves_nccl_streamed,
)


def _loader_module() -> Any:
    loader_path = Path(__file__).resolve().parents[2] / "examples/abot_world/_loader.py"
    spec = importlib.util.spec_from_file_location("abot_nccl_validation_loader", loader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load ABot loader: {loader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _take_output(service: ABotWorldLiveKitService, session_id: str, timeout: float = 120.0) -> dict:
    state = service._session(session_id)  # Validation intentionally observes the service output boundary.
    if state is None:
        raise KeyError(session_id)
    return state.output_queue.get(timeout=timeout)


def _drain_outputs(service: ABotWorldLiveKitService, session_id: str, stop: threading.Event) -> None:
    """Mirror the parent LiveKit transport's continuous model-output pull."""
    state = service._session(session_id)
    if state is None:
        return
    while not stop.is_set():
        try:
            state.output_queue.get(timeout=0.05)
        except Exception:
            continue


def _rank_main(rank: int, args: argparse.Namespace, port: int) -> None:
    torch.cuda.set_device(rank)
    service: ABotWorldLiveKitService | None = None
    try:
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=2,
        )
        print(f"rank={rank} phase=nccl_ready", flush=True)
        warmup_started = time.monotonic()
        warmup_send = torch.zeros(1, dtype=torch.uint8, device=torch.device(f"cuda:{rank}"))
        warmup_recv = torch.empty_like(warmup_send)
        send_op = dist.P2POp(dist.isend, warmup_send, 1 - rank)
        recv_op = dist.P2POp(dist.irecv, warmup_recv, 1 - rank)
        warmup_requests = dist.batch_isend_irecv(
            [send_op, recv_op] if rank == 0 else [recv_op, send_op]
        )
        for request in warmup_requests:
            request.wait()
        torch.cuda.synchronize(rank)
        dist.barrier()
        if rank == 0:
            print(f"nccl_peer_warmup_seconds={time.monotonic() - warmup_started:.6f}", flush=True)

        # Process workers are started serially in the real serving pool.  Do
        # the same here: concurrently materializing two 25-GB checkpoints on
        # this shared filesystem can make a migration smoke test look like an
        # NCCL deadlock before either rank reaches the communicator.
        for loading_rank in range(2):
            if rank == loading_rank:
                print(f"rank={rank} phase=load_replica", flush=True)
                loader = _loader_module()
                pipeline = loader.get_pipeline(
                    model_root=args.model_root,
                    pipeline_class=ABotWorldInteractivePipeline,
                    device_id=rank,
                )
                service = ABotWorldLiveKitService(
                    pipeline,
                    max_batch_size=1,
                    default_session_config={
                        "image_path": str(args.image),
                        "prompt": args.prompt,
                        "fps": 12,
                        "control_latent_frames": args.control_latent_frames,
                        "seed": args.seed,
                    },
                )
                service.start()
                torch.cuda.synchronize(rank)
                print(f"rank={rank} phase=replica_ready", flush=True)
            dist.barrier()
        assert service is not None
        session_id = "nccl-validation-session"
        metadata: dict[str, Any] | None = None
        leaves: dict[tuple[Any, ...], torch.Tensor] | None = None
        if rank == 0:
            service.create_session({"session_id": session_id})
            preview = _take_output(service, session_id)
            assert preview["type"] == "preview"
            service.push_chunk(session_id, {"type": "control_state", "controls": ["W"]})
            source_chunk = _take_output(service, session_id)
            if source_chunk.get("type") not in {"chunk", "video"}:
                raise RuntimeError(f"Expected source generated output, got {source_chunk.get('type')!r}")
            # Stop new scheduling and emulate the parent transport draining any
            # in-flight output while the source reaches a migration boundary.
            service.push_chunk(session_id, {"type": "control_state", "controls": []})
            drain_stop = threading.Event()
            drain_thread = threading.Thread(
                target=_drain_outputs,
                args=(service, session_id, drain_stop),
                daemon=True,
            )
            drain_thread.start()
            metadata = service.prepare_migration_nccl_metadata(session_id, timeout=120.0)
            drain_stop.set()
            drain_thread.join(timeout=1.0)
            leaves = metadata.pop("_nccl_tensor_leaves")
            print(
                f"source_chunk={source_chunk.get('index')} state_bytes={metadata['state_bytes']} "
                f"tensor_leaves={len(leaves)}",
                flush=True,
            )

        object_list: list[Any] = [metadata]
        dist.broadcast_object_list(object_list, src=0, device=torch.device(f"cuda:{rank}"))
        metadata = object_list[0]
        assert isinstance(metadata, dict)
        if rank == 1:
            leaves = allocate_tensor_tree_leaves(metadata["tensor_manifest"], torch.device(f"cuda:{rank}"))
        assert leaves is not None
        groups = build_layer_transfer_groups(metadata["tensor_manifest"])
        transfer_stream = torch.cuda.Stream(device=torch.device(f"cuda:{rank}"))
        if rank == 0:
            report = transfer_tensor_leaves_nccl_streamed(
                leaves,
                peer_rank=1,
                send=True,
                groups=groups,
                stream=transfer_stream,
            )
            print(
                f"nccl_copy_bytes={report.total_bytes} "
                f"nccl_copy_seconds={report.total_duration_ms / 1000.0:.6f} "
                f"transfer_groups={len(report.groups)}",
                flush=True,
            )
            service.commit_migration(session_id)
        else:
            layer_count = max(
                (group.layer_index for group in groups if group.layer_index is not None),
                default=-1,
            ) + 1
            progress = LayerTransferProgress(layer_count)
            transfer_result: dict[str, Any] = {}
            transfer_errors: list[BaseException] = []

            def receive_progressively() -> None:
                torch.cuda.set_device(rank)

                def group_complete(group, event) -> None:
                    if group.layer_index is not None:
                        progress.mark_layer_ready(group.layer_index, event)

                try:
                    transfer_result["report"] = transfer_tensor_leaves_nccl_streamed(
                        leaves,
                        peer_rank=0,
                        send=False,
                        groups=groups,
                        stream=transfer_stream,
                        on_group_complete=group_complete,
                    )
                except BaseException as exc:
                    transfer_errors.append(exc)
                    progress.mark_failed(exc)
                else:
                    progress.mark_complete()

            transfer_thread = threading.Thread(
                target=receive_progressively,
                name="sst-progressive-recv",
                daemon=True,
            )
            transfer_started = time.monotonic()
            transfer_thread.start()
            if not progress.first_layer_ready.wait(timeout=120.0):
                raise TimeoutError("Timed out waiting for SST layer 0")
            if transfer_errors:
                raise RuntimeError("SST receive failed before layer 0") from transfer_errors[0]
            installed = service.import_migration_nccl(
                metadata,
                leaves,
                owner_worker_id="worker-1",
                ownership_epoch=1,
                migration_layer_readiness=progress,
            )
            assert installed == session_id
            residual_active_at_dispatch = transfer_thread.is_alive()
            compute_started = time.monotonic()
            service.push_chunk(session_id, {"type": "control_state", "controls": ["W"]})
            target_chunk = _take_output(service, session_id)
            compute_seconds = time.monotonic() - compute_started
            output_before_transfer_complete = transfer_thread.is_alive()
            transfer_thread.join(timeout=120.0)
            if transfer_thread.is_alive():
                raise TimeoutError("Timed out waiting for residual SST layers")
            if transfer_errors:
                raise RuntimeError("SST residual receive failed") from transfer_errors[0]
            report = transfer_result["report"]
            if target_chunk.get("type") not in {"chunk", "video"}:
                raise RuntimeError(f"Expected target generated output, got {target_chunk.get('type')!r}")
            session = service._session(session_id)
            assert session is not None
            print(
                f"target_chunk={target_chunk.get('index')} "
                f"next_latent_frame={session.pipeline_session.next_latent_frame} "
                f"emitted_frames={session.pipeline_session.emitted_frames} "
                f"first_layer_ready_ms={progress.snapshot()['first_layer_ready_ms']} "
                f"transfer_complete_ms={progress.snapshot()['transfer_complete_ms']} "
                f"compute_seconds={compute_seconds:.6f} "
                f"residual_active_at_dispatch={residual_active_at_dispatch} "
                f"output_before_transfer_complete={output_before_transfer_complete} "
                f"host_wait_ms={progress.snapshot()['host_wait_ms']} "
                f"nccl_copy_seconds={report.total_duration_ms / 1000.0:.6f} "
                f"wall_transfer_seconds={time.monotonic() - transfer_started:.6f}",
                flush=True,
            )
        dist.barrier()
    finally:
        if service is not None:
            service.stop()
        if dist.is_initialized():
            dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", default="A smooth first-person exploration through a vivid natural landscape.")
    parser.add_argument("--control-latent-frames", choices=(1, 2, 3), type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mp.spawn(_rank_main, args=(args, _free_port()), nprocs=2, join=True)


if __name__ == "__main__":
    main()
