"""Stress concurrent phase peer channels and reusable receive pages on GPUs."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from telefuser.service.livekit.nccl_transfer import PersistentTensorPagePool, TensorTransferGroup
from telefuser.service.livekit.phase_transport import PhasePeerTransport


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    rank, world = dist.get_rank(), dist.get_world_size()
    transport = PhasePeerTransport.bootstrap(device)
    pages = PersistentTensorPagePool()
    started = time.monotonic()

    @torch.inference_mode()
    def peer_loop(peer: int) -> int:
        torch.cuda.set_device(device)
        sent = 0
        for step in range(args.iterations):
            send = (rank < peer) == (step % 2 == 0)
            value = (rank if send else peer) * args.iterations + step
            rows = 64 + step % 3 * 64
            manifest = [
                {"path": ["kv"], "shape": [1, rows, 8, 16], "dtype": "float16"},
                {"path": ["cursor"], "shape": [1], "dtype": "int64"},
                {"path": ["bytes"], "shape": [16], "dtype": "uint8"},
                {"path": ["empty"], "shape": [1, 0, 8, 16], "dtype": "float16"},
            ]
            if send:
                leaves = {
                    ("kv",): torch.full((1, rows, 8, 16), value, dtype=torch.float16, device=device),
                    ("cursor",): torch.full((1,), value, dtype=torch.int64, device=device),
                    ("bytes",): torch.full((16,), value % 251, dtype=torch.uint8, device=device),
                    ("empty",): torch.empty((1, 0, 8, 16), dtype=torch.float16, device=device),
                }
                slot = None
                ready = torch.cuda.Event()
                ready.record()
            else:
                with torch.cuda.stream(transport.lane.stream_for(peer)):
                    leaves, slot = pages.acquire(manifest, device)
                ready = None
            report = transport.transfer(
                leaves,
                peer=peer,
                send=send,
                groups=(TensorTransferGroup("payload", tuple(tuple(item["path"]) for item in manifest)),),
                ready=ready,
            )
            if not send:
                assert bool((leaves["kv",] == value).all()), (rank, peer, step, "KV mismatch")
                assert int(leaves["cursor",].item()) == value, (rank, peer, step, "cursor mismatch")
                assert bool((leaves["bytes",] == value % 251).all()), (rank, peer, step, "byte mismatch")
                pages.release(slot)
            else:
                sent += report.total_bytes
        return sent

    with ThreadPoolExecutor(max_workers=world - 1) as workers:
        sent = sum(workers.map(peer_loop, sorted(transport.groups)))
    elapsed = time.monotonic() - started
    counter = torch.tensor([sent, (world - 1) * args.iterations], device=device, dtype=torch.int64)
    dist.all_reduce(counter)
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "world_size": world,
                    "iterations_per_pair": args.iterations,
                    "logical_transfers": int(counter[1].item()) // 2,
                    "sent_bytes": int(counter[0].item()),
                    "elapsed_seconds": elapsed,
                    "checks": ["exact KV", "int64 cursor", "uint8 bytes", "zero-size tensors"],
                },
                indent=2,
            )
            + "\n"
        )
    transport.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
