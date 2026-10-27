"""Compare whole-chunk, chunk-step and whole-latent-step ABot execution.

Run as a module from the repository root. Latent mode must be launched with
torchrun. Whole-latent-step mode supports arbitrary rank counts. Only
rank zero decodes video. No serving scheduler or LiveKit process is started.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image

from examples.abot_world._loader import get_pipeline
from telefuser.core.config import AttentionConfig, AttnImplType
from telefuser.pipelines.abot_world.interactive import ABotWorldInteractivePipeline
from tools.validation.abot_benchmark_metrics import percentile


def _summary(samples: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(samples),
        "p50": percentile(samples, 0.5),
        "p95": percentile(samples, 0.95),
        "maximum": max(samples),
    }


def benchmark(args: argparse.Namespace) -> dict[str, object] | None:
    distributed = args.mode == "latent"
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA; use the Gloo unit tests for CPU validation")
    device_id = int(os.environ["LOCAL_RANK"]) if distributed else args.device_id
    torch.cuda.set_device(device_id)
    if distributed:
        dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    pipeline = None
    hook = None
    driver = None
    task_mode = args.execution == "latent-step"
    try:
        pipeline = get_pipeline(
            args.model_root,
            height=args.height,
            width=args.width,
            device_id=device_id,
            pipeline_class=ABotWorldInteractivePipeline,
            latent_parallel=distributed,
        )
        # Keep the comparison eager and use the same explicitly chosen backend.
        pipeline.denoise_stage.configure_cuda_graph(False)
        backend = {"sdpa": AttnImplType.TORCH_SDPA, "sage-sm90": AttnImplType.SAGE_ATTN_2_8_8_SM90}[args.attention]
        attention = AttentionConfig.dense_attention(backend)
        pipeline.denoise_stage.model_runtime_config.attention_config = attention
        pipeline.denoise_stage.dit.set_attention_config(attention)
        with Image.open(args.image) as source:
            image = source.convert("RGB")
        sessions = [
            pipeline.create_interactive_session(
                image, args.prompt, seed=args.seed + index, session_id=f"latent-benchmark-{index}"
            )
            for index in range(args.batch_size)
        ]
        model = pipeline.denoise_stage.dit
        forward_count = 0
        forward_shapes: list[tuple[int, int]] = []

        def count_forward(_module: object, _inputs: object, kwargs: dict[str, object], _output: object) -> None:
            nonlocal forward_count
            forward_count += 1
            forward_shapes.append((kwargs["x"].shape[0], kwargs["x"].shape[2]))

        hook = model.register_forward_hook(count_forward, with_kwargs=True)
        world_size = dist.get_world_size() if distributed else 1
        if task_mode:
            driver = pipeline.create_continuous_runtime(
                max_batch_size=args.latent_slots or (3 * args.batch_size + world_size - 1) // world_size,
                granularity="latent_step",
                backend=args.backend,
                max_graphs=args.max_graphs,
            )
        elif args.execution == "step":
            driver = pipeline.create_continuous_runtime(max_batch_size=args.batch_size)
        samples: list[dict[str, float | int]] = []
        last_output = None
        for index in range(args.warmup + args.chunks):
            torch.cuda.synchronize(device_id)
            before = forward_count
            task_before = (
                (
                    driver.dispatch_count,
                    driver.compute_forward_count,
                    driver.latent_step_items,
                    driver.logical_kv_send_bytes,
                    driver.logical_output_send_bytes,
                )
                if task_mode
                else None
            )
            before_frames = sessions[0].emitted_frames
            started = time.monotonic()
            if driver is None:
                output = pipeline.generate_next_blocks_raw(sessions, [{"W": True} for _ in sessions])
                last_output = output.decoded
                phase_metrics = pipeline.last_stage_metrics()
            else:
                admitted, iteration = 0, 0
                outputs = {}
                while admitted < len(sessions) or driver.has_pending:
                    while admitted < len(sessions) and admitted * args.stagger_steps <= iteration:
                        driver.submit(sessions[admitted], {"W": True})
                        admitted += 1
                    for item in driver.step():
                        outputs[item.session_id] = item.decoded
                    iteration += 1
                last_output = torch.cat([outputs[session.session_id] for session in sessions])
                # Chunk-stage timers do not describe an interleaved step run.
                phase_metrics = {}
            torch.cuda.synchronize(device_id)
            wall_seconds = time.monotonic() - started
            if index >= args.warmup:
                samples.append(
                    {
                        **phase_metrics,
                        "wall_seconds": wall_seconds,
                        "dit_forward_calls": driver.dispatch_count - task_before[0]
                        if task_mode
                        else forward_count - before,
                        "dit_compute_forward_calls": driver.compute_forward_count - task_before[1]
                        if task_mode
                        else forward_count - before,
                        "frames_per_session": sessions[0].emitted_frames - before_frames,
                        "dit_batch_items": driver.latent_step_items - task_before[2]
                        if task_mode
                        else sum(batch for batch, _ in forward_shapes[before:]),
                        "dit_local_latent_items": driver.latent_step_items - task_before[2]
                        if task_mode
                        else sum(batch * frames for batch, frames in forward_shapes[before:]),
                        "logical_kv_send_bytes": driver.logical_kv_send_bytes - task_before[3] if task_mode else 0,
                        "logical_output_send_bytes": driver.logical_output_send_bytes - task_before[4]
                        if task_mode
                        else 0,
                    }
                )

        # Reduce once after measurement, not at every denoising step or chunk.
        slowest_wall = torch.tensor([sample["wall_seconds"] for sample in samples], device=f"cuda:{device_id}")
        if distributed:
            dist.all_reduce(slowest_wall, op=dist.ReduceOp.MAX)
        rank_stats = torch.tensor(
            [
                sum(sample["dit_compute_forward_calls"] for sample in samples),
                sum(sample["dit_batch_items"] for sample in samples),
                sum(sample["logical_kv_send_bytes"] for sample in samples),
                sum(sample["logical_output_send_bytes"] for sample in samples),
                torch.cuda.max_memory_allocated(device_id),
            ],
            device=f"cuda:{device_id}",
            dtype=torch.float64,
        )
        if distributed:
            gathered_stats = [torch.empty_like(rank_stats) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_stats, rank_stats)
        else:
            gathered_stats = [rank_stats]
        if not pipeline.is_output_rank:
            return None
        wall = slowest_wall.cpu().tolist()
        latent = sessions[0].first_frame_latent
        frame_tokens = (latent.shape[-2] // model.patch_size[1]) * (latent.shape[-1] // model.patch_size[2])
        world_size = dist.get_world_size() if distributed else 1
        calls = [int(sample["dit_forward_calls"]) for sample in samples]
        # Ring-allgather logical bytes, not a measurement of NCCL wire traffic.
        bytes_per_local_latent = frame_tokens * model.dim * latent.element_size()
        kv_collectives = 2 * model.num_layers * statistics.fmean(calls) if distributed else 0
        sent_bytes = (
            (
                2
                * model.num_layers
                * (world_size - 1)
                * bytes_per_local_latent
                * statistics.fmean([sample["dit_local_latent_items"] for sample in samples])
            )
            if distributed
            else 0
        )
        if task_mode:
            sent_bytes = statistics.fmean(sample["logical_kv_send_bytes"] for sample in samples)
        runtime = driver.exchange if task_mode else pipeline.denoise_stage.latent_parallel_runtime
        frames = [int(sample["frames_per_session"]) for sample in samples]
        result = {
            "mode": args.mode,
            "execution": args.execution,
            "backend": args.backend if task_mode else "eager",
            "batch_item_unit": "whole_latent_step" if task_mode else "session_chunk_step",
            "latent_slots_per_rank": driver.max_batch_size if task_mode else None,
            "graph_captures": driver.executor.captures if task_mode else 0,
            "graph_replays": driver.executor.replays if task_mode else 0,
            "per_rank_compute": [
                {
                    "rank": rank,
                    "compute_forward_calls": int(row[0]),
                    "batch_items": int(row[1]),
                    "logical_kv_send_bytes": int(row[2]),
                    "logical_output_send_bytes": int(row[3]),
                    "peak_allocated_bytes": int(row[4]),
                }
                for rank, row in enumerate([item.cpu().tolist() for item in gathered_stats])
            ],
            "stagger_steps": args.stagger_steps,
            "world_size": world_size,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "prompt": args.prompt,
            "model_root": str(args.model_root.resolve()),
            "image": str(args.image.resolve()),
            "height": args.height,
            "width": args.width,
            "attention_requested": args.attention,
            "dtype": str(pipeline.torch_dtype),
            "torch_version": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(device_id),
            "warmup_chunks": args.warmup,
            "measured_chunks": args.chunks,
            "batch_wall_seconds_slowest_rank": _summary(wall),
            "root_denoise_seconds": _summary([float(sample["denoise_seconds"]) for sample in samples])
            if driver is None
            else None,
            "root_decode_seconds": _summary([float(sample["vae_decode_seconds"]) for sample in samples])
            if driver is None
            else None,
            "mean_physical_batch_size": sum(item[1].item() for item in gathered_stats)
            / sum(item[0].item() for item in gathered_stats),
            "aggregate_model_decode_fps": args.batch_size * sum(frames) / sum(wall),
            "communication_model": {
                "kind": "logical ring-allgather send bytes; excludes broadcasts/protocol overhead; not measured",
                "current_kv_collectives_per_chunk": kv_collectives,
                "output_collectives_per_chunk": int(distributed) if driver is None else None,
                "current_kv_send_bytes_per_rank_per_chunk": sent_bytes,
                "full_kv_cache_replicated": not task_mode,
                "historical_kv_policy": "retained per local whole-latent task" if task_mode else "replicated per rank",
                "persistent_scratch_bytes_per_rank": sum(
                    value.numel() * value.element_size() for value in runtime._buffers.values()
                )
                if runtime
                else 0,
            },
            "samples": samples,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if args.save_decoded and last_output is not None:
            # Save outside the measured region for numerical/visual comparison.
            torch.save(last_output.cpu(), args.output.with_suffix(".decoded.pt"))
        return result
    finally:
        if hook is not None:
            hook.remove()
        try:
            if task_mode and driver is not None and not driver.has_pending:
                driver.close(materialize=False)
            if pipeline is not None:
                pipeline.close()
        finally:
            if distributed and dist.is_initialized():
                dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single", "latent"), required=True)
    parser.add_argument("--execution", choices=("chunk", "step", "latent-step"), default="chunk")
    parser.add_argument("--backend", choices=("eager", "cuda_graph"), default="eager")
    parser.add_argument(
        "--latent-slots", type=int, help="Whole-latent task slots per rank; default ceil(3*sessions/ranks)"
    )
    parser.add_argument("--max-graphs", type=int, default=4)
    parser.add_argument(
        "--stagger-steps", type=int, default=0, help="Step-mode admission spacing in dispatch iterations"
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0, help="Logical device for single mode only")
    parser.add_argument("--prompt", default="A smooth first-person exploration through a vivid natural landscape.")
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--chunks", type=int, default=12)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--attention", choices=("sdpa", "sage-sm90"), default="sdpa")
    parser.add_argument("--save-decoded", action="store_true")
    args = parser.parse_args()
    if min(args.chunks, args.batch_size) < 1 or args.warmup < 0:
        parser.error("chunks and batch-size must be positive; warmup must be nonnegative")
    if args.stagger_steps < 0 or (args.stagger_steps and args.execution == "chunk"):
        parser.error("stagger-steps must be nonnegative and requires step or latent-step execution")
    if args.backend != "eager" and args.execution != "latent-step":
        parser.error("--backend cuda_graph requires --execution latent-step")
    if args.max_graphs < 1 or (args.latent_slots is not None and args.latent_slots < 1):
        parser.error("max-graphs and latent-slots must be positive")
    return args


def main() -> None:
    result = benchmark(parse_args())
    if result is not None:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
