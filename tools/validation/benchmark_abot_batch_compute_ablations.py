"""Measure how ABot native-batch benefit changes with smaller compute profiles.

This is an offline, eager-only microbenchmark.  It holds the serving chunk
shape at LF=3 (12 output frames per session), warms every retained session to
its configured causal-KV window, then compares one native ``B``-session call
against ``B`` equivalent singleton calls.  It is intentionally a throughput
experiment, not a quality evaluation: fewer denoising calls, a shorter KV
window, and a smaller output width each change the generated-video quality or
temporal receptive field.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import os
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from telefuser.pipelines.abot_world.interactive import ABotWorldInteractivePipeline


@dataclass(frozen=True)
class ComputeProfile:
    """One deliberately controlled model-compute ablation."""

    name: str
    denoise_step_positions: tuple[int, ...]
    local_attn_frames: int
    sink_frames: int
    height: int
    width: int
    description: str

    @property
    def spatial_tokens_per_latent_frame(self) -> int:
        return (self.height // 32) * (self.width // 32)


_PROFILES: dict[str, ComputeProfile] = {
    "turboserve_public_scale": ComputeProfile(
        name="turboserve_public_scale",
        denoise_step_positions=(0, 1, 2, 3),
        local_attn_frames=12,
        sink_frames=3,
        height=224,
        width=416,
        description=(
            "ABot compute-shape match for TurboServe's released public demo: 4 denoise calls, "
            "12-frame KV with a 3-frame sink, and a 14x26 latent grid / 91 DiT spatial tokens per frame. "
            "Run this profile with --control-latent-frames 1 to match TurboServe's one-latent-frame block."
        ),
    ),
    "baseline": ComputeProfile(
        name="baseline",
        denoise_step_positions=(0, 1, 2, 3),
        local_attn_frames=18,
        sink_frames=6,
        height=480,
        width=832,
        description="Published eager LF3 runtime shape: 4 denoise calls, 18-frame KV, 390 tokens/frame.",
    ),
    "steps2": ComputeProfile(
        name="steps2",
        denoise_step_positions=(0, 3),
        local_attn_frames=18,
        sink_frames=6,
        height=480,
        width=832,
        description="Synthetic 2-call schedule using the first and last official ABot training times.",
    ),
    "kv9": ComputeProfile(
        name="kv9",
        denoise_step_positions=(0, 1, 2, 3),
        local_attn_frames=9,
        sink_frames=3,
        height=480,
        width=832,
        description="Half-length causal KV window; sink-to-window ratio is retained at one third.",
    ),
    "half_tokens": ComputeProfile(
        name="half_tokens",
        denoise_step_positions=(0, 1, 2, 3),
        local_attn_frames=18,
        sink_frames=6,
        height=480,
        width=416,
        description="Half the spatial DiT tokens/frame (480x416 instead of 480x832).",
    ),
    "reduced_all": ComputeProfile(
        name="reduced_all",
        denoise_step_positions=(0, 3),
        local_attn_frames=9,
        sink_frames=3,
        height=480,
        width=416,
        description="Combined 2-call, 9-frame-KV, half-token compute ablation.",
    ),
}


def _load_example_loader() -> Any:
    path = Path(__file__).resolve().parents[2] / "examples/abot_world/_loader.py"
    spec = importlib.util.spec_from_file_location("abot_batch_compute_ablation_loader", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load ABot example loader: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_profiles(value: str) -> list[ComputeProfile]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(names).difference(_PROFILES))
    if not names or unknown:
        raise argparse.ArgumentTypeError(
            f"profiles must be a nonempty comma-separated subset of {','.join(_PROFILES)}; got {value!r}"
        )
    return [_PROFILES[name] for name in names]


def _parse_positive_ints(value: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from exc
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "count": float(len(values)),
        "mean": statistics.fmean(values),
        "p50": ordered[(len(ordered) - 1) // 2],
        "p95": ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)],
        "min": ordered[0],
        "max": ordered[-1],
    }


def _warmup_chunks(profile: ComputeProfile, control_latent_frames: int, extra_chunks: int) -> int:
    """Return chunks after the distinct first chunk needed for full KV state."""
    fill_after_first = max(0, profile.local_attn_frames - control_latent_frames)
    return math.ceil(fill_after_first / control_latent_frames) + extra_chunks


def _configure_profile(
    pipeline: ABotWorldInteractivePipeline,
    profile: ComputeProfile,
) -> list[float]:
    """Install local benchmark-only cache and denoise-call overrides."""
    pipeline.config.local_attn_size = profile.local_attn_frames
    pipeline.config.sink_size = profile.sink_frames
    pipeline.config.height = profile.height
    pipeline.config.width = profile.width
    stage = pipeline.denoise_stage
    stage.configure_cuda_graph(False)
    stage.dit.set_causal_attention_window(profile.local_attn_frames, profile.sink_frames)
    original: Callable[[Any], torch.Tensor] = stage._official_denoising_timesteps

    def selected_timesteps(scheduler: Any) -> torch.Tensor:
        source = original(scheduler)
        positions = torch.tensor(profile.denoise_step_positions, device=source.device, dtype=torch.long)
        return source.index_select(0, positions)

    stage._official_denoising_timesteps = selected_timesteps
    scheduler = stage._scheduler()
    return [float(item) for item in selected_timesteps(scheduler).cpu().tolist()]


def _create_sessions(
    pipeline: ABotWorldInteractivePipeline,
    image: Image.Image,
    prompt: str,
    seed: int,
    batch_size: int,
    profile: ComputeProfile,
    control_latent_frames: int,
    extra_warmup_chunks: int,
    tag: str,
) -> tuple[list[Any], list[dict[str, bool]], int]:
    sessions = [
        pipeline.create_interactive_session(
            image,
            prompt,
            seed=seed + index,
            session_id=f"{profile.name}-{tag}-b{batch_size}-s{index}",
        )
        for index in range(batch_size)
    ]
    controls = [{"W": True} for _ in sessions]
    expected_frames = 4 * control_latent_frames
    warmup_chunks = _warmup_chunks(profile, control_latent_frames, extra_warmup_chunks)
    try:
        first = pipeline.generate_next_blocks(sessions, controls, control_latent_frames=control_latent_frames)
        if any(len(frames) != expected_frames for frames in first):
            raise RuntimeError(f"First chunk emitted unexpected frame counts: {[len(frames) for frames in first]}")
        for _ in range(warmup_chunks):
            frames = pipeline.generate_next_blocks(sessions, controls, control_latent_frames=control_latent_frames)
            if any(len(item) != expected_frames for item in frames):
                raise RuntimeError(f"Warmup emitted unexpected frame counts: {[len(item) for item in frames]}")
    except Exception:
        for session in sessions:
            pipeline.close_interactive_session(session)
        raise
    return sessions, controls, warmup_chunks


def _close_sessions(pipeline: ABotWorldInteractivePipeline, sessions: Sequence[Any]) -> None:
    for session in sessions:
        pipeline.close_interactive_session(session)
    torch.cuda.empty_cache()


def _run_native_batch(
    pipeline: ABotWorldInteractivePipeline,
    image: Image.Image,
    args: argparse.Namespace,
    profile: ComputeProfile,
    batch_size: int,
) -> dict[str, Any]:
    device = torch.device(pipeline.device)
    sessions, controls, warmup_chunks = _create_sessions(
        pipeline,
        image,
        args.prompt,
        args.seed,
        batch_size,
        profile,
        args.control_latent_frames,
        args.extra_warmup_chunks,
        "native",
    )
    expected_frames = 4 * args.control_latent_frames
    samples: list[float] = []
    stage_samples: list[dict[str, Any]] = []
    try:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(args.repeats):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            frames = pipeline.generate_next_blocks(sessions, controls, control_latent_frames=args.control_latent_frames)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            if any(len(item) != expected_frames for item in frames):
                raise RuntimeError(f"Native batch emitted unexpected frame counts: {[len(item) for item in frames]}")
            samples.append(elapsed)
            stage_samples.append(dict(pipeline.last_stage_metrics()))
        return {
            "timing_seconds": _summary(samples),
            "samples_seconds": samples,
            "warmup_chunks_after_first": warmup_chunks,
            "stage_metric_samples": stage_samples,
            "taew_decode_modes": sorted({int(item.get("taew_decode_mode", -1)) for item in stage_samples}),
            "taew_effective_batch_sizes": sorted(
                {int(item.get("taew_decode_batch_size", 0)) for item in stage_samples}
            ),
            "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
    finally:
        _close_sessions(pipeline, sessions)


def _run_serial_group(
    pipeline: ABotWorldInteractivePipeline,
    image: Image.Image,
    args: argparse.Namespace,
    profile: ComputeProfile,
    batch_size: int,
) -> dict[str, Any]:
    """Measure B session chunks one after another without a model microbatch."""
    device = torch.device(pipeline.device)
    sessions, controls, warmup_chunks = _create_sessions(
        pipeline,
        image,
        args.prompt,
        args.seed,
        batch_size,
        profile,
        args.control_latent_frames,
        args.extra_warmup_chunks,
        "serial",
    )
    expected_frames = 4 * args.control_latent_frames
    samples: list[float] = []
    stage_samples: list[dict[str, Any]] = []
    try:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(args.repeats):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            for session, controls_for_session in zip(sessions, controls, strict=True):
                frames = pipeline.generate_next_block(
                    session,
                    controls_for_session,
                    control_latent_frames=args.control_latent_frames,
                )
                if len(frames) != expected_frames:
                    raise RuntimeError(f"Serial session emitted {len(frames)} instead of {expected_frames} frames")
                stage_samples.append(dict(pipeline.last_stage_metrics()))
            torch.cuda.synchronize(device)
            samples.append(time.perf_counter() - started)
        return {
            "timing_seconds": _summary(samples),
            "samples_seconds": samples,
            "warmup_chunks_after_first": warmup_chunks,
            "stage_metric_samples": stage_samples,
            "taew_decode_modes": sorted({int(item.get("taew_decode_mode", -1)) for item in stage_samples}),
            "taew_effective_batch_sizes": sorted(
                {int(item.get("taew_decode_batch_size", 0)) for item in stage_samples}
            ),
            "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
    finally:
        _close_sessions(pipeline, sessions)


def _row_for_batch(
    native: dict[str, Any],
    serial: dict[str, Any],
    batch_size: int,
    frames_per_session: int,
) -> dict[str, Any]:
    native_mean = float(native["timing_seconds"]["mean"])
    serial_mean = float(serial["timing_seconds"]["mean"])
    return {
        "batch_size": batch_size,
        "native_batch_mean_seconds": native_mean,
        "native_batch_p95_seconds": float(native["timing_seconds"]["p95"]),
        "serial_mean_seconds": serial_mean,
        "serial_p95_seconds": float(serial["timing_seconds"]["p95"]),
        # Match the supplied table: time_serial / time_batch - 1.
        "batch_gain_percent": (serial_mean / native_mean - 1.0) * 100.0 if native_mean else 0.0,
        "native_aggregate_fps": frames_per_session * batch_size / native_mean if native_mean else 0.0,
        "serial_aggregate_fps": frames_per_session * batch_size / serial_mean if serial_mean else 0.0,
        "native_per_user_fps": frames_per_session / native_mean if native_mean else 0.0,
        "native": native,
        "serial": serial,
    }


def _write_summary_markdown(result: dict[str, Any], path: Path) -> None:
    control_latent_frames = int(result["plan"]["control_latent_frames"])
    frames_per_session = int(result["plan"]["frames_per_session_per_chunk"])
    lines = [
        "# ABot batch-versus-serial compute ablations",
        "",
        f"All rows are eager-only LF={control_latent_frames} continuation chunks "
        f"({frames_per_session} output frames/session). "
        "`batch_gain_percent = T_serial / T_native_batch - 1`; P95 is the highest sample at this small repeat count.",
        "",
    ]
    for profile_result in result["profiles"]:
        profile = profile_result["profile"]
        lines.extend(
            [
                f"## {profile['name']}",
                "",
                profile["description"],
                "",
                f"- denoise calls: {len(profile['denoise_step_positions'])}; KV: {profile['local_attn_frames']} frames "
                f"(sink {profile['sink_frames']}); resolution: {profile['height']}x{profile['width']}; "
                f"DiT spatial tokens/frame: {profile['spatial_tokens_per_latent_frame']}",
                "",
                "| B | Native batch time / P95 | Serial time / P95 | Batch gain | "
                "Batch aggregate FPS | Serial aggregate FPS | Batch per-user FPS |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in profile_result["rows"]:
            lines.append(
                "| {batch_size} | {native_batch_mean_seconds:.3f} / {native_batch_p95_seconds:.3f} s | "
                "{serial_mean_seconds:.3f} / {serial_p95_seconds:.3f} s | {batch_gain_percent:+.1f}% | "
                "{native_aggregate_fps:.2f} | {serial_aggregate_fps:.2f} | {native_per_user_fps:.2f} |".format(**row)
            )
        lines.append("")
    lines.extend(
        [
            "## Scope",
            "",
            "This changes compute only. It does not validate video quality, action fidelity, "
            "or an end-to-end serving SLO. The reduced-step schedule is an endpoint "
            "subsampling of ABot's four official training times and is not a claim of an "
            "equivalent production sampler.",
            "",
            "`turboserve_public_scale` matches TurboServe's public *compute shape*, not its Wan architecture: "
            "ABot remains a different 0.5B model with a different VAE, conditioning path, and scheduler.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profiles", type=_parse_profiles, default=list(_PROFILES.values()))
    parser.add_argument("--batch-sizes", type=_parse_positive_ints, default=[1, 2, 3, 4])
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--extra-warmup-chunks", type=int, default=1)
    parser.add_argument("--control-latent-frames", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="A smooth first-person exploration through a vivid natural landscape.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("repeats must be at least 2")
    if args.extra_warmup_chunks < 0:
        parser.error("extra-warmup-chunks must be non-negative")
    if args.device_id < 0:
        parser.error("device-id must be non-negative")
    return args


def main() -> None:
    args = _parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "script": Path(__file__).name,
        "eager_only": True,
        "control_latent_frames": args.control_latent_frames,
        "frames_per_session_per_chunk": 4 * args.control_latent_frames,
        "batch_sizes": args.batch_sizes,
        "repeats": args.repeats,
        "extra_warmup_chunks": args.extra_warmup_chunks,
        "profiles": [
            asdict(profile) | {"spatial_tokens_per_latent_frame": profile.spatial_tokens_per_latent_frame}
            for profile in args.profiles
        ],
    }
    (args.output_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    if args.device_id >= torch.cuda.device_count():
        raise ValueError(f"device-id {args.device_id} is outside visible CUDA devices [0,{torch.cuda.device_count()})")
    # Disable graph capture before the loader creates the pipeline; this is a
    # native eager batch experiment rather than a graph benchmark.
    os.environ["TELEFUSER_ABOT_CUDA_GRAPH_ENABLED"] = "0"
    torch.cuda.set_device(args.device_id)
    loader = _load_example_loader()
    image = Image.open(args.image).convert("RGB")
    profile_results: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    try:
        for profile in args.profiles:
            print(f"loading profile={profile.name}", flush=True)
            pipeline = loader.get_pipeline(
                model_root=args.model_root,
                height=profile.height,
                width=profile.width,
                device_id=args.device_id,
                pipeline_class=ABotWorldInteractivePipeline,
            )
            try:
                selected_timesteps = _configure_profile(pipeline, profile)
                rows: list[dict[str, Any]] = []
                for batch_size in args.batch_sizes:
                    print(f"profile={profile.name} batch={batch_size} native", flush=True)
                    native = _run_native_batch(pipeline, image, args, profile, batch_size)
                    print(f"profile={profile.name} batch={batch_size} serial", flush=True)
                    serial = _run_serial_group(pipeline, image, args, profile, batch_size)
                    row = _row_for_batch(native, serial, batch_size, 4 * args.control_latent_frames)
                    rows.append(row)
                    csv_rows.append(
                        {
                            "profile": profile.name,
                            "denoise_calls": len(profile.denoise_step_positions),
                            "local_attn_frames": profile.local_attn_frames,
                            "sink_frames": profile.sink_frames,
                            "height": profile.height,
                            "width": profile.width,
                            "spatial_tokens_per_latent_frame": profile.spatial_tokens_per_latent_frame,
                            **{key: value for key, value in row.items() if key not in {"native", "serial"}},
                        }
                    )
                profile_result = {
                    "profile": asdict(profile)
                    | {"spatial_tokens_per_latent_frame": profile.spatial_tokens_per_latent_frame},
                    "selected_scheduler_timesteps": selected_timesteps,
                    "rows": rows,
                }
                profile_results.append(profile_result)
                result = {"plan": plan, "profiles": profile_results}
                (args.output_dir / "results.json").write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                _write_summary_markdown(result, args.output_dir / "summary.md")
            finally:
                pipeline.close()
                del pipeline
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        if csv_rows:
            fields = list(csv_rows[0])
            with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(csv_rows)
    print(
        json.dumps(
            {"output_dir": str(args.output_dir), "profiles": [item["profile"]["name"] for item in profile_results]}
        )
    )


if __name__ == "__main__":
    main()
