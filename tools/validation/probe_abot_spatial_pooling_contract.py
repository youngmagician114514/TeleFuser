#!/usr/bin/env python3
"""Audit current-chunk pooling against ABot retained-session contracts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

VIDEO = (480, 832)
LATENT = (30, 52)
PATCH = 2
CHUNK_FRAMES = 3
WINDOW_FRAMES = 18
HIGH_FRAME_TOKENS = (LATENT[0] // PATCH) * (LATENT[1] // PATCH)
HIGH_CAPACITY = WINDOW_FRAMES * HIGH_FRAME_TOKENS
HIGH_HISTORY = CHUNK_FRAMES * HIGH_FRAME_TOKENS


@dataclass(frozen=True)
class Geometry:
    config: str
    pool_ratio: float
    factor: int
    adjusted_h: int
    adjusted_w: int
    pooled_h: int
    pooled_w: int
    dit_h: int
    dit_w: int

    @property
    def frame_tokens(self) -> int:
        return self.dit_h * self.dit_w


def nearest_multiple(value: int, multiple: int) -> int:
    below = max(multiple, value // multiple * multiple)
    above = math.ceil(value / multiple) * multiple
    return min((below, above), key=lambda item: (abs(item - value), item))


def geometry(config: str, factor: int) -> Geometry:
    # Pooled latent dimensions must still divide the DiT 2x2 patch exactly.
    multiple = factor * PATCH
    adjusted_h = nearest_multiple(LATENT[0], multiple)
    adjusted_w = nearest_multiple(LATENT[1], multiple)
    pooled_h, pooled_w = adjusted_h // factor, adjusted_w // factor
    return Geometry(
        config,
        1.0 / factor,
        factor,
        adjusted_h,
        adjusted_w,
        pooled_h,
        pooled_w,
        pooled_h // PATCH,
        pooled_w // PATCH,
    )


GEOMETRIES = (geometry("high", 1), geometry("medium", 2), geometry("low", 4))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def contract(item: Geometry) -> dict[str, Any]:
    chunk_tokens = CHUNK_FRAMES * item.frame_tokens
    new_end = 2 * chunk_tokens
    high_grid = (LATENT[0] // PATCH, LATENT[1] // PATCH)
    pooled_grid = (item.dit_h, item.dit_w)
    return {
        **asdict(item),
        "frame_tokens": item.frame_tokens,
        "latent_shape_before": [1, 48, CHUNK_FRAMES, *LATENT],
        "latent_shape_after": [1, 48, CHUNK_FRAMES, item.pooled_h, item.pooled_w],
        "dit_grid_after": [CHUNK_FRAMES, item.dit_h, item.dit_w],
        "current_chunk_tokens": chunk_tokens,
        "token_ratio": chunk_tokens / (CHUNK_FRAMES * HIGH_FRAME_TOKENS),
        "unchanged_kv_shape": [1, HIGH_CAPACITY, 24, 128],
        "fixed_kv_capacity": HIGH_CAPACITY,
        "required_capacity": WINDOW_FRAMES * item.frame_tokens,
        "steady_capacity_matches": HIGH_CAPACITY == WINDOW_FRAMES * item.frame_tokens,
        "history_alignment_remainder": HIGH_HISTORY % item.frame_tokens,
        "history_alignment_valid": HIGH_HISTORY % item.frame_tokens == 0,
        "literal_action_grid": list(high_grid),
        "literal_action_matches": high_grid == pooled_grid,
        "old_global_end": HIGH_HISTORY,
        "new_current_start": chunk_tokens,
        "new_current_end": new_end,
        "cursor_would_regress": new_end < HIGH_HISTORY,
    }


def failure_reason(item: dict[str, Any]) -> str:
    failures = []
    if not item["literal_action_matches"]:
        failures.append("action and DiT grids differ")
    if not item["steady_capacity_matches"]:
        failures.append("fixed KV capacity does not match pooled query grid")
    if not item["history_alignment_valid"]:
        failures.append("historical K cannot be reshaped into pooled-grid frames for 3D RoPE")
    if item["cursor_would_regress"]:
        failures.append("token-unit cache cursor would regress")
    return "; ".join(failures)


def analysis(items: list[dict[str, Any]], baseline_dir: Path, baseline: dict[str, str]) -> str:
    high, medium, low = items
    table = "\n".join(
        f"| {x['config']} | {x['pool_ratio']:.2f} | {x['adjusted_h']}×{x['adjusted_w']} | "
        f"{x['pooled_h']}×{x['pooled_w']} | {x['dit_h']}×{x['dit_w']} | "
        f"{x['current_chunk_tokens']} | {x['fixed_kv_capacity']} / {x['required_capacity']} |"
        for x in items
    )
    return textwrap.dedent(
        f"""
        # ABot-World spatial pooling contract probe

        ## Outcome

        The current interactive architecture does **not** support pooling only the current generation chunk while
        retaining the full-resolution KV/session state. Medium and low are `unsupported`; their timing, memory, SSIM,
        CLIP and video fields are intentionally empty.

        High imports the identical three-run native baseline from `{baseline_dir}`: mean block latency
        {float(baseline['latency_ms']):.2f} ms, {float(baseline['fps']):.2f} generated FPS, and
        {float(baseline['gpu_memory_gib']):.3f} GiB peak allocated memory.

        ## Geometry

        `tokens` means current LF=3 chunk tokens. The baseline 7,020 figure is the complete 18-frame KV capacity;
        the current high chunk has 3 × 390 = 1,170 tokens.

        | Config | Pool | Adjusted pre-pool latent | Pooled latent | DiT grid | Chunk tokens | Fixed / required KV |
        | --- | ---: | ---: | ---: | ---: | ---: | ---: |
        {table}

        The adjusted sizes include the DiT 2×2 patch constraint. Medium changes 30×52 to 28×52 before pooling;
        low changes it to 32×48. Otherwise DiT unpatchify cannot reproduce the pooled latent shape.

        ## Exact blockers

        1. **Historical K depends on its spatial grid.** K is stored unrotated, but all visible K is reshaped with the
           current Q grid before 3D RoPE. One high chunk leaves {HIGH_HISTORY} tokens. Medium expects
           {medium['frame_tokens']} tokens/frame (remainder {medium['history_alignment_remainder']}); low expects
           {low['frame_tokens']} (remainder {low['history_alignment_remainder']}). Neither describes the same old
           frames.
        2. **The rolling cache assumes one frame stride.** Its size is `18 × frame_tokens`, and cursors count tokens.
           The old cursor is {HIGH_HISTORY}; switching computes end {medium['new_current_end']} or
           {low['new_current_end']}, moving the current cursor backwards and corrupting history.
        3. **Literal latent-only pooling breaks action conditioning.** The unchanged action map produces a 15×26
           adapter grid, versus pooled grids 7×13 and 4×6. ABot requires identical embedding shapes.
        4. **Every denoise call writes KV.** The four sampling forwards and final t=0 commit all receive the session
           cache. There is no read-only-history, heterogeneous-Q path.

        Keeping allocation shape `{high['unchanged_kv_shape']}` is insufficient: its 18-frame interpretation is lost.

        ## Requested questions

        1. **Does pooling lower DiT latency?** Not validly measurable. A pooled run with a newly sized cache would be
           resolution scaling under another name, not the requested fixed-history experiment.
        2. **Is the KV temporal window preserved?** No; tokens/frame and token-unit cursor semantics change.
        3. **Does it affect batching?** Yes. Different Q lengths need separate buckets, padding, or ragged attention.
        4. **Is it better than resolution scaling?** Not currently. Resolution scaling is session-static but internally
           consistent; current-only pooling requires new mixed-grid attention/cache semantics.

        ## Required future runtime design

        A supported version needs per-frame K grid metadata, normalized heterogeneous-grid RoPE, read-only historical
        KV during pooled denoising, resized action conditioning, a final full-resolution t=0 cache commit after latent
        upsampling, and ragged/padded batching. These runtime-contract changes were not forced in this pre-experiment.

        ## Quality scope

        Only baseline videos and a baseline contact sheet are emitted. Baseline self-SSIM is 1.0. CLIP and manual
        comparisons are not applicable without valid medium/low generations.

        [Baseline contact sheet](baseline_contact_sheet.png)
        """
    ).lstrip().replace("\n        ", "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    baseline_dir, output_dir = args.baseline_dir.resolve(), args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir()

    profiles = [x for x in read_csv(baseline_dir / "profile.csv") if x["config"] == "high"]
    runs = [x for x in read_csv(baseline_dir / "runs.csv") if x["config"] == "high"]
    if len(profiles) != 1 or len(runs) != 3:
        raise ValueError("baseline must contain one high profile and three high runs")
    baseline = profiles[0]
    imported_runs, quality_rows = [], []
    for row in runs:
        source = Path(row["video_path"])
        destination = videos_dir / source.name
        shutil.copy2(source, destination)
        imported_runs.append(
            {
                "config": "high",
                "run": row["run"],
                "pool_ratio": 1.0,
                "latency_ms": row["latency_ms"],
                "fps": row["fps"],
                "peak_gpu_memory_gib": row["gpu_memory_gib"],
                "reference_ssim": row["reference_ssim"],
                "video_path": str(destination.resolve()),
                "measurement_source": str((baseline_dir / "runs.csv").resolve()),
            }
        )
        quality_rows.append(
            {
                "config": "high",
                "run": row["run"],
                "video_path": str(destination.resolve()),
                "reference_ssim": row["reference_ssim"],
                "status": "baseline_only",
            }
        )

    items = [contract(x) for x in GEOMETRIES]
    for item in items[1:]:
        if (
            item["literal_action_matches"]
            or item["steady_capacity_matches"]
            or item["history_alignment_valid"]
            or not item["cursor_would_regress"]
        ):
            raise AssertionError(f"Unexpected native compatibility for {item['config']}: {item}")
    rows = []
    for item in items:
        supported = item["config"] == "high"
        rows.append(
            {
                "config": item["config"],
                "pool_ratio": item["pool_ratio"],
                "tokens": item["current_chunk_tokens"],
                "latent_shape_before": json.dumps(item["latent_shape_before"]),
                "latent_shape_after": json.dumps(item["latent_shape_after"]),
                "dit_grid_after": json.dumps(item["dit_grid_after"]),
                "kv_capacity_tokens": item["fixed_kv_capacity"],
                "latency_ms": baseline["latency_ms"] if supported else "",
                "fps": baseline["fps"] if supported else "",
                "peak_gpu_memory_gib": baseline["gpu_memory_gib"] if supported else "",
                "reference_ssim": 1.0 if supported else "",
                "video_path": str((videos_dir / "high_run1.mp4").resolve()) if supported else "",
                "status": "supported_baseline" if supported else "unsupported",
                "failure_reason": "" if supported else failure_reason(item),
                "measurement_source": str((baseline_dir / "profile.csv").resolve()) if supported else "",
            }
        )
    write_csv(output_dir / "profile.csv", rows)
    write_csv(output_dir / "runs.csv", imported_runs)
    write_csv(output_dir / "quality_manifest.csv", quality_rows)
    (output_dir / "contract.json").write_text(
        json.dumps(
            {"native_support": False, "baseline_source": str(baseline_dir), "profiles": items},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "analysis.md").write_text(analysis(items, baseline_dir, baseline), encoding="utf-8")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(videos_dir / "high_run1.mp4"),
            "-vf",
            "select='eq(n,0)+eq(n,11)+eq(n,23)+eq(n,35)',scale=416:240,tile=4x1",
            "-frames:v",
            "1",
            str(output_dir / "baseline_contact_sheet.png"),
        ],
        check=True,
    )
    print(json.dumps({"output_dir": str(output_dir), "status": "unsupported"}, sort_keys=True))


if __name__ == "__main__":
    main()
