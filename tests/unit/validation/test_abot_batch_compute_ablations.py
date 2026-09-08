from __future__ import annotations

import json
import sys
from pathlib import Path

from tools.validation import benchmark_abot_batch_compute_ablations as benchmark


def test_turboserve_public_scale_profile_maps_the_published_compute_shape() -> None:
    profile = benchmark._PROFILES["turboserve_public_scale"]

    assert (profile.height, profile.width) == (224, 416)
    # ABot uses a 16x VAE spatial compression followed by a 2x2 DiT patch.
    assert (profile.height // 16, profile.width // 16) == (14, 26)
    assert profile.spatial_tokens_per_latent_frame == 91
    assert profile.local_attn_frames == 12
    assert profile.sink_frames == 3
    assert profile.denoise_step_positions == (0, 1, 2, 3)
    assert benchmark._warmup_chunks(profile, control_latent_frames=1, extra_chunks=1) == 12


def test_turboserve_public_scale_dry_run_records_one_latent_frame_chunk(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "plan"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_abot_batch_compute_ablations.py",
            "--model-root",
            str(tmp_path / "unused-model-root"),
            "--image",
            str(tmp_path / "unused-image.png"),
            "--output-dir",
            str(output_dir),
            "--profiles",
            "turboserve_public_scale",
            "--batch-sizes",
            "1,2,3,4,6",
            "--control-latent-frames",
            "1",
            "--dry-run",
        ],
    )

    benchmark.main()

    plan = json.loads((output_dir / "plan.json").read_text(encoding="utf-8"))
    assert plan["control_latent_frames"] == 1
    assert plan["frames_per_session_per_chunk"] == 4
    assert plan["batch_sizes"] == [1, 2, 3, 4, 6]
    assert plan["profiles"] == [
        {
            "denoise_step_positions": [0, 1, 2, 3],
            "description": benchmark._PROFILES["turboserve_public_scale"].description,
            "height": 224,
            "local_attn_frames": 12,
            "name": "turboserve_public_scale",
            "sink_frames": 3,
            "spatial_tokens_per_latent_frame": 91,
            "width": 416,
        }
    ]
