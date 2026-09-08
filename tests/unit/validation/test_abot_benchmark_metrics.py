from __future__ import annotations

import pytest

from tools.validation.abot_benchmark_metrics import percentile, summarize, summarize_slo


def test_summary_matches_livekit_phase_metric_shape() -> None:
    assert summarize([1.0, 2.0, 3.0, 4.0]) == {
        "count": 4,
        "mean": 2.5,
        "p50": 2.0,
        "p95": 4.0,
        "p99": 4.0,
        "maximum": 4.0,
    }


def test_slo_summary_is_backend_neutral() -> None:
    assert summarize_slo([11.75, 11.5, 12.0], target_fps=12.0, tolerance_fps=0.25) == {
        "observation_samples": 3,
        "satisfied_samples": 2,
        "sample_attainment": 0.666667,
    }


def test_metric_primitives_validate_inputs() -> None:
    assert percentile([], 0.95) == 0.0
    with pytest.raises(ValueError, match="quantile"):
        percentile([1.0], 1.1)
    with pytest.raises(ValueError, match="target_fps"):
        summarize_slo([1.0], target_fps=0.0, tolerance_fps=0.0)
