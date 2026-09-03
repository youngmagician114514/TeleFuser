from __future__ import annotations

import pytest

from tools.validation import extract_motivation_diagnostics as extractor

_MINIMAL_SCHEDULER = {
    "epoch": 17,
    "current_time": 12.5,
    "diagnostics": {"search_count": 3, "dispatch_count": 2},
}


def test_extract_includes_optional_migration_diagnostics() -> None:
    migration = {
        "attempts_total": 4,
        "success_total": 3,
        "failure_total": 1,
        "worker_exits_total": 1,
    }
    metadata = {
        "motivation_scheduler": _MINIMAL_SCHEDULER,
        "turboserve_routing": {"migration_diagnostics": migration},
    }

    report = extractor._extract(metadata)

    assert report["schema_version"] == "motivation_diagnostics_v1"
    assert report["motivation_scheduler_epoch"] == 17
    assert report["diagnostics"] == _MINIMAL_SCHEDULER["diagnostics"]
    assert report["migration_diagnostics"] == migration


def test_extract_keeps_legacy_shape_without_migration_snapshot() -> None:
    metadata = {"motivation_scheduler": _MINIMAL_SCHEDULER}

    report = extractor._extract(metadata)

    assert report == {
        "schema_version": "motivation_diagnostics_v1",
        "motivation_scheduler_epoch": 17,
        "motivation_scheduler_time": 12.5,
        "diagnostics": _MINIMAL_SCHEDULER["diagnostics"],
    }


@pytest.mark.parametrize("routing", [None, [], {"migration_diagnostics": []}, {"other": {}}])
def test_extract_ignores_malformed_optional_migration_snapshot(routing: object) -> None:
    metadata = {"motivation_scheduler": _MINIMAL_SCHEDULER, "turboserve_routing": routing}

    report = extractor._extract(metadata)

    assert "migration_diagnostics" not in report


def test_extract_requires_scheduler_diagnostics() -> None:
    with pytest.raises(ValueError, match="motivation_scheduler"):
        extractor._extract({})
