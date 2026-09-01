#!/usr/bin/env python3
"""Extract the bounded Motivation scheduler diagnostics from service metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _extract(metadata: dict[str, Any]) -> dict[str, Any]:
    scheduler = metadata.get("motivation_scheduler")
    if not isinstance(scheduler, dict):
        raise ValueError("metadata does not contain motivation_scheduler")
    diagnostics = scheduler.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError(
            "metadata does not contain motivation_scheduler.diagnostics; "
            "run the server with --motivation-profile on the diagnostics-enabled revision"
        )
    return {
        "schema_version": "motivation_diagnostics_v1",
        "motivation_scheduler_epoch": scheduler.get("epoch"),
        "motivation_scheduler_time": scheduler.get("current_time"),
        "diagnostics": diagnostics,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True, help="metadata-after.json from a LiveKit run")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON artifact")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("service metadata must be a JSON object")
    diagnostics = _extract(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote Motivation diagnostics: {args.output}")


if __name__ == "__main__":
    main()
