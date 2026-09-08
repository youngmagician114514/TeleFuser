"""Small, transport-independent helpers for model dispatch audit traces.

The process and process-NCCL worker pools share one parent-side trace format.
Keeping the bounded writer here prevents the two transports from growing
slightly different schemas while leaving model and LiveKit code unaware of
the experiment artifact.
"""

from __future__ import annotations

import contextlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from telefuser.utils.logging import logger


class DispatchTraceWriter:
    """Bounded, parent-owned JSONL writer for dispatch audit records."""

    def __init__(self, path: str, *, max_events: int, workers: dict[str, list[str]]) -> None:
        self.path = Path(path).expanduser().resolve()
        self.max_events = int(max_events)
        if self.max_events < 0:
            raise ValueError("max_events must be non-negative")
        self.received_events = 0
        self.written_events = 0
        self.dropped_events = 0
        self.write_errors = 0
        self._write_error_logged = False
        self._handle: Any | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(f"dispatch trace path already exists; choose a fresh run-scoped path: {self.path}")
        self._handle = self.path.open("x", encoding="utf-8")
        self._write_line(
            {
                "schema_version": 1,
                "event_type": "trace_metadata",
                "trace_started_monotonic_seconds": time.monotonic(),
                "trace_started_unix_seconds": time.time(),
                "trace_started_utc": datetime.now(timezone.utc).isoformat(),
                "max_dispatch_events": self.max_events,
                "configured_workers": workers,
            }
        )

    def _write_line(self, record: dict[str, Any]) -> bool:
        handle = self._handle
        if handle is None:
            return False
        try:
            handle.write(json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n")
            handle.flush()
            return True
        except (OSError, TypeError, ValueError) as exc:
            self.write_errors += 1
            if not self._write_error_logged:
                self._write_error_logged = True
                logger.warning("Failed to write ABot dispatch trace %s: %s", self.path, exc)
            return False

    def append(self, record: dict[str, Any]) -> None:
        """Append one event, dropping only events over the configured bound."""
        self.received_events += 1
        if self.received_events > self.max_events:
            self.dropped_events += 1
            return
        enriched = dict(record)
        enriched["parent_sequence"] = self.received_events
        enriched["parent_received_monotonic_seconds"] = time.monotonic()
        enriched["parent_received_unix_seconds"] = time.time()
        if self._write_line(enriched):
            self.written_events += 1
        else:
            self.dropped_events += 1

    def snapshot(self) -> dict[str, object]:
        """Return bounded writer counters for service metadata."""
        return {
            "enabled": True,
            "path": str(self.path),
            "max_events": self.max_events,
            "received_events": self.received_events,
            "written_events": self.written_events,
            "dropped_events": self.dropped_events,
            "write_errors": self.write_errors,
        }

    def close(self) -> None:
        """Close the file once; repeated calls are harmless."""
        handle = self._handle
        self._handle = None
        if handle is not None:
            with contextlib.suppress(OSError):
                handle.close()

__all__ = ["DispatchTraceWriter"]
