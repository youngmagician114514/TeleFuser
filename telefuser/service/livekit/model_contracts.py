"""Small model-facing contracts used by the LiveKit control plane.

The scheduler deliberately knows nothing about a particular video model.  These
contracts describe the three pieces of model information that a new backend
must provide while keeping the existing ABot-World path as the default:

* :class:`ModelServingCapabilities` describes the model's output contract;
* :class:`SessionStateCodec` is the model-owned retained-session state boundary;
* :class:`BatchCompatibilityProvider` exposes the state/layout key used by
  cross-session batching.

All interfaces are intentionally narrow.  A backend can implement them
without importing the scheduler or LiveKit transport classes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ModelServingCapabilities:
    """Static serving properties required by the generic control plane.

    ``frames_per_job`` is the number of output frames represented by one
    logical model job.  ``output_fps`` is the playback rate used to translate
    those frames into a playout duration.  The remaining flags are descriptive
    capabilities and do not change scheduling by themselves.
    """

    model_name: str
    output_fps: float
    frames_per_job: int
    supports_cross_session_batching: bool = True
    supports_session_state_transfer: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if not math.isfinite(float(self.output_fps)) or float(self.output_fps) <= 0:
            raise ValueError("output_fps must be positive and finite")
        if (
            isinstance(self.frames_per_job, bool)
            or not isinstance(self.frames_per_job, int)
            or self.frames_per_job <= 0
        ):
            raise ValueError("frames_per_job must be a positive integer")
        if not isinstance(self.supports_cross_session_batching, bool):
            raise TypeError("supports_cross_session_batching must be a bool")
        if not isinstance(self.supports_session_state_transfer, bool):
            raise TypeError("supports_session_state_transfer must be a bool")

    @property
    def job_duration_seconds(self) -> float:
        """Nominal playback duration represented by one logical job."""

        return float(self.frames_per_job) / float(self.output_fps)


# Preserve the ABot-World contract currently used by all existing entrypoints.
ABOT_WORLD_CAPABILITIES = ModelServingCapabilities(
    model_name="abot-world",
    output_fps=12.0,
    frames_per_job=12,
)


@runtime_checkable
class SessionStateCodec(Protocol):
    """Model-owned export/import boundary for one retained session.

    The returned object is opaque to the scheduler and may be a CPU snapshot,
    device-resident paged state, or a transport handle.  Ownership and
    transport lifecycle remain the responsibility of the migration backend.
    """

    def export_state(self, session: Any) -> Any:
        """Export a quiescent session state at a model-safe boundary."""

    def import_state(
        self,
        state: Any,
        *,
        owner_worker_id: str | None = None,
        ownership_epoch: int | None = None,
    ) -> Any:
        """Restore a state object on the target model backend."""


@runtime_checkable
class IncrementalSessionStateCodec(Protocol):
    """Optional copy-on-write state contract for asynchronous rehoming.

    A backend implementing this protocol may retain a lightweight version
    handle while the session continues to execute and later export only the
    state changed after that handle.  The serving runtime falls back to the
    regular :class:`SessionStateCodec` path when these hooks are absent; this
    keeps existing models correct while allowing GPU-native paged backends to
    remove full-session exports from the migration critical path.
    """

    def begin_incremental_snapshot(self, session: Any) -> Any:
        """Return an opaque immutable/copy-on-write snapshot handle."""

    def export_incremental_state(self, session: Any, base_snapshot: Any) -> Any:
        """Export state changed since ``base_snapshot``."""

    def release_incremental_snapshot(self, base_snapshot: Any) -> None:
        """Release resources retained by a snapshot handle."""


@runtime_checkable
class BatchCompatibilityProvider(Protocol):
    """Provide the structural key used to form a physical model batch."""

    def batch_compatibility_key(
        self,
        session_id: str,
        *,
        fidelity: str | None = None,
    ) -> tuple[object, ...]:
        """Return a stable key for the session's current model layout."""
