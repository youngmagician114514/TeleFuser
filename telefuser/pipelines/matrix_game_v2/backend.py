"""Optional Matrix-Game runtime boundary.

The upstream Matrix-Game repository exposes a full streaming inference script,
but does not expose a retained-session, multi-session service API.  Keeping
that model-specific refactor behind this small protocol prevents optional
Matrix dependencies from leaking into TeleFuser's control plane.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from telefuser.service.livekit.model_contracts import ModelServingCapabilities

from .action import MatrixGameAction


class MatrixGameBackendError(RuntimeError):
    """Base error raised by the optional Matrix-Game runtime."""


class MatrixGameBackendUnavailableError(MatrixGameBackendError):
    """Raised when no sessionized Matrix runtime has been configured."""


@dataclass(frozen=True)
class MatrixGameBackendConfig:
    """Configuration passed to an external Matrix-Game backend factory."""

    model_root: str | None = None
    config_path: str | None = None
    checkpoint_path: str | None = None
    pretrained_model_path: str | None = None
    device: str = "cuda"
    mode: str = "universal"
    num_frame_per_block: int = 3
    output_fps: float = 25.0
    frames_per_job: int = 12

    @classmethod
    def from_environment(cls, *, device: str = "cuda") -> "MatrixGameBackendConfig":
        def optional(name: str) -> str | None:
            value = os.getenv(name)
            return value.strip() if value and value.strip() else None

        return cls(
            model_root=optional("MATRIX_GAME_ROOT"),
            config_path=optional("MATRIX_GAME_CONFIG"),
            checkpoint_path=optional("MATRIX_GAME_CHECKPOINT"),
            pretrained_model_path=optional("MATRIX_GAME_PRETRAINED_MODEL"),
            device=os.getenv("MATRIX_GAME_DEVICE", device),
            mode=os.getenv("MATRIX_GAME_MODE", "universal"),
            num_frame_per_block=int(os.getenv("MATRIX_GAME_FRAMES_PER_BLOCK", "3")),
            output_fps=float(os.getenv("MATRIX_GAME_OUTPUT_FPS", "25")),
            frames_per_job=int(os.getenv("MATRIX_GAME_FRAMES_PER_JOB", "12")),
        )


@dataclass(frozen=True)
class MatrixGameStep:
    """One generated output chunk returned by a session backend."""

    frames: Sequence[Any]
    index: int
    fps: float | None = None


class MatrixGameSessionBackend(Protocol):
    """Synchronous per-session model API used by :class:`MatrixGameV2Service`."""

    capabilities: ModelServingCapabilities

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def create_session(self, session_id: str, config: Mapping[str, Any]) -> Any: ...

    def step(self, session: Any, action: MatrixGameAction) -> MatrixGameStep: ...

    def close_session(self, session: Any) -> None: ...


def create_matrix_game_backend(
    config: MatrixGameBackendConfig,
    *,
    factory: Callable[[MatrixGameBackendConfig], MatrixGameSessionBackend] | None = None,
) -> MatrixGameSessionBackend:
    """Create a model backend without importing Matrix-Game at TeleFuser load time.

    Production integrations provide a small module through
    ``TELEFUSER_MATRIX_GAME_BACKEND_MODULE``.  The module must expose
    ``create_backend(config)``.  This is deliberate: the official upstream
    inference script owns a single long-running pipeline and has no safe
    session state boundary that TeleFuser could infer automatically.
    """

    if factory is not None:
        return factory(config)
    module_name = os.getenv("TELEFUSER_MATRIX_GAME_BACKEND_MODULE")
    if not module_name:
        raise MatrixGameBackendUnavailableError(
            "Matrix-Game 2.0 needs a sessionized backend factory. Set "
            "TELEFUSER_MATRIX_GAME_BACKEND_MODULE to a module exposing "
            "create_backend(config). The upstream inference script is a "
            "single-session application and cannot be safely shared by TeleFuser yet."
        )
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - depends on optional environment
        raise MatrixGameBackendUnavailableError(
            f"Could not import Matrix backend module {module_name!r}: {exc}"
        ) from exc
    backend_factory = getattr(module, "create_backend", None)
    if not callable(backend_factory):
        raise MatrixGameBackendUnavailableError(
            f"Matrix backend module {module_name!r} must expose callable create_backend(config)"
        )
    try:
        backend = backend_factory(config)
    except Exception as exc:  # pragma: no cover - depends on optional environment
        raise MatrixGameBackendError(f"Matrix backend factory failed: {exc}") from exc
    for name in ("start", "stop", "create_session", "step", "close_session"):
        if not callable(getattr(backend, name, None)):
            raise MatrixGameBackendError(f"Matrix backend is missing required method {name}()")
    capabilities = getattr(backend, "capabilities", None)
    if not isinstance(capabilities, ModelServingCapabilities):
        raise MatrixGameBackendError("Matrix backend must expose ModelServingCapabilities as capabilities")
    return backend
