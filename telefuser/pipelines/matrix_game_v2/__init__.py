"""TeleFuser adapter contracts for Matrix-Game 2.0.

The package intentionally contains no Matrix-Game or CUDA imports at module
import time.  The actual model runtime is injected through
``TELEFUSER_MATRIX_GAME_BACKEND_MODULE`` so this backend can be tested and
loaded independently from the optional upstream model environment.
"""

from .action import MatrixGameAction
from .backend import (
    MatrixGameBackendConfig,
    MatrixGameBackendError,
    MatrixGameBackendUnavailableError,
    MatrixGameSessionBackend,
    MatrixGameStep,
    create_matrix_game_backend,
)
from .official_backend import MatrixGameOfficialBackend
from .service import MatrixGameV2Service

__all__ = [
    "MatrixGameAction",
    "MatrixGameBackendConfig",
    "MatrixGameBackendError",
    "MatrixGameBackendUnavailableError",
    "MatrixGameSessionBackend",
    "MatrixGameOfficialBackend",
    "MatrixGameStep",
    "MatrixGameV2Service",
    "create_matrix_game_backend",
]
