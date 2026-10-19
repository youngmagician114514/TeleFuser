"""Matrix-Game 2.0 LiveKit pipeline entrypoint.

Set ``TELEFUSER_MATRIX_GAME_BACKEND_MODULE`` to a module that implements the
sessionized Matrix backend contract.  Keeping this file free of Matrix's
optional imports lets TeleFuser validate and unit-test the entrypoint on a
CPU-only machine.
"""

from __future__ import annotations

import os

from telefuser.pipelines.matrix_game_v2.backend import MatrixGameBackendConfig, create_matrix_game_backend
from telefuser.pipelines.matrix_game_v2.service import MatrixGameV2Service


def get_service(gpu_num: int = 1, gpu_ids: list[str] | None = None) -> MatrixGameV2Service:
    """Create one Matrix-Game replica for the worker's assigned GPU."""

    assigned = list(gpu_ids) if gpu_ids else ["0"]
    if gpu_num != 1 or len(assigned) != 1:
        raise ValueError("Each Matrix-Game worker owns exactly one GPU; use multiple workers for multiple GPUs")
    raw_device = str(assigned[0])
    device = raw_device if raw_device.startswith("cuda:") else f"cuda:{raw_device}"
    config = MatrixGameBackendConfig.from_environment(device=device)

    def backend_factory():
        return create_matrix_game_backend(config)

    return MatrixGameV2Service(
        backend_factory=backend_factory,
        output_queue_size=int(os.getenv("MATRIX_GAME_OUTPUT_QUEUE_SIZE", "4")),
    )
