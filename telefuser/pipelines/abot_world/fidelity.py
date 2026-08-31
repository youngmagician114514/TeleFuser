"""Runtime fidelity choices supported by the ABot-World serving path."""

from __future__ import annotations

import re
from dataclasses import dataclass

_PROFILE_RE = re.compile(
    r"^b(?P<batch>[1-9][0-9]*)_s(?P<steps>[1-9][0-9]*)_w(?P<window>[1-9][0-9]*)"
    r"_rho(?P<rho>[^_]+)_(?P<precision>[a-z0-9]+)$"
)


@dataclass(frozen=True)
class ABotWorldFidelity:
    """A profile-table fidelity that can be applied to an interactive batch.

    Offline ABot profiles use four official sampler positions and a causal KV
    window.  The batch size is deliberately excluded: it is a scheduler
    property, while every member of one batch shares this fidelity.
    """

    name: str
    denoise_step_positions: tuple[int, ...]
    local_attn_size: int
    sink_size: int
    precision: str = "bf16"
    rho: str = "0"

    @property
    def denoise_steps(self) -> int:
        return len(self.denoise_step_positions)

    @property
    def is_default(self) -> bool:
        return self.denoise_step_positions == (0, 1, 2, 3) and self.local_attn_size == 18

    @classmethod
    def from_profile_name(cls, name: str) -> "ABotWorldFidelity":
        """Parse and validate a standard offline profile configuration name."""
        if not isinstance(name, str) or not name:
            raise ValueError("ABot fidelity name must be a non-empty string")
        match = _PROFILE_RE.fullmatch(name)
        if match is None:
            raise ValueError(f"unsupported ABot fidelity profile: {name!r}")
        steps = int(match.group("steps"))
        window = int(match.group("window"))
        rho = match.group("rho")
        precision = match.group("precision")
        positions = {
            2: (0, 3),
            3: (0, 2, 3),
            4: (0, 1, 2, 3),
        }.get(steps)
        if positions is None:
            raise ValueError(f"unsupported ABot denoising step count: {steps}")
        if window not in {6, 12, 18}:
            raise ValueError(f"unsupported ABot causal KV window: {window}")
        if rho not in {"0", "0.0"} or precision != "bf16":
            raise ValueError(
                "ABot runtime currently supports only dense rho=0 BF16 profiles; "
                f"got rho={rho!r}, precision={precision!r}"
            )
        return cls(
            name=name,
            denoise_step_positions=positions,
            local_attn_size=window,
            sink_size=window // 3,
            precision=precision,
            rho=rho,
        )
