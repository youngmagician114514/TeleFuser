"""Normalize TeleFuser input chunks to Matrix-Game action conditions."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping
from typing import Any


_KEY_ALIASES = {
    "arrowup": "w",
    "arrowdown": "s",
    "arrowleft": "a",
    "arrowright": "d",
    "keyup": "w",
    "keydown": "s",
    "keyleft": "a",
    "keyright": "d",
    "keyw": "w",
    "keya": "a",
    "keys": "s",
    "keyd": "d",
    "keyi": "i",
    "keyj": "j",
    "keyk": "k",
    "keyl": "l",
    "up": "w",
    "down": "s",
    "left": "a",
    "right": "d",
    "forward": "w",
    "backward": "s",
    "w": "w",
    "a": "a",
    "s": "s",
    "d": "d",
    "i": "i",
    "j": "j",
    "k": "k",
    "l": "l",
}


def _as_float(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Matrix-Game {field} must be numeric, got {value!r}") from exc
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"Matrix-Game {field} must be finite, got {value!r}")
    return result


def _key_values(value: Any) -> Iterable[Any]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        # Browser events often arrive as {"key": "w", "pressed": true}.
        if value.get("pressed", True) is False:
            return ()
        return (value.get("key", value.get("code", "")),)
    if isinstance(value, Iterable):
        return value
    return (value,)


def _normalise_key(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    key = value.strip().lower()
    if not key or key in {"idle", "none", "noop", "neutral"}:
        return None
    return _KEY_ALIASES.get(key, key)


@dataclass(frozen=True)
class MatrixGameAction:
    """Canonical keyboard/mouse action passed to a Matrix backend."""

    keyboard: tuple[str, ...] = ()
    mouse_dx: float = 0.0
    mouse_dy: float = 0.0

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "MatrixGameAction":
        data = dict(payload or {})
        nested = data.get("action")
        if isinstance(nested, Mapping):
            merged = dict(nested)
            merged.update({key: value for key, value in data.items() if key != "action"})
            data = merged

        raw_keys: list[Any] = []
        for field in ("keyboard", "keys", "controls", "control", "key", "code"):
            if field in data:
                raw_keys.extend(_key_values(data[field]))
        keys: list[str] = []
        for value in raw_keys:
            key = _normalise_key(value)
            if key is not None and key not in keys:
                keys.append(key)

        mouse = data.get("mouse")
        mouse_dx = data.get("mouse_dx", data.get("dx", 0.0))
        mouse_dy = data.get("mouse_dy", data.get("dy", 0.0))
        if isinstance(mouse, Mapping):
            mouse_dx = mouse.get("dx", mouse.get("x", mouse_dx))
            mouse_dy = mouse.get("dy", mouse.get("y", mouse_dy))
        elif isinstance(mouse, (tuple, list)) and len(mouse) >= 2:
            mouse_dx, mouse_dy = mouse[0], mouse[1]

        return cls(
            keyboard=tuple(keys),
            mouse_dx=_as_float(mouse_dx, field="mouse_dx"),
            mouse_dy=_as_float(mouse_dy, field="mouse_dy"),
        )

    def as_conditions(self) -> dict[str, Any]:
        """Return names used by the external Matrix session backend."""

        return {
            "keyboard": list(self.keyboard),
            "mouse": {"dx": self.mouse_dx, "dy": self.mouse_dy},
        }
