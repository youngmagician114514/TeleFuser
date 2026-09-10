"""Quality semantics shared by the runtime profile loader and reports.

The offline ABot table contains measurements for several native batch sizes.
Batching changes execution time/throughput, not the intended user-visible
quality point.  This module therefore provides one small, dependency-free
normalization rule:

* the canonical reference is the B1, S4, W18, BF16 point;
* all batch sizes in one ``S/W/rho/precision`` family use that family's B1
  quality (a measured B2/B4 difference is treated as evaluator noise);
* quality is reported as a factor relative to the reference and is clipped to
  ``[0, 1]`` so a noisy point cannot claim quality above the reference.

If a reduced test table does not contain an S4/W18 B1 row, callers receive a
transparent raw-quality fallback rather than an arbitrary denominator.  The
returned metadata makes that fallback visible to diagnostics.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass


DEFAULT_QUALITY_REFERENCE_FIDELITY = "b1_s4_w18_rho0_bf16"
_PROFILE_RE = re.compile(r"^b(?P<batch>[1-9][0-9]*)_(?P<family>.+)$")


@dataclass(frozen=True)
class QualityNormalization:
    """Normalized values and the denominator used to derive them."""

    values: dict[str, float]
    reference_config: str | None
    reference_raw: float | None
    normalized: bool
    batch_invariant: bool


def profile_batch_family(fidelity: str) -> tuple[int | None, str]:
    """Return ``(batch, family suffix)`` for a profile name.

    Names outside the ABot ``bN_<family>`` convention are retained as an
    unparsed singleton family.  This keeps small unit-test tables and custom
    baselines usable without silently grouping unrelated rows.
    """

    match = _PROFILE_RE.fullmatch(str(fidelity).strip())
    if match is None:
        return None, str(fidelity).strip()
    return int(match.group("batch")), match.group("family")


def _is_s4_w18_family(family: str) -> bool:
    tokens = set(str(family).split("_"))
    return "s4" in tokens and "w18" in tokens


def _reference_config(
    raw_by_config: Mapping[str, float],
    requested: str,
) -> str | None:
    if requested in raw_by_config:
        return requested

    # Profile captures sometimes omit rho/precision in the config label.  A
    # semantic S4/W18 match is preferable to falling back to the maximum or a
    # random row; prefer B1 and then the smallest available batch.
    candidates: list[tuple[int, str]] = []
    for config in raw_by_config:
        batch, family = profile_batch_family(config)
        if batch is not None and batch >= 1 and _is_s4_w18_family(family):
            candidates.append((batch, config))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][1]


def normalize_profile_qualities(
    raw_by_config: Mapping[str, float],
    *,
    reference_fidelity: str = DEFAULT_QUALITY_REFERENCE_FIDELITY,
    batch_invariant: bool = True,
    clip: bool = True,
) -> QualityNormalization:
    """Normalize profile quality while making native batch quality invariant.

    ``raw_by_config`` is deliberately a mapping of fidelity name to the raw
    ``Q_world`` (or component proxy) value.  The function does not mutate the
    input.  When the explicit S4/W18 reference is absent, values are returned
    unchanged and ``normalized`` is false; this is useful for compact custom
    profile tables while production ABot tables still fail visibly in their
    metadata rather than using a misleading max-quality denominator.
    """

    raw: dict[str, float] = {}
    for config, value in raw_by_config.items():
        key = str(config).strip()
        numeric = float(value)
        if not key:
            raise ValueError("profile quality config names must be non-empty")
        if not math.isfinite(numeric) or numeric <= 0:
            raise ValueError(f"profile row {key!r} has invalid quality {numeric!r}")
        if key in raw:
            raise ValueError(f"duplicate profile quality row: {key!r}")
        raw[key] = numeric

    if not raw:
        raise ValueError("at least one profile quality value is required")

    reference_config = _reference_config(raw, str(reference_fidelity).strip())
    if reference_config is None:
        return QualityNormalization(
            values=dict(raw),
            reference_config=None,
            reference_raw=None,
            normalized=False,
            batch_invariant=False,
        )

    reference_raw = raw[reference_config]
    family_canonical: dict[str, float] = {}
    family_batch: dict[str, int] = {}
    for config, value in raw.items():
        batch, family = profile_batch_family(config)
        if not batch_invariant or batch is None:
            continue
        # B1 is the semantic quality representative.  If a reduced table has
        # no B1, retain the smallest available batch as an explicit fallback.
        previous_batch = family_batch.get(family)
        if previous_batch is None or batch < previous_batch:
            family_batch[family] = batch
            family_canonical[family] = value

    values: dict[str, float] = {}
    for config, value in raw.items():
        batch, family = profile_batch_family(config)
        canonical = family_canonical.get(family, value) if batch_invariant else value
        normalized = canonical / reference_raw
        if clip:
            normalized = min(1.0, max(0.0, normalized))
        values[config] = normalized

    return QualityNormalization(
        values=values,
        reference_config=reference_config,
        reference_raw=reference_raw,
        normalized=True,
        batch_invariant=batch_invariant,
    )


__all__ = [
    "DEFAULT_QUALITY_REFERENCE_FIDELITY",
    "QualityNormalization",
    "normalize_profile_qualities",
    "profile_batch_family",
]
