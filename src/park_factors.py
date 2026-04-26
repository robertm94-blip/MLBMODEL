"""Park factor loader.

Reads multiplicative runs park factors from data/park_factors.json. Falls back
to the hardcoded src/features.py:PARK_FACTORS dict so existing call sites keep
working if the JSON is missing.
"""

from __future__ import annotations

import json
import os
from typing import Any

DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "park_factors.json"
)

_cache: dict[int, float] | None = None
_meta_cache: dict[int, dict[str, Any]] | None = None


def _load(path: str = DEFAULT_PATH) -> tuple[dict[int, float], dict[int, dict[str, Any]]]:
    global _cache, _meta_cache
    if _cache is not None and _meta_cache is not None:
        return _cache, _meta_cache

    factors: dict[int, float] = {}
    meta: dict[int, dict[str, Any]] = {}

    if os.path.exists(path):
        with open(path) as fh:
            data = json.load(fh)
        for vid_str, entry in (data.get("venues") or {}).items():
            try:
                vid = int(vid_str)
            except (TypeError, ValueError):
                continue
            factor = entry.get("runs_factor")
            if factor is None:
                continue
            factors[vid] = float(factor)
            meta[vid] = entry
    else:
        from src.features import PARK_FACTORS

        for vid, factor in PARK_FACTORS.items():
            factors[int(vid)] = float(factor)
            meta[int(vid)] = {"runs_factor": float(factor)}

    _cache = factors
    _meta_cache = meta
    return factors, meta


def get_park_factor(venue_id: int | str | None, path: str = DEFAULT_PATH) -> float:
    """Return the multiplicative runs park factor for `venue_id`.

    Defaults to 1.0 (neutral) for unknown venues.
    """
    if venue_id is None:
        return 1.0
    try:
        vid = int(venue_id)
    except (TypeError, ValueError):
        return 1.0
    factors, _ = _load(path)
    if vid in factors:
        return factors[vid]
    from src.features import PARK_FACTORS

    return float(PARK_FACTORS.get(vid, 1.0))


def get_park_meta(venue_id: int | str | None, path: str = DEFAULT_PATH) -> dict[str, Any]:
    """Return the full venue record (name, team, runs_factor) or {} if unknown."""
    if venue_id is None:
        return {}
    try:
        vid = int(venue_id)
    except (TypeError, ValueError):
        return {}
    _, meta = _load(path)
    return dict(meta.get(vid, {}))


def reset_cache() -> None:
    """Reset the module-level cache. Useful for tests."""
    global _cache, _meta_cache
    _cache = None
    _meta_cache = None
