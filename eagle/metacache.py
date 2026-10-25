"""Helpers for comparing stage metadata when deciding whether a cache is reusable.

Downstream stages record a snapshot of the upstream stage's meta file instead of the
upstream CSV modification time, so regenerating an upstream artifact with identical
settings no longer invalidates the downstream cache.
"""

import json
from pathlib import Path
from typing import Any


def load_meta_snapshot(meta_path: Path) -> dict[str, Any] | None:
    """Read a stage meta JSON file, or return None when it is missing/unreadable."""
    try:
        with Path(meta_path).open("r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def meta_snapshot_matches(cached: Any, current: dict[str, Any] | None) -> bool:
    """Return True when a recorded upstream snapshot still equals the current meta.

    A missing snapshot on either side counts as a mismatch so that caches written by
    older versions are recomputed once instead of being reused unchecked.
    """
    if not isinstance(cached, dict) or not isinstance(current, dict):
        return False
    return cached == current
