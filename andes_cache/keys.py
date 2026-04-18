"""Deterministic cache key helpers."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def normalize_query(query: str) -> str:
    """Normalize query text for stable cache keys."""
    text = (query or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def stable_hash(payload: Any) -> str:
    """Hash JSON-serializable payload with deterministic ordering."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_key(prefix: str, **parts: Any) -> str:
    """Build deterministic cache key from named parts."""
    digest = stable_hash(parts)
    return f"{prefix}:{digest}"
