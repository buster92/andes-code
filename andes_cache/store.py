"""Disk-backed cache store with per-layer namespaces and schema versioning."""

from __future__ import annotations

import json
import time
from pathlib import Path


class DiskCacheStore:
    def __init__(self, base_dir: Path, schema_version: str):
        self.base_dir = Path(base_dir)
        self.schema_version = schema_version
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        marker = self.base_dir / ".schema"
        current = marker.read_text().strip() if marker.exists() else ""
        if current != self.schema_version:
            for fp in self.base_dir.glob("**/*.json"):
                fp.unlink(missing_ok=True)
            marker.write_text(self.schema_version)

    def _path(self, layer: str, key: str) -> Path:
        layer_dir = self.base_dir / layer
        layer_dir.mkdir(parents=True, exist_ok=True)
        return layer_dir / f"{key}.json"

    def get(self, layer: str, key: str):
        path = self._path(layer, key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
        except Exception:
            return None
        return payload.get("value")

    def set(self, layer: str, key: str, value, metadata: dict | None = None) -> None:
        path = self._path(layer, key)
        payload = {
            "saved_at": time.time(),
            "value": value,
            "metadata": metadata or {},
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def invalidate_layer(self, layer: str) -> int:
        layer_dir = self.base_dir / layer
        if not layer_dir.exists():
            return 0
        removed = 0
        for fp in layer_dir.glob("*.json"):
            fp.unlink(missing_ok=True)
            removed += 1
        return removed

    def invalidate_where(self, layer: str, predicate) -> int:
        layer_dir = self.base_dir / layer
        if not layer_dir.exists():
            return 0
        removed = 0
        for fp in layer_dir.glob("*.json"):
            try:
                payload = json.loads(fp.read_text())
            except Exception:
                fp.unlink(missing_ok=True)
                removed += 1
                continue
            if predicate(payload):
                fp.unlink(missing_ok=True)
                removed += 1
        return removed
