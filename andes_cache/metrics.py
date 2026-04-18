"""Lightweight cache metrics and latency instrumentation."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path


class CacheMetrics:
    def __init__(self, sink_path: Path):
        self.sink_path = Path(sink_path)
        self.stats = defaultdict(lambda: {
            "hits": 0,
            "misses": 0,
            "invalidations": 0,
            "stale_prevented": 0,
            "latency_saved_ms": 0.0,
        })

    def hit(self, layer: str, saved_ms: float = 0.0) -> None:
        self.stats[layer]["hits"] += 1
        self.stats[layer]["latency_saved_ms"] += max(saved_ms, 0.0)

    def miss(self, layer: str) -> None:
        self.stats[layer]["misses"] += 1

    def invalidate(self, layer: str, count: int = 1) -> None:
        self.stats[layer]["invalidations"] += max(count, 0)

    def stale_prevented(self, layer: str, count: int = 1) -> None:
        self.stats[layer]["stale_prevented"] += max(count, 0)

    def snapshot(self) -> dict:
        return {k: dict(v) for k, v in self.stats.items()}

    def flush(self) -> None:
        payload = {
            "ts": time.time(),
            "layers": self.snapshot(),
        }
        self.sink_path.parent.mkdir(parents=True, exist_ok=True)
        self.sink_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
