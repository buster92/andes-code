"""Lightweight cache benchmark utility (cold vs warm)."""

import time
from pathlib import Path

from andes_cache.manager import AndesCacheManager


def run() -> None:
    mgr = AndesCacheManager(Path("index") / "cache")
    repo_fp = "bench-repo"
    query = "architecture overview"

    t0 = time.perf_counter()
    _ = mgr.retrieval_get(repo_fp=repo_fp, query=query, index_version="bench-v1")
    cold_ms = (time.perf_counter() - t0) * 1000

    mgr.retrieval_set(
        repo_fp=repo_fp,
        query=query,
        index_version="bench-v1",
        value=[{"file": "server.py", "score": 0.1}],
    )

    t1 = time.perf_counter()
    _ = mgr.retrieval_get(repo_fp=repo_fp, query=query, index_version="bench-v1")
    warm_ms = (time.perf_counter() - t1) * 1000

    mgr.flush_metrics()

    print(f"Cold retrieval lookup: {cold_ms:.2f}ms")
    print(f"Warm retrieval lookup: {warm_ms:.2f}ms")
    if warm_ms > 0:
        print(f"Approx speedup: {cold_ms / warm_ms:.2f}x")
    print("Metrics written to index/cache/metrics/cache_metrics.json")


if __name__ == "__main__":
    run()
