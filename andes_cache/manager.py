"""High-level multi-layer cache manager for AndesCode workflows."""

from __future__ import annotations

from pathlib import Path

from .keys import build_key, normalize_query, stable_hash
from .metrics import CacheMetrics
from .store import DiskCacheStore
from .versions import (
    CACHE_SCHEMA_VERSION,
    PATCH_PLAN_VERSION,
    PROMPT_TEMPLATE_VERSION,
    RETRIEVAL_POLICY_VERSION,
    SEMANTIC_CACHE_VERSION,
)


class CacheLayers:
    WORKSPACE = "workspace"
    RETRIEVAL = "retrieval"
    FILE_NEIGHBORHOOD = "file_neighborhood"
    PROMPT_PREFIX = "prompt_prefix"
    PATCH_PLAN = "patch_plan"
    SEMANTIC = "semantic"


class AndesCacheManager:
    def __init__(self, base_dir: Path):
        self.store = DiskCacheStore(base_dir, CACHE_SCHEMA_VERSION)
        self.metrics = CacheMetrics(Path(base_dir) / "metrics" / "cache_metrics.json")

    # Workspace layer
    def workspace_get(self, repo_fp: str, artifact: str):
        key = build_key("ws", repo_fp=repo_fp, artifact=artifact)
        value = self.store.get(CacheLayers.WORKSPACE, key)
        if value is None:
            self.metrics.miss(CacheLayers.WORKSPACE)
        else:
            self.metrics.hit(CacheLayers.WORKSPACE)
        return value

    def workspace_set(self, repo_fp: str, artifact: str, value):
        key = build_key("ws", repo_fp=repo_fp, artifact=artifact)
        self.store.set(
            CacheLayers.WORKSPACE,
            key,
            value,
            metadata={"repo_fp": repo_fp, "artifact": artifact},
        )

    def invalidate_workspace_for_repo(self, repo_fp: str) -> int:
        removed = self.store.invalidate_where(
            CacheLayers.WORKSPACE,
            lambda p: p.get("metadata", {}).get("repo_fp") == repo_fp,
        )
        self.metrics.invalidate(CacheLayers.WORKSPACE, removed)
        return removed

    # Retrieval layer
    def retrieval_get(self, *, repo_fp: str, query: str, index_version: str):
        nquery = normalize_query(query)
        key = build_key(
            "ret",
            repo_fp=repo_fp,
            query=nquery,
            retrieval_policy_version=RETRIEVAL_POLICY_VERSION,
            index_version=index_version,
        )
        value = self.store.get(CacheLayers.RETRIEVAL, key)
        if value is None:
            self.metrics.miss(CacheLayers.RETRIEVAL)
        else:
            self.metrics.hit(CacheLayers.RETRIEVAL)
        return value

    def retrieval_set(self, *, repo_fp: str, query: str, index_version: str, value):
        nquery = normalize_query(query)
        key = build_key(
            "ret",
            repo_fp=repo_fp,
            query=nquery,
            retrieval_policy_version=RETRIEVAL_POLICY_VERSION,
            index_version=index_version,
        )
        self.store.set(
            CacheLayers.RETRIEVAL,
            key,
            value,
            metadata={"repo_fp": repo_fp, "query": nquery},
        )

    # File neighborhood
    def neighborhood_get(self, *, repo_fp: str, mode: str, anchor_file: str):
        key = build_key("nb", repo_fp=repo_fp, mode=mode, anchor_file=anchor_file)
        value = self.store.get(CacheLayers.FILE_NEIGHBORHOOD, key)
        if value is None:
            self.metrics.miss(CacheLayers.FILE_NEIGHBORHOOD)
        else:
            self.metrics.hit(CacheLayers.FILE_NEIGHBORHOOD)
        return value

    def neighborhood_set(self, *, repo_fp: str, mode: str, anchor_file: str, value):
        key = build_key("nb", repo_fp=repo_fp, mode=mode, anchor_file=anchor_file)
        self.store.set(
            CacheLayers.FILE_NEIGHBORHOOD,
            key,
            value,
            metadata={"repo_fp": repo_fp, "mode": mode, "anchor_file": anchor_file},
        )

    # Prompt-prefix
    def prompt_prefix_get(self, *, repo_fp: str, workspace_signature: str):
        key = build_key(
            "pp",
            repo_fp=repo_fp,
            workspace_signature=workspace_signature,
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
        )
        value = self.store.get(CacheLayers.PROMPT_PREFIX, key)
        if value is None:
            self.metrics.miss(CacheLayers.PROMPT_PREFIX)
        else:
            self.metrics.hit(CacheLayers.PROMPT_PREFIX)
        return value

    def prompt_prefix_set(self, *, repo_fp: str, workspace_signature: str, value):
        key = build_key(
            "pp",
            repo_fp=repo_fp,
            workspace_signature=workspace_signature,
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
        )
        self.store.set(
            CacheLayers.PROMPT_PREFIX,
            key,
            value,
            metadata={"repo_fp": repo_fp, "workspace_signature": workspace_signature},
        )

    # Patch plan
    def patch_plan_get(self, *, repo_fp: str, query: str, target_signature: str):
        key = build_key(
            "ppc",
            repo_fp=repo_fp,
            query=normalize_query(query),
            target_signature=target_signature,
            patch_plan_version=PATCH_PLAN_VERSION,
        )
        value = self.store.get(CacheLayers.PATCH_PLAN, key)
        if value is None:
            self.metrics.miss(CacheLayers.PATCH_PLAN)
        else:
            self.metrics.hit(CacheLayers.PATCH_PLAN)
        return value

    def patch_plan_set(self, *, repo_fp: str, query: str, target_signature: str, value):
        key = build_key(
            "ppc",
            repo_fp=repo_fp,
            query=normalize_query(query),
            target_signature=target_signature,
            patch_plan_version=PATCH_PLAN_VERSION,
        )
        self.store.set(
            CacheLayers.PATCH_PLAN,
            key,
            value,
            metadata={"repo_fp": repo_fp, "target_signature": target_signature},
        )

    # Scoped semantic cache
    def semantic_get(self, *, repo_fp: str, query: str, retrieval_signature: str, safe_class: str):
        key = build_key(
            "sem",
            repo_fp=repo_fp,
            query=normalize_query(query),
            retrieval_signature=retrieval_signature,
            safe_class=safe_class,
            semantic_cache_version=SEMANTIC_CACHE_VERSION,
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
        )
        value = self.store.get(CacheLayers.SEMANTIC, key)
        if value is None:
            self.metrics.miss(CacheLayers.SEMANTIC)
        else:
            self.metrics.hit(CacheLayers.SEMANTIC)
        return value

    def semantic_set(self, *, repo_fp: str, query: str, retrieval_signature: str, safe_class: str, value):
        key = build_key(
            "sem",
            repo_fp=repo_fp,
            query=normalize_query(query),
            retrieval_signature=retrieval_signature,
            safe_class=safe_class,
            semantic_cache_version=SEMANTIC_CACHE_VERSION,
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
        )
        self.store.set(
            CacheLayers.SEMANTIC,
            key,
            value,
            metadata={"repo_fp": repo_fp, "safe_class": safe_class},
        )

    def invalidate_repo(self, repo_fp: str, include_workspace: bool = False) -> dict:
        layers = [
            CacheLayers.RETRIEVAL,
            CacheLayers.FILE_NEIGHBORHOOD,
            CacheLayers.PROMPT_PREFIX,
            CacheLayers.PATCH_PLAN,
            CacheLayers.SEMANTIC,
        ]
        if include_workspace:
            layers.append(CacheLayers.WORKSPACE)

        removed = {}
        for layer in layers:
            count = self.store.invalidate_where(
                layer,
                lambda p: p.get("metadata", {}).get("repo_fp") == repo_fp,
            )
            removed[layer] = count
            self.metrics.invalidate(layer, count)
        return removed

    def workspace_signature(self, workspace: dict) -> str:
        return stable_hash(workspace)

    def flush_metrics(self):
        self.metrics.flush()
