import os
os.environ.setdefault("TRANSFORMERS_OFFLINE",              "1")
os.environ.setdefault("HF_DATASETS_OFFLINE",              "1")
os.environ.setdefault("HF_HUB_OFFLINE",                   "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM",            "false")

import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path
from typing import Generator

import chromadb
from sentence_transformers import SentenceTransformer

from andes_cache import AndesCacheManager, RepoFingerprinter
from andes_cache.routing import (
    classify_query_intent_details,
    RUNTIME_USAGE_OR_REFERENCE,
)
from andes_cache.source_of_truth import (
    config_priority_files,
    expected_authority_candidates,
    rank_recovery_authoritative_paths,
    score_authoritative_path,
    select_best_authoritative_path,
    summarize_declared_permissions,
    annotate_sources,
    missing_manifest_notice,
    classify_source_type,
    authority_level_for_source,
    wants_runtime_usage,
    is_declaration_query,
)
from andes_cache.integrity import (
    INTEGRITY_HEALTHY,
    IntegrityValidationMode,
    validate_authoritative_integrity_for_mode,
    deep_repair_integrity_validation,
    repair_authoritative_integrity,
    select_healthy_authoritative_path,
    prune_missing_on_disk_hashes,
    lightweight_integrity_probe,
)
from andes_cache.versions import (
    INDEX_VERSION,
    MODULE_DETECTION_VERSION,
    PARSER_VERSION,
    PROMPT_TEMPLATE_VERSION,
    RETRIEVAL_POLICY_VERSION,
    SOURCE_OF_TRUTH_VERSION,
    WORKSPACE_EXTRACTION_VERSION,
    WORKSPACE_SCHEMA_VERSION,
)
from andes_cache.debug import (
    resolve_debug_mode,
    initialize_payload,
    finalize_payload,
    apply_failure_signals,
    populate_retrieval_snapshot,
)

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL   = "all-MiniLM-L6-v2"
CHROMA_PATH   = str(Path(__file__).parent / "index")
COLLECTION    = "codebase"
HASH_STORE    = Path(__file__).parent / "index" / ".file_hashes.json"
PROJECT_MAP   = Path(__file__).parent / "index" / "project_map.json"
SYMBOL_INDEX  = Path(__file__).parent / "index" / "symbol_index.json"
WORKSPACE_INDEX = Path(__file__).parent / "index" / "workspace_index.json"
INDEX_STATE = Path(__file__).parent / "index" / "index_state.json"
INTEGRITY_STATE = Path(__file__).parent / "index" / "integrity_state.json"
CHUNK_COUNT_STATE = Path(__file__).parent / "index" / "chunk_count_state.json"
CACHE_DIR = Path(__file__).parent / "index" / "cache"
CACHE = AndesCacheManager(CACHE_DIR)
CURRENT_REPO_FINGERPRINT = ""
EMBED_BATCH   = 64
CHROMA_BATCH  = 4096
CHUNK_LINES   = 80
CHUNK_OVERLAP = 15

DECISION_REUSE_ALL = "reuse_all"
DECISION_REBUILD_WORKSPACE_ONLY = "rebuild_workspace_only"
DECISION_INCREMENTAL_REINDEX = "incremental_reindex"
DECISION_FULL_REBUILD = "full_rebuild"


@dataclass
class IntegrityRuntimeState:
    startup_probe: dict = field(default_factory=dict)
    refreshed_at: str = ""
    owner_root: str = ""


INTEGRITY_RUNTIME_STATE = IntegrityRuntimeState()
STARTUP_INTEGRITY_PROBE_MAX_AGE_SECONDS = 600

# ── Language boundary patterns ────────────────────────────────────────────────
_BOUNDARY = {
    "py":   re.compile(r"^(def |class |async def )", re.MULTILINE),
    "js":   re.compile(r"^(function |const \w+ ?= ?(async ?)?\(|class )", re.MULTILINE),
    "ts":   re.compile(r"^(function |const |class |interface |type |export )", re.MULTILINE),
    "jsx":  re.compile(r"^(function |const |class |export )", re.MULTILINE),
    "tsx":  re.compile(r"^(function |const |class |export |interface )", re.MULTILINE),
    "java": re.compile(r"^\s*(public|private|protected|static)[^\n]*\(", re.MULTILINE),
    "kt":   re.compile(r"^\s*(fun |class |object |interface )", re.MULTILINE),
    "swift":re.compile(r"^\s*(func |class |struct |protocol |extension )", re.MULTILINE),
    "go":   re.compile(r"^(func |type )", re.MULTILINE),
    "rs":   re.compile(r"^(fn |pub fn |impl |struct |enum |trait )", re.MULTILINE),
}

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".go", ".rs", ".java", ".cpp", ".c", ".h",
    ".rb", ".php", ".cs", ".swift", ".kt",
    ".gradle", ".xml",
}

MANIFEST_FILES = {
    "requirements.txt", "pyproject.toml", "poetry.lock", "Pipfile",
    "package.json", "pnpm-workspace.yaml", "yarn.lock",
    "build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts",
    "Cargo.toml", "go.mod", "pom.xml", "build.sbt",
    "Podfile", "Package.swift", "AndroidManifest.xml", "Dockerfile", "docker-compose.yml",
}

# Canonical authoritative file discovery patterns.
# Keep this easy to extend as new ecosystems/build systems are added.
AUTHORITATIVE_FILE_PATTERNS = [
    # Basenames from MANIFEST_FILES (supports both root and nested discovery).
    *[f"**/{name}" for name in sorted(MANIFEST_FILES)],
]

SKIP_DIRS = {
    ".git", ".svn", ".hg",
    "node_modules", ".next", ".nuxt", "dist", "build",
    "__pycache__", ".venv", "venv", ".pytest_cache", ".mypy_cache", "coverage",
    "vendor", "bundle", "Pods", "Carthage",
    ".gradle", ".idea", "target", "bin", "gen", "intermediates", "generated",
    "pkg", "DerivedData", "xcuserdata",
    "tmp", "temp", "cache", ".cache", "logs",
    ".build", "release", "debug",
}

# ── Setup ─────────────────────────────────────────────────────────────────────
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

print("🔍  Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)
chroma   = chromadb.PersistentClient(path=CHROMA_PATH)
col      = chroma.get_or_create_collection(COLLECTION)
print("✅  Indexer ready.")


# ── Public API ─────────────────────────────────────────────────────────────────

def index_codebase(root: str) -> dict:
    result = {}
    for event in index_codebase_stream(root):
        if event["type"] == "done":
            result = event
    return result


def index_codebase_stream(root: str) -> Generator[dict, None, None]:
    """
    Index a codebase with incremental support. Yields SSE progress events.
    On completion builds project map + symbol index for smart retrieval.
    """
    global col
    root_path = Path(root).resolve()

    if not root_path.exists():
        yield {"type": "error", "message": f"Path not found: {root_path}"}
        return

    # ── Collect and diff ──────────────────────────────────────────────────────
    hashes = _load_hashes()
    previous_repo_fp = hashes.get("__fingerprint__", "")
    all_files = _collect_files(root_path)
    new_files, unchanged = [], []
    changed_rel_paths = []

    for fp in all_files:
        key   = str(fp.relative_to(root_path))
        fhash = _file_hash(fp)
        if hashes.get(key) == fhash:
            unchanged.append(fp)
        else:
            new_files.append((fp, fhash))
            changed_rel_paths.append(key)

    known_files = {k for k in hashes.keys() if not k.startswith("__")}
    current_files = {str(fp.relative_to(root_path)) for fp in all_files}
    removed_paths = sorted(known_files - current_files)

    repo_changed = bool(new_files or removed_paths)

    # Recompute repo fingerprint from the current filesystem snapshot.
    next_hashes = {"__root__": str(root_path)}
    new_rel_set = {str(p.relative_to(root_path)) for p, _ in new_files}
    for fp in all_files:
        rel = str(fp.relative_to(root_path))
        if rel in new_rel_set:
            continue
        next_hashes[rel] = hashes.get(rel, _file_hash(fp))
    for fp, fhash in new_files:
        next_hashes[str(fp.relative_to(root_path))] = fhash

    repo_fp = RepoFingerprinter.build(
        root_path,
        next_hashes,
        index_version=INDEX_VERSION,
        parser_version=PARSER_VERSION,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
        retrieval_policy_version=RETRIEVAL_POLICY_VERSION,
    )
    _set_repo_fingerprint(repo_fp)
    current_state = _build_current_index_state(root_path, repo_fp)
    stored_state = _load_index_state()
    decision = evaluate_index_state(current_state, stored_state, repo_changed=repo_changed)
    if removed_paths and decision["decision"] != DECISION_FULL_REBUILD:
        decision = {
            "decision": DECISION_FULL_REBUILD,
            "reasons": ["Detected removed files; performing full rebuild to avoid stale vectors"],
        }

    if decision["decision"] == DECISION_FULL_REBUILD:
        for reason in decision["reasons"]:
            logging.info(reason)
            yield {"type": "decision", "level": DECISION_FULL_REBUILD, "message": reason}
        yield {"type": "clear"}
        try:
            chroma.delete_collection(COLLECTION)
        except Exception:
            pass
        col = chroma.get_or_create_collection(COLLECTION)
        hashes = {"__root__": str(root_path)}
        new_files = [(fp, _file_hash(fp)) for fp in all_files]
        changed_rel_paths = [str(fp.relative_to(root_path)) for fp, _ in new_files] + removed_paths
        unchanged = []
        for f in [PROJECT_MAP, SYMBOL_INDEX, WORKSPACE_INDEX]:
            try:
                f.unlink()
            except Exception:
                pass
        if previous_repo_fp:
            CACHE.invalidate_repo(previous_repo_fp, include_workspace=True)
        _save_chunk_count_state({})
    elif decision["decision"] == DECISION_REBUILD_WORKSPACE_ONLY:
        for reason in decision["reasons"]:
            logging.info(reason)
            yield {"type": "decision", "level": DECISION_REBUILD_WORKSPACE_ONLY, "message": reason}
        yield {"type": "clear"}
        CACHE.invalidate_workspace_for_repo(repo_fp)
        CACHE.invalidate_repo(repo_fp, include_workspace=False)
    elif decision["decision"] == DECISION_INCREMENTAL_REINDEX:
        for reason in decision["reasons"]:
            logging.info(reason)
            yield {"type": "decision", "level": DECISION_INCREMENTAL_REINDEX, "message": reason}
        yield {"type": "clear"}
    else:
        for reason in decision["reasons"]:
            logging.info(reason)
            yield {"type": "decision", "level": DECISION_REUSE_ALL, "message": reason}
        yield {"type": "clear"}

    yield {
        "type": "scan", "files": len(all_files),
        "new": len(new_files), "unchanged": len(unchanged),
    }

    if not all_files:
        yield {"type": "error", "message": "No source files found in that directory"}
        return

    if decision["decision"] == DECISION_REBUILD_WORKSPACE_ONLY:
        yield {"type": "mapping", "message": "Rebuilding workspace metadata only..."}
        # Use the full current filesystem snapshot, not just "unchanged" files.
        # Older index logic may have produced an incomplete indexed set; workspace-only
        # rebuilds must apply current detection/extraction logic to all files without
        # forcing re-embedding or vector-store writes.
        #
        # Invariant: discovered in workspace metadata != embedded in vector index.
        # Workspace-only rebuild must not promote newly discovered files into the
        # embedded hash state unless chunks were actually stored in Chroma.
        all_indexed = all_files
        workspace_chunks = []
        for fp in all_indexed:
            try:
                workspace_chunks.extend(_chunk_file(fp, root_path))
            except Exception as e:
                logging.warning(f"Skipped {fp} while rebuilding workspace metadata: {e}")
        workspace = build_workspace_index(
            root_path,
            all_indexed,
            workspace_chunks,
            repo_fingerprint=repo_fp,
            changed_paths=changed_rel_paths,
            force_refresh=True,
        )
        pmap = build_project_map(root_path, all_indexed, workspace_chunks, workspace)
        symidx = build_symbol_index(workspace_chunks)
        _save_json(PROJECT_MAP, pmap)
        _save_json(SYMBOL_INDEX, symidx)
        _save_json(WORKSPACE_INDEX, workspace)
        CACHE.workspace_set(repo_fp, "symbol_index", symidx)
        _save_hashes(_preserve_embedded_hash_state(hashes, root_path, repo_fp))
        _validate_and_repair_authoritative_integrity(root_path)
        _save_index_state(_with_timestamp(current_state, "workspace_rebuilt_at"))
        CACHE.flush_metrics()
        yield {
            "type": "done",
            "indexed": len(all_indexed),
            "chunks": col.count(),
            "project": root_path.name,
            "new": 0,
            "reused": len(unchanged),
            "map": pmap,
            "repo_fingerprint": repo_fp,
            "decision": DECISION_REBUILD_WORKSPACE_ONLY,
            "workspace_rebuild_scope": "all_files",
            "workspace_only_rebuild_preserved_embedding_state": True,
        }
        return

    if not new_files and unchanged:
        # Nothing changed — still emit map event so UI can display it
        pmap = _load_project_map()
        _save_hashes(next_hashes | {"__root__": str(root_path), "__fingerprint__": repo_fp})
        if removed_paths:
            chunk_counts = _load_chunk_count_state()
            for rel in removed_paths:
                chunk_counts.pop(rel, None)
            _save_chunk_count_state(chunk_counts)
        _validate_and_repair_authoritative_integrity(root_path)
        _save_index_state(_with_timestamp(current_state, "last_reused_at"))
        CACHE.flush_metrics()
        yield {
            "type": "done", "indexed": len(unchanged),
            "chunks": col.count(), "project": root_path.name,
            "new": 0, "reused": len(unchanged),
            "map": pmap,
            "repo_fingerprint": repo_fp,
            "decision": DECISION_REUSE_ALL,
        }
        return

    # ── Chunk ─────────────────────────────────────────────────────────────────
    all_chunks = []
    for fp, _ in new_files:
        try:
            all_chunks.extend(_chunk_file(fp, root_path))
        except Exception as e:
            logging.warning(f"Skipped {fp}: {e}")

    total_chunks = len(all_chunks)
    yield {"type": "chunks", "total": total_chunks}

    if not all_chunks:
        yield {"type": "error", "message": "No chunks generated — check file contents"}
        return

    # ── Embed ─────────────────────────────────────────────────────────────────
    all_embeddings = []
    for start in range(0, total_chunks, EMBED_BATCH):
        batch = all_chunks[start: start + EMBED_BATCH]
        vecs  = embedder.encode([c["content"] for c in batch],
                                show_progress_bar=False).tolist()
        all_embeddings.extend(vecs)
        done = min(start + EMBED_BATCH, total_chunks)
        yield {"type": "embed", "done": done, "total": total_chunks,
               "pct": int(done / total_chunks * 100)}

    # ── Store ─────────────────────────────────────────────────────────────────
    stored = 0
    for start in range(0, total_chunks, CHROMA_BATCH):
        bc = all_chunks[start: start + CHROMA_BATCH]
        bv = all_embeddings[start: start + CHROMA_BATCH]
        col.upsert(
            ids        = [c["id"]      for c in bc],
            embeddings = bv,
            documents  = [c["content"] for c in bc],
            metadatas  = [{"file": c["file"], "language": c["language"],
                           "line": c["line"], "symbols": c.get("symbols", "")}
                          for c in bc],
        )
        stored += len(bc)
        yield {"type": "store", "done": stored, "total": total_chunks,
               "pct": int(stored / total_chunks * 100)}

    # ── Save hashes and invalidate dependent cache layers ────────────────────
    if previous_repo_fp and previous_repo_fp != repo_fp:
        CACHE.invalidate_repo(previous_repo_fp, include_workspace=False)

    for fp, fhash in new_files:
        hashes[str(fp.relative_to(root_path))] = fhash
    _save_hashes(next_hashes | {"__root__": str(root_path), "__fingerprint__": repo_fp})
    chunk_counts = _load_chunk_count_state()
    for rel in removed_paths:
        chunk_counts.pop(rel, None)
    chunk_counts.update({path: count for path, count in _chunk_counts_by_path(all_chunks).items()})
    _save_chunk_count_state(chunk_counts)

    # ── Build project intelligence ────────────────────────────────────────────
    yield {"type": "mapping", "message": "Building project map..."}

    all_indexed = [fp for fp, _ in new_files] + unchanged
    workspace = build_workspace_index(
        root_path,
        all_indexed,
        all_chunks,
        repo_fingerprint=repo_fp,
        changed_paths=changed_rel_paths,
    )
    pmap      = build_project_map(root_path, all_indexed, all_chunks, workspace)
    symidx    = build_symbol_index(all_chunks)

    _save_json(PROJECT_MAP, pmap)
    _save_json(SYMBOL_INDEX, symidx)
    _save_json(WORKSPACE_INDEX, workspace)
    CACHE.workspace_set(repo_fp, "symbol_index", symidx)
    _validate_and_repair_authoritative_integrity(root_path)
    _save_index_state(_with_timestamp(current_state, "last_indexed_at"))
    CACHE.flush_metrics()

    yield {
        "type":    "done",
        "indexed": len({c["file"] for c in all_chunks}) + len(unchanged),
        "chunks":  col.count(),
        "project": root_path.name,
        "new":     len(new_files),
        "reused":  len(unchanged),
        "map":     pmap,
        "repo_fingerprint": repo_fp,
        "decision": DECISION_INCREMENTAL_REINDEX if repo_changed else DECISION_REUSE_ALL,
    }


def search(
    query: str,
    n_results: int = 5,
    debug_mode: bool | None = None,
    debug_payload: dict | None = None,
    return_debug: bool = False,
) -> list[dict] | tuple[list[dict], dict | None]:
    """
    Smart retrieval with query routing and re-ranking.

    Routes:
      - Filename mentioned   → fetch ALL chunks from that file
      - Symbol name match    → boost chunks from that symbol's file
      - Architectural query  → fetch more candidates (8 instead of 5)
      - Default              → semantic search + re-ranking
    """
    debug_enabled = resolve_debug_mode(param_flag=debug_mode)
    payload_out = None

    def _ret(results: list[dict], payload: dict | None = None):
        if return_debug:
            return results, payload
        return results
    count = col.count()
    if count == 0:
        if debug_enabled:
            decision = classify_query_intent_details(query)
            payload = debug_payload or initialize_payload(query, decision, _load_workspace_index())
            payload_out = finalize_payload(payload, [])
            payload_out = apply_failure_signals(
                payload_out,
                query=query,
                intent=decision["intent"],
                retrieval_route=decision["retrieval_route"],
                top_score=None,
                final_chunks=[],
            )
            return _ret([], payload_out)
        return _ret([])
    decision = classify_query_intent_details(query)
    intent = decision["intent"]
    retrieval_route = decision["retrieval_route"]
    workspace_index = _load_workspace_index()
    payload = debug_payload or (initialize_payload(query, decision, workspace_index) if debug_enabled else None)
    authoritative_files_detected = (
        sorted(
            {
                *(workspace_index.get("manifests", []) or []),
                *((workspace_index.get("config_graph", {}) or {}).get("config_files", []) or []),
            }
        )
        if isinstance(workspace_index, dict)
        else []
    )
    decl_query = is_declaration_query(query, intent)
    authoritative_files_required = list(authoritative_files_detected) if decl_query else []

    def _matches_authoritative(path: str) -> bool:
        if path in authoritative_files_required:
            return True
        base = Path(path).name
        return any(Path(p).name == base for p in authoritative_files_required)

    def _chunk_identity(chunk: dict) -> tuple[str, int, str]:
        file_path = str(chunk.get("file", "") or "")
        line = int(chunk.get("line", 0) or 0)
        content = str(chunk.get("content", "") or "")
        return (file_path, line, content)

    def _update_authoritative_debug(final_chunks: list[dict], *, mode: str = "", reason: str = "") -> None:
        if payload is None:
            return
        retrieved = sorted(
            {
                c.get("file", "")
                for c in (final_chunks or [])
                if c.get("file") and _matches_authoritative(c.get("file", ""))
            }
        )
        payload["retrieval"]["authoritative_files_detected"] = list(authoritative_files_detected)
        payload["retrieval"]["authoritative_files_required"] = list(authoritative_files_required)
        payload["retrieval"]["authoritative_files_retrieved"] = retrieved
        payload["retrieval"]["authoritative_files_missing"] = [p for p in authoritative_files_required if p not in retrieved]
        payload["retrieval"]["forced_authoritative_file"] = bool(retrieved)
        if reason:
            payload["retrieval"]["authority_selection_reason"] = reason
        if mode:
            payload["retrieval"]["authority_retrieval_mode"] = mode
        if decl_query:
            missing = payload["retrieval"]["authoritative_files_missing"]
            has_runtime_or_inferred = any(
                c.get("file")
                and not _matches_authoritative(c.get("file", ""))
                and c.get("source_type") in {"source_code", "inferred"}
                for c in (final_chunks or [])
            )
            if retrieved and not missing and not has_runtime_or_inferred:
                payload["retrieval"]["declaration_answer_mode"] = "declared_only"
            elif retrieved and has_runtime_or_inferred:
                payload["retrieval"]["declaration_answer_mode"] = "declared_plus_runtime"
            elif retrieved and missing and not has_runtime_or_inferred:
                payload["retrieval"]["declaration_answer_mode"] = "declared_partial_only"
            elif payload["retrieval"]["authority_retrieval_mode"] == "runtime_fallback_used":
                payload["retrieval"]["declaration_answer_mode"] = "runtime_only_fallback"
            else:
                payload["retrieval"]["declaration_answer_mode"] = "missing_declarations"

    if payload is not None:
        payload["retrieval"]["route_taken"] = retrieval_route
        _update_authoritative_debug([], mode="semantic_fallback_blocked")
    repo_fp = get_repo_fingerprint()
    if repo_fp:
        cached = CACHE.retrieval_get(
            repo_fp=repo_fp,
            query=query,
            index_version=INDEX_VERSION,
            intent=intent,
            retrieval_route=retrieval_route,
        )
        if cached:
            cached_final = list(cached)
            authoritative_source_types = {"manifest", "build_file", "dependency_file", "config_file"}
            authoritative_cached = [
                c
                for c in cached_final
                if c.get("source_type") in authoritative_source_types
                or _matches_authoritative(c.get("file", ""))
            ]
            preserve_authoritative_overflow = (
                decl_query
                and len(cached_final) > n_results
                and len(authoritative_cached) > n_results
            )
            cached_visible = cached_final if preserve_authoritative_overflow else cached_final[:n_results]
            if payload is not None:
                payload = populate_retrieval_snapshot(
                    payload,
                    chunks=cached_visible,
                    raw_candidates=[c.get("file", "") for c in cached_visible],
                    cache_hit=True,
                )
                _update_authoritative_debug(cached_visible, mode="direct_chunk_load", reason="cache hit")
                payload_out = finalize_payload(payload, cached_visible)
                payload_out = apply_failure_signals(
                    payload_out,
                    query=query,
                    intent=intent,
                    retrieval_route=retrieval_route,
                    top_score=None,
                    final_chunks=cached_visible,
                )
                return _ret(cached_visible, payload_out)
            return _ret(cached_visible)

    if retrieval_route == "source_of_truth":
        final = _retrieve_config_first(
            query,
            intent,
            n_results=n_results,
            ambiguous=decision.get("ambiguous", False),
            allow_runtime_fallback=decision.get("allow_runtime_fallback", False),
            strict_authority_mode=decision.get("strict_authority_mode", True),
            debug_payload=payload,
        )
        if repo_fp and final:
            CACHE.retrieval_set(
                repo_fp=repo_fp,
                query=query,
                index_version=INDEX_VERSION,
                value=final,
                intent=intent,
                retrieval_route=retrieval_route,
            )
        if payload is not None:
            payload = populate_retrieval_snapshot(
                payload,
                chunks=final,
                raw_candidates=payload.get("source_of_truth", {}).get("priority_files", []),
            )
            mode = (
                "workspace_only_detected_not_indexed"
                if any(c.get("file") == "__source_of_truth_integrity__" for c in final)
                else "direct_chunk_load"
            )
            _update_authoritative_debug(
                final,
                mode=mode,
                reason="source_of_truth route deterministic authoritative retrieval",
            )
            payload_out = finalize_payload(payload, final)
            payload_out = apply_failure_signals(
                payload_out,
                query=query,
                intent=intent,
                retrieval_route=retrieval_route,
                top_score=None,
                final_chunks=final,
            )
        return _ret(final, payload_out)

    # ── Route 1: Filename detected ────────────────────────────────────────────
    file_match = re.search(
        r'\b[\w/]+\.(py|js|ts|jsx|tsx|java|kt|swift|go|rs|cpp|c|rb|cs)\b',
        query, re.IGNORECASE
    )
    if file_match:
        fname   = file_match.group(0).lower()
        results = _fetch_all_from_file(fname, n_results)
        if results:
            final = _add_coverage(results)
            final = annotate_sources(final, source_type="source_code", authority_level="referenced")
            if repo_fp:
                CACHE.retrieval_set(
                    repo_fp=repo_fp,
                    query=query,
                    index_version=INDEX_VERSION,
                    value=final,
                    intent=intent,
                    retrieval_route=retrieval_route,
                )
            if payload is not None:
                payload = populate_retrieval_snapshot(
                    payload,
                    chunks=final,
                    raw_candidates=[c.get("file", "") for c in results],
                )
                _update_authoritative_debug(final)
                payload_out = finalize_payload(payload, final)
                payload_out = apply_failure_signals(
                    payload_out,
                    query=query,
                    intent=intent,
                    retrieval_route=retrieval_route,
                    top_score=None,
                    final_chunks=final,
                )
            return _ret(final, payload_out)

    # ── Route 1b: Structured workspace questions ─────────────────────────────
    structured = _structured_query_results(query)
    if structured:
        final = structured[:n_results]
        if repo_fp:
            CACHE.retrieval_set(
                repo_fp=repo_fp,
                query=query,
                index_version=INDEX_VERSION,
                value=final,
                intent=intent,
                retrieval_route=retrieval_route,
            )
        if payload is not None:
            payload = populate_retrieval_snapshot(
                payload,
                chunks=final,
                raw_candidates=[c.get("file", "") for c in structured],
            )
            _update_authoritative_debug(final)
            payload_out = finalize_payload(payload, final)
            payload_out = apply_failure_signals(
                payload_out,
                query=query,
                intent=intent,
                retrieval_route=retrieval_route,
                top_score=None,
                final_chunks=final,
            )
        return _ret(final, payload_out)

    # ── Route 2: Symbol name lookup ───────────────────────────────────────────
    symidx = _load_json(SYMBOL_INDEX)
    if symidx:
        query_words = set(re.findall(r"\w+", query))
        matched_files = set()
        for word in query_words:
            if word in symidx:
                matched_files.update(symidx[word])
        if matched_files:
            # Boost n_results for symbol queries — likely need more context
            n_results = max(n_results, 6)

    # ── Route 3: Architectural query ──────────────────────────────────────────
    arch_patterns = re.compile(
        r'\b(how does|how is|where is|where are|explain|overview|'
        r'architecture|structure|flow|pipeline|what is the|'
        r'how do you|entry point|main|startup)\b',
        re.IGNORECASE
    )
    if arch_patterns.search(query):
        n_results = max(n_results, 8)

    # ── Semantic search + re-ranking ──────────────────────────────────────────
    embedding = embedder.encode(query).tolist()
    fetch     = min(n_results * 6, count)   # wider candidate pool → better rerank
    results   = col.query(query_embeddings=[embedding], n_results=fetch)

    candidates = [
        {
            "content":  doc,
            "file":     results["metadatas"][0][i].get("file", ""),
            "language": results["metadatas"][0][i].get("language", ""),
            "line":     results["metadatas"][0][i].get("line", 0),
            "symbols":  results["metadatas"][0][i].get("symbols", ""),
            "score":    results["distances"][0][i],
        }
        for i, doc in enumerate(results["documents"][0])
    ]
    authoritative_forced_candidates = []
    authoritative_missing = []
    if decl_query and authoritative_files_required:
        for authoritative_path in authoritative_files_required:
            file_chunks = _fetch_exact_file(authoritative_path, max_results=6)
            if not file_chunks:
                basename_candidates = sorted(
                    _fetch_indexed_candidates_by_basename(authoritative_path, limit=120).keys()
                )
                logging.warning(
                    "AUTHORITATIVE_FETCH_EMPTY requested=%s candidates=%s",
                    authoritative_path,
                    basename_candidates,
                )
                authoritative_missing.append(authoritative_path)
                continue
            source_type = classify_source_type(authoritative_path)
            authority = authority_level_for_source(intent, source_type)
            annotate_sources(file_chunks, source_type=source_type, authority_level=authority)
            for ch in file_chunks:
                authoritative_forced_candidates.append(
                    {
                        "content": ch.get("content", ""),
                        "file": ch.get("file", authoritative_path),
                        "language": ch.get("language", ""),
                        "line": ch.get("line", 0),
                        "symbols": ch.get("symbols", ""),
                        "score": -1.0,
                    }
                )

    # Boost candidates from symbol-matched files
    if symidx and matched_files:
        for c in candidates:
            if any(mf in c["file"] for mf in matched_files):
                c["score"] *= 0.8   # lower distance = better match

    ranked = _rerank(query, candidates, track_reasons=payload is not None)
    final_ranked = ranked[:n_results]
    if decl_query and authoritative_files_required:
        if not authoritative_forced_candidates and authoritative_missing:
            limitation = {
                "content": (
                    "# Source-of-Truth Limitation\n"
                    "- Authoritative file detected but not retrievable from index.\n"
                    "- Declared/configured facts cannot be confirmed from authoritative artifacts.\n"
                ),
                "file": "__source_of_truth_missing__",
                "language": "meta",
                "symbols": "",
                "score": 0.0,
                "_rank": 1.0,
                "full_file": True,
                "coverage": {"returned": 1, "total": 1, "partial": False},
                "source_type": "inferred",
                "authority_level": "inferred",
            }
            final = [limitation]
            final = _add_coverage(final)
            final = annotate_sources(final, source_type="inferred", authority_level="inferred")
            if payload is not None:
                payload = populate_retrieval_snapshot(
                    payload,
                    chunks=final,
                    raw_candidates=[c.get("file", "") for c in candidates],
                )
                _update_authoritative_debug(
                    final,
                    mode="workspace_only_detected_not_indexed",
                    reason="authoritative file detected in workspace metadata but no chunks in index",
                )
                payload_out = finalize_payload(payload, final)
                payload_out = apply_failure_signals(
                    payload_out,
                    query=query,
                    intent=intent,
                    retrieval_route=retrieval_route,
                    top_score=None,
                    final_chunks=final,
                )
            return _ret(final, payload_out)
        authoritative_chunks_injected: set[tuple[str, int, str]] = set()
        merged = []
        for c in authoritative_forced_candidates:
            key = _chunk_identity(c)
            if key in authoritative_chunks_injected:
                continue
            authoritative_chunks_injected.add(key)
            merged.append(c)
        target_size = max(n_results, len(authoritative_chunks_injected))
        for c in final_ranked:
            if len(merged) >= target_size:
                break
            if _chunk_identity(c) in authoritative_chunks_injected:
                continue
            merged.append(c)
        final_ranked = merged

    # Add coverage metadata — lets server tell model how much of each file it has
    final = _add_coverage(final_ranked)
    if decl_query and authoritative_files_required:
        for chunk in final:
            fpath = chunk.get("file", "")
            if _matches_authoritative(fpath):
                s_type = classify_source_type(fpath)
                chunk["source_type"] = s_type
                chunk["authority_level"] = authority_level_for_source(intent, s_type)
            else:
                chunk["source_type"] = "source_code"
                chunk["authority_level"] = "inferred"
    else:
        authority = "referenced" if intent == RUNTIME_USAGE_OR_REFERENCE else "inferred"
        final = annotate_sources(final, source_type="source_code", authority_level=authority)
    if repo_fp:
        CACHE.retrieval_set(
            repo_fp=repo_fp,
            query=query,
            index_version=INDEX_VERSION,
            value=final,
            intent=intent,
            retrieval_route=retrieval_route,
        )
    if payload is not None:
        payload = populate_retrieval_snapshot(
            payload,
            chunks=final,
            raw_candidates=[c.get("file", "") for c in candidates],
        )
        _update_authoritative_debug(
            final,
            mode=("direct_chunk_load" if decl_query and authoritative_files_required else "runtime_fallback_used"),
            reason=("authoritative chunks force-included before semantic merge" if decl_query and authoritative_files_required else "semantic retrieval"),
        )
        top = []
        for rank, c in enumerate(ranked[: min(5, len(ranked))], start=1):
            top.append({
                "file": c.get("file", ""),
                "score": round(c.get("_rank", 0.0), 5),
                "rank": rank,
                "reason": c.get("_debug_reason", "semantic"),
            })
        payload["ranking"]["top_candidates"] = top
        payload_out = finalize_payload(payload, final)
        payload_out = apply_failure_signals(
            payload_out,
            query=query,
            intent=intent,
            retrieval_route=retrieval_route,
            top_score=(top[0]["score"] if top else None),
            final_chunks=final,
        )
    return _ret(final, payload_out)


def inspect_query_debug(query: str, n_results: int = 5, debug_mode: bool = True) -> dict:
    payload = initialize_payload(query, classify_query_intent_details(query), _load_workspace_index())
    _, dbg = search(
        query,
        n_results=n_results,
        debug_mode=debug_mode,
        debug_payload=payload,
        return_debug=True,
    )
    return dbg or payload


def explain_retrieval_decision(query: str, n_results: int = 5) -> dict:
    return inspect_query_debug(query, n_results=n_results, debug_mode=True).get("retrieval", {})


def explain_source_of_truth(query: str, n_results: int = 5) -> dict:
    return inspect_query_debug(query, n_results=n_results, debug_mode=True).get("source_of_truth", {})


def explain_missing_authority(query: str, n_results: int = 5) -> dict:
    payload = inspect_query_debug(query, n_results=n_results, debug_mode=True)
    return {
        "query": query,
        "missing_expected": payload.get("source_of_truth", {}).get("missing_expected", []),
        "expected_but_missing_authority": payload.get("failure_signals", {}).get("expected_but_missing_authority", False),
    }


def get_chunks_for_file(filename: str) -> list[dict]:
    """Retrieve ALL indexed chunks for a specific file."""
    return _fetch_all_from_file(filename, max_results=999)


def build_project_map(root_path: Path, files: list, chunks: list, workspace: dict | None = None) -> dict:
    """
    Analyse the project structure and build a structured summary.
    This gets injected into every system prompt so the model always
    knows what project it's working with.
    """
    # ── Language detection ────────────────────────────────────────────────────
    lang_counts = defaultdict(int)
    for fp in files:
        lang_counts[fp.suffix] += 1
    primary_ext  = max(lang_counts, key=lang_counts.get) if lang_counts else ""
    EXT_TO_LANG  = {
        ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
        ".kt": "Kotlin", ".java": "Java", ".swift": "Swift",
        ".go": "Go", ".rs": "Rust", ".cpp": "C++", ".cs": "C#",
    }
    primary_lang = EXT_TO_LANG.get(primary_ext, primary_ext.lstrip(".").upper())

    # ── Framework / stack detection ───────────────────────────────────────────
    root_files = {f.name for f in root_path.iterdir() if f.is_file()} \
                 if root_path.exists() else set()

    stack = []
    if "requirements.txt" in root_files or "pyproject.toml" in root_files:
        stack.append("Python")
        reqs = ""
        for rf in ["requirements.txt", "pyproject.toml"]:
            try:
                reqs = (root_path / rf).read_text(errors="ignore").lower()
                break
            except Exception:
                pass
        if "fastapi" in reqs or "flask" in reqs or "django" in reqs:
            stack.append("Web API")
        if "torch" in reqs or "tensorflow" in reqs:
            stack.append("ML/Deep Learning")
        if "pandas" in reqs or "numpy" in reqs:
            stack.append("Data Science")
        if "binance" in reqs or "ccxt" in reqs:
            stack.append("Crypto Trading")
        if "scikit" in reqs or "sklearn" in reqs:
            stack.append("Machine Learning")

    if "build.gradle" in root_files or "build.gradle.kts" in root_files:
        stack.append("Android")
    if "package.json" in root_files:
        try:
            pkg = json.loads((root_path / "package.json").read_text(errors="ignore"))
            deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
            if "react" in deps:     stack.append("React")
            if "next" in deps:      stack.append("Next.js")
            if "vue" in deps:       stack.append("Vue")
            if "express" in deps:   stack.append("Express")
        except Exception:
            stack.append("Node.js")
    if "Podfile" in root_files or any(f.endswith(".xcodeproj") for f in root_files):
        stack.append("iOS")
    if "Cargo.toml" in root_files:
        stack.append("Rust")
    if "go.mod" in root_files:
        stack.append("Go")

    # ── Entry point detection ─────────────────────────────────────────────────
    entry_candidates = [
        "main.py", "app.py", "server.py", "run.py", "manage.py", "__main__.py",
        "index.js", "app.js", "server.js", "main.js",
        "index.ts", "app.ts", "main.ts",
        "Main.java", "Application.java",
        "main.go", "main.rs", "main.swift",
    ]
    entry_point = None
    file_names  = {fp.name: fp for fp in files}
    for candidate in entry_candidates:
        if candidate in file_names:
            entry_point = str(file_names[candidate].relative_to(root_path))
            break

    # ── Module list (top-level source files) ──────────────────────────────────
    # Group by directory to give a clean module view
    module_dirs = defaultdict(list)
    for fp in files:
        rel  = fp.relative_to(root_path)
        parts = rel.parts
        if len(parts) == 1:
            module_dirs["root"].append(fp.name)
        else:
            module_dirs[parts[0]].append("/".join(parts[1:]))

    # Keep top 15 most populated directories
    top_modules = dict(
        sorted(module_dirs.items(), key=lambda x: -len(x[1]))[:15]
    )

    # ── Domain detection from source content ──────────────────────────────────
    domain_keywords = {
        "crypto/trading": ["btcusdt", "binance", "cryptocurrency", "trading", "candlestick",
                           "ohlcv", "binance", "coinbase", "kraken"],
        "web backend":    ["endpoint", "router", "middleware", "request", "response",
                           "http", "rest", "graphql"],
        "mobile/android": ["activity", "fragment", "viewmodel", "recyclerview",
                           "manifest", "gradle"],
        "mobile/ios":     ["uiviewcontroller", "swiftui", "storyboard", "appdelegate"],
        "data science":   ["dataframe", "numpy", "pandas", "sklearn", "model.fit"],
        "ML/AI":          ["neural", "epoch", "loss", "gradient", "tensor"],
        "devops/infra":   ["dockerfile", "kubernetes", "terraform", "deploy"],
    }

    domain_scores = defaultdict(int)
    # Sample first 30 files to detect domain
    sample_files = files[:30]
    for fp in sample_files:
        try:
            text = fp.read_text(errors="ignore").lower()[:3000]
            for domain, keywords in domain_keywords.items():
                domain_scores[domain] += sum(1 for kw in keywords if kw in text)
        except Exception:
            pass

    detected_domain = max(domain_scores, key=domain_scores.get) \
                      if any(v > 0 for v in domain_scores.values()) else None

    # ── Key symbols per file ──────────────────────────────────────────────────
    # Build a compact function/class map: {file: [symbol1, symbol2, ...]}
    file_symbols = defaultdict(list)
    for c in chunks:
        if c.get("symbols"):
            syms = c["symbols"].split()
            file_symbols[c["file"]].extend(syms)

    # Deduplicate and keep top files by symbol count
    file_symbol_map = {
        f: list(dict.fromkeys(syms))[:10]   # preserve order, cap at 10
        for f, syms in sorted(file_symbols.items(), key=lambda x: -len(x[1]))[:20]
    }

    workspace = workspace or {}

    return {
        "project":      root_path.name,
        "language":     primary_lang,
        "stack":        stack,
        "entry_point":  entry_point,
        "file_count":   len(files),
        "modules":      top_modules,
        "domain":       detected_domain,
        "file_symbols": file_symbol_map,
        "workspace": {
            "repo_types": workspace.get("repo_types", []),
            "package_managers": workspace.get("package_managers", []),
            "manifests": workspace.get("manifests", []),
            "entry_points": workspace.get("entry_points", [])[:15],
            "module_count": len(workspace.get("modules", [])),
        },
    }


def build_workspace_index(
    root_path: Path,
    files: list,
    chunks: list,
    repo_fingerprint: str = "",
    changed_paths: list[str] | None = None,
    force_refresh: bool = False,
) -> dict:
    """
    Build structure-first workspace intelligence for dependency/config analysis.

    Workspace artifacts are cached independently so unchanged artifacts are reused.
    """
    changed_paths = changed_paths or []
    repo_fp = repo_fingerprint or get_repo_fingerprint()

    cached_manifests = None if force_refresh else CACHE.workspace_get(repo_fp, "manifests")
    manifests = cached_manifests if cached_manifests is not None else _discover_manifests(root_path)
    CACHE.workspace_set(repo_fp, "manifests", manifests)

    cached_modules = None if force_refresh else CACHE.workspace_get(repo_fp, "module_graph")
    modules = cached_modules if cached_modules is not None else _discover_modules(root_path, files)
    CACHE.workspace_set(repo_fp, "module_graph", modules)

    has_manifest_change = any(Path(p).name in MANIFEST_FILES for p in changed_paths)
    has_code_change = any(Path(p).suffix in SUPPORTED_EXTENSIONS for p in changed_paths)

    dependencies = None if force_refresh else CACHE.workspace_get(repo_fp, "dependency_inventory")
    if dependencies is None or has_manifest_change:
        dependencies = _collect_declared_dependencies(root_path, manifests)
        CACHE.workspace_set(repo_fp, "dependency_inventory", dependencies)

    import_graph = None if force_refresh else CACHE.workspace_get(repo_fp, "import_graph")
    if import_graph is None or has_code_change:
        import_graph = _build_import_graph(files, root_path)
        CACHE.workspace_set(repo_fp, "import_graph", import_graph)

    config_graph = None if force_refresh else CACHE.workspace_get(repo_fp, "config_graph")
    if config_graph is None or has_manifest_change:
        config_graph = _build_config_graph(root_path, manifests)
        CACHE.workspace_set(repo_fp, "config_graph", config_graph)

    entry_points = None if force_refresh else CACHE.workspace_get(repo_fp, "entry_points")
    if entry_points is None or has_code_change:
        entry_points = _discover_entry_points(root_path, files)
        CACHE.workspace_set(repo_fp, "entry_points", entry_points)

    file_to_module = None if force_refresh else CACHE.workspace_get(repo_fp, "file_to_module_map")
    if file_to_module is None or has_code_change:
        file_to_module = _build_file_to_module_map(root_path, files)
        CACHE.workspace_set(repo_fp, "file_to_module_map", file_to_module)

    return {
        "project": root_path.name,
        "repo_types": _detect_repo_types(manifests),
        "package_managers": _detect_package_managers(manifests),
        "modules": modules,
        "manifests": manifests,
        "entry_points": entry_points,
        "dependencies": dependencies,
        "import_graph": import_graph,
        "config_graph": config_graph,
        "file_to_module_map": file_to_module,
    }


def build_symbol_index(chunks: list) -> dict:
    """
    Build a symbol-to-file lookup: {"build_insights": ["market_insights.py"], ...}
    Used for smart retrieval when query mentions a function/class name.
    """
    index = defaultdict(set)
    for c in chunks:
        if c.get("symbols"):
            for sym in c["symbols"].split():
                index[sym].add(c["file"])
    return {k: list(v) for k, v in index.items()}


def _structured_query_results(query: str) -> list[dict]:
    """
    Return synthetic results for dependency/config/module questions.
    This allows structure-first answers without relying on semantic chunk matches.
    """
    q = query.lower()
    ws = _load_workspace_index()
    if not ws:
        return []

    def _as_chunk(title: str, lines: list[str]) -> dict:
        text = f"# {title}\n" + "\n".join(lines)
        return {
            "content": text,
            "file": "__workspace_index__",
            "language": "meta",
            "symbols": "",
            "score": 0.0,
            "_rank": 1.0,
            "full_file": True,
            "coverage": {"returned": 1, "total": 1, "partial": False},
        }

    dep_intent = re.search(r"\b(dependenc|library|libraries|package|uses?|framework)\b", q)
    cfg_intent = re.search(r"\b(config|manifest|permission|capabilit|build|gradle|docker|env)\b", q)
    mod_intent = re.search(r"\b(module|monorepo|workspace|service|package|architecture|entry)\b", q)

    chunks = []
    if dep_intent:
        deps = ws.get("dependencies", {})
        lines = []
        for eco, items in deps.items():
            if items:
                lines.append(f"- {eco}: {', '.join(items[:25])}")
        if lines:
            chunks.append(_as_chunk("Declared Dependencies", lines))

    if cfg_intent:
        cfg = ws.get("config_graph", {})
        lines = []
        if cfg.get("build_systems"):
            lines.append(f"- Build systems: {', '.join(cfg['build_systems'])}")
        if cfg.get("capabilities"):
            lines.append(f"- Capabilities/permissions: {', '.join(cfg['capabilities'][:20])}")
        if cfg.get("config_files"):
            lines.append(f"- Config files: {', '.join(cfg['config_files'][:20])}")
        if lines:
            chunks.append(_as_chunk("Build & Config Graph", lines))

    if mod_intent:
        lines = []
        mods = ws.get("modules", [])
        if mods:
            lines.append("- Modules:")
            for m in mods[:15]:
                lines.append(f"  - {m['name']} ({m['kind']}) files={m['file_count']}")
        if ws.get("entry_points"):
            lines.append("- Entry points: " + ", ".join(ws["entry_points"][:20]))
        if lines:
            chunks.append(_as_chunk("Workspace Modules", lines))

    return chunks


def format_project_map_for_prompt(pmap: dict) -> str:
    """
    Format project map as a compact header for the system prompt.
    Keeps it under ~300 tokens so it doesn't crowd out code context.
    """
    if not pmap:
        return ""

    lines = [
        "## Project Context",
        f"Name: {pmap.get('project', 'unknown')}",
        f"Language: {pmap.get('language', 'unknown')}",
    ]

    if pmap.get("stack"):
        lines.append(f"Stack: {', '.join(pmap['stack'])}")
    if pmap.get("domain"):
        lines.append(f"Domain: {pmap['domain']}")
    if pmap.get("entry_point"):
        lines.append(f"Entry point: {pmap['entry_point']}")
    workspace = pmap.get("workspace", {})
    if workspace.get("repo_types"):
        lines.append(f"Repo types: {', '.join(workspace['repo_types'])}")
    if workspace.get("package_managers"):
        lines.append(f"Package managers: {', '.join(workspace['package_managers'])}")
    if workspace.get("entry_points"):
        lines.append("Detected entry points: " + ", ".join(workspace["entry_points"][:6]))

    # Key modules with their symbols
    fsyms = pmap.get("file_symbols", {})
    if fsyms:
        lines.append("Key modules:")
        for fname, syms in list(fsyms.items())[:10]:
            sym_str = ", ".join(syms[:6])
            lines.append(f"  - {fname}: {sym_str}")

    return "\n".join(lines)


def _load_project_map() -> dict:
    return _load_json(PROJECT_MAP)


def _load_workspace_index() -> dict:
    return _load_json(WORKSPACE_INDEX)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _discover_manifests(root_path: Path) -> list[str]:
    """
    Discover authoritative project files (manifests/build/config declarations).

    Backward-compatible: returned values continue to populate workspace["manifests"].
    """
    authoritative_files = set()
    for pattern in AUTHORITATIVE_FILE_PATTERNS:
        for p in root_path.glob(pattern):
            if not p.is_file():
                continue
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            authoritative_files.add(str(p.relative_to(root_path)))
    return sorted(authoritative_files)


def _detect_repo_types(manifests: list[str]) -> list[str]:
    types = set()
    joined = " ".join(manifests).lower()
    if any(x in joined for x in ("requirements.txt", "pyproject.toml", "pipfile")):
        types.add("python")
    if "package.json" in joined:
        types.add("node")
    if any(x in joined for x in ("build.gradle", "pom.xml")):
        types.add("jvm")
    if "go.mod" in joined:
        types.add("go")
    if "cargo.toml" in joined:
        types.add("rust")
    if any(x in joined for x in ("androidmanifest.xml", "build.gradle")):
        types.add("android")
    if any(x in joined for x in ("podfile", "package.swift")):
        types.add("ios")
    if len(types) > 1:
        types.add("polyglot")
    return sorted(types)


def _detect_package_managers(manifests: list[str]) -> list[str]:
    mgrs = set()
    for m in manifests:
        base = Path(m).name.lower()
        if base in {"requirements.txt", "pyproject.toml", "poetry.lock", "pipfile"}:
            mgrs.add("pip/poetry")
        if base in {"package.json", "pnpm-workspace.yaml", "yarn.lock"}:
            mgrs.add("npm/yarn/pnpm")
        if base in {"build.gradle", "build.gradle.kts"}:
            mgrs.add("gradle")
        if base == "cargo.toml":
            mgrs.add("cargo")
        if base == "go.mod":
            mgrs.add("go-mod")
        if base == "pom.xml":
            mgrs.add("maven")
    return sorted(mgrs)


def _discover_modules(root_path: Path, files: list[Path]) -> list[dict]:
    groups = defaultdict(list)
    for fp in files:
        rel = fp.relative_to(root_path)
        if len(rel.parts) == 1:
            groups["root"].append(rel)
        else:
            groups[rel.parts[0]].append(rel)

    modules = []
    for name, rels in groups.items():
        ext_counts = defaultdict(int)
        for r in rels:
            ext_counts[r.suffix] += 1
        modules.append({
            "name": name,
            "kind": "root" if name == "root" else "directory",
            "file_count": len(rels),
            "languages": sorted(ext_counts, key=ext_counts.get, reverse=True)[:3],
        })
    modules.sort(key=lambda m: m["file_count"], reverse=True)
    return modules[:40]


def _discover_entry_points(root_path: Path, files: list[Path]) -> list[str]:
    names = {str(fp.relative_to(root_path)): fp.name for fp in files}
    patterns = (
        r"(^|/)(main|app|server|index|run)\.(py|js|ts|go|rs|java|kt|swift)$",
        r"(^|/)__main__\.py$",
    )
    found = []
    for rel in names:
        if any(re.search(p, rel) for p in patterns):
            found.append(rel)
    return sorted(found)[:30]


def _collect_declared_dependencies(root_path: Path, manifests: list[str]) -> dict:
    deps = defaultdict(list)
    for manifest in manifests:
        mf = root_path / manifest
        base = mf.name
        try:
            text = mf.read_text(errors="ignore")
        except Exception:
            continue
        if base == "requirements.txt":
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                pkg = re.split(r"[<>=!~\[]", line, maxsplit=1)[0].strip()
                if pkg:
                    deps["python"].append(pkg)
        elif base == "pyproject.toml":
            for m in re.finditer(r'^\s*([A-Za-z0-9_.\-]+)\s*=', text, re.MULTILINE):
                name = m.group(1)
                if name not in {"name", "version", "description", "requires-python"}:
                    deps["python"].append(name)
        elif base == "package.json":
            try:
                pkg = json.loads(text)
                combo = {}
                combo.update(pkg.get("dependencies", {}))
                combo.update(pkg.get("devDependencies", {}))
                combo.update(pkg.get("peerDependencies", {}))
                deps["node"].extend(combo.keys())
            except Exception:
                pass
        elif base == "go.mod":
            for m in re.finditer(r"^\s*require\s+([^\s]+)", text, re.MULTILINE):
                deps["go"].append(m.group(1))
        elif base == "Cargo.toml":
            for m in re.finditer(r'^\s*([A-Za-z0-9_\-]+)\s*=', text, re.MULTILINE):
                name = m.group(1)
                if name not in {"package", "dependencies", "dev-dependencies", "features"}:
                    deps["rust"].append(name)
        elif base in {"build.gradle", "build.gradle.kts"}:
            for m in re.finditer(r'["\']([A-Za-z0-9_.\-]+:[A-Za-z0-9_.\-]+:[^"\']+)["\']', text):
                deps["jvm"].append(m.group(1))

    # normalize
    cleaned = {}
    for eco, items in deps.items():
        unique = sorted(set(i for i in items if len(i) > 1))
        if unique:
            cleaned[eco] = unique[:250]
    return cleaned


def _build_import_graph(files: list[Path], root_path: Path) -> dict:
    edges = []
    samples = defaultdict(list)
    for fp in files:
        try:
            text = fp.read_text(errors="ignore")
        except Exception:
            continue
        rel = str(fp.relative_to(root_path))
        suffix = fp.suffix
        imported = set()

        if suffix == ".py":
            imported.update(re.findall(r"^\s*import\s+([A-Za-z0-9_\.]+)", text, re.MULTILINE))
            imported.update(re.findall(r"^\s*from\s+([A-Za-z0-9_\.]+)\s+import", text, re.MULTILINE))
        elif suffix in {".js", ".ts", ".jsx", ".tsx"}:
            imported.update(re.findall(r"from\s+[\"']([^\"']+)[\"']", text))
            imported.update(re.findall(r"require\([\"']([^\"']+)[\"']\)", text))
        elif suffix in {".go"}:
            imported.update(re.findall(r'"([^"]+)"', "\n".join(re.findall(r"import\s*\((.*?)\)", text, re.DOTALL))))
        elif suffix in {".java", ".kt"}:
            imported.update(re.findall(r"^\s*import\s+([A-Za-z0-9_.*]+)", text, re.MULTILINE))
        elif suffix == ".rs":
            imported.update(re.findall(r"^\s*use\s+([A-Za-z0-9_:]+)", text, re.MULTILINE))

        for dst in sorted(imported):
            edges.append({"from": rel, "to": dst})
            if len(samples[rel]) < 5:
                samples[rel].append(dst)

    return {"edge_count": len(edges), "samples": dict(list(samples.items())[:120])}


def _build_config_graph(root_path: Path, manifests: list[str]) -> dict:
    build_systems = set()
    capabilities = set()
    config_files = sorted(manifests)[:300]
    for mf in manifests:
        base = Path(mf).name.lower()
        if base in {"package.json", "pnpm-workspace.yaml", "yarn.lock"}:
            build_systems.add("node")
        if base in {"build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts"}:
            build_systems.add("gradle")
        if base in {"go.mod"}:
            build_systems.add("go")
        if base in {"cargo.toml"}:
            build_systems.add("cargo")
        if base in {"dockerfile", "docker-compose.yml"}:
            build_systems.add("docker")

        try:
            text = (root_path / mf).read_text(errors="ignore").lower()
        except Exception:
            continue
        if "internet" in text or "network" in text:
            capabilities.add("network")
        if "camera" in text:
            capabilities.add("camera")
        if "microphone" in text:
            capabilities.add("microphone")
        if "location" in text:
            capabilities.add("location")
        if "bluetooth" in text:
            capabilities.add("bluetooth")
        if "notification" in text:
            capabilities.add("notifications")
        if "filesystem" in text or "storage" in text:
            capabilities.add("storage")

    return {
        "build_systems": sorted(build_systems),
        "capabilities": sorted(capabilities),
        "config_files": config_files,
    }


def _retrieve_config_first(
    query: str,
    intent: str,
    n_results: int = 5,
    ambiguous: bool = False,
    allow_runtime_fallback: bool = False,
    strict_authority_mode: bool = True,
    debug_payload: dict | None = None,
) -> list[dict]:
    """
    Deterministic source-of-truth retrieval path for config/declaration/dependency
    questions. Pull config/manifests first, then fallback to inferred source code.
    """
    workspace = _load_workspace_index()
    priority_files = config_priority_files(
        intent,
        query,
        workspace.get("manifests", []),
        workspace.get("config_graph", {}).get("config_files", []),
    )
    root_path_for_integrity = _repo_root_path_from_hashes()
    if debug_payload is not None:
        startup_probe = get_startup_integrity_probe() or run_startup_integrity_probe(root_path_for_integrity)
        debug_payload["source_of_truth"]["priority_files"] = priority_files
        debug_payload["source_of_truth"]["ranked_paths_with_scores"] = [
            {"path": p, "score": score_authoritative_path(p, query, intent)}
            for p in priority_files
        ]
        debug_payload["source_of_truth"]["selection_reason"] = ""
        debug_payload["source_of_truth"]["exact_path_used"] = ""
        debug_payload["source_of_truth"]["fallback_used"] = False
        debug_payload["source_of_truth"]["integrity"] = {"startup_probe": startup_probe}

    selected_healthy_path, integrity_attempts = select_healthy_authoritative_path(
        priority_files,
        validate_path_fn=lambda p: validate_authoritative_integrity_for_mode(
            mode=IntegrityValidationMode.NORMAL,
            workspace=workspace,
            hash_state=_load_hashes(),
            fetch_exact_file=_fetch_exact_file,
            file_hash_lookup=lambda rp: _file_hash(root_path_for_integrity / rp) if (root_path_for_integrity / rp).exists() else None,
            file_exists_lookup=lambda rp: (root_path_for_integrity / rp).exists() and (root_path_for_integrity / rp).is_file(),
            expected_chunk_count_lookup=lambda rp: _expected_chunk_count_for_file(root_path_for_integrity, rp),
            candidate_paths=[p],
        ),
        max_candidates=min(6, max(1, len(priority_files))),
    )
    if not selected_healthy_path and priority_files:
        # Attempt targeted repair only for the highest-priority authoritative path first.
        repaired_attempt = _validate_and_repair_authoritative_integrity(
            root_path_for_integrity,
            candidate_paths=[priority_files[0]],
        )
        integrity_attempts.append(repaired_attempt | {"candidate": priority_files[0], "repair_ran": True})
        if repaired_attempt.get("overall_status") == INTEGRITY_HEALTHY:
            selected_healthy_path = priority_files[0]
    if debug_payload is not None:
        latest_attempt = integrity_attempts[-1] if integrity_attempts else {}
        latest_files = latest_attempt.get("files", [])
        failing_paths = latest_attempt.get("failing_files", [])
        reason_codes = sorted({reason for f in latest_files for reason in f.get("reasons", [])})
        debug_payload["source_of_truth"]["integrity"] = {
            "selected_healthy_path": selected_healthy_path,
            "overall_status": latest_attempt.get("overall_status", ""),
            "failing_paths": failing_paths,
            "reason_codes": reason_codes,
            "repair_ran": bool(latest_attempt.get("repair_ran", False)),
            "repair_succeeded": bool(latest_attempt.get("repair_succeeded", False)),
            "attempts": integrity_attempts,
        }
    if not selected_healthy_path:
        limitation = {
            "content": (
                "# Source-of-Truth Index Integrity Limitation\n"
                "- The file was discovered in workspace metadata, but the current index could not retrieve it.\n"
                "- The index appears stale or incomplete for authoritative file retrieval.\n"
                "- AndesCode attempted targeted repair for authoritative files before returning this limitation.\n"
            ),
            "file": "__source_of_truth_integrity__",
            "language": "meta",
            "symbols": "",
            "score": 0.0,
            "_rank": 1.0,
            "full_file": True,
            "coverage": {"returned": 1, "total": 1, "partial": False},
            "source_type": "inferred",
            "authority_level": "inferred",
        }
        if debug_payload is not None:
            debug_payload["failure_signals"]["expected_but_missing_authority"] = True
        return [limitation]
    priority_files = [selected_healthy_path] + [p for p in priority_files if p != selected_healthy_path]

    collected = []
    found_authoritative = False
    for fname in priority_files:
        file_chunks = _fetch_all_from_file(fname, query=query, intent=intent, max_results=40)
        if not file_chunks:
            continue
        found_authoritative = True
        source_type = classify_source_type(fname)
        authority = authority_level_for_source(intent, source_type)
        annotate_sources(file_chunks, source_type=source_type, authority_level=authority)
        collected.extend(file_chunks[:6])
        if debug_payload is not None and not debug_payload["source_of_truth"]["exact_path_used"]:
            debug_payload["source_of_truth"]["exact_path_used"] = fname if "/" in fname else file_chunks[0].get("file", "")
            debug_payload["source_of_truth"]["fallback_used"] = bool(file_chunks and file_chunks[0].get("_fallback_used", False))
            debug_payload["source_of_truth"]["selection_reason"] = (
                "selected highest ranked authoritative path for this query"
            )
        if len(collected) >= n_results:
            break

    recovery_ran = False
    if not collected:
        recovery_ran = True
        recovery_candidates = expected_authority_candidates(
            intent,
            query,
            workspace.get("manifests", []),
            workspace.get("config_graph", {}).get("config_files", []),
        )
        ranked_paths = rank_recovery_authoritative_paths(
            intent,
            query,
            workspace.get("manifests", []),
            workspace.get("config_graph", {}).get("config_files", []),
            recovery_candidates,
        )
        if debug_payload is not None:
            debug_payload["source_of_truth"]["recovery_candidates"] = recovery_candidates
            debug_payload["source_of_truth"]["ranked_paths"] = ranked_paths
            debug_payload["source_of_truth"]["ranked_paths_with_scores"] = [
                {"path": p, "score": score_authoritative_path(p, query, intent)}
                for p in ranked_paths
            ]
        recovery_chunks = _recover_authoritative_files(
            ranked_paths,
            intent=intent,
            n_results=n_results,
        )
        if recovery_chunks:
            collected.extend(recovery_chunks)
            found_authoritative = True
    if debug_payload is not None:
        debug_payload["source_of_truth"]["selected_files"] = [c.get("file", "") for c in collected]

    q = query.lower()
    asks_permissions = any(k in q for k in ("permission", "permissions", "declared", "manifest"))
    if asks_permissions:
        permissions = summarize_declared_permissions(collected)
        if permissions:
            summary = {
                "content": "# Manifest Declarations\n"
                + "\n".join([f"- declared permission: {p}" for p in permissions]),
                "file": "AndroidManifest.xml",
                "language": "xml",
                "symbols": "",
                "score": 0.0,
                "_rank": 1.0,
                "full_file": True,
                "coverage": {"returned": 1, "total": 1, "partial": False},
                "source_type": "manifest",
                "authority_level": "declared",
            }
            if debug_payload is not None:
                debug_payload["failure_signals"]["expected_but_missing_authority"] = False
            return [summary] + collected[: max(0, n_results - 1)]

        # Explicit missing-manifest declaration path
        if not any(c.get("file", "").endswith("AndroidManifest.xml") for c in collected):
            if debug_payload is not None:
                debug_payload["source_of_truth"]["missing_expected"].append("AndroidManifest.xml")
                debug_payload["failure_signals"]["expected_but_missing_authority"] = True
            explicit = missing_manifest_notice()
            explicit.update({
                "symbols": "",
                "score": 0.0,
                "_rank": 1.0,
                "full_file": True,
                "coverage": {"returned": 1, "total": 1, "partial": False},
            })
            if (not strict_authority_mode) and allow_runtime_fallback and wants_runtime_usage(query):
                fallback = search_semantic_only(query, n_results=max(1, n_results - 1))
                annotate_sources(fallback, source_type="source_code", authority_level="referenced")
                return [explicit] + fallback
            return [explicit]

    if collected:
        return collected[:n_results]

    limitation = {
        "content": (
            "# Source-of-Truth Limitation\n"
            "- AndesCode could not find the required authoritative file in indexed source-of-truth candidates.\n"
            "- Declared/configured facts cannot be confirmed from authoritative artifacts.\n"
            "- Ask for runtime usage/references explicitly to include inferred code behavior.\n"
            + ("- Query intent appears ambiguous; clarify whether you want declarations or runtime usage.\n" if ambiguous else "")
        ),
        "file": "__source_of_truth_missing__",
        "language": "meta",
        "symbols": "",
        "score": 0.0,
        "_rank": 1.0,
        "full_file": True,
        "coverage": {"returned": 1, "total": 1, "partial": False},
        "source_type": "inferred",
        "authority_level": "inferred",
    }
    if (
        not strict_authority_mode
        and not found_authoritative
        and recovery_ran
        and allow_runtime_fallback
        and wants_runtime_usage(query)
    ):
        fallback = search_semantic_only(query, n_results=max(1, n_results - 1))
        annotate_sources(fallback, source_type="source_code", authority_level="referenced")
        if debug_payload is not None:
            debug_payload["failure_signals"]["expected_but_missing_authority"] = True
        return [limitation] + fallback
    if debug_payload is not None:
        debug_payload["failure_signals"]["expected_but_missing_authority"] = True
    return [limitation]


def _recover_authoritative_files(candidates: list[str], intent: str, n_results: int) -> list[dict]:
    """
    Strict authority recovery pass using deterministic filename/path matching only.
    Never falls back to semantic/runtime source code retrieval.
    """
    if not candidates:
        return []

    recovered = []
    scanned = 0
    max_candidates = min(max(24, n_results * 8), 80)
    for candidate in candidates[:max_candidates]:
        file_chunks = _fetch_exact_file(candidate, max_results=60)
        if not file_chunks:
            continue
        scanned += 1
        source_type = classify_source_type(candidate)
        authority = authority_level_for_source(intent, source_type)
        annotate_sources(file_chunks, source_type=source_type, authority_level=authority)
        recovered.extend(file_chunks[:8])
        if len(recovered) >= n_results:
            break
        if scanned >= 8:
            break
    return recovered[:n_results]


def _normalize_index_path(path: str) -> str:
    """Normalize index/workspace paths into a comparable relative POSIX form."""
    if not path:
        return ""
    p = str(path).replace("\\", "/").strip()
    if not p:
        return ""
    root = str(Path.cwd()).replace("\\", "/").rstrip("/")
    if root and p.startswith(root + "/"):
        p = p[len(root) + 1:]
    repo_name = Path.cwd().name
    marker = f"/{repo_name}/"
    if marker in f"/{p}":
        idx = f"/{p}".find(marker)
        if idx >= 0:
            p = f"/{p}"[idx + len(marker):]
    while p.startswith("./"):
        p = p[2:]
    p = p.lstrip("/")
    return p


def _path_suffix_match(left: str, right: str) -> bool:
    if not left or not right:
        return False
    l = _normalize_index_path(left)
    r = _normalize_index_path(right)
    if not l or not r:
        return False
    return l == r or l.endswith("/" + r) or r.endswith("/" + l)


def _fetch_indexed_candidates_by_basename(path: str, limit: int = 240) -> dict[str, list[dict]]:
    """Fetch indexed chunks grouped by file for files sharing the same basename."""
    base = Path(str(path).replace("\\", "/")).name
    if not base:
        return {}
    try:
        lookup = col.get(where={"file": {"$contains": base}}, limit=limit)
    except Exception:
        return {}
    if not lookup or not lookup.get("documents"):
        return {}

    candidates_by_file: dict[str, list[dict]] = defaultdict(list)
    for i, doc in enumerate(lookup["documents"]):
        if not doc:
            continue
        meta = lookup["metadatas"][i] if i < len(lookup.get("metadatas", [])) else {}
        file_path = meta.get("file", "")
        if not file_path or Path(file_path).name != base:
            continue
        candidates_by_file[file_path].append(
            {
                "content": doc,
                "file": file_path,
                "language": meta.get("language", ""),
                "line": meta.get("line", 0),
                "symbols": meta.get("symbols", ""),
                "score": 0.0,
                "_rank": 1.0,
                "full_file": True,
                "_fallback_used": True,
            }
        )
    return candidates_by_file


def _choose_preferred_candidate_path(requested_path: str, candidate_paths: list[str]) -> str:
    if not candidate_paths:
        return ""
    requested_norm = _normalize_index_path(requested_path)
    build_hints = ("buildsrc", "gradle", "settings.gradle", "build.gradle", "deps", "dependenc", "config")

    def _score(path: str) -> tuple[int, int, int, int, str]:
        p_norm = _normalize_index_path(path)
        exact_norm = int(bool(requested_norm and p_norm == requested_norm))
        suffix_match = int(_path_suffix_match(p_norm, requested_norm))
        build_related = int(any(h in p_norm.lower() for h in build_hints))
        depth = p_norm.count("/")
        return (exact_norm, suffix_match, build_related, -depth, p_norm)

    return sorted(candidate_paths, key=_score, reverse=True)[0]


def _fetch_exact_file(path: str, max_results: int = 60) -> list:
    """
    Fetch indexed chunks for an exact relative path with a strict fallback to
    basename search + exact-path filtering.
    """
    count = col.count()
    if count == 0:
        return []

    normalized_path = _normalize_index_path(path)

    try:
        exact = col.get(where={"file": path}, limit=max_results)
        if exact and exact.get("documents"):
            chunks = [
                {
                    "content": doc,
                    "file": exact["metadatas"][i].get("file", ""),
                    "language": exact["metadatas"][i].get("language", ""),
                    "line": exact["metadatas"][i].get("line", 0),
                    "symbols": exact["metadatas"][i].get("symbols", ""),
                    "score": 0.0,
                    "_rank": 1.0,
                    "full_file": True,
                    "_fallback_used": False,
                }
                for i, doc in enumerate(exact["documents"])
                if doc and _path_suffix_match(exact["metadatas"][i].get("file", ""), path)
            ]
            chunks.sort(key=lambda c: int(c.get("line", 0) or 0))
            return chunks[:max_results]
    except Exception:
        pass

    try:
        candidates_by_file = _fetch_indexed_candidates_by_basename(path, limit=max_results * 12)
        if not candidates_by_file:
            return []
        selected_path = _choose_preferred_candidate_path(
            normalized_path or path,
            list(candidates_by_file.keys()),
        )
        chunks = candidates_by_file.get(selected_path, [])
        chunks.sort(key=lambda c: int(c.get("line", 0) or 0))
        return chunks[:max_results]
    except Exception:
        return []


def search_semantic_only(query: str, n_results: int = 5) -> list[dict]:
    """Semantic retrieval path without config-first routing."""
    count = col.count()
    if count == 0:
        return []

    embedding = embedder.encode(query).tolist()
    fetch = min(n_results * 6, count)
    results = col.query(query_embeddings=[embedding], n_results=fetch)

    candidates = [
        {
            "content": doc,
            "file": results["metadatas"][0][i].get("file", ""),
            "language": results["metadatas"][0][i].get("language", ""),
            "symbols": results["metadatas"][0][i].get("symbols", ""),
            "score": results["distances"][0][i],
        }
        for i, doc in enumerate(results["documents"][0])
    ]
    ranked = _rerank(query, candidates)
    return _add_coverage(ranked[:n_results])

def _fetch_all_from_file(filename: str, query: str = "", intent: str = "", max_results: int = 20) -> list:
    """Retrieve all indexed chunks for a file; exact-path first, deterministic fallback."""
    count = col.count()
    if count == 0:
        return []

    if "/" in filename:
        exact = _fetch_exact_file(filename, max_results=max_results)
        if exact:
            return exact

    try:
        results = col.get(
            where={"file": {"$contains": filename.split("/")[-1]}},
            limit=max_results * 4,
        )
        if not results or not results.get("documents"):
            return []

        candidates_by_file: dict[str, list[dict]] = defaultdict(list)
        for i, doc in enumerate(results["documents"]):
            if not doc:
                continue
            meta = results["metadatas"][i]
            fpath = meta.get("file", "")
            if not fpath:
                continue
            candidates_by_file[fpath].append(
                {
                    "content": doc,
                    "file": fpath,
                    "language": meta.get("language", ""),
                    "line": meta.get("line", 0),
                    "symbols": meta.get("symbols", ""),
                    "score": 0.0,
                    "_rank": 1.0,
                    "full_file": True,
                    "_fallback_used": True,
                }
            )
        if not candidates_by_file:
            return []
        # Duplicate basenames are resolved deterministically to one concrete path.
        selected_path = select_best_authoritative_path(list(candidates_by_file.keys()), query, intent)
        if not selected_path:
            selected_path = sorted(candidates_by_file.keys(), key=lambda p: (p.count("/"), p))[0]
        chunks = candidates_by_file[selected_path]
        chunks.sort(key=lambda c: int(c.get("line", 0) or 0))
        return chunks[:max_results]
    except Exception:
        return []


def _fetch_all_from_file_legacy(filename: str, max_results: int = 20) -> list:
    """Legacy compatibility wrapper."""
    return _fetch_all_from_file(filename, max_results=max_results)


def _add_coverage(chunks: list) -> list:
    """
    Add coverage metadata to each chunk:
    how many chunks exist for that file vs how many we're returning.
    This lets the server tell the model if it has partial file coverage.
    """
    # Count total chunks per file in the index
    file_chunk_counts = {}
    for c in chunks:
        fname = c["file"]
        if fname not in file_chunk_counts:
            try:
                result = col.get(where={"file": {"$contains": fname.split("/")[-1]}})
                file_chunk_counts[fname] = len(result.get("documents", []))
            except Exception:
                file_chunk_counts[fname] = 0

    returned_per_file = defaultdict(int)
    for c in chunks:
        returned_per_file[c["file"]] += 1

    for c in chunks:
        fname   = c["file"]
        total   = file_chunk_counts.get(fname, 0)
        returned= returned_per_file[fname]
        c["coverage"] = {
            "returned": returned,
            "total":    total,
            "partial":  total > 0 and returned < total,
        }
    return chunks


def _chunk_counts_by_path(chunks: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for chunk in chunks:
        path = chunk.get("file", "")
        if path:
            counts[path] += 1
    return dict(counts)


def _camel_tokens(s: str) -> set:
    """Split camelCase/PascalCase/snake_case into component words."""
    parts = re.sub(r"([A-Z][a-z]+|[A-Z]+(?=[A-Z]|$))", r"_\1", s).lower()
    return set(re.findall(r"\w+", parts))


def _rerank(query: str, candidates: list, track_reasons: bool = False) -> list:
    """
    Score each candidate on four axes then sort descending.

    Axes:
      base       — cosine similarity converted to 0-1 (1 = best)
      kw_bonus   — query term overlap with chunk text, tf-weighted
      sym_bonus  — query term overlap with declared symbols (higher weight)
      phrase     — bonus when a multi-word query phrase appears verbatim
      len_penalty— penalise very short chunks (< 5 lines) — usually noise
    """
    q_lower = query.lower()
    terms   = set(re.findall(r"\w+", q_lower))
    # Expand query terms with camelCase decomposition
    q_tokens = terms.copy()
    for t in list(terms):
        q_tokens |= _camel_tokens(t)

    # Build bigrams from query for phrase-match bonus
    q_words  = re.findall(r"\w+", q_lower)
    q_bigrams = {f"{q_words[i]} {q_words[i+1]}" for i in range(len(q_words)-1)}

    for c in candidates:
        content  = c["content"].lower()
        cwords   = set(re.findall(r"\w+", content))
        n_lines  = content.count("\n") + 1

        # Base similarity score (ChromaDB distance → similarity)
        base = max(0.0, 1.0 - (c["score"] / 2.0))

        # Keyword overlap — weight by rarity (terms that appear in few words get more credit)
        matched  = q_tokens & cwords
        kw_bonus = min(len(matched) / max(len(q_tokens), 1) * 0.25, 0.25)

        # Symbol bonus — exact symbol name match carries more signal
        sym_bonus = 0.0
        if c.get("symbols"):
            sw = set(re.findall(r"\w+", c["symbols"].lower()))
            sw |= _camel_tokens(c["symbols"])
            sym_bonus = min(len(q_tokens & sw) / max(len(q_tokens), 1) * 0.20, 0.20)

        # Phrase bonus — verbatim multi-word matches are very strong signal
        phrase_bonus = 0.0
        for bg in q_bigrams:
            if bg in content:
                phrase_bonus = min(phrase_bonus + 0.08, 0.16)

        # Length penalty — very short chunks are usually incomplete / noise
        len_penalty = 0.05 if n_lines < 4 else 0.0

        c["_rank"] = base + kw_bonus + sym_bonus + phrase_bonus - len_penalty
        if track_reasons:
            reasons = []
            if kw_bonus > 0:
                reasons.append("keyword overlap")
            if sym_bonus > 0:
                reasons.append("symbol match")
            if phrase_bonus > 0:
                reasons.append("phrase match")
            if not reasons:
                reasons.append("semantic distance")
            c["_debug_reason"] = ", ".join(reasons)

    return sorted(candidates, key=lambda x: x["_rank"], reverse=True)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _with_timestamp(state: dict, field: str) -> dict:
    out = dict(state)
    timestamps = dict(out.get("timestamps", {}))
    timestamps[field] = _utc_now_iso()
    out["timestamps"] = timestamps
    return out


def _build_current_index_state(root_path: Path, repo_fp: str) -> dict:
    return {
        "repo_root": str(root_path),
        "repo_fingerprint": repo_fp,
        "index_version": INDEX_VERSION,
        "parser_version": PARSER_VERSION,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "retrieval_policy_version": RETRIEVAL_POLICY_VERSION,
        "workspace_schema_version": WORKSPACE_SCHEMA_VERSION,
        "workspace_extraction_version": WORKSPACE_EXTRACTION_VERSION,
        "source_of_truth_version": SOURCE_OF_TRUTH_VERSION,
        "module_detection_version": MODULE_DETECTION_VERSION,
        "timestamps": {"evaluated_at": _utc_now_iso()},
    }


def evaluate_index_state(current_state: dict, stored_state: dict | None, repo_changed: bool) -> dict:
    if not stored_state:
        return {
            "decision": DECISION_FULL_REBUILD,
            "reasons": ["Stored index state missing or unreadable; performing safe full rebuild"],
        }

    if stored_state.get("repo_root") != current_state.get("repo_root"):
        return {
            "decision": DECISION_FULL_REBUILD,
            "reasons": ["Repository root changed; performing full rebuild"],
        }

    for field in ("index_version", "parser_version"):
        if stored_state.get(field) != current_state.get(field):
            return {
                "decision": DECISION_FULL_REBUILD,
                "reasons": [f"{field.replace('_', ' ').capitalize()} changed; performing full rebuild"],
            }

    workspace_only_fields = (
        ("workspace_schema_version", "Workspace schema version changed; rebuilding workspace metadata"),
        ("workspace_extraction_version", "Workspace extraction version changed; rebuilding workspace metadata"),
        ("source_of_truth_version", "Source-of-truth version changed; refreshing authoritative file map"),
        ("module_detection_version", "Module detection version changed; rebuilding workspace metadata"),
    )
    workspace_reasons = [
        reason
        for field, reason in workspace_only_fields
        if stored_state.get(field) != current_state.get(field)
    ]
    if workspace_reasons:
        return {"decision": DECISION_REBUILD_WORKSPACE_ONLY, "reasons": workspace_reasons}

    if repo_changed:
        return {
            "decision": DECISION_INCREMENTAL_REINDEX,
            "reasons": ["Detected changed files; running incremental reindex"],
        }

    return {
        "decision": DECISION_REUSE_ALL,
        "reasons": ["Index state compatible and repository unchanged; reusing all artifacts"],
    }


def _collect_files(root: Path) -> list:
    files = []
    for fp in root.rglob("*"):
        if fp.is_file() and fp.suffix in SUPPORTED_EXTENSIONS:
            if not any(s in fp.parts for s in SKIP_DIRS):
                files.append(fp)
    return sorted(files)


def _chunk_file(fp: Path, root: Path) -> list:
    text = fp.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return []

    rel_path = str(fp.relative_to(root))
    language = fp.suffix.lstrip(".")
    lines    = text.splitlines()
    symbols  = _extract_symbols(text, language)
    pattern  = _BOUNDARY.get(language)

    if pattern and len(lines) > CHUNK_LINES:
        chunks = _boundary_chunks(text, rel_path, language, symbols, pattern)
        if chunks:
            return chunks
    return _line_chunks(lines, rel_path, language, symbols)


def _boundary_chunks(text, rel_path, language, symbols, pattern) -> list:
    lines      = text.splitlines()
    matches    = list(pattern.finditer(text))
    if len(matches) < 2:
        return []

    boundaries = [text[:m.start()].count("\n") for m in matches] + [len(lines)]
    chunks     = []

    for idx, start in enumerate(boundaries[:-1]):
        end   = boundaries[idx + 1]
        block = lines[start:end]
        if len(block) > CHUNK_LINES * 2:
            chunks.extend(_line_chunks(block, rel_path, language, symbols, start))
        elif block:
            ct = "\n".join(block).strip()
            if ct:
                chunks.append(_make_chunk(rel_path, language, symbols, ct, start))
    return chunks


def _line_chunks(lines, rel_path, language, symbols, line_offset=0) -> list:
    chunks = []
    i = 0
    while i < len(lines):
        text = "\n".join(lines[i: i + CHUNK_LINES]).strip()
        if text:
            chunks.append(_make_chunk(rel_path, language, symbols, text, line_offset + i))
        i += CHUNK_LINES - CHUNK_OVERLAP
    return chunks


def _make_chunk(rel_path, language, symbols, text, line) -> dict:
    chunk_id = hashlib.md5(f"{rel_path}:{line}".encode()).hexdigest()
    return {
        "id":       chunk_id,
        "content":  f"# File: {rel_path}\n\n{text}",
        "file":     rel_path,
        "language": language,
        "line":     line,
        "symbols":  symbols,
    }


def _extract_symbols(text: str, language: str) -> str:
    patterns = {
        "py":   r"(?:def|class)\s+(\w+)",
        "js":   r"(?:function|class)\s+(\w+)",
        "ts":   r"(?:function|class|interface|type)\s+(\w+)",
        "java": r"(?:class|interface)\s+(\w+)",
        "kt":   r"(?:fun|class|object|interface)\s+(\w+)",
        "swift":r"(?:func|class|struct|protocol)\s+(\w+)",
        "go":   r"(?:func|type)\s+(\w+)",
        "rs":   r"(?:fn|struct|enum|trait|impl)\s+(\w+)",
    }
    pat = patterns.get(language)
    if not pat:
        return ""
    return " ".join(re.findall(pat, text, re.MULTILINE)[:50])


def _set_repo_fingerprint(fingerprint: str) -> None:
    global CURRENT_REPO_FINGERPRINT
    CURRENT_REPO_FINGERPRINT = fingerprint or ""


def get_repo_fingerprint() -> str:
    if CURRENT_REPO_FINGERPRINT:
        return CURRENT_REPO_FINGERPRINT
    hashes = _load_hashes()
    return hashes.get("__fingerprint__", "")


def _build_file_to_module_map(root_path: Path, files: list[Path]) -> dict:
    mapping = {}
    for fp in files:
        rel = str(fp.relative_to(root_path))
        if len(fp.relative_to(root_path).parts) == 1:
            mapping[rel] = "root"
        else:
            mapping[rel] = fp.relative_to(root_path).parts[0]
    return mapping


def _file_hash(fp: Path) -> str:
    return hashlib.md5(fp.read_bytes()).hexdigest()


def _load_hashes() -> dict:
    try:
        if HASH_STORE.exists():
            return json.loads(HASH_STORE.read_text())
    except Exception:
        pass
    return {}


def _preserve_embedded_hash_state(hashes: dict, root_path: Path, repo_fp: str) -> dict:
    """
    Keep only files that were already tracked as embedded/indexed.
    Used during workspace-only rebuilds, where metadata is refreshed but
    vector-store writes are intentionally skipped.
    """
    preserved = {k: v for k, v in hashes.items() if not k.startswith("__")}
    preserved["__root__"] = str(root_path)
    preserved["__fingerprint__"] = repo_fp
    return preserved


def _repo_root_path_from_hashes() -> Path:
    hashes = _load_hashes()
    root = hashes.get("__root__")
    if root:
        return Path(root)
    return Path(".").resolve()


def _expected_chunk_count_for_file(root_path: Path, rel_path: str) -> int | None:
    cached = _load_chunk_count_state()
    if rel_path in cached:
        try:
            return int(cached[rel_path])
        except Exception:
            pass
    return None


def _expected_chunk_count_for_file_deep(root_path: Path, rel_path: str) -> int | None:
    fp = root_path / rel_path
    if not fp.exists() or not fp.is_file():
        return None
    try:
        return len(_chunk_file(fp, root_path))
    except Exception:
        return None


def _repair_index_paths(root_path: Path, rel_paths: list[str]) -> bool:
    """
    Best-effort, rollback-safe targeted repair for specific files.
    This flow is intentionally not fully transactional across all files:
    we apply per-path updates, and if a later path fails we attempt to roll
    back previously-applied paths to their captured vector snapshot.
    """
    if not rel_paths:
        return True
    required_collection_ops = ("get", "upsert", "delete")
    if any(not hasattr(col, op) for op in required_collection_ops):
        logging.warning("Targeted integrity repair skipped: collection does not expose required rollback-safe APIs")
        return False
    hashes = _load_hashes()
    next_hashes = dict(hashes)
    sorted_paths = sorted(set(rel_paths))
    prepared_paths = []

    for rel in sorted_paths:
        fp = root_path / rel
        previous = col.get(
            where={"file": rel},
            include=["embeddings", "documents", "metadatas"],
        )
        previous_ids = list(previous.get("ids") or [])
        previous_embeddings = list(previous.get("embeddings") or [])
        previous_documents = list(previous.get("documents") or [])
        previous_metadatas = list(previous.get("metadatas") or [])
        missing_on_disk = not fp.exists() or not fp.is_file()
        if missing_on_disk:
            prepared_paths.append(
                {
                    "path": rel,
                    "missing_on_disk": True,
                    "previous_ids": previous_ids,
                    "previous_embeddings": previous_embeddings,
                    "previous_documents": previous_documents,
                    "previous_metadatas": previous_metadatas,
                    "new_ids": [],
                    "new_embeddings": [],
                    "new_documents": [],
                    "new_metadatas": [],
                }
            )
            next_hashes.pop(rel, None)
            continue
        try:
            chunks = _chunk_file(fp, root_path)
        except Exception as e:
            logging.warning(f"Integrity repair skipped {rel}: {e}")
            return False
        new_ids = [c["id"] for c in chunks]
        new_documents = [c["content"] for c in chunks]
        new_metadatas = [
            {"file": c["file"], "language": c["language"], "line": c["line"], "symbols": c.get("symbols", "")}
            for c in chunks
        ]
        new_embeddings = []
        for start in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[start: start + EMBED_BATCH]
            if not batch:
                continue
            embedded = embedder.encode([c["content"] for c in batch], show_progress_bar=False).tolist()
            new_embeddings.extend(embedded)
        prepared_paths.append(
            {
                "path": rel,
                "missing_on_disk": False,
                "previous_ids": previous_ids,
                "previous_embeddings": previous_embeddings,
                "previous_documents": previous_documents,
                "previous_metadatas": previous_metadatas,
                "new_ids": new_ids,
                "new_embeddings": new_embeddings,
                "new_documents": new_documents,
                "new_metadatas": new_metadatas,
            }
        )
        try:
            next_hashes[rel] = _file_hash(fp)
        except Exception:
            pass

    if not prepared_paths:
        return True

    try:
        applied_paths = []
        chunk_counts = _load_chunk_count_state()
        for prepared in prepared_paths:
            rel = prepared["path"]
            if prepared["new_ids"]:
                for start in range(0, len(prepared["new_ids"]), CHROMA_BATCH):
                    col.upsert(
                        ids=prepared["new_ids"][start: start + CHROMA_BATCH],
                        embeddings=prepared["new_embeddings"][start: start + CHROMA_BATCH],
                        documents=prepared["new_documents"][start: start + CHROMA_BATCH],
                        metadatas=prepared["new_metadatas"][start: start + CHROMA_BATCH],
                    )
            stale_ids = sorted(set(prepared["previous_ids"]) - set(prepared["new_ids"]))
            if stale_ids:
                col.delete(ids=stale_ids)
            if prepared["missing_on_disk"]:
                chunk_counts.pop(rel, None)
            else:
                chunk_counts[rel] = len(prepared["new_ids"])
            applied_paths.append(prepared)

        next_hashes["__root__"] = str(root_path)
        if "__fingerprint__" not in next_hashes:
            next_hashes["__fingerprint__"] = get_repo_fingerprint()
        _save_hashes(next_hashes)
        _save_chunk_count_state(chunk_counts)
        return True
    except Exception as e:
        logging.warning(f"Targeted integrity repair failed (best-effort rollback will run): {e}")
        for prepared in reversed(applied_paths):
            try:
                if prepared["new_ids"]:
                    col.delete(ids=prepared["new_ids"])
            except Exception as rollback_err:
                logging.warning(f"Integrity rollback step failed while clearing new vectors for {prepared['path']}: {rollback_err}")
            try:
                if prepared["previous_ids"]:
                    for start in range(0, len(prepared["previous_ids"]), CHROMA_BATCH):
                        col.upsert(
                            ids=prepared["previous_ids"][start: start + CHROMA_BATCH],
                            embeddings=prepared["previous_embeddings"][start: start + CHROMA_BATCH],
                            documents=prepared["previous_documents"][start: start + CHROMA_BATCH],
                            metadatas=prepared["previous_metadatas"][start: start + CHROMA_BATCH],
                        )
            except Exception as rollback_err:
                logging.warning(f"Integrity rollback step failed while restoring previous vectors for {prepared['path']}: {rollback_err}")
        return False


def _refresh_startup_integrity_probe(root_path: Path | None = None, max_files: int = 6, reason: str = "") -> dict:
    root_path = root_path or _repo_root_path_from_hashes()
    workspace = _load_workspace_index()
    hash_state = _load_hashes()

    probe = lightweight_integrity_probe(
        workspace=workspace,
        hash_state=hash_state,
        fetch_exact_file=_fetch_exact_file,
        file_hash_lookup=lambda rel: _file_hash(root_path / rel) if (root_path / rel).exists() else None,
        file_exists_lookup=lambda rel: (root_path / rel).exists() and (root_path / rel).is_file(),
        max_files=max_files,
    )
    INTEGRITY_RUNTIME_STATE.startup_probe = dict(probe)
    INTEGRITY_RUNTIME_STATE.owner_root = str(root_path)
    INTEGRITY_RUNTIME_STATE.refreshed_at = datetime.now(timezone.utc).isoformat()
    _save_integrity_state({
        "startup_probe": probe,
        "startup_probe_refreshed_at": INTEGRITY_RUNTIME_STATE.refreshed_at,
        "startup_probe_owner_root": INTEGRITY_RUNTIME_STATE.owner_root,
        "startup_probe_reason": reason,
    }, merge=True)
    return probe


def run_startup_integrity_probe(root_path: Path | None = None, max_files: int = 6) -> dict:
    return _refresh_startup_integrity_probe(root_path=root_path, max_files=max_files, reason="startup")


def _is_same_root(root_path: Path) -> bool:
    owner_root = (INTEGRITY_RUNTIME_STATE.owner_root or "").strip()
    if not owner_root:
        return False
    try:
        return Path(owner_root).resolve() == root_path.resolve()
    except Exception:
        return False


def _is_startup_probe_fresh(refreshed_at: str, max_age_seconds: int = STARTUP_INTEGRITY_PROBE_MAX_AGE_SECONDS) -> bool:
    if not refreshed_at:
        return False
    try:
        refreshed = datetime.fromisoformat(refreshed_at)
    except Exception:
        return False
    if refreshed.tzinfo is None:
        refreshed = refreshed.replace(tzinfo=timezone.utc)
    age_seconds = (datetime.now(timezone.utc) - refreshed.astimezone(timezone.utc)).total_seconds()
    return 0 <= age_seconds <= max_age_seconds


def get_startup_integrity_probe() -> dict:
    current_root = _repo_root_path_from_hashes()
    if INTEGRITY_RUNTIME_STATE.startup_probe:
        if not _is_same_root(current_root):
            INTEGRITY_RUNTIME_STATE.startup_probe = {}
            INTEGRITY_RUNTIME_STATE.owner_root = ""
            INTEGRITY_RUNTIME_STATE.refreshed_at = ""
            return {}
        if not _is_startup_probe_fresh(INTEGRITY_RUNTIME_STATE.refreshed_at):
            INTEGRITY_RUNTIME_STATE.startup_probe = {}
            INTEGRITY_RUNTIME_STATE.owner_root = ""
            INTEGRITY_RUNTIME_STATE.refreshed_at = ""
            return {}
        return dict(INTEGRITY_RUNTIME_STATE.startup_probe)
    persisted = _load_integrity_state()
    probe = persisted.get("startup_probe")
    persisted_owner_root = str(persisted.get("startup_probe_owner_root", "")).strip()
    if persisted_owner_root:
        try:
            if Path(persisted_owner_root).resolve() != current_root.resolve():
                return {}
        except Exception:
            return {}
    else:
        return {}
    if not _is_startup_probe_fresh(str(persisted.get("startup_probe_refreshed_at", ""))):
        return {}
    if isinstance(probe, dict):
        return dict(probe)
    return {}


def _validate_and_repair_authoritative_integrity(root_path: Path, candidate_paths: list[str] | None = None) -> dict:
    workspace = _load_workspace_index()
    hash_state = _load_hashes()

    def _file_hash_lookup(rel_path: str) -> str | None:
        fp = root_path / rel_path
        if not fp.exists() or not fp.is_file():
            return None
        try:
            return _file_hash(fp)
        except Exception:
            return None

    def _file_exists_lookup(rel_path: str) -> bool:
        fp = root_path / rel_path
        return fp.exists() and fp.is_file()

    report = validate_authoritative_integrity_for_mode(
        mode=IntegrityValidationMode.NORMAL,
        workspace=workspace,
        hash_state=hash_state,
        fetch_exact_file=_fetch_exact_file,
        file_hash_lookup=_file_hash_lookup,
        file_exists_lookup=_file_exists_lookup,
        expected_chunk_count_lookup=lambda p: _expected_chunk_count_for_file(root_path, p),
        candidate_paths=candidate_paths,
    )
    cleaned_hash_state, missing_paths = prune_missing_on_disk_hashes(hash_state, report)
    if missing_paths:
        _save_hashes(cleaned_hash_state)
        hash_state = cleaned_hash_state

    if report.overall_status == INTEGRITY_HEALTHY:
        _save_integrity_state({"report": report.to_dict()}, merge=False)
        _refresh_startup_integrity_probe(root_path=root_path, reason="validate_healthy")
        return report.to_dict()

    repaired = repair_authoritative_integrity(
        report,
        repair_paths_fn=lambda paths: _repair_index_paths(root_path, paths),
        revalidate_fn=lambda paths: deep_repair_integrity_validation(
            workspace=workspace,
            hash_state=_load_hashes(),
            fetch_exact_file=_fetch_exact_file,
            file_hash_lookup=_file_hash_lookup,
            file_exists_lookup=_file_exists_lookup,
            expected_chunk_count_lookup=lambda p: _expected_chunk_count_for_file_deep(root_path, p),
            candidate_paths=paths,
        ),
    )
    _save_integrity_state({"report": repaired.to_dict()}, merge=False)
    _refresh_startup_integrity_probe(root_path=root_path, reason="validate_repair")
    return repaired.to_dict()


def _save_hashes(hashes: dict) -> None:
    try:
        _save_state_json(HASH_STORE, hashes, indent=None)
    except Exception as e:
        logging.warning(f"Could not save hashes: {e}")


def _load_chunk_count_state() -> dict:
    try:
        data = _load_state_json(CHUNK_COUNT_STATE, {})
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _save_chunk_count_state(state: dict[str, int]) -> None:
    try:
        cleaned = {}
        for path, count in state.items():
            try:
                cleaned[path] = int(count)
            except Exception:
                continue
        _save_state_json(CHUNK_COUNT_STATE, cleaned, indent=None)
    except Exception as e:
        logging.warning(f"Could not save chunk count state: {e}")


def _load_integrity_state() -> dict:
    data = _load_state_json(INTEGRITY_STATE, {})
    return data if isinstance(data, dict) else {}


def _save_integrity_state(state: dict, merge: bool = True) -> None:
    payload = dict(_load_integrity_state()) if merge else {}
    payload.update(state)
    _save_state_json(INTEGRITY_STATE, payload, indent=2)


def _load_index_state() -> dict:
    try:
        if INDEX_STATE.exists():
            return json.loads(INDEX_STATE.read_text())
    except Exception as e:
        logging.warning(f"Could not load index state; forcing rebuild: {e}")
    return {}


def _save_index_state(state: dict) -> None:
    try:
        INDEX_STATE.parent.mkdir(parents=True, exist_ok=True)
        INDEX_STATE.write_text(json.dumps(state, indent=2))
    except Exception as e:
        logging.warning(f"Could not save index state: {e}")


def _load_json(path: Path) -> dict:
    data = _load_state_json(path, {})
    return data if isinstance(data, dict) else {}


def _save_json(path: Path, data: dict) -> None:
    try:
        _save_state_json(path, data, indent=2)
    except Exception as e:
        logging.warning(f"Could not save {path.name}: {e}")


def _load_state_json(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        return default
    return default


def _save_state_json(path: Path, data, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if indent is None:
        path.write_text(json.dumps(data))
    else:
        path.write_text(json.dumps(data, indent=indent))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 indexer.py <path-to-codebase>")
        sys.exit(1)

    for event in index_codebase_stream(sys.argv[1]):
        t = event["type"]
        if t == "scan":
            reused = event.get("unchanged", 0)
            suffix = f" ({event['new']} new, {reused} reused)" if reused else ""
            print(f"📂  Found {event['files']} source files{suffix}")
        elif t == "chunks":
            print(f"⚙️   {event['total']} chunks to embed")
        elif t == "embed":
            print(f"\r   Embedding {event['done']}/{event['total']} ({event['pct']}%)",
                  end="", flush=True)
        elif t == "store" and event["done"] == event["total"]:
            print(f"\n   Stored {event['done']} chunks")
        elif t == "mapping":
            print(f"🗺   {event['message']}")
        elif t == "done":
            reused = event.get("reused", 0)
            suffix = f" ({event['new']} new, {reused} reused)" if reused else ""
            pmap   = event.get("map", {})
            print(f"✅  Done — {event['indexed']} files{suffix} · {event['chunks']} chunks")
            if pmap:
                print(f"    Language: {pmap.get('language')} · Domain: {pmap.get('domain')}")
                print(f"    Entry: {pmap.get('entry_point')} · Stack: {pmap.get('stack')}")
        elif t == "error":
            print(f"❌  {event['message']}")
