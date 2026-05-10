from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Any

from .import_graph import expand_import_neighbors

GraphDebug = dict[str, Any]
FetchFile = Callable[[str, int], list[dict]]


def hybrid_retrieve(
    *,
    query: str,
    semantic_candidates: list[dict],
    symbol_graph: dict,
    import_graph: dict,
    repo_graph_state: dict | None = None,
    fetch_file: FetchFile | None = None,
    n_results: int = 5,
    authority_chunks: list[dict] | None = None,
) -> tuple[list[dict], GraphDebug]:
    """Blend semantic candidates with deterministic code-graph expansion.

    This function is model-free and side-effect-free.  It does not replace the
    existing retriever; callers opt into it and provide already-fetched semantic
    candidates plus an optional file loader for graph-selected files.
    """
    repo_graph_state = repo_graph_state or {}
    authority_chunks = authority_chunks or []
    semantic_files = _files_from_chunks(semantic_candidates)
    authority_files = _files_from_chunks(authority_chunks)
    symbols_matched = match_symbols(query, symbol_graph)
    symbol_files = {s.get("file_path", "") for s in symbols_matched if s.get("file_path")}
    filename_files = set(match_filenames(query, repo_graph_state))
    import_seed_files = set(semantic_files[: max(n_results, 1)]) | filename_files
    seed_files = import_seed_files | symbol_files
    graph_neighbors = expand_import_neighbors(import_seed_files, import_graph, limit=max(n_results * 2, 8))
    reference_neighbors = expand_reference_neighbors(seed_files, symbols_matched, symbol_graph, repo_graph_state, limit=max(n_results, 5))
    graph_files = [f for f in sorted(symbol_files | filename_files | set(graph_neighbors) | set(reference_neighbors)) if f]

    graph_chunks: list[dict] = []
    if fetch_file:
        for file_path in graph_files:
            if file_path in semantic_files or file_path in authority_files:
                continue
            for chunk in fetch_file(file_path, 3):
                enriched = dict(chunk)
                enriched.setdefault("score", 0.05)
                enriched["_graph_selected"] = True
                graph_chunks.append(enriched)

    combined = _dedupe_chunks([*authority_chunks, *semantic_candidates, *graph_chunks])
    notes = []
    if graph_chunks:
        notes.append(f"Graph retrieval added {len(_files_from_chunks(graph_chunks))} neighboring file(s).")
    if symbols_matched:
        notes.append(f"Matched symbol(s): {', '.join(sorted({s.get('name', '') for s in symbols_matched})[:8])}.")
    if not notes:
        notes.append("Graph retrieval found no additional high-confidence neighbors.")

    routes = ["semantic_vector"]
    if authority_chunks:
        routes.append("source_of_truth")
    if symbols_matched:
        routes.append("exact_symbol")
    if filename_files:
        routes.append("file_name")
    if graph_neighbors:
        routes.append("import_neighbors")
    if reference_neighbors:
        routes.append("reference_neighbors")

    debug = {
        "retrieval_routes_used": routes,
        "graph_neighbors_added": [f for f in graph_neighbors + reference_neighbors if f not in semantic_files],
        "symbols_matched": [s for s in symbols_matched],
        "files_selected_by_graph": sorted(set(graph_files)),
        "files_selected_by_semantic": semantic_files,
        "files_selected_by_authority": authority_files,
        "context_sufficiency_notes": notes,
    }
    return combined, debug


def match_symbols(query: str, symbol_graph: dict) -> list[dict]:
    by_name = symbol_graph.get("by_name", {}) if isinstance(symbol_graph, dict) else {}
    words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query))
    lower_words = {w.lower() for w in words}
    matches: list[dict] = []
    for name, entries in by_name.items():
        if name in words or name.lower() in lower_words:
            matches.extend(entries if isinstance(entries, list) else [])
    return matches


def match_filenames(query: str, repo_graph_state: dict) -> list[str]:
    files = (repo_graph_state or {}).get("files", {})
    if not isinstance(files, dict):
        return []
    q = query.lower()
    found = []
    for path in files:
        base = Path(path).name.lower()
        if base and (base in q or path.lower() in q):
            found.append(path)
    return sorted(found)


def expand_reference_neighbors(seed_files: set[str], symbols_matched: list[dict], symbol_graph: dict, repo_graph_state: dict, limit: int = 8) -> list[str]:
    symbol_names = {s.get("name", "") for s in symbols_matched if s.get("name")}
    if not symbol_names:
        refs = (repo_graph_state or {}).get("references", {})
        symbol_names = {name for f in seed_files for name in refs.get(f, [])}
    by_file = symbol_graph.get("by_file", {}) if isinstance(symbol_graph, dict) else {}
    out = []
    for file_path, entries in by_file.items():
        if file_path in seed_files:
            continue
        for entry in entries:
            if symbol_names.intersection(set(entry.get("references", []) or [])):
                out.append(file_path)
                break
        if len(out) >= limit:
            break
    return sorted(set(out))


def _files_from_chunks(chunks: list[dict]) -> list[str]:
    seen = set()
    files = []
    for chunk in chunks or []:
        path = chunk.get("file") or chunk.get("path")
        if path and path not in seen:
            seen.add(path)
            files.append(path)
    return files


def _dedupe_chunks(chunks: list[dict]) -> list[dict]:
    out = []
    seen = set()
    for chunk in chunks:
        key = (chunk.get("file", ""), chunk.get("line", 0), chunk.get("content", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(chunk)
    return out
