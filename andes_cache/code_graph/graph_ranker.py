from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Any

GraphDebug = dict[str, Any]
FetchFile = Callable[[str, int], list[dict]]

_GRAPH_ROUTE_SCORES = {
    "exact_symbol": 0.10,
    "filename_match": 0.15,
    "direct_import_neighbor": 0.30,
    "reverse_import_neighbor": 0.40,
    "reference_neighbor": 0.55,
}
_DEFAULT_REFERENCE_NEIGHBOR_LIMIT = 3


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
    already_selected_files = set(semantic_files) | set(authority_files)

    symbols_matched = match_symbols(query, symbol_graph)
    symbol_files = {s.get("file_path", "") for s in symbols_matched if s.get("file_path")}
    filename_files = set(match_filenames(query, repo_graph_state))
    import_seed_files = set(semantic_files[: max(n_results, 1)]) | filename_files | symbol_files
    seed_files = import_seed_files | symbol_files
    graph_artifacts_available = _graph_artifacts_available(symbol_graph, import_graph, repo_graph_state)

    import_neighbor_limit = max(n_results * 2, 8)
    reference_neighbor_limit = min(max(n_results, 1), _DEFAULT_REFERENCE_NEIGHBOR_LIMIT)
    direct_import_neighbors, reverse_import_neighbors = expand_import_neighbors_by_route(
        import_seed_files,
        import_graph,
        limit=import_neighbor_limit,
    )
    existing_direct_import_routes, existing_reverse_import_routes = _import_routes_between_selected_files(
        import_seed_files,
        import_graph,
        already_selected_files | symbol_files | filename_files,
    )
    reference_neighbors = expand_reference_neighbors(
        seed_files,
        symbols_matched,
        symbol_graph,
        repo_graph_state,
        limit=reference_neighbor_limit,
    )

    graph_route_by_file: dict[str, str] = {}
    graph_score_by_file: dict[str, float] = {}
    _record_graph_routes(graph_route_by_file, graph_score_by_file, symbol_files, "exact_symbol")
    _record_graph_routes(graph_route_by_file, graph_score_by_file, filename_files, "filename_match")
    _record_graph_routes(
        graph_route_by_file,
        graph_score_by_file,
        [*direct_import_neighbors, *existing_direct_import_routes],
        "direct_import_neighbor",
    )
    _record_graph_routes(
        graph_route_by_file,
        graph_score_by_file,
        [*reverse_import_neighbors, *existing_reverse_import_routes],
        "reverse_import_neighbor",
    )
    _record_graph_routes(
        graph_route_by_file,
        graph_score_by_file,
        reference_neighbors,
        "reference_neighbor",
    )

    graph_files = sorted(graph_route_by_file, key=lambda f: (graph_score_by_file[f], f))

    boosted_authority_chunks, authority_boosted_files = _apply_graph_metadata_to_chunks(
        authority_chunks,
        graph_route_by_file,
        graph_score_by_file,
    )
    boosted_semantic_candidates, semantic_boosted_files = _apply_graph_metadata_to_chunks(
        semantic_candidates,
        graph_route_by_file,
        graph_score_by_file,
    )
    graph_boosted_existing_files = sorted(set(authority_boosted_files) | set(semantic_boosted_files))

    graph_chunks: list[dict] = []
    if fetch_file:
        for file_path in graph_files:
            if file_path in already_selected_files:
                continue
            for chunk in fetch_file(file_path, 3):
                enriched = dict(chunk)
                route = graph_route_by_file[file_path]
                score = graph_score_by_file[file_path]
                # Graph-only neighbors are useful context but are not semantic
                # vector hits. Assign deterministic route-aware distances so
                # reranking can prefer exact matches over broader expansions.
                enriched["score"] = score
                enriched["_graph_selected"] = True
                enriched["_graph_route"] = route
                enriched["_graph_score_reason"] = _graph_score_reason(route, score)
                graph_chunks.append(enriched)

    combined = _dedupe_chunks([*boosted_authority_chunks, *boosted_semantic_candidates, *graph_chunks])
    notes = []
    if not graph_artifacts_available:
        notes.append("Graph artifacts missing or empty; hybrid retrieval could not expand neighbors.")
    if graph_chunks:
        notes.append(f"Graph retrieval added {len(_files_from_chunks(graph_chunks))} neighboring file(s).")
    if graph_boosted_existing_files:
        notes.append(f"Graph retrieval boosted {len(graph_boosted_existing_files)} existing selected file(s).")
    if reference_neighbors:
        notes.append(
            f"Reference-neighbor expansion added up to {reference_neighbor_limit} lower-confidence file(s); "
            "exact symbols and import routes keep priority."
        )
    if symbols_matched:
        notes.append(f"Matched symbol(s): {', '.join(sorted({s.get('name', '') for s in symbols_matched})[:8])}.")
    if graph_artifacts_available and not graph_chunks and not symbols_matched:
        notes.append("Graph retrieval found no additional high-confidence neighbors.")

    import_neighbors = [*direct_import_neighbors, *reverse_import_neighbors]
    graph_neighbors_added = [
        f
        for f in [*import_neighbors, *reference_neighbors]
        if f and f not in already_selected_files and f in graph_route_by_file
    ]

    routes = ["semantic_vector"]
    if authority_chunks:
        routes.append("source_of_truth")
    if symbols_matched:
        routes.append("exact_symbol")
    if filename_files:
        routes.append("file_name")
    if (
        direct_import_neighbors
        or reverse_import_neighbors
        or existing_direct_import_routes
        or existing_reverse_import_routes
    ):
        routes.append("import_neighbors")
    if reference_neighbors:
        routes.append("reference_neighbors")

    debug = {
        "retrieval_routes_used": routes,
        "graph_neighbors_added": graph_neighbors_added,
        "symbols_matched": [s for s in symbols_matched],
        "files_selected_by_graph": graph_files,
        "files_selected_by_semantic": semantic_files,
        "files_selected_by_authority": authority_files,
        "context_sufficiency_notes": notes,
        "graph_route_by_file": {file_path: graph_route_by_file[file_path] for file_path in graph_files},
        "graph_score_by_file": {file_path: graph_score_by_file[file_path] for file_path in graph_files},
        "graph_seed_files": sorted(f for f in seed_files if f),
        "graph_expansion_limits": {
            "import_neighbors": import_neighbor_limit,
            "reference_neighbors": reference_neighbor_limit,
        },
        "graph_boosted_existing_files": graph_boosted_existing_files,
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


def expand_import_neighbors_by_route(seed_files: set[str], graph: dict, limit: int = 12) -> tuple[list[str], list[str]]:
    adjacency = graph.get("adjacency", {}) if isinstance(graph, dict) else {}
    reverse = graph.get("reverse_adjacency", {}) if isinstance(graph, dict) else {}
    seed_set = set(seed_files)
    direct: list[str] = []
    reverse_neighbors: list[str] = []
    seen_neighbors: set[str] = set(seed_set)

    # Prefer direct-import routes globally when a file is reachable through both
    # directions, and make every selected neighbor consume capacity at most once.
    for seed in sorted(seed_set):
        for neighbor in adjacency.get(seed, []) or []:
            if neighbor in seen_neighbors:
                continue
            seen_neighbors.add(neighbor)
            direct.append(neighbor)
            if len(direct) + len(reverse_neighbors) >= limit:
                return direct, reverse_neighbors
    for seed in sorted(seed_set):
        for neighbor in reverse.get(seed, []) or []:
            if neighbor in seen_neighbors:
                continue
            seen_neighbors.add(neighbor)
            reverse_neighbors.append(neighbor)
            if len(direct) + len(reverse_neighbors) >= limit:
                return direct, reverse_neighbors
    return direct, reverse_neighbors


def _import_routes_between_selected_files(
    seed_files: set[str],
    graph: dict,
    selected_files: set[str],
) -> tuple[list[str], list[str]]:
    adjacency = graph.get("adjacency", {}) if isinstance(graph, dict) else {}
    reverse = graph.get("reverse_adjacency", {}) if isinstance(graph, dict) else {}
    seed_set = set(seed_files)
    direct: list[str] = []
    reverse_neighbors: list[str] = []
    seen_direct: set[str] = set()
    seen_reverse: set[str] = set()
    for seed in sorted(seed_set):
        for neighbor in adjacency.get(seed, []) or []:
            if neighbor == seed or neighbor not in selected_files or neighbor in seen_direct:
                continue
            seen_direct.add(neighbor)
            direct.append(neighbor)
        for neighbor in reverse.get(seed, []) or []:
            if neighbor == seed or neighbor not in selected_files or neighbor in seen_direct or neighbor in seen_reverse:
                continue
            seen_reverse.add(neighbor)
            reverse_neighbors.append(neighbor)
    return direct, reverse_neighbors

def expand_reference_neighbors(
    seed_files: set[str],
    symbols_matched: list[dict],
    symbol_graph: dict,
    repo_graph_state: dict,
    limit: int = _DEFAULT_REFERENCE_NEIGHBOR_LIMIT,
) -> list[str]:
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


def _record_graph_routes(
    graph_route_by_file: dict[str, str],
    graph_score_by_file: dict[str, float],
    files: set[str] | list[str],
    route: str,
) -> None:
    score = _GRAPH_ROUTE_SCORES[route]
    for file_path in files:
        if not file_path:
            continue
        current_score = graph_score_by_file.get(file_path)
        if current_score is None or score < current_score:
            graph_route_by_file[file_path] = route
            graph_score_by_file[file_path] = score


def _apply_graph_metadata_to_chunks(
    chunks: list[dict],
    graph_route_by_file: dict[str, str],
    graph_score_by_file: dict[str, float],
) -> tuple[list[dict], list[str]]:
    boosted_chunks: list[dict] = []
    boosted_files: list[str] = []
    for chunk in chunks or []:
        file_path = chunk.get("file") or chunk.get("path")
        if not file_path or file_path not in graph_route_by_file:
            boosted_chunks.append(chunk)
            continue

        route = graph_route_by_file[file_path]
        graph_score = graph_score_by_file[file_path]
        boosted = dict(chunk)
        existing_score = boosted.get("score")
        boosted["score"] = min(existing_score, graph_score) if isinstance(existing_score, (int, float)) else graph_score
        boosted["_graph_selected"] = True
        boosted["_graph_route"] = route
        boosted["_graph_score_reason"] = _graph_score_reason(route, graph_score)
        boosted_chunks.append(boosted)
        boosted_files.append(file_path)
    return boosted_chunks, boosted_files


def _graph_score_reason(route: str, score: float) -> str:
    return f"{route} graph route assigned deterministic score {score:.2f}"


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


def _graph_artifacts_available(symbol_graph: dict, import_graph: dict, repo_graph_state: dict) -> bool:
    if not any(isinstance(obj, dict) and obj for obj in (symbol_graph, import_graph, repo_graph_state)):
        return False
    return bool(
        (isinstance(symbol_graph, dict) and (symbol_graph.get("symbols") or symbol_graph.get("by_name")))
        or (isinstance(import_graph, dict) and (import_graph.get("edges") or import_graph.get("adjacency")))
    )
