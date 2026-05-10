from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .import_graph import build_import_graph
from .models import RepoGraph
from .parser_registry import ParserRegistry
from .symbol_extractor import extract_symbols_for_file

SYMBOL_GRAPH_FILE = "symbol_graph.json"
IMPORT_GRAPH_FILE = "import_graph.json"
REPO_GRAPH_STATE_FILE = "repo_graph_state.json"


def build_repo_graph(root: Path, files: list[Path], index_dir: Path | None = None) -> RepoGraph:
    registry = ParserRegistry()
    symbols = []
    file_meta: dict[str, dict[str, Any]] = {}
    for fp in files:
        rel = str(fp.relative_to(root))
        language = registry.language_for_path(fp)
        file_meta[rel] = {"language": language}
        try:
            file_symbols = extract_symbols_for_file(fp, root, registry=registry)
            symbols.extend(file_symbols)
        except Exception as exc:
            file_meta[rel]["symbol_error"] = str(exc)

    import_graph = build_import_graph(files, root, registry=registry)
    graph = RepoGraph(
        symbols=symbols,
        imports=[],
        references=_reference_index(symbols),
        files=file_meta,
    )
    # Store import edges as dict state separately; RepoGraph.imports remains a
    # typed compact edge list for callers that deserialize via models.
    graph.files["__import_graph__"] = import_graph
    if index_dir is not None:
        persist_repo_graph(graph, import_graph, index_dir, root)
    return graph


def persist_repo_graph(graph: RepoGraph, import_graph: dict, index_dir: Path, root: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    _write_json(index_dir / SYMBOL_GRAPH_FILE, {
        "symbols": [s.to_dict() for s in graph.symbols],
        "by_name": _symbols_by_name(graph.symbols),
        "by_file": _symbols_by_file(graph.symbols),
    })
    _write_json(index_dir / IMPORT_GRAPH_FILE, import_graph)
    _write_json(index_dir / REPO_GRAPH_STATE_FILE, {
        "repo_root": str(root),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "symbol_count": len(graph.symbols),
        "import_edge_count": import_graph.get("edge_count", 0),
        "files": {k: v for k, v in graph.files.items() if not k.startswith("__")},
        "references": graph.references,
    })


def load_graph_artifacts(index_dir: Path) -> dict[str, Any]:
    return {
        "symbol_graph": _read_json(index_dir / SYMBOL_GRAPH_FILE, {}),
        "import_graph": _read_json(index_dir / IMPORT_GRAPH_FILE, {}),
        "repo_graph_state": _read_json(index_dir / REPO_GRAPH_STATE_FILE, {}),
    }


def _reference_index(symbols) -> dict[str, list[str]]:
    known = {s.name for s in symbols}
    by_file: dict[str, set[str]] = {}
    for symbol in symbols:
        matches = known.intersection(symbol.references or []) - {symbol.name}
        if matches:
            by_file.setdefault(symbol.file_path, set()).update(matches)
    return {k: sorted(v) for k, v in by_file.items()}


def _symbols_by_name(symbols) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for symbol in symbols:
        out.setdefault(symbol.name, []).append(symbol.to_dict())
    return out


def _symbols_by_file(symbols) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for symbol in symbols:
        out.setdefault(symbol.file_path, []).append(symbol.to_dict())
    return out


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return default
