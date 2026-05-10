from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from andes_cache.code_graph.graph_ranker import hybrid_retrieve
from andes_cache.code_graph.import_graph import build_import_graph, expand_import_neighbors
from andes_cache.code_graph.parser_registry import ParserHandle, ParserRegistry
from andes_cache.code_graph.repo_graph import CODE_GRAPH_VERSION, build_repo_graph, graph_artifacts_current, load_graph_artifacts
from andes_cache.code_graph.symbol_extractor import extract_symbols


def test_python_symbol_extraction_with_methods_and_references() -> None:
    text = """
import helpers

API_URL = "https://example.test"

class Service:
    def fetch(self):
        return helpers.load(API_URL)

def make_service():
    return Service()
"""

    symbols = extract_symbols(text, "service.py", "py")

    by_name = {symbol.name: symbol for symbol in symbols}
    assert by_name["Service"].kind == "class"
    assert by_name["fetch"].kind == "method"
    assert by_name["fetch"].parent == "Service"
    assert by_name["make_service"].kind == "function"
    assert by_name["API_URL"].kind == "constant"
    assert "helpers" in by_name["Service"].imports
    assert by_name["Service"].start_line < by_name["Service"].end_line


def test_kotlin_symbol_extraction_fallback() -> None:
    text = """
package com.example
import com.example.data.Repository

interface LoginView
object LoginRoutes
class LoginController {
    fun authenticate(user: String): Boolean = Repository().check(user)
}
typealias UserName = String
"""

    symbols = extract_symbols(text, "LoginController.kt", "kt")

    by_name = {symbol.name: symbol for symbol in symbols}
    assert by_name["LoginView"].kind == "interface"
    assert by_name["LoginRoutes"].kind == "object"
    assert by_name["LoginController"].kind == "class"
    assert by_name["authenticate"].kind == "method"
    assert by_name["authenticate"].parent == "LoginController"
    assert by_name["UserName"].kind == "type"


def test_import_graph_expands_python_neighbors(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("from services.auth import AuthService\n", encoding="utf-8")
    services = tmp_path / "services"
    services.mkdir()
    (services / "auth.py").write_text("class AuthService: pass\n", encoding="utf-8")

    graph = build_import_graph([tmp_path / "app.py", services / "auth.py"], tmp_path)
    neighbors = expand_import_neighbors({"app.py"}, graph)

    assert "services/auth.py" in neighbors
    assert graph["edge_count"] == 1


def test_hybrid_retrieval_includes_graph_neighbor_files() -> None:
    semantic = [{"file": "app.py", "content": "AuthService()", "line": 1, "score": 0.2, "symbols": ""}]
    symbol_graph = {
        "by_name": {"AuthService": [{"name": "AuthService", "file_path": "services/auth.py", "references": []}]},
        "by_file": {"services/auth.py": [{"name": "AuthService", "file_path": "services/auth.py", "references": []}]},
    }
    import_graph = {"adjacency": {"app.py": ["services/auth.py"]}, "reverse_adjacency": {"services/auth.py": ["app.py"]}}

    def fetch_file(path: str, limit: int) -> list[dict]:
        return [{"file": path, "content": "class AuthService: pass", "line": 1, "score": 0.0, "symbols": "AuthService"}]

    chunks, debug = hybrid_retrieve(
        query="Where is AuthService used?",
        semantic_candidates=semantic,
        symbol_graph=symbol_graph,
        import_graph=import_graph,
        repo_graph_state={"files": {"app.py": {}, "services/auth.py": {}}},
        fetch_file=fetch_file,
        n_results=5,
    )

    assert any(chunk["file"] == "services/auth.py" for chunk in chunks)
    assert "services/auth.py" in debug["files_selected_by_graph"]
    assert "import_neighbors" in debug["retrieval_routes_used"]


def test_symbol_extraction_fallback_when_tree_sitter_unavailable() -> None:
    class NoTreeSitterRegistry(ParserRegistry):
        def get_parser(self, language: str) -> ParserHandle:
            return ParserHandle(language=language, parser=None, available=False, reason="forced unavailable")

    symbols = extract_symbols("class Greeter:\n    def hello(self):\n        return 'hi'\n", "greeter.py", "py", registry=NoTreeSitterRegistry())

    assert {symbol.name for symbol in symbols} == {"Greeter", "hello"}


def test_repo_graph_persists_artifacts(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("from helper import run\nrun()\n", encoding="utf-8")
    (tmp_path / "helper.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    index_dir = tmp_path / ".andes-index"

    build_repo_graph(tmp_path, [tmp_path / "main.py", tmp_path / "helper.py"], index_dir=index_dir)
    artifacts = load_graph_artifacts(index_dir)

    assert artifacts["symbol_graph"]["by_name"]["run"][0]["file_path"] == "helper.py"
    assert "main.py" in artifacts["import_graph"]["adjacency"]
    assert artifacts["repo_graph_state"]["symbol_count"] == 1


def test_js_ts_relative_import_resolution(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "app.ts").write_text('import { helper } from "./helper"\nhelper()\n', encoding="utf-8")
    (src / "helper.ts").write_text("export function helper() {}\n", encoding="utf-8")

    graph = build_import_graph([src / "app.ts", src / "helper.ts"], tmp_path)

    assert graph["adjacency"]["src/app.ts"] == ["src/helper.ts"]
    assert not graph["unresolved"].get("src/app.ts")


def test_js_ts_relative_import_resolution_across_parent_directories(tmp_path: Path) -> None:
    service = tmp_path / "src" / "features" / "auth"
    token_dir = tmp_path / "src" / "features" / "utils"
    service.mkdir(parents=True)
    token_dir.mkdir(parents=True)
    (service / "service.ts").write_text('import { token } from "../utils/token"\n', encoding="utf-8")
    (token_dir / "token.ts").write_text("export const token = 't'\n", encoding="utf-8")

    graph = build_import_graph([service / "service.ts", token_dir / "token.ts"], tmp_path)

    assert graph["adjacency"]["src/features/auth/service.ts"] == ["src/features/utils/token.ts"]


def test_js_ts_relative_import_resolution_across_multiple_parent_directories(tmp_path: Path) -> None:
    service = tmp_path / "src" / "features" / "auth"
    shared = tmp_path / "src" / "shared"
    service.mkdir(parents=True)
    shared.mkdir(parents=True)
    (service / "service.ts").write_text('import api from "../../shared/api"\n', encoding="utf-8")
    (shared / "api.ts").write_text("export default function api() {}\n", encoding="utf-8")

    graph = build_import_graph([service / "service.ts", shared / "api.ts"], tmp_path)

    assert graph["adjacency"]["src/features/auth/service.ts"] == ["src/shared/api.ts"]


def test_js_ts_relative_import_resolution_supports_index_files(tmp_path: Path) -> None:
    src = tmp_path / "src"
    helper = src / "helper"
    helper.mkdir(parents=True)
    (src / "app.ts").write_text('import { helper } from "./helper"\n', encoding="utf-8")
    (helper / "index.ts").write_text("export function helper() {}\n", encoding="utf-8")

    graph = build_import_graph([src / "app.ts", helper / "index.ts"], tmp_path)

    assert graph["adjacency"]["src/app.ts"] == ["src/helper/index.ts"]


def test_js_ts_missing_and_alias_imports_remain_unresolved(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "app.ts").write_text('import missing from "./missing"\nimport api from "@/shared/api"\n', encoding="utf-8")

    graph = build_import_graph([src / "app.ts"], tmp_path)

    assert graph["unresolved"]["src/app.ts"] == ["./missing", "@/shared/api"]
    assert "src/app.ts" not in graph["adjacency"]


def test_graph_artifact_version_validation_detects_missing_or_stale_version(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    index_dir = tmp_path / ".andes-index"

    build_repo_graph(tmp_path, [tmp_path / "main.py"], index_dir=index_dir)
    assert graph_artifacts_current(index_dir)

    state_path = index_dir / "repo_graph_state.json"
    stale_state = state_path.read_text(encoding="utf-8").replace(CODE_GRAPH_VERSION, "stale-version")
    state_path.write_text(stale_state, encoding="utf-8")
    assert not graph_artifacts_current(index_dir)

    build_repo_graph(tmp_path, [tmp_path / "main.py"], index_dir=index_dir)
    assert graph_artifacts_current(index_dir)


def test_indexer_search_hybrid_retrieval_adds_import_neighbor_and_uses_hybrid_cache(monkeypatch) -> None:
    class _FakeEmbedding:
        def tolist(self):
            return [0.1, 0.2, 0.3]

    class _FakeModel:
        def __init__(self, *_args, **_kwargs):
            pass

        def encode(self, *_args, **_kwargs):
            return _FakeEmbedding()

    class _FakeCollection:
        def count(self):
            return 2

        def query(self, query_embeddings, n_results):  # noqa: ARG002
            return {
                "documents": [["import { helper } from './helper'\nhelper()"]],
                "metadatas": [[{"file": "src/app.ts", "language": "ts", "line": 1, "symbols": ""}]],
                "distances": [[0.05]],
            }

        def get(self, where=None, limit=None):  # noqa: ARG002
            if where == {"file": "src/helper.ts"}:
                return {
                    "documents": ["export function helper() { return 1 }"],
                    "metadatas": [{"file": "src/helper.ts", "language": "ts", "line": 1, "symbols": "helper"}],
                }
            return {"documents": [], "metadatas": []}

    class _FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_or_create_collection(self, *_args, **_kwargs):
            return _FakeCollection()

    fake_sentence_transformers = types.ModuleType("sentence_transformers")
    fake_sentence_transformers.SentenceTransformer = _FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_sentence_transformers)
    fake_chromadb = types.ModuleType("chromadb")
    fake_chromadb.PersistentClient = _FakeClient
    monkeypatch.setitem(sys.modules, "chromadb", fake_chromadb)
    monkeypatch.setenv("ANDESCODE_HYBRID_RETRIEVAL", "1")

    import indexer

    indexer = importlib.reload(indexer)
    indexer.classify_query_intent_details = lambda _q: {
        "intent": "runtime_usage_or_reference",
        "retrieval_route": "semantic",
        "ambiguous": False,
        "allow_runtime_fallback": False,
        "strict_authority_mode": True,
    }
    indexer._load_workspace_index = lambda: {"manifests": [], "config_graph": {"config_files": []}}
    indexer._structured_query_results = lambda _q: []
    indexer._load_json = lambda _path: {}
    indexer.get_repo_fingerprint = lambda: "repo-fp"
    indexer._add_coverage = lambda chunks: chunks
    indexer._rerank = lambda _q, candidates, track_reasons=False: [dict(c, _rank=1.0) for c in candidates]
    indexer._load_code_graph_artifacts = lambda: {
        "symbol_graph": {"symbols": [], "by_name": {}, "by_file": {}},
        "import_graph": {"edges": [{"source": "src/app.ts", "target": "src/helper.ts", "import_name": "./helper", "resolved": True}], "adjacency": {"src/app.ts": ["src/helper.ts"]}, "reverse_adjacency": {"src/helper.ts": ["src/app.ts"]}},
        "repo_graph_state": {"code_graph_version": indexer.CODE_GRAPH_VERSION, "files": {"src/app.ts": {}, "src/helper.ts": {}}},
    }

    cache_get_routes = []
    cache_set_routes = []

    def _cache_get(repo_fp, query, index_version, intent, retrieval_route):  # noqa: ARG001
        cache_get_routes.append(retrieval_route)
        return None

    def _cache_set(repo_fp, query, index_version, value, intent, retrieval_route):  # noqa: ARG001
        cache_set_routes.append(retrieval_route)

    indexer.CACHE = SimpleNamespace(retrieval_get=_cache_get, retrieval_set=_cache_set)

    results, debug = indexer.search("explain helper flow", n_results=2, debug_mode=True, return_debug=True)

    assert [chunk["file"] for chunk in results] == ["src/app.ts", "src/helper.ts"]
    assert debug["retrieval"]["files_selected_by_semantic"] == ["src/app.ts"]
    assert "src/helper.ts" in debug["retrieval"]["files_selected_by_graph"]
    assert "import_neighbors" in debug["retrieval"]["retrieval_routes_used"]
    assert "semantic:hybrid" in cache_get_routes
    assert "semantic:hybrid" in cache_set_routes


def test_ambiguous_basename_import_does_not_create_edge(tmp_path: Path) -> None:
    src = tmp_path / "src"
    left = src / "left"
    right = src / "right"
    left.mkdir(parents=True)
    right.mkdir(parents=True)
    (src / "app.ts").write_text('import helper from "helper"\n', encoding="utf-8")
    (left / "helper.ts").write_text("export const helper = 1\n", encoding="utf-8")
    (right / "helper.ts").write_text("export const helper = 2\n", encoding="utf-8")

    graph = build_import_graph([src / "app.ts", left / "helper.ts", right / "helper.ts"], tmp_path)

    assert graph["unresolved"]["src/app.ts"] == ["helper"]
    assert "src/app.ts" not in graph["adjacency"]
