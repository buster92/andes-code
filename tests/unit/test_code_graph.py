from __future__ import annotations

from pathlib import Path

from andes_cache.code_graph.graph_ranker import hybrid_retrieve
from andes_cache.code_graph.import_graph import build_import_graph, expand_import_neighbors
from andes_cache.code_graph.parser_registry import ParserHandle, ParserRegistry
from andes_cache.code_graph.repo_graph import build_repo_graph, load_graph_artifacts
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
