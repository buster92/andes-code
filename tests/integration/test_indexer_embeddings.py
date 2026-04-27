import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration]


@pytest.fixture()
def sample_codebase():
    tmpdir = tempfile.mkdtemp()
    try:
        (Path(tmpdir) / "auth.py").write_text(
            "def login(username, password):\n"
            "    return username == 'admin' and password == 'secret'\n"
        )
        (Path(tmpdir) / "models.py").write_text(
            "class User:\n"
            "    def __init__(self, name, email):\n"
            "        self.name = name\n"
            "        self.email = email\n"
        )
        (Path(tmpdir) / "README.md").write_text("# Test project\nThis is a test.")
        (Path(tmpdir) / "requirements.txt").write_text("fastapi==0.111.0\npydantic>=2.0\n")
        (Path(tmpdir) / "package.json").write_text(
            json.dumps({
                "name": "tmp-test",
                "dependencies": {"react": "^18.2.0", "axios": "^1.7.0"},
                "devDependencies": {"typescript": "^5.0.0"},
            })
        )
        (Path(tmpdir) / "api.py").write_text(
            "from fastapi import FastAPI\n"
            "from auth import login\n"
            "app = FastAPI()\n"
        )
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _import_indexer():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import indexer

    return indexer


def test_index_codebase(sample_codebase):
    indexer = _import_indexer()
    result = indexer.index_codebase(sample_codebase)
    assert "indexed" in result
    assert result["indexed"] > 0


def test_search_returns_results(sample_codebase):
    indexer = _import_indexer()
    indexer.index_codebase(sample_codebase)
    results = indexer.search("user authentication login", n_results=2)
    assert isinstance(results, list)
    assert len(results) > 0


def test_project_map_contains_workspace_summary(sample_codebase):
    indexer = _import_indexer()
    result = indexer.index_codebase(sample_codebase)
    ws = result.get("map", {}).get("workspace", {})
    assert len(ws.get("manifests", [])) > 0
    assert len(ws.get("package_managers", [])) > 0
