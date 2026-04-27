"""
Tests for universal text indexing:
- Extended SUPPORTED_EXTENSIONS (R, SQL, Markdown, etc.)
- Binary file detection and skipping
- File size guard
- Jupyter notebook cell extraction
"""

import json
import importlib
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class _FakeSentenceTransformer:
    def __init__(self, *_args, **_kwargs):
        pass

    def encode(self, _query):  # pragma: no cover - not used in these tests
        return [0.1, 0.2, 0.3]


class _FakeCollection:
    def count(self):  # pragma: no cover - not used in these tests
        return 0

    def get(self, **_kwargs):  # pragma: no cover - not used in these tests
        return {"documents": [], "metadatas": []}


class _FakeChromaClient:
    def __init__(self, *_args, **_kwargs):
        pass

    def get_or_create_collection(self, *_args, **_kwargs):
        return _FakeCollection()


def _import_indexer_with_stubs():
    fake_sentence_transformers = types.ModuleType("sentence_transformers")
    fake_sentence_transformers.SentenceTransformer = _FakeSentenceTransformer
    sys.modules["sentence_transformers"] = fake_sentence_transformers

    fake_chromadb = types.ModuleType("chromadb")
    fake_chromadb.PersistentClient = _FakeChromaClient
    sys.modules["chromadb"] = fake_chromadb

    import indexer
    return importlib.reload(indexer)


_indexer = _import_indexer_with_stubs()
MAX_FILE_BYTES = _indexer.MAX_FILE_BYTES
SUPPORTED_EXTENSIONS = _indexer.SUPPORTED_EXTENSIONS
_chunk_notebook = _indexer._chunk_notebook
_collect_files = _indexer._collect_files
_is_binary = _indexer._is_binary
_chunk_file = _indexer._chunk_file


class TestSupportedExtensions(unittest.TestCase):
    """Critical extensions that must be present for data-science / stats teams."""

    def test_r_scripts_supported(self):
        self.assertIn(".r", SUPPORTED_EXTENSIONS)

    def test_sql_supported(self):
        self.assertIn(".sql", SUPPORTED_EXTENSIONS)

    def test_markdown_supported(self):
        self.assertIn(".md", SUPPORTED_EXTENSIONS)

    def test_jupyter_notebooks_supported(self):
        self.assertIn(".ipynb", SUPPORTED_EXTENSIONS)

    def test_yaml_supported(self):
        self.assertIn(".yaml", SUPPORTED_EXTENSIONS)
        self.assertIn(".yml", SUPPORTED_EXTENSIONS)

    def test_shell_scripts_supported(self):
        self.assertIn(".sh", SUPPORTED_EXTENSIONS)

    def test_plaintext_supported(self):
        self.assertIn(".txt", SUPPORTED_EXTENSIONS)

    def test_original_code_extensions_still_present(self):
        for ext in (".py", ".js", ".ts", ".go", ".rs", ".java"):
            with self.subTest(ext=ext):
                self.assertIn(ext, SUPPORTED_EXTENSIONS)


class TestBinaryDetection(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _write(self, name, data, mode="wb"):
        p = self.tmp / name
        p.write_bytes(data) if mode == "wb" else p.write_text(data, encoding="utf-8")
        return p

    def test_null_byte_is_binary(self):
        p = self._write("blob.r", b"some text\x00more text")
        self.assertTrue(_is_binary(p))

    def test_pdf_header_is_binary(self):
        p = self._write("report.pdf", b"%PDF-1.4\x00\x01\x02\x03" + b"\x80" * 100)
        self.assertTrue(_is_binary(p))

    def test_png_header_is_binary(self):
        p = self._write("chart.png", b"\x89PNG\r\n\x1a\n" + b"\x00\x01\x02" * 100)
        self.assertTrue(_is_binary(p))

    def test_plain_r_script_is_not_binary(self):
        r_code = "library(dplyr)\ndf <- read.csv('data.csv')\ndf %>% filter(year == 2024)\n"
        p = self._write("analysis.r", r_code.encode())
        self.assertFalse(_is_binary(p))

    def test_plain_sql_is_not_binary(self):
        sql = "SELECT year, COUNT(*) as total\nFROM census\nWHERE region = 'Norte'\nGROUP BY year;\n"
        p = self._write("query.sql", sql.encode())
        self.assertFalse(_is_binary(p))

    def test_plain_markdown_is_not_binary(self):
        md = "# Project Readme\n\nThis project processes census data.\n\n## Usage\n\nRun `main.R`.\n"
        p = self._write("README.md", md.encode())
        self.assertFalse(_is_binary(p))

    def test_empty_file_is_not_binary(self):
        p = self._write("empty.md", b"")
        self.assertFalse(_is_binary(p))


class TestCollectFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _write(self, rel, content, encoding="utf-8"):
        p = self.tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_text(content, encoding=encoding)
        return p

    def test_collects_r_files(self):
        self._write("analysis.R", "x <- 1\n")
        files = _collect_files(self.tmp)
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].suffix, ".R")

    def test_collects_uppercase_supported_extensions(self):
        self._write("QUERY.SQL", "SELECT 1;\n")
        self._write("README.MD", "# Docs\n")
        self._write("SCRIPT.SH", "echo hello\n")
        self._write("MODEL.R", "x <- 1\n")
        files = _collect_files(self.tmp)
        names = {f.name for f in files}
        self.assertIn("QUERY.SQL", names)
        self.assertIn("README.MD", names)
        self.assertIn("SCRIPT.SH", names)
        self.assertIn("MODEL.R", names)

    def test_collects_sql_files(self):
        self._write("query.sql", "SELECT 1;\n")
        files = _collect_files(self.tmp)
        self.assertEqual(len(files), 1)

    def test_collects_canonical_dotenv_basename(self):
        self._write(".env", "API_TOKEN=abc123\n")
        files = _collect_files(self.tmp)
        self.assertEqual([f.name for f in files], [".env"])

    def test_collects_dotenv_with_extension(self):
        self._write("config.env", "LOG_LEVEL=debug\n")
        files = _collect_files(self.tmp)
        self.assertEqual([f.name for f in files], ["config.env"])

    def test_skips_binary_canonical_dotenv(self):
        self._write(".env", b"\x00\x01\x02\x03" + b"\x80" * 64)
        files = _collect_files(self.tmp)
        self.assertEqual(files, [])

    def test_skips_oversized_files(self):
        self._write("huge.md", "x\n" * (MAX_FILE_BYTES + 1))
        files = _collect_files(self.tmp)
        self.assertEqual(len(files), 0)

    def test_collects_oversized_manifest_file(self):
        self._write("poetry.lock", "pkg = 'x'\n" * (MAX_FILE_BYTES + 1))
        files = _collect_files(self.tmp)
        self.assertEqual([f.name for f in files], ["poetry.lock"])

    def test_skips_binary_files(self):
        # A .md file that is actually binary
        self._write("corrupt.md", b"%PDF\x00\x01\x02\x03" + b"\x80" * 200)
        files = _collect_files(self.tmp)
        self.assertEqual(len(files), 0)

    def test_skips_skip_dirs(self):
        self._write("__pycache__/analysis.py", "x = 1\n")
        self._write("src/main.py", "def main(): pass\n")
        files = _collect_files(self.tmp)
        paths = [f.name for f in files]
        self.assertIn("main.py", paths)
        self.assertNotIn("analysis.py", paths)

    def test_collects_mixed_types(self):
        self._write("src/pipeline.py", "def run(): pass\n")
        self._write("src/query.sql", "SELECT * FROM table;\n")
        self._write("src/model.R", "lm(y ~ x, data=df)\n")
        self._write("README.md", "# Docs\n")
        files = _collect_files(self.tmp)
        suffixes = {f.suffix for f in files}
        self.assertIn(".py", suffixes)
        self.assertIn(".sql", suffixes)
        self.assertIn(".R", suffixes)
        self.assertIn(".md", suffixes)


class TestNotebookChunking(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _make_notebook(self, cells):
        nb = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {},
            "cells": cells,
        }
        p = self.tmp / "notebook.ipynb"
        p.write_text(json.dumps(nb), encoding="utf-8")
        return p

    def test_extracts_code_cells(self):
        p = self._make_notebook([
            {"cell_type": "code", "source": ["import pandas as pd\n", "df = pd.read_csv('data.csv')\n"], "metadata": {}, "outputs": []},
        ])
        chunks = _chunk_notebook(p, self.tmp)
        self.assertGreater(len(chunks), 0)
        combined = " ".join(c["content"] for c in chunks)
        self.assertIn("import pandas", combined)

    def test_extracts_markdown_cells(self):
        p = self._make_notebook([
            {"cell_type": "markdown", "source": ["## Analysis\n", "This section loads the census data."], "metadata": {}},
        ])
        chunks = _chunk_notebook(p, self.tmp)
        self.assertGreater(len(chunks), 0)
        combined = " ".join(c["content"] for c in chunks)
        self.assertIn("Analysis", combined)

    def test_cell_type_prefix_present(self):
        p = self._make_notebook([
            {"cell_type": "code", "source": ["x = 1"], "metadata": {}, "outputs": []},
            {"cell_type": "markdown", "source": ["# Title"], "metadata": {}},
        ])
        chunks = _chunk_notebook(p, self.tmp)
        combined = " ".join(c["content"] for c in chunks)
        self.assertIn("[code]", combined)
        self.assertIn("[markdown]", combined)

    def test_empty_cells_skipped(self):
        p = self._make_notebook([
            {"cell_type": "code", "source": [], "metadata": {}, "outputs": []},
            {"cell_type": "code", "source": ["x = 1"], "metadata": {}, "outputs": []},
        ])
        chunks = _chunk_notebook(p, self.tmp)
        combined = " ".join(c["content"] for c in chunks)
        self.assertIn("x = 1", combined)

    def test_invalid_json_returns_empty(self):
        p = self.tmp / "bad.ipynb"
        p.write_text("this is not json", encoding="utf-8")
        chunks = _chunk_notebook(p, self.tmp)
        self.assertEqual(chunks, [])

    def test_source_as_string_handled(self):
        """Some notebook formats store source as a single string, not a list."""
        p = self._make_notebook([
            {"cell_type": "code", "source": "import os\nprint(os.getcwd())", "metadata": {}, "outputs": []},
        ])
        chunks = _chunk_notebook(p, self.tmp)
        combined = " ".join(c["content"] for c in chunks)
        self.assertIn("import os", combined)

    def test_uppercase_ipynb_chunked_as_notebook_cells(self):
        nb = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {},
            "cells": [
                {"cell_type": "markdown", "source": ["# Header\n"], "metadata": {}},
                {"cell_type": "code", "source": ["print('hello')\n"], "metadata": {}, "outputs": []},
            ],
        }
        p = self.tmp / "NOTEBOOK.IPYNB"
        p.write_text(json.dumps(nb), encoding="utf-8")

        chunks = _chunk_file(p, self.tmp)
        self.assertGreater(len(chunks), 0)
        combined = " ".join(c["content"] for c in chunks)

        self.assertIn("[markdown]", combined)
        self.assertIn("[code]", combined)
        self.assertIn("Header", combined)
        self.assertIn("print('hello')", combined)
        self.assertNotIn('"cells"', combined)


if __name__ == "__main__":
    unittest.main()
