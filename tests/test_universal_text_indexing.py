"""
Tests for universal text indexing:
- Extended SUPPORTED_EXTENSIONS (R, SQL, Markdown, etc.)
- Binary file detection and skipping
- File size guard
- Jupyter notebook cell extraction
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from indexer import (
    MAX_FILE_BYTES,
    SUPPORTED_EXTENSIONS,
    _chunk_notebook,
    _collect_files,
    _is_binary,
)


class TestSupportedExtensions(unittest.TestCase):
    """Critical extensions that must be present for data-science / stats teams."""

    def test_r_scripts_supported(self):
        self.assertIn(".r", SUPPORTED_EXTENSIONS)
        self.assertIn(".R", SUPPORTED_EXTENSIONS)

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

    def test_collects_sql_files(self):
        self._write("query.sql", "SELECT 1;\n")
        files = _collect_files(self.tmp)
        self.assertEqual(len(files), 1)

    def test_skips_oversized_files(self):
        self._write("huge.md", "x\n" * (MAX_FILE_BYTES + 1))
        files = _collect_files(self.tmp)
        self.assertEqual(len(files), 0)

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


if __name__ == "__main__":
    unittest.main()
