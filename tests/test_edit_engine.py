import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from edit_engine import EditOperation, FileEditEngine, generate_diff_preview


class TestFileEditEngine(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=Path.cwd()))
        self.root = self.tmp / "repo"
        self.root.mkdir(parents=True, exist_ok=True)
        self.file_path = self.root / "src" / "main.py"
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.write_text("def greet():\n    return 'hello'\n", encoding="utf-8")

        self.hash_store = self.tmp / ".file_hashes.json"
        self._write_hash_store(self.file_path)
        self.reindexed = []
        self.engine = FileEditEngine(
            hash_store_path=self.hash_store,
            reindex_file=lambda root, rel: self._capture_reindex(root, rel),
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_hash_store(self, tracked_file: Path):
        rel = str(tracked_file.relative_to(self.root))
        content_hash = hashlib.md5(tracked_file.read_bytes()).hexdigest()
        self.hash_store.write_text(
            json.dumps({"__root__": str(self.root), rel: content_hash}),
            encoding="utf-8",
        )

    def _capture_reindex(self, root_path: Path, rel_path: str) -> bool:
        self.reindexed.append((str(root_path), rel_path))
        return True

    def test_successful_exact_replacement(self):
        edit = EditOperation(
            file_path="src/main.py",
            old_content="return 'hello'",
            new_content="return 'hello world'",
        )
        result = self.engine.apply_edit_operation(edit)

        self.assertTrue(result.success)
        self.assertIsNone(result.error)
        self.assertIn("hello world", self.file_path.read_text(encoding="utf-8"))
        self.assertEqual(len(self.reindexed), 1)
        self.assertEqual(self.reindexed[0][1], "src/main.py")

    def test_failure_when_old_content_not_found(self):
        edit = EditOperation(
            file_path="src/main.py",
            old_content="return 'missing'",
            new_content="return 'updated'",
        )
        result = self.engine.apply_edit_operation(edit)

        self.assertFalse(result.success)
        self.assertIn("old_content not found", result.error)

    def test_failure_on_hash_mismatch(self):
        self.file_path.write_text("def greet():\n    return 'changed'\n", encoding="utf-8")

        edit = EditOperation(
            file_path="src/main.py",
            old_content="return 'changed'",
            new_content="return 'updated'",
        )
        result = self.engine.apply_edit_operation(edit)

        self.assertFalse(result.success)
        self.assertIn("Hash mismatch", result.error)

    def test_failure_when_file_missing(self):
        self.file_path.unlink()

        edit = EditOperation(
            file_path="src/main.py",
            old_content="return 'hello'",
            new_content="return 'updated'",
        )
        result = self.engine.apply_edit_operation(edit)

        self.assertFalse(result.success)
        self.assertEqual("File missing on disk", result.error)


class TestDiffPreview(unittest.TestCase):
    def test_generates_unified_diff(self):
        diff = generate_diff_preview(
            "line1\nline2\n",
            "line1\nline2 changed\n",
            file_path="src/main.py",
        )

        self.assertIn("--- a/src/main.py", diff)
        self.assertIn("+++ b/src/main.py", diff)
        self.assertIn("-line2", diff)
        self.assertIn("+line2 changed", diff)


if __name__ == "__main__":
    unittest.main()
