import importlib
import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path


def _import_indexer_with_stubs():
    fake_st = types.ModuleType("sentence_transformers")

    class FakeModel:
        def __init__(self, *_args, **_kwargs):
            pass

        def encode(self, texts, show_progress_bar=False):
            return [[0.0] * 3 for _ in texts]

    fake_st.SentenceTransformer = FakeModel
    sys.modules["sentence_transformers"] = fake_st

    fake_chroma = types.ModuleType("chromadb")

    class FakeCollection:
        def count(self):
            return 0

        def upsert(self, **_kwargs):
            return None

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def get_or_create_collection(self, *_args, **_kwargs):
            return FakeCollection()

        def delete_collection(self, *_args, **_kwargs):
            return None

    fake_chroma.PersistentClient = FakeClient
    sys.modules["chromadb"] = fake_chroma

    import indexer  # noqa: WPS433

    return importlib.reload(indexer)


class TestIndexStateDecisions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.indexer = _import_indexer_with_stubs()

    def test_workspace_only_rebuild_when_workspace_version_changes(self):
        current = {"repo_root": "/repo", "index_version": "1", "parser_version": "1", "workspace_extraction_version": "new"}
        stored = {"repo_root": "/repo", "index_version": "1", "parser_version": "1", "workspace_extraction_version": "old"}
        decision = self.indexer.evaluate_index_state(current, stored, repo_changed=False)
        self.assertEqual(decision["decision"], self.indexer.DECISION_REBUILD_WORKSPACE_ONLY)

    def test_full_rebuild_when_core_index_version_changes(self):
        current = {"repo_root": "/repo", "index_version": "2", "parser_version": "1"}
        stored = {"repo_root": "/repo", "index_version": "1", "parser_version": "1"}
        decision = self.indexer.evaluate_index_state(current, stored, repo_changed=False)
        self.assertEqual(decision["decision"], self.indexer.DECISION_FULL_REBUILD)

    def test_incremental_reindex_when_files_changed(self):
        current = {"repo_root": "/repo", "index_version": "1", "parser_version": "1"}
        stored = {"repo_root": "/repo", "index_version": "1", "parser_version": "1"}
        decision = self.indexer.evaluate_index_state(current, stored, repo_changed=True)
        self.assertEqual(decision["decision"], self.indexer.DECISION_INCREMENTAL_REINDEX)

    def test_reuse_all_when_nothing_changed(self):
        current = {"repo_root": "/repo", "index_version": "1", "parser_version": "1"}
        stored = {"repo_root": "/repo", "index_version": "1", "parser_version": "1"}
        decision = self.indexer.evaluate_index_state(current, stored, repo_changed=False)
        self.assertEqual(decision["decision"], self.indexer.DECISION_REUSE_ALL)

    def test_missing_or_corrupt_state_forces_safe_rebuild(self):
        d1 = self.indexer.evaluate_index_state({"repo_root": "/repo"}, None, repo_changed=False)
        self.assertEqual(d1["decision"], self.indexer.DECISION_FULL_REBUILD)

        tmp = Path(tempfile.mkdtemp())
        try:
            self.indexer.INDEX_STATE = tmp / "index_state.json"
            self.indexer.INDEX_STATE.write_text("{not-json")
            loaded = self.indexer._load_index_state()
            self.assertEqual(loaded, {})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_manifest_discovery_refresh_path_with_workspace_rebuild(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / "src").mkdir(parents=True, exist_ok=True)
            (tmp / "src" / "a.py").write_text("def a():\n    return 1\n")
            files = [tmp / "src" / "a.py"]
            chunks = [{"file": "src/a.py", "symbols": "a"}]

            ws1 = self.indexer.build_workspace_index(
                tmp,
                files,
                chunks,
                repo_fingerprint="fp-test",
                changed_paths=[],
                force_refresh=True,
            )
            self.assertEqual(ws1["manifests"], [])

            (tmp / "package.json").write_text(json.dumps({"name": "x"}))
            ws2 = self.indexer.build_workspace_index(
                tmp,
                files,
                chunks,
                repo_fingerprint="fp-test",
                changed_paths=[],
                force_refresh=True,
            )
            self.assertIn("package.json", ws2["manifests"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
