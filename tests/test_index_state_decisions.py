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
            class _FakeVectors(list):
                def tolist(self):
                    return list(self)

            return _FakeVectors([[0.0] * 3 for _ in texts])

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

        tmp = Path(tempfile.mkdtemp(dir=Path.cwd()))
        try:
            self.indexer.INDEX_STATE = tmp / "index_state.json"
            self.indexer.INDEX_STATE.write_text("{not-json")
            loaded = self.indexer._load_index_state()
            self.assertEqual(loaded, {})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_manifest_discovery_refresh_path_with_workspace_rebuild(self):
        tmp = Path(tempfile.mkdtemp(dir=Path.cwd()))
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

    def test_workspace_only_rebuild_uses_all_files_scope_for_missed_files(self):
        tmp = Path(tempfile.mkdtemp(dir=Path.cwd()))
        original_hash_store = self.indexer.HASH_STORE
        original_project_map = self.indexer.PROJECT_MAP
        original_symbol_index = self.indexer.SYMBOL_INDEX
        original_workspace_index = self.indexer.WORKSPACE_INDEX
        original_index_state = self.indexer.INDEX_STATE
        original_cache = self.indexer.CACHE
        original_eval = self.indexer.evaluate_index_state
        try:
            self.indexer.HASH_STORE = tmp / ".file_hashes.json"
            self.indexer.PROJECT_MAP = tmp / "project_map.json"
            self.indexer.SYMBOL_INDEX = tmp / "symbol_index.json"
            self.indexer.WORKSPACE_INDEX = tmp / "workspace_index.json"
            self.indexer.INDEX_STATE = tmp / "index_state.json"
            self.indexer.CACHE = self.indexer.AndesCacheManager(tmp / "cache")

            src = tmp / "src"
            src.mkdir(parents=True, exist_ok=True)
            code = src / "a.py"
            manifest = tmp / "AndroidManifest.xml"
            code.write_text("def a():\n    return 1\n")

            # Baseline index snapshot when manifest does not yet exist.
            py_hash = self.indexer._file_hash(code)
            self.indexer._save_hashes(
                {
                    "__root__": str(tmp.resolve()),
                    "src/a.py": py_hash,
                    "__fingerprint__": "prev-fp",
                }
            )
            self.indexer._save_index_state({"repo_root": str(tmp.resolve()), "index_version": "1"})
            baseline_workspace = self.indexer.build_workspace_index(
                tmp,
                [code],
                self.indexer._chunk_file(code, tmp),
                repo_fingerprint="prev-fp",
                changed_paths=[],
                force_refresh=True,
            )
            self.assertNotIn("AndroidManifest.xml", baseline_workspace["manifests"])

            # Add a newly detectable file and force workspace-only rebuild decision.
            manifest.write_text("<manifest package=\"com.example\"/>")

            def _force_workspace_only(_current, _stored, repo_changed):
                return {
                    "decision": self.indexer.DECISION_REBUILD_WORKSPACE_ONLY,
                    "reasons": ["Workspace extraction version changed; rebuilding workspace metadata"],
                }

            self.indexer.evaluate_index_state = _force_workspace_only
            events = list(self.indexer.index_codebase_stream(str(tmp)))
            done = [e for e in events if e.get("type") == "done"][-1]
            self.assertEqual(done.get("decision"), self.indexer.DECISION_REBUILD_WORKSPACE_ONLY)
            self.assertEqual(done.get("workspace_rebuild_scope"), "all_files")
            self.assertTrue(done.get("workspace_only_rebuild_preserved_embedding_state"))
            self.assertIn("AndroidManifest.xml", done["map"]["workspace"]["manifests"])

            # Workspace-only rebuild must not mark newly discovered files as embedded.
            post_workspace_hashes = self.indexer._load_hashes()
            self.assertNotIn("AndroidManifest.xml", post_workspace_hashes)

            # A subsequent run should still detect the manifest as needing embedding.
            self.indexer.evaluate_index_state = original_eval
            events_after = list(self.indexer.index_codebase_stream(str(tmp)))
            done_after = [e for e in events_after if e.get("type") == "done"][-1]
            self.assertEqual(done_after.get("decision"), self.indexer.DECISION_INCREMENTAL_REINDEX)
            self.assertEqual(done_after.get("new"), 1)
        finally:
            self.indexer.HASH_STORE = original_hash_store
            self.indexer.PROJECT_MAP = original_project_map
            self.indexer.SYMBOL_INDEX = original_symbol_index
            self.indexer.WORKSPACE_INDEX = original_workspace_index
            self.indexer.INDEX_STATE = original_index_state
            self.indexer.CACHE = original_cache
            self.indexer.evaluate_index_state = original_eval
            shutil.rmtree(tmp, ignore_errors=True)

    def test_authoritative_discovery_finds_deep_nested_files_and_respects_skip_dirs(self):
        tmp = Path(tempfile.mkdtemp(dir=Path.cwd()))
        try:
            # Nested multi-module / monorepo-like structure.
            (tmp / "android" / "app" / "src" / "main").mkdir(parents=True, exist_ok=True)
            (tmp / "android" / "feature" / "payments").mkdir(parents=True, exist_ok=True)
            (tmp / "services" / "api").mkdir(parents=True, exist_ok=True)
            (tmp / "packages" / "python-core").mkdir(parents=True, exist_ok=True)
            (tmp / "apps" / "mobile").mkdir(parents=True, exist_ok=True)
            (tmp / "node_modules" / "left-pad").mkdir(parents=True, exist_ok=True)
            (tmp / ".git" / "hooks").mkdir(parents=True, exist_ok=True)

            (tmp / "android" / "app" / "src" / "main" / "AndroidManifest.xml").write_text("<manifest/>")
            (tmp / "android" / "feature" / "payments" / "build.gradle.kts").write_text("plugins {}\n")
            (tmp / "android" / "settings.gradle.kts").write_text("rootProject.name = \"android\"\n")
            (tmp / "services" / "api" / "package.json").write_text("{\"name\": \"api\"}\n")
            (tmp / "packages" / "python-core" / "pyproject.toml").write_text("[project]\nname='core'\n")
            (tmp / "apps" / "mobile" / "settings.gradle.kts").write_text("include(\":app\")\n")

            # Should be skipped due to skip dirs.
            (tmp / "node_modules" / "left-pad" / "package.json").write_text("{\"name\": \"left-pad\"}\n")
            (tmp / ".git" / "hooks" / "package.json").write_text("{\"name\": \"hook\"}\n")

            manifests = self.indexer._discover_manifests(tmp)

            self.assertIn("android/app/src/main/AndroidManifest.xml", manifests)
            self.assertIn("android/feature/payments/build.gradle.kts", manifests)
            self.assertIn("android/settings.gradle.kts", manifests)
            self.assertIn("apps/mobile/settings.gradle.kts", manifests)
            self.assertIn("services/api/package.json", manifests)
            self.assertIn("packages/python-core/pyproject.toml", manifests)

            self.assertNotIn("node_modules/left-pad/package.json", manifests)
            self.assertNotIn(".git/hooks/package.json", manifests)
            self.assertNotIn("package.json", manifests)  # ensure relative nested paths are preserved
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
