import tempfile
import shutil
import time
import unittest
from pathlib import Path

from andes_cache.fingerprint import RepoFingerprinter
from andes_cache.keys import build_key, normalize_query
from andes_cache.manager import AndesCacheManager
from andes_cache.prompt import build_prompt_sections, serialize_prompt_sections
from andes_cache.store import DiskCacheStore
from andes_cache.versions import INDEX_VERSION, PARSER_VERSION, PROMPT_TEMPLATE_VERSION, RETRIEVAL_POLICY_VERSION


class TestCacheKeys(unittest.TestCase):
    def test_query_normalization(self):
        self.assertEqual(normalize_query("  Explain   Foo  "), "explain foo")

    def test_deterministic_key(self):
        a = build_key("ret", repo_fp="x", q="y")
        b = build_key("ret", q="y", repo_fp="x")
        self.assertEqual(a, b)


class TestCacheStore(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_store_roundtrip(self):
        store = DiskCacheStore(self.tmp, "1")
        store.set("retrieval", "abc", {"ok": True})
        self.assertEqual(store.get("retrieval", "abc")["ok"], True)

    def test_schema_invalidation(self):
        store = DiskCacheStore(self.tmp, "1")
        store.set("retrieval", "abc", {"ok": True})
        DiskCacheStore(self.tmp, "2")
        self.assertIsNone(DiskCacheStore(self.tmp, "2").get("retrieval", "abc"))


class TestFingerprintAndInvalidation(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _fp(self, hashes):
        return RepoFingerprinter.build(
            self.tmp,
            hashes,
            index_version=INDEX_VERSION,
            parser_version=PARSER_VERSION,
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
            retrieval_policy_version=RETRIEVAL_POLICY_VERSION,
        )

    def test_fingerprint_changes_when_file_changes(self):
        fp1 = self._fp({"a.py": "1"})
        fp2 = self._fp({"a.py": "2"})
        self.assertNotEqual(fp1, fp2)

    def test_no_cross_repo_reuse(self):
        mgr = AndesCacheManager(self.tmp / "cache")
        mgr.retrieval_set(repo_fp="repoA", query="q", index_version="v1", value=[{"x": 1}])
        self.assertIsNone(mgr.retrieval_get(repo_fp="repoB", query="q", index_version="v1"))


class TestPromptDeterminism(unittest.TestCase):
    def test_deterministic_prompt_serialization(self):
        sections = build_prompt_sections(
            system_prefix="S",
            workspace_prefix="W",
            retrieval_context="R",
            user_turn="U",
        )
        p1 = serialize_prompt_sections(sections)
        p2 = serialize_prompt_sections(sections)
        self.assertEqual(p1, p2)
        self.assertIn("## SYSTEM PREFIX", p1)


class TestLayeredCaches(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.mgr = AndesCacheManager(self.tmp / "cache")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_workspace_cache(self):
        self.mgr.workspace_set("fp", "module_graph", {"m": 1})
        self.assertEqual(self.mgr.workspace_get("fp", "module_graph"), {"m": 1})

    def test_retrieval_cache(self):
        self.mgr.retrieval_set(repo_fp="fp", query="Explain auth", index_version="v1", value=[{"f": "auth.py"}])
        self.assertEqual(self.mgr.retrieval_get(repo_fp="fp", query="explain   auth", index_version="v1")[0]["f"], "auth.py")

    def test_partial_invalidation(self):
        self.mgr.workspace_set("fp", "module_graph", {"m": 1})
        self.mgr.retrieval_set(repo_fp="fp", query="q", index_version="v1", value=[1])
        removed = self.mgr.invalidate_repo("fp", include_workspace=False)
        self.assertGreater(removed["retrieval"], 0)
        self.assertIsNotNone(self.mgr.workspace_get("fp", "module_graph"))

    def test_patch_plan_reuse(self):
        self.mgr.patch_plan_set(repo_fp="fp", query="fix auth", target_signature="auth.py", value={"plan": ["x"]})
        self.assertIsNotNone(self.mgr.patch_plan_get(repo_fp="fp", query="fix auth", target_signature="auth.py"))
        self.assertIsNone(self.mgr.patch_plan_get(repo_fp="fp", query="fix auth", target_signature="other.py"))

    def test_unsafe_semantic_not_reused_for_patch(self):
        self.mgr.semantic_set(
            repo_fp="fp",
            query="architecture overview",
            retrieval_signature="sig",
            safe_class="descriptive",
            value="cached",
        )
        hit = self.mgr.semantic_get(
            repo_fp="fp",
            query="architecture overview",
            retrieval_signature="sig",
            safe_class="code_patch",
        )
        self.assertIsNone(hit)


class TestBenchmarkUtility(unittest.TestCase):
    def test_cold_warm_metrics(self):
        tmp = Path(tempfile.mkdtemp())
        mgr = AndesCacheManager(tmp / "cache")
        t0 = time.perf_counter()
        self.assertIsNone(mgr.retrieval_get(repo_fp="fp", query="q", index_version="1"))
        cold = (time.perf_counter() - t0) * 1000
        mgr.retrieval_set(repo_fp="fp", query="q", index_version="1", value=[{"ok": True}])
        t1 = time.perf_counter()
        self.assertIsNotNone(mgr.retrieval_get(repo_fp="fp", query="q", index_version="1"))
        warm = (time.perf_counter() - t1) * 1000
        mgr.flush_metrics()
        self.assertTrue((tmp / "cache" / "metrics" / "cache_metrics.json").exists())
        self.assertGreaterEqual(cold, 0)
        self.assertGreaterEqual(warm, 0)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
