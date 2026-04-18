import unittest

from andes_cache.routing import (
    classify_query_intent,
    retrieval_route_for_intent,
    orchestration_plan,
)
from andes_cache.source_of_truth import (
    config_priority_files,
    summarize_declared_permissions,
    missing_manifest_notice,
)
from andes_cache.manager import AndesCacheManager


class TestIntentClassification(unittest.TestCase):
    def test_permissions_query_routes_to_config(self):
        q = "list the permissions declared"
        intent = classify_query_intent(q)
        self.assertEqual(intent, "declaration_or_configuration")
        self.assertEqual(retrieval_route_for_intent(intent), "source_of_truth")

    def test_dependency_query_routes_to_config(self):
        q = "what libraries/dependencies are used"
        intent = classify_query_intent(q)
        self.assertEqual(intent, "dependency_or_build_inventory")
        self.assertEqual(retrieval_route_for_intent(intent), "source_of_truth")

    def test_runtime_usage_query_routes_to_runtime(self):
        q = "where is auth token used"
        intent = classify_query_intent(q)
        self.assertEqual(intent, "runtime_usage_or_reference")
        self.assertEqual(retrieval_route_for_intent(intent), "runtime_usage")


class TestSourceOfTruthBehavior(unittest.TestCase):
    def test_android_manifest_permissions_extraction(self):
        chunks = [{
            "file": "app/src/main/AndroidManifest.xml",
            "content": '<uses-permission android:name="android.permission.CAMERA"/>\n'
                       '<uses-permission android:name="android.permission.INTERNET"/>',
        }]
        perms = summarize_declared_permissions(chunks)
        self.assertEqual(
            perms,
            ["android.permission.CAMERA", "android.permission.INTERNET"],
        )

    def test_missing_manifest_message(self):
        msg = missing_manifest_notice()["content"].lower()
        self.assertIn("no androidmanifest.xml", msg)
        self.assertIn("cannot be confirmed", msg)
        self.assertIn("inferences", msg)

    def test_dependency_files_are_prioritized(self):
        manifests = ["package.json", "app/build.gradle", "requirements.txt", "Dockerfile"]
        ordered = config_priority_files(
            intent="dependency_or_build_inventory",
            query="what dependencies are declared",
            manifests=manifests,
            config_files=[],
        )
        self.assertTrue(ordered.index("app/build.gradle") < ordered.index("requirements.txt"))
        self.assertIn("package.json", ordered)
        self.assertIn("Dockerfile", ordered)

    def test_declaration_without_authoritative_file_requires_limitation(self):
        msg = missing_manifest_notice()["content"]
        self.assertIn("Decla", msg)


class TestRouteIsolationAndFastPath(unittest.TestCase):
    def test_retrieval_cache_route_isolation(self):
        import tempfile
        from pathlib import Path
        import shutil

        tmp = tempfile.mkdtemp()
        try:
            mgr = AndesCacheManager(Path(tmp) / "cache")
            mgr.retrieval_set(
                repo_fp="fp",
                query="where is x configured",
                index_version="v1",
                intent="declaration_or_configuration",
                retrieval_route="source_of_truth",
                value=[{"route": "source_of_truth"}],
            )
            self.assertIsNone(
                mgr.retrieval_get(
                    repo_fp="fp",
                    query="where is x configured",
                    index_version="v1",
                    intent="generic_semantic",
                    retrieval_route="semantic",
                )
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_fast_path_skips_patch_plan(self):
        plan = orchestration_plan("declaration_or_configuration")
        self.assertTrue(plan["skip_patch_plan"])
        self.assertTrue(plan["skip_patch_diagnosis"])
        self.assertTrue(plan["skip_neighborhood"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
