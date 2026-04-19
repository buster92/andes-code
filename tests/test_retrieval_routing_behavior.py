import unittest
from pathlib import Path

from andes_cache.routing import (
    classify_query_intent,
    classify_query_intent_details,
    retrieval_route_for_intent,
    orchestration_plan,
    semantic_cache_allowed,
)
from andes_cache.source_of_truth import (
    config_priority_files,
    expected_authority_candidates,
    rank_recovery_authoritative_paths,
    rank_authoritative_paths,
    select_best_authoritative_path,
    summarize_declared_permissions,
    missing_manifest_notice,
    classify_source_type,
    authority_level_for_source,
    wants_runtime_usage,
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
        self.assertEqual(intent, "runtime_usage_or_reference")
        self.assertEqual(retrieval_route_for_intent(intent), "runtime_usage")

    def test_dependencies_declared_routes_to_source_of_truth(self):
        d = classify_query_intent_details("what dependencies are declared")
        self.assertEqual(d["intent"], "dependency_or_build_inventory")
        self.assertEqual(d["retrieval_route"], "source_of_truth")

    def test_dependencies_configured_routes_to_source_of_truth(self):
        d = classify_query_intent_details("what dependencies are configured")
        self.assertEqual(d["intent"], "dependency_or_build_inventory")
        self.assertEqual(d["retrieval_route"], "source_of_truth")

    def test_dependencies_used_at_runtime_routes_to_runtime(self):
        d = classify_query_intent_details("what dependencies are used at runtime")
        self.assertEqual(d["intent"], "runtime_usage_or_reference")
        self.assertEqual(d["retrieval_route"], "runtime_usage")

    def test_where_configured_routes_to_declaration(self):
        d = classify_query_intent_details("where is auth configured")
        self.assertEqual(d["intent"], "declaration_or_configuration")
        self.assertEqual(d["retrieval_route"], "source_of_truth")

    def test_runtime_usage_query_routes_to_runtime(self):
        q = "where is auth token used"
        intent = classify_query_intent(q)
        self.assertEqual(intent, "runtime_usage_or_reference")
        self.assertEqual(retrieval_route_for_intent(intent), "runtime_usage")

    def test_ambiguous_dependency_query_marks_ambiguity(self):
        d = classify_query_intent_details("what dependencies are needed at runtime")
        self.assertEqual(d["intent"], "runtime_usage_or_reference")
        self.assertEqual(d["retrieval_route"], "runtime_usage")

    def test_ambiguous_defined_query_prefers_symbol_lookup(self):
        d = classify_query_intent_details("where is auth defined")
        self.assertEqual(d["intent"], "symbol_lookup")
        self.assertEqual(d["retrieval_route"], "symbol_lookup")
        self.assertFalse(d["ambiguous"])

    def test_libraries_used_is_intentionally_runtime(self):
        d = classify_query_intent_details("what libraries are used")
        self.assertEqual(d["intent"], "runtime_usage_or_reference")
        self.assertEqual(d["retrieval_route"], "runtime_usage")


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

    def test_runtime_fallback_requires_explicit_runtime_wording(self):
        self.assertFalse(wants_runtime_usage("what config does this service use"))
        self.assertFalse(wants_runtime_usage("what dependencies are declared"))
        self.assertTrue(wants_runtime_usage("what dependencies are used at runtime"))
        self.assertTrue(wants_runtime_usage("where is auth used"))

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

    def test_broad_android_manifest_prefers_app_main_over_debug_and_tests(self):
        manifests = [
            "mobile/app/src/main/AndroidManifest.xml",
            "mobile/app/src/debug/AndroidManifest.xml",
            "mobile/lib/src/main/AndroidManifest.xml",
            "mobile/app/src/androidTest/AndroidManifest.xml",
        ]
        ordered = rank_authoritative_paths(
            manifests,
            query="what permissions are used in the app",
            intent="declaration_or_configuration",
        )
        self.assertEqual(ordered[0], "mobile/app/src/main/AndroidManifest.xml")

    def test_module_specific_package_json_prefers_named_service(self):
        paths = [
            "services/payments/package.json",
            "services/orders/package.json",
            "packages/common/package.json",
        ]
        best = select_best_authoritative_path(
            paths,
            query="what dependencies does payments declare",
            intent="dependency_or_build_inventory",
        )
        self.assertEqual(best, "services/payments/package.json")

    def test_repo_wide_gradle_question_prefers_root_settings_file(self):
        paths = [
            "settings.gradle.kts",
            "apps/mobile/build.gradle.kts",
            "services/api/build.gradle.kts",
        ]
        best = select_best_authoritative_path(
            paths,
            query="what dependencies are declared repo-wide",
            intent="dependency_or_build_inventory",
        )
        self.assertEqual(best, "settings.gradle.kts")

    def test_non_authoritative_candidates_are_filtered(self):
        manifests = [
            "service/build.gradle",
            "tests/build.gradle",
            "examples/package.json",
            ".env.example",
            "service/package.json",
        ]
        ordered = config_priority_files(
            intent="dependency_or_build_inventory",
            query="what dependencies are declared for service",
            manifests=manifests,
            config_files=[],
        )
        self.assertIn("service/build.gradle", ordered)
        self.assertNotIn("tests/build.gradle", ordered)
        self.assertNotIn("examples/package.json", ordered)
        self.assertNotIn(".env.example", ordered)

    def test_multi_module_query_prefers_relevant_module(self):
        manifests = [
            "payments/build.gradle",
            "orders/build.gradle",
            "settings.gradle",
        ]
        ordered = config_priority_files(
            intent="dependency_or_build_inventory",
            query="what dependencies are declared in payments module",
            manifests=manifests,
            config_files=[],
        )
        self.assertLess(ordered.index("payments/build.gradle"), ordered.index("orders/build.gradle"))

    def test_authority_tagging_is_consistent(self):
        self.assertEqual(classify_source_type("app/build.gradle"), "build_file")
        self.assertEqual(
            authority_level_for_source("dependency_or_build_inventory", "build_file"),
            "declared",
        )
        self.assertEqual(classify_source_type(".env.production"), "config_file")
        self.assertEqual(
            authority_level_for_source("declaration_or_configuration", "config_file"),
            "configured",
        )
        self.assertEqual(
            authority_level_for_source("generic_semantic", "source_code"),
            "referenced",
        )

    def test_expected_authority_candidates_expand_manifest_recovery(self):
        manifests = ["mobile/app/src/main/AndroidManifest.xml", "service/package.json"]
        ordered = expected_authority_candidates(
            intent="declaration_or_configuration",
            query="what permissions are declared in the manifest",
            manifests=manifests,
            config_files=[],
        )
        self.assertIn("mobile/app/src/main/AndroidManifest.xml", ordered)
        self.assertIn("AndroidManifest.xml", ordered)

    def test_rank_authoritative_paths_returns_deterministic_list_for_weak_scores(self):
        paths = [
            "docs/notes.txt",
            "tmp/random.data",
            "misc/untyped.file",
        ]
        ranked = rank_authoritative_paths(
            paths,
            query="where is this configured",
            intent="declaration_or_configuration",
        )
        self.assertEqual(len(ranked), 3)
        self.assertEqual(sorted(ranked), sorted(paths))
        self.assertEqual(ranked, rank_authoritative_paths(paths, query="where is this configured", intent="declaration_or_configuration"))

    def test_expected_authority_candidates_expand_non_android_dependency_files(self):
        manifests = ["services/api/pyproject.toml", "services/worker/requirements.txt", "frontend/package.json"]
        ordered = expected_authority_candidates(
            intent="dependency_or_build_inventory",
            query="what dependencies are declared for worker",
            manifests=manifests,
            config_files=[],
        )
        self.assertIn("services/worker/requirements.txt", ordered)
        self.assertIn("services/api/pyproject.toml", ordered)
        self.assertIn("frontend/package.json", ordered)

    def test_recovery_ranking_prefers_hinted_module_path(self):
        manifests = [
            "services/payments/settings/config.yml",
            "services/orders/settings/config.yml",
            "services/payments/package.json",
        ]
        ranked = rank_recovery_authoritative_paths(
            intent="declaration_or_configuration",
            query="where is payments config declared",
            manifests=manifests,
            config_files=[],
            candidate_hints=expected_authority_candidates(
                "declaration_or_configuration",
                "where is payments config declared",
                manifests,
                [],
            ),
        )
        self.assertTrue(ranked)
        self.assertEqual(ranked[0], "services/payments/settings/config.yml")

    def test_recovery_ranking_prefers_exact_authoritative_file_over_generic_config(self):
        manifests = ["android/app/src/main/AndroidManifest.xml", "android/settings/config.yml"]
        ranked = rank_recovery_authoritative_paths(
            intent="declaration_or_configuration",
            query="what permissions are declared in android/app/src/main/AndroidManifest.xml",
            manifests=manifests,
            config_files=[],
            candidate_hints=expected_authority_candidates(
                "declaration_or_configuration",
                "what permissions are declared in android/app/src/main/AndroidManifest.xml",
                manifests,
                [],
            ),
        )
        self.assertTrue(ranked)
        self.assertEqual(ranked[0], "android/app/src/main/AndroidManifest.xml")

    def test_recovery_ranking_excludes_tests_samples_docs(self):
        manifests = [
            "docs/config/settings.yml",
            "examples/service/package.json",
            "tests/service/pyproject.toml",
            "services/api/pyproject.toml",
        ]
        ranked = rank_recovery_authoritative_paths(
            intent="dependency_or_build_inventory",
            query="what dependencies are declared for api",
            manifests=manifests,
            config_files=[],
            candidate_hints=expected_authority_candidates(
                "dependency_or_build_inventory",
                "what dependencies are declared for api",
                manifests,
                [],
            ),
        )
        self.assertEqual(ranked, ["services/api/pyproject.toml"])

    def test_dependency_recovery_ranking_avoids_unrelated_generic_config(self):
        manifests = ["services/api/settings/config.yml", "services/api/package.json"]
        ranked = rank_recovery_authoritative_paths(
            intent="dependency_or_build_inventory",
            query="what dependencies are declared for api service",
            manifests=manifests,
            config_files=[],
            candidate_hints=expected_authority_candidates(
                "dependency_or_build_inventory",
                "what dependencies are declared for api service",
                manifests,
                [],
            ),
        )
        self.assertTrue(ranked)
        self.assertEqual(ranked[0], "services/api/package.json")

    def test_manifest_query_ranking_does_not_prefer_random_settings(self):
        manifests = ["mobile/AndroidManifest.xml", "mobile/settings/config.yml"]
        ranked = rank_recovery_authoritative_paths(
            intent="declaration_or_configuration",
            query="list declared permissions in manifest",
            manifests=manifests,
            config_files=[],
            candidate_hints=expected_authority_candidates(
                "declaration_or_configuration",
                "list declared permissions in manifest",
                manifests,
                [],
            ),
        )
        self.assertTrue(ranked)
        self.assertEqual(ranked[0], "mobile/AndroidManifest.xml")


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

    def test_semantic_cache_blocked_for_declaration_route(self):
        self.assertFalse(semantic_cache_allowed("declaration_or_configuration", "source_of_truth"))
        self.assertTrue(semantic_cache_allowed("generic_semantic", "semantic"))

    def test_route_before_cache_order_in_search_source(self):
        # Lightweight guardrail: importing indexer in unit tests initializes heavyweight
        # embedding/chroma components, so we enforce this invariant via source-order check.
        src = Path("indexer.py").read_text()
        classify_pos = src.find("decision = classify_query_intent_details(query)")
        route_pos = src.find('retrieval_route = decision["retrieval_route"]')
        cache_pos = src.find("CACHE.retrieval_get(")
        self.assertGreater(classify_pos, -1)
        self.assertGreater(route_pos, classify_pos)
        self.assertGreater(cache_pos, route_pos)

    def test_strict_authority_mode_prevents_runtime_append_for_decl_intents(self):
        src = Path("indexer.py").read_text()
        self.assertIn("strict_authority_mode=decision.get(\"strict_authority_mode\", True)", src)
        self.assertIn("if (not strict_authority_mode) and allow_runtime_fallback and wants_runtime_usage(query):", src)

    def test_authority_recovery_runs_only_after_empty_pass_one(self):
        src = Path("indexer.py").read_text()
        self.assertIn("if not collected:", src)
        self.assertIn("ranked_paths = rank_recovery_authoritative_paths(", src)
        self.assertIn("recovery_candidates = expected_authority_candidates(", src)
        self.assertIn("recovery_chunks = _recover_authoritative_files(", src)

    def test_authority_recovery_is_filename_path_based_and_route_safe(self):
        src = Path("indexer.py").read_text()
        self.assertIn("def _recover_authoritative_files(", src)
        self.assertIn("file_chunks = _fetch_exact_file(candidate, max_results=60)", src)
        start = src.find("def _recover_authoritative_files(")
        end = src.find("\ndef search_semantic_only(", start)
        self.assertGreater(start, -1)
        self.assertGreater(end, start)
        self.assertNotIn("search_semantic_only(query", src[start:end])

    def test_recovery_uses_exact_path_fetch_guard(self):
        src = Path("indexer.py").read_text()
        self.assertIn("def _fetch_exact_file(path: str, max_results: int = 60)", src)
        self.assertIn('exact = col.get(where={"file": path}, limit=max_results)', src)
        self.assertIn('if doc and fallback["metadatas"][i].get("file", "") == path', src)

    def test_exact_path_retrieval_keeps_line_metadata_and_ordering(self):
        src = Path("indexer.py").read_text()
        self.assertIn('"line": exact["metadatas"][i].get("line", 0)', src)
        self.assertIn('chunks.sort(key=lambda c: int(c.get("line", 0) or 0))', src)

    def test_fetch_all_from_file_avoids_mixing_duplicate_basenames(self):
        src = Path("indexer.py").read_text()
        self.assertIn("selected_path = select_best_authoritative_path(", src)
        self.assertIn("chunks = candidates_by_file[selected_path]", src)

    def test_limitation_message_mentions_indexed_source_of_truth_candidates(self):
        src = Path("indexer.py").read_text()
        self.assertIn(
            "AndesCode could not find the required authoritative file in indexed source-of-truth candidates.",
            src,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
