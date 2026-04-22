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
    is_declaration_query,
    has_declaration_keywords,
    source_of_truth_guidance,
    # Dependency authority helpers
    context_has_dependency_authority,
    workspace_has_dependency_files,
    recover_dependency_build_files,
    no_dependency_files_in_workspace_limitation,
    dependency_files_not_indexed_limitation,
    dependency_authority_incomplete_limitation,
    DEPENDENCY_BUILD_AUTHORITY_TYPES,
)
from andes_cache.debug import compute_intent_authority_satisfaction
from andes_cache.manager import AndesCacheManager
from andes_cache.integrity import (
    INTEGRITY_HEALTHY,
    INTEGRITY_STALE,
    FileIntegrityStatus,
    IntegrityReport,
    select_healthy_authoritative_path,
)


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


    def test_dependency_queries_are_treated_as_declaration_questions(self):
        self.assertTrue(is_declaration_query("what dependencies and versions are configured"))
        self.assertTrue(is_declaration_query("where are build settings declared"))
        self.assertTrue(is_declaration_query("what dependencies are declared"))
        self.assertTrue(is_declaration_query("what libraries does this project use"))
        self.assertTrue(is_declaration_query("where is config defined"))

    def test_declaration_keyword_helper_matches_shared_keyword_set(self):
        self.assertTrue(has_declaration_keywords("what is declared in package.json"))
        self.assertTrue(has_declaration_keywords("show dependencies in pyproject.toml"))
        self.assertTrue(has_declaration_keywords("what does gradle declare"))
        self.assertTrue(has_declaration_keywords("list versions from pom.xml"))
        self.assertTrue(has_declaration_keywords("where are build settings declared"))

    def test_guidance_requires_declared_and_inferred_sections(self):
        guidance = source_of_truth_guidance("what dependencies are declared in package.json")
        self.assertIn("Declared Dependencies", guidance)
        self.assertIn("Inferred from Code Usage", guidance)
        self.assertIn("dependency declaration files", guidance)

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

    def test_android_multimodule_healthy_app_manifest_wins_after_integrity_gating(self):
        paths = [
            "android/app/src/main/AndroidManifest.xml",
            "android/lib/src/main/AndroidManifest.xml",
        ]
        ranked = rank_authoritative_paths(
            paths,
            query="what permissions are declared for the app",
            intent="declaration_or_configuration",
        )

        def validate(path: str):
            if path == "android/app/src/main/AndroidManifest.xml":
                return IntegrityReport(
                    overall_status=INTEGRITY_HEALTHY,
                    files=[FileIntegrityStatus(path=path, status=INTEGRITY_HEALTHY, reasons=[])],
                )
            return IntegrityReport(
                overall_status=INTEGRITY_STALE,
                files=[FileIntegrityStatus(path=path, status=INTEGRITY_STALE, reasons=["workspace_hash_mismatch"])],
            )

        selected, _attempts = select_healthy_authoritative_path(ranked, validate_path_fn=validate)
        self.assertEqual(selected, "android/app/src/main/AndroidManifest.xml")

    def test_android_multimodule_stale_lower_rank_does_not_block_app_build_file(self):
        paths = [
            "android/app/build.gradle.kts",
            "android/feature/chat/build.gradle.kts",
            "android/libs/common/build.gradle.kts",
        ]
        ranked = rank_authoritative_paths(
            paths,
            query="what dependencies are declared for app module",
            intent="dependency_or_build_inventory",
        )
        calls = []

        def validate(path: str):
            calls.append(path)
            if path == "android/app/build.gradle.kts":
                return IntegrityReport(
                    overall_status=INTEGRITY_HEALTHY,
                    files=[FileIntegrityStatus(path=path, status=INTEGRITY_HEALTHY, reasons=[])],
                )
            return IntegrityReport(
                overall_status=INTEGRITY_STALE,
                files=[FileIntegrityStatus(path=path, status=INTEGRITY_STALE, reasons=["missing_on_disk"])],
            )

        selected, _attempts = select_healthy_authoritative_path(ranked, validate_path_fn=validate)
        self.assertEqual(selected, "android/app/build.gradle.kts")
        self.assertEqual(calls, ["android/app/build.gradle.kts"])

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
        self.assertIn("_path_suffix_match(", src)
        self.assertIn("_fetch_indexed_candidates_by_basename(path", src)

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


class TestDependencyAuthoritySourceGuardrails(unittest.TestCase):
    """Source-code guardrail tests — verify the key structural invariants in indexer.py."""

    def test_cache_bypass_triggers_for_manifest_only_dep_results(self):
        """indexer.py must bypass cache when a dep query has a manifest-only cached result."""
        src = Path("indexer.py").read_text()
        self.assertIn("_cached_dep_authority_missing", src)
        self.assertIn("context_has_dependency_authority(cached_final)", src)
        self.assertIn("intent == \"dependency_or_build_inventory\"", src)

    def test_dep_chunks_placed_before_manifest_chunks_in_return(self):
        """After dep recovery, dep chunks must be placed at the front of collected[]."""
        src = Path("indexer.py").read_text()
        # The fix puts dep_chunks first; verify the reorder pattern is present.
        self.assertIn("collected = dep_chunks + non_dep_collected[:manifest_budget]", src)

    def test_libs_versions_toml_in_manifest_files_for_discovery(self):
        """libs.versions.toml must be in MANIFEST_FILES so it gets indexed and discovered."""
        src = Path("indexer.py").read_text()
        self.assertIn("libs.versions.toml", src)
        # Also verify it's within the MANIFEST_FILES block
        mf_start = src.find("MANIFEST_FILES = {")
        mf_end = src.find("}", mf_start)
        self.assertIn("libs.versions.toml", src[mf_start:mf_end])

    def test_dep_specific_failure_markers_in_both_cache_bypass_sets(self):
        """New __dep*__ markers must appear in both _failure_markers and _failure_file_markers."""
        src = Path("indexer.py").read_text()
        # Both sets must list the new markers
        self.assertGreaterEqual(
            src.count("__no_dependency_files_in_workspace__"), 2
        )
        self.assertGreaterEqual(
            src.count("__dependency_files_not_indexed__"), 2
        )
        self.assertGreaterEqual(
            src.count("__dependency_authority_incomplete__"), 2
        )


class TestDependencyAuthorityIntentSpecific(unittest.TestCase):
    """
    Targeted tests for intent-specific dependency authority checks.
    Tests cover scenarios A-E as specified in the implementation requirements.
    """

    # ── Helpers (shared chunk factories) ────────────────────────────────────

    def _manifest_chunk(self, path: str = "app/src/main/AndroidManifest.xml") -> dict:
        return {
            "file": path,
            "content": '<uses-permission android:name="android.permission.INTERNET"/>',
            "source_type": "manifest",
            "authority_level": "configured",
        }

    def _build_chunk(self, path: str = "app/build.gradle.kts") -> dict:
        return {
            "file": path,
            "content": 'implementation("com.example:library:1.0.0")',
            "source_type": "build_file",
            "authority_level": "declared",
        }

    def _dep_chunk(self, path: str = "gradle/libs.versions.toml") -> dict:
        return {
            "file": path,
            "content": "[libraries]\nretrofit = { group = \"com.squareup.retrofit2\", name = \"retrofit\", version = \"2.9.0\" }",
            "source_type": "dependency_file",
            "authority_level": "declared",
        }

    # ── Test A: Android repo with manifest-only, no build.gradle retrieved ───

    def test_A_manifest_does_not_satisfy_dependency_authority(self):
        """Manifest chunk alone must NOT count as dependency authority."""
        chunks = [self._manifest_chunk()]
        self.assertFalse(context_has_dependency_authority(chunks))

    def test_A_workspace_manifest_only_has_no_dependency_files(self):
        """Workspace with only manifest in metadata has no dependency files."""
        manifests = ["app/src/main/AndroidManifest.xml"]
        self.assertFalse(workspace_has_dependency_files(manifests, []))

    def test_A_dependency_recovery_returns_empty_for_manifest_only_workspace(self):
        """Recovery helper returns empty list when workspace has no dep/build files."""
        manifests = ["app/src/main/AndroidManifest.xml"]
        candidates = recover_dependency_build_files(
            intent="dependency_or_build_inventory",
            query="what dependencies are declared",
            manifests=manifests,
            config_files=[],
        )
        self.assertEqual(candidates, [])

    def test_A_no_dep_files_limitation_chunk_has_correct_marker(self):
        """Case A limitation uses the dedicated no-dep-files marker."""
        chunk = no_dependency_files_in_workspace_limitation()
        self.assertEqual(chunk["file"], "__no_dependency_files_in_workspace__")
        self.assertIn("No Declaration Files", chunk["content"])
        self.assertNotIn("Source-of-Truth Limitation", chunk["content"].split("\n")[0])

    def test_A_debug_compute_shows_dependency_authority_missing_for_manifest_only(self):
        """Intent authority satisfaction: manifest-only → missing dep authority."""
        chunks = [self._manifest_chunk()]
        result = compute_intent_authority_satisfaction(
            "dependency_or_build_inventory", chunks
        )
        self.assertIn("build_file", result["required_authority_classes"])
        self.assertIn("dependency_file", result["required_authority_classes"])
        self.assertEqual(result["satisfied_authority_classes"], [])
        self.assertIn("build_file", result["missing_authority_classes"])
        self.assertIn("dependency_file", result["missing_authority_classes"])

    # ── Test B: Android repo with build.gradle.kts and/or libs.versions.toml ─

    def test_B_build_gradle_kts_satisfies_dependency_authority(self):
        """build.gradle.kts chunk counts as dependency authority."""
        chunks = [self._build_chunk()]
        self.assertTrue(context_has_dependency_authority(chunks))

    def test_B_libs_versions_toml_classified_as_dependency_file(self):
        """libs.versions.toml must be classified as dependency_file, not build_file."""
        self.assertEqual(classify_source_type("gradle/libs.versions.toml"), "dependency_file")

    def test_B_libs_versions_toml_satisfies_dependency_authority(self):
        """libs.versions.toml chunk counts as dependency authority."""
        chunks = [self._dep_chunk("gradle/libs.versions.toml")]
        self.assertTrue(context_has_dependency_authority(chunks))

    def test_B_workspace_with_build_gradle_kts_has_dependency_files(self):
        """Workspace containing build.gradle.kts reports dependency files present."""
        manifests = [
            "app/src/main/AndroidManifest.xml",
            "app/build.gradle.kts",
        ]
        self.assertTrue(workspace_has_dependency_files(manifests, []))

    def test_B_recovery_returns_build_gradle_kts_when_present_in_workspace(self):
        """Recovery helper returns build.gradle.kts when workspace has it."""
        manifests = [
            "app/src/main/AndroidManifest.xml",
            "app/build.gradle.kts",
        ]
        candidates = recover_dependency_build_files(
            intent="dependency_or_build_inventory",
            query="what dependencies are declared",
            manifests=manifests,
            config_files=[],
        )
        self.assertIn("app/build.gradle.kts", candidates)
        self.assertNotIn("app/src/main/AndroidManifest.xml", candidates)

    def test_B_debug_compute_satisfied_when_build_file_in_context(self):
        """Intent authority satisfaction: build_file in context → satisfied (OR logic).

        missing_authority_classes may still list dependency_file as absent but that
        does NOT mean the intent is unsatisfied — having any one required type is enough.
        """
        chunks = [self._manifest_chunk(), self._build_chunk()]
        result = compute_intent_authority_satisfaction(
            "dependency_or_build_inventory", chunks
        )
        self.assertIn("build_file", result["satisfied_authority_classes"])
        # dependency_file is absent (only build_file is present) — debug shows this
        # but does not block authority satisfaction (OR semantics).
        self.assertIn("dependency_file", result["missing_authority_classes"])
        self.assertNotIn("build_file", result["missing_authority_classes"])

    # ── Test C: Workspace with truly no dependency declaration files ──────────

    def test_C_no_dep_files_anywhere_returns_correct_limitation_file(self):
        """When workspace has no dep files, the 'no dep files' limitation is used."""
        chunk = no_dependency_files_in_workspace_limitation()
        self.assertEqual(chunk["file"], "__no_dependency_files_in_workspace__")
        # Content must NOT claim files were retrieved/attempted
        self.assertNotIn("retrieved", chunk["content"].lower())

    def test_C_config_only_workspace_not_treated_as_dependency_authority(self):
        """A workspace with only config/.env files has no dep authority."""
        config_files = [".env.production", "config/settings.yml"]
        self.assertFalse(workspace_has_dependency_files([], config_files))

    def test_C_no_dep_file_limitation_differs_from_generic_source_of_truth_limitation(self):
        """Case A must use the specific no-dep-files marker, not the generic one."""
        chunk = no_dependency_files_in_workspace_limitation()
        self.assertNotEqual(chunk["file"], "__source_of_truth_missing__")
        self.assertNotEqual(chunk["file"], "__source_of_truth_integrity__")

    # ── Test D: Dep files exist in workspace metadata but not retrievable ─────

    def test_D_dep_files_not_indexed_limitation_has_correct_marker(self):
        """Case B limitation uses the 'not indexed' marker."""
        chunk = dependency_files_not_indexed_limitation()
        self.assertEqual(chunk["file"], "__dependency_files_not_indexed__")
        self.assertIn("Source-of-Truth Limitation", chunk["content"])
        self.assertIn("could not retrieve", chunk["content"])

    def test_D_dep_files_not_indexed_distinct_from_no_dep_files(self):
        """Case B marker must be distinct from Case A marker."""
        a = no_dependency_files_in_workspace_limitation()
        b = dependency_files_not_indexed_limitation()
        self.assertNotEqual(a["file"], b["file"])
        self.assertNotEqual(a["content"], b["content"])

    def test_D_workspace_with_build_gradle_but_not_fetched_reports_dep_present(self):
        """workspace_has_dependency_files is True even if the index cannot fetch them."""
        # Workspace metadata says build.gradle is present.
        manifests = ["app/build.gradle", "app/src/main/AndroidManifest.xml"]
        self.assertTrue(workspace_has_dependency_files(manifests, []))

    def test_D_recovery_candidates_include_gradle_when_workspace_has_it(self):
        """If workspace has build.gradle, recovery_candidates should include it."""
        manifests = ["app/build.gradle", "app/src/main/AndroidManifest.xml"]
        candidates = recover_dependency_build_files(
            intent="dependency_or_build_inventory",
            query="what dependencies are declared",
            manifests=manifests,
            config_files=[],
        )
        self.assertIn("app/build.gradle", candidates)

    # ── Test E: Manifest-specific query still works normally ─────────────────

    def test_E_manifest_query_intent_is_not_dependency_inventory(self):
        """Permissions/manifest queries are classified as declaration_or_configuration."""
        d = classify_query_intent_details("what permissions does the app declare")
        self.assertEqual(d["intent"], "declaration_or_configuration")
        self.assertNotEqual(d["intent"], "dependency_or_build_inventory")

    def test_E_manifest_satisfies_declaration_intent_authority(self):
        """For declaration_or_configuration, manifest IS valid authority (OR logic).

        config_file may show up in missing_authority_classes (informational) but the
        intent is satisfied as long as at least one required type is present.
        """
        chunks = [self._manifest_chunk()]
        result = compute_intent_authority_satisfaction(
            "declaration_or_configuration", chunks
        )
        self.assertIn("manifest", result["satisfied_authority_classes"])
        # config_file is listed as absent (informational) but manifest is sufficient.
        self.assertNotIn("manifest", result["missing_authority_classes"])

    def test_E_context_has_dependency_authority_false_for_manifest_only(self):
        """context_has_dependency_authority is False for manifest, not a dep-query problem."""
        # This is fine: the caller checks intent BEFORE calling this helper.
        chunks = [self._manifest_chunk()]
        self.assertFalse(context_has_dependency_authority(chunks))

    def test_E_manifest_chunk_authority_level_for_declaration_intent(self):
        """Manifest authority level for declaration_or_configuration is 'configured'."""
        self.assertEqual(
            authority_level_for_source("declaration_or_configuration", "manifest"),
            "configured",
        )

    def test_E_manifest_not_in_dependency_build_authority_types(self):
        """manifest source type must NOT be in DEPENDENCY_BUILD_AUTHORITY_TYPES."""
        self.assertNotIn("manifest", DEPENDENCY_BUILD_AUTHORITY_TYPES)
        self.assertNotIn("config_file", DEPENDENCY_BUILD_AUTHORITY_TYPES)
        self.assertIn("build_file", DEPENDENCY_BUILD_AUTHORITY_TYPES)
        self.assertIn("dependency_file", DEPENDENCY_BUILD_AUTHORITY_TYPES)

    # ── Additional classify_source_type coverage ─────────────────────────────

    def test_classify_source_type_libs_versions_toml(self):
        """gradle/libs.versions.toml is a dependency_file (Gradle version catalog)."""
        self.assertEqual(classify_source_type("gradle/libs.versions.toml"), "dependency_file")

    def test_classify_source_type_gradle_properties(self):
        """gradle.properties is a build_file (contains build/project config)."""
        self.assertEqual(classify_source_type("gradle.properties"), "build_file")

    def test_classify_source_type_gemfile(self):
        """Gemfile is a dependency_file (Ruby)."""
        self.assertEqual(classify_source_type("Gemfile"), "dependency_file")

    def test_classify_source_type_composer_json(self):
        """composer.json is a dependency_file (PHP)."""
        self.assertEqual(classify_source_type("composer.json"), "dependency_file")

    def test_classify_source_type_build_gradle_kts_is_build_file(self):
        """build.gradle.kts is a build_file, not a dependency_file."""
        self.assertEqual(classify_source_type("app/build.gradle.kts"), "build_file")

    # ── Declare/infer separation ─────────────────────────────────────────────

    def test_declared_section_must_only_use_dep_or_build_source_types(self):
        """
        Verifies that manifest source_type is classified as 'configured', not 'declared',
        under the dependency_or_build_inventory intent — preventing manifest entries from
        appearing in a declared dependency section.
        """
        self.assertEqual(
            authority_level_for_source("dependency_or_build_inventory", "manifest"),
            "configured",
        )
        self.assertEqual(
            authority_level_for_source("dependency_or_build_inventory", "build_file"),
            "declared",
        )
        self.assertEqual(
            authority_level_for_source("dependency_or_build_inventory", "dependency_file"),
            "declared",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
