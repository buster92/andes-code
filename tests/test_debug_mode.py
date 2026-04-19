import os
import unittest

from andes_cache.debug import (
    env_debug_mode,
    resolve_debug_mode,
    initialize_payload,
    finalize_payload,
    apply_failure_signals,
    populate_retrieval_snapshot,
    format_debug_sse_event,
    infer_expected_authority,
)


class TestDebugMode(unittest.TestCase):
    def test_debug_mode_off_by_default(self):
        prev = os.environ.pop("ANDESCODE_DEBUG_MODE", None)
        try:
            self.assertFalse(env_debug_mode())
            self.assertFalse(resolve_debug_mode())
        finally:
            if prev is not None:
                os.environ["ANDESCODE_DEBUG_MODE"] = prev

    def test_debug_mode_structured_payload(self):
        decision = {"intent": "declaration_or_configuration", "retrieval_route": "source_of_truth", "ambiguous": False}
        workspace = {"repo_types": ["android"], "modules": [{"name": "app"}], "manifests": ["app/src/main/AndroidManifest.xml"]}
        payload = initialize_payload("what permissions are declared", decision, workspace)
        payload = finalize_payload(payload, [{"file": "app/src/main/AndroidManifest.xml", "content": "x", "source_type": "manifest"}])
        self.assertIn("workspace_summary", payload)
        self.assertIn("retrieval", payload)
        self.assertIn("failure_signals", payload)
        self.assertTrue(payload["final_context"]["authoritative_files_present"])

    def test_failure_flags_missing_manifest_and_wrong_route_and_empty(self):
        decision = {"intent": "declaration_or_configuration", "retrieval_route": "semantic", "ambiguous": True}
        workspace = {"repo_types": ["android"], "modules": [{"name": "app"}, {"name": "feature"}], "manifests": []}
        payload = initialize_payload("what permissions are declared", decision, workspace)
        payload["source_of_truth"]["missing_expected"] = ["AndroidManifest.xml"]
        payload["failure_signals"]["expected_but_missing_authority"] = True
        payload = apply_failure_signals(
            payload,
            query="what permissions are declared",
            intent="declaration_or_configuration",
            retrieval_route="semantic",
            top_score=None,
            final_chunks=[],
        )
        self.assertFalse(payload["failure_signals"]["expected_but_missing_authority"])
        self.assertTrue(payload["failure_signals"]["wrong_retrieval_route"])
        self.assertTrue(payload["failure_signals"]["empty_retrieval"])
        self.assertTrue(payload["failure_signals"]["multi_module_ambiguity"])

    def test_request_isolation_no_cross_payload_leakage(self):
        decision = {"intent": "declaration_or_configuration", "retrieval_route": "source_of_truth", "ambiguous": False}
        p1 = initialize_payload("what permissions are declared", decision, {"modules": [{"name": "app"}], "manifests": []})
        p2 = initialize_payload("how does payments module work", {"intent": "architecture_overview", "retrieval_route": "semantic", "ambiguous": False}, {"modules": [{"name": "payments"}], "manifests": []})
        p1["source_of_truth"]["missing_expected"].append("manifest")
        self.assertEqual(p2["source_of_truth"]["missing_expected"], [])

    def test_cache_hit_debug_consistency(self):
        decision = {"intent": "runtime_usage_or_reference", "retrieval_route": "runtime_usage", "ambiguous": False}
        payload = initialize_payload("where is auth used", decision, {"modules": [{"name": "auth"}], "manifests": []})
        chunks = [{"file": "auth/service.py", "content": "x", "coverage": {"partial": False}, "source_type": "source_code"}]
        payload = populate_retrieval_snapshot(payload, chunks=chunks, raw_candidates=["auth/service.py"], cache_hit=True)
        payload = finalize_payload(payload, chunks)
        self.assertTrue(payload["retrieval"]["cache_hit"])
        self.assertEqual(payload["retrieval"]["selected_candidates"], ["auth/service.py"])
        self.assertEqual(payload["final_context"]["files_used"], ["auth/service.py"])

    def test_streaming_debug_transport_is_separate(self):
        event = format_debug_sse_event({"query": "q"})
        self.assertIn("event: debug", event)
        self.assertIn("\"object\": \"debug.payload\"", event)
        self.assertNotIn("[DEBUG_PAYLOAD]", event)

    def test_route_aware_low_confidence_source_of_truth(self):
        decision = {"intent": "declaration_or_configuration", "retrieval_route": "source_of_truth", "ambiguous": False}
        payload = initialize_payload("where is auth configured", decision, {"modules": [{"name": "api"}], "manifests": ["service/config.yml"]})
        good_chunks = [{"file": "service/config.yml", "content": "x", "source_type": "config_file"}]
        checked = apply_failure_signals(
            payload,
            query="where is auth configured",
            intent="declaration_or_configuration",
            retrieval_route="source_of_truth",
            top_score=None,
            final_chunks=good_chunks,
        )
        self.assertFalse(checked["failure_signals"]["low_confidence_retrieval"])

    def test_non_android_source_of_truth_expectations(self):
        decision = {"intent": "dependency_or_build_inventory", "retrieval_route": "source_of_truth", "ambiguous": False}
        payload = initialize_payload("what dependencies are declared", decision, {"modules": [{"name": "worker"}], "manifests": ["pyproject.toml"]})
        checked = apply_failure_signals(
            payload,
            query="what dependencies are declared",
            intent="dependency_or_build_inventory",
            retrieval_route="source_of_truth",
            top_score=None,
            final_chunks=[],
        )
        self.assertIn("dependency_file", checked["source_of_truth"]["missing_expected"])
        self.assertTrue(checked["failure_signals"]["expected_but_missing_authority"])

    def test_expected_authority_is_query_specific(self):
        dep = infer_expected_authority("dependency_or_build_inventory", "what dependencies are declared")
        self.assertEqual(sorted(dep["expected_classes"]), ["build_file", "dependency_file"])

        cfg = infer_expected_authority("declaration_or_configuration", "where is auth configured")
        self.assertEqual(sorted(cfg["expected_classes"]), ["config_file"])

        perms = infer_expected_authority("declaration_or_configuration", "what permissions are declared")
        self.assertEqual(sorted(perms["expected_classes"]), ["manifest"])
        self.assertIn("AndroidManifest.xml", perms["expected_files"])


if __name__ == "__main__":
    unittest.main()
