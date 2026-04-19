import os
import unittest

from andes_cache.debug import (
    env_debug_mode,
    resolve_debug_mode,
    initialize_payload,
    finalize_payload,
    apply_failure_signals,
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
            retrieval_route="semantic",
            top_score=None,
            final_chunks=[],
        )
        self.assertTrue(payload["failure_signals"]["expected_but_missing_authority"])
        self.assertTrue(payload["failure_signals"]["wrong_retrieval_route"])
        self.assertTrue(payload["failure_signals"]["empty_retrieval"])
        self.assertTrue(payload["failure_signals"]["multi_module_ambiguity"])


if __name__ == "__main__":
    unittest.main()
