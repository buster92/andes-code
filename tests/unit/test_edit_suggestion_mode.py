import unittest

from andes_cache.routing import classify_query_intent, retrieval_route_for_intent
from edit_suggestion import (
    EDIT_SUGGESTION,
    build_edit_suggestion_context,
    edit_suggestion_policy,
    enforce_edit_suggestion_output,
    is_edit_suggestion_query,
)
from tests.unit.test_server_stream_debug_mode import _import_server_with_stubs


class TestEditSuggestionMode(unittest.TestCase):
    def test_edit_suggestion_mode_triggers_for_change_fix_and_performance_requests(self):
        queries = [
            "improve this",
            "fix this bug",
            "make this faster",
            "suggest one update",
            "change this behavior",
            "why is this failing?",
            "what code should I edit?",
        ]
        for query in queries:
            with self.subTest(query=query):
                self.assertTrue(is_edit_suggestion_query(query))
                self.assertEqual(classify_query_intent(query), EDIT_SUGGESTION)
                self.assertEqual(retrieval_route_for_intent(EDIT_SUGGESTION), "edit_suggestion")

    def test_output_contract_requires_file_paths_and_symbols(self):
        chunks = [
            {
                "file": "src/cache.py",
                "content": "class CacheManager:\n    def refresh_cache(self):\n        return None\n",
                "symbols": "CacheManager refresh_cache",
            }
        ]
        ctx = build_edit_suggestion_context(chunks, query="fix cache refresh")
        grounded = (
            "Finding: `src/cache.py` currently routes refresh through `CacheManager.refresh_cache`.\n\n"
            "Evidence: `src/cache.py`, `CacheManager`, `refresh_cache`.\n\n"
            "Recommended change: Update one guard in `refresh_cache`.\n\n"
            "Patch plan: In `src/cache.py`, change `CacheManager.refresh_cache`.\n\n"
            "Validation: pytest\n\n"
            "Confidence: high"
        )
        self.assertEqual(enforce_edit_suggestion_output(grounded, ctx), grounded)

    def test_generic_recommendations_are_rejected_when_context_is_missing(self):
        ctx = build_edit_suggestion_context([], query="fix this bug")
        answer = "You should add a repository layer and improve error handling."
        filtered = enforce_edit_suggestion_output(answer, ctx)
        self.assertIn("I do not have enough repo-grounded context to propose a safe edit.", filtered)
        self.assertIn("relevant files", filtered)

    def test_existing_mechanism_is_reported_instead_of_suggested_as_new(self):
        chunks = [
            {
                "file": "services/search_service.py",
                "content": "class SearchService:\n    def query(self):\n        cached = self.cache.get('q')\n        return cached\n",
                "symbols": "SearchService query cache",
            }
        ]
        ctx = build_edit_suggestion_context(chunks, query="suggest one update to add cache")
        self.assertIn("cache exists in services/search_service.py", ctx.existing_mechanisms)
        policy = edit_suggestion_policy()
        self.assertIn("If it exists, mention the existing file/symbol instead of proposing it as new", policy)

    def test_validation_commands_are_inferred_from_repo_structure(self):
        chunks = [
            {"file": "pytest.ini", "content": "[pytest]", "symbols": ""},
            {"file": "tests/test_cache.py", "content": "def test_cache(): assert True", "symbols": "test_cache"},
            {"file": "src/cache.py", "content": "def get_cache(): return {}", "symbols": "get_cache"},
        ]
        ctx = build_edit_suggestion_context(chunks, query="fix cache bug")
        self.assertIn("pytest", ctx.validation_commands)

    def test_pack_context_includes_edit_retrieval_checklist(self):
        server = _import_server_with_stubs()
        server.MODEL_CONTEXT_WINDOW = 4000
        chunks = [
            {
                "file": "src/cache.py",
                "content": "class CacheManager:\n    def refresh_cache(self):\n        cached = True\n        return cached\n",
                "symbols": "CacheManager refresh_cache",
            },
            {"file": "pytest.ini", "content": "[pytest]", "symbols": ""},
        ]
        context, _info = server._pack_context_section(
            query="fix this cache bug",
            map_section="",
            chunks=chunks,
            request_id="req-edit",
        )
        self.assertIn("Edit Suggestion Retrieval Checklist", context)
        self.assertIn("src/cache.py", context)
        self.assertIn("CacheManager", context)
        self.assertIn("pytest", context)


if __name__ == "__main__":
    unittest.main()
