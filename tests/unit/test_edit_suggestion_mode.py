import asyncio
import json
import types
import unittest

from andes_cache.routing import (
    DEPENDENCY_OR_BUILD_INVENTORY,
    EDIT_SUGGESTION as ROUTING_EDIT_SUGGESTION,
    GENERIC_SEMANTIC,
    SYMBOL_LOOKUP,
    classify_query_intent,
    retrieval_route_for_intent,
)
from edit_suggestion import (
    EDIT_SUGGESTION,
    build_edit_suggestion_context,
    edit_suggestion_policy,
    enforce_edit_suggestion_output,
    is_edit_suggestion_query,
)
from tests.unit.test_server_stream_debug_mode import _import_server_with_stubs


def _stream_text(events):
    parts = []
    for raw in events:
        if not raw.startswith("data: ") or raw.strip() == "data: [DONE]":
            continue
        try:
            payload = json.loads(raw[len("data: "):].strip())
        except Exception:
            continue
        parts.append(payload.get("choices", [{}])[0].get("delta", {}).get("content", ""))
    return "".join(parts)


async def _collect_stream(server, query="fix this bug"):
    messages = [{"role": "user", "content": query}]
    events = []
    async for event in server._stream(messages, 128, "req-edit-regression", 0.0, debug_mode=False):
        events.append(event)
    return events


class _ValidEditAnswerLlm:
    def __call__(self, _prompt, max_tokens=0, stream=False, echo=False):
        answer = (
            "Finding: `src/cache.py` currently routes refresh through `CacheManager.refresh_cache`.\n\n"
            "Evidence: `src/cache.py`, `CacheManager`, `refresh_cache`.\n\n"
            "Recommended change: Add one guard in `refresh_cache`.\n\n"
            "Patch plan: In `src/cache.py`, update `CacheManager.refresh_cache`.\n\n"
            "Validation: pytest\n\n"
            "Confidence: high"
        )
        if stream:
            return iter([{"choices": [{"text": answer}]}])
        return {"choices": [{"text": "src/cache.py"}]}


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

    def test_broad_analysis_questions_do_not_trigger_edit_suggestion_mode(self):
        cases = {
            "explain the performance path": GENERIC_SEMANTIC,
            "where is AddToCart defined?": SYMBOL_LOOKUP,
            "what dependencies does this add?": DEPENDENCY_OR_BUILD_INVENTORY,
            "how does updateSchedule work?": GENERIC_SEMANTIC,
            "where is this configured?": "declaration_or_configuration",
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                self.assertFalse(is_edit_suggestion_query(query))
                self.assertNotEqual(classify_query_intent(query), ROUTING_EDIT_SUGGESTION)
                self.assertEqual(classify_query_intent(query), expected)

    def test_streaming_planned_edit_context_survives_with_debug_mode_off(self):
        server = _import_server_with_stubs()
        server.INDEXER_READY = True
        server.llm = _ValidEditAnswerLlm()
        server._plan_files = lambda _query, _pmap: ["src/cache.py"]
        server.search_codebase = lambda *_args, **_kwargs: []
        server._indexer_module = types.SimpleNamespace(
            _load_project_map=lambda: {"project": "demo", "file_symbols": {"src/cache.py": ["CacheManager", "refresh_cache"]}},
            _load_workspace_index=lambda: {
                "import_graph": {"samples": {}},
                "file_to_module_map": {"src/cache.py": "src", "tests/test_cache.py": "tests"},
                "config_graph": {"config_files": ["pytest.ini"]},
                "manifests": [],
            },
            get_repo_fingerprint=lambda: "fp-edit",
            get_chunks_for_file=lambda fname: {
                "src/cache.py": [
                    {
                        "file": "src/cache.py",
                        "content": "class CacheManager:\n    def refresh_cache(self):\n        return None\n",
                        "symbols": "CacheManager refresh_cache",
                        "line": 1,
                    }
                ],
                "tests/test_cache.py": [{"file": "tests/test_cache.py", "content": "def test_refresh_cache(): assert True", "symbols": "test_refresh_cache", "line": 1}],
                "pytest.ini": [{"file": "pytest.ini", "content": "[pytest]", "symbols": "", "line": 1}],
            }.get(fname, []),
            CACHE=None,
        )

        captured = {}
        original_enforce = server.enforce_edit_suggestion_output

        def _capture(answer, ctx):
            captured["files"] = ctx.files
            captured["symbols"] = ctx.symbols
            return original_enforce(answer, ctx)

        server.enforce_edit_suggestion_output = _capture
        try:
            events = asyncio.run(_collect_stream(server, query="fix this bug"))
        finally:
            server.enforce_edit_suggestion_output = original_enforce

        text = _stream_text(events)
        self.assertNotIn("I do not have enough repo-grounded context", text)
        self.assertIn("Finding:", text)
        self.assertIn("src/cache.py", text)
        self.assertIn("CacheManager", text)
        self.assertIn("src/cache.py", captured.get("files", ()))
        self.assertIn("CacheManager", captured.get("symbols", ()))

    def test_streaming_safe_fallback_only_when_context_is_truly_missing(self):
        server = _import_server_with_stubs()
        server.INDEXER_READY = True
        server.llm = _ValidEditAnswerLlm()
        server._plan_files = lambda _query, _pmap: []
        server.search_codebase = lambda *_args, **_kwargs: []
        server._indexer_module = types.SimpleNamespace(
            _load_project_map=lambda: {"project": "demo", "file_symbols": {}},
            _load_workspace_index=lambda: {},
            get_repo_fingerprint=lambda: "fp-missing",
            get_chunks_for_file=lambda _fname: [],
            CACHE=None,
        )

        events = asyncio.run(_collect_stream(server, query="fix this bug"))
        text = _stream_text(events)
        self.assertIn("I do not have enough repo-grounded context to propose a safe edit.", text)
        self.assertIn("relevant files", text)


if __name__ == "__main__":
    unittest.main()
