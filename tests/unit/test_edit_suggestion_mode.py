import asyncio
import json
import tempfile
import types
import unittest
from pathlib import Path

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
    is_symbol_optional_file,
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


def _remote_request_body(*, stream=False, debug=False):
    return {
        "client": {
            "client_version": "test",
            "protocol_version": "andes.remote.v1",
            "platform": "test",
        },
        "workspace": {
            "workspace_id": "repo",
            "repo_name": "repo",
            "repo_root_name": "repo",
            "branch": "main",
            "commit_hash": "abc",
            "is_dirty": False,
        },
        "query": {
            "request_id": "req-remote-edit",
            "text": "fix this bug",
            "requested_at": "2026-05-14T00:00:00+00:00",
        },
        "retrieval": {
            "strategy": "test",
            "top_k": 1,
            "indexed_at": "2026-05-14T00:00:00+00:00",
            "index_state": "ready",
            "total_candidate_files": 1,
            "retrieved_chunk_count": 1,
            "retrieval_mode": "remote_inference",
        },
        "chunks": [
            {
                "chunk_id": "c1",
                "path": "src/cache.py",
                "language": "py",
                "start_line": 1,
                "end_line": 3,
                "score": 1.0,
                "source_type": "source_code",
                "authority": "referenced",
                "authority_reason": "test",
                "content": "class CacheManager:\n    def refresh_cache(self):\n        return None\n",
            }
        ],
        "options": {"stream": stream, "debug": debug, "max_answer_tokens": 128},
    }


class _JsonRequest:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


class _FakeRemoteStreamResponse:
    def __init__(self, events):
        self._chunks = [event.encode("utf-8") for event in events]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _size):
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


def _remote_chunk_event(content):
    return "data: " + json.dumps({
        "object": "chat.completion.chunk",
        "choices": [{"delta": {"content": content}}],
    }) + "\n\n"


def _remote_error_event(code="remote_failed", message="remote failed"):
    return "data: " + json.dumps({
        "object": "andescode.error",
        "error": {"code": code, "message": message, "phase": "remote_inference"},
    }) + "\n\n"


def _remote_debug_event():
    return "event: debug\n" + "data: " + json.dumps({
        "object": "debug.payload",
        "debug": {"query": "fix this bug", "retrieval": {"strategy": "test"}},
    }) + "\n\n"


class _GenericRemoteLlm:
    def __call__(self, _prompt, max_tokens=0, stream=False, echo=False):
        text = "You should consider improving architecture."
        if stream:
            return iter([{"choices": [{"text": text}]}])
        return {"choices": [{"text": text}]}


class _GroundedRemoteLlm:
    def __call__(self, _prompt, max_tokens=0, stream=False, echo=False):
        text = (
            "Finding: `src/cache.py` currently routes refresh through `CacheManager.refresh_cache`.\n\n"
            "Evidence: `src/cache.py`, `CacheManager`, `refresh_cache`.\n\n"
            "Recommended change: Add one guard in `refresh_cache`.\n\n"
            "Patch plan: In `src/cache.py`, update `CacheManager.refresh_cache`.\n\n"
            "Validation: pytest\n\n"
            "Confidence: high"
        )
        if stream:
            return iter([{"choices": [{"text": text}]}])
        return {"choices": [{"text": text}]}


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


def _contract_answer(file_path, evidence="file-only"):
    return (
        f"Finding: `{file_path}` currently contains the behavior to update.\n\n"
        f"Evidence: `{file_path}` and {evidence}.\n\n"
        f"Recommended change: Make one minimal change in `{file_path}`.\n\n"
        f"Patch plan: Update `{file_path}` only.\n\n"
        "Validation: pytest\n\n"
        "Confidence: high"
    )


class TestEditSuggestionMode(unittest.TestCase):
    def test_edit_suggestion_mode_triggers_for_change_fix_and_performance_requests(self):
        queries = [
            "implement retry handling",
            "please implement cache invalidation",
            "improve this",
            "fix this bug",
            "make this faster",
            "suggest one update",
            "suggest one change",
            "change this behavior",
            "refactor refresh_cache to avoid duplicate work",
            "optimize cache lookup",
            "why is this failing?",
            "what code should I edit?",
            "what files should I change?",
        ]
        for query in queries:
            with self.subTest(query=query):
                self.assertTrue(is_edit_suggestion_query(query))
                self.assertEqual(classify_query_intent(query), EDIT_SUGGESTION)
                self.assertEqual(retrieval_route_for_intent(EDIT_SUGGESTION), "edit_suggestion")

    def test_bare_edit_words_in_explanatory_queries_do_not_trigger_edit_mode(self):
        cases = {
            "how is retry implemented?": GENERIC_SEMANTIC,
            "where is the patch logic?": GENERIC_SEMANTIC,
            "explain the refactor": GENERIC_SEMANTIC,
            "what does this change do?": GENERIC_SEMANTIC,
            "where is this configured?": "declaration_or_configuration",
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                self.assertFalse(is_edit_suggestion_query(query))
                self.assertNotEqual(classify_query_intent(query), ROUTING_EDIT_SUGGESTION)
                self.assertEqual(classify_query_intent(query), expected)

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

    def test_config_build_docs_file_only_evidence_survives_without_symbols(self):
        cases = [
            ("package.json", "{}"),
            (".github/workflows/ci.yml", "name: ci"),
            ("Dockerfile", "FROM python:3.11"),
            ("README.md", "# Project"),
        ]
        for path, content in cases:
            with self.subTest(path=path):
                self.assertTrue(is_symbol_optional_file(path))
                ctx = build_edit_suggestion_context(
                    [{"file": path, "content": content, "symbols": ""}],
                    query="suggest one update",
                )
                self.assertEqual(ctx.missing_context, ())
                answer = _contract_answer(path)
                self.assertEqual(enforce_edit_suggestion_output(answer, ctx), answer)

    def test_source_code_without_symbols_still_falls_back(self):
        ctx = build_edit_suggestion_context(
            [{"file": "src/cache.py", "content": "CACHE_ENABLED = True", "symbols": ""}],
            query="fix this bug",
        )
        answer = _contract_answer("src/cache.py")
        filtered = enforce_edit_suggestion_output(answer, ctx)
        self.assertIn("I do not have enough repo-grounded context", filtered)
        self.assertIn("symbols or methods in the relevant source files", filtered)

    def test_source_code_with_file_and_symbol_still_passes(self):
        ctx = build_edit_suggestion_context(
            [
                {
                    "file": "src/cache.py",
                    "content": "class CacheManager:\n    pass\n",
                    "symbols": "CacheManager",
                }
            ],
            query="fix this bug",
        )
        answer = _contract_answer("src/cache.py", evidence="`CacheManager`")
        self.assertEqual(enforce_edit_suggestion_output(answer, ctx), answer)

    def test_safe_fallback_still_appears_when_no_relevant_files_are_retrieved(self):
        ctx = build_edit_suggestion_context([], query="fix this bug")
        filtered = enforce_edit_suggestion_output(_contract_answer("package.json"), ctx)
        self.assertIn("I do not have enough repo-grounded context", filtered)
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


class TestEditSuggestionChunkMerging(unittest.TestCase):
    def _server_with_chunks(self, chunks):
        server = _import_server_with_stubs()
        server.INDEXER_READY = True
        server._indexer_module = types.SimpleNamespace(
            get_chunks_for_file=lambda fname: chunks if fname == "src/cache.py" else [],
        )
        return server

    def test_direct_file_contents_are_preferred_when_index_root_is_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "src").mkdir()
            (root / "src" / "cache.py").write_text("real_line1\nreal_line2\n", encoding="utf-8")
            server = _import_server_with_stubs()
            server.INDEXER_READY = True
            server._indexer_module = types.SimpleNamespace(
                _load_hashes=lambda: {"__root__": str(root)},
                get_chunks_for_file=lambda fname: [
                    {"file": fname, "line": 1, "end_line": 2, "content": "stale_chunk1\nstale_chunk2", "symbols": "CacheManager"}
                ],
            )

            merged = server._merge_indexed_file_chunks("src/cache.py")

            self.assertEqual(merged["content"], "real_line1\nreal_line2\n")
            self.assertTrue(merged["full_file"])
            self.assertEqual(merged["coverage"]["source"], "disk_file")

    def test_non_overlapping_chunks_merge_normally(self):
        server = self._server_with_chunks(
            [
                {"file": "src/cache.py", "line": 1, "end_line": 2, "content": "line1\nline2", "symbols": "CacheManager"},
                {"file": "src/cache.py", "line": 3, "end_line": 4, "content": "line3\nline4", "symbols": "refresh_cache"},
            ]
        )

        merged = server._merge_indexed_file_chunks("src/cache.py")

        self.assertEqual(merged["content"], "line1\nline2\nline3\nline4")
        self.assertTrue(merged["full_file"])
        self.assertFalse(merged["coverage"]["partial"])

    def test_overlapping_chunks_do_not_duplicate_overlap_lines(self):
        server = self._server_with_chunks(
            [
                {"file": "src/cache.py", "line": 1, "end_line": 3, "content": "line1\nline2\nline3", "symbols": "CacheManager"},
                {"file": "src/cache.py", "line": 3, "end_line": 5, "content": "line3\nline4\nline5", "symbols": "refresh_cache"},
            ]
        )

        merged = server._merge_indexed_file_chunks("src/cache.py")

        self.assertEqual(merged["content"], "line1\nline2\nline3\nline4\nline5")
        self.assertEqual(merged["content"].count("line3"), 1)
        self.assertTrue(merged["full_file"])

    def test_zero_based_overlapping_chunks_with_headers_deoverlap_cleanly(self):
        first_lines = [f"line{i}" for i in range(1, 81)]
        second_lines = [f"line{i}" for i in range(66, 101)]
        server = self._server_with_chunks(
            [
                {
                    "file": "src/cache.py",
                    "line": 0,
                    "content": "# File: src/cache.py\n\n" + "\n".join(first_lines),
                    "symbols": "CacheManager",
                },
                {
                    "file": "src/cache.py",
                    "line": 65,
                    "content": "# File: src/cache.py\n\n" + "\n".join(second_lines),
                    "symbols": "refresh_cache",
                },
            ]
        )

        merged = server._merge_indexed_file_chunks("src/cache.py")

        self.assertTrue(merged["full_file"])
        self.assertFalse(merged["coverage"]["partial"])
        self.assertEqual(merged["coverage"]["source"], "indexed_chunks_deoverlapped")
        self.assertEqual(merged["coverage"]["removed_repeated_file_headers"], 2)
        self.assertNotIn("# File:", merged["content"])
        self.assertEqual(merged["content"].count("line66"), 1)
        self.assertEqual(merged["content"].splitlines(), [f"line{i}" for i in range(1, 101)])

    def test_repeated_file_headers_are_not_duplicated(self):
        server = self._server_with_chunks(
            [
                {"file": "src/cache.py", "line": 1, "end_line": 2, "content": "# File: src/cache.py\nline1", "symbols": "CacheManager"},
                {"file": "src/cache.py", "line": 3, "end_line": 4, "content": "# File: src/cache.py\nline2", "symbols": "refresh_cache"},
            ]
        )

        merged = server._merge_indexed_file_chunks("src/cache.py")

        self.assertEqual(merged["content"], "line1\nline2")
        self.assertNotIn("# File:", merged["content"])
        self.assertEqual(merged["coverage"]["removed_repeated_file_headers"], 2)

    def test_unranged_chunks_are_not_labeled_clean_full_file(self):
        server = self._server_with_chunks(
            [
                {"file": "src/cache.py", "content": "line1\nline2", "symbols": "CacheManager"},
                {"file": "src/cache.py", "content": "line2\nline3", "symbols": "refresh_cache"},
            ]
        )

        merged = server._merge_indexed_file_chunks("src/cache.py")

        self.assertFalse(merged["full_file"])
        self.assertTrue(merged["coverage"]["partial"])
        self.assertEqual(merged["coverage"]["source"], "indexed_chunks_merged_partial")

    def test_edit_suggestion_context_includes_deoverlapped_merged_file(self):
        server = self._server_with_chunks(
            [
                {"file": "src/cache.py", "line": 1, "end_line": 3, "content": "class CacheManager:\n    def refresh_cache(self):\n        return None", "symbols": "CacheManager refresh_cache"},
                {"file": "src/cache.py", "line": 3, "end_line": 5, "content": "        return None\n# tail\nDONE = True", "symbols": "refresh_cache"},
            ]
        )
        server.MODEL_CONTEXT_WINDOW = 4000

        chunks, loaded = server._expand_edit_suggestion_chunks(
            "fix src/cache.py bug",
            [{"file": "src/cache.py", "content": "snippet", "symbols": "CacheManager"}],
            {},
            {},
        )
        context, _info = server._pack_context_section(
            query="fix src/cache.py bug",
            map_section="",
            chunks=chunks,
            request_id="req-merge",
        )

        self.assertIn("src/cache.py", loaded)
        self.assertIn("class CacheManager", context)
        self.assertEqual(context.count("return None"), 1)


class TestEditSuggestionRemoteAndPackedEnforcement(unittest.TestCase):
    def test_edit_context_prefers_post_pack_chunks_over_dropped_retrieval(self):
        server = _import_server_with_stubs()
        debug_payload = {
            "normalized_retrieval": {
                "chunks": [
                    {"path": "src/dropped.py", "content": "class DroppedThing:\n    pass\n", "symbols": "DroppedThing"}
                ]
            },
            "final_context": {
                "packed_chunks": [
                    {"file": "src/kept.py", "content": "class KeptThing:\n    pass\n", "symbols": "KeptThing"}
                ]
            },
        }
        dropped_answer = _contract_answer("src/dropped.py", evidence="`DroppedThing`")
        kept_answer = _contract_answer("src/kept.py", evidence="`KeptThing`")

        self.assertIn(
            "I do not have enough repo-grounded context",
            server._enforce_edit_suggestion_answer(dropped_answer, debug_payload, "fix this bug"),
        )
        self.assertEqual(
            server._enforce_edit_suggestion_answer(kept_answer, debug_payload, "fix this bug"),
            kept_answer,
        )

    def test_edit_context_fails_closed_without_post_pack_chunks(self):
        server = _import_server_with_stubs()
        debug_payload = {
            "normalized_retrieval": {
                "chunks": [
                    {"path": "src/cache.py", "content": "class CacheManager:\n    pass\n", "symbols": "CacheManager"}
                ]
            }
        }
        answer = _contract_answer("src/cache.py", evidence="`CacheManager`")

        self.assertIn(
            "I do not have enough repo-grounded context",
            server._enforce_edit_suggestion_answer(answer, debug_payload, "fix this bug"),
        )

    def test_remote_non_stream_edit_suggestion_response_is_enforced(self):
        server = _import_server_with_stubs()
        server.llm = _GenericRemoteLlm()

        response = asyncio.run(server.remote_inference_ask(_JsonRequest(_remote_request_body(stream=False))))

        self.assertTrue(response["ok"])
        self.assertIn("I do not have enough repo-grounded context", response["answer"])
        self.assertIn("answer with concrete file paths and symbols", response["answer"])

    def test_remote_non_stream_grounded_edit_suggestion_survives(self):
        server = _import_server_with_stubs()
        server.llm = _GroundedRemoteLlm()

        response = asyncio.run(server.remote_inference_ask(_JsonRequest(_remote_request_body(stream=False))))

        self.assertTrue(response["ok"])
        self.assertIn("Finding:", response["answer"])
        self.assertIn("src/cache.py", response["answer"])
        self.assertIn("CacheManager", response["answer"])
        self.assertNotIn("I do not have enough repo-grounded context", response["answer"])

    def test_remote_stream_edit_suggestion_is_buffered_and_enforced(self):
        server = _import_server_with_stubs()
        server.llm = _GenericRemoteLlm()
        original_streaming_response = server.StreamingResponse
        try:
            server.StreamingResponse = lambda gen, **kwargs: {"gen": gen, "kwargs": kwargs}
            response = asyncio.run(server.remote_inference_ask(_JsonRequest(_remote_request_body(stream=True))))
            async def _collect():
                return [event async for event in response["gen"]]
            events = asyncio.run(_collect())
        finally:
            server.StreamingResponse = original_streaming_response

        text = _stream_text(events)
        self.assertIn("I do not have enough repo-grounded context", text)
        self.assertNotIn("You should consider improving architecture.", text)
        self.assertEqual(events[-1].strip(), "data: [DONE]")

    def _collect_proxy_events(self, server, *, query="fix this bug", remote_events=None, debug_mode=False):
        payload = _remote_request_body(stream=True, debug=debug_mode)
        payload["query"]["text"] = query
        client_debug = {
            "final_context": {
                "packed_chunks": [
                    {
                        "file": "src/cache.py",
                        "content": "class CacheManager:\n    def refresh_cache(self):\n        return None\n",
                        "symbols": "CacheManager refresh_cache",
                    }
                ]
            }
        }
        original_freshness = server._index_freshness_payload
        original_collect = server._collect_local_remote_payload
        original_urlopen = server.url_request.urlopen
        try:
            server._index_freshness_payload = lambda: {"ok": True, "has_index": True, "changed": False}
            server._collect_local_remote_payload = lambda **_kwargs: (payload, client_debug)
            server.url_request.urlopen = lambda *_args, **_kwargs: _FakeRemoteStreamResponse(remote_events or [])

            async def _collect():
                messages = [{"role": "user", "content": query}]
                return [
                    event
                    async for event in server._remote_proxy_stream_with_freshness(
                        messages, 128, "req-proxy-edit", debug_mode=debug_mode
                    )
                ]

            return asyncio.run(_collect())
        finally:
            server._index_freshness_payload = original_freshness
            server._collect_local_remote_payload = original_collect
            server.url_request.urlopen = original_urlopen

    def test_remote_proxy_edit_stream_normal_chunks_are_buffered_and_enforced(self):
        server = _import_server_with_stubs()
        events = self._collect_proxy_events(
            server,
            remote_events=[
                _remote_chunk_event("You should consider improving architecture."),
                "data: [DONE]\n\n",
            ],
        )

        text = _stream_text(events)
        self.assertIn("I do not have enough repo-grounded context", text)
        self.assertNotIn("You should consider improving architecture.", text)
        self.assertEqual(events[-1].strip(), "data: [DONE]")

    def test_remote_proxy_edit_stream_debug_enabled_preserves_debug_event_after_answer(self):
        server = _import_server_with_stubs()
        events = self._collect_proxy_events(
            server,
            remote_events=[
                _remote_chunk_event("You should consider improving architecture."),
                _remote_debug_event(),
                "data: [DONE]\n\n",
            ],
            debug_mode=True,
        )
        joined = "".join(events)

        self.assertIn("I do not have enough repo-grounded context", _stream_text(events))
        self.assertIn("event: debug", joined)
        self.assertIn('"object": "debug.payload"', joined)
        debug_index = next(i for i, event in enumerate(events) if event.startswith("event: debug"))
        self.assertGreater(debug_index, 0)
        self.assertEqual(events[-1].strip(), "data: [DONE]")

    def test_remote_proxy_edit_stream_debug_disabled_drops_debug_event(self):
        server = _import_server_with_stubs()
        events = self._collect_proxy_events(
            server,
            remote_events=[
                _remote_chunk_event("You should consider improving architecture."),
                _remote_debug_event(),
                "data: [DONE]\n\n",
            ],
            debug_mode=False,
        )
        joined = "".join(events)

        self.assertIn("I do not have enough repo-grounded context", _stream_text(events))
        self.assertNotIn("event: debug", joined)
        self.assertNotIn('"object": "debug.payload"', joined)
        self.assertEqual(events[-1].strip(), "data: [DONE]")

    def test_remote_proxy_edit_stream_without_done_reports_interrupted_not_fallback(self):
        server = _import_server_with_stubs()
        events = self._collect_proxy_events(
            server,
            remote_events=[
                _remote_chunk_event("Finding: partial answer from upstream"),
            ],
        )
        joined = "".join(events)

        self.assertIn("andescode.error", joined)
        self.assertIn("remote_stream_interrupted", joined)
        self.assertNotIn("I do not have enough repo-grounded context", joined)
        self.assertEqual(events[-1].strip(), "data: [DONE]")

    def test_remote_proxy_edit_stream_done_text_without_marker_reports_interrupted(self):
        server = _import_server_with_stubs()
        events = self._collect_proxy_events(
            server,
            remote_events=[
                _remote_chunk_event("Finding: literal [DONE] text but no completion marker"),
            ],
        )
        joined = "".join(events)

        self.assertIn("andescode.error", joined)
        self.assertIn("remote_stream_interrupted", joined)
        self.assertNotIn("I do not have enough repo-grounded context", joined)
        self.assertNotIn("literal [DONE] text", joined)
        self.assertEqual(events[-1].strip(), "data: [DONE]")

    def test_remote_proxy_edit_stream_error_surfaces_without_safe_fallback(self):
        server = _import_server_with_stubs()
        events = self._collect_proxy_events(
            server,
            remote_events=[
                _remote_error_event(code="remote_failed", message="boom"),
                "data: [DONE]\n\n",
            ],
        )
        joined = "".join(events)

        self.assertIn("andescode.error", joined)
        self.assertIn("remote_failed", joined)
        self.assertIn("boom", joined)
        self.assertNotIn("I do not have enough repo-grounded context", joined)
        self.assertEqual(events[-1].strip(), "data: [DONE]")

    def test_remote_proxy_non_edit_stream_passes_through_normally(self):
        server = _import_server_with_stubs()
        remote = _remote_chunk_event("pass-through-token")
        events = self._collect_proxy_events(
            server,
            query="how does this work?",
            remote_events=[remote, "data: [DONE]\n\n"],
        )

        self.assertIn(remote, events)
        self.assertIn("pass-through-token", "".join(events))
        self.assertEqual(events[-1].strip(), "data: [DONE]")

    def test_remote_proxy_non_edit_debug_stream_passes_through_unchanged(self):
        server = _import_server_with_stubs()
        remote = _remote_chunk_event("pass-through-token")
        debug = _remote_debug_event()
        events = self._collect_proxy_events(
            server,
            query="how does this work?",
            remote_events=[remote, debug, "data: [DONE]\n\n"],
            debug_mode=True,
        )

        self.assertIn(remote, events)
        self.assertIn(debug, events)
        self.assertEqual(events[-1].strip(), "data: [DONE]")

    def test_remote_proxy_non_edit_truncated_stream_still_passes_partial_then_error(self):
        server = _import_server_with_stubs()
        remote = _remote_chunk_event("partial-pass-through-token")
        events = self._collect_proxy_events(
            server,
            query="how does this work?",
            remote_events=[remote],
        )
        joined = "".join(events)

        self.assertIn(remote, events)
        self.assertIn("partial-pass-through-token", joined)
        self.assertIn("andescode.error", joined)
        self.assertIn("remote_stream_interrupted", joined)
        self.assertEqual(events[-1].strip(), "data: [DONE]")


if __name__ == "__main__":
    unittest.main()
