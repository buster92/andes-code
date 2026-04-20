import asyncio
import importlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


def _import_server_with_stubs():
    tmp_model = Path(tempfile.mkdtemp()) / "fake.gguf"
    tmp_model.write_text("stub")
    os.environ["MODEL_PATH"] = str(tmp_model)

    fake_llama_cpp = types.ModuleType("llama_cpp")

    class FakeLlamaCache:
        def __init__(self, *args, **kwargs):
            pass

    class FakeLlama:
        def __init__(self, *args, **kwargs):
            pass

        def set_cache(self, *_args, **_kwargs):
            return None

        def __call__(self, _prompt, max_tokens=0, stream=False, echo=False):
            if stream:
                def _gen():
                    yield {"choices": [{"text": "stub-token"}]}
                return _gen()
            return {"choices": [{"text": "a.py"}]}

    fake_llama_cpp.Llama = FakeLlama
    fake_llama_cpp.LlamaCache = FakeLlamaCache
    sys.modules["llama_cpp"] = fake_llama_cpp

    fake_indexer = types.ModuleType("indexer")
    fake_indexer.search = lambda query, n_results=5, debug_mode=False, return_debug=False: []
    fake_indexer._load_project_map = lambda: {}
    fake_indexer._load_workspace_index = lambda: {}
    fake_indexer.get_repo_fingerprint = lambda: ""
    fake_indexer.get_chunks_for_file = lambda _fname: []
    fake_indexer.format_project_map_for_prompt = lambda _pmap: ""
    fake_indexer.CACHE = None
    sys.modules["indexer"] = fake_indexer

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *_args, **_kwargs: None
    sys.modules["dotenv"] = fake_dotenv

    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.run = lambda *_args, **_kwargs: None
    sys.modules["uvicorn"] = fake_uvicorn

    fake_fastapi = types.ModuleType("fastapi")

    class _FakeFastAPI:
        def __init__(self, *args, **kwargs):
            pass

        def mount(self, *args, **kwargs):
            return None

        def get(self, *_args, **_kwargs):
            return lambda fn: fn

        def post(self, *_args, **_kwargs):
            return lambda fn: fn

    class _FakeRequest:
        async def json(self):
            return {}

    fake_fastapi.FastAPI = _FakeFastAPI
    fake_fastapi.Request = _FakeRequest
    sys.modules["fastapi"] = fake_fastapi

    fake_fastapi_responses = types.ModuleType("fastapi.responses")
    fake_fastapi_responses.StreamingResponse = lambda *args, **kwargs: {"stream": True}
    fake_fastapi_responses.HTMLResponse = lambda body, status_code=200: {"body": body, "status_code": status_code}
    sys.modules["fastapi.responses"] = fake_fastapi_responses

    fake_fastapi_staticfiles = types.ModuleType("fastapi.staticfiles")
    fake_fastapi_staticfiles.StaticFiles = lambda *args, **kwargs: None
    sys.modules["fastapi.staticfiles"] = fake_fastapi_staticfiles

    import server

    return importlib.reload(server)


def _collect_stream(server_module, *, debug_mode=True):
    async def _collect():
        events = []
        messages = [{"role": "user", "content": "how does this work?"}]
        async for event in server_module._stream(messages, 16, "req123", 0.0, debug_mode=debug_mode):
            events.append(event)
        return events

    return asyncio.run(_collect())


class TestServerStreamingDebugMode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = _import_server_with_stubs()

    def test_debug_checkbox_request_flag_enables_debug_when_env_off(self):
        prev = os.environ.pop("ANDESCODE_DEBUG_MODE", None)
        try:
            self.assertTrue(self.server._resolve_request_debug_mode(True))
            self.assertFalse(self.server._resolve_request_debug_mode(False))
        finally:
            if prev is not None:
                os.environ["ANDESCODE_DEBUG_MODE"] = prev

    def test_build_context_from_plan_returns_debug_payload(self):
        server = self.server
        server.INDEXER_READY = True
        server._indexer_module = types.SimpleNamespace(
            _load_project_map=lambda: {"project": "demo", "file_symbols": {"src/a.py": ["run"]}},
            _load_workspace_index=lambda: {},
            get_repo_fingerprint=lambda: "fp",
            get_chunks_for_file=lambda fname: [{"file": fname, "content": "def run():\n    return 1", "full_file": True}],
        )
        server.search_codebase = lambda *_args, **_kwargs: [{"file": "src/extra.py", "content": "x = 1"}]

        messages, files_loaded, debug_payload = server._build_context_from_plan(
            [{"role": "user", "content": "find run"}],
            ["src/a.py"],
            "req123",
            diagnosis={"intent": "code_fix_or_patch", "mode": "bugfix"},
            debug_mode=True,
            return_debug=True,
        )

        self.assertTrue(messages)
        self.assertIn("src/a.py", files_loaded)
        self.assertEqual(debug_payload["query"], "find run")
        self.assertEqual(debug_payload["orchestration_path"], "planned_context")
        self.assertIn("planned_files", debug_payload["planning"])
        self.assertIn("files_used", debug_payload["final_context"])

    def test_stream_direct_retrieval_emits_debug_event(self):
        server = self.server
        server._indexer_module = None
        server.orchestration_plan = lambda _intent: {"skip_patch_plan": True, "skip_neighborhood": True}
        server.classify_query_intent_details = lambda _query: {
            "intent": "runtime_usage_or_reference",
            "retrieval_route": "semantic",
        }
        server._build_context = lambda messages, request_id, debug_mode=False, return_debug=False: (
            messages,
            {"query": "q", "retrieval": {}, "orchestration_path": "direct_retrieval"},
        )

        events = _collect_stream(server, debug_mode=True)
        debug_events = [e for e in events if "event: debug" in e]
        self.assertTrue(debug_events)
        self.assertIn('"orchestration_path": "direct_retrieval"', debug_events[-1])

    def test_stream_planner_route_emits_debug_event(self):
        server = self.server
        server.INDEXER_READY = True
        server._indexer_module = types.SimpleNamespace(
            _load_project_map=lambda: {"project": "demo", "file_symbols": {"src/a.py": ["run"]}},
            get_repo_fingerprint=lambda: "",
            CACHE=None,
        )
        server.classify_query_intent_details = lambda _query: {
            "intent": "code_fix_or_patch",
            "retrieval_route": "semantic",
        }
        server.orchestration_plan = lambda _intent: {"skip_patch_plan": False, "skip_neighborhood": False}
        server._diagnose_query = lambda _query, _intent: {"intent": "code_fix_or_patch", "mode": "bugfix"}
        server._plan_files = lambda _query, _pmap: ["src/a.py"]
        server._build_context_from_plan = lambda messages, planned_files, request_id, diagnosis, debug_mode=False, return_debug=False: (
            messages,
            planned_files,
            {"query": "q", "orchestration_path": "planned_context", "retrieval": {"route_taken": "planned_context"}},
        )

        events = _collect_stream(server, debug_mode=True)
        debug_events = [e for e in events if "event: debug" in e]
        self.assertTrue(debug_events)
        self.assertIn('"orchestration_path": "planned_context"', debug_events[-1])

    def test_root_and_index_state_include_integrity_probe(self):
        server = self.server
        server.INDEXER_READY = True
        server._indexer_module = types.SimpleNamespace(
            col=types.SimpleNamespace(count=lambda: 3),
            get_startup_integrity_probe=lambda: {
                "warning_active": True,
                "warning_message": "Index incomplete — results may be partial",
            },
        )
        root = server.root()
        state = server.index_state()
        self.assertIn("integrity_probe", root)
        self.assertTrue(root["integrity_probe"]["warning_active"])
        self.assertIn("integrity_probe", state)

    def test_index_state_tolerates_missing_integrity_probe_getter(self):
        server = self.server
        server.INDEXER_READY = True
        server._indexer_module = types.SimpleNamespace(col=types.SimpleNamespace(count=lambda: 1))
        root = server.root()
        state = server.index_state()
        self.assertEqual(root["integrity_probe"], {})
        self.assertEqual(state["integrity_probe"], {})

    def test_performance_query_enables_high_signal_policy(self):
        server = self.server
        policy, query_type = server._reasoning_policy_for_query("Why is scroll jank on main thread?")
        self.assertEqual(query_type, "performance")
        self.assertIn("High-Signal Performance Analysis Mode", policy)

    def test_high_signal_output_validation_is_conservative_for_mixed_items(self):
        server = self.server
        text = (
            "- Path A frequency: per frame, thread: main, cost: high\n"
            "- Path B frequency: once, thread: background, cost: low\n"
        )
        filtered, removed = server._validate_high_signal_output(text, True)
        self.assertEqual(removed, 0)
        self.assertIn("Path A", filtered)
        self.assertIn("Path B", filtered)

    def test_build_context_injects_reasoning_policy_for_performance_query(self):
        server = self.server
        server.INDEXER_READY = False
        messages = [{"role": "user", "content": "Investigate scroll lag in ScheduleFragment"}]
        built = server._build_context(messages, "req123", debug_mode=False, return_debug=False)
        self.assertIn("REASONING POLICY", built[0]["content"])
        self.assertIn("High-Signal Performance Analysis Mode", built[0]["content"])

    def test_streaming_still_emits_incremental_answer_tokens(self):
        server = self.server
        server._indexer_module = None
        server.orchestration_plan = lambda _intent: {"skip_patch_plan": True, "skip_neighborhood": True}
        server.classify_query_intent_details = lambda _query: {
            "intent": "runtime_usage_or_reference",
            "retrieval_route": "semantic",
        }
        server._build_context = lambda messages, request_id, debug_mode=False, return_debug=False: (
            messages,
            {"query": "q", "retrieval": {}, "orchestration_path": "direct_retrieval"},
        )

        class _TokenLlm:
            def __call__(self, _prompt, max_tokens=0, stream=False, echo=False):
                if stream:
                    def _gen():
                        yield {"choices": [{"text": "hot-path-1 "}]}
                        yield {"choices": [{"text": "hot-path-2"}]}
                    return _gen()
                return {"choices": [{"text": "unused"}]}

        server.llm = _TokenLlm()
        events = _collect_stream(server, debug_mode=False)
        hot_path_1_idx = next(i for i, e in enumerate(events) if "hot-path-1" in e)
        hot_path_2_idx = next(i for i, e in enumerate(events) if "hot-path-2" in e)
        done_idx = next(i for i, e in enumerate(events) if "[DONE]" in e)
        self.assertLess(hot_path_1_idx, done_idx)
        self.assertLess(hot_path_2_idx, done_idx)
        self.assertLess(hot_path_1_idx, hot_path_2_idx)

    def test_high_signal_validation_preserves_multiline_structured_finding(self):
        server = self.server
        text = (
            "1. Scroll diff churn\n"
            "   - frequency: per frame\n"
            "   - thread: main\n"
            "   - cost: high\n"
            "   - data flow → execution → UI impact: VM emit → adapter submitList → bind storm → frame drop\n"
        )
        filtered, removed = server._validate_high_signal_output(text, True)
        self.assertEqual(removed, 0)
        self.assertIn("Scroll diff churn", filtered)
        self.assertIn("frame drop", filtered)

    def test_high_signal_validation_filters_architecture_only_block_but_keeps_hot_path(self):
        server = self.server
        text = (
            "Architecture notes:\n"
            "Use repository + use case with dependency injection for cleaner separation.\n\n"
            "Hot path finding:\n"
            "ViewModel emit → Fragment accept → adapter update → RecyclerView bind → frame drop.\n"
            "frequency: per frame, thread: main, cost: high\n"
        )
        filtered, removed = server._validate_high_signal_output(text, True)
        self.assertEqual(removed, 1)
        self.assertNotIn("Architecture notes", filtered)
        self.assertIn("Hot path finding", filtered)


if __name__ == "__main__":
    unittest.main()
