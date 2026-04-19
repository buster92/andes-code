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


if __name__ == "__main__":
    unittest.main()
