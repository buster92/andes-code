import asyncio
import logging
import types
import unittest
from pathlib import Path

from tests.unit.test_server_stream_debug_mode import _import_server_with_stubs


async def _collect_events(server_module):
    events = []
    messages = [{"role": "user", "content": "why is this failing?"}]
    async for event in server_module._stream(messages, 32, "req999", 0.0, debug_mode=False):
        events.append(event)
    return events


def _chunk_contents(events):
    out = []
    for raw in events:
        if not raw.startswith("data: ") or raw.strip() == "data: [DONE]":
            continue
        payload = raw[len("data: "):].strip()
        try:
            import json
            data = json.loads(payload)
            out.append(data["choices"][0]["delta"].get("content", ""))
        except Exception:
            continue
    return out


def _chunk_objects(events):
    out = []
    for raw in events:
        if not raw.startswith("data: ") or raw.strip() == "data: [DONE]":
            continue
        payload = raw[len("data: "):].strip()
        try:
            import json
            out.append(json.loads(payload))
        except Exception:
            continue
    return out


class _CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


class TestStreamingResilience(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = _import_server_with_stubs()

    def test_streaming_success_emits_real_content_and_done(self):
        server = self.server
        server._indexer_module = None
        server.orchestration_plan = lambda _intent: {"skip_patch_plan": True, "skip_neighborhood": True}
        server.classify_query_intent_details = lambda _query: {
            "intent": "runtime_usage_or_reference",
            "retrieval_route": "semantic",
        }
        server._build_context = lambda messages, request_id, debug_mode=False, return_debug=False: (messages, None)

        events = asyncio.run(_collect_events(server))
        contents = "".join(_chunk_contents(events))

        self.assertIn("stub-token", contents)
        self.assertEqual(events[-1].strip(), "data: [DONE]")

    def test_streaming_failure_before_generation_emits_error_and_done(self):
        server = self.server
        server._indexer_module = None
        server.orchestration_plan = lambda _intent: {"skip_patch_plan": True, "skip_neighborhood": True}
        server.classify_query_intent_details = lambda _query: {
            "intent": "runtime_usage_or_reference",
            "retrieval_route": "semantic",
        }

        def _boom(*_args, **_kwargs):
            raise RuntimeError("context exploded")

        server._build_context = _boom

        events = asyncio.run(_collect_events(server))
        contents = "\n".join(_chunk_contents(events))
        objects = _chunk_objects(events)

        self.assertIn("❌ Sorry", contents)
        self.assertIn("phase: context_build", contents)
        self.assertTrue(any(obj.get("object") == "andescode.error" for obj in objects))
        self.assertEqual(events[-1].strip(), "data: [DONE]")

    def test_phase_logs_and_external_log_path(self):
        server = self.server
        server._indexer_module = None
        server.orchestration_plan = lambda _intent: {"skip_patch_plan": True, "skip_neighborhood": True}
        server.classify_query_intent_details = lambda _query: {
            "intent": "runtime_usage_or_reference",
            "retrieval_route": "semantic",
        }
        server._build_context = lambda messages, request_id, debug_mode=False, return_debug=False: (messages, None)

        handler = _CaptureHandler()
        server.audit.addHandler(handler)
        try:
            asyncio.run(_collect_events(server))
        finally:
            server.audit.removeHandler(handler)

        combined = "\n".join(handler.messages)
        self.assertIn("phase=request_received", combined)
        self.assertIn("phase=intent_classified", combined)
        self.assertIn("phase=context_build_start", combined)
        self.assertIn("phase=generation_start", combined)
        self.assertIn("phase=generation_completed", combined)

        server._build_context = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("context failed"))
        handler_fail = _CaptureHandler()
        server.audit.addHandler(handler_fail)
        try:
            asyncio.run(_collect_events(server))
        finally:
            server.audit.removeHandler(handler_fail)
        self.assertIn("phase=pipeline_failed", "\n".join(handler_fail.messages))

        repo_root = Path(__file__).resolve().parents[1]
        log_path = Path(server.LOG_PATH).resolve()
        self.assertNotEqual(log_path, repo_root / "audit.log")
        self.assertFalse(str(log_path).startswith(str(repo_root)))

    def test_index_logging_emits_started_phases_once(self):
        server = self.server
        server.INDEXER_READY = True
        server._indexer_module = object()
        old_auto_manager = server._auto_index_manager
        server._auto_index_manager = None

        import sys
        fake_indexer = sys.modules["indexer"]
        fake_indexer.index_codebase_stream = lambda _path: iter([
            {"type": "scan", "files": 3, "new": 2, "unchanged": 1},
            {"type": "embed", "done": 1, "total": 2},
            {"type": "embed", "done": 2, "total": 2},
            {"type": "store", "done": 1, "total": 2},
            {"type": "store", "done": 2, "total": 2},
            {"type": "mapping", "message": "Building project map..."},
            {"type": "done", "indexed": 2, "chunks": 2, "decision": "incremental"},
        ])

        handler = _CaptureHandler()
        server.audit.addHandler(handler)
        events = []
        try:
            ok = server._run_index_stream("/tmp/project", "manual", events.append)
        finally:
            server.audit.removeHandler(handler)
            server._auto_index_manager = old_auto_manager

        self.assertTrue(ok)
        logs = "\n".join(handler.messages)
        self.assertEqual(logs.count("phase=embedding_started"), 1)
        self.assertEqual(logs.count("phase=storage_started"), 1)
        self.assertIn("phase=embedding_completed", logs)
        self.assertIn("phase=storage_completed", logs)

    def test_file_neighborhood_handles_structured_import_graph_without_type_error(self):
        server = self.server
        workspace = {
            "import_graph": {
                "edge_count": 5,
                "samples": {
                    "ScheduleFragment.kt": ["GuideViewModel.kt"],
                },
            },
            "file_to_module_map": {
                "ScheduleFragment.kt": "app",
                "GuideViewModel.kt": "app",
            },
        }

        result = server._file_neighborhood("ScheduleFragment.kt", "bugfix", workspace, repo_fp="")

        self.assertIn("ScheduleFragment.kt", result)
        self.assertIn("GuideViewModel.kt", result)

    def test_file_neighborhood_ignores_malformed_samples_safely(self):
        server = self.server
        workspace = {
            "import_graph": {
                "edge_count": 5,
                "samples": 123,
            },
            "file_to_module_map": {
                "ScheduleFragment.kt": "app",
                "GuideViewModel.kt": "app",
            },
        }

        result = server._file_neighborhood("ScheduleFragment.kt", "bugfix", workspace, repo_fp="")

        self.assertIn("ScheduleFragment.kt", result)

    def test_planned_context_streaming_with_structured_import_graph_does_not_emit_pipeline_error(self):
        server = self.server
        server.INDEXER_READY = True
        server._indexer_module = types.SimpleNamespace(
            _load_project_map=lambda: {"project": "demo", "file_symbols": {"ScheduleFragment.kt": ["render"]}},
            _load_workspace_index=lambda: {
                "import_graph": {
                    "edge_count": 5,
                    "samples": {
                        "ScheduleFragment.kt": ["GuideViewModel.kt"],
                    },
                },
                "file_to_module_map": {
                    "ScheduleFragment.kt": "app",
                    "GuideViewModel.kt": "app",
                },
            },
            get_repo_fingerprint=lambda: "fp-structured-import-graph",
            get_chunks_for_file=lambda fname: [{"file": fname, "content": f"// chunk for {fname}", "full_file": True}],
            CACHE=None,
        )
        server.classify_query_intent_details = lambda _query: {
            "intent": "code_fix_or_patch",
            "retrieval_route": "semantic",
        }
        server.orchestration_plan = lambda _intent: {"skip_patch_plan": False, "skip_neighborhood": False}
        server._diagnose_query = lambda _query, _intent: {"intent": "code_fix_or_patch", "mode": "bugfix"}
        server._plan_files = lambda _query, _pmap: ["ScheduleFragment.kt"]
        server.search_codebase = lambda *_args, **_kwargs: []

        events = asyncio.run(_collect_events(server))
        contents = "\n".join(_chunk_contents(events))
        objects = _chunk_objects(events)

        self.assertIn("stub-token", contents)
        self.assertFalse(any(obj.get("object") == "andescode.error" for obj in objects))
        self.assertNotIn("phase: context_build", contents)
        self.assertEqual(events[-1].strip(), "data: [DONE]")


if __name__ == "__main__":
    unittest.main()
