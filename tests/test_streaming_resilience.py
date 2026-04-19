import asyncio
import logging
import unittest
from pathlib import Path

from tests.test_server_stream_debug_mode import _import_server_with_stubs


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

        self.assertIn("❌ Sorry", contents)
        self.assertIn("phase: context_build", contents)
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


if __name__ == "__main__":
    unittest.main()
