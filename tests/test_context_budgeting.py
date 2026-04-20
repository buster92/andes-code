import asyncio
import types
import unittest

from tests.test_server_stream_debug_mode import _import_server_with_stubs


async def _collect_events(server_module, query):
    events = []
    messages = [{"role": "user", "content": query}]
    async for event in server_module._stream(messages, 64, "req-budget", 0.0, debug_mode=False):
        events.append(event)
    return events


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


class TestContextBudgeting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = _import_server_with_stubs()

    def setUp(self):
        self.server.MODEL_CONTEXT_WINDOW = 1200
        self.server.CONTEXT_RESERVED_RESPONSE_TOKENS = 320
        self.server.CONTEXT_SAFETY_MARGIN_TOKENS = 128

    def test_budget_packing_truncates_and_marks_context(self):
        server = self.server
        server.MODEL_CONTEXT_WINDOW = 900
        chunks = [
            {"file": f"src/file_{i}.py", "content": "x" * 500, "full_file": i == 0}
            for i in range(10)
        ]

        context, info = server._pack_context_section(
            query="explain behavior",
            map_section="",
            chunks=chunks,
            request_id="req-pack",
        )

        self.assertTrue(info["truncated"])
        self.assertLess(info["packed_chunks"], info["considered_chunks"])
        self.assertIn("Context truncated to fit model window", context)

    def test_priority_preservation_under_tight_budget(self):
        server = self.server
        server.MODEL_CONTEXT_WINDOW = 950
        chunks = [
            {"file": "fallback.py", "content": "f" * 300},
            {"file": "neighbor.py", "content": "n" * 300},
            {"file": "planned.py", "content": "p" * 300},
            {"file": "ScheduleFragment.kt", "content": "a" * 300},
        ]

        context, info = server._pack_context_section(
            query="Trace ScheduleFragment.kt update path",
            map_section="",
            chunks=chunks,
            anchor_files=["ScheduleFragment.kt"],
            planned_files=["planned.py"],
            neighbor_files=["neighbor.py"],
            request_id="req-priority",
        )

        self.assertTrue(info["packed_chunks"] >= 1)
        kept_files = info["kept_files"]
        self.assertIn("ScheduleFragment.kt", kept_files)
        if "fallback.py" in kept_files:
            self.fail("fallback chunk should not survive before higher priority chunks")
        self.assertIn("ScheduleFragment.kt", context)

    def test_planned_context_overflow_regression_stream_completes(self):
        server = self.server
        server.MODEL_CONTEXT_WINDOW = 1100
        server.INDEXER_READY = True
        server._indexer_module = types.SimpleNamespace(
            _load_project_map=lambda: {"project": "demo", "file_symbols": {"ScheduleFragment.kt": ["render"]}},
            _load_workspace_index=lambda: {
                "import_graph": {"samples": {"ScheduleFragment.kt": ["GuideViewModel.kt"]}},
                "file_to_module_map": {
                    "ScheduleFragment.kt": "app",
                    "GuideViewModel.kt": "app",
                    "neighbor.py": "app",
                },
            },
            get_repo_fingerprint=lambda: "fp-budget",
            get_chunks_for_file=lambda fname: [{"file": fname, "content": (fname + "\n") * 60, "full_file": True}],
            CACHE=None,
        )
        server.classify_query_intent_details = lambda _query: {
            "intent": "code_fix_or_patch",
            "retrieval_route": "semantic",
        }
        server.orchestration_plan = lambda _intent: {"skip_patch_plan": False, "skip_neighborhood": False}
        server._diagnose_query = lambda _query, _intent: {"intent": "code_fix_or_patch", "mode": "bugfix"}
        server._plan_files = lambda _query, _pmap: ["ScheduleFragment.kt"]
        server.search_codebase = lambda *_args, **_kwargs: [{"file": "fallback.py", "content": "z" * 2200}]

        messages, files_loaded, debug_payload = server._build_context_from_plan(
            [{"role": "user", "content": "Trace flow into ScheduleFragment.kt on scroll update"}],
            ["ScheduleFragment.kt"],
            "req-overflow",
            diagnosis={"intent": "code_fix_or_patch", "mode": "bugfix"},
            debug_mode=True,
            return_debug=True,
        )
        system_msg = messages[0]["content"]

        events = asyncio.run(_collect_events(server, "Trace flow into ScheduleFragment.kt on scroll update"))
        objects = _chunk_objects(events)

        self.assertIn("Context truncated to fit model window", system_msg)
        self.assertIn("ScheduleFragment.kt", debug_payload["final_context"]["files_used"])
        self.assertFalse(any(obj.get("object") == "andescode.error" for obj in objects))
        self.assertEqual(events[-1].strip(), "data: [DONE]")
        self.assertIn("ScheduleFragment.kt", files_loaded)


if __name__ == "__main__":
    unittest.main()
