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

    def test_fallback_allowed_when_top_tier_cannot_fit_any_chunk(self):
        server = self.server
        server.MODEL_CONTEXT_WINDOW = 820
        chunks = [
            {"file": "src/ui/ScheduleFragment.kt", "content": "A" * 1400},
            {"file": "fallback.py", "content": "f" * 120},
        ]

        _context, info = server._pack_context_section(
            query="Analyze ScheduleFragment.kt scroll update",
            map_section="",
            chunks=chunks,
            anchor_files=["ScheduleFragment.kt"],
            request_id="req-strict-priority",
        )

        self.assertGreater(info["packed_chunks"], 0)
        self.assertIn("fallback.py", info["kept_files"])
        self.assertNotIn("fallback.py", info["dropped_files"])

    def test_strict_priority_blocks_fallback_when_anchor_tier_partially_fits(self):
        server = self.server
        server.MODEL_CONTEXT_WINDOW = 900
        chunks = [
            {"file": "src/ui/ScheduleFragment.kt", "content": "A" * 220},
            {"file": "src/ui/ScheduleFragment.kt", "content": "A" * 1400},
            {"file": "fallback.py", "content": "f" * 120},
        ]

        _context, info = server._pack_context_section(
            query="Analyze ScheduleFragment.kt scroll update",
            map_section="",
            chunks=chunks,
            anchor_files=["ScheduleFragment.kt"],
            request_id="req-strict-priority-partial",
        )

        self.assertGreaterEqual(info["packed_chunks"], 1)
        self.assertIn("src/ui/ScheduleFragment.kt", info["kept_files"])
        self.assertNotIn("fallback.py", info["kept_files"])

    def test_conversation_history_is_accounted_for_in_budget(self):
        server = self.server
        server.MODEL_CONTEXT_WINDOW = 1050
        chunks = [{"file": f"src/f{i}.py", "content": "x" * 450} for i in range(8)]
        short_messages = [{"role": "user", "content": "quick question"}]
        long_messages = [
            {"role": "user", "content": "long context " + ("u" * 1400)},
            {"role": "assistant", "content": "prior answer " + ("a" * 1400)},
            {"role": "user", "content": "follow-up " + ("q" * 1200)},
        ]

        _short_context, short_info = server._pack_context_section(
            query="analyze flow",
            map_section="",
            chunks=chunks,
            conversation_messages=short_messages,
            request_id="req-short-history",
        )
        long_context, long_info = server._pack_context_section(
            query="analyze flow",
            map_section="",
            chunks=chunks,
            conversation_messages=long_messages,
            request_id="req-long-history",
        )

        self.assertGreater(short_info["packed_chunks"], long_info["packed_chunks"])
        self.assertTrue(long_info["truncated"])
        self.assertIn("Context truncated to fit model window", long_context)

    def test_anchor_basename_matching_works(self):
        server = self.server
        candidates = server._prioritize_chunk_candidates(
            [{"file": "app/feature/ScheduleFragment.kt", "content": "render()"}],
            anchor_files=["ScheduleFragment.kt"],
        )
        self.assertEqual(candidates[0]["tier"], 0)


    def test_declaration_query_prioritizes_authoritative_over_runtime_usage(self):
        server = self.server
        candidates = server._prioritize_chunk_candidates(
            [
                {"file": "src/runtime_usage.py", "content": "import requests", "source_type": "source_code"},
                {"file": "pyproject.toml", "content": "[tool.poetry.dependencies]", "source_type": "dependency_file"},
            ],
            query="what dependencies are declared",
        )
        self.assertEqual(candidates[0]["file"], "pyproject.toml")
        self.assertEqual(candidates[0]["authority_rank"], 0)
        self.assertEqual(candidates[1]["authority_rank"], 1)

    def test_declaration_guidance_is_injected_when_authoritative_chunks_are_missing(self):
        server = self.server
        context, _info = server._pack_context_section(
            query="what dependencies are declared",
            map_section="",
            chunks=[
                {"file": "src/runtime_usage.py", "content": "import requests", "source_type": "source_code"},
            ],
            request_id="req-decl-missing-auth",
        )
        self.assertIn("## Source-of-Truth Guidance", context)
        self.assertIn("Declared", context)
        self.assertIn("Inferred from usage", context)
        self.assertIn("state declaration files are missing", context)

    def test_dependency_context_packing_prefers_build_or_dependency_chunk_over_usage(self):
        server = self.server
        server.MODEL_CONTEXT_WINDOW = 960
        chunks = [
            {
                "file": "src/runtime_usage.py",
                "content": "u" * 520,
                "source_type": "source_code",
            },
            {
                "file": "build.gradle.kts",
                "content": "d" * 520,
                "source_type": "build_file",
            },
        ]

        _context, info = server._pack_context_section(
            query="what dependencies are declared",
            map_section="",
            chunks=chunks,
            request_id="req-decl-pack-priority",
        )

        self.assertGreaterEqual(info["packed_chunks"], 1)
        self.assertIn("build.gradle.kts", info["kept_files"])
        self.assertNotIn("src/runtime_usage.py", info["kept_files"])

    def test_declaration_queries_force_include_authoritative_chunk_before_trim(self):
        server = self.server
        server.MODEL_CONTEXT_WINDOW = 860
        chunks = [
            {
                "file": "app/src/runtime_usage.kt",
                "content": "u" * 320,
                "source_type": "source_code",
            },
            {
                "file": "app/buildSrc/dependencies.kt",
                "content": "d" * 5000,
                "source_type": "dependency_file",
            },
        ]

        context, info = server._pack_context_section(
            query="what dependencies are declared for this module",
            map_section="",
            chunks=chunks,
            authoritative_files=["app/buildSrc/dependencies.kt"],
            request_id="req-force-authoritative",
        )

        self.assertGreaterEqual(info["packed_chunks"], 1)
        self.assertIn("app/buildSrc/dependencies.kt", info["kept_files"])
        self.assertIn("dependencies.kt", context)
        self.assertEqual(info.get("forced_authoritative_file"), "app/buildSrc/dependencies.kt")

    def test_planned_context_injects_authoritative_candidates_before_packing(self):
        server = self.server
        server.INDEXER_READY = True
        seen_chunks = {}
        original_pack = server._pack_context_section

        def _capture_pack(**kwargs):
            seen_chunks["files"] = [c.get("file") for c in kwargs.get("chunks", [])]
            return "ctx", {
                "packed_chunks": 1,
                "kept_files": ["app/buildSrc/dependencies.kt"],
                "packed_chunks_raw": [{"file": "app/buildSrc/dependencies.kt", "content": "deps"}],
            }

        try:
            server._pack_context_section = _capture_pack
            server.search_codebase = lambda *_args, **_kwargs: [{"file": "src/runtime_usage.kt", "content": "runtime"}]
            server._indexer_module = types.SimpleNamespace(
                _load_project_map=lambda: {},
                _load_workspace_index=lambda: {
                    "import_graph": {"samples": {}},
                    "file_to_module_map": {},
                    "manifests": [],
                    "config_graph": {"config_files": ["app/buildSrc/dependencies.kt"]},
                },
                get_repo_fingerprint=lambda: "",
                get_chunks_for_file=lambda fname: (
                    [{"file": fname, "content": "implementation(\"x:y:1.0\")", "source_type": "dependency_file"}]
                    if fname == "app/buildSrc/dependencies.kt"
                    else []
                ),
                CACHE=None,
            )

            messages = [{"role": "user", "content": "what dependencies are declared"}]
            _msg, files_loaded = server._build_context_from_plan(messages, ["src/runtime_usage.kt"], "req-plan-auth")

            self.assertIn("app/buildSrc/dependencies.kt", seen_chunks.get("files", []))
            self.assertIn("app/buildSrc/dependencies.kt", files_loaded)
        finally:
            server._pack_context_section = original_pack

    def test_build_context_with_long_history_still_packs_without_overflow(self):
        server = self.server
        server.MODEL_CONTEXT_WINDOW = 1100
        server.INDEXER_READY = True
        server._indexer_module = types.SimpleNamespace(
            _load_project_map=lambda: {},
            _load_workspace_index=lambda: {},
            get_repo_fingerprint=lambda: "",
            CACHE=None,
        )
        server.search_codebase = lambda *_args, **_kwargs: [
            {"file": f"src/file_{i}.py", "content": "line\n" * 260} for i in range(6)
        ]
        messages = [
            {"role": "user", "content": "previous question " + ("u" * 1200)},
            {"role": "assistant", "content": "previous answer " + ("a" * 1200)},
            {"role": "user", "content": "trace ScheduleFragment.kt event flow"},
        ]

        result = server._build_context(messages, "req-long-convo")
        system_prompt = result[0]["content"]

        self.assertIn("Context truncated to fit model window", system_prompt)

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
