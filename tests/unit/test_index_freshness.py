import asyncio
import json
import tempfile
import types
import unittest
from pathlib import Path

from tests.unit.test_remote_inference_server_path import _import_server_with_stubs


class _Req:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


class TestIndexFreshness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server, _ = _import_server_with_stubs()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.file = self.root / "app.py"
        self.file.write_text("print('one')\n")
        self.server.INDEXER_READY = True
        self.server.search_codebase = lambda *args, **kwargs: [{"file": "app.py", "content": "print('one')", "score": 1.0}]
        self.server._set_active_index_session(True)
        self.indexer = types.SimpleNamespace()
        initial_hash = "hash:" + self.file.read_text()
        self.indexer._file_hash = lambda fp: "hash:" + Path(fp).read_text()
        self.indexer._collect_files = lambda root: [p for p in Path(root).rglob("*.py") if p.is_file()]
        self.indexer._load_hashes = lambda: {"__root__": str(self.root), "app.py": initial_hash}
        self.indexer._load_project_map = lambda: {"project": "tmp"}
        self.indexer._load_index_state = lambda: {"state": "ready"}
        self.indexer.get_repo_fingerprint = lambda: "fp-1"
        self.indexer.CACHE = None
        self.server._indexer_module = self.indexer

    def tearDown(self):
        self.tmp.cleanup()

    def test_freshness_unchanged_when_hashes_match(self):
        result = self.server._index_freshness_payload()
        self.assertTrue(result["ok"])
        self.assertTrue(result["has_index"])
        self.assertFalse(result["changed"])
        self.assertEqual(result["changed_count"], 0)
        self.assertEqual(result["deleted_count"], 0)

    def test_freshness_changed_when_file_hash_differs(self):
        self.file.write_text("print('two')\n")
        result = self.server._index_freshness_payload()
        self.assertTrue(result["changed"])
        self.assertEqual(result["changed_count"], 1)
        self.assertEqual(result["deleted_count"], 0)

    def test_freshness_reports_deleted_indexed_file(self):
        self.file.unlink()
        result = self.server._index_freshness_payload()
        self.assertTrue(result["changed"])
        self.assertEqual(result["changed_count"], 0)
        self.assertEqual(result["deleted_count"], 1)

    def test_query_time_refresh_before_non_stream_answer_uses_incremental(self):
        self.file.write_text("print('two')\n")
        calls = []
        self.server.get_execution_mode = lambda: self.server.ExecutionMode.LOCAL
        self.server._build_context = lambda messages, request_id, debug_mode=False, return_debug=False: (messages, [], {}) if return_debug else (messages, [])
        class _Orchestrator:
            def __init__(self, **_kwargs):
                pass
            def run_non_stream(self, **_kwargs):
                calls.append(("retrieval", None))
                return "answer", {}
        self.server.LocalAskOrchestrator = _Orchestrator
        def _refresh(path, source, emit_event, change_batch=None, force_refresh=False):
            calls.append(("refresh", force_refresh))
            emit_event({"type": "done", "indexed": 1, "chunks": 1})
            return True
        self.server._run_index_stream = _refresh

        result = asyncio.run(self.server.chat(_Req({"messages": [{"role": "user", "content": "hi"}], "stream": False})))

        self.assertEqual(calls[0], ("refresh", False))
        self.assertEqual(calls[1], ("retrieval", None))
        self.assertEqual(result["choices"][0]["message"]["content"], "answer")

    def test_query_time_refresh_failure_stops_non_stream_answer(self):
        self.file.write_text("print('two')\n")
        self.server.get_execution_mode = lambda: self.server.ExecutionMode.LOCAL
        self.server.LocalAskOrchestrator = lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not answer"))
        self.server._run_index_stream = lambda *args, **kwargs: False

        result = asyncio.run(self.server.chat(_Req({"messages": [{"role": "user", "content": "hi"}], "stream": False})))

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "index_refresh_failed")

    def test_remote_inference_refreshes_locally_before_payload(self):
        self.file.write_text("print('two')\n")
        calls = []
        self.server.get_execution_mode = lambda: self.server.ExecutionMode.REMOTE_INFERENCE
        self.server._collect_local_remote_payload = lambda **_kwargs: (calls.append(("payload", None)) or {"workspace": {}, "retrieval": {}}, {})
        self.server.url_request.urlopen = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stop after payload"))
        def _refresh(path, source, emit_event, change_batch=None, force_refresh=False):
            calls.append(("refresh", force_refresh))
            emit_event({"type": "done", "indexed": 1, "chunks": 1})
            return True
        self.server._run_index_stream = _refresh

        result = asyncio.run(self.server.chat(_Req({"messages": [{"role": "user", "content": "hi"}], "stream": False})))

        self.assertEqual(calls[0], ("refresh", False))
        self.assertEqual(calls[1], ("payload", None))
        self.assertEqual(result["error"]["code"], "remote_proxy_error")


class TestStaticFreshnessUi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ui = Path("static/index.html").read_text()
        cls.readme = Path("README.md").read_text()
        cls.server_text = Path("server.py").read_text()

    def test_auto_index_language_removed(self):
        combined = "\n".join([self.ui, self.readme, self.server_text])
        self.assertNotIn("ANDESCODE" + "_AUTO" + "_INDEX", combined)
        self.assertNotIn("Auto" + "-index: " + "watch" + "ing", combined)

    def test_focus_check_prompts_but_does_not_index_automatically(self):
        self.assertIn("window.addEventListener('focus', maybeCheckIndexFreshnessOnFocus)", self.ui)
        self.assertIn("/v1/index/freshness", self.ui)
        focus_body = self.ui.split("async function maybeCheckIndexFreshnessOnFocus", 1)[1].split("async function reindexCodebase", 1)[0]
        self.assertNotIn("indexCodebase(", focus_body)
        self.assertIn("await indexCodebase(false)", self.ui)
        self.assertIn("await indexCodebase(true)", self.ui)
