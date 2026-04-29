import asyncio
import importlib
import json
import os
import sys
import tempfile
import types
import unittest
import urllib.error
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
                    yield {"choices": [{"text": "remote-token"}]}
                return _gen()
            return {"choices": [{"text": "remote-non-stream-answer"}]}

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
        def __init__(self, payload):
            self._payload = payload

        async def json(self):
            return self._payload

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

    return importlib.reload(server), _FakeRequest


def _valid_remote_request():
    return {
        "client": {
            "client_version": "1.0.0",
            "protocol_version": "andes.remote.v1",
            "platform": "darwin-arm64",
        },
        "workspace": {
            "workspace_id": "ws-1",
            "repo_name": "andes-code",
            "repo_root_name": "andes-code",
            "branch": "main",
            "commit_hash": "abc123",
            "is_dirty": False,
        },
        "query": {
            "request_id": "req-1",
            "text": "Where does indexing start?",
            "requested_at": "2026-01-01T00:00:00+00:00",
        },
        "retrieval": {
            "strategy": "semantic",
            "top_k": 5,
            "indexed_at": "2026-01-01T00:00:00+00:00",
            "index_state": "ready",
            "total_candidate_files": 10,
            "retrieved_chunk_count": 1,
        },
        "chunks": [
            {
                "chunk_id": "c1",
                "path": "server.py",
                "language": "python",
                "start_line": 1,
                "end_line": 3,
                "score": 0.9,
                "source_type": "source_code",
                "authority": "declared",
                "authority_reason": "test",
                "content": "def start():\n  pass",
            }
        ],
        "options": {"stream": False, "debug": True, "max_answer_tokens": 32},
    }


class TestRemoteInferenceServerPath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server, cls.req_type = _import_server_with_stubs()

    def setUp(self):
        self.server._set_active_index_session(True)

    def test_remote_ask_non_stream_happy_path(self):
        req = self.req_type(_valid_remote_request())
        result = asyncio.run(self.server.remote_inference_ask(req))
        self.assertTrue(result["ok"])
        self.assertEqual(result["event"], "final_answer")
        self.assertEqual(result["request_id"], "req-1")
        self.assertIn("debug", result)

    def test_remote_ask_rejects_missing_chunks(self):
        payload = _valid_remote_request()
        payload["chunks"] = []
        req = self.req_type(payload)
        result = asyncio.run(self.server.remote_inference_ask(req))
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "validation_error")

    def test_remote_ask_rejects_unsupported_protocol(self):
        payload = _valid_remote_request()
        payload["client"]["protocol_version"] = "andes.remote.v2"
        req = self.req_type(payload)
        result = asyncio.run(self.server.remote_inference_ask(req))
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "validation_error")

    def test_chat_remote_inference_mode_returns_payload_error_when_no_chunks(self):
        self.server.get_execution_mode = lambda: self.server.ExecutionMode.REMOTE_INFERENCE
        self.server.INDEXER_READY = True
        self.server._indexer_module = types.SimpleNamespace(_load_index_state=lambda: {}, ROOT=".")
        self.server.search_codebase = lambda *_args, **_kwargs: []

        class _Req:
            async def json(self):
                return {
                    "messages": [{"role": "user", "content": "missing context?"}],
                    "stream": False,
                    "max_tokens": 16,
                }

        result = asyncio.run(self.server.chat(_Req()))
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "empty_retrieval")

    def test_chat_remote_inference_mode_non_stream_success(self):
        self.server.get_execution_mode = lambda: self.server.ExecutionMode.REMOTE_INFERENCE
        self.server.INDEXER_READY = True
        self.server._indexer_module = types.SimpleNamespace(_load_index_state=lambda: {}, ROOT=".")
        self.server.search_codebase = lambda *_args, **_kwargs: [
            {"file": "server.py", "content": "def a():\n  pass", "line": 1, "score": 0.7}
        ]
        self.server.subprocess.check_output = lambda *args, **kwargs: "stub"

        captured = {"payload": None}

        class _Resp:
            def __init__(self, body: dict):
                self._body = json.dumps(body).encode("utf-8")

            def read(self, _size: int | None = None):
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        class _ReqObj:
            def __init__(self, data):
                self.data = data

        def _request_stub(_url, data=None, headers=None, method=None):
            captured["payload"] = json.loads(data.decode("utf-8"))
            return _ReqObj(data)

        self.server.url_request.Request = _request_stub
        self.server.url_request.urlopen = lambda *_args, **_kwargs: _Resp(
            {"ok": True, "event": "final_answer", "answer": "remote answer"}
        )

        class _Req:
            async def json(self):
                return {
                    "messages": [{"role": "user", "content": "what is a?"}],
                    "stream": False,
                    "max_tokens": 16,
                }

        result = asyncio.run(self.server.chat(_Req()))
        self.assertEqual(result["object"], "chat.completion")
        self.assertEqual(result["choices"][0]["message"]["content"], "remote answer")
        self.assertEqual(captured["payload"]["options"]["stream"], False)

    def test_chat_remote_inference_mode_non_stream_unreachable(self):
        self.server.get_execution_mode = lambda: self.server.ExecutionMode.REMOTE_INFERENCE
        self.server.INDEXER_READY = True
        self.server._indexer_module = types.SimpleNamespace(_load_index_state=lambda: {}, ROOT=".")
        self.server.search_codebase = lambda *_args, **_kwargs: [
            {"file": "server.py", "content": "def a():\n  pass", "line": 1, "score": 0.7}
        ]
        self.server.subprocess.check_output = lambda *args, **kwargs: "stub"
        self.server.url_request.urlopen = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            urllib.error.URLError("connection refused")
        )

        class _Req:
            async def json(self):
                return {
                    "messages": [{"role": "user", "content": "what is a?"}],
                    "stream": False,
                    "max_tokens": 16,
                }

        result = asyncio.run(self.server.chat(_Req()))
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "remote_unreachable")

    def test_chat_remote_stream_done_marker_split_across_reads(self):
        self.server.get_execution_mode = lambda: self.server.ExecutionMode.REMOTE_INFERENCE
        self.server.INDEXER_READY = True
        self.server._indexer_module = types.SimpleNamespace(_load_index_state=lambda: {}, ROOT=".")
        self.server.search_codebase = lambda *_args, **_kwargs: [
            {"file": "server.py", "content": "def a():\n  pass", "line": 1, "score": 0.7}
        ]
        self.server.subprocess.check_output = lambda *args, **kwargs: "stub"
        self.server.StreamingResponse = lambda gen, **_kwargs: gen

        class _Resp:
            def __init__(self):
                self._parts = [
                    b"data: token1\n\n",
                    b"data: [DO",
                    b"NE]\n\n",
                    b"",
                ]
                self._idx = 0

            def read(self, _size: int | None = None):
                part = self._parts[self._idx]
                self._idx += 1
                return part

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        self.server.url_request.urlopen = lambda *_args, **_kwargs: _Resp()

        class _Req:
            async def json(self):
                return {
                    "messages": [{"role": "user", "content": "what is a?"}],
                    "stream": True,
                    "max_tokens": 16,
                }

        stream_gen = asyncio.run(self.server.chat(_Req()))

        async def _collect():
            out = []
            async for part in stream_gen:
                out.append(part)
            return "".join(out)

        stream_payload = asyncio.run(_collect())
        self.assertIn("data: [DONE]", stream_payload)
        self.assertNotIn("remote_stream_interrupted", stream_payload)


if __name__ == "__main__":
    unittest.main()
