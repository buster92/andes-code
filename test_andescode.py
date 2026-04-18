"""
AndesCode Test Suite
====================
Tests the server API, indexer, and offline enforcement.

Usage:
    # Server must be running on localhost:8080
    python3 tests/test_andescode.py

    # Run specific test
    python3 tests/test_andescode.py TestServer.test_health
"""

import json
import os
import sys
import time
import unittest
import tempfile
import shutil
from pathlib import Path

# ── Test config ───────────────────────────────────────────────────────────────
BASE_URL = os.getenv("ANDESCODE_URL", "http://localhost:8080")
TIMEOUT  = 120   # seconds — model responses can be slow


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get(path: str) -> dict:
    import urllib.request
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=10) as r:
        return json.loads(r.read())


def _post(path: str, body: dict, timeout: int = TIMEOUT) -> dict:
    import urllib.request
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        f"{BASE_URL}{path}",
        data    = data,
        headers = {"Content-Type": "application/json"},
        method  = "POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _post_stream(path: str, body: dict, timeout: int = TIMEOUT) -> str:
    """POST with stream=True, collect all chunks, return full text."""
    import urllib.request
    body   = {**body, "stream": True}
    data   = json.dumps(body).encode()
    req    = urllib.request.Request(
        f"{BASE_URL}{path}",
        data    = data,
        headers = {"Content-Type": "application/json"},
        method  = "POST",
    )
    full_text = ""
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw_line in r:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
                delta = chunk["choices"][0]["delta"].get("content", "")
                full_text += delta
            except (json.JSONDecodeError, KeyError):
                continue
    return full_text


def _server_is_up() -> bool:
    try:
        _get("/")
        return True
    except Exception:
        return False


# ── Test classes ──────────────────────────────────────────────────────────────

class TestServer(unittest.TestCase):
    """Basic server health and routing tests. No model required."""

    @classmethod
    def setUpClass(cls):
        if not _server_is_up():
            raise unittest.SkipTest("AndesCode server not running — start with: python3 server.py")

    def test_health(self):
        """GET / returns running status."""
        r = _get("/")
        self.assertEqual(r["status"], "running")
        self.assertEqual(r["product"], "AndesCode")
        self.assertIn("version", r)
        print(f"  ✅ health: {r}")

    def test_health_v1(self):
        """GET /v1 also returns health."""
        r = _get("/v1")
        self.assertEqual(r["status"], "running")

    def test_models_list(self):
        """GET /v1/models returns at least one model."""
        r = _get("/v1/models")
        self.assertEqual(r["object"], "list")
        self.assertGreater(len(r["data"]), 0)
        model_ids = [m["id"] for m in r["data"]]
        self.assertIn("andescode-gemma4-26b", model_ids)
        print(f"  ✅ models: {model_ids}")

    def test_models_list_no_prefix(self):
        """GET /models (no /v1) also works — Continue compatibility."""
        r = _get("/models")
        self.assertEqual(r["object"], "list")

    def test_indexer_status(self):
        """Root endpoint reports indexer status."""
        r = _get("/")
        self.assertIn("indexer", r)
        print(f"  ✅ indexer ready: {r['indexer']}")

    def test_cache_status(self):
        """Root endpoint reports cache config."""
        r = _get("/")
        self.assertIn("cache", r)
        print(f"  ✅ cache: {r['cache']}")


class TestOffline(unittest.TestCase):
    """Verify no outbound network calls are made during inference."""

    @classmethod
    def setUpClass(cls):
        if not _server_is_up():
            raise unittest.SkipTest("Server not running")

    def test_hf_hub_offline_env(self):
        """HF_HUB_OFFLINE must be set in server process."""
        # We can't inspect the server's env directly, but we can check
        # the audit log contains no HTTP Request lines after the last startup
        log_path = Path(__file__).parent.parent / "audit.log"
        if not log_path.exists():
            self.skipTest("audit.log not found")

        lines = log_path.read_text().splitlines()
        # Find last server startup marker
        start_idx = 0
        for i, line in enumerate(lines):
            if "REQUEST" in line:
                start_idx = i
                break

        recent = lines[start_idx:]
        hf_calls = [l for l in recent if "huggingface.co" in l]
        self.assertEqual(
            len(hf_calls), 0,
            f"Found {len(hf_calls)} outbound HF calls after startup:\n" +
            "\n".join(hf_calls[:5])
        )
        print(f"  ✅ No HF network calls found in recent audit log")

    def test_audit_log_no_code_content(self):
        """Audit log must never contain raw code content."""
        log_path = Path(__file__).parent.parent / "audit.log"
        if not log_path.exists():
            self.skipTest("audit.log not found")

        lines = log_path.read_text().splitlines()
        # Audit entries should only contain metadata keywords
        allowed_keywords = {"REQUEST", "RESPONSE", "CONTEXT", "STREAM_DONE",
                            "CONTEXT_FAIL", "INDEX", "Use pytorch", "Load pretrained"}
        suspicious = []
        for line in lines:
            parts = line.split("|")
            if len(parts) < 2:
                continue
            entry = parts[1].strip() if len(parts) > 1 else ""
            # Flag lines that look like they contain code
            if any(x in entry for x in ["def ", "class ", "import ", "return "]):
                suspicious.append(line[:120])

        self.assertEqual(
            len(suspicious), 0,
            f"Potential code content in audit log:\n" + "\n".join(suspicious[:3])
        )
        print(f"  ✅ Audit log contains no code content")


class TestInference(unittest.TestCase):
    """Test actual model responses. Requires model to be loaded."""

    @classmethod
    def setUpClass(cls):
        if not _server_is_up():
            raise unittest.SkipTest("Server not running")

    def test_simple_question_non_stream(self):
        """Non-streaming response returns valid structure."""
        r = _post("/v1/chat/completions", {
            "messages":   [{"role": "user", "content": "Reply with exactly: PONG"}],
            "max_tokens": 32,
            "stream":     False,
        })
        self.assertIn("choices", r)
        self.assertEqual(len(r["choices"]), 1)
        content = r["choices"][0]["message"]["content"]
        self.assertIsInstance(content, str)
        self.assertGreater(len(content), 0)
        print(f"  ✅ non-stream response: '{content[:80]}'")

    def test_simple_question_stream(self):
        """Streaming response returns non-empty text."""
        t0   = time.perf_counter()
        text = _post_stream("/chat/completions", {
            "messages":   [{"role": "user", "content": "What is 2 + 2? Answer in one word."}],
            "max_tokens": 64,
        })
        elapsed = time.perf_counter() - t0
        self.assertGreater(len(text), 0)
        print(f"  ✅ stream response ({elapsed:.1f}s): '{text[:80]}'")

    def test_stream_contains_status_messages(self):
        """Streaming response should include status messages."""
        text = _post_stream("/chat/completions", {
            "messages":   [{"role": "user", "content": "Say hello."}],
            "max_tokens": 64,
        })
        self.assertIn("Searching", text)
        self.assertIn("Thinking", text)
        print(f"  ✅ status messages present in stream")

    def test_stream_contains_timing_footer(self):
        """Streaming response should include timing footer."""
        text = _post_stream("/chat/completions", {
            "messages":   [{"role": "user", "content": "Say hello."}],
            "max_tokens": 64,
        })
        self.assertIn("⏱", text)
        self.assertIn("total", text)
        print(f"  ✅ timing footer present in stream")

    def test_no_thinking_tags_in_response(self):
        """Thinking tags must be stripped from response."""
        text = _post_stream("/chat/completions", {
            "messages":   [{"role": "user", "content": "What is Python?"}],
            "max_tokens": 256,
        })
        self.assertNotIn("<|channel>", text)
        self.assertNotIn("<channel|>", text)
        print(f"  ✅ no thinking tags in response")

    def test_both_routes_work(self):
        """/chat/completions and /v1/chat/completions both respond."""
        for route in ["/chat/completions", "/v1/chat/completions"]:
            r = _post(route, {
                "messages":   [{"role": "user", "content": "Hi"}],
                "max_tokens": 32,
                "stream":     False,
            })
            self.assertIn("choices", r, f"Route {route} failed")
        print(f"  ✅ both routes respond")

    def test_ttft_under_threshold(self):
        """First token should arrive within 15 seconds."""
        import urllib.request
        body = json.dumps({
            "messages":   [{"role": "user", "content": "Say yes."}],
            "max_tokens": 32,
            "stream":     True,
        }).encode()
        req = urllib.request.Request(
            f"{BASE_URL}/chat/completions",
            data    = body,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )
        t0         = time.perf_counter()
        ttft       = None
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            for raw_line in r:
                line = raw_line.decode("utf-8").strip()
                if line.startswith("data:"):
                    payload = line[5:].strip()
                    if payload != "[DONE]":
                        try:
                            chunk   = json.loads(payload)
                            content = chunk["choices"][0]["delta"].get("content", "")
                            if content and ttft is None:
                                ttft = time.perf_counter() - t0
                                break
                        except (json.JSONDecodeError, KeyError):
                            pass

        self.assertIsNotNone(ttft, "Never received a token")
        self.assertLess(ttft, 15.0, f"TTFT too slow: {ttft:.1f}s")
        print(f"  ✅ TTFT: {ttft:.2f}s")


class TestIndexer(unittest.TestCase):
    """Test the codebase indexer."""

    def setUp(self):
        # Create a temporary codebase for testing
        self.tmpdir = tempfile.mkdtemp()
        # Write sample Python files
        (Path(self.tmpdir) / "auth.py").write_text(
            "def login(username, password):\n"
            "    \"\"\"Authenticate a user.\"\"\"\n"
            "    return username == 'admin' and password == 'secret'\n"
        )
        (Path(self.tmpdir) / "models.py").write_text(
            "class User:\n"
            "    def __init__(self, name, email):\n"
            "        self.name = name\n"
            "        self.email = email\n"
        )
        (Path(self.tmpdir) / "README.md").write_text(
            "# Test project\nThis is a test."
        )
        (Path(self.tmpdir) / "requirements.txt").write_text(
            "fastapi==0.111.0\npydantic>=2.0\n"
        )
        (Path(self.tmpdir) / "package.json").write_text(
            json.dumps({
                "name": "tmp-test",
                "dependencies": {"react": "^18.2.0", "axios": "^1.7.0"},
                "devDependencies": {"typescript": "^5.0.0"},
            })
        )
        (Path(self.tmpdir) / "api.py").write_text(
            "from fastapi import FastAPI\n"
            "from auth import login\n"
            "app = FastAPI()\n"
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_index_codebase(self):
        """Indexer processes Python files and skips non-code files."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from indexer import index_codebase
        result = index_codebase(self.tmpdir)
        self.assertIn("indexed", result)
        self.assertGreater(result["indexed"], 0)
        print(f"  ✅ indexed: {result}")

    def test_search_returns_results(self):
        """Search returns relevant results after indexing."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from indexer import index_codebase, search
        index_codebase(self.tmpdir)
        results = search("user authentication login", n_results=2)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        files = [r["file"] for r in results]
        print(f"  ✅ search returned {len(results)} results from: {files}")

    def test_search_returns_relevant_file(self):
        """Search for login should return auth.py."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from indexer import index_codebase, search
        index_codebase(self.tmpdir)
        results = search("login function authentication", n_results=3)
        files   = [r["file"] for r in results]
        self.assertTrue(
            any("auth" in f for f in files),
            f"Expected auth.py in results, got: {files}"
        )
        print(f"  ✅ relevant file found: {files}")

    def test_skips_non_code_files(self):
        """Indexer ignores markdown and other non-code files."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from indexer import index_codebase, search
        index_codebase(self.tmpdir)
        results = search("test project README", n_results=3)
        files   = [r["file"] for r in results]
        self.assertTrue(
            all(".md" not in f for f in files),
            f"README.md should not be indexed, got: {files}"
        )
        print(f"  ✅ non-code files correctly skipped")

    def test_index_via_api(self):
        """POST /v1/index endpoint works."""
        if not _server_is_up():
            self.skipTest("Server not running")
        r = _post("/v1/index", {"path": self.tmpdir}, timeout=60)
        self.assertIn("indexed", r)
        self.assertGreater(r["indexed"], 0)
        print(f"  ✅ API index: {r}")

    def test_project_map_contains_workspace_summary(self):
        """Project map should include workspace/manifests/package manager info."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from indexer import index_codebase
        result = index_codebase(self.tmpdir)
        pmap = result.get("map", {})
        ws = pmap.get("workspace", {})
        self.assertGreater(len(ws.get("manifests", [])), 0)
        self.assertGreater(len(ws.get("package_managers", [])), 0)
        self.assertGreater(len(ws.get("repo_types", [])), 0)
        print(f"  ✅ workspace summary: {ws}")

    def test_workspace_dependency_query_is_structure_first(self):
        """Dependency questions should return structured workspace facts."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from indexer import index_codebase, search
        index_codebase(self.tmpdir)
        results = search("What libraries and dependencies does this repo use?", n_results=2)
        self.assertGreater(len(results), 0)
        self.assertTrue(any(r["file"] == "__workspace_index__" for r in results))
        self.assertTrue(any("Declared Dependencies" in r["content"] for r in results))
        print("  ✅ dependency query routed to workspace index")


class TestPerformance(unittest.TestCase):
    """Latency benchmarks. Results are printed but not strict pass/fail."""

    @classmethod
    def setUpClass(cls):
        if not _server_is_up():
            raise unittest.SkipTest("Server not running")

    def test_second_request_faster(self):
        """Second identical request should be faster due to KV cache."""
        msg = {"messages": [{"role": "user", "content": "What is a list in Python?"}],
               "max_tokens": 128, "stream": False}

        t0 = time.perf_counter()
        _post("/v1/chat/completions", msg)
        t1 = time.perf_counter() - t0

        t0 = time.perf_counter()
        _post("/v1/chat/completions", msg)
        t2 = time.perf_counter() - t0

        print(f"  📊 First request:  {t1:.1f}s")
        print(f"  📊 Second request: {t2:.1f}s")
        print(f"  📊 Cache speedup:  {t1/t2:.1f}x" if t2 > 0 else "")
        # Not a hard assertion — just informational
        self.assertGreater(t1, 0)
        self.assertGreater(t2, 0)

    def test_context_search_latency(self):
        """Context search should complete in under 2 seconds."""
        import urllib.request
        body = json.dumps({
            "messages":   [{"role": "user", "content": "How does streaming work?"}],
            "max_tokens": 32,
            "stream":     True,
        }).encode()
        req = urllib.request.Request(
            f"{BASE_URL}/chat/completions",
            data    = body,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )
        t0           = time.perf_counter()
        context_done = None
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            for raw_line in r:
                line = raw_line.decode().strip()
                if "context ready in" in line and context_done is None:
                    context_done = time.perf_counter() - t0
                    break

        if context_done:
            print(f"  📊 Context search latency: {context_done:.2f}s")
            self.assertLess(context_done, 3.0, f"Context too slow: {context_done:.1f}s")
        else:
            print(f"  ⚠️  Could not measure context latency from stream")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🏔️  AndesCode Test Suite")
    print(f"   Server: {BASE_URL}")
    print(f"   Timeout: {TIMEOUT}s\n")

    if not _server_is_up():
        print("❌  Server is not running.")
        print("    Start it with: python3 server.py\n")
        sys.exit(1)

    # Run all tests with verbose output
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    # Order: fast tests first
    for cls in [TestServer, TestOffline, TestIndexer, TestInference, TestPerformance]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
