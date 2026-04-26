import json
import os
import time
import urllib.request

import pytest

BASE_URL = os.getenv("ANDESCODE_URL", "http://localhost:8080")
TIMEOUT = 120

pytestmark = [pytest.mark.integration, pytest.mark.server, pytest.mark.model]


def _post(path: str, body: dict, timeout: int = TIMEOUT) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _post_stream(path: str, body: dict, timeout: int = TIMEOUT) -> str:
    body = {**body, "stream": True}
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
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
                full_text += chunk["choices"][0]["delta"].get("content", "")
            except (json.JSONDecodeError, KeyError):
                continue
    return full_text


def test_simple_question_non_stream():
    r = _post("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Reply with exactly: PONG"}],
        "max_tokens": 32,
        "stream": False,
    })
    assert "choices" in r
    assert len(r["choices"]) == 1


def test_simple_question_stream():
    t0 = time.perf_counter()
    text = _post_stream("/chat/completions", {
        "messages": [{"role": "user", "content": "What is 2 + 2? Answer in one word."}],
        "max_tokens": 64,
    })
    assert len(text) > 0
    assert time.perf_counter() - t0 < TIMEOUT
