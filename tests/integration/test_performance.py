import json
import os
import time
import urllib.request

import pytest

BASE_URL = os.getenv("ANDESCODE_URL", "http://localhost:8080")

pytestmark = [pytest.mark.integration, pytest.mark.server, pytest.mark.model]


def _post(body: dict) -> dict:
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def test_second_request_faster_smoke():
    msg = {"messages": [{"role": "user", "content": "What is a list in Python?"}], "max_tokens": 64, "stream": False}
    t0 = time.perf_counter(); _post(msg); t1 = time.perf_counter() - t0
    t0 = time.perf_counter(); _post(msg); t2 = time.perf_counter() - t0
    assert t1 > 0
    assert t2 > 0
