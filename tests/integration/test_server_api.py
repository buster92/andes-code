import json
import os
import urllib.request

import pytest

BASE_URL = os.getenv("ANDESCODE_URL", "http://localhost:8080")

pytestmark = [pytest.mark.integration, pytest.mark.server]


def _get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=10) as r:
        return json.loads(r.read())


def test_health():
    r = _get("/")
    assert r["status"] == "running"
    assert r["product"] == "AndesCode"
    assert "version" in r


def test_health_v1():
    r = _get("/v1")
    assert r["status"] == "running"


def test_models_list():
    r = _get("/v1/models")
    assert r["object"] == "list"
    assert len(r["data"]) > 0


def test_models_list_no_prefix():
    r = _get("/models")
    assert r["object"] == "list"


def test_indexer_status():
    r = _get("/")
    assert "indexer" in r


def test_cache_status():
    r = _get("/")
    assert "cache" in r
