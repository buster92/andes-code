from __future__ import annotations

import os
from pathlib import Path

import pytest


def _flag_enabled(name: str) -> bool:
    return os.getenv(name, "0") == "1"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        path = Path(str(item.fspath)).as_posix()
        if "/tests/unit/" in path:
            item.add_marker(pytest.mark.unit)
        if "/tests/integration/" in path:
            item.add_marker(pytest.mark.integration)
        if "/tests/eval/" in path:
            item.add_marker(pytest.mark.eval)

        if item.get_closest_marker("integration") and not _flag_enabled("ANDESCODE_RUN_INTEGRATION_TESTS"):
            item.add_marker(pytest.mark.skip(reason="Set ANDESCODE_RUN_INTEGRATION_TESTS=1 to run integration tests"))
        if item.get_closest_marker("model") and not _flag_enabled("ANDESCODE_RUN_MODEL_TESTS"):
            item.add_marker(pytest.mark.skip(reason="Set ANDESCODE_RUN_MODEL_TESTS=1 to run model tests"))
        if item.get_closest_marker("server") and not _flag_enabled("ANDESCODE_RUN_SERVER_TESTS"):
            item.add_marker(pytest.mark.skip(reason="Set ANDESCODE_RUN_SERVER_TESTS=1 to run server tests"))
        if item.get_closest_marker("eval") and not _flag_enabled("ANDESCODE_RUN_EVAL_TESTS"):
            item.add_marker(pytest.mark.skip(reason="Set ANDESCODE_RUN_EVAL_TESTS=1 to run eval tests"))
