"""Shared fixture registry for AndesCode eval suites."""

from __future__ import annotations

import importlib

# name -> (module_path, human-readable description)
FIXTURES: dict[str, tuple[str, str]] = {
    "android": (
        "fixtures.golden_android",
        "SecureCam Android — Kotlin / RxJava 3 / Hilt / Room / CameraX / BLE",
    ),
    "python_api": (
        "fixtures.golden_python_api",
        "TaskFlow API — FastAPI / SQLAlchemy async / Celery / Redis / Pytest",
    ),
    "rust_cli": (
        "fixtures.golden_rust_cli",
        "Ferox CLI — Tokio / Serde / Clap / async traits / Cargo workspace",
    ),
}

DEFAULT_FIXTURE = "android"


def load_fixture(name: str):
    """Load and return (fixture_module, fixture_description)."""
    if name not in FIXTURES:
        raise ValueError(f"Unknown fixture '{name}'. Available: {list(FIXTURES)}")
    module_path, description = FIXTURES[name]
    module = importlib.import_module(module_path)
    return module, description
