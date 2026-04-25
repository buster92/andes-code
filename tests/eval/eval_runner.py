"""
AndesCode Eval Runner
=====================
Stack-agnostic entrypoint for the AndesCode eval framework.
Each fixture is an independent golden codebase that tests a different
language/stack combination. The retrieval and answer-quality harnesses
are reused across all fixtures.

Registered fixtures
-------------------
  android      Kotlin · RxJava 3 · Hilt · Room · CameraX · Bluetooth LE
               The most retrieval-demanding fixture: deep RxJava chains,
               4-level dependency graphs, hardware integration.
  python_api   FastAPI · SQLAlchemy async · Celery · Redis · Pytest
               Async session management, DI chains, JWT auth, background jobs.
  rust_cli     Tokio · Serde · Clap · async traits · Cargo workspace
               Channel-based worker pool, trait objects, cross-crate dep graph.

Suites
------
  fast    Retrieval precision only — no model needed. Runs in ~5–15s.
          Use on every commit / PR.

  full    Retrieval precision + answer quality eval.
          Requires the AndesCode server to be running.
          Produces a graded per-category report.

  eval    Answer quality only — skips retrieval tests.
          Useful when you only want to measure model answer quality.

Usage
-----
  # Fast battery — no model, run on every commit
  python3 tests/eval_runner.py --suite fast

  # Full eval against the default (android) fixture
  python3 tests/eval_runner.py --suite full

  # Full eval, auto-index the fixture before running
  python3 tests/eval_runner.py --suite full --auto-index

  # Specific fixture
  python3 tests/eval_runner.py --suite full --fixture android

  # Point at a non-default server
  python3 tests/eval_runner.py --suite full --url http://localhost:9090

Adding a new fixture
--------------------
  1. Create tests/fixtures/golden_<name>.py
     Export GOLDEN_FILES: dict[str, str] and write_golden_codebase(target_dir)
     following the same pattern as golden_android.py.

  2. Register it in FIXTURES below.

  3. That's it — the runner picks it up automatically.
"""

import argparse
import os
import sys
import unittest
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

from path_setup import find_repo_root, prepend_sys_path

REPO_ROOT = find_repo_root(__file__)
prepend_sys_path(REPO_ROOT)
prepend_sys_path(EVAL_DIR)

# ── Fixture registry ──────────────────────────────────────────────────────────
# Each entry: name → (module_path, human description)
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _server_up(url: str) -> bool:
    import json, urllib.request
    try:
        with urllib.request.urlopen(f"{url}/", timeout=5) as r:
            return json.loads(r.read()).get("status") == "running"
    except Exception:
        return False


def _load_fixture(name: str):
    import importlib
    if name not in FIXTURES:
        print(f"❌ Unknown fixture '{name}'. Available: {list(FIXTURES)}")
        sys.exit(1)
    module_path, description = FIXTURES[name]
    module = importlib.import_module(module_path)
    return module, description


def _auto_index(fixture_module, tmpdir: str):
    from indexer import index_codebase
    fixture_module.write_golden_codebase(tmpdir)
    result = index_codebase(tmpdir)
    print(f"  Indexed {result['indexed']} files from golden codebase.\n")
    return result


# ── Suite runners ─────────────────────────────────────────────────────────────

def run_fast(fixture_name: str, auto_index: bool):
    """Retrieval precision only — no model needed."""
    import tempfile, shutil
    from test_retrieval_precision import (
        RetrievalBaseline, RetrievalCodeUnderstanding, RetrievalArchitecture,
        RetrievalNestedDeps, RetrievalHardware, RetrievalFramework, RetrievalReactiveX,
        GoldenBaseTest,
    )

    fixture_module, description = _load_fixture(fixture_name)
    print(f"  Fixture:  {description}")
    print(f"  Suite:    fast (retrieval precision, no model)\n")

    tmpdir = tempfile.mkdtemp(prefix="andescode_eval_")
    try:
        fixture_module.write_golden_codebase(tmpdir)
        # Patch the base class so all test classes use this tmpdir
        GoldenBaseTest._override_tmpdir = tmpdir

        loader = unittest.TestLoader()
        suite = unittest.TestSuite([
            loader.loadTestsFromTestCase(cls) for cls in [
                RetrievalBaseline, RetrievalCodeUnderstanding, RetrievalArchitecture,
                RetrievalNestedDeps, RetrievalHardware, RetrievalFramework, RetrievalReactiveX,
            ]
        ])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return result.wasSuccessful()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_eval(url: str, fixture_name: str, auto_index: bool):
    """Answer quality eval only — model required."""
    import tempfile, shutil

    fixture_module, description = _load_fixture(fixture_name)
    print(f"  Fixture:  {description}")
    print(f"  Suite:    eval (answer quality, model required)")
    print(f"  Server:   {url}\n")

    if not _server_up(url):
        print("❌ AndesCode server is not running.")
        print(f"   Start it with: python3 server.py")
        sys.exit(1)

    os.environ["ANDESCODE_URL"] = url

    tmpdir = None
    if auto_index:
        tmpdir = tempfile.mkdtemp(prefix="andescode_eval_")
        fixture_module.write_golden_codebase(tmpdir)
        os.environ["AUTO_INDEX"] = "1"
        # Patch indexer path so test_answer_eval picks it up
        os.environ["GOLDEN_FIXTURE_DIR"] = tmpdir

    from test_answer_eval import TestAnswerQuality
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestAnswerQuality)
    result = unittest.TextTestRunner(verbosity=1).run(suite)

    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return result.wasSuccessful()


def run_full(url: str, fixture_name: str, auto_index: bool):
    """Retrieval precision + answer quality."""
    print("── Phase 1: Retrieval Precision ──────────────────────────────")
    fast_ok = run_fast(fixture_name, auto_index=False)

    print("\n── Phase 2: Answer Quality ───────────────────────────────────")
    eval_ok = run_eval(url, fixture_name, auto_index)

    return fast_ok and eval_ok


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AndesCode Eval Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python3 tests/eval_runner.py --suite fast
  python3 tests/eval_runner.py --suite full --auto-index
  python3 tests/eval_runner.py --suite eval --url http://localhost:9090
  python3 tests/eval_runner.py --list-fixtures
        """
    )
    parser.add_argument(
        "--suite", choices=["fast", "eval", "full"], default="fast",
        help="fast=retrieval only (no model)  eval=answer quality  full=both"
    )
    parser.add_argument(
        "--fixture", default=DEFAULT_FIXTURE,
        help=f"Golden codebase fixture to use (default: {DEFAULT_FIXTURE})"
    )
    parser.add_argument(
        "--url", default="http://localhost:8080",
        help="AndesCode server URL (default: http://localhost:8080)"
    )
    parser.add_argument(
        "--auto-index", action="store_true",
        help="Write the golden codebase to a temp dir and index it before running"
    )
    parser.add_argument(
        "--list-fixtures", action="store_true",
        help="List available fixtures and exit"
    )
    args = parser.parse_args()

    if args.list_fixtures:
        print("\nRegistered fixtures:\n")
        for name, (_, desc) in FIXTURES.items():
            marker = "✓" if name == DEFAULT_FIXTURE else " "
            print(f"  {marker} {name:<16} {desc}")
        print()
        sys.exit(0)

    print(f"\n🏔️  AndesCode Eval Runner\n")

    dispatch = {
        "fast": lambda: run_fast(args.fixture, args.auto_index),
        "eval": lambda: run_eval(args.url, args.fixture, args.auto_index),
        "full": lambda: run_full(args.url, args.fixture, args.auto_index),
    }
    success = dispatch[args.suite]()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
