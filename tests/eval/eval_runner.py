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
          Still requires index/retrieval embeddings.
          Use on every commit / PR.

  full    Retrieval precision + answer quality eval.
          Requires the AndesCode server to be running.
          Produces a graded per-category report.

  eval    Answer quality only — skips retrieval tests.
          Currently Android-only (`--fixture android`).
          Useful when you only want to measure model answer quality.

  hybrid-ab
          Model-free A/B comparison: baseline retrieval vs graph-aware hybrid
          retrieval. Emits JSON and Markdown reports under tests/eval/reports/.

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

  # Hybrid retrieval A/B report (no model)
  python3 tests/eval_runner.py --suite hybrid-ab --fixture android

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
from fixture_registry import DEFAULT_FIXTURE, FIXTURES, load_fixture

REPO_ROOT = find_repo_root(__file__)
prepend_sys_path(REPO_ROOT)
prepend_sys_path(EVAL_DIR)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _server_up(url: str) -> bool:
    import json, urllib.request
    try:
        with urllib.request.urlopen(f"{url}/", timeout=5) as r:
            return json.loads(r.read()).get("status") == "running"
    except Exception:
        return False


# ── Suite runners ─────────────────────────────────────────────────────────────

def run_fast(fixture_name: str, auto_index: bool):
    """Retrieval precision only — no model needed."""
    import tempfile, shutil, os
    from test_retrieval_precision import (
        RetrievalFixtureSmoke,
        RetrievalBaseline, RetrievalCodeUnderstanding, RetrievalArchitecture,
        RetrievalNestedDeps, RetrievalHardware, RetrievalFramework, RetrievalReactiveX,
        GoldenBaseTest,
    )

    fixture_module, description = load_fixture(fixture_name)
    print(f"  Fixture:  {description}")
    print(f"  Suite:    fast (retrieval precision, no model)\n")

    tmpdir = tempfile.mkdtemp(prefix="andescode_eval_")
    try:
        fixture_module.write_golden_codebase(tmpdir)
        os.environ["FIXTURE"] = fixture_name
        os.environ["GOLDEN_FIXTURE_DIR"] = tmpdir
        GoldenBaseTest._reset_shared_state()

        loader = unittest.TestLoader()
        suite = unittest.TestSuite([
            loader.loadTestsFromTestCase(cls) for cls in [
                RetrievalFixtureSmoke,
                RetrievalBaseline, RetrievalCodeUnderstanding, RetrievalArchitecture,
                RetrievalNestedDeps, RetrievalHardware, RetrievalFramework, RetrievalReactiveX,
            ]
        ])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return result.wasSuccessful()
    finally:
        os.environ.pop("GOLDEN_FIXTURE_DIR", None)
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_eval(url: str, fixture_name: str, auto_index: bool):
    """Answer quality eval only — model required."""
    import tempfile, shutil

    fixture_module, description = load_fixture(fixture_name)
    print(f"  Fixture:  {description}")
    print(f"  Suite:    eval (answer quality, model required)")
    print(f"  Server:   {url}\n")

    if not _server_up(url):
        print("❌ AndesCode server is not running.")
        print(f"   Start it with: python3 server.py")
        sys.exit(1)

    os.environ["ANDESCODE_URL"] = url

    tmpdir = None
    os.environ["FIXTURE"] = fixture_name
    if auto_index:
        tmpdir = tempfile.mkdtemp(prefix="andescode_eval_")
        fixture_module.write_golden_codebase(tmpdir)
        os.environ["AUTO_INDEX"] = "1"
        os.environ["GOLDEN_FIXTURE_DIR"] = tmpdir
    else:
        os.environ.pop("AUTO_INDEX", None)
        os.environ.pop("GOLDEN_FIXTURE_DIR", None)

    from test_answer_eval import TestAnswerQuality
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestAnswerQuality)
    result = unittest.TextTestRunner(verbosity=1).run(suite)

    if tmpdir:
        os.environ.pop("GOLDEN_FIXTURE_DIR", None)
        shutil.rmtree(tmpdir, ignore_errors=True)

    return result.wasSuccessful()


def run_full(url: str, fixture_name: str, auto_index: bool):
    """Retrieval precision + answer quality."""
    print("── Phase 1: Retrieval Precision ──────────────────────────────")
    fast_ok = run_fast(fixture_name, auto_index=False)

    print("\n── Phase 2: Answer Quality ───────────────────────────────────")
    eval_ok = run_eval(url, fixture_name, auto_index)

    return fast_ok and eval_ok


def run_hybrid_ab(fixture_name: str):
    """Baseline vs hybrid retrieval comparison — no model needed."""
    from hybrid_retrieval_ab_eval import DEFAULT_REPORT_DIR, run_fixture

    _, description = load_fixture(fixture_name)
    print(f"  Fixture:  {description}")
    print("  Suite:    hybrid-ab (baseline vs hybrid retrieval, no model)\n")
    try:
        report, json_path, md_path = run_fixture(fixture_name, DEFAULT_REPORT_DIR)
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return False
    summary = report["summary"]
    print(f"  Baseline: {summary['baseline_pass_count']}/{summary['total_cases']} pass")
    print(f"  Hybrid:   {summary['hybrid_pass_count']}/{summary['total_cases']} pass")
    print(f"  Improvements={summary['net_improvements']} Regressions={summary['net_regressions']} Unchanged={summary['unchanged_cases']}")
    print(f"  JSON:     {json_path}")
    print(f"  Markdown: {md_path}")
    return True


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
  python3 tests/eval_runner.py --suite hybrid-ab --fixture android
  python3 tests/eval_runner.py --list-fixtures
        """
    )
    parser.add_argument(
        "--suite", choices=["fast", "eval", "full", "hybrid-ab"], default="fast",
        help="fast=retrieval only (no model)  hybrid-ab=baseline vs hybrid report  eval=answer quality  full=both"
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

    if args.fixture not in FIXTURES:
        print(f"❌ Unknown fixture '{args.fixture}'. Available: {list(FIXTURES)}")
        sys.exit(1)

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
        "hybrid-ab": lambda: run_hybrid_ab(args.fixture),
    }
    success = dispatch[args.suite]()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
