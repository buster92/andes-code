"""Model-free A/B eval for AndesCode graph-aware hybrid retrieval.

Runs the same retrieval precision cases twice against a golden fixture:

* baseline retrieval: ``ANDESCODE_HYBRID_RETRIEVAL=0``
* hybrid retrieval: ``ANDESCODE_HYBRID_RETRIEVAL=1``

The eval measures retrieval quality, not answer quality. It emits pilot-friendly
JSON and Markdown reports under ``tests/eval/reports/`` by default.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

from fixture_registry import DEFAULT_FIXTURE, FIXTURES, load_fixture
from path_setup import find_repo_root, prepend_sys_path

REPO_ROOT = find_repo_root(__file__)
prepend_sys_path(REPO_ROOT)
prepend_sys_path(EVAL_DIR)

DEFAULT_REPORT_DIR = EVAL_DIR / "reports"

HYBRID_DEBUG_DEFAULTS: dict[str, Any] = {
    "retrieval_routes_used": [],
    "graph_route_by_file": {},
    "graph_score_by_file": {},
    "graph_expansion_limits": {},
    "graph_seed_files": [],
    "context_sufficiency_notes": [],
    "graph_neighbors_added": [],
    "graph_boosted_existing_files": [],
}


@dataclass(frozen=True)
class RetrievalCase:
    """A single model-free retrieval precision case."""

    fixture: str
    test_id: str
    query: str
    expected_files: list[str]
    n_results: int = 8
    context_sufficiency_notes: str = ""


# These cases are deliberately hard multi-file/codebase-understanding queries,
# not answer-quality prompts. They keep the model-free A/B eval focused on
# retrieval evidence while stressing graph expansion paths that hybrid retrieval
# is meant to improve.
EVAL_CASES: list[RetrievalCase] = [
    RetrievalCase(
        fixture="android",
        test_id="android/upload-usecase-chain",
        query="what does UploadVideoUseCase depend on VideoRepository VideoDao VideoApiService GpsTracker",
        expected_files=["UploadVideoUseCase.kt", "VideoRepository.kt"],
        context_sufficiency_notes="Good context includes the use case and repository; DAO/API/GPS neighbors are useful supporting evidence.",
    ),
    RetrievalCase(
        fixture="android",
        test_id="android/auth-token-interceptor-chain",
        query="auth token read UserRepository injected OkHttp interceptor API request Authorization Bearer",
        expected_files=["NetworkModule.kt"],
        context_sufficiency_notes="Good context links token storage/repository code with the OkHttp Authorization interceptor wiring.",
    ),
    RetrievalCase(
        fixture="android",
        test_id="android/gps-video-data-flow",
        query="GPS coordinates latitude longitude stored video metadata saveRecording GpsTracker Video entity",
        expected_files=["VideoRepository.kt", "Video.kt"],
        context_sufficiency_notes="Good context traces GPS capture into persisted latitude/longitude fields on the video model.",
    ),
    RetrievalCase(
        fixture="android",
        test_id="android/rx-sync-retry-flow",
        query="SyncVideosUseCase retries pending failed uploads retryWhen RetryWithDelay VideoRepository concatMapSingle",
        expected_files=["SyncVideosUseCase.kt", "VideoRepository.kt"],
        context_sufficiency_notes="Good context includes sequential sync orchestration and retry behavior used by repository uploads.",
    ),
    RetrievalCase(
        fixture="python_api",
        test_id="python_api/auth-token-flow",
        query="AuthService create_access_token decode_token verify_password bcrypt jose jwt dependency user routes",
        expected_files=["app/services/auth_service.py"],
        context_sufficiency_notes="Good context identifies token creation/verification plus the API route or dependency that consumes it.",
    ),
    RetrievalCase(
        fixture="python_api",
        test_id="python_api/task-route-service-db-flow",
        query="FastAPI task routes create list endpoint TaskService repository async SQLAlchemy session",
        expected_files=["app/api/routes/tasks.py", "app/services/task_service.py"],
        context_sufficiency_notes="Good context follows request handling from route to service and persistence/session code.",
    ),
    RetrievalCase(
        fixture="python_api",
        test_id="python_api/celery-worker-flow",
        query="Celery background jobs worker queue retries task processing redis broker",
        expected_files=["app/workers/tasks.py"],
        context_sufficiency_notes="Good context shows the worker entry points and retry/queue configuration.",
    ),
    RetrievalCase(
        fixture="rust_cli",
        test_id="rust_cli/pipeline-source-discovery",
        query="DirectorySource WalkDir spawn_blocking include_extensions max_file_size_bytes FileSource discover pipeline",
        expected_files=["ferox-core/src/pipeline.rs"],
        context_sufficiency_notes="Good context identifies source discovery and async/blocking boundary files.",
    ),
    RetrievalCase(
        fixture="rust_cli",
        test_id="rust_cli/scheduler-channel-backpressure",
        query="worker pool scheduler channel dispatch backpressure tokio mpsc WorkerMessage",
        expected_files=["ferox-worker/src/scheduler.rs", "ferox-worker/src/channel.rs"],
        context_sufficiency_notes="Good context shows scheduler orchestration and channel/backpressure behavior together.",
    ),
    RetrievalCase(
        fixture="rust_cli",
        test_id="rust_cli/cli-to-core-flow",
        query="clap command line args parse subcommands run pipeline config ferox cli main",
        expected_files=["ferox-cli/src/main.rs"],
        context_sufficiency_notes="Good context starts at CLI parsing and includes the files it calls for execution/configuration.",
    ),
]


def _cases_for_fixture(fixture: str) -> list[RetrievalCase]:
    return [case for case in EVAL_CASES if case.fixture == fixture]


def _matches_expected(retrieved_files: list[str], expected: str) -> bool:
    return any(expected in file_path for file_path in retrieved_files)


def _passes_expected_files(retrieved_files: list[str], expected_files: list[str]) -> bool:
    return all(_matches_expected(retrieved_files, expected) for expected in expected_files)


def _unique_files(results: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    files: list[str] = []
    for result in results:
        file_path = str(result.get("file", "") or "")
        if file_path and file_path not in seen:
            seen.add(file_path)
            files.append(file_path)
    return files


def _copy_default(value: Any) -> Any:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return list(value)
    return value


def _extract_hybrid_debug(retrieval_debug: dict[str, Any]) -> dict[str, Any]:
    """Return the graph-aware debug fields used by JSON and Markdown reports."""
    return {
        field: retrieval_debug.get(field, _copy_default(default))
        for field, default in HYBRID_DEBUG_DEFAULTS.items()
    }


def _run_retrieval_case(case: RetrievalCase, hybrid_enabled: bool) -> dict[str, Any]:
    from indexer import search

    os.environ["ANDESCODE_HYBRID_RETRIEVAL"] = "1" if hybrid_enabled else "0"
    results, debug = search(
        case.query,
        n_results=case.n_results,
        debug_mode=True,
        return_debug=True,
    )
    retrieved_files = _unique_files(results)
    retrieval_debug = (debug or {}).get("retrieval", {}) if isinstance(debug, dict) else {}
    return {
        "retrieved_files": retrieved_files,
        "pass": _passes_expected_files(retrieved_files, case.expected_files),
        "debug": _extract_hybrid_debug(retrieval_debug),
    }


def _classification(baseline_pass: bool, hybrid_pass: bool) -> str:
    if not baseline_pass and hybrid_pass:
        return "improvement"
    if baseline_pass and not hybrid_pass:
        return "regression"
    return "unchanged"


def _case_report(case: RetrievalCase) -> dict[str, Any]:
    baseline = _run_retrieval_case(case, hybrid_enabled=False)
    hybrid = _run_retrieval_case(case, hybrid_enabled=True)
    baseline_files = baseline["retrieved_files"]
    hybrid_files = hybrid["retrieved_files"]
    files_added = [file_path for file_path in hybrid_files if file_path not in baseline_files]
    files_removed = [file_path for file_path in baseline_files if file_path not in hybrid_files]
    noisy_additions = [
        file_path
        for file_path in files_added
        if not any(expected in file_path for expected in case.expected_files)
    ]

    return {
        "fixture": case.fixture,
        "query_test_id": case.test_id,
        "query": case.query,
        "n_results": case.n_results,
        "expected_files": case.expected_files,
        "baseline_retrieved_files": baseline_files,
        "hybrid_retrieved_files": hybrid_files,
        "files_added_by_hybrid": files_added,
        "files_removed_by_hybrid": files_removed,
        "baseline_pass": baseline["pass"],
        "hybrid_pass": hybrid["pass"],
        "classification": _classification(baseline["pass"], hybrid["pass"]),
        "graph_routes_used": hybrid["debug"].get("retrieval_routes_used", []),
        "graph_route_by_file": hybrid["debug"].get("graph_route_by_file", {}),
        "graph_score_by_file": hybrid["debug"].get("graph_score_by_file", {}),
        "graph_expansion_limits": hybrid["debug"].get("graph_expansion_limits", {}),
        "graph_seed_files": hybrid["debug"].get("graph_seed_files", []),
        "graph_neighbors_added": hybrid["debug"].get("graph_neighbors_added", []),
        "graph_boosted_existing_files": hybrid["debug"].get("graph_boosted_existing_files", []),
        "context_sufficiency_notes": [
            case.context_sufficiency_notes,
            *hybrid["debug"].get("context_sufficiency_notes", []),
        ],
        "noisy_additions": noisy_additions,
    }


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_pass_count = sum(1 for case in cases if case["baseline_pass"])
    hybrid_pass_count = sum(1 for case in cases if case["hybrid_pass"])
    improvements = sum(1 for case in cases if case["classification"] == "improvement")
    regressions = sum(1 for case in cases if case["classification"] == "regression")
    unchanged = sum(1 for case in cases if case["classification"] == "unchanged")
    noisy_additions_count = sum(len(case.get("noisy_additions", [])) for case in cases)
    return {
        "total_cases": len(cases),
        "baseline_pass_count": baseline_pass_count,
        "hybrid_pass_count": hybrid_pass_count,
        "net_improvements": improvements,
        "net_regressions": regressions,
        "unchanged_cases": unchanged,
        "noisy_additions_count": noisy_additions_count,
    }


def _write_json(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _md_list(values: list[Any]) -> str:
    if not values:
        return "_(none)_"
    return "<br>".join(f"`{value}`" for value in values)


def _md_mapping(mapping: dict[str, Any]) -> list[str]:
    if not mapping:
        return ["  - _(none)_"]
    return [f"  - `{key}` → `{value}`" for key, value in sorted(mapping.items())]


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = report["summary"]
    lines = [
        "# Hybrid Retrieval A/B Evaluation Report",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Fixture: `{report['fixture']}`",
        f"Fixture description: {report['fixture_description']}",
        "",
        "## Scoring Summary",
        "",
        f"- Baseline pass count: **{summary['baseline_pass_count']} / {summary['total_cases']}**",
        f"- Hybrid pass count: **{summary['hybrid_pass_count']} / {summary['total_cases']}**",
        f"- Net improvements: **{summary['net_improvements']}**",
        f"- Net regressions: **{summary['net_regressions']}**",
        f"- Unchanged cases: **{summary['unchanged_cases']}**",
        f"- Noisy additions count: **{summary['noisy_additions_count']}**",
        "",
        "## Case Comparison",
        "",
        "| Fixture | Query/test id | Expected files | Baseline retrieved files | Hybrid retrieved files | Added by hybrid | Removed by hybrid | Baseline | Hybrid | Class | Graph routes |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for case in report["cases"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{case['fixture']}`",
                    f"`{case['query_test_id']}`",
                    _md_list(case["expected_files"]),
                    _md_list(case["baseline_retrieved_files"]),
                    _md_list(case["hybrid_retrieved_files"]),
                    _md_list(case["files_added_by_hybrid"]),
                    _md_list(case["files_removed_by_hybrid"]),
                    "PASS" if case["baseline_pass"] else "FAIL",
                    "PASS" if case["hybrid_pass"] else "FAIL",
                    f"**{case['classification']}**",
                    _md_list(case["graph_routes_used"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Graph Debug Evidence", ""])
    for case in report["cases"]:
        lines.extend(
            [
                f"### `{case['query_test_id']}`",
                "",
                f"- Query: `{case['query']}`",
                f"- Graph seed files: {_md_list(case['graph_seed_files'])}",
                f"- Graph neighbors added: {_md_list(case['graph_neighbors_added'])}",
                f"- Graph boosted existing files: {_md_list(case['graph_boosted_existing_files'])}",
                "- Graph route by file:",
            ]
        )
        lines.extend(_md_mapping(case["graph_route_by_file"]))
        lines.append("- Graph score by file:")
        lines.extend(_md_mapping(case["graph_score_by_file"]))
        lines.append("- Context sufficiency notes:")
        for note in case["context_sufficiency_notes"]:
            lines.append(f"  - {note}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_fixture(fixture_name: str, report_dir: Path) -> tuple[dict[str, Any], Path, Path]:
    try:
        from indexer import index_codebase
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown dependency"
        raise RuntimeError(
            f"Hybrid retrieval A/B eval requires indexer dependencies; missing '{missing}'. "
            "Install the full retrieval dependencies before running this eval."
        ) from exc

    fixture_module, description = load_fixture(fixture_name)
    cases = _cases_for_fixture(fixture_name)
    if not cases:
        raise ValueError(f"No A/B retrieval cases registered for fixture '{fixture_name}'.")

    tmpdir = tempfile.mkdtemp(prefix=f"andescode_hybrid_ab_{fixture_name}_")
    old_fixture = os.environ.get("FIXTURE")
    old_fixture_dir = os.environ.get("GOLDEN_FIXTURE_DIR")
    old_hybrid = os.environ.get("ANDESCODE_HYBRID_RETRIEVAL")
    try:
        fixture_module.write_golden_codebase(tmpdir)
        os.environ["FIXTURE"] = fixture_name
        os.environ["GOLDEN_FIXTURE_DIR"] = tmpdir
        indexed = index_codebase(tmpdir)
        if indexed.get("indexed", 0) <= 0:
            raise RuntimeError(f"Fixture '{fixture_name}' indexed no files from {tmpdir}")
        case_reports = [_case_report(case) for case in cases]
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "fixture": fixture_name,
            "fixture_description": description,
            "n_results_policy": (
                "Each case fixes n_results and runs the same value for "
                "baseline retrieval and hybrid retrieval."
            ),
            "modes": {
                "baseline": {"ANDESCODE_HYBRID_RETRIEVAL": "0"},
                "hybrid": {"ANDESCODE_HYBRID_RETRIEVAL": "1"},
            },
            "summary": _summary(case_reports),
            "cases": case_reports,
        }
    finally:
        if old_fixture is None:
            os.environ.pop("FIXTURE", None)
        else:
            os.environ["FIXTURE"] = old_fixture
        if old_fixture_dir is None:
            os.environ.pop("GOLDEN_FIXTURE_DIR", None)
        else:
            os.environ["GOLDEN_FIXTURE_DIR"] = old_fixture_dir
        if old_hybrid is None:
            os.environ.pop("ANDESCODE_HYBRID_RETRIEVAL", None)
        else:
            os.environ["ANDESCODE_HYBRID_RETRIEVAL"] = old_hybrid
        shutil.rmtree(tmpdir, ignore_errors=True)

    stem = f"hybrid-retrieval-ab-{fixture_name}"
    json_path = report_dir / f"{stem}.json"
    md_path = report_dir / f"{stem}.md"
    _write_json(report, json_path)
    _write_markdown(report, md_path)
    return report, json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the model-free A/B eval for baseline retrieval vs hybrid retrieval."
    )
    parser.add_argument(
        "--fixture",
        default=DEFAULT_FIXTURE,
        choices=[*FIXTURES.keys(), "all"],
        help="Golden fixture to evaluate, or 'all'.",
    )
    parser.add_argument(
        "--report-dir",
        default=str(DEFAULT_REPORT_DIR),
        help="Directory for JSON and Markdown reports.",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero when any case regresses. Reports are always written first.",
    )
    args = parser.parse_args(argv)

    fixtures = list(FIXTURES) if args.fixture == "all" else [args.fixture]
    report_dir = Path(args.report_dir)
    any_regression = False
    for fixture_name in fixtures:
        print(f"🏔️  AndesCode model-free A/B eval — fixture: {fixture_name}")
        try:
            report, json_path, md_path = run_fixture(fixture_name, report_dir)
        except RuntimeError as exc:
            print(f"❌ {exc}")
            return 1
        summary = report["summary"]
        print(f"  Baseline: {summary['baseline_pass_count']}/{summary['total_cases']} pass")
        print(f"  Hybrid:   {summary['hybrid_pass_count']}/{summary['total_cases']} pass")
        print(f"  Improvements={summary['net_improvements']} Regressions={summary['net_regressions']} Unchanged={summary['unchanged_cases']}")
        print(f"  JSON:     {json_path}")
        print(f"  Markdown: {md_path}\n")
        any_regression = any_regression or summary["net_regressions"] > 0
    return 1 if args.fail_on_regression and any_regression else 0


if __name__ == "__main__":
    raise SystemExit(main())
