"""Structured deterministic debug payload helpers for AndesCode retrieval."""

from __future__ import annotations

import os
from copy import deepcopy

from andes_cache.routing import retrieval_route_for_intent


def env_debug_mode() -> bool:
    """Global debug toggle from environment (off by default)."""
    return str(os.getenv("ANDESCODE_DEBUG_MODE", "")).strip().lower() in {"1", "true", "yes", "on"}


def resolve_debug_mode(*, api_flag: bool | None = None, param_flag: bool | None = None) -> bool:
    """Resolve debug mode with precedence: API flag > function param > env."""
    if api_flag is not None:
        return bool(api_flag)
    if param_flag is not None:
        return bool(param_flag)
    return env_debug_mode()


def infer_expected_authority(intent: str, query: str) -> dict:
    q = (query or "").lower()
    expected_classes: list[str] = []
    expected_files: list[str] = []

    # Query-specific expectations first (more precise than broad intent defaults)
    if any(k in q for k in ("permission", "permissions")):
        expected_classes.append("manifest")
        expected_files.append("AndroidManifest.xml")
    elif any(k in q for k in ("dependenc", "library", "package", "requirement", "declared")):
        expected_classes.extend(["dependency_file", "build_file"])
    elif any(k in q for k in ("config", "configured", "setting", "env", "environment")):
        expected_classes.append("config_file")
        if "manifest" in q:
            expected_classes.append("manifest")

    # Specific file hint remains optional and additive.
    if any(k in q for k in ("permission", "permissions", "manifest")):
        expected_classes.append("manifest")
    if any(k in q for k in ("build", "gradle", "cargo", "pom", "requirements")):
        expected_classes.extend(["dependency_file", "build_file"])

    # Minimal fallback by intent only when query provides no strong hints.
    if not expected_classes:
        if intent == "dependency_or_build_inventory":
            expected_classes.extend(["dependency_file", "build_file"])
        elif intent == "declaration_or_configuration":
            expected_classes.append("config_file")

    return {
        "expected_classes": sorted(set(expected_classes)),
        "expected_files": sorted(set(expected_files)),
    }


def initialize_payload(query: str, decision: dict, workspace: dict | None) -> dict:
    ws = workspace or {}
    modules = ws.get("modules", []) if isinstance(ws.get("modules", []), list) else []
    manifests = ws.get("manifests", []) if isinstance(ws.get("manifests", []), list) else []
    primary_module = modules[0]["name"] if modules and isinstance(modules[0], dict) else ""
    expected = infer_expected_authority(decision.get("intent", ""), query)

    route_reason = (
        "Intent requires authoritative declaration/config artifacts"
        if decision.get("retrieval_route") == "source_of_truth"
        else "Intent favors runtime/symbol/semantic evidence"
    )

    return {
        "query": query,
        "intent": decision.get("intent", "unknown"),
        "retrieval_route": decision.get("retrieval_route", "semantic"),
        "strict_authority_mode": bool(decision.get("strict_authority_mode", False)),
        "workspace_summary": {
            "repo_types": ws.get("repo_types", []),
            "modules_detected": [m.get("name", "") for m in modules if isinstance(m, dict)],
            "manifests_found": manifests,
            "primary_module": primary_module,
            "entry_points": ws.get("entry_points", []),
        },
        "source_of_truth": {
            "expected_classes": expected["expected_classes"],
            "expected_files": expected["expected_files"],
            "priority_files": [],
            "recovery_candidates": [],
            "ranked_paths": [],
            "ranked_paths_with_scores": [],
            "selection_reason": "",
            "exact_path_used": "",
            "fallback_used": False,
            "selected_files": [],
            "missing_expected": [],
        },
        "retrieval": {
            "route_taken": decision.get("retrieval_route", "semantic"),
            "route_reason": route_reason,
            "files_retrieved": [],
            "raw_candidates": [],
            "selected_candidates": [],
            "authoritative_files_detected": [],
            "authoritative_files_required": [],
            "authoritative_files_retrieved": [],
            "authoritative_files_missing": [],
            "forced_authoritative_file": False,
            "authority_selection_reason": "",
            "authority_retrieval_mode": "",
            "declaration_answer_mode": "",
            "chunks_per_file": {},
            "coverage": {},
            "cache_hit": False,
        },
        "ranking": {"top_candidates": []},
        "final_context": {
            "files_used": [],
            "authoritative_files_present": False,
            "context_size": 0,
        },
        "summaries": {
            "why_route": route_reason,
            "why_files": "",
            "missing_authority": "",
        },
        "failure_signals": {
            "expected_but_missing_authority": False,
            "low_confidence_retrieval": False,
            "ambiguous_intent": bool(decision.get("ambiguous", False)),
            "fragmented_context": False,
            "wrong_retrieval_route": False,
            "empty_retrieval": False,
            "multi_module_ambiguity": False,
        },
    }


def apply_failure_signals(
    payload: dict,
    *,
    query: str,
    intent: str,
    retrieval_route: str,
    top_score: float | None = None,
    final_chunks: list[dict] | None = None,
) -> dict:
    p = deepcopy(payload)
    chunks = final_chunks or []

    expected_route = retrieval_route_for_intent(intent)
    p["failure_signals"]["wrong_retrieval_route"] = retrieval_route != expected_route

    p["failure_signals"]["empty_retrieval"] = len(chunks) == 0

    if retrieval_route == "source_of_truth":
        has_authoritative = any(
            c.get("source_type") in {"manifest", "build_file", "dependency_file", "config_file"} for c in chunks
        )
        p["failure_signals"]["low_confidence_retrieval"] = (len(chunks) == 0) or (not has_authoritative)
    else:
        p["failure_signals"]["low_confidence_retrieval"] = (top_score is None) or (top_score < 0.25)

    modules = p.get("workspace_summary", {}).get("modules_detected", [])
    q = (query or "").lower()
    module_hits = [m for m in modules if m and m.lower() in q]
    p["failure_signals"]["multi_module_ambiguity"] = (
        p["failure_signals"].get("ambiguous_intent", False)
        and len(modules) >= 2
        and len(module_hits) == 0
        and retrieval_route in {"source_of_truth", "semantic"}
    )

    expected_classes = set(p.get("source_of_truth", {}).get("expected_classes", []))
    found_classes = {
        c.get("source_type") for c in chunks if c.get("source_type") in {"manifest", "build_file", "dependency_file", "config_file"}
    }
    missing_classes = sorted(expected_classes - found_classes)
    p["source_of_truth"]["missing_expected"] = sorted(set(p["source_of_truth"].get("missing_expected", []) + missing_classes))
    p["failure_signals"]["expected_but_missing_authority"] = (
        retrieval_route == "source_of_truth" and len(expected_classes) > 0 and len(found_classes) == 0
    )

    p["summaries"]["missing_authority"] = (
        "Missing authoritative classes: " + ", ".join(p["source_of_truth"]["missing_expected"])
        if p["source_of_truth"]["missing_expected"] else ""
    )
    return p


def populate_retrieval_snapshot(
    payload: dict,
    *,
    chunks: list[dict],
    raw_candidates: list[str] | None = None,
    cache_hit: bool = False,
) -> dict:
    p = deepcopy(payload)
    selected = [c.get("file", "") for c in chunks]
    p["retrieval"]["cache_hit"] = cache_hit
    p["retrieval"]["raw_candidates"] = list(raw_candidates or selected)
    p["retrieval"]["selected_candidates"] = selected
    p["retrieval"]["files_retrieved"] = selected
    p["retrieval"]["chunks_per_file"] = {f: selected.count(f) for f in sorted(set(selected)) if f}
    p["retrieval"]["coverage"] = {c.get("file", ""): c.get("coverage", {}) for c in chunks}
    return p


def format_debug_sse_event(payload: dict) -> str:
    import json

    return f"event: debug\\ndata: {json.dumps({'object': 'debug.payload', 'debug': payload})}\\n\\n"


def finalize_payload(payload: dict, final_chunks: list[dict]) -> dict:
    p = deepcopy(payload)
    files = [c.get("file", "") for c in final_chunks if c.get("file")]
    p["final_context"]["files_used"] = files
    p["final_context"]["context_size"] = sum(len(c.get("content", "")) for c in final_chunks)
    p["final_context"]["authoritative_files_present"] = any(
        c.get("source_type") in {"manifest", "build_file", "config_file", "dependency_file"} for c in final_chunks
    )
    p["failure_signals"]["empty_retrieval"] = len(final_chunks) == 0
    p["failure_signals"]["fragmented_context"] = _is_fragmented(final_chunks)

    sel = p.get("retrieval", {}).get("selected_candidates", [])
    if sel:
        p["summaries"]["why_files"] = f"Selected {len(sel)} files from {p['retrieval'].get('route_taken')} route"
    return p


def _is_fragmented(chunks: list[dict]) -> bool:
    if len(chunks) < 4:
        return False
    by_file = {}
    for c in chunks:
        by_file[c.get("file", "")] = by_file.get(c.get("file", ""), 0) + 1
    if not by_file:
        return False
    avg = sum(by_file.values()) / len(by_file)
    return len(by_file) >= 4 and avg <= 1.5
