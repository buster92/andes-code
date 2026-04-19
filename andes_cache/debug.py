"""Structured deterministic debug payload helpers for AndesCode retrieval."""

from __future__ import annotations

import os
from copy import deepcopy


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


def initialize_payload(query: str, decision: dict, workspace: dict | None) -> dict:
    ws = workspace or {}
    modules = ws.get("modules", []) if isinstance(ws.get("modules", []), list) else []
    manifests = ws.get("manifests", []) if isinstance(ws.get("manifests", []), list) else []
    primary_module = modules[0]["name"] if modules and isinstance(modules[0], dict) else ""
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
            "priority_files": [],
            "recovery_candidates": [],
            "ranked_paths": [],
            "selected_files": [],
            "missing_expected": [],
        },
        "retrieval": {
            "route_taken": decision.get("retrieval_route", "semantic"),
            "files_retrieved": [],
            "raw_candidates": [],
            "selected_candidates": [],
            "chunks_per_file": {},
            "coverage": {},
        },
        "ranking": {"top_candidates": []},
        "final_context": {
            "files_used": [],
            "authoritative_files_present": False,
            "context_size": 0,
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
    return p


def apply_failure_signals(
    payload: dict,
    *,
    query: str,
    retrieval_route: str,
    top_score: float | None = None,
    final_chunks: list[dict] | None = None,
) -> dict:
    p = deepcopy(payload)
    q = (query or "").lower()
    chunks = final_chunks or []
    p["failure_signals"]["wrong_retrieval_route"] = ("permission" in q and retrieval_route != "source_of_truth")
    p["failure_signals"]["empty_retrieval"] = len(chunks) == 0
    p["failure_signals"]["low_confidence_retrieval"] = (top_score is None) or (top_score < 0.25)
    modules = p.get("workspace_summary", {}).get("modules_detected", [])
    p["failure_signals"]["multi_module_ambiguity"] = len(modules) > 1 and p["failure_signals"].get("ambiguous_intent", False)
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
