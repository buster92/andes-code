"""Structured deterministic debug payload helpers for AndesCode retrieval."""

from __future__ import annotations

import os
from copy import deepcopy

from andes_cache.routing import retrieval_route_for_intent


def _coerce_debug_flag(value) -> bool | None:
    """Normalize common debug flag representations into bool/None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
        return None
    return bool(value)


def env_debug_mode() -> bool:
    """Global debug toggle from environment (off by default)."""
    return bool(_coerce_debug_flag(os.getenv("ANDESCODE_DEBUG_MODE", "")))


def resolve_debug_mode(
    *,
    api_flag: bool | str | None = None,
    param_flag: bool | str | None = None,
    request_flag: bool | str | None = None,
    env_flag: bool | str | None = None,
) -> bool:
    """Resolve debug mode with precedence: request/API flag > explicit/env parameter > env."""
    api_value = _coerce_debug_flag(api_flag if api_flag is not None else request_flag)
    if api_value is not None:
        return api_value
    param_value = _coerce_debug_flag(param_flag if param_flag is not None else env_flag)
    if param_value is not None:
        return param_value
    return env_debug_mode()


def compute_intent_authority_satisfaction(intent: str, chunks: list[dict]) -> dict:
    """Compute intent-specific authority class satisfaction from a list of retrieved chunks.

    For ``dependency_or_build_inventory`` queries the *required* classes are
    ``{"build_file", "dependency_file"}`` — manifest-only context does NOT satisfy this.
    For ``declaration_or_configuration`` queries the required classes are
    ``{"config_file", "manifest"}``.
    For all other intents no specific class is required (empty sets are returned).

    Returns a dict with three sorted lists:
      - ``required_authority_classes``  — what this intent needs
      - ``satisfied_authority_classes`` — what was found in ``chunks``
      - ``missing_authority_classes``   — what is still absent
    """
    from andes_cache.source_of_truth import DEPENDENCY_BUILD_AUTHORITY_TYPES  # local to avoid circular import

    AUTHORITATIVE_SOURCE_TYPES = {"manifest", "build_file", "dependency_file", "config_file"}

    if intent == "dependency_or_build_inventory":
        required: set[str] = DEPENDENCY_BUILD_AUTHORITY_TYPES.copy()
    elif intent == "declaration_or_configuration":
        required = {"config_file", "manifest"}
    else:
        required = set()

    found_types = {
        c.get("source_type")
        for c in chunks
        if c.get("source_type") in AUTHORITATIVE_SOURCE_TYPES
    }
    satisfied = required & found_types
    missing = required - found_types
    return {
        "required_authority_classes": sorted(required),
        "satisfied_authority_classes": sorted(satisfied),
        "missing_authority_classes": sorted(missing),
    }


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
            "declaration_query_trigger_reason": "",
            # Intent-specific authority class tracking (dependency/declaration queries)
            "required_authority_classes": [],
            "satisfied_authority_classes": [],
            "missing_authority_classes": [],
            # Dependency-specific recovery tracking
            "dependency_recovery_attempted": False,
            "dependency_recovery_candidates": [],
            "dependency_recovery_selected_files": [],
            "dependency_recovery_succeeded": False,
            "chunks_per_file": {},
            "coverage": {},
            "cache_hit": False,
            "retrieval_routes_used": [],
            "graph_neighbors_added": [],
            "symbols_matched": [],
            "files_selected_by_graph": [],
            "files_selected_by_semantic": [],
            "files_selected_by_authority": [],
            "context_sufficiency_notes": [],
        },
        "ranking": {"top_candidates": []},
        "final_context": {
            "files_used": [],
            "authoritative_files_present": False,
            # Intent-specific: True only when the required authority classes for THIS intent
            # are satisfied (e.g. for dep queries, manifest-only context returns False here).
            "authoritative_files_present_for_intent": False,
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
        # For dependency queries, manifest-only context is not sufficient authority.
        if intent == "dependency_or_build_inventory":
            from andes_cache.source_of_truth import DEPENDENCY_BUILD_AUTHORITY_TYPES
            has_authoritative = any(
                c.get("source_type") in DEPENDENCY_BUILD_AUTHORITY_TYPES for c in chunks
            )
        else:
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

    # Intent-specific authority gap detection.
    # For dependency queries: expected_but_missing_authority fires when no dependency/build
    # authority files are found — manifest-only context does NOT satisfy this.
    if retrieval_route == "source_of_truth" and len(expected_classes) > 0:
        if intent == "dependency_or_build_inventory":
            from andes_cache.source_of_truth import DEPENDENCY_BUILD_AUTHORITY_TYPES
            dep_expected = expected_classes & DEPENDENCY_BUILD_AUTHORITY_TYPES
            dep_found = found_classes & DEPENDENCY_BUILD_AUTHORITY_TYPES
            p["failure_signals"]["expected_but_missing_authority"] = (
                len(dep_expected) > 0 and len(dep_found) == 0
            )
        else:
            p["failure_signals"]["expected_but_missing_authority"] = len(found_classes) == 0
    else:
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

    return f"event: debug\ndata: {json.dumps({'object': 'debug.payload', 'debug': payload})}\n\n"


def finalize_payload(payload: dict, final_chunks: list[dict]) -> dict:
    p = deepcopy(payload)
    intent = p.get("intent", "")
    files = [c.get("file", "") for c in final_chunks if c.get("file")]
    p["final_context"]["files_used"] = files
    p["final_context"]["context_size"] = sum(len(c.get("content", "")) for c in final_chunks)
    p["final_context"]["authoritative_files_present"] = any(
        c.get("source_type") in {"manifest", "build_file", "config_file", "dependency_file"} for c in final_chunks
    )

    # Intent-specific authority satisfaction — manifest-only context does NOT satisfy
    # dependency_or_build_inventory queries.
    auth = compute_intent_authority_satisfaction(intent, final_chunks)
    p["retrieval"]["required_authority_classes"] = auth["required_authority_classes"]
    p["retrieval"]["satisfied_authority_classes"] = auth["satisfied_authority_classes"]
    p["retrieval"]["missing_authority_classes"] = auth["missing_authority_classes"]
    if auth["required_authority_classes"]:
        # OR semantics: having *any one* of the required classes is sufficient
        # (e.g. build.gradle alone satisfies a dep query even if requirements.txt is absent).
        # missing_authority_classes is informational (shows what else could be present)
        # but does NOT by itself mean the intent is unsatisfied.
        p["final_context"]["authoritative_files_present_for_intent"] = (
            len(auth["satisfied_authority_classes"]) > 0
        )
    else:
        # No specific class requirement for this intent — fall back to the generic flag.
        p["final_context"]["authoritative_files_present_for_intent"] = (
            p["final_context"]["authoritative_files_present"]
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
