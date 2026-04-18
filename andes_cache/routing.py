"""Deterministic query-intent routing helpers used before retrieval/planning."""

from __future__ import annotations

import re

DECLARATION_OR_CONFIGURATION = "declaration_or_configuration"
DEPENDENCY_OR_BUILD_INVENTORY = "dependency_or_build_inventory"
RUNTIME_USAGE_OR_REFERENCE = "runtime_usage_or_reference"
ARCHITECTURE_OVERVIEW = "architecture_overview"
SYMBOL_LOOKUP = "symbol_lookup"
CODE_FIX_OR_PATCH = "code_fix_or_patch"
GENERIC_SEMANTIC = "generic_semantic"


def classify_query_intent_details(query: str) -> dict:
    """
    Rule-based classifier with lightweight ambiguity metadata.
    Classification is deterministic and runs before any cache lookups/retrieval.
    """
    q = (query or "").strip().lower()
    words = set(re.findall(r"\w+", q))

    patch_score = _score(
        q, words, {"fix", "patch", "edit", "change", "refactor", "bug", "implement"}
    )
    decl_score = _score(
        q,
        words,
        {
            "declared",
            "configured",
            "configuration",
            "manifest",
            "setting",
            "settings",
            "permission",
            "permissions",
            "env",
            "environment",
            "defined",
            "define",
        },
    )
    dep_score = _score(
        q,
        words,
        {
            "dependency",
            "dependencies",
            "library",
            "libraries",
            "package",
            "packages",
            "version",
            "versions",
            "requirements",
            "gradle",
            "pom",
            "maven",
            "cargo",
            "go",
            "build",
        },
    )
    runtime_score = _score(
        q, words, {"used", "usage", "called", "invoked", "referenced", "checked", "runtime"}
    )
    symbol_score = _score(
        q, words, {"symbol", "function", "class", "method", "definition", "defined"}
    )
    architecture_score = _score(
        q,
        words,
        {"architecture", "overview", "flow", "pipeline", "entry", "startup", "module"},
    )

    if patch_score > 0:
        intent = CODE_FIX_OR_PATCH
    elif dep_score > 0 and dep_score >= decl_score:
        intent = DEPENDENCY_OR_BUILD_INVENTORY
    elif runtime_score > 0 and runtime_score > decl_score:
        intent = RUNTIME_USAGE_OR_REFERENCE
    elif symbol_score > 0 and re.search(r"\b(where is .*defined|definition of)\b", q):
        intent = SYMBOL_LOOKUP
    elif decl_score > 0:
        intent = DECLARATION_OR_CONFIGURATION
    elif architecture_score > 0:
        intent = ARCHITECTURE_OVERVIEW
    elif symbol_score > 0:
        intent = SYMBOL_LOOKUP
    else:
        intent = GENERIC_SEMANTIC

    # Ambiguity: declaration/dependency + runtime signals without explicit declaration wording.
    explicit_decl = bool(
        re.search(r"\b(declared|configured|manifest|config(?:uration)?|build file)\b", q)
    )
    ambiguous = (
        intent in {DECLARATION_OR_CONFIGURATION, DEPENDENCY_OR_BUILD_INVENTORY}
        and runtime_score > 0
        and not explicit_decl
    )
    if intent == DEPENDENCY_OR_BUILD_INVENTORY and re.search(r"\b(needed|required)\b", q):
        ambiguous = True

    retrieval_route = retrieval_route_for_intent(intent)
    return {
        "intent": intent,
        "retrieval_route": retrieval_route,
        "ambiguous": ambiguous,
        "allow_runtime_fallback": runtime_score > 0,
    }


def classify_query_intent(query: str) -> str:
    """Compatibility helper returning only the intent class."""
    return classify_query_intent_details(query)["intent"]


def retrieval_route_for_intent(intent: str) -> str:
    if intent in {DECLARATION_OR_CONFIGURATION, DEPENDENCY_OR_BUILD_INVENTORY}:
        return "source_of_truth"
    if intent == RUNTIME_USAGE_OR_REFERENCE:
        return "runtime_usage"
    if intent == SYMBOL_LOOKUP:
        return "symbol_lookup"
    return "semantic"


def is_fast_path_intent(intent: str) -> bool:
    return intent in {DECLARATION_OR_CONFIGURATION, DEPENDENCY_OR_BUILD_INVENTORY}


def semantic_cache_allowed(intent: str, retrieval_route: str) -> bool:
    return intent in {ARCHITECTURE_OVERVIEW, GENERIC_SEMANTIC} and retrieval_route == "semantic"


def orchestration_plan(intent: str) -> dict:
    fast = is_fast_path_intent(intent)
    return {
        "skip_patch_plan": fast,
        "skip_patch_diagnosis": fast,
        "skip_neighborhood": fast,
    }


def _score(q: str, words: set[str], terms: set[str]) -> int:
    return sum(1 for t in terms if t in words or t in q)
