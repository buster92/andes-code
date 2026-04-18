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


def classify_query_intent(query: str) -> str:
    """Rule-based intent classifier with explainable lexical rules."""
    q = (query or "").strip().lower()

    if re.search(r"\b(fix|patch|edit|change|refactor|bug|implement)\b", q):
        return CODE_FIX_OR_PATCH

    if re.search(r"\b(declared|configured|configuration|manifest|setting|settings|permission|permissions|env|environment)\b", q):
        return DECLARATION_OR_CONFIGURATION

    if re.search(r"\b(dependenc|library|libraries|package|version|requirements|gradle|cargo|go\.mod|pom|build inventory)\b", q):
        return DEPENDENCY_OR_BUILD_INVENTORY

    if re.search(r"\b(used|usage|called|invoked|referenced|checked|where is this called|where is this used)\b", q):
        return RUNTIME_USAGE_OR_REFERENCE

    if re.search(r"\b(symbol|function|class|method|where is .*defined|definition of)\b", q):
        return SYMBOL_LOOKUP

    if re.search(r"\b(architecture|overview|flow|pipeline|entry point|how does|how is|where is)\b", q):
        return ARCHITECTURE_OVERVIEW

    return GENERIC_SEMANTIC


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
