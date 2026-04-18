"""Deterministic query-intent routing helpers used before retrieval/planning."""

from __future__ import annotations

import re

CONFIG_DECLARATION = "config_declaration"
DEPENDENCY_INVENTORY = "dependency_inventory"
ARCHITECTURE_OVERVIEW = "architecture_overview"
SYMBOL_LOOKUP = "symbol_lookup"
CODE_FIX_PATCH = "code_fix_patch"
GENERIC_SEMANTIC = "generic_semantic"


def classify_query_intent(query: str) -> str:
    """Rule-based intent classifier with explainable lexical rules."""
    q = (query or "").strip().lower()

    if re.search(r"\b(fix|patch|edit|change|refactor|bug|implement)\b", q):
        return CODE_FIX_PATCH

    if re.search(r"\b(permission|permissions|declared|manifest)\b", q):
        return CONFIG_DECLARATION

    if re.search(r"\b(dependenc|library|libraries|package|version|requirements|gradle|cargo|go\.mod|pom)\b", q):
        return DEPENDENCY_INVENTORY

    if re.search(r"\b(config|configure|configuration|env|build|capabilit|settings)\b", q):
        return CONFIG_DECLARATION

    if re.search(r"\b(symbol|function|class|method|where is .*defined|definition of)\b", q):
        return SYMBOL_LOOKUP

    if re.search(r"\b(architecture|overview|flow|pipeline|entry point|how does|how is|where is)\b", q):
        return ARCHITECTURE_OVERVIEW

    return GENERIC_SEMANTIC


def retrieval_route_for_intent(intent: str) -> str:
    if intent in {CONFIG_DECLARATION, DEPENDENCY_INVENTORY}:
        return "config_first"
    if intent == SYMBOL_LOOKUP:
        return "symbol_lookup"
    return "semantic"


def is_fast_path_intent(intent: str) -> bool:
    return intent in {CONFIG_DECLARATION, DEPENDENCY_INVENTORY}


def semantic_cache_allowed(intent: str, retrieval_route: str) -> bool:
    return intent in {ARCHITECTURE_OVERVIEW, GENERIC_SEMANTIC} and retrieval_route == "semantic"


def orchestration_plan(intent: str) -> dict:
    fast = is_fast_path_intent(intent)
    return {
        "skip_patch_plan": fast,
        "skip_patch_diagnosis": fast,
        "skip_neighborhood": fast,
    }
