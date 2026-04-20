"""Token-aware prompt budgeting and deterministic context chunk packing."""

from __future__ import annotations

import math
from dataclasses import dataclass


DEFAULT_MODEL_CTX = 8192
DEFAULT_RESPONSE_TOKENS = 1400
DEFAULT_SAFETY_MARGIN = 256
# Conservative heuristic: over-estimate tokens so we stay under hard window.
DEFAULT_CHARS_PER_TOKEN = 3.0


@dataclass(frozen=True)
class BudgetResult:
    total_ctx: int
    reserved_response: int
    safety_margin: int
    fixed_prompt_tokens: int
    context_budget_tokens: int


@dataclass(frozen=True)
class PackedContext:
    chunks: list[dict]
    used_tokens: int
    considered_chunks: int
    packed_chunks: int
    truncated: bool
    dropped_files: list[str]
    kept_files: list[str]


def estimate_tokens(text: str, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    """Conservative rough token estimate from text length."""
    if not text:
        return 0
    return int(math.ceil(len(text) / max(chars_per_token, 1e-6)))


def compute_context_budget(
    *,
    system_prompt: str,
    workspace_prefix: str,
    user_query: str,
    total_ctx: int = DEFAULT_MODEL_CTX,
    reserved_response: int = DEFAULT_RESPONSE_TOKENS,
    safety_margin: int = DEFAULT_SAFETY_MARGIN,
) -> BudgetResult:
    fixed_prompt_tokens = estimate_tokens(system_prompt) + estimate_tokens(workspace_prefix) + estimate_tokens(user_query)
    context_budget_tokens = max(total_ctx - reserved_response - safety_margin - fixed_prompt_tokens, 0)
    return BudgetResult(
        total_ctx=total_ctx,
        reserved_response=reserved_response,
        safety_margin=safety_margin,
        fixed_prompt_tokens=fixed_prompt_tokens,
        context_budget_tokens=context_budget_tokens,
    )


def pack_chunks_to_budget(chunks: list[dict], budget_tokens: int) -> PackedContext:
    """Pack already-prioritized chunk entries into the available context budget."""
    used_tokens = 0
    packed: list[dict] = []
    dropped: list[dict] = []
    blocked_lower_tiers = False

    tiers = sorted({int(chunk.get("tier", 999)) for chunk in chunks})
    for tier in tiers:
        tier_chunks = [c for c in chunks if int(c.get("tier", 999)) == tier]
        tier_overflowed = False
        for chunk in tier_chunks:
            chunk_tokens = int(chunk.get("est_tokens", 0))
            if used_tokens + chunk_tokens > budget_tokens:
                dropped.append(chunk)
                tier_overflowed = True
                continue
            packed.append(chunk)
            used_tokens += chunk_tokens
        if tier_overflowed:
            blocked_lower_tiers = True
            for chunk in chunks:
                c_tier = int(chunk.get("tier", 999))
                if c_tier > tier and chunk not in dropped and chunk not in packed:
                    dropped.append(chunk)
            break

    packed_files = sorted({c.get("file", "") for c in packed if c.get("file")})
    dropped_files = sorted({c.get("file", "") for c in dropped if c.get("file") and c.get("file") not in packed_files})
    truncated = blocked_lower_tiers or len(packed) < len(chunks)

    return PackedContext(
        chunks=packed,
        used_tokens=used_tokens,
        considered_chunks=len(chunks),
        packed_chunks=len(packed),
        truncated=truncated,
        dropped_files=dropped_files,
        kept_files=packed_files,
    )
