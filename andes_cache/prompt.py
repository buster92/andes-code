"""Deterministic prompt section serialization for prefix-cache reuse."""

from __future__ import annotations


def build_prompt_sections(
    *,
    system_prefix: str,
    workspace_prefix: str,
    retrieval_context: str,
    user_turn: str,
) -> dict:
    """Build deterministic section object for prompt rendering."""
    return {
        "system_prefix": system_prefix.strip(),
        "workspace_prefix": workspace_prefix.strip(),
        "retrieval_context": retrieval_context.strip(),
        "user_turn": user_turn.strip(),
    }


def serialize_prompt_sections(sections: dict) -> str:
    """Render prompt with stable ordering and separators."""
    ordered = [
        ("SYSTEM PREFIX", sections.get("system_prefix", "")),
        ("WORKSPACE PREFIX", sections.get("workspace_prefix", "")),
        ("RETRIEVAL CONTEXT", sections.get("retrieval_context", "")),
        ("USER TURN", sections.get("user_turn", "")),
    ]
    lines = []
    for title, content in ordered:
        lines.append(f"## {title}")
        lines.append(content)
        lines.append("")
    return "\n".join(lines).strip()
