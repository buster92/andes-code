"""Conversation export helpers for AndesCode."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from runtime_paths import ensure_app_data_dir

EXPORT_DIR_NAME = "exports"


def default_export_dir() -> Path:
    """Return the predictable user-visible conversation export directory."""
    return ensure_app_data_dir() / EXPORT_DIR_NAME


def sanitize_filename_part(value: str, fallback: str = "conversation") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "-", (value or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ._-")
    return (cleaned or fallback)[:80]


def _parse_export_datetime(value: str | None) -> datetime:
    if value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except Exception:
            pass
    return datetime.now(timezone.utc)


def render_conversation_export(messages: Iterable[Mapping[str, object]], title: str, exported_at: datetime) -> str:
    lines = [
        f"# {title or 'AndesCode Conversation'}",
        f"Exported: {exported_at.isoformat()}",
        "",
    ]
    for message in messages:
        role = str(message.get("role", "message") or "message").upper()
        content = str(message.get("content", "") or "")
        lines.append(f"[{role}]")
        lines.append(content)
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_conversation_export(
    messages: list[Mapping[str, object]],
    title: str = "AndesCode Conversation",
    created_at: str | None = None,
    export_dir: Path | None = None,
) -> Path:
    if not messages:
        raise ValueError("No conversation messages to export.")

    exported_at = _parse_export_datetime(created_at)
    target_dir = export_dir or default_export_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    safe_title = sanitize_filename_part(title, fallback="conversation")
    timestamp = exported_at.strftime("%Y-%m-%d_%H-%M-%S")
    path = target_dir / f"{safe_title}_{timestamp}.md"

    suffix = 2
    while path.exists():
        path = target_dir / f"{safe_title}_{timestamp}_{suffix}.md"
        suffix += 1

    path.write_text(render_conversation_export(messages, title, exported_at), encoding="utf-8")
    return path
