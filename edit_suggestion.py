"""Repo-grounded edit suggestion helpers.

Edit Suggestion Mode is intentionally read-only.  These helpers classify edit
intent, build stricter prompt instructions, and validate generated answers
without applying patches or running shell commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable

EDIT_SUGGESTION = "edit_suggestion"

_SAFE_FALLBACK = "I do not have enough repo-grounded context to propose a safe edit."

_EDIT_INTENT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bimprove\s+(?:this|it|the|[\w./-]+|performance)\b",
        r"\bfix\s+(?:this|it|the|a\s+)?(?:bug|failure|error|issue|test|crash)?\b",
        r"\bmake\s+(?:this|it|the|[\w./-]+)\s+faster\b",
        r"\bsuggest\s+(?:one|a|an)\s+(?:concrete\s+)?(?:update|change|edit|fix|improvement)\b",
        r"\bchange\s+(?:this|it|the)\s+behavior\b",
        r"\bwhy\s+is\s+(?:this|it|the|[\w./-]+)\s+failing\b",
        r"\bwhat\s+code\s+should\s+i\s+edit\b",
        r"^\s*(?:please\s+)?update\s+[\w./-]+\s+(?:to|so|for|with)\b",
        r"^\s*(?:please\s+)?add\s+(?:a|an|the|one|new|missing|some)?\s*[\w-]+\s*(?:check|guard|test|feature|method|function|class|field|parameter|validation|cache|retry|handler|endpoint|implementation)?\b",
        r"\b(?:implement|modify|refactor|patch|optimi[sz]e)\b",
    )
]

_SYMBOL_RE = re.compile(r"\b(?:class|def|function|method|symbol|import|from)\s+([A-Za-z_]\w*)\b|\b([A-Z][A-Za-z0-9_]{2,}|[a-z_]\w{3,})\s*\(")


@dataclass(frozen=True)
class EditSuggestionContext:
    files: tuple[str, ...]
    symbols: tuple[str, ...]
    validation_commands: tuple[str, ...]
    existing_mechanisms: tuple[str, ...]
    missing_context: tuple[str, ...] = ()


def is_edit_suggestion_query(query: str) -> bool:
    """Return True when a user is asking for a concrete code edit suggestion."""
    q = (query or "").strip()
    if not q:
        return False
    return any(pattern.search(q) for pattern in _EDIT_INTENT_PATTERNS)


def is_symbol_optional_file(path: str) -> bool:
    """Return True for config/build/docs files where indexed symbols are not expected."""
    normalized = (path or "").replace("\\", "/").lower().strip()
    if not normalized:
        return False
    name = Path(normalized).name
    if normalized.startswith(".github/workflows/") or "/.github/workflows/" in normalized:
        return True
    if name in {
        "package.json",
        "dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        "settings.gradle",
        "settings.gradle.kts",
        "build.gradle",
        "build.gradle.kts",
    }:
        return True
    return any(
        normalized.endswith(ext)
        for ext in (".yml", ".yaml", ".json", ".toml", ".ini", ".md", ".txt")
    )


def requires_symbol_evidence(path: str) -> bool:
    """Return True when edit evidence should include a symbol/method/class name."""
    return bool(path and not is_symbol_optional_file(path))


def edit_suggestion_policy() -> str:
    """Strict prompt contract for read-only, repo-grounded edit suggestions."""
    return (
        "Edit Suggestion Mode v1 (read-only, repo-grounded):\n"
        "- The user is asking for a likely code change; do not give generic architecture advice.\n"
        "- First identify entry files, related ViewModel/service/repository/test/config files, and likely call/data path from imports, symbols, and references in the retrieved files.\n"
        "- Treat full-file chunks as the primary edit targets; use excerpts only as supporting context.\n"
        "- Before recommending a mechanism, explicitly check whether it already exists in the retrieved context. If it exists, mention the existing file/symbol instead of proposing it as new.\n"
        "- Do not claim to edit files and do not provide automatic file-write instructions.\n"
        f"- If relevant files or symbols are not present, say exactly: \"{_SAFE_FALLBACK}\" Then list the missing files or symbols needed.\n"
        "- Final answer MUST use exactly these sections: Finding, Evidence, Recommended change, Patch plan, Validation, Confidence.\n"
        "- Finding: short explanation of current behavior from retrieved files.\n"
        "- Evidence: concrete file paths and symbol/method/class names used to reach the conclusion.\n"
        "- Recommended change: one minimal change only.\n"
        "- Patch plan: file-by-file changes with method/function names; include snippets only when grounded in retrieved files.\n"
        "- Validation: specific commands inferred from repo files; if none can be inferred, say no test command could be inferred.\n"
        "- Confidence: high/medium/low based on completeness of retrieved context."
    )


def extract_symbols_from_chunks(chunks: Iterable[dict[str, Any]]) -> list[str]:
    """Extract explicit indexed symbols and common declaration/call names."""
    seen: set[str] = set()
    symbols: list[str] = []
    for chunk in chunks:
        raw_symbols = str(chunk.get("symbols") or "")
        candidates = [s for s in re.split(r"\s+", raw_symbols) if s]
        content = str(chunk.get("content") or "")
        for match in _SYMBOL_RE.finditer(content):
            candidates.extend([g for g in match.groups() if g])
        for sym in candidates:
            if sym in seen or len(sym) < 3:
                continue
            seen.add(sym)
            symbols.append(sym)
    return symbols[:40]


def infer_validation_commands(files: Iterable[str]) -> list[str]:
    """Infer safe validation commands from repo structure and loaded file names."""
    file_set = {str(f) for f in files if f}
    names = {Path(f).name for f in file_set}
    commands: list[str] = []
    if "pytest.ini" in names or "requirements.txt" in names or any(f.startswith("tests/") or "/tests/" in f for f in file_set):
        commands.append("pytest")
    if "package.json" in names:
        commands.append("npm test")
    if "pyproject.toml" in names and "pytest" not in commands:
        commands.append("python -m pytest")
    if "Cargo.toml" in names:
        commands.append("cargo test")
    if any(name in names for name in ("build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts")):
        commands.append("./gradlew test")
    if "go.mod" in names:
        commands.append("go test ./...")
    return list(dict.fromkeys(commands))[:3]


def detect_existing_mechanisms(chunks: Iterable[dict[str, Any]], query: str) -> list[str]:
    """Find retrieved mechanisms matching common requested changes."""
    q = (query or "").lower()
    mechanisms = {
        "cache": re.compile(r"\b(cache|cached|memoiz|lru_cache)\b", re.IGNORECASE),
        "retry": re.compile(r"\b(retry|backoff)\b", re.IGNORECASE),
        "debounce": re.compile(r"\b(debounce|throttle)\b", re.IGNORECASE),
        "validation": re.compile(r"\b(validate|validation|schema)\b", re.IGNORECASE),
        "error handling": re.compile(r"\b(except|catch|error|failure)\b", re.IGNORECASE),
        "test": re.compile(r"\b(test_|assert|pytest|unittest)\b", re.IGNORECASE),
    }
    requested = [name for name in mechanisms if name in q]
    if not requested:
        requested = list(mechanisms)
    found: list[str] = []
    for chunk in chunks:
        content = str(chunk.get("content") or "")
        file_path = str(chunk.get("file") or "")
        for name in requested:
            if mechanisms[name].search(content):
                found.append(f"{name} exists in {file_path}")
    return list(dict.fromkeys(found))[:8]


def build_edit_suggestion_context(chunks: list[dict[str, Any]], *, query: str = "") -> EditSuggestionContext:
    files = tuple(dict.fromkeys(str(c.get("file") or "") for c in chunks if c.get("file")))
    symbols = tuple(extract_symbols_from_chunks(chunks))
    commands = tuple(infer_validation_commands(files))
    existing = tuple(detect_existing_mechanisms(chunks, query))
    missing: list[str] = []
    if not files:
        missing.append("relevant files")
    has_symbol_optional_file = any(is_symbol_optional_file(path) for path in files)
    if not symbols and files and not has_symbol_optional_file:
        missing.append("symbols or methods in the relevant source files")
    return EditSuggestionContext(files=files, symbols=symbols, validation_commands=commands, existing_mechanisms=existing, missing_context=tuple(missing))


def safe_context_fallback(missing_context: Iterable[str]) -> str:
    missing = [m for m in missing_context if m]
    if not missing:
        missing = ["relevant files or symbols"]
    return f"{_SAFE_FALLBACK}\n\nMissing context needed:\n" + "\n".join(f"- {m}" for m in missing)


def answer_has_file_and_symbol_evidence(answer: str, ctx: EditSuggestionContext) -> bool:
    text = answer or ""
    cited_files = [f for f in ctx.files if f and f in text]
    if not cited_files:
        return False
    if any(is_symbol_optional_file(f) for f in cited_files):
        return True
    return any(s and re.search(rf"\b{re.escape(s)}\b", text) for s in ctx.symbols)


def enforce_edit_suggestion_output(answer: str, ctx: EditSuggestionContext) -> str:
    """Reject generic edit advice when retrieved context is insufficient."""
    if ctx.missing_context:
        return safe_context_fallback(ctx.missing_context)
    required = ("Finding", "Evidence", "Recommended change", "Patch plan", "Validation", "Confidence")
    missing_sections = [section for section in required if not re.search(rf"(^|\n)(?:#+\s*)?{re.escape(section)}\s*:?", answer or "", re.IGNORECASE)]
    if missing_sections or not answer_has_file_and_symbol_evidence(answer, ctx):
        return safe_context_fallback(["answer with concrete file paths and symbols"])
    return answer
