from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXTENSION_LANGUAGE = {
    ".py": "py",
    ".kt": "kt",
    ".kts": "kt",
    ".java": "java",
    ".js": "js",
    ".jsx": "jsx",
    ".ts": "ts",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rs",
    ".swift": "swift",
}


@dataclass(frozen=True)
class ParserHandle:
    language: str
    parser: Any | None
    available: bool
    reason: str = ""


class ParserRegistry:
    """Small optional tree-sitter registry.

    The code graph never depends on tree-sitter being installed.  Callers can ask
    for a parser and will receive ``available=False`` when the runtime lacks a
    grammar, allowing regex/built-in AST extraction to continue deterministically.
    """

    def __init__(self) -> None:
        self._cache: dict[str, ParserHandle] = {}

    def language_for_path(self, path: str | Path) -> str:
        return EXTENSION_LANGUAGE.get(Path(path).suffix.lower(), Path(path).suffix.lstrip(".").lower())

    def get_parser(self, language: str) -> ParserHandle:
        if language in self._cache:
            return self._cache[language]
        try:
            from tree_sitter import Parser  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            handle = ParserHandle(language=language, parser=None, available=False, reason=f"tree-sitter unavailable: {exc}")
            self._cache[language] = handle
            return handle

        # Prefer the common tree_sitter_languages package when present.  If it is
        # absent, return a clean fallback handle rather than failing indexing.
        try:  # pragma: no cover - depends on optional grammar packages
            from tree_sitter_languages import get_language  # type: ignore

            grammar_name = {"py": "python", "kt": "kotlin"}.get(language, language)
            ts_language = get_language(grammar_name)
            parser = Parser()
            parser.set_language(ts_language)
            handle = ParserHandle(language=language, parser=parser, available=True)
        except Exception as exc:
            handle = ParserHandle(language=language, parser=None, available=False, reason=f"grammar unavailable: {exc}")
        self._cache[language] = handle
        return handle
