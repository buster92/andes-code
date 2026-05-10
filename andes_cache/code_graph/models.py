from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class CodeSymbol:
    file_path: str
    language: str
    name: str
    kind: str
    start_line: int
    end_line: int
    signature: str = ""
    parent: str = ""
    imports: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImportEdge:
    source: str
    target: str
    import_name: str
    resolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RepoGraph:
    symbols: list[CodeSymbol] = field(default_factory=list)
    imports: list[ImportEdge] = field(default_factory=list)
    references: dict[str, list[str]] = field(default_factory=dict)
    files: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbols": [s.to_dict() for s in self.symbols],
            "imports": [e.to_dict() for e in self.imports],
            "references": self.references,
            "files": self.files,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RepoGraph":
        data = data or {}
        return cls(
            symbols=[CodeSymbol(**s) for s in data.get("symbols", []) if isinstance(s, dict)],
            imports=[ImportEdge(**e) for e in data.get("imports", []) if isinstance(e, dict)],
            references={str(k): list(v) for k, v in (data.get("references", {}) or {}).items()},
            files={str(k): dict(v) for k, v in (data.get("files", {}) or {}).items()},
        )
