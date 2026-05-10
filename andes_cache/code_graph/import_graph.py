from __future__ import annotations

import posixpath
import re
from collections import defaultdict
from pathlib import Path

from .models import ImportEdge
from .parser_registry import ParserRegistry

_INDEX_BASENAMES = ("index",)


def extract_import_names(text: str, language: str) -> list[str]:
    imports: set[str] = set()
    if language == "py":
        imports.update(re.findall(r"^\s*import\s+([A-Za-z0-9_\.]+)", text, re.MULTILINE))
        imports.update(re.findall(r"^\s*from\s+([A-Za-z0-9_\.]+)\s+import", text, re.MULTILINE))
    elif language in {"kt", "java"}:
        imports.update(re.findall(r"^\s*import\s+([A-Za-z0-9_.*]+)", text, re.MULTILINE))
    elif language in {"js", "jsx", "ts", "tsx"}:
        imports.update(re.findall(r"from\s+[\"']([^\"']+)[\"']", text))
        imports.update(re.findall(r"require\([\"']([^\"']+)[\"']\)", text))
    elif language == "go":
        block = "\n".join(re.findall(r"import\s*\((.*?)\)", text, re.DOTALL))
        imports.update(re.findall(r'"([^"]+)"', block))
        imports.update(re.findall(r'^\s*import\s+"([^"]+)"', text, re.MULTILINE))
    elif language == "rs":
        imports.update(re.findall(r"^\s*use\s+([A-Za-z0-9_:]+)", text, re.MULTILINE))
    return sorted(i for i in imports if i)


def build_import_graph(files: list[Path], root: Path, registry: ParserRegistry | None = None) -> dict:
    registry = registry or ParserRegistry()
    module_map = _module_map(files, root, registry)
    edges: list[ImportEdge] = []
    adjacency: dict[str, set[str]] = defaultdict(set)
    reverse: dict[str, set[str]] = defaultdict(set)
    unresolved: dict[str, list[str]] = defaultdict(list)

    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        rel = str(fp.relative_to(root))
        language = registry.language_for_path(fp)
        for import_name in extract_import_names(text, language):
            target = resolve_import(import_name, rel, module_map)
            resolved = bool(target)
            target_name = target or import_name
            edges.append(ImportEdge(rel, target_name, import_name, resolved))
            if resolved:
                adjacency[rel].add(target_name)
                reverse[target_name].add(rel)
            else:
                unresolved[rel].append(import_name)

    return {
        "edge_count": len(edges),
        "edges": [e.to_dict() for e in edges],
        "adjacency": {k: sorted(v) for k, v in adjacency.items()},
        "reverse_adjacency": {k: sorted(v) for k, v in reverse.items()},
        "unresolved": {k: sorted(v) for k, v in unresolved.items()},
    }


def expand_import_neighbors(seed_files: list[str] | set[str], graph: dict, limit: int = 12) -> list[str]:
    adjacency = graph.get("adjacency", {}) if isinstance(graph, dict) else {}
    reverse = graph.get("reverse_adjacency", {}) if isinstance(graph, dict) else {}
    out: list[str] = []
    seen = set(seed_files)
    for seed in sorted(seed_files):
        for neighbor in list(adjacency.get(seed, [])) + list(reverse.get(seed, [])):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            out.append(neighbor)
            if len(out) >= limit:
                return out
    return out


def resolve_import(import_name: str, source: str, module_map: dict[str, str]) -> str:
    if import_name.startswith(("./", "../")):
        return _resolve_relative_path_import(import_name, source, module_map)

    candidates = [import_name]
    if import_name.startswith("."):
        parent = Path(source).parent.as_posix().replace("/", ".")
        stripped = import_name.lstrip(".")
        candidates.append(f"{parent}.{stripped}".strip("."))
    if "." in import_name:
        parts = import_name.split(".")
        candidates.extend(".".join(parts[:i]) for i in range(len(parts), 0, -1))
    for candidate in candidates:
        if candidate in module_map:
            return module_map[candidate]
    return ""


def _resolve_relative_path_import(import_name: str, source: str, module_map: dict[str, str]) -> str:
    source_dir = Path(source).parent.as_posix()
    joined = posixpath.normpath(posixpath.join(source_dir, import_name))
    if joined.startswith("../") or joined == "..":
        return ""
    candidates = [joined]
    candidates.extend(f"{joined}/{basename}" for basename in _INDEX_BASENAMES)
    for candidate in candidates:
        if candidate in module_map:
            return module_map[candidate]
    return ""


def _module_map(files: list[Path], root: Path, registry: ParserRegistry) -> dict[str, str]:
    mapping: dict[str, str] = {}
    basename_candidates: dict[str, set[str]] = defaultdict(set)

    for fp in files:
        rel = str(fp.relative_to(root))
        language = registry.language_for_path(fp)
        no_suffix = fp.relative_to(root).with_suffix("").as_posix()
        # Slash-style keys support JS/TS relative imports; dotted keys support
        # Python/JVM imports.  Keep exact keys deterministic and do not invent
        # edges unless an indexed file already maps to the candidate.
        mapping.setdefault(no_suffix, rel)
        mapping.setdefault(no_suffix.replace("/", "."), rel)
        basename_candidates[Path(rel).stem].add(rel)
        if language in {"kt", "java"}:
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                text = ""
            package = re.search(r"^\s*package\s+([A-Za-z0-9_.]+)", text, re.MULTILINE)
            if package:
                mapping.setdefault(f"{package.group(1)}.{fp.stem}", rel)

    # Basename-only imports are a useful fallback for small Python scripts, but
    # they are unsafe when multiple files share the same basename.  In that case
    # omit the ambiguous key so callers preserve the import as unresolved rather
    # than creating a misleading edge to whichever file happened to appear first.
    for basename, rels in basename_candidates.items():
        if len(rels) == 1:
            mapping.setdefault(basename, next(iter(rels)))
    return mapping
