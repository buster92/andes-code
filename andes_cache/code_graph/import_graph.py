from __future__ import annotations

import posixpath
import re
from collections import defaultdict
from pathlib import Path

from .models import ImportEdge
from .parser_registry import ParserRegistry

_INDEX_BASENAMES = ("index", "__init__")
_JS_TS_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}


def extract_import_names(text: str, language: str) -> list[str]:
    imports: set[str] = set()
    if language == "py":
        imports.update(_extract_python_import_names(text))
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
        candidates.extend(_relative_python_import_candidates(import_name, source))
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
    suffix = Path(joined).suffix.lower()
    if suffix in _JS_TS_EXTENSIONS:
        candidates.append(str(Path(joined).with_suffix("")))
    candidates.extend(f"{joined}/{basename}" for basename in _INDEX_BASENAMES)
    for candidate in candidates:
        if candidate in module_map:
            return module_map[candidate]
    return ""


def _extract_python_import_names(text: str) -> set[str]:
    imports: set[str] = set()
    imports.update(re.findall(r"^\s*import\s+([A-Za-z0-9_\.]+)", text, re.MULTILINE))
    for match in re.finditer(r"^\s*from\s+([A-Za-z0-9_\.]*|\.+[A-Za-z0-9_\.]*)\s+import\s+([^#\n]+)", text, re.MULTILINE):
        module = match.group(1).strip()
        imported_names = _python_imported_names(match.group(2))
        if module and module != ".":
            imports.add(module)
        for name in imported_names:
            if name == "*":
                continue
            if module in {"", "."}:
                imports.add(f".{name}")
            elif not module.startswith(".") and "." not in module:
                imports.add(f"{module}.{name}")
    return imports


def _python_imported_names(import_clause: str) -> list[str]:
    names = []
    for item in import_clause.strip().strip("()").split(","):
        name = item.strip().split(" as ", 1)[0].strip()
        if name:
            names.append(name)
    return names


def _relative_python_import_candidates(import_name: str, source: str) -> list[str]:
    leading = len(import_name) - len(import_name.lstrip("."))
    remainder = import_name[leading:].strip(".")
    source_parts = Path(source).parent.parts
    keep = max(len(source_parts) - max(leading - 1, 0), 0)
    base_parts = list(source_parts[:keep])
    if remainder:
        base_parts.extend(part for part in remainder.split(".") if part)
    slash_key = "/".join(base_parts)
    dotted_key = ".".join(base_parts)
    return [key for key in (dotted_key, slash_key) if key]


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
        mapping.setdefault(rel, rel)
        mapping.setdefault(no_suffix, rel)
        mapping.setdefault(no_suffix.replace("/", "."), rel)
        if Path(rel).stem == "__init__":
            package_key = Path(no_suffix).parent.as_posix()
            if package_key and package_key != ".":
                mapping.setdefault(package_key, rel)
                mapping.setdefault(package_key.replace("/", "."), rel)
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
