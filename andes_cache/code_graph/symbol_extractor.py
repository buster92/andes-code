from __future__ import annotations

import ast
import re
from pathlib import Path

from .models import CodeSymbol
from .parser_registry import ParserRegistry

_SYMBOL_KINDS = {"function", "class", "method", "interface", "object", "type", "constant"}


def extract_symbols_for_file(path: Path, root: Path, registry: ParserRegistry | None = None) -> list[CodeSymbol]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    language = (registry or ParserRegistry()).language_for_path(path)
    rel = str(path.relative_to(root))
    return extract_symbols(text, rel, language, registry=registry)


def extract_symbols(text: str, file_path: str, language: str, registry: ParserRegistry | None = None) -> list[CodeSymbol]:
    registry = registry or ParserRegistry()
    handle = registry.get_parser(language)
    if handle.available and handle.parser is not None:
        try:
            # Parse once when a grammar exists so syntax-aware integrations can
            # attach richer node extraction without changing this public API.
            handle.parser.parse(text.encode("utf-8"))
        except Exception:
            pass
    # First version uses Python's built-in AST for Python (more stable than an
    # optional grammar) and regex fallbacks for all languages.
    if language == "py":
        try:
            return _python_ast_symbols(text, file_path)
        except SyntaxError:
            return _regex_symbols(text, file_path, language)
    return _regex_symbols(text, file_path, language)


def _python_ast_symbols(text: str, file_path: str) -> list[CodeSymbol]:
    tree = ast.parse(text)
    imports = _extract_imports(text, "py")
    lines = text.splitlines()
    symbols: list[CodeSymbol] = []

    def visit_body(body: list[ast.stmt], parent: str = "") -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "method" if parent else "function"
                symbols.append(CodeSymbol(
                    file_path=file_path,
                    language="py",
                    name=node.name,
                    kind=kind,
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno),
                    signature=_line_signature(lines, node.lineno),
                    parent=parent,
                    imports=imports,
                    references=_name_refs(node),
                ))
            elif isinstance(node, ast.ClassDef):
                symbols.append(CodeSymbol(
                    file_path=file_path,
                    language="py",
                    name=node.name,
                    kind="class",
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno),
                    signature=_line_signature(lines, node.lineno),
                    parent=parent,
                    imports=imports,
                    references=_name_refs(node),
                ))
                visit_body(node.body, node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)) and not parent:
                for name in _assigned_names(node):
                    if name.isupper():
                        symbols.append(CodeSymbol(file_path, "py", name, "constant", node.lineno, getattr(node, "end_lineno", node.lineno), _line_signature(lines, node.lineno), "", imports, []))
    visit_body(tree.body)
    return symbols


def _assigned_names(node: ast.stmt) -> list[str]:
    targets = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    names = []
    for target in targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
    return names


def _name_refs(node: ast.AST) -> list[str]:
    refs = sorted({n.id for n in ast.walk(node) if isinstance(n, ast.Name)})
    return refs[:100]


def _line_signature(lines: list[str], lineno: int) -> str:
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1].strip()[:300]
    return ""


def _extract_imports(text: str, language: str) -> list[str]:
    imports: set[str] = set()
    if language == "py":
        imports.update(re.findall(r"^\s*import\s+([A-Za-z0-9_\.]+)", text, re.MULTILINE))
        imports.update(re.findall(r"^\s*from\s+([A-Za-z0-9_\.]+)\s+import", text, re.MULTILINE))
    elif language == "kt":
        imports.update(re.findall(r"^\s*import\s+([A-Za-z0-9_.*]+)", text, re.MULTILINE))
    return sorted(imports)


def _regex_symbols(text: str, file_path: str, language: str) -> list[CodeSymbol]:
    imports = _extract_imports(text, language)
    lines = text.splitlines()
    symbols: list[CodeSymbol] = []
    parent_stack: list[tuple[str, int]] = []

    if language == "kt":
        pattern = re.compile(r"^\s*(?:public\s+|private\s+|internal\s+|protected\s+|open\s+|data\s+|sealed\s+)*(fun|class|object|interface|typealias)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
    elif language in {"js", "jsx", "ts", "tsx"}:
        pattern = re.compile(
            r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?"
            r"(function|class|interface|type|const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)",
            re.MULTILINE,
        )
    else:
        pattern = re.compile(r"^\s*(def|class|async\s+def|function|interface|type|object|const)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)

    matches = list(pattern.finditer(text))
    for idx, match in enumerate(matches):
        line_no = text[:match.start()].count("\n") + 1
        end_line = (text[:matches[idx + 1].start()].count("\n") if idx + 1 < len(matches) else len(lines))
        raw_kind, name = match.group(1), match.group(2)
        kind = _normalize_kind(raw_kind)
        indent = len(lines[line_no - 1]) - len(lines[line_no - 1].lstrip()) if line_no <= len(lines) else 0
        while parent_stack and parent_stack[-1][1] >= indent:
            parent_stack.pop()
        parent = parent_stack[-1][0] if parent_stack and kind in {"function", "method"} else ""
        if parent and kind == "function":
            kind = "method"
        symbols.append(CodeSymbol(file_path, language, name, kind, line_no, max(end_line, line_no), _line_signature(lines, line_no), parent, imports, _text_refs(text)))
        if kind in {"class", "object", "interface"}:
            parent_stack.append((name, indent))
    return symbols


def _normalize_kind(kind: str) -> str:
    kind = kind.replace("async ", "")
    if kind == "fun" or kind == "def":
        return "function"
    if kind == "typealias":
        return "type"
    if kind not in _SYMBOL_KINDS:
        return "constant" if kind in {"const", "let", "var"} else kind
    return kind


def _text_refs(text: str) -> list[str]:
    return sorted(set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", text)))[:100]
