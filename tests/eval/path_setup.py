"""Path helpers for AndesCode eval modules."""

from __future__ import annotations

import sys
from pathlib import Path


def find_repo_root(start_file: str) -> Path:
    """Find the repository root by locating the directory containing indexer.py."""
    start = Path(start_file).resolve()
    for candidate in (start.parent, *start.parents):
        if (candidate / "indexer.py").exists():
            return candidate
    raise RuntimeError(
        f"Could not locate repository root from '{start}'. "
        "Expected an ancestor directory containing indexer.py."
    )


def prepend_sys_path(path: Path) -> None:
    """Prepend path to sys.path once, preserving precedence."""
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)
