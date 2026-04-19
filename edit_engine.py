import difflib
import hashlib
import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

HASH_STORE = Path(__file__).parent / "index" / ".file_hashes.json"


@dataclass(frozen=True)
class EditOperation:
    file_path: str
    old_content: str
    new_content: str


@dataclass(frozen=True)
class ApplyResult:
    success: bool
    error: str | None = None


@dataclass(frozen=True)
class ApplyAttemptLog:
    file_path: str
    exists: bool
    indexed: bool
    hash_match: bool
    content_match: bool
    message: str


def generate_diff_preview(old_text: str, new_text: str, file_path: str = "file") -> str:
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm="",
        )
    )


class FileEditEngine:
    def __init__(
        self,
        hash_store_path: Path = HASH_STORE,
        reindex_file: Callable[[Path, str], bool] | None = None,
    ):
        self.hash_store_path = hash_store_path
        self._reindex_file = reindex_file or self._default_reindex_file

    def apply_edit_operation(self, edit: EditOperation) -> ApplyResult:
        hashes = self._load_hashes()
        root_path = self._repo_root(hashes)
        if root_path is None:
            return self._fail(
                edit.file_path,
                exists=False,
                indexed=False,
                hash_match=False,
                content_match=False,
                message="Missing indexed repo root metadata (__root__)",
            )
        target_path = self._resolve_target_path(root_path, edit.file_path)
        if target_path is None:
            return self._fail(
                edit.file_path,
                exists=False,
                indexed=False,
                hash_match=False,
                content_match=False,
                message="Path escapes repository root",
            )
        rel_path = self._relative_to_root(root_path, target_path)
        if rel_path is None:
            return self._fail(
                edit.file_path,
                exists=False,
                indexed=False,
                hash_match=False,
                content_match=False,
                message="Path escapes repository root",
            )

        logging.info("[edit-apply] file=%s step=exists", rel_path)
        if not target_path.exists() or not target_path.is_file():
            return self._fail(rel_path, exists=False, indexed=False, hash_match=False, content_match=False, message="File missing on disk")

        logging.info("[edit-apply] file=%s step=indexed", rel_path)
        stored_hash = hashes.get(rel_path)
        if stored_hash is None:
            return self._fail(rel_path, exists=True, indexed=False, hash_match=False, content_match=False, message="File is not indexed")

        current_hash = self._file_hash(target_path)
        logging.info("[edit-apply] file=%s step=hash_match", rel_path)
        if current_hash != stored_hash:
            return self._fail(
                rel_path,
                exists=True,
                indexed=True,
                hash_match=False,
                content_match=False,
                message="Hash mismatch: file changed since index (stale context)",
            )

        text = target_path.read_text(encoding="utf-8")
        occurrences = text.count(edit.old_content)
        logging.info("[edit-apply] file=%s step=content_match occurrences=%s", rel_path, occurrences)
        if occurrences == 0:
            return self._fail(
                rel_path,
                exists=True,
                indexed=True,
                hash_match=True,
                content_match=False,
                message="old_content not found exactly in file",
            )
        if occurrences > 1:
            return self._fail(
                rel_path,
                exists=True,
                indexed=True,
                hash_match=True,
                content_match=False,
                message="old_content is ambiguous (multiple exact matches)",
            )

        updated = text.replace(edit.old_content, edit.new_content, 1)
        self._atomic_write(target_path, updated)

        if not self._reindex_file(root_path, rel_path):
            logging.error("[edit-apply] file=%s step=reindex status=failed", rel_path)
            return ApplyResult(success=False, error="Edit applied but re-index failed — index may be inconsistent")

        logging.info("[edit-apply] file=%s status=success", rel_path)
        return ApplyResult(success=True, error=None)

    def _fail(self, file_path: str, exists: bool, indexed: bool, hash_match: bool, content_match: bool, message: str) -> ApplyResult:
        log = ApplyAttemptLog(
            file_path=file_path,
            exists=exists,
            indexed=indexed,
            hash_match=hash_match,
            content_match=content_match,
            message=message,
        )
        logging.error("[edit-apply] %s", log)
        return ApplyResult(success=False, error=message)

    def _load_hashes(self) -> dict:
        if not self.hash_store_path.exists():
            return {}
        try:
            data = json.loads(self.hash_store_path.read_text())
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _repo_root(self, hashes: dict) -> Path | None:
        root = hashes.get("__root__")
        if root:
            return Path(root).resolve()
        return None

    def _resolve_target_path(self, root_path: Path, file_path: str) -> Path | None:
        path = Path(file_path)
        if path.is_absolute():
            resolved = path.resolve()
        else:
            resolved = (root_path / path).resolve()

        try:
            resolved.relative_to(root_path)
            return resolved
        except ValueError:
            return None

    def _relative_to_root(self, root_path: Path, file_path: Path) -> str | None:
        try:
            return str(file_path.relative_to(root_path))
        except ValueError:
            return None

    def _file_hash(self, path: Path) -> str:
        return hashlib.md5(path.read_bytes()).hexdigest()

    def _atomic_write(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as tmp:
            tmp.write(content)
            temp_path = Path(tmp.name)
        temp_path.replace(path)

    def _default_reindex_file(self, root_path: Path, rel_path: str) -> bool:
        try:
            import indexer

            return bool(indexer._repair_index_paths(root_path, [rel_path]))
        except Exception as exc:
            logging.error("[edit-apply] file=%s step=reindex exception=%s", rel_path, exc)
            return False
