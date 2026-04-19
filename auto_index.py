import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

TEMP_FILE_PREFIXES = (".#", "~$")
TEMP_FILE_SUFFIXES = (
    "~", ".tmp", ".temp", ".swp", ".swo", ".swx", ".bak", ".orig", ".lock", ".DS_Store"
)


@dataclass
class ChangeBatch:
    changed_paths: set[str]
    deleted_paths: set[str]

    @property
    def count(self) -> int:
        return len(self.changed_paths) + len(self.deleted_paths)


class AutoIndexManager:
    """Polling-based filesystem watcher with debounced auto-index orchestration."""

    def __init__(
        self,
        *,
        snapshot_fn: Callable[[Path], dict[str, str]],
        run_index_fn: Callable[[str, ChangeBatch], bool],
        status_logger: Callable[[str], None] | None = None,
        debounce_seconds: float = 2.0,
        poll_interval: float = 1.0,
        enabled: bool = True,
    ):
        self.snapshot_fn = snapshot_fn
        self.run_index_fn = run_index_fn
        self.status_logger = status_logger or (lambda _msg: None)
        self.debounce_seconds = debounce_seconds
        self.poll_interval = poll_interval
        self.enabled = enabled

        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._indexed_root: str | None = None
        self._snapshot: dict[str, str] = {}

        self._pending_changed: set[str] = set()
        self._pending_deleted: set[str] = set()
        self._last_change_at: float | None = None

        self._auto_index_in_progress = False
        self._rerun_requested = False
        self._last_auto_index_at: str | None = None
        self._watcher_status = "disabled" if not enabled else "idle"

    @staticmethod
    def env_enabled() -> bool:
        return os.getenv("ANDESCODE_AUTO_INDEX", "1").strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def is_relevant_project_path(
        rel_path: str,
        *,
        supported_suffixes: set[str],
        authoritative_basenames: set[str],
        skip_dirs: set[str],
    ) -> bool:
        """
        Relevant means either:
        - source code file with a supported extension, OR
        - authoritative/build/config/dependency file by canonical basename.
        """
        parts = Path(rel_path).parts
        if any(part in skip_dirs for part in parts):
            return False
        name = Path(rel_path).name
        if name.startswith(TEMP_FILE_PREFIXES):
            return False
        if any(name.endswith(suffix) for suffix in TEMP_FILE_SUFFIXES):
            return False
        if name.startswith(".") and name not in {".env", ".editorconfig"}:
            return False
        if name in authoritative_basenames:
            return True
        return Path(rel_path).suffix in supported_suffixes

    @staticmethod
    def is_relevant_path(rel_path: str, *, supported_suffixes: set[str], skip_dirs: set[str]) -> bool:
        """Backward-compatible wrapper for older tests/callers."""
        return AutoIndexManager.is_relevant_project_path(
            rel_path,
            supported_suffixes=supported_suffixes,
            authoritative_basenames=set(),
            skip_dirs=skip_dirs,
        )

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self.enabled = enabled
            if not enabled:
                self._watcher_status = "disabled"
                self._pending_changed.clear()
                self._pending_deleted.clear()
                self._last_change_at = None

    def start_for_root(self, root: str) -> None:
        root_path = str(Path(root).resolve())
        with self._lock:
            if not self.enabled:
                self._indexed_root = root_path
                self._watcher_status = "disabled"
                return
            if self._thread and self._thread.is_alive() and self._indexed_root == root_path:
                self._watcher_status = "watching"
                return

        self.stop()

        with self._lock:
            self._indexed_root = root_path
            self._snapshot = self.snapshot_fn(Path(root_path))
            self._pending_changed.clear()
            self._pending_deleted.clear()
            self._last_change_at = None
            self._stop_event = threading.Event()
            self._watcher_status = "watching"
            self._thread = threading.Thread(target=self._watch_loop, daemon=True)
            self._thread.start()
            self.status_logger(f"Watching {root_path} for file changes")

    def stop(self) -> None:
        thread = None
        stop_event = None
        with self._lock:
            thread = self._thread
            stop_event = self._stop_event
            self._thread = None
            self._stop_event = None
            self._snapshot = {}
            self._pending_changed.clear()
            self._pending_deleted.clear()
            self._last_change_at = None
            self._indexed_root = None
            if self.enabled:
                self._watcher_status = "idle"

        if stop_event:
            stop_event.set()
        if thread and thread.is_alive():
            thread.join(timeout=2)

    def request_rerun_if_busy(self) -> None:
        with self._lock:
            self._rerun_requested = True

    def notify_auto_run_start(self) -> None:
        with self._lock:
            self._auto_index_in_progress = True
            self._watcher_status = "indexing"

    def notify_auto_run_end(self) -> bool:
        with self._lock:
            self._auto_index_in_progress = False
            self._last_auto_index_at = datetime.now(timezone.utc).isoformat()
            rerun = self._rerun_requested
            self._rerun_requested = False
            self._watcher_status = "watching" if self.enabled and self._indexed_root else "idle"
            return rerun

    def status(self) -> dict:
        with self._lock:
            return {
                "indexed_root": self._indexed_root,
                "watcher_enabled": self.enabled,
                "watcher_status": self._watcher_status,
                "auto_index_in_progress": self._auto_index_in_progress,
                "last_auto_index_at": self._last_auto_index_at,
                "pending_change_count": len(self._pending_changed) + len(self._pending_deleted),
            }

    def _watch_loop(self) -> None:
        while True:
            with self._lock:
                stop_event = self._stop_event
                root = self._indexed_root
            if not stop_event or stop_event.is_set() or not root:
                return

            try:
                current = self.snapshot_fn(Path(root))
                with self._lock:
                    previous = self._snapshot
                    changed = {p for p, h in current.items() if previous.get(p) != h}
                    deleted = set(previous.keys()) - set(current.keys())
                    if changed or deleted:
                        self._pending_changed |= changed
                        self._pending_deleted |= deleted
                        self._last_change_at = time.monotonic()
                        self._watcher_status = "changes_detected"
                    self._snapshot = current

                    last_change_at = self._last_change_at
                    debounce_ready = bool(last_change_at) and (time.monotonic() - last_change_at) >= self.debounce_seconds
                    if debounce_ready and (self._pending_changed or self._pending_deleted):
                        batch = ChangeBatch(
                            changed_paths=set(self._pending_changed),
                            deleted_paths=set(self._pending_deleted),
                        )
                        self._pending_changed.clear()
                        self._pending_deleted.clear()
                        self._last_change_at = None
                    else:
                        batch = None
            except Exception as exc:
                self.status_logger(f"Auto-index watcher error: {exc}")
                time.sleep(self.poll_interval)
                continue

            if batch:
                if batch.deleted_paths:
                    self.status_logger(f"{batch.count} file changes detected (including deletion); refreshing index")
                else:
                    self.status_logger(f"{batch.count} file changes detected; refreshing index")
                started = self.run_index_fn(root, batch)
                if not started:
                    with self._lock:
                        self._rerun_requested = True

            time.sleep(self.poll_interval)
