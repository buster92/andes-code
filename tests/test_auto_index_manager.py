import tempfile
import time
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auto_index import AutoIndexManager, ChangeBatch


class TestAutoIndexManager(unittest.TestCase):
    def test_is_relevant_project_path_includes_authoritative_files(self):
        supported = {".py", ".ts"}
        authoritative = {
            "package.json",
            "pyproject.toml",
            "Cargo.toml",
            "go.mod",
            "build.gradle.kts",
            "settings.gradle.kts",
            "docker-compose.yml",
            "pnpm-workspace.yaml",
            "Dockerfile",
        }
        skip_dirs = {".git", "node_modules"}

        self.assertTrue(
            AutoIndexManager.is_relevant_project_path(
                "apps/web/package.json",
                supported_suffixes=supported,
                authoritative_basenames=authoritative,
                skip_dirs=skip_dirs,
            )
        )
        self.assertTrue(
            AutoIndexManager.is_relevant_project_path(
                "services/api/pyproject.toml",
                supported_suffixes=supported,
                authoritative_basenames=authoritative,
                skip_dirs=skip_dirs,
            )
        )
        self.assertTrue(
            AutoIndexManager.is_relevant_project_path(
                "rust/Cargo.toml",
                supported_suffixes=supported,
                authoritative_basenames=authoritative,
                skip_dirs=skip_dirs,
            )
        )
        self.assertTrue(
            AutoIndexManager.is_relevant_project_path(
                "go/go.mod",
                supported_suffixes=supported,
                authoritative_basenames=authoritative,
                skip_dirs=skip_dirs,
            )
        )
        self.assertTrue(
            AutoIndexManager.is_relevant_project_path(
                "android/build.gradle.kts",
                supported_suffixes=supported,
                authoritative_basenames=authoritative,
                skip_dirs=skip_dirs,
            )
        )
        self.assertTrue(
            AutoIndexManager.is_relevant_project_path(
                "android/settings.gradle.kts",
                supported_suffixes=supported,
                authoritative_basenames=authoritative,
                skip_dirs=skip_dirs,
            )
        )
        self.assertTrue(
            AutoIndexManager.is_relevant_project_path(
                "infra/docker-compose.yml",
                supported_suffixes=supported,
                authoritative_basenames=authoritative,
                skip_dirs=skip_dirs,
            )
        )
        self.assertTrue(
            AutoIndexManager.is_relevant_project_path(
                "monorepo/pnpm-workspace.yaml",
                supported_suffixes=supported,
                authoritative_basenames=authoritative,
                skip_dirs=skip_dirs,
            )
        )
        self.assertTrue(
            AutoIndexManager.is_relevant_project_path(
                "infra/Dockerfile",
                supported_suffixes=supported,
                authoritative_basenames=authoritative,
                skip_dirs=skip_dirs,
            )
        )

    def test_is_relevant_path_ignores_temp_and_skip_dirs(self):
        supported = {".py", ".ts"}
        authoritative = {"package.json", "Dockerfile"}
        skip_dirs = {".git", "node_modules"}
        self.assertTrue(
            AutoIndexManager.is_relevant_project_path(
                "src/app.py",
                supported_suffixes=supported,
                authoritative_basenames=authoritative,
                skip_dirs=skip_dirs,
            )
        )
        self.assertFalse(
            AutoIndexManager.is_relevant_project_path(
                "node_modules/package.json",
                supported_suffixes=supported,
                authoritative_basenames=authoritative,
                skip_dirs=skip_dirs,
            )
        )
        self.assertFalse(
            AutoIndexManager.is_relevant_project_path(
                "src/app.py.swp",
                supported_suffixes=supported,
                authoritative_basenames=authoritative,
                skip_dirs=skip_dirs,
            )
        )
        self.assertFalse(
            AutoIndexManager.is_relevant_project_path(
                "src/.DS_Store",
                supported_suffixes=supported,
                authoritative_basenames=authoritative,
                skip_dirs=skip_dirs,
            )
        )
        self.assertFalse(
            AutoIndexManager.is_relevant_project_path(
                "src/.#Dockerfile",
                supported_suffixes=supported,
                authoritative_basenames=authoritative,
                skip_dirs=skip_dirs,
            )
        )

    def test_debounce_collapses_multiple_changes_into_one_run(self):
        state = {"a.py": "1"}
        runs = []

        def snapshot(_root: Path):
            return dict(state)

        def run_index(_root: str, batch: ChangeBatch) -> bool:
            runs.append(batch)
            return True

        mgr = AutoIndexManager(snapshot_fn=snapshot, run_index_fn=run_index, debounce_seconds=0.08, poll_interval=0.02)
        with tempfile.TemporaryDirectory() as tmp:
            mgr.start_for_root(tmp)
            time.sleep(0.05)
            state["a.py"] = "2"
            time.sleep(0.03)
            state["b.py"] = "1"
            time.sleep(0.2)
            mgr.stop()

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].count, 2)

    def test_delete_event_included_in_batch(self):
        state = {"a.py": "1", "b.py": "1"}
        runs = []

        def snapshot(_root: Path):
            return dict(state)

        def run_index(_root: str, batch: ChangeBatch) -> bool:
            runs.append(batch)
            return True

        mgr = AutoIndexManager(snapshot_fn=snapshot, run_index_fn=run_index, debounce_seconds=0.06, poll_interval=0.02)
        with tempfile.TemporaryDirectory() as tmp:
            mgr.start_for_root(tmp)
            time.sleep(0.05)
            del state["b.py"]
            time.sleep(0.15)
            mgr.stop()

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].deleted_paths, {"b.py"})

    def test_switch_root_stops_previous_watch_target(self):
        roots = {}
        runs = []

        def snapshot(root: Path):
            return dict(roots.get(str(root), {}))

        def run_index(root: str, batch: ChangeBatch) -> bool:
            runs.append((Path(root).name, batch.count))
            return True

        mgr = AutoIndexManager(snapshot_fn=snapshot, run_index_fn=run_index, debounce_seconds=0.06, poll_interval=0.02)
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            r1 = str(Path(tmp1).resolve())
            r2 = str(Path(tmp2).resolve())
            roots[r1] = {"a.py": "1"}
            roots[r2] = {"x.py": "1"}

            mgr.start_for_root(r1)
            time.sleep(0.05)
            mgr.start_for_root(r2)
            roots[r1]["a.py"] = "2"  # should be ignored now
            roots[r2]["x.py"] = "2"
            time.sleep(0.2)
            mgr.stop()

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0][0], Path(tmp2).name)

    def test_rerun_flag_coalesces_overlapping_requests(self):
        mgr = AutoIndexManager(snapshot_fn=lambda _root: {}, run_index_fn=lambda _root, _batch: True)
        mgr.notify_auto_run_start()
        mgr.request_rerun_if_busy()
        self.assertTrue(mgr.notify_auto_run_end())
        self.assertFalse(mgr.notify_auto_run_end())


if __name__ == "__main__":
    unittest.main(verbosity=2)
