import unittest
from pathlib import Path

from andes_cache.integrity import (
    INTEGRITY_DEGRADED,
    INTEGRITY_HEALTHY,
    INTEGRITY_STALE,
    REASON_DISCOVERED_NOT_EMBEDDED,
    REASON_EMBEDDED_NOT_RETRIEVABLE,
    validate_authoritative_integrity,
    repair_authoritative_integrity,
)


class TestAuthoritativeIntegrity(unittest.TestCase):
    def test_discovered_in_workspace_but_not_embedded(self):
        workspace = {"manifests": ["app/build.gradle"], "config_graph": {"config_files": []}}
        report = validate_authoritative_integrity(
            workspace=workspace,
            hash_state={},
            fetch_exact_file=lambda _p, _m: [],
        )
        self.assertEqual(report.overall_status, INTEGRITY_STALE)
        self.assertIn(REASON_DISCOVERED_NOT_EMBEDDED, report.files[0].reasons)

    def test_embedded_but_not_retrievable(self):
        workspace = {"manifests": ["package.json"], "config_graph": {"config_files": []}}
        report = validate_authoritative_integrity(
            workspace=workspace,
            hash_state={"package.json": "abc"},
            fetch_exact_file=lambda _p, _m: [],
        )
        self.assertEqual(report.overall_status, INTEGRITY_DEGRADED)
        self.assertIn(REASON_EMBEDDED_NOT_RETRIEVABLE, report.files[0].reasons)

    def test_targeted_repair_success(self):
        workspace = {"manifests": ["pyproject.toml"], "config_graph": {"config_files": []}}

        state = {"retrievable": False}

        def fetch(path, _max):
            if path == "pyproject.toml" and state["retrievable"]:
                return [{"content": "[tool.poetry]", "line": 0, "file": path}]
            return []

        initial = validate_authoritative_integrity(
            workspace=workspace,
            hash_state={"pyproject.toml": "h"},
            fetch_exact_file=fetch,
            expected_chunk_count_lookup=lambda _p: 1,
        )

        def repair(paths):
            self.assertEqual(paths, ["pyproject.toml"])
            state["retrievable"] = True
            return True

        repaired = repair_authoritative_integrity(
            initial,
            repair_paths_fn=repair,
            revalidate_fn=lambda paths: validate_authoritative_integrity(
                workspace=workspace,
                hash_state={"pyproject.toml": "h"},
                fetch_exact_file=fetch,
                expected_chunk_count_lookup=lambda _p: 1,
                candidate_paths=paths,
            ),
        )
        self.assertEqual(repaired.overall_status, INTEGRITY_HEALTHY)
        self.assertTrue(repaired.repair_succeeded)


class TestIntegrationGuardrails(unittest.TestCase):
    def test_strict_source_truth_gated_by_integrity(self):
        src = Path("indexer.py").read_text(encoding="utf-8")
        self.assertIn("integrity_report = _validate_and_repair_authoritative_integrity(", src)
        self.assertIn("Source-of-Truth Index Integrity Limitation", src)

    def test_no_full_rebuild_dependency_in_targeted_repair(self):
        src = Path("indexer.py").read_text(encoding="utf-8")
        start = src.find("def _validate_and_repair_authoritative_integrity(")
        end = src.find("def _save_hashes(", start)
        self.assertGreater(start, -1)
        self.assertGreater(end, start)
        segment = src[start:end]
        self.assertIn("_repair_index_paths", segment)
        self.assertNotIn("delete_collection", segment)

    def test_integrity_checks_cover_multi_ecosystem_authoritative_files(self):
        workspace = {
            "manifests": ["android/app/build.gradle", "services/api/package.json"],
            "config_graph": {"config_files": []},
        }

        def fetch(path, _max):
            if path.endswith("build.gradle"):
                return [{"content": "plugins {}", "line": 0, "file": path}]
            if path.endswith("package.json"):
                return [{"content": "{\"name\": \"api\"}", "line": 0, "file": path}]
            return []

        report = validate_authoritative_integrity(
            workspace=workspace,
            hash_state={"android/app/build.gradle": "h1", "services/api/package.json": "h2"},
            fetch_exact_file=fetch,
            expected_chunk_count_lookup=lambda _p: 1,
        )
        self.assertEqual(report.overall_status, INTEGRITY_HEALTHY)


if __name__ == "__main__":
    unittest.main(verbosity=2)
