import unittest
from pathlib import Path

from andes_cache.integrity import (
    INTEGRITY_DEGRADED,
    INTEGRITY_HEALTHY,
    INTEGRITY_STALE,
    REASON_DISCOVERED_NOT_EMBEDDED,
    REASON_EMBEDDED_NOT_RETRIEVABLE,
    REASON_MISSING_ON_DISK,
    REASON_REPAIR_FAILED,
    validate_authoritative_integrity,
    repair_authoritative_integrity,
    prune_missing_on_disk_hashes,
    select_healthy_authoritative_path,
    IntegrityReport,
    FileIntegrityStatus,
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

    def test_missing_file_is_reported_explicitly(self):
        workspace = {"manifests": ["deleted/build.gradle"], "config_graph": {"config_files": []}}
        report = validate_authoritative_integrity(
            workspace=workspace,
            hash_state={"deleted/build.gradle": "h"},
            fetch_exact_file=lambda _p, _m: [],
            file_hash_lookup=lambda _p: None,
        )
        self.assertEqual(report.overall_status, INTEGRITY_STALE)
        self.assertIn(REASON_MISSING_ON_DISK, report.files[0].reasons)

    def test_repair_failure_marks_reason_code_and_keeps_status(self):
        workspace = {"manifests": ["package.json"], "config_graph": {"config_files": []}}
        initial = validate_authoritative_integrity(
            workspace=workspace,
            hash_state={"package.json": "h"},
            fetch_exact_file=lambda _p, _m: [],
            expected_chunk_count_lookup=lambda _p: 1,
        )
        repaired = repair_authoritative_integrity(
            initial,
            repair_paths_fn=lambda _paths: False,
            revalidate_fn=lambda paths: validate_authoritative_integrity(
                workspace=workspace,
                hash_state={"package.json": "h"},
                fetch_exact_file=lambda _p, _m: [],
                expected_chunk_count_lookup=lambda _p: 1,
                candidate_paths=paths,
            ),
        )
        self.assertFalse(repaired.repair_succeeded)
        self.assertIn(REASON_REPAIR_FAILED, repaired.files[0].reasons)
        self.assertEqual(repaired.overall_status, INTEGRITY_DEGRADED)

    def test_partial_repair_failure_consistency(self):
        workspace = {"manifests": ["good.gradle", "bad.gradle"], "config_graph": {"config_files": []}}
        state = {"good": False, "bad": False}

        def fetch(path, _max):
            if path == "good.gradle" and state["good"]:
                return [{"content": "plugins {}", "line": 0, "file": path}]
            if path == "bad.gradle" and state["bad"]:
                return [{"content": "plugins {}", "line": 0, "file": path}]
            return []

        initial = validate_authoritative_integrity(
            workspace=workspace,
            hash_state={"good.gradle": "hg", "bad.gradle": "hb"},
            fetch_exact_file=fetch,
            expected_chunk_count_lookup=lambda _p: 1,
        )

        def repair(paths):
            self.assertEqual(sorted(paths), ["bad.gradle", "good.gradle"])
            state["good"] = True
            return False

        repaired = repair_authoritative_integrity(
            initial,
            repair_paths_fn=repair,
            revalidate_fn=lambda paths: validate_authoritative_integrity(
                workspace=workspace,
                hash_state={"good.gradle": "hg", "bad.gradle": "hb"},
                fetch_exact_file=fetch,
                expected_chunk_count_lookup=lambda _p: 1,
                candidate_paths=paths,
            ),
        )
        self.assertFalse(repaired.repair_succeeded)
        by_path = {f.path: f for f in repaired.files}
        self.assertEqual(by_path["good.gradle"].status, INTEGRITY_HEALTHY)
        self.assertEqual(by_path["bad.gradle"].status, INTEGRITY_DEGRADED)
        self.assertIn(REASON_REPAIR_FAILED, by_path["bad.gradle"].reasons)

    def test_delete_and_add_rename_style_scenario(self):
        workspace = {"manifests": ["new/package.json"], "config_graph": {"config_files": ["old/package.json"]}}
        fetch_map = {
            "new/package.json": [{"content": "{\"name\":\"new\"}", "line": 0, "file": "new/package.json"}],
            "old/package.json": [],
        }
        report = validate_authoritative_integrity(
            workspace=workspace,
            hash_state={"old/package.json": "old-hash"},
            fetch_exact_file=lambda p, _m: fetch_map.get(p, []),
            file_hash_lookup=lambda p: None if p == "old/package.json" else "new-hash",
            expected_chunk_count_lookup=lambda _p: 1,
        )
        by_path = {f.path: f for f in report.files}
        self.assertEqual(by_path["new/package.json"].status, INTEGRITY_STALE)
        self.assertIn(REASON_DISCOVERED_NOT_EMBEDDED, by_path["new/package.json"].reasons)
        self.assertEqual(by_path["old/package.json"].status, INTEGRITY_STALE)
        self.assertIn(REASON_MISSING_ON_DISK, by_path["old/package.json"].reasons)

    def test_report_contains_aggregated_reason_codes(self):
        report = IntegrityReport(
            overall_status=INTEGRITY_STALE,
            files=[
                FileIntegrityStatus(path="a", status=INTEGRITY_STALE, reasons=[REASON_MISSING_ON_DISK]),
                FileIntegrityStatus(path="b", status=INTEGRITY_DEGRADED, reasons=[REASON_EMBEDDED_NOT_RETRIEVABLE]),
            ],
        )
        payload = report.to_dict()
        self.assertEqual(
            payload["reason_codes"],
            sorted([REASON_EMBEDDED_NOT_RETRIEVABLE, REASON_MISSING_ON_DISK]),
        )

    def test_missing_file_cleanup_prunes_stale_hash_entry(self):
        report = IntegrityReport(
            overall_status=INTEGRITY_STALE,
            files=[
                FileIntegrityStatus(
                    path="removed/AndroidManifest.xml",
                    status=INTEGRITY_STALE,
                    reasons=[REASON_MISSING_ON_DISK],
                )
            ],
        )
        cleaned, removed = prune_missing_on_disk_hashes(
            {"removed/AndroidManifest.xml": "old", "__fingerprint__": "fp"},
            report,
        )
        self.assertEqual(removed, ["removed/AndroidManifest.xml"])
        self.assertNotIn("removed/AndroidManifest.xml", cleaned)
        self.assertEqual(cleaned["__fingerprint__"], "fp")


class TestIntegrationGuardrails(unittest.TestCase):
    def test_selector_allows_healthy_top_ranked_path_with_stale_lower_candidate(self):
        calls = []

        def validate_path(path: str):
            calls.append(path)
            if path == "app/build.gradle":
                return IntegrityReport(
                    overall_status=INTEGRITY_HEALTHY,
                    files=[FileIntegrityStatus(path=path, status=INTEGRITY_HEALTHY, reasons=[])],
                )
            return IntegrityReport(
                overall_status=INTEGRITY_STALE,
                files=[FileIntegrityStatus(path=path, status=INTEGRITY_STALE, reasons=[REASON_DISCOVERED_NOT_EMBEDDED])],
            )

        selected, attempts = select_healthy_authoritative_path(
            ["app/build.gradle", "services/api/package.json"],
            validate_path_fn=validate_path,
        )
        self.assertEqual(selected, "app/build.gradle")
        self.assertEqual(calls, ["app/build.gradle"])
        self.assertEqual(attempts[0]["overall_status"], INTEGRITY_HEALTHY)

    def test_strict_source_truth_gated_by_integrity(self):
        src = Path("indexer.py").read_text(encoding="utf-8")
        self.assertIn("selected_healthy_path, integrity_attempts = select_healthy_authoritative_path(", src)
        self.assertIn("Source-of-Truth Index Integrity Limitation", src)
        self.assertIn("priority_files = [selected_healthy_path]", src)

    def test_targeted_repair_is_rollback_safe_by_design(self):
        src = Path("indexer.py").read_text(encoding="utf-8")
        start = src.find("def _validate_and_repair_authoritative_integrity(")
        end = src.find("def _save_hashes(", start)
        self.assertGreater(start, -1)
        self.assertGreater(end, start)
        segment = src[start:end]
        self.assertIn("_repair_index_paths", segment)
        self.assertNotIn("delete_collection", segment)
        self.assertIn("stale_ids = sorted(set(prepared[\"previous_ids\"]) - set(prepared[\"new_ids\"]))", src)
        self.assertIn("Integrity rollback failed while restoring previous vectors", src)

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
