"""Authoritative file index integrity validation and targeted repair orchestration."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable

from .source_of_truth import rank_authoritative_paths

INTEGRITY_HEALTHY = "healthy"
INTEGRITY_DEGRADED = "degraded"
INTEGRITY_STALE = "stale"
INTEGRITY_REPAIRING = "repairing"
INTEGRITY_WARNING_PARTIAL_RESULTS = "Index incomplete — results may be partial"

REASON_DISCOVERED_NOT_EMBEDDED = "discovered_not_embedded"
REASON_EMBEDDED_NOT_RETRIEVABLE = "embedded_not_retrievable"
REASON_RETRIEVABLE_BUT_INCOMPLETE = "retrievable_but_incomplete"
REASON_WORKSPACE_HASH_MISMATCH = "workspace_hash_mismatch"
REASON_MISSING_ON_DISK = "missing_on_disk"
REASON_REPAIR_FAILED = "repair_failed"


@dataclass
class FileIntegrityStatus:
    path: str
    status: str
    reasons: list[str]
    repair_attempted: bool = False
    repair_succeeded: bool = False
    discovered: bool = True
    embedded: bool = False
    retrievable: bool = False
    expected_chunks: int | None = None
    retrieved_chunks: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IntegrityReport:
    overall_status: str
    files: list[FileIntegrityStatus]
    repair_ran: bool = False
    repair_succeeded: bool = False

    def to_dict(self) -> dict:
        reason_codes = sorted({reason for f in self.files for reason in f.reasons})
        return {
            "overall_status": self.overall_status,
            "repair_ran": self.repair_ran,
            "repair_succeeded": self.repair_succeeded,
            "failing_files": [f.path for f in self.files if f.status != INTEGRITY_HEALTHY],
            "reason_codes": reason_codes,
            "files": [f.to_dict() for f in self.files],
        }



def authoritative_paths_from_workspace(workspace: dict, query: str = "", intent: str = "") -> list[str]:
    manifests = workspace.get("manifests", []) if workspace else []
    config_files = workspace.get("config_graph", {}).get("config_files", []) if workspace else []
    paths = sorted({p for p in (manifests + config_files) if p})
    return rank_authoritative_paths(paths, query=query, intent=intent)



def _assess_chunks(chunks: list[dict], expected_chunk_count: int | None) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not chunks:
        return False, [REASON_EMBEDDED_NOT_RETRIEVABLE]

    if any(not (c.get("content") or "").strip() for c in chunks):
        reasons.append(REASON_RETRIEVABLE_BUT_INCOMPLETE)

    lines: list[int] = []
    for c in chunks:
        try:
            lines.append(int(c.get("line", 0) or 0))
        except Exception:
            lines.append(0)
    sorted_lines = sorted(lines)
    if lines != sorted_lines or len(sorted_lines) != len(set(sorted_lines)):
        reasons.append(REASON_RETRIEVABLE_BUT_INCOMPLETE)

    if expected_chunk_count is not None and len(chunks) < expected_chunk_count:
        reasons.append(REASON_RETRIEVABLE_BUT_INCOMPLETE)

    return len(reasons) == 0, sorted(set(reasons))



def validate_authoritative_integrity(
    workspace: dict,
    hash_state: dict,
    fetch_exact_file: Callable[[str, int], list[dict]],
    file_hash_lookup: Callable[[str], str | None] | None = None,
    file_exists_lookup: Callable[[str], bool] | None = None,
    expected_chunk_count_lookup: Callable[[str], int | None] | None = None,
    candidate_paths: list[str] | None = None,
    max_files: int = 24,
    validate_expected_chunks: bool = True,
) -> IntegrityReport:
    paths = candidate_paths or authoritative_paths_from_workspace(workspace)
    statuses: list[FileIntegrityStatus] = []

    for path in paths[:max_files]:
        embedded_hash = (hash_state or {}).get(path)
        embedded = embedded_hash is not None
        reasons: list[str] = []

        if not embedded:
            reasons.append(REASON_DISCOVERED_NOT_EMBEDDED)

        if file_exists_lookup and not file_exists_lookup(path):
            reasons.append(REASON_MISSING_ON_DISK)
        if file_hash_lookup:
            current_hash = file_hash_lookup(path)
            if current_hash is None and not file_exists_lookup:
                reasons.append(REASON_MISSING_ON_DISK)
            elif embedded and current_hash is not None and current_hash != embedded_hash:
                reasons.append(REASON_WORKSPACE_HASH_MISMATCH)

        chunks = fetch_exact_file(path, 120)
        retrievable = bool(chunks)
        expected_chunks = None
        if validate_expected_chunks and expected_chunk_count_lookup:
            expected_chunks = expected_chunk_count_lookup(path)

        ok_chunks, chunk_reasons = _assess_chunks(chunks, expected_chunks)
        if not ok_chunks and embedded:
            reasons.extend(chunk_reasons)

        status = INTEGRITY_HEALTHY if not reasons else (
            INTEGRITY_STALE if any(r in {REASON_DISCOVERED_NOT_EMBEDDED, REASON_WORKSPACE_HASH_MISMATCH, REASON_MISSING_ON_DISK} for r in reasons)
            else INTEGRITY_DEGRADED
        )
        statuses.append(
            FileIntegrityStatus(
                path=path,
                status=status,
                reasons=sorted(set(reasons)),
                discovered=True,
                embedded=embedded,
                retrievable=retrievable,
                expected_chunks=expected_chunks,
                retrieved_chunks=len(chunks),
            )
        )

    overall = INTEGRITY_HEALTHY
    if any(f.status == INTEGRITY_STALE for f in statuses):
        overall = INTEGRITY_STALE
    elif any(f.status == INTEGRITY_DEGRADED for f in statuses):
        overall = INTEGRITY_DEGRADED

    return IntegrityReport(overall_status=overall, files=statuses)


def lightweight_integrity_probe(
    workspace: dict,
    hash_state: dict,
    fetch_exact_file: Callable[[str, int], list[dict]],
    file_hash_lookup: Callable[[str], str | None] | None = None,
    file_exists_lookup: Callable[[str], bool] | None = None,
    candidate_paths: list[str] | None = None,
    max_files: int = 6,
) -> dict:
    """
    Cheap startup/open-time integrity signal.
    - validates only a short ranked subset
    - skips expensive expected-chunk recomputation
    - does not trigger repair
    """
    report = validate_authoritative_integrity(
        workspace=workspace,
        hash_state=hash_state,
        fetch_exact_file=fetch_exact_file,
        file_hash_lookup=file_hash_lookup,
        file_exists_lookup=file_exists_lookup,
        expected_chunk_count_lookup=None,
        candidate_paths=candidate_paths,
        max_files=max_files,
        validate_expected_chunks=False,
    )
    warning_active = report.overall_status != INTEGRITY_HEALTHY
    failing = [f.path for f in report.files if f.status != INTEGRITY_HEALTHY]
    reason_codes = sorted({r for f in report.files for r in f.reasons})
    return {
        "overall_status": report.overall_status,
        "warning_active": warning_active,
        "warning_message": INTEGRITY_WARNING_PARTIAL_RESULTS if warning_active else "",
        "failing_paths": failing,
        "reason_codes": reason_codes,
        "checked_paths": [f.path for f in report.files],
        "probe_mode": "lightweight",
    }



def repair_authoritative_integrity(
    report: IntegrityReport,
    repair_paths_fn: Callable[[list[str]], bool],
    revalidate_fn: Callable[[list[str]], IntegrityReport],
) -> IntegrityReport:
    failing_paths = [f.path for f in report.files if f.status != INTEGRITY_HEALTHY]
    if not failing_paths:
        return report

    for f in report.files:
        if f.path in failing_paths:
            f.repair_attempted = True

    repair_ok = repair_paths_fn(failing_paths)
    repaired = revalidate_fn(failing_paths)
    repaired.repair_ran = True
    repaired.repair_succeeded = repair_ok and repaired.overall_status == INTEGRITY_HEALTHY

    if not repaired.repair_succeeded:
        for f in repaired.files:
            if f.status != INTEGRITY_HEALTHY and REASON_REPAIR_FAILED not in f.reasons:
                f.reasons.append(REASON_REPAIR_FAILED)
    for f in repaired.files:
        if f.path in failing_paths:
            f.repair_attempted = True
            f.repair_succeeded = f.status == INTEGRITY_HEALTHY
    return repaired


def prune_missing_on_disk_hashes(hash_state: dict, report: IntegrityReport) -> tuple[dict, list[str]]:
    if not hash_state:
        return {}, []
    missing_paths = sorted(
        {
            f.path
            for f in report.files
            if REASON_MISSING_ON_DISK in f.reasons and hash_state.get(f.path) is not None
        }
    )
    if not missing_paths:
        return dict(hash_state), []
    cleaned = {k: v for k, v in hash_state.items() if k not in set(missing_paths)}
    return cleaned, missing_paths


def select_healthy_authoritative_path(
    candidate_paths: list[str],
    validate_path_fn: Callable[[str], IntegrityReport],
    max_candidates: int = 6,
) -> tuple[str, list[dict]]:
    """
    Validate a small ranked shortlist and return the first healthy path.
    This keeps strict-answer gating narrow and avoids blocking on lower-ranked
    stale candidates once a healthy authoritative source is confirmed.
    """
    attempts: list[dict] = []
    for path in candidate_paths[:max_candidates]:
        report = validate_path_fn(path)
        report_dict = report.to_dict()
        report_dict["candidate"] = path
        attempts.append(report_dict)
        if report.overall_status == INTEGRITY_HEALTHY:
            return path, attempts
    return "", attempts
