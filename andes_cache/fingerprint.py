"""Repository fingerprinting for strict cache isolation and invalidation."""

from __future__ import annotations

from pathlib import Path

from .keys import stable_hash


class RepoFingerprinter:
    @staticmethod
    def build(
        root_path: Path,
        file_hashes: dict,
        *,
        index_version: str,
        parser_version: str,
        prompt_template_version: str,
        retrieval_policy_version: str,
    ) -> str:
        filtered = {
            k: v
            for k, v in file_hashes.items()
            if not k.startswith("__")
        }
        payload = {
            "root": str(Path(root_path).resolve()),
            "index_version": index_version,
            "parser_version": parser_version,
            "prompt_template_version": prompt_template_version,
            "retrieval_policy_version": retrieval_policy_version,
            "files": filtered,
        }
        return stable_hash(payload)
