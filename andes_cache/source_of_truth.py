"""Deterministic source-of-truth retrieval helpers for config/declaration questions."""

from __future__ import annotations

import re


def config_priority_files(intent: str, query: str, manifests: list[str], config_files: list[str]) -> list[str]:
    q = (query or "").lower()
    preferred: list[str] = []
    authoritative_paths = [
        p for p in (manifests + config_files) if p and _is_authoritative_candidate(p)
    ]
    query_hints = _query_path_hints(q)

    if any(k in q for k in ("permission", "permissions", "declared", "manifest")):
        preferred.extend([m for m in authoritative_paths if m.endswith("AndroidManifest.xml")])
        preferred.append("AndroidManifest.xml")

    if intent == "dependency_or_build_inventory":
        dep_files = [
            "build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts",
            "package.json", "requirements.txt", "pyproject.toml", "Cargo.toml",
            "go.mod", "pom.xml", "poetry.lock", "Pipfile", "Dockerfile", "docker-compose.yml",
            "Package.swift", ".entitlements", ".plist",
        ]
        for df in dep_files:
            matched = [m for m in authoritative_paths if m.endswith(df)]
            preferred.extend(_rank_by_query_hints(matched, query_hints))
            preferred.append(df)
    else:
        cfg_files = [
            "AndroidManifest.xml", "build.gradle", "build.gradle.kts",
            "settings.gradle", "settings.gradle.kts", ".env", "docker-compose.yml",
            "package.json", "pyproject.toml",
        ]
        for cf in cfg_files:
            matched = [m for m in authoritative_paths if m.endswith(cf)]
            preferred.extend(_rank_by_query_hints(matched, query_hints))
            preferred.append(cf)
        preferred.extend(_rank_by_query_hints(authoritative_paths, query_hints))

    seen = set()
    ordered = []
    for p in preferred:
        if p and p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


def summarize_declared_permissions(chunks: list[dict]) -> list[str]:
    perms = set()
    for c in chunks:
        if not c.get("file", "").endswith("AndroidManifest.xml"):
            continue
        for perm in re.findall(r'uses-permission[^>]*android:name="([^"]+)"', c.get("content", "")):
            perms.add(perm.strip())
    return sorted(perms)


def annotate_sources(chunks: list[dict], source_type: str, authority_level: str) -> list[dict]:
    for c in chunks:
        c.setdefault("source_type", source_type)
        c.setdefault("authority_level", authority_level)
    return chunks


def classify_source_type(path: str) -> str:
    p = (path or "").lower()
    if "manifest" in p:
        return "manifest"
    if any(x in p for x in ("gradle", "pom.xml", "dockerfile", "compose", "package.swift")):
        return "build_file"
    if any(x in p for x in ("requirements", "pyproject", "cargo.toml", "go.mod", "package.json", "pipfile", "poetry.lock")):
        return "dependency_file"
    if any(x in p for x in ("config", ".env", "entitlements", ".plist", "settings")):
        return "config_file"
    return "source_code"


def authority_level_for_source(intent: str, source_type: str) -> str:
    if source_type in {"manifest", "config_file"}:
        return "configured"
    if source_type in {"build_file", "dependency_file"}:
        return "declared" if intent == "dependency_or_build_inventory" else "configured"
    if source_type == "source_code":
        return "referenced"
    return "inferred"


def wants_runtime_usage(query: str) -> bool:
    q = (query or "").lower()
    return bool(
        re.search(
            r"\b(used at runtime|needed at runtime|required at runtime|runtime usage|referenced in code|checked in code|where is .* used)\b",
            q,
        )
    )


def missing_manifest_notice() -> dict:
    return {
        "content": (
            "# Manifest Availability\n"
            "- No AndroidManifest.xml was found in indexed source-of-truth files.\n"
            "- Declared permissions cannot be confirmed.\n"
            "- Any fallback findings below are references/inferences, not declarations."
        ),
        "file": "__manifest_status__",
        "language": "meta",
        "source_type": "inferred",
        "authority_level": "inferred",
    }


def _is_authoritative_candidate(path: str) -> bool:
    p = (path or "").lower()
    if not p:
        return False
    non_authoritative = (
        "test/", "/test", "tests/", "/tests", "spec/", "specs/", "fixture", "fixtures",
        "example", "examples", "sample", "samples", "mock", "/docs/",
        ".env.example", ".env.sample",
    )
    return not any(tok in p for tok in non_authoritative)


def _query_path_hints(query: str) -> list[str]:
    hints = []
    for tok in re.findall(r"[a-z0-9_\-\.]+", query):
        if len(tok) < 3:
            continue
        if tok in {"what", "where", "declared", "configured", "dependencies", "config"}:
            continue
        hints.append(tok)
    return hints


def _rank_by_query_hints(paths: list[str], hints: list[str]) -> list[str]:
    def score(path: str) -> tuple[int, int]:
        p = path.lower()
        hint_hits = sum(1 for h in hints if h in p)
        # Prefer less nested files for repo-level questions.
        depth = p.count("/")
        return (hint_hits, -depth)

    return sorted(paths, key=score, reverse=True)
