"""Deterministic source-of-truth retrieval helpers for config/declaration questions."""

from __future__ import annotations

import re


def config_priority_files(intent: str, query: str, manifests: list[str], config_files: list[str]) -> list[str]:
    q = (query or "").lower()
    preferred = []

    if any(k in q for k in ("permission", "permissions", "declared", "manifest")):
        preferred.extend([m for m in manifests if m.endswith("AndroidManifest.xml")])
        preferred.append("AndroidManifest.xml")

    if intent == "dependency_inventory":
        dep_files = [
            "build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts",
            "package.json", "requirements.txt", "pyproject.toml", "Cargo.toml",
            "go.mod", "pom.xml", "poetry.lock", "Pipfile",
        ]
        for df in dep_files:
            preferred.extend([m for m in manifests if m.endswith(df)])
            preferred.append(df)
    else:
        cfg_files = [
            "AndroidManifest.xml", "build.gradle", "build.gradle.kts",
            "settings.gradle", "settings.gradle.kts", ".env", "docker-compose.yml",
            "package.json", "pyproject.toml",
        ]
        for cf in cfg_files:
            preferred.extend([m for m in manifests if m.endswith(cf)])
            preferred.append(cf)
        preferred.extend(config_files)

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
