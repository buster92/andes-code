"""Deterministic source-of-truth retrieval helpers for config/declaration questions."""

from __future__ import annotations

import re

AUTHORITATIVE_NAME_HINTS = [
    "androidmanifest.xml",
    "manifest",
    "build.gradle",
    "build.gradle.kts",
    "settings.gradle",
    "settings.gradle.kts",
    "package.json",
    "pyproject.toml",
    "requirements.txt",
    "cargo.toml",
    "go.mod",
    "pom.xml",
    "dockerfile",
    "docker-compose.yml",
    "package.swift",
    "pipfile",
    "poetry.lock",
    ".env",
    "config",
    "settings",
    "entitlements",
    ".plist",
]


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


def expected_authority_candidates(intent: str, query: str, manifests: list[str], config_files: list[str]) -> list[str]:
    """
    Deterministic expansion for strict-authority recovery when primary retrieval
    does not find authoritative files.
    """
    q = (query or "").lower()
    authoritative_paths = [
        p for p in (manifests + config_files) if p and _is_authoritative_candidate(p)
    ]
    candidates: list[str] = []

    # Start from the normal deterministic ranking.
    candidates.extend(config_priority_files(intent, query, manifests, config_files))

    # Intent-focused, cross-ecosystem authoritative expansions.
    intent_hints: list[str] = []
    if intent == "dependency_or_build_inventory":
        intent_hints.extend([
            "build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts",
            "package.json", "pyproject.toml", "requirements.txt", "cargo.toml", "go.mod",
            "pom.xml", "dockerfile", "docker-compose.yml", "package.swift", "pipfile", "poetry.lock",
        ])
    else:
        intent_hints.extend([
            "androidmanifest.xml", "manifest", "config", "settings", ".env", ".plist", "entitlements",
            "docker-compose.yml", "dockerfile",
        ])

    # Query-focused expansions.
    if any(k in q for k in ("permission", "permissions", "declared", "manifest")):
        intent_hints.extend(["androidmanifest.xml", "manifest"])
    if any(k in q for k in ("dependenc", "library", "package", "build", "module")):
        intent_hints.extend([
            "build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts",
            "package.json", "pyproject.toml", "requirements.txt", "cargo.toml", "go.mod", "pom.xml",
        ])
    if any(k in q for k in ("config", "configured", "setting", "environment", "env", "entitlement", "plist")):
        intent_hints.extend(["config", "settings", ".env", ".plist", "entitlements"])

    path_hints = sorted(set(_query_path_hints(q)))
    for hint in path_hints:
        if "." in hint or "/" in hint or "-" in hint:
            intent_hints.append(hint)

    for hint in AUTHORITATIVE_NAME_HINTS + intent_hints:
        hint_l = hint.lower()
        matched = [p for p in authoritative_paths if hint_l in p.lower()]
        candidates.extend(_rank_by_query_hints(matched, path_hints))
        if "/" not in hint and "." in hint:
            candidates.append(hint)

    seen = set()
    ordered = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def rank_recovery_authoritative_paths(
    intent: str,
    query: str,
    manifests: list[str],
    config_files: list[str],
    candidate_hints: list[str],
) -> list[str]:
    """
    Rank concrete authoritative paths for recovery with deterministic priority:
    exact path > basename > suffix/segment > substring.
    """
    q = (query or "").lower()
    path_hints = sorted(set(_query_path_hints(q)))
    normalized_candidate_hints = sorted({(h or "").lower().strip() for h in candidate_hints if (h or "").strip()})
    authoritative_paths = [
        p for p in (manifests + config_files) if p and _is_authoritative_candidate(p)
    ]
    if not authoritative_paths:
        return []

    scored = []
    for path in sorted(set(authoritative_paths)):
        p = path.lower()
        base = p.rsplit("/", 1)[-1]
        score = 0
        source_type = classify_source_type(path)
        candidate_factor = _candidate_match_factor(intent, q, source_type)

        # Candidate hint matching priority.
        for h in normalized_candidate_hints:
            if p == h:
                score += int(120 * candidate_factor)
            elif base == h:
                score += int(100 * candidate_factor)
            elif p.endswith(f"/{h}") or f"/{h}/" in p:
                score += int(70 * candidate_factor)
            elif h in p:
                score += int(20 * candidate_factor)

        # Query/module path overlap (prefer hinted modules/services/subtrees).
        for hint in path_hints:
            if p == hint:
                score += 110
            elif base == hint:
                score += 90
            elif p.endswith(f"/{hint}") or f"/{hint}/" in p:
                score += 55
            elif hint in p:
                score += 15

        # Intent relevance weighting.
        score += _intent_source_priority(intent, q, source_type, base)

        # Prefer shallower paths if relevance ties.
        depth_penalty = p.count("/")
        scored.append((score, -depth_penalty, path))

    scored.sort(reverse=True)
    return [path for score, _, path in scored if score > 0]


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
        "test/", "/test", "tests/", "/tests", "__tests__", "spec/", "specs/",
        "fixture", "fixtures", "testdata", "example", "examples", "sample", "samples",
        "mock", "mocks", "/docs/", "/doc/", "docs/", "doc/",
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


def _intent_source_priority(intent: str, query: str, source_type: str, basename: str) -> int:
    asks_dependency = any(k in query for k in ("dependenc", "library", "package", "build", "module"))
    if intent == "dependency_or_build_inventory":
        if source_type == "dependency_file":
            return 70
        if source_type == "build_file":
            return 60
        if source_type == "manifest":
            return 20
        return 0

    asks_manifest = any(k in query for k in ("permission", "permissions", "manifest"))
    asks_config = any(k in query for k in ("config", "configured", "settings", "env", "environment"))
    if asks_manifest:
        if source_type == "manifest":
            return 80
        if source_type in {"build_file", "dependency_file"}:
            return 10
        return 0
    if asks_config:
        if source_type == "config_file":
            return 90
        if source_type == "manifest":
            return 35
        if source_type in {"build_file", "dependency_file"}:
            return 8 if not asks_dependency else 25
        return 0
    if basename in {"dockerfile", "docker-compose.yml"}:
        return 25
    if source_type in {"manifest", "config_file"}:
        return 40
    if source_type in {"build_file", "dependency_file"}:
        return 35
    return 0


def _candidate_match_factor(intent: str, query: str, source_type: str) -> float:
    asks_manifest = any(k in query for k in ("permission", "permissions", "manifest"))
    asks_config = any(k in query for k in ("config", "configured", "settings", "env", "environment"))
    asks_dependency = any(k in query for k in ("dependenc", "library", "package", "build", "module"))

    if intent == "dependency_or_build_inventory":
        return 1.0 if source_type in {"dependency_file", "build_file"} else 0.25
    if asks_manifest:
        return 1.0 if source_type == "manifest" else 0.2
    if asks_config and not asks_dependency:
        return 1.0 if source_type in {"config_file", "manifest"} else 0.2
    return 1.0
