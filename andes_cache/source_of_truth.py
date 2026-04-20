"""Deterministic source-of-truth retrieval helpers for config/declaration questions."""

from __future__ import annotations

import re

AUTHORITATIVE_NAME_HINTS = [
    "AndroidManifest.xml",
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

PRIMARY_PATH_POSITIVE_TOKENS = {
    "app", "apps", "service", "services", "package", "packages", "core", "api",
    "src", "main",
}
PRIMARY_PATH_NEGATIVE_TOKENS = {
    "test", "tests", "androidtest", "debug", "spec", "specs", "mock", "mocks",
    "fixture", "fixtures", "example", "examples", "sample", "samples",
    "node_modules", "dist", "build", "generated", "out", ".gradle",
}
_BROAD_QUERY_WORDS = {"repo", "repository", "project", "app", "application", "overall", "global"}


def config_priority_files(intent: str, query: str, manifests: list[str], config_files: list[str]) -> list[str]:
    authoritative_paths = [
        p for p in (manifests + config_files) if p and _is_authoritative_candidate(p)
    ]
    return rank_authoritative_paths(authoritative_paths, query, intent)


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
    authoritative_paths = [
        p for p in (manifests + config_files) if p and _is_authoritative_candidate(p)
    ]
    if not authoritative_paths:
        return []

    normalized_candidate_hints = sorted({(h or "").lower().strip() for h in candidate_hints if (h or "").strip()})
    ranked = rank_authoritative_paths(authoritative_paths, query, intent)
    if not normalized_candidate_hints:
        return ranked

    with_hint_scores = []
    for path in ranked:
        p = path.lower()
        base = p.rsplit("/", 1)[-1]
        source_type = classify_source_type(path)
        candidate_factor = _candidate_match_factor(intent, (query or "").lower(), source_type)
        score = 0
        for h in normalized_candidate_hints:
            if p == h:
                score += int(120 * candidate_factor)
            elif base == h:
                score += int(85 * candidate_factor)
            elif p.endswith(f"/{h}") or f"/{h}/" in p:
                score += int(45 * candidate_factor)
            elif h in p:
                score += int(15 * candidate_factor)
        with_hint_scores.append((score, path))
    with_hint_scores.sort(key=lambda t: (t[0], -t[1].count("/"), t[1]), reverse=True)
    return [p for _, p in with_hint_scores if _ > 0] or ranked


def score_authoritative_path(path: str, query: str, intent: str) -> int:
    q = (query or "").lower()
    path_l = (path or "").lower()
    base = path_l.rsplit("/", 1)[-1]
    tokens = _path_tokens(path_l)
    hint_tokens = set(_query_path_hints(q))
    broad_query = _is_broad_query(q, hint_tokens, set(tokens))

    score = 0
    source_type = classify_source_type(path_l)
    score += _intent_source_priority(intent, q, source_type, base)
    score += 18 * sum(1 for t in tokens if t in PRIMARY_PATH_POSITIVE_TOKENS)
    score -= 35 * sum(1 for t in tokens if t in PRIMARY_PATH_NEGATIVE_TOKENS)

    if "/src/main/" in path_l:
        score += 70
    if "/app/" in path_l or path_l.startswith("app/"):
        score += 38
    if "/services/" in path_l or "/packages/" in path_l or "/apps/" in path_l:
        score += 26

    query_hits = sum(1 for tok in hint_tokens if tok in tokens or tok in base)
    score += query_hits * 42

    if broad_query:
        depth = path_l.count("/")
        score += max(0, 36 - (depth * 4))
        if base in {"settings.gradle", "settings.gradle.kts", "pyproject.toml", "package.json"} and depth <= 2:
            score += 32
    else:
        score += min(20, path_l.count("/") * 2)
    return score


def rank_authoritative_paths(paths: list[str], query: str, intent: str) -> list[str]:
    scored: list[tuple[int, int, str]] = []
    for p in sorted(set(paths)):
        score = score_authoritative_path(p, query, intent)
        scored.append((score, -p.count("/"), p))
    scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    return [p for _, _, p in scored]


def select_best_authoritative_path(paths: list[str], query: str, intent: str) -> str:
    ranked = rank_authoritative_paths(paths, query, intent)
    return ranked[0] if ranked else ""


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


def is_declaration_query(query: str, intent: str = "") -> bool:
    """Return True when user asks for config/build/dependency declarations."""
    q = (query or "").lower()
    declaration_keywords = (
        "dependenc",
        "declared",
        "librar",
        "version",
        "manifest",
        "permission",
        "config",
        "build",
        "settings",
        "requirements",
        "package.json",
        "pyproject",
        "gradle",
        "pom.xml",
    )
    if intent in {
        "declaration_or_configuration",
        "dependency_or_build_inventory",
        "config_lookup",
        "dependency_lookup",
    }:
        return True
    return any(k in q for k in declaration_keywords)


def has_declaration_keywords(query: str) -> bool:
    q = (query or "").lower()
    return any(
        k in q
        for k in (
            "dependenc",
            "declared",
            "librar",
            "version",
            "build",
            "config",
            "requirements",
            "manifest",
            "permission",
        )
    )


def source_of_truth_guidance(query: str, intent: str = "") -> str:
    """Prompt guidance for declaration/config/build questions."""
    if not is_declaration_query(query, intent):
        return ""
    return (
        "## Source-of-Truth Guidance\n"
        "- Prioritize authoritative declaration/config/build files first.\n"
        "- Prefer explicit declarations over inference from runtime code usage.\n"
        "- Downgrade inferred statements unless declaration files are missing.\n"
        "- Format the final answer in two sections: `Declared` and `Inferred from usage`.\n"
        "- If declaration files are missing, state that explicitly before any inferred findings.\n\n"
    )

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
        if tok in {"what", "where", "declared", "configured", "dependencies", "config", "used"}:
            continue
        hints.append(tok)
    return hints


def _path_tokens(path: str) -> list[str]:
    return [tok for tok in re.split(r"[^a-z0-9]+", path.lower()) if tok]


def _is_broad_query(query: str, hint_tokens: set[str], discovered_path_tokens: set[str] | None = None) -> bool:
    q = query.lower()
    if any(w in q for w in _BROAD_QUERY_WORDS):
        return True

    # Explicit target scope generally indicates module/service-specific intent.
    if any(m in q for m in {"service", "module", "package", "feature"}):
        return False

    # Path-like or file-like hints imply specificity.
    if "/" in q or "\\" in q:
        return False
    if any("." in tok for tok in hint_tokens):
        return False

    # Strong overlap between query hints and discovered path tokens implies specificity.
    path_tokens = discovered_path_tokens or set()
    overlap = len([h for h in hint_tokens if h in path_tokens])
    if overlap >= 2 and overlap >= max(1, len(hint_tokens) // 2):
        return False

    # If query provides many specific hints, treat as focused.
    specific_hints = [t for t in hint_tokens if len(t) >= 4 and t not in _BROAD_QUERY_WORDS]
    if len(specific_hints) >= 3:
        return False

    # Few hints with no explicit target markers defaults to broad.
    return len(hint_tokens) <= 2


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
            return -20 if not asks_dependency else 25
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
