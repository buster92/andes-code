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
    "libs.versions.toml",
    "gradle.properties",
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
    "gemfile",
    "composer.json",
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
            "libs.versions.toml", "gradle.properties",
            "package.json", "pyproject.toml", "requirements.txt", "cargo.toml", "go.mod",
            "pom.xml", "dockerfile", "docker-compose.yml", "package.swift", "pipfile", "poetry.lock",
            "gemfile", "composer.json",
        ])
    else:
        intent_hints.extend([
            "androidmanifest.xml", "manifest", "config", "settings", ".env", ".plist", "entitlements",
            "docker-compose.yml", "dockerfile",
        ])

    # Query-focused expansions.
    if any(k in q for k in ("permission", "permissions", "manifest")):
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
    # Dependency-file patterns are checked before build-file patterns so more-specific
    # names (e.g. libs.versions.toml, which lives under a gradle/ directory) win.
    if any(x in p for x in (
        "requirements", "pyproject", "cargo.toml", "go.mod", "package.json",
        "pipfile", "poetry.lock", "libs.versions.toml", "gemfile", "composer.json",
    )):
        return "dependency_file"
    if any(x in p for x in (
        "gradle", "pom.xml", "dockerfile", "compose", "package.swift",
    )):
        return "build_file"
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
            "# Source-of-Truth Limitation\n"
            "- No AndroidManifest.xml was found in indexed source-of-truth files.\n"
            "- Declared permissions cannot be confirmed.\n"
            "- Any fallback findings below are references/inferences, not declarations."
        ),
        "file": "__manifest_status__",
        "language": "meta",
        "source_type": "meta",
        "authority_level": "notice",
    }



# ---------------------------------------------------------------------------
# Dependency / build-declaration authority helpers
# ---------------------------------------------------------------------------

#: Source types that constitute genuine dependency/build declaration authority.
#: A manifest-only context does NOT satisfy this requirement.
DEPENDENCY_BUILD_AUTHORITY_TYPES: frozenset[str] = frozenset({"dependency_file", "build_file"})


def context_has_dependency_authority(chunks: list[dict]) -> bool:
    """Return True if at least one chunk comes from a dependency/build declaration file.

    Manifest files (AndroidManifest.xml, etc.) and generic config files are
    explicitly excluded — they do not constitute dependency declaration authority.
    """
    return any(c.get("source_type") in DEPENDENCY_BUILD_AUTHORITY_TYPES for c in chunks)


def workspace_has_dependency_files(manifests: list[str], config_files: list[str]) -> bool:
    """Return True if the workspace metadata contains at least one dependency/build file.

    Used to distinguish between:
      - No dependency files exist at all in the workspace (Case A)
      - Files exist but are not indexed/retrievable (Case B)
    """
    all_paths = (manifests or []) + (config_files or [])
    return any(
        classify_source_type(p) in DEPENDENCY_BUILD_AUTHORITY_TYPES
        for p in all_paths
        if p
    )


def recover_dependency_build_files(
    intent: str,
    query: str,
    manifests: list[str],
    config_files: list[str],
) -> list[str]:
    """Dedicated recovery helper that returns only dependency/build declaration file
    candidates filtered from workspace metadata.

    Called when a dependency query has collected manifest-only context and needs a
    force-recovery pass for the required dependency/build authority.  Uses
    filename/path-driven ranking (via rank_authoritative_paths) — no semantic fallback.
    """
    all_paths = [p for p in (manifests or []) + (config_files or []) if p]
    dep_paths = [
        p for p in all_paths
        if classify_source_type(p) in DEPENDENCY_BUILD_AUTHORITY_TYPES
        and _is_authoritative_candidate(p)
    ]
    return rank_authoritative_paths(dep_paths, query, intent)


def no_dependency_files_in_workspace_limitation() -> dict:
    """Limitation chunk for Case A: no dependency declaration files exist in workspace.

    Distinct from the generic source-of-truth limitation so callers and the LLM
    can clearly distinguish 'nothing to find' from 'something exists but is stale'.
    """
    return {
        "content": (
            "# No Declaration Files\n"
            "- No dependency declaration files (build.gradle, build.gradle.kts, package.json, "
            "requirements.txt, pyproject.toml, Cargo.toml, go.mod, pom.xml, etc.) were found "
            "in this workspace.\n"
            "- Declared dependencies cannot be sourced from a build or dependency declaration file.\n"
            "- AndroidManifest.xml and other config files are present but do not declare "
            "build dependencies."
        ),
        "file": "__no_dependency_files_in_workspace__",
        "language": "meta",
        "source_type": "meta",
        "authority_level": "notice",
    }


def dependency_files_not_indexed_limitation() -> dict:
    """Limitation chunk for Case B: dependency files exist in workspace but are not
    indexed/retrievable.

    Distinct from Case A so the user knows re-indexing should fix this.
    """
    return {
        "content": (
            "# Source-of-Truth Limitation\n"
            "- Dependency declaration files were found in workspace metadata, but the current "
            "index could not retrieve them.\n"
            "- The index may be stale or the files may not have been indexed yet.\n"
            "- AndesCode attempted deterministic dependency/build file recovery before returning "
            "this limitation.\n"
            "- Re-indexing the workspace should resolve this."
        ),
        "file": "__dependency_files_not_indexed__",
        "language": "meta",
        "source_type": "meta",
        "authority_level": "notice",
    }


def dependency_authority_incomplete_limitation() -> dict:
    """Limitation chunk for Case C: partial recovery succeeded but required dependency
    authority is still incomplete (e.g. only manifest/config files in context).
    """
    return {
        "content": (
            "# Source-of-Truth Limitation\n"
            "- Dependency declaration file recovery was attempted but could not satisfy the "
            "required dependency authority for this query.\n"
            "- Some authoritative files were found, but none are recognized dependency/build "
            "declaration files (build.gradle, package.json, requirements.txt, etc.).\n"
            "- Declared dependencies cannot be confirmed from the retrieved context."
        ),
        "file": "__dependency_authority_incomplete__",
        "language": "meta",
        "source_type": "meta",
        "authority_level": "notice",
    }


# ---------------------------------------------------------------------------
# Declaration-domain keyword detection
# ---------------------------------------------------------------------------
# Rules for this regex:
#
#  1. Always use \b word-boundary anchors so that domain terms are never
#     matched as sub-strings of unrelated words:
#       "independence" must NOT trigger dependency routing.
#       "misconfigured" must NOT trigger config routing.
#
#  2. Use prefix patterns (dependenc\w+, librar\w*, config\w*) to catch
#     all inflected forms (dependency/dependencies, library/libraries,
#     configured/configuration) with a single branch.
#
#  3. "declared" is intentionally EXCLUDED.  It is a common English
#     adjective ("What variables are declared?", "What is declared vs
#     defined?") and its presence alone is not evidence of a build-file /
#     manifest query.  Dependency queries that happen to contain "declared"
#     are correctly classified by the dependency-domain terms themselves
#     ("dependenc\w+", "librar\w*", etc.).
#
# If you need to add a new ecosystem keyword, add it here (not in ad-hoc
# `any(k in q for k in ...)` checks scattered in the codebase).
# ---------------------------------------------------------------------------
_DECL_KW_RE = re.compile(
    r"\b(?:"
    r"dependenc\w+"      # dependency, dependencies, dependent, …
    r"|librar\w*"        # library, libraries
    r"|version\w*"       # version, versions, versioned
    r"|manifest"
    r"|permission\w*"    # permission, permissions
    r"|config\w*"        # config, configuration, configured, configs
    r"|build\w*"         # build, builds, building, build.gradle
    r"|setting\w*"       # setting, settings
    r"|requirements"
    r"|gradle\w*"
    r"|pom"              # pom, pom.xml
    r"|pyproject"
    r"|package\.json"
    r"|pom\.xml"
    r")",
    re.IGNORECASE,
)


def is_declaration_query(query: str, intent: str = "") -> bool:
    """Return True when user asks for config/build/dependency declarations."""
    if intent in {
        "declaration_or_configuration",
        "dependency_or_build_inventory",
        "config_lookup",
        "dependency_lookup",
    }:
        return True
    return has_declaration_keywords(query)


def has_declaration_keywords(query: str) -> bool:
    return bool(_DECL_KW_RE.search(query or ""))


def source_of_truth_guidance(query: str, intent: str = "") -> str:
    """Prompt guidance for declaration/config/build questions."""
    if not is_declaration_query(query, intent):
        return ""
    return (
        "## Source-of-Truth Guidance — STRICT RULES FOR THIS QUERY\n\n"
        "This is a dependency/declaration query. You MUST follow every rule below exactly.\n\n"
        "**Definitions:**\n"
        "- DECLARED = a dependency or setting that appears explicitly in a build or dependency "
        "declaration file (build.gradle, build.gradle.kts, package.json, requirements.txt, "
        "pyproject.toml, Cargo.toml, go.mod, pom.xml, Podfile, Package.swift, Dockerfile, etc.).\n"
        "- INFERRED = a dependency or library observed only from import statements, `require()` "
        "calls, or other code usage — NOT from a declaration file.\n\n"
        "**Mandatory Output Structure:**\n"
        "Your answer MUST be split into exactly these two sections. Never merge them.\n\n"
        "### Declared Dependencies\n"
        "List only what you can directly cite from a declaration file present in the retrieved "
        "context. For each entry include the source file name. If no declaration files are in "
        "the retrieved context, write: *No dependency declaration files were found.*\n\n"
        "### Inferred from Code Usage\n"
        "List only what you observed from imports, `require()` calls, or usage patterns in source "
        "files — NOT from declaration files. If none found, write: *None identified.*\n\n"
        "**Critical prohibitions:**\n"
        "- NEVER present inferred findings as declared dependencies.\n"
        "- NEVER merge the two sections into a single list.\n"
        "- NEVER omit the 'Declared Dependencies' section even when it is empty.\n"
        "- NEVER omit the 'Inferred from Code Usage' section even when it is empty.\n"
        "- If a retrieved chunk is labelled `source_type: source_code` or "
        "`authority_level: inferred`, it belongs in 'Inferred from Code Usage' only.\n"
        "- If a retrieved chunk is labelled `source_type: meta` or `authority_level: notice`, "
        "it is a system notice — NOT a dependency entry. Quote it verbatim at the very top "
        "of your response, before either section header.\n"
        "- The only recognized system notice markers are '# Source-of-Truth Limitation' and "
        "'# No Declaration Files'. If either appears in the context, quote it verbatim at the "
        "top of the response.\n\n"
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
    # Use word-tokenised set so that short terms never match mid-word
    # (e.g. "config" must not fire inside "misconfigured").
    _qw = set(re.findall(r"\w+", (query or "").lower()))
    asks_dependency = bool(
        _qw & {"library", "libraries", "package", "packages", "module", "modules"}
        or re.search(r"\bdependenc\w+|\bbuild\w*", query, re.IGNORECASE)
    )
    if intent == "dependency_or_build_inventory":
        if source_type == "dependency_file":
            return 70
        if source_type == "build_file":
            return 60
        if source_type == "manifest":
            return 20
        return 0

    asks_manifest = bool(_qw & {"permission", "permissions", "manifest"})
    asks_config = bool(_qw & {"config", "configured", "settings", "env", "environment"})
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
    _qw = set(re.findall(r"\w+", (query or "").lower()))
    asks_manifest = bool(_qw & {"permission", "permissions", "manifest"})
    asks_config = bool(_qw & {"config", "configured", "settings", "env", "environment"})
    asks_dependency = bool(
        _qw & {"library", "libraries", "package", "packages", "module", "modules"}
        or re.search(r"\bdependenc\w+|\bbuild\w*", query, re.IGNORECASE)
    )

    if intent == "dependency_or_build_inventory":
        return 1.0 if source_type in {"dependency_file", "build_file"} else 0.25
    if asks_manifest:
        return 1.0 if source_type == "manifest" else 0.2
    if asks_config and not asks_dependency:
        return 1.0 if source_type in {"config_file", "manifest"} else 0.2
    return 1.0
