import os
os.environ.setdefault("TRANSFORMERS_OFFLINE",              "1")
os.environ.setdefault("HF_DATASETS_OFFLINE",              "1")
os.environ.setdefault("HF_HUB_OFFLINE",                   "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM",            "false")

import hashlib
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Generator

import chromadb
from sentence_transformers import SentenceTransformer

from andes_cache import AndesCacheManager, RepoFingerprinter
from andes_cache.routing import (
    classify_query_intent_details,
    RUNTIME_USAGE_OR_REFERENCE,
)
from andes_cache.source_of_truth import (
    config_priority_files,
    summarize_declared_permissions,
    annotate_sources,
    missing_manifest_notice,
    classify_source_type,
    authority_level_for_source,
    wants_runtime_usage,
)
from andes_cache.versions import (
    INDEX_VERSION,
    PARSER_VERSION,
    PROMPT_TEMPLATE_VERSION,
    RETRIEVAL_POLICY_VERSION,
)

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL   = "all-MiniLM-L6-v2"
CHROMA_PATH   = str(Path(__file__).parent / "index")
COLLECTION    = "codebase"
HASH_STORE    = Path(__file__).parent / "index" / ".file_hashes.json"
PROJECT_MAP   = Path(__file__).parent / "index" / "project_map.json"
SYMBOL_INDEX  = Path(__file__).parent / "index" / "symbol_index.json"
WORKSPACE_INDEX = Path(__file__).parent / "index" / "workspace_index.json"
CACHE_DIR = Path(__file__).parent / "index" / "cache"
CACHE = AndesCacheManager(CACHE_DIR)
CURRENT_REPO_FINGERPRINT = ""
EMBED_BATCH   = 64
CHROMA_BATCH  = 4096
CHUNK_LINES   = 80
CHUNK_OVERLAP = 15

# ── Language boundary patterns ────────────────────────────────────────────────
_BOUNDARY = {
    "py":   re.compile(r"^(def |class |async def )", re.MULTILINE),
    "js":   re.compile(r"^(function |const \w+ ?= ?(async ?)?\(|class )", re.MULTILINE),
    "ts":   re.compile(r"^(function |const |class |interface |type |export )", re.MULTILINE),
    "jsx":  re.compile(r"^(function |const |class |export )", re.MULTILINE),
    "tsx":  re.compile(r"^(function |const |class |export |interface )", re.MULTILINE),
    "java": re.compile(r"^\s*(public|private|protected|static)[^\n]*\(", re.MULTILINE),
    "kt":   re.compile(r"^\s*(fun |class |object |interface )", re.MULTILINE),
    "swift":re.compile(r"^\s*(func |class |struct |protocol |extension )", re.MULTILINE),
    "go":   re.compile(r"^(func |type )", re.MULTILINE),
    "rs":   re.compile(r"^(fn |pub fn |impl |struct |enum |trait )", re.MULTILINE),
}

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".go", ".rs", ".java", ".cpp", ".c", ".h",
    ".rb", ".php", ".cs", ".swift", ".kt",
    ".gradle", ".xml",
}

MANIFEST_FILES = {
    "requirements.txt", "pyproject.toml", "poetry.lock", "Pipfile",
    "package.json", "pnpm-workspace.yaml", "yarn.lock",
    "build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts",
    "Cargo.toml", "go.mod", "pom.xml", "build.sbt",
    "Podfile", "Package.swift", "AndroidManifest.xml", "Dockerfile", "docker-compose.yml",
}

SKIP_DIRS = {
    ".git", ".svn", ".hg",
    "node_modules", ".next", ".nuxt", "dist", "build",
    "__pycache__", ".venv", "venv", ".pytest_cache", ".mypy_cache", "coverage",
    "vendor", "bundle", "Pods", "Carthage",
    ".gradle", ".idea", "target", "bin", "gen", "intermediates", "generated",
    "pkg", "DerivedData", "xcuserdata",
    "tmp", "temp", "cache", ".cache", "logs",
    ".build", "release", "debug",
}

# ── Setup ─────────────────────────────────────────────────────────────────────
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

print("🔍  Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)
chroma   = chromadb.PersistentClient(path=CHROMA_PATH)
col      = chroma.get_or_create_collection(COLLECTION)
print("✅  Indexer ready.")


# ── Public API ─────────────────────────────────────────────────────────────────

def index_codebase(root: str) -> dict:
    result = {}
    for event in index_codebase_stream(root):
        if event["type"] == "done":
            result = event
    return result


def index_codebase_stream(root: str) -> Generator[dict, None, None]:
    """
    Index a codebase with incremental support. Yields SSE progress events.
    On completion builds project map + symbol index for smart retrieval.
    """
    global col
    root_path = Path(root).resolve()

    if not root_path.exists():
        yield {"type": "error", "message": f"Path not found: {root_path}"}
        return

    # ── Hash cache / project change detection ─────────────────────────────────
    hashes = _load_hashes()
    is_same_project = hashes.get("__root__") == str(root_path)
    previous_repo_fp = hashes.get("__fingerprint__", "")

    if not is_same_project:
        try:
            chroma.delete_collection(COLLECTION)
        except Exception:
            pass
        col    = chroma.get_or_create_collection(COLLECTION)
        hashes = {"__root__": str(root_path)}
        # Clear old maps
        for f in [PROJECT_MAP, SYMBOL_INDEX, WORKSPACE_INDEX]:
            try:
                f.unlink()
            except Exception:
                pass
        if previous_repo_fp:
            CACHE.invalidate_repo(previous_repo_fp, include_workspace=True)

    yield {"type": "clear"}

    # ── Collect and diff ──────────────────────────────────────────────────────
    all_files = _collect_files(root_path)
    new_files, unchanged = [], []
    changed_rel_paths = []

    for fp in all_files:
        key   = str(fp.relative_to(root_path))
        fhash = _file_hash(fp)
        if hashes.get(key) == fhash:
            unchanged.append(fp)
        else:
            new_files.append((fp, fhash))
            changed_rel_paths.append(key)

    known_files = {k for k in hashes.keys() if not k.startswith("__")}
    current_files = {str(fp.relative_to(root_path)) for fp in all_files}
    removed_paths = sorted(known_files - current_files)

    # File deletions can leave stale vector records. Force a safe full rebuild.
    if removed_paths:
        try:
            chroma.delete_collection(COLLECTION)
        except Exception:
            pass
        col = chroma.get_or_create_collection(COLLECTION)
        new_files = [(fp, _file_hash(fp)) for fp in all_files]
        changed_rel_paths = [str(fp.relative_to(root_path)) for fp, _ in new_files] + removed_paths
        unchanged = []

    # Recompute repo fingerprint from the current filesystem snapshot.
    next_hashes = {"__root__": str(root_path)}
    new_rel_set = {str(p.relative_to(root_path)) for p, _ in new_files}
    for fp in all_files:
        rel = str(fp.relative_to(root_path))
        if rel in new_rel_set:
            continue
        next_hashes[rel] = hashes.get(rel, _file_hash(fp))
    for fp, fhash in new_files:
        next_hashes[str(fp.relative_to(root_path))] = fhash

    repo_fp = RepoFingerprinter.build(
        root_path,
        next_hashes,
        index_version=INDEX_VERSION,
        parser_version=PARSER_VERSION,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
        retrieval_policy_version=RETRIEVAL_POLICY_VERSION,
    )
    _set_repo_fingerprint(repo_fp)

    yield {
        "type": "scan", "files": len(all_files),
        "new": len(new_files), "unchanged": len(unchanged),
    }

    if not all_files:
        yield {"type": "error", "message": "No source files found in that directory"}
        return

    if not new_files and unchanged:
        # Nothing changed — still emit map event so UI can display it
        pmap = _load_project_map()
        CACHE.flush_metrics()
        yield {
            "type": "done", "indexed": len(unchanged),
            "chunks": col.count(), "project": root_path.name,
            "new": 0, "reused": len(unchanged),
            "map": pmap,
            "repo_fingerprint": repo_fp,
        }
        return

    # ── Chunk ─────────────────────────────────────────────────────────────────
    all_chunks = []
    for fp, _ in new_files:
        try:
            all_chunks.extend(_chunk_file(fp, root_path))
        except Exception as e:
            logging.warning(f"Skipped {fp}: {e}")

    total_chunks = len(all_chunks)
    yield {"type": "chunks", "total": total_chunks}

    if not all_chunks:
        yield {"type": "error", "message": "No chunks generated — check file contents"}
        return

    # ── Embed ─────────────────────────────────────────────────────────────────
    all_embeddings = []
    for start in range(0, total_chunks, EMBED_BATCH):
        batch = all_chunks[start: start + EMBED_BATCH]
        vecs  = embedder.encode([c["content"] for c in batch],
                                show_progress_bar=False).tolist()
        all_embeddings.extend(vecs)
        done = min(start + EMBED_BATCH, total_chunks)
        yield {"type": "embed", "done": done, "total": total_chunks,
               "pct": int(done / total_chunks * 100)}

    # ── Store ─────────────────────────────────────────────────────────────────
    stored = 0
    for start in range(0, total_chunks, CHROMA_BATCH):
        bc = all_chunks[start: start + CHROMA_BATCH]
        bv = all_embeddings[start: start + CHROMA_BATCH]
        col.upsert(
            ids        = [c["id"]      for c in bc],
            embeddings = bv,
            documents  = [c["content"] for c in bc],
            metadatas  = [{"file": c["file"], "language": c["language"],
                           "line": c["line"], "symbols": c.get("symbols", "")}
                          for c in bc],
        )
        stored += len(bc)
        yield {"type": "store", "done": stored, "total": total_chunks,
               "pct": int(stored / total_chunks * 100)}

    # ── Save hashes and invalidate dependent cache layers ────────────────────
    if previous_repo_fp and previous_repo_fp != repo_fp:
        CACHE.invalidate_repo(previous_repo_fp, include_workspace=False)

    for fp, fhash in new_files:
        hashes[str(fp.relative_to(root_path))] = fhash
    hashes["__root__"] = str(root_path)
    hashes["__fingerprint__"] = repo_fp
    _save_hashes(hashes)

    # ── Build project intelligence ────────────────────────────────────────────
    yield {"type": "mapping", "message": "Building project map..."}

    all_indexed = [fp for fp, _ in new_files] + unchanged
    workspace = build_workspace_index(
        root_path,
        all_indexed,
        all_chunks,
        repo_fingerprint=repo_fp,
        changed_paths=changed_rel_paths,
    )
    pmap      = build_project_map(root_path, all_indexed, all_chunks, workspace)
    symidx    = build_symbol_index(all_chunks)

    _save_json(PROJECT_MAP, pmap)
    _save_json(SYMBOL_INDEX, symidx)
    _save_json(WORKSPACE_INDEX, workspace)
    CACHE.workspace_set(repo_fp, "symbol_index", symidx)
    CACHE.flush_metrics()

    yield {
        "type":    "done",
        "indexed": len({c["file"] for c in all_chunks}) + len(unchanged),
        "chunks":  col.count(),
        "project": root_path.name,
        "new":     len(new_files),
        "reused":  len(unchanged),
        "map":     pmap,
        "repo_fingerprint": repo_fp,
    }


def search(query: str, n_results: int = 5) -> list[dict]:
    """
    Smart retrieval with query routing and re-ranking.

    Routes:
      - Filename mentioned   → fetch ALL chunks from that file
      - Symbol name match    → boost chunks from that symbol's file
      - Architectural query  → fetch more candidates (8 instead of 5)
      - Default              → semantic search + re-ranking
    """
    count = col.count()
    if count == 0:
        return []
    decision = classify_query_intent_details(query)
    intent = decision["intent"]
    retrieval_route = decision["retrieval_route"]
    repo_fp = get_repo_fingerprint()
    if repo_fp:
        cached = CACHE.retrieval_get(
            repo_fp=repo_fp,
            query=query,
            index_version=INDEX_VERSION,
            intent=intent,
            retrieval_route=retrieval_route,
        )
        if cached:
            return cached[:n_results]

    if retrieval_route == "source_of_truth":
        final = _retrieve_config_first(
            query,
            intent,
            n_results=n_results,
            ambiguous=decision.get("ambiguous", False),
            allow_runtime_fallback=decision.get("allow_runtime_fallback", False),
            strict_authority_mode=decision.get("strict_authority_mode", True),
        )
        if repo_fp and final:
            CACHE.retrieval_set(
                repo_fp=repo_fp,
                query=query,
                index_version=INDEX_VERSION,
                value=final,
                intent=intent,
                retrieval_route=retrieval_route,
            )
        return final

    # ── Route 1: Filename detected ────────────────────────────────────────────
    file_match = re.search(
        r'\b[\w/]+\.(py|js|ts|jsx|tsx|java|kt|swift|go|rs|cpp|c|rb|cs)\b',
        query, re.IGNORECASE
    )
    if file_match:
        fname   = file_match.group(0).lower()
        results = _fetch_all_from_file(fname, n_results)
        if results:
            final = _add_coverage(results)
            final = annotate_sources(final, source_type="source_code", authority_level="referenced")
            if repo_fp:
                CACHE.retrieval_set(
                    repo_fp=repo_fp,
                    query=query,
                    index_version=INDEX_VERSION,
                    value=final,
                    intent=intent,
                    retrieval_route=retrieval_route,
                )
            return final

    # ── Route 1b: Structured workspace questions ─────────────────────────────
    structured = _structured_query_results(query)
    if structured:
        final = structured[:n_results]
        if repo_fp:
            CACHE.retrieval_set(
                repo_fp=repo_fp,
                query=query,
                index_version=INDEX_VERSION,
                value=final,
                intent=intent,
                retrieval_route=retrieval_route,
            )
        return final

    # ── Route 2: Symbol name lookup ───────────────────────────────────────────
    symidx = _load_json(SYMBOL_INDEX)
    if symidx:
        query_words = set(re.findall(r"\w+", query))
        matched_files = set()
        for word in query_words:
            if word in symidx:
                matched_files.update(symidx[word])
        if matched_files:
            # Boost n_results for symbol queries — likely need more context
            n_results = max(n_results, 6)

    # ── Route 3: Architectural query ──────────────────────────────────────────
    arch_patterns = re.compile(
        r'\b(how does|how is|where is|where are|explain|overview|'
        r'architecture|structure|flow|pipeline|what is the|'
        r'how do you|entry point|main|startup)\b',
        re.IGNORECASE
    )
    if arch_patterns.search(query):
        n_results = max(n_results, 8)

    # ── Semantic search + re-ranking ──────────────────────────────────────────
    embedding = embedder.encode(query).tolist()
    fetch     = min(n_results * 6, count)   # wider candidate pool → better rerank
    results   = col.query(query_embeddings=[embedding], n_results=fetch)

    candidates = [
        {
            "content":  doc,
            "file":     results["metadatas"][0][i].get("file", ""),
            "language": results["metadatas"][0][i].get("language", ""),
            "symbols":  results["metadatas"][0][i].get("symbols", ""),
            "score":    results["distances"][0][i],
        }
        for i, doc in enumerate(results["documents"][0])
    ]

    # Boost candidates from symbol-matched files
    if symidx and matched_files:
        for c in candidates:
            if any(mf in c["file"] for mf in matched_files):
                c["score"] *= 0.8   # lower distance = better match

    ranked = _rerank(query, candidates)

    # Add coverage metadata — lets server tell model how much of each file it has
    final = _add_coverage(ranked[:n_results])
    authority = "referenced" if intent == RUNTIME_USAGE_OR_REFERENCE else "inferred"
    final = annotate_sources(final, source_type="source_code", authority_level=authority)
    if repo_fp:
        CACHE.retrieval_set(
            repo_fp=repo_fp,
            query=query,
            index_version=INDEX_VERSION,
            value=final,
            intent=intent,
            retrieval_route=retrieval_route,
        )
    return final


def get_chunks_for_file(filename: str) -> list[dict]:
    """Retrieve ALL indexed chunks for a specific file."""
    return _fetch_all_from_file(filename, max_results=999)


def build_project_map(root_path: Path, files: list, chunks: list, workspace: dict | None = None) -> dict:
    """
    Analyse the project structure and build a structured summary.
    This gets injected into every system prompt so the model always
    knows what project it's working with.
    """
    # ── Language detection ────────────────────────────────────────────────────
    lang_counts = defaultdict(int)
    for fp in files:
        lang_counts[fp.suffix] += 1
    primary_ext  = max(lang_counts, key=lang_counts.get) if lang_counts else ""
    EXT_TO_LANG  = {
        ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
        ".kt": "Kotlin", ".java": "Java", ".swift": "Swift",
        ".go": "Go", ".rs": "Rust", ".cpp": "C++", ".cs": "C#",
    }
    primary_lang = EXT_TO_LANG.get(primary_ext, primary_ext.lstrip(".").upper())

    # ── Framework / stack detection ───────────────────────────────────────────
    root_files = {f.name for f in root_path.iterdir() if f.is_file()} \
                 if root_path.exists() else set()

    stack = []
    if "requirements.txt" in root_files or "pyproject.toml" in root_files:
        stack.append("Python")
        reqs = ""
        for rf in ["requirements.txt", "pyproject.toml"]:
            try:
                reqs = (root_path / rf).read_text(errors="ignore").lower()
                break
            except Exception:
                pass
        if "fastapi" in reqs or "flask" in reqs or "django" in reqs:
            stack.append("Web API")
        if "torch" in reqs or "tensorflow" in reqs:
            stack.append("ML/Deep Learning")
        if "pandas" in reqs or "numpy" in reqs:
            stack.append("Data Science")
        if "binance" in reqs or "ccxt" in reqs:
            stack.append("Crypto Trading")
        if "scikit" in reqs or "sklearn" in reqs:
            stack.append("Machine Learning")

    if "build.gradle" in root_files or "build.gradle.kts" in root_files:
        stack.append("Android")
    if "package.json" in root_files:
        try:
            pkg = json.loads((root_path / "package.json").read_text(errors="ignore"))
            deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
            if "react" in deps:     stack.append("React")
            if "next" in deps:      stack.append("Next.js")
            if "vue" in deps:       stack.append("Vue")
            if "express" in deps:   stack.append("Express")
        except Exception:
            stack.append("Node.js")
    if "Podfile" in root_files or any(f.endswith(".xcodeproj") for f in root_files):
        stack.append("iOS")
    if "Cargo.toml" in root_files:
        stack.append("Rust")
    if "go.mod" in root_files:
        stack.append("Go")

    # ── Entry point detection ─────────────────────────────────────────────────
    entry_candidates = [
        "main.py", "app.py", "server.py", "run.py", "manage.py", "__main__.py",
        "index.js", "app.js", "server.js", "main.js",
        "index.ts", "app.ts", "main.ts",
        "Main.java", "Application.java",
        "main.go", "main.rs", "main.swift",
    ]
    entry_point = None
    file_names  = {fp.name: fp for fp in files}
    for candidate in entry_candidates:
        if candidate in file_names:
            entry_point = str(file_names[candidate].relative_to(root_path))
            break

    # ── Module list (top-level source files) ──────────────────────────────────
    # Group by directory to give a clean module view
    module_dirs = defaultdict(list)
    for fp in files:
        rel  = fp.relative_to(root_path)
        parts = rel.parts
        if len(parts) == 1:
            module_dirs["root"].append(fp.name)
        else:
            module_dirs[parts[0]].append("/".join(parts[1:]))

    # Keep top 15 most populated directories
    top_modules = dict(
        sorted(module_dirs.items(), key=lambda x: -len(x[1]))[:15]
    )

    # ── Domain detection from source content ──────────────────────────────────
    domain_keywords = {
        "crypto/trading": ["btcusdt", "binance", "cryptocurrency", "trading", "candlestick",
                           "ohlcv", "binance", "coinbase", "kraken"],
        "web backend":    ["endpoint", "router", "middleware", "request", "response",
                           "http", "rest", "graphql"],
        "mobile/android": ["activity", "fragment", "viewmodel", "recyclerview",
                           "manifest", "gradle"],
        "mobile/ios":     ["uiviewcontroller", "swiftui", "storyboard", "appdelegate"],
        "data science":   ["dataframe", "numpy", "pandas", "sklearn", "model.fit"],
        "ML/AI":          ["neural", "epoch", "loss", "gradient", "tensor"],
        "devops/infra":   ["dockerfile", "kubernetes", "terraform", "deploy"],
    }

    domain_scores = defaultdict(int)
    # Sample first 30 files to detect domain
    sample_files = files[:30]
    for fp in sample_files:
        try:
            text = fp.read_text(errors="ignore").lower()[:3000]
            for domain, keywords in domain_keywords.items():
                domain_scores[domain] += sum(1 for kw in keywords if kw in text)
        except Exception:
            pass

    detected_domain = max(domain_scores, key=domain_scores.get) \
                      if any(v > 0 for v in domain_scores.values()) else None

    # ── Key symbols per file ──────────────────────────────────────────────────
    # Build a compact function/class map: {file: [symbol1, symbol2, ...]}
    file_symbols = defaultdict(list)
    for c in chunks:
        if c.get("symbols"):
            syms = c["symbols"].split()
            file_symbols[c["file"]].extend(syms)

    # Deduplicate and keep top files by symbol count
    file_symbol_map = {
        f: list(dict.fromkeys(syms))[:10]   # preserve order, cap at 10
        for f, syms in sorted(file_symbols.items(), key=lambda x: -len(x[1]))[:20]
    }

    workspace = workspace or {}

    return {
        "project":      root_path.name,
        "language":     primary_lang,
        "stack":        stack,
        "entry_point":  entry_point,
        "file_count":   len(files),
        "modules":      top_modules,
        "domain":       detected_domain,
        "file_symbols": file_symbol_map,
        "workspace": {
            "repo_types": workspace.get("repo_types", []),
            "package_managers": workspace.get("package_managers", []),
            "manifests": workspace.get("manifests", []),
            "entry_points": workspace.get("entry_points", [])[:15],
            "module_count": len(workspace.get("modules", [])),
        },
    }


def build_workspace_index(
    root_path: Path,
    files: list,
    chunks: list,
    repo_fingerprint: str = "",
    changed_paths: list[str] | None = None,
) -> dict:
    """
    Build structure-first workspace intelligence for dependency/config analysis.

    Workspace artifacts are cached independently so unchanged artifacts are reused.
    """
    changed_paths = changed_paths or []
    repo_fp = repo_fingerprint or get_repo_fingerprint()

    cached_manifests = CACHE.workspace_get(repo_fp, "manifests")
    manifests = cached_manifests if cached_manifests is not None else _discover_manifests(root_path)
    CACHE.workspace_set(repo_fp, "manifests", manifests)

    cached_modules = CACHE.workspace_get(repo_fp, "module_graph")
    modules = cached_modules if cached_modules is not None else _discover_modules(root_path, files)
    CACHE.workspace_set(repo_fp, "module_graph", modules)

    has_manifest_change = any(Path(p).name in MANIFEST_FILES for p in changed_paths)
    has_code_change = any(Path(p).suffix in SUPPORTED_EXTENSIONS for p in changed_paths)

    dependencies = CACHE.workspace_get(repo_fp, "dependency_inventory")
    if dependencies is None or has_manifest_change:
        dependencies = _collect_declared_dependencies(root_path, manifests)
        CACHE.workspace_set(repo_fp, "dependency_inventory", dependencies)

    import_graph = CACHE.workspace_get(repo_fp, "import_graph")
    if import_graph is None or has_code_change:
        import_graph = _build_import_graph(files, root_path)
        CACHE.workspace_set(repo_fp, "import_graph", import_graph)

    config_graph = CACHE.workspace_get(repo_fp, "config_graph")
    if config_graph is None or has_manifest_change:
        config_graph = _build_config_graph(root_path, manifests)
        CACHE.workspace_set(repo_fp, "config_graph", config_graph)

    entry_points = CACHE.workspace_get(repo_fp, "entry_points")
    if entry_points is None or has_code_change:
        entry_points = _discover_entry_points(root_path, files)
        CACHE.workspace_set(repo_fp, "entry_points", entry_points)

    file_to_module = CACHE.workspace_get(repo_fp, "file_to_module_map")
    if file_to_module is None or has_code_change:
        file_to_module = _build_file_to_module_map(root_path, files)
        CACHE.workspace_set(repo_fp, "file_to_module_map", file_to_module)

    return {
        "project": root_path.name,
        "repo_types": _detect_repo_types(manifests),
        "package_managers": _detect_package_managers(manifests),
        "modules": modules,
        "manifests": manifests,
        "entry_points": entry_points,
        "dependencies": dependencies,
        "import_graph": import_graph,
        "config_graph": config_graph,
        "file_to_module_map": file_to_module,
    }


def build_symbol_index(chunks: list) -> dict:
    """
    Build a symbol-to-file lookup: {"build_insights": ["market_insights.py"], ...}
    Used for smart retrieval when query mentions a function/class name.
    """
    index = defaultdict(set)
    for c in chunks:
        if c.get("symbols"):
            for sym in c["symbols"].split():
                index[sym].add(c["file"])
    return {k: list(v) for k, v in index.items()}


def _structured_query_results(query: str) -> list[dict]:
    """
    Return synthetic results for dependency/config/module questions.
    This allows structure-first answers without relying on semantic chunk matches.
    """
    q = query.lower()
    ws = _load_workspace_index()
    if not ws:
        return []

    def _as_chunk(title: str, lines: list[str]) -> dict:
        text = f"# {title}\n" + "\n".join(lines)
        return {
            "content": text,
            "file": "__workspace_index__",
            "language": "meta",
            "symbols": "",
            "score": 0.0,
            "_rank": 1.0,
            "full_file": True,
            "coverage": {"returned": 1, "total": 1, "partial": False},
        }

    dep_intent = re.search(r"\b(dependenc|library|libraries|package|uses?|framework)\b", q)
    cfg_intent = re.search(r"\b(config|manifest|permission|capabilit|build|gradle|docker|env)\b", q)
    mod_intent = re.search(r"\b(module|monorepo|workspace|service|package|architecture|entry)\b", q)

    chunks = []
    if dep_intent:
        deps = ws.get("dependencies", {})
        lines = []
        for eco, items in deps.items():
            if items:
                lines.append(f"- {eco}: {', '.join(items[:25])}")
        if lines:
            chunks.append(_as_chunk("Declared Dependencies", lines))

    if cfg_intent:
        cfg = ws.get("config_graph", {})
        lines = []
        if cfg.get("build_systems"):
            lines.append(f"- Build systems: {', '.join(cfg['build_systems'])}")
        if cfg.get("capabilities"):
            lines.append(f"- Capabilities/permissions: {', '.join(cfg['capabilities'][:20])}")
        if cfg.get("config_files"):
            lines.append(f"- Config files: {', '.join(cfg['config_files'][:20])}")
        if lines:
            chunks.append(_as_chunk("Build & Config Graph", lines))

    if mod_intent:
        lines = []
        mods = ws.get("modules", [])
        if mods:
            lines.append("- Modules:")
            for m in mods[:15]:
                lines.append(f"  - {m['name']} ({m['kind']}) files={m['file_count']}")
        if ws.get("entry_points"):
            lines.append("- Entry points: " + ", ".join(ws["entry_points"][:20]))
        if lines:
            chunks.append(_as_chunk("Workspace Modules", lines))

    return chunks


def format_project_map_for_prompt(pmap: dict) -> str:
    """
    Format project map as a compact header for the system prompt.
    Keeps it under ~300 tokens so it doesn't crowd out code context.
    """
    if not pmap:
        return ""

    lines = [
        "## Project Context",
        f"Name: {pmap.get('project', 'unknown')}",
        f"Language: {pmap.get('language', 'unknown')}",
    ]

    if pmap.get("stack"):
        lines.append(f"Stack: {', '.join(pmap['stack'])}")
    if pmap.get("domain"):
        lines.append(f"Domain: {pmap['domain']}")
    if pmap.get("entry_point"):
        lines.append(f"Entry point: {pmap['entry_point']}")
    workspace = pmap.get("workspace", {})
    if workspace.get("repo_types"):
        lines.append(f"Repo types: {', '.join(workspace['repo_types'])}")
    if workspace.get("package_managers"):
        lines.append(f"Package managers: {', '.join(workspace['package_managers'])}")
    if workspace.get("entry_points"):
        lines.append("Detected entry points: " + ", ".join(workspace["entry_points"][:6]))

    # Key modules with their symbols
    fsyms = pmap.get("file_symbols", {})
    if fsyms:
        lines.append("Key modules:")
        for fname, syms in list(fsyms.items())[:10]:
            sym_str = ", ".join(syms[:6])
            lines.append(f"  - {fname}: {sym_str}")

    return "\n".join(lines)


def _load_project_map() -> dict:
    return _load_json(PROJECT_MAP)


def _load_workspace_index() -> dict:
    return _load_json(WORKSPACE_INDEX)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _discover_manifests(root_path: Path) -> list[str]:
    manifests = []
    for mf in MANIFEST_FILES:
        if (root_path / mf).exists():
            manifests.append(mf)

    # common nested manifests in multi-module repos
    for pattern in ("**/package.json", "**/go.mod", "**/Cargo.toml", "**/pom.xml", "**/build.gradle"):
        for p in root_path.glob(pattern):
            if any(s in p.parts for s in SKIP_DIRS):
                continue
            rel = str(p.relative_to(root_path))
            if rel not in manifests:
                manifests.append(rel)
    return sorted(manifests)


def _detect_repo_types(manifests: list[str]) -> list[str]:
    types = set()
    joined = " ".join(manifests).lower()
    if any(x in joined for x in ("requirements.txt", "pyproject.toml", "pipfile")):
        types.add("python")
    if "package.json" in joined:
        types.add("node")
    if any(x in joined for x in ("build.gradle", "pom.xml")):
        types.add("jvm")
    if "go.mod" in joined:
        types.add("go")
    if "cargo.toml" in joined:
        types.add("rust")
    if any(x in joined for x in ("androidmanifest.xml", "build.gradle")):
        types.add("android")
    if any(x in joined for x in ("podfile", "package.swift")):
        types.add("ios")
    if len(types) > 1:
        types.add("polyglot")
    return sorted(types)


def _detect_package_managers(manifests: list[str]) -> list[str]:
    mgrs = set()
    for m in manifests:
        base = Path(m).name.lower()
        if base in {"requirements.txt", "pyproject.toml", "poetry.lock", "pipfile"}:
            mgrs.add("pip/poetry")
        if base in {"package.json", "pnpm-workspace.yaml", "yarn.lock"}:
            mgrs.add("npm/yarn/pnpm")
        if base in {"build.gradle", "build.gradle.kts"}:
            mgrs.add("gradle")
        if base == "cargo.toml":
            mgrs.add("cargo")
        if base == "go.mod":
            mgrs.add("go-mod")
        if base == "pom.xml":
            mgrs.add("maven")
    return sorted(mgrs)


def _discover_modules(root_path: Path, files: list[Path]) -> list[dict]:
    groups = defaultdict(list)
    for fp in files:
        rel = fp.relative_to(root_path)
        if len(rel.parts) == 1:
            groups["root"].append(rel)
        else:
            groups[rel.parts[0]].append(rel)

    modules = []
    for name, rels in groups.items():
        ext_counts = defaultdict(int)
        for r in rels:
            ext_counts[r.suffix] += 1
        modules.append({
            "name": name,
            "kind": "root" if name == "root" else "directory",
            "file_count": len(rels),
            "languages": sorted(ext_counts, key=ext_counts.get, reverse=True)[:3],
        })
    modules.sort(key=lambda m: m["file_count"], reverse=True)
    return modules[:40]


def _discover_entry_points(root_path: Path, files: list[Path]) -> list[str]:
    names = {str(fp.relative_to(root_path)): fp.name for fp in files}
    patterns = (
        r"(^|/)(main|app|server|index|run)\.(py|js|ts|go|rs|java|kt|swift)$",
        r"(^|/)__main__\.py$",
    )
    found = []
    for rel in names:
        if any(re.search(p, rel) for p in patterns):
            found.append(rel)
    return sorted(found)[:30]


def _collect_declared_dependencies(root_path: Path, manifests: list[str]) -> dict:
    deps = defaultdict(list)
    for manifest in manifests:
        mf = root_path / manifest
        base = mf.name
        try:
            text = mf.read_text(errors="ignore")
        except Exception:
            continue
        if base == "requirements.txt":
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                pkg = re.split(r"[<>=!~\[]", line, maxsplit=1)[0].strip()
                if pkg:
                    deps["python"].append(pkg)
        elif base == "pyproject.toml":
            for m in re.finditer(r'^\s*([A-Za-z0-9_.\-]+)\s*=', text, re.MULTILINE):
                name = m.group(1)
                if name not in {"name", "version", "description", "requires-python"}:
                    deps["python"].append(name)
        elif base == "package.json":
            try:
                pkg = json.loads(text)
                combo = {}
                combo.update(pkg.get("dependencies", {}))
                combo.update(pkg.get("devDependencies", {}))
                combo.update(pkg.get("peerDependencies", {}))
                deps["node"].extend(combo.keys())
            except Exception:
                pass
        elif base == "go.mod":
            for m in re.finditer(r"^\s*require\s+([^\s]+)", text, re.MULTILINE):
                deps["go"].append(m.group(1))
        elif base == "Cargo.toml":
            for m in re.finditer(r'^\s*([A-Za-z0-9_\-]+)\s*=', text, re.MULTILINE):
                name = m.group(1)
                if name not in {"package", "dependencies", "dev-dependencies", "features"}:
                    deps["rust"].append(name)
        elif base in {"build.gradle", "build.gradle.kts"}:
            for m in re.finditer(r'["\']([A-Za-z0-9_.\-]+:[A-Za-z0-9_.\-]+:[^"\']+)["\']', text):
                deps["jvm"].append(m.group(1))

    # normalize
    cleaned = {}
    for eco, items in deps.items():
        unique = sorted(set(i for i in items if len(i) > 1))
        if unique:
            cleaned[eco] = unique[:250]
    return cleaned


def _build_import_graph(files: list[Path], root_path: Path) -> dict:
    edges = []
    samples = defaultdict(list)
    for fp in files:
        try:
            text = fp.read_text(errors="ignore")
        except Exception:
            continue
        rel = str(fp.relative_to(root_path))
        suffix = fp.suffix
        imported = set()

        if suffix == ".py":
            imported.update(re.findall(r"^\s*import\s+([A-Za-z0-9_\.]+)", text, re.MULTILINE))
            imported.update(re.findall(r"^\s*from\s+([A-Za-z0-9_\.]+)\s+import", text, re.MULTILINE))
        elif suffix in {".js", ".ts", ".jsx", ".tsx"}:
            imported.update(re.findall(r"from\s+[\"']([^\"']+)[\"']", text))
            imported.update(re.findall(r"require\([\"']([^\"']+)[\"']\)", text))
        elif suffix in {".go"}:
            imported.update(re.findall(r'"([^"]+)"', "\n".join(re.findall(r"import\s*\((.*?)\)", text, re.DOTALL))))
        elif suffix in {".java", ".kt"}:
            imported.update(re.findall(r"^\s*import\s+([A-Za-z0-9_.*]+)", text, re.MULTILINE))
        elif suffix == ".rs":
            imported.update(re.findall(r"^\s*use\s+([A-Za-z0-9_:]+)", text, re.MULTILINE))

        for dst in sorted(imported):
            edges.append({"from": rel, "to": dst})
            if len(samples[rel]) < 5:
                samples[rel].append(dst)

    return {"edge_count": len(edges), "samples": dict(list(samples.items())[:120])}


def _build_config_graph(root_path: Path, manifests: list[str]) -> dict:
    build_systems = set()
    capabilities = set()
    config_files = sorted(manifests)[:300]
    for mf in manifests:
        base = Path(mf).name.lower()
        if base in {"package.json", "pnpm-workspace.yaml", "yarn.lock"}:
            build_systems.add("node")
        if base in {"build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts"}:
            build_systems.add("gradle")
        if base in {"go.mod"}:
            build_systems.add("go")
        if base in {"cargo.toml"}:
            build_systems.add("cargo")
        if base in {"dockerfile", "docker-compose.yml"}:
            build_systems.add("docker")

        try:
            text = (root_path / mf).read_text(errors="ignore").lower()
        except Exception:
            continue
        if "internet" in text or "network" in text:
            capabilities.add("network")
        if "camera" in text:
            capabilities.add("camera")
        if "microphone" in text:
            capabilities.add("microphone")
        if "location" in text:
            capabilities.add("location")
        if "bluetooth" in text:
            capabilities.add("bluetooth")
        if "notification" in text:
            capabilities.add("notifications")
        if "filesystem" in text or "storage" in text:
            capabilities.add("storage")

    return {
        "build_systems": sorted(build_systems),
        "capabilities": sorted(capabilities),
        "config_files": config_files,
    }


def _retrieve_config_first(
    query: str,
    intent: str,
    n_results: int = 5,
    ambiguous: bool = False,
    allow_runtime_fallback: bool = False,
    strict_authority_mode: bool = True,
) -> list[dict]:
    """
    Deterministic source-of-truth retrieval path for config/declaration/dependency
    questions. Pull config/manifests first, then fallback to inferred source code.
    """
    workspace = _load_workspace_index()
    priority_files = config_priority_files(
        intent,
        query,
        workspace.get("manifests", []),
        workspace.get("config_graph", {}).get("config_files", []),
    )

    collected = []
    found_authoritative = False
    for fname in priority_files:
        file_chunks = _fetch_all_from_file(fname, max_results=40)
        if not file_chunks:
            continue
        found_authoritative = True
        source_type = classify_source_type(fname)
        authority = authority_level_for_source(intent, source_type)
        annotate_sources(file_chunks, source_type=source_type, authority_level=authority)
        collected.extend(file_chunks[:6])
        if len(collected) >= n_results:
            break

    q = query.lower()
    asks_permissions = any(k in q for k in ("permission", "permissions", "declared", "manifest"))
    if asks_permissions:
        permissions = summarize_declared_permissions(collected)
        if permissions:
            summary = {
                "content": "# Manifest Declarations\n"
                + "\n".join([f"- declared permission: {p}" for p in permissions]),
                "file": "AndroidManifest.xml",
                "language": "xml",
                "symbols": "",
                "score": 0.0,
                "_rank": 1.0,
                "full_file": True,
                "coverage": {"returned": 1, "total": 1, "partial": False},
                "source_type": "manifest",
                "authority_level": "declared",
            }
            return [summary] + collected[: max(0, n_results - 1)]

        # Explicit missing-manifest declaration path
        if not any(c.get("file", "").endswith("AndroidManifest.xml") for c in collected):
            explicit = missing_manifest_notice()
            explicit.update({
                "symbols": "",
                "score": 0.0,
                "_rank": 1.0,
                "full_file": True,
                "coverage": {"returned": 1, "total": 1, "partial": False},
            })
            if (not strict_authority_mode) and allow_runtime_fallback and wants_runtime_usage(query):
                fallback = search_semantic_only(query, n_results=max(1, n_results - 1))
                annotate_sources(fallback, source_type="source_code", authority_level="referenced")
                return [explicit] + fallback
            return [explicit]

    if collected:
        return collected[:n_results]

    limitation = {
        "content": (
            "# Source-of-Truth Limitation\n"
            "- No authoritative declaration/configuration files were retrieved.\n"
            "- Declared/configured facts cannot be confirmed from source-of-truth artifacts.\n"
            "- Ask for runtime usage/references explicitly to include inferred code behavior.\n"
            + ("- Query intent appears ambiguous; clarify whether you want declarations or runtime usage.\n" if ambiguous else "")
        ),
        "file": "__source_of_truth_missing__",
        "language": "meta",
        "symbols": "",
        "score": 0.0,
        "_rank": 1.0,
        "full_file": True,
        "coverage": {"returned": 1, "total": 1, "partial": False},
        "source_type": "inferred",
        "authority_level": "inferred",
    }
    if (
        not strict_authority_mode
        and not found_authoritative
        and allow_runtime_fallback
        and wants_runtime_usage(query)
    ):
        fallback = search_semantic_only(query, n_results=max(1, n_results - 1))
        annotate_sources(fallback, source_type="source_code", authority_level="referenced")
        return [limitation] + fallback
    return [limitation]


def search_semantic_only(query: str, n_results: int = 5) -> list[dict]:
    """Semantic retrieval path without config-first routing."""
    count = col.count()
    if count == 0:
        return []

    embedding = embedder.encode(query).tolist()
    fetch = min(n_results * 6, count)
    results = col.query(query_embeddings=[embedding], n_results=fetch)

    candidates = [
        {
            "content": doc,
            "file": results["metadatas"][0][i].get("file", ""),
            "language": results["metadatas"][0][i].get("language", ""),
            "symbols": results["metadatas"][0][i].get("symbols", ""),
            "score": results["distances"][0][i],
        }
        for i, doc in enumerate(results["documents"][0])
    ]
    ranked = _rerank(query, candidates)
    return _add_coverage(ranked[:n_results])

def _fetch_all_from_file(filename: str, max_results: int = 20) -> list:
    """Retrieve all indexed chunks for a given filename."""
    count = col.count()
    if count == 0:
        return []

    # Query with a high n_results and filter by file metadata
    try:
        results = col.get(
            where={"file": {"$contains": filename.split("/")[-1]}},
            limit=max_results,
        )
        if not results or not results.get("documents"):
            return []

        chunks = [
            {
                "content":   doc,
                "file":      results["metadatas"][i].get("file", ""),
                "language":  results["metadatas"][i].get("language", ""),
                "symbols":   results["metadatas"][i].get("symbols", ""),
                "score":     0.0,
                "_rank":     1.0,
                "full_file": True,
            }
            for i, doc in enumerate(results["documents"])
            if doc
        ]
        # Sort by line number for coherent reading order
        chunks.sort(key=lambda c: c.get("line", 0) if "line" in c else 0)
        return chunks
    except Exception:
        return []


def _add_coverage(chunks: list) -> list:
    """
    Add coverage metadata to each chunk:
    how many chunks exist for that file vs how many we're returning.
    This lets the server tell the model if it has partial file coverage.
    """
    # Count total chunks per file in the index
    file_chunk_counts = {}
    for c in chunks:
        fname = c["file"]
        if fname not in file_chunk_counts:
            try:
                result = col.get(where={"file": {"$contains": fname.split("/")[-1]}})
                file_chunk_counts[fname] = len(result.get("documents", []))
            except Exception:
                file_chunk_counts[fname] = 0

    returned_per_file = defaultdict(int)
    for c in chunks:
        returned_per_file[c["file"]] += 1

    for c in chunks:
        fname   = c["file"]
        total   = file_chunk_counts.get(fname, 0)
        returned= returned_per_file[fname]
        c["coverage"] = {
            "returned": returned,
            "total":    total,
            "partial":  total > 0 and returned < total,
        }
    return chunks


def _camel_tokens(s: str) -> set:
    """Split camelCase/PascalCase/snake_case into component words."""
    parts = re.sub(r"([A-Z][a-z]+|[A-Z]+(?=[A-Z]|$))", r"_\1", s).lower()
    return set(re.findall(r"\w+", parts))


def _rerank(query: str, candidates: list) -> list:
    """
    Score each candidate on four axes then sort descending.

    Axes:
      base       — cosine similarity converted to 0-1 (1 = best)
      kw_bonus   — query term overlap with chunk text, tf-weighted
      sym_bonus  — query term overlap with declared symbols (higher weight)
      phrase     — bonus when a multi-word query phrase appears verbatim
      len_penalty— penalise very short chunks (< 5 lines) — usually noise
    """
    q_lower = query.lower()
    terms   = set(re.findall(r"\w+", q_lower))
    # Expand query terms with camelCase decomposition
    q_tokens = terms.copy()
    for t in list(terms):
        q_tokens |= _camel_tokens(t)

    # Build bigrams from query for phrase-match bonus
    q_words  = re.findall(r"\w+", q_lower)
    q_bigrams = {f"{q_words[i]} {q_words[i+1]}" for i in range(len(q_words)-1)}

    for c in candidates:
        content  = c["content"].lower()
        cwords   = set(re.findall(r"\w+", content))
        n_lines  = content.count("\n") + 1

        # Base similarity score (ChromaDB distance → similarity)
        base = max(0.0, 1.0 - (c["score"] / 2.0))

        # Keyword overlap — weight by rarity (terms that appear in few words get more credit)
        matched  = q_tokens & cwords
        kw_bonus = min(len(matched) / max(len(q_tokens), 1) * 0.25, 0.25)

        # Symbol bonus — exact symbol name match carries more signal
        sym_bonus = 0.0
        if c.get("symbols"):
            sw = set(re.findall(r"\w+", c["symbols"].lower()))
            sw |= _camel_tokens(c["symbols"])
            sym_bonus = min(len(q_tokens & sw) / max(len(q_tokens), 1) * 0.20, 0.20)

        # Phrase bonus — verbatim multi-word matches are very strong signal
        phrase_bonus = 0.0
        for bg in q_bigrams:
            if bg in content:
                phrase_bonus = min(phrase_bonus + 0.08, 0.16)

        # Length penalty — very short chunks are usually incomplete / noise
        len_penalty = 0.05 if n_lines < 4 else 0.0

        c["_rank"] = base + kw_bonus + sym_bonus + phrase_bonus - len_penalty

    return sorted(candidates, key=lambda x: x["_rank"], reverse=True)


def _collect_files(root: Path) -> list:
    files = []
    for fp in root.rglob("*"):
        if fp.is_file() and fp.suffix in SUPPORTED_EXTENSIONS:
            if not any(s in fp.parts for s in SKIP_DIRS):
                files.append(fp)
    return sorted(files)


def _chunk_file(fp: Path, root: Path) -> list:
    text = fp.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return []

    rel_path = str(fp.relative_to(root))
    language = fp.suffix.lstrip(".")
    lines    = text.splitlines()
    symbols  = _extract_symbols(text, language)
    pattern  = _BOUNDARY.get(language)

    if pattern and len(lines) > CHUNK_LINES:
        chunks = _boundary_chunks(text, rel_path, language, symbols, pattern)
        if chunks:
            return chunks
    return _line_chunks(lines, rel_path, language, symbols)


def _boundary_chunks(text, rel_path, language, symbols, pattern) -> list:
    lines      = text.splitlines()
    matches    = list(pattern.finditer(text))
    if len(matches) < 2:
        return []

    boundaries = [text[:m.start()].count("\n") for m in matches] + [len(lines)]
    chunks     = []

    for idx, start in enumerate(boundaries[:-1]):
        end   = boundaries[idx + 1]
        block = lines[start:end]
        if len(block) > CHUNK_LINES * 2:
            chunks.extend(_line_chunks(block, rel_path, language, symbols, start))
        elif block:
            ct = "\n".join(block).strip()
            if ct:
                chunks.append(_make_chunk(rel_path, language, symbols, ct, start))
    return chunks


def _line_chunks(lines, rel_path, language, symbols, line_offset=0) -> list:
    chunks = []
    i = 0
    while i < len(lines):
        text = "\n".join(lines[i: i + CHUNK_LINES]).strip()
        if text:
            chunks.append(_make_chunk(rel_path, language, symbols, text, line_offset + i))
        i += CHUNK_LINES - CHUNK_OVERLAP
    return chunks


def _make_chunk(rel_path, language, symbols, text, line) -> dict:
    chunk_id = hashlib.md5(f"{rel_path}:{line}".encode()).hexdigest()
    return {
        "id":       chunk_id,
        "content":  f"# File: {rel_path}\n\n{text}",
        "file":     rel_path,
        "language": language,
        "line":     line,
        "symbols":  symbols,
    }


def _extract_symbols(text: str, language: str) -> str:
    patterns = {
        "py":   r"(?:def|class)\s+(\w+)",
        "js":   r"(?:function|class)\s+(\w+)",
        "ts":   r"(?:function|class|interface|type)\s+(\w+)",
        "java": r"(?:class|interface)\s+(\w+)",
        "kt":   r"(?:fun|class|object|interface)\s+(\w+)",
        "swift":r"(?:func|class|struct|protocol)\s+(\w+)",
        "go":   r"(?:func|type)\s+(\w+)",
        "rs":   r"(?:fn|struct|enum|trait|impl)\s+(\w+)",
    }
    pat = patterns.get(language)
    if not pat:
        return ""
    return " ".join(re.findall(pat, text, re.MULTILINE)[:50])


def _set_repo_fingerprint(fingerprint: str) -> None:
    global CURRENT_REPO_FINGERPRINT
    CURRENT_REPO_FINGERPRINT = fingerprint or ""


def get_repo_fingerprint() -> str:
    if CURRENT_REPO_FINGERPRINT:
        return CURRENT_REPO_FINGERPRINT
    hashes = _load_hashes()
    return hashes.get("__fingerprint__", "")


def _build_file_to_module_map(root_path: Path, files: list[Path]) -> dict:
    mapping = {}
    for fp in files:
        rel = str(fp.relative_to(root_path))
        if len(fp.relative_to(root_path).parts) == 1:
            mapping[rel] = "root"
        else:
            mapping[rel] = fp.relative_to(root_path).parts[0]
    return mapping


def _file_hash(fp: Path) -> str:
    return hashlib.md5(fp.read_bytes()).hexdigest()


def _load_hashes() -> dict:
    try:
        if HASH_STORE.exists():
            return json.loads(HASH_STORE.read_text())
    except Exception:
        pass
    return {}


def _save_hashes(hashes: dict) -> None:
    try:
        HASH_STORE.parent.mkdir(parents=True, exist_ok=True)
        HASH_STORE.write_text(json.dumps(hashes))
    except Exception as e:
        logging.warning(f"Could not save hashes: {e}")


def _load_json(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}


def _save_json(path: Path, data: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logging.warning(f"Could not save {path.name}: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 indexer.py <path-to-codebase>")
        sys.exit(1)

    for event in index_codebase_stream(sys.argv[1]):
        t = event["type"]
        if t == "scan":
            reused = event.get("unchanged", 0)
            suffix = f" ({event['new']} new, {reused} reused)" if reused else ""
            print(f"📂  Found {event['files']} source files{suffix}")
        elif t == "chunks":
            print(f"⚙️   {event['total']} chunks to embed")
        elif t == "embed":
            print(f"\r   Embedding {event['done']}/{event['total']} ({event['pct']}%)",
                  end="", flush=True)
        elif t == "store" and event["done"] == event["total"]:
            print(f"\n   Stored {event['done']} chunks")
        elif t == "mapping":
            print(f"🗺   {event['message']}")
        elif t == "done":
            reused = event.get("reused", 0)
            suffix = f" ({event['new']} new, {reused} reused)" if reused else ""
            pmap   = event.get("map", {})
            print(f"✅  Done — {event['indexed']} files{suffix} · {event['chunks']} chunks")
            if pmap:
                print(f"    Language: {pmap.get('language')} · Domain: {pmap.get('domain')}")
                print(f"    Entry: {pmap.get('entry_point')} · Stack: {pmap.get('stack')}")
        elif t == "error":
            print(f"❌  {event['message']}")
