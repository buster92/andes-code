# ── Offline enforcement — MUST be before any library import ──────────────────
# Zero egress enforced at OS level — no outbound connections during inference.
import os
os.environ["TRANSFORMERS_OFFLINE"]              = "1"
os.environ["HF_DATASETS_OFFLINE"]              = "1"
os.environ["HF_HUB_OFFLINE"]                   = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"]            = "false"

import contextlib
import hashlib
import inspect
import json
import logging
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid
import threading
from datetime import datetime, timezone
from pathlib import Path
from urllib import error as url_error
from urllib import request as url_request

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")
from runtime_paths import get_runtime_log_path
from conversation_export import write_conversation_export

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from llama_cpp import Llama, LlamaCache
from andes_cache import (
    build_prompt_sections,
    serialize_prompt_sections,
    compute_context_budget,
    estimate_tokens,
    pack_chunks_to_budget,
)
from andes_cache.debug import resolve_debug_mode, env_debug_mode, format_debug_sse_event
from andes_cache.source_of_truth import source_of_truth_guidance, is_declaration_query, has_declaration_keywords
from andes_cache.routing import (
    classify_query_intent,
    classify_query_intent_details,
    retrieval_route_for_intent,
    is_fast_path_intent,
    semantic_cache_allowed,
    orchestration_plan,
)
from ask_orchestrator import (
    FunctionAnswerContextBuilder,
    FunctionRetrievalProvider,
    LlamaAnswerEngine,
    LocalAskOrchestrator,
)
from execution_mode import ExecutionMode, execution_mode_env_key, get_execution_mode
from local_retrieval import normalize_local_retrieval
from edit_suggestion import (
    EDIT_SUGGESTION,
    build_edit_suggestion_context,
    edit_suggestion_policy,
    enforce_edit_suggestion_output,
    is_edit_suggestion_query,
    safe_context_fallback,
)
from remote_inference_schema import (
    RemoteInferenceRequest,
    RemoteProtocol,
    SchemaValidationError,
)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
MODEL_PATH     = str(BASE_DIR / os.getenv("MODEL_PATH", "models/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"))
LOG_PATH       = get_runtime_log_path("server")
PORT           = int(os.getenv("PORT", 8080))
CONTEXT_CHUNKS = int(os.getenv("CONTEXT_CHUNKS", 5))
CACHE_SIZE_GB  = float(os.getenv("CACHE_SIZE_GB", 2.0))
MODEL_CONTEXT_WINDOW = int(os.getenv("MODEL_CONTEXT_WINDOW", 8192))
CONTEXT_RESERVED_RESPONSE_TOKENS = int(os.getenv("CONTEXT_RESERVED_RESPONSE_TOKENS", 1400))
CONTEXT_SAFETY_MARGIN_TOKENS = int(os.getenv("CONTEXT_SAFETY_MARGIN_TOKENS", 256))
REMOTE_INFERENCE_SERVER_URL = os.getenv("ANDESCODE_REMOTE_SERVER_URL", "http://127.0.0.1:8080").rstrip("/")
# Snapshot for startup visibility only.
# NOTE: If you change ANDESCODE_DEBUG_MODE in .env while the server is running,
# this startup value will not change until restart. Request-level debug still
# resolves per request via _resolve_request_debug_mode().
DEBUG_MODE_STARTUP = env_debug_mode()


def _resolve_request_debug_mode(api_debug: bool | str | None) -> bool:
    """Resolve debug mode for each request: API checkbox/body flag > current env."""
    return resolve_debug_mode(request_flag=api_debug, env_flag=env_debug_mode())

# Core system prompt — kept short and static for KV cache reuse
_BASE_SYSTEM = (
    "You are AndesCode, an expert AI coding assistant with deep knowledge of "
    "the developer's actual codebase. You have been given:\n"
    "1. A structured map of the project (entry points, modules, domain)\n"
    "2. The most relevant code excerpts for the current question\n\n"
    "Rules:\n"
    "- Reference specific files and functions in your answers\n"
    "- If you see a coverage warning, acknowledge you may have partial context\n"
    "- Never invent code that wasn't in the provided excerpts\n"
    "- Be direct and precise — this is a professional dev environment"
)

_PERFORMANCE_QUERY_RE = re.compile(
    r"\b(slow|lag|jank|main\s+thread|scroll|performance|frame|cpu|render)\b",
    re.IGNORECASE,
)

_HIGH_SIGNAL_PERFORMANCE_POLICY = (
      "High-Signal Performance Analysis Mode (auto-enforced for performance queries):\n"
    "- Use structured execution paths first as candidate hot-path hints.\n"
    "- Validate each candidate path against the supporting retrieved code snippets before final claims.\n"
    "- For each path finding, explicitly include:\n"
    "  a) execution frequency: once / per event / per frame\n"
    "  b) thread: main / background (mark proven vs inferred)\n"
    "  c) relative cost: low / medium / high\n"
    "  d) risk: main-thread blocking risk yes/no with rationale\n"
    "  e) evidence class tags on claims: PROVEN / INFERRED / SPECULATIVE\n"
    "- Classification rules (strict):\n"
    "  * PROVEN = directly visible in retrieved code.\n"
    "  * INFERRED = framework behavior or partial evidence.\n"
    "  * SPECULATIVE = requires runtime confirmation or missing context.\n"
    "  * Thread execution MUST NOT be PROVEN unless full Rx/Coroutine chain is visible.\n"
    "  * If thread is inferred from patterns (e.g., observeOn(AndroidSchedulers.mainThread())), mark INFERRED.\n"
    "  * Any performance impact claim (frame drops/jank/etc.) is SPECULATIVE unless measured evidence is shown.\n"
    "- Explicitly map each path step to cost contribution (which step is expensive and why).\n"
    "- Ignore architecture-only explanations (DI, repository, use case) and lifecycle-only setup (onCreate/init/setup).\n"
    "- Keep only hot paths: high frequency AND non-trivial cost.\n"
    "- Rank findings by (frequency × cost × thread impact) in descending order.\n"
    "- Return only top relevant items (3–5 max).\n"
    "- Every finding must include a causal chain with explicit separation:\n"
    "  Code path: data flow → execution\n"
    "  Predicted UI impact: <impact> (SPECULATIVE unless measured)\n"
    "  Example:\n"
    "  Code path: ViewModel emit (PROVEN) → adapter update (PROVEN) → RecyclerView bind (INFERRED)\n"
    "  Predicted UI impact: May drop frames (SPECULATIVE, depends on runtime profile).\n"
    "- Before finalizing, remove any finding that is infrequent or negligible cost.\n"
    "- If no high-signal findings remain, explicitly say: No high-signal hot paths found."
)

# ── Audit log ─────────────────────────────────────────────────────────────────
audit = logging.getLogger("andescode.audit")
audit.setLevel(logging.INFO)
audit.propagate = False
_fh = logging.FileHandler(LOG_PATH)
_fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
audit.addHandler(_fh)

# ── Log path sanitizer ────────────────────────────────────────────────────────
# Strips absolute paths / home directory so audit.log never leaks usernames
# or internal dir structure.  Only short relative paths survive.
_HOME = str(Path.home())

def _safe(value: object) -> str:
    """Return a log-safe string: strip home prefix, truncate long paths."""
    text = str(value)
    # Remove home directory prefix (contains username on all platforms)
    text = text.replace(_HOME + "/", "~/").replace(_HOME + "\\", "~/")
    # Replace any remaining absolute paths  /foo/bar/baz → .../baz
    text = re.sub(r"(?:^|\s)/(?:[\w./\-]+/)(\w[\w.\-]*)", r".../ \1", text)
    return text


def _phase_log(request_id: str, phase: str, **fields) -> None:
    payload = " | ".join([f"{k}={_safe(v)}" for k, v in fields.items()])
    message = f"CHAT {request_id} | phase={phase}"
    if payload:
        message = f"{message} | {payload}"
    audit.info(message)


def _is_performance_query(query: str) -> bool:
    return bool(query and _PERFORMANCE_QUERY_RE.search(query))


def _reasoning_policy_for_query(query: str) -> tuple[str, str]:
    # Explicit edit requests get the strict read-only patch recommendation contract.
    if is_edit_suggestion_query(query):
        return (edit_suggestion_policy(), EDIT_SUGGESTION)
    # Pure performance analysis remains high-signal and execution-path focused.
    if _is_performance_query(query):
        return (_HIGH_SIGNAL_PERFORMANCE_POLICY, "performance")
    # Declaration/dependency queries get structured source-of-truth guidance so the
    # model always separates declared findings from inferred ones and never silently
    # falls back to code-usage inference without acknowledging missing authority files.
    intent = classify_query_intent(query)
    sot_guidance = source_of_truth_guidance(query, intent)
    if sot_guidance:
        return (sot_guidance, "declaration")
    return ("", "general")


def _validate_high_signal_output(text: str, enabled: bool) -> tuple[str, int]:
    if not enabled or not text.strip():
        return text, 0
    blocks = [b for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    if not blocks:
        return text, 0

    architecture_markers = (
        "dependency injection", "di", "repository", "use case",
        "clean architecture", "mvvm", "mvi", "oncreate", "init", "setup",
    )
    hot_path_markers = (
        "per frame", "per event", "frame", "scroll", "jank", "render",
        "recyclerview", "main thread", "cpu", "bind", "gesture",
    )

    kept_blocks: list[str] = []
    filtered_out = 0
    for block in blocks:
        normalized = block.lower()
        has_architecture = any(marker in normalized for marker in architecture_markers)
        has_hot_path = any(marker in normalized for marker in hot_path_markers)
        if has_architecture and not has_hot_path:
            filtered_out += 1
            continue
        kept_blocks.append(_annotate_high_signal_block(block))

    if not kept_blocks:
        return "No high-signal hot paths found.", filtered_out
    return "\n\n".join(kept_blocks).strip(), filtered_out


def _annotate_high_signal_block(block: str) -> str:
    """Nudge performance findings toward explicit evidence-class labeling."""
    lines = block.splitlines()
    out: list[str] = []
    class_marker_re = re.compile(r"\b(PROVEN|INFERRED|SPECULATIVE)\b", re.IGNORECASE)
    thread_line_re = re.compile(r"^(\s*[-*]?\s*thread\s*:\s*)(.+)$", re.IGNORECASE)
    impact_line_re = re.compile(r"^(\s*[-*]?\s*(?:impact|ui impact|predicted ui impact)\s*:\s*)(.+)$", re.IGNORECASE)
    impact_keywords_re = re.compile(r"\b(frame\s*drops?|jank|stutter|slow|lag|fps)\b", re.IGNORECASE)
    main_thread_hint_re = re.compile(r"\b(main|ui)\s*thread\b|\bmain\b|\bui\b", re.IGNORECASE)

    for line in lines:
        if class_marker_re.search(line):
            out.append(line)
            continue

        thread_match = thread_line_re.match(line)
        if thread_match:
            prefix, value = thread_match.groups()
            if main_thread_hint_re.search(value):
                out.append(
                    f"{prefix}Likely Main Thread (INFERRED from thread-affinity pattern; full Rx/Coroutine chain not fully visible)"
                )
            else:
                out.append(
                    f"{prefix}Likely {value.strip()} (INFERRED from partial code evidence)"
                )
            continue

        impact_match = impact_line_re.match(line)
        if impact_match:
            prefix, value = impact_match.groups()
            candidate = value.strip()
            if candidate:
                lowered = candidate.lower()
                if lowered.startswith("causes "):
                    candidate = f"May cause {candidate[7:]}"
                elif lowered.startswith("cause "):
                    candidate = f"May cause {candidate[6:]}"
                elif not lowered.startswith("may "):
                    candidate = f"May {candidate[0].lower() + candidate[1:]}"
            else:
                candidate = "May impact UI smoothness"
            out.append(
                f"{prefix}{candidate} (SPECULATIVE, requires runtime profiling/measurement)"
            )
            continue

        if impact_keywords_re.search(line):
            out.append(f"{line} (SPECULATIVE, predicted UI impact)")
            continue

        out.append(line)

    return "\n".join(out)


def _index_phase_log(source: str, phase: str, **fields) -> None:
    payload = " | ".join([f"{k}={_safe(v)}" for k, v in fields.items()])
    message = f"INDEX | source={source} | phase={phase}"
    if payload:
        message = f"{message} | {payload}"
    audit.info(message)


def _make_error_chunk(request_id: str, phase: str, err: Exception | str) -> str:
    err_text = _safe(err)
    return _make_chunk(
        (
            "\n\n❌ Sorry — AndesCode failed while processing this request. "
            f"(phase: {phase})\nDetails: {err_text}"
        ),
        request_id,
    )


def _make_pipeline_error_event(request_id: str, phase: str, err: Exception | str) -> str:
    payload = {
        "id": f"chatcmpl-{request_id}",
        "object": "andescode.error",
        "created": int(time.time()),
        "error": {
            "phase": phase,
            "message": _safe(err),
        },
    }
    return f"data: {json.dumps(payload)}\n\n"


def _minimal_debug_payload(
    *,
    query: str,
    request_id: str,
    stream_path: str,
    reason: str,
    cache_hit: bool = False,
    intent: str = "unknown",
    retrieval_route: str = "unknown",
    final_context: dict | None = None,
) -> dict:
    """Fallback debug payload so debug-mode requests always produce one visible debug event."""
    context_snapshot = final_context or {}
    files_used = context_snapshot.get("files_used") or []
    context_size = context_snapshot.get("context_size")
    if context_size is None:
        context_size = 0
    return {
        "query": query,
        "debug_enabled": True,
        "stream_path": stream_path or "unknown",
        "cache_hit": bool(cache_hit),
        "payload_kind": "fallback",
        "reason": reason,
        "request_id": request_id,
        "intent": intent or "unknown",
        "retrieval_route": retrieval_route or "unknown",
        "final_context": {
            "files_used": files_used,
            "context_size": context_size,
        },
    }


def _build_cache_debug_payload(
    *,
    query: str,
    request_id: str,
    repo_fp: str,
    retrieval_signature: str,
    intent: str,
    retrieval_route: str,
    semantic_hit,
) -> dict:
    """Build a minimal but structured debug payload for semantic cache short-circuits."""
    metadata = {}
    retrieved_chunks_count = 0
    if isinstance(semantic_hit, dict):
        metadata = semantic_hit.get("metadata", {}) or {}
        cached_chunks = semantic_hit.get("retrieved_chunks") or semantic_hit.get("chunks") or []
        if isinstance(cached_chunks, list):
            retrieved_chunks_count = len(cached_chunks)
    return {
        "query": query,
        "debug_enabled": True,
        "request_id": request_id,
        "stream_path": "semantic_cache_hit",
        "orchestration_path": "semantic_cache_hit",
        "payload_kind": "cache",
        "cache_hit": True,
        "retrieval_route": retrieval_route or "unknown",
        "intent": intent or "unknown",
        "retrieval": {
            "route_taken": "semantic_cache_hit",
            "retrieved_chunks_count": int(retrieved_chunks_count),
            "source": "cache",
        },
        "cache": {
            "repo_fp": repo_fp,
            "retrieval_signature": retrieval_signature,
            "metadata": metadata,
        },
        "final_context": {
            "files_used": [],
            "context_size": 0,
        },
    }

for _lib in ("httpx", "httpcore", "sentence_transformers", "transformers", "huggingface_hub"):
    logging.getLogger(_lib).setLevel(logging.ERROR)

# ── Startup helpers ───────────────────────────────────────────────────────────
def _print(msg: str) -> None:
    print(msg, flush=True)

@contextlib.contextmanager
def _suppress_stderr():
    devnull = open(os.devnull, "w")
    old_fd  = os.dup(2)
    os.dup2(devnull.fileno(), 2)
    try:
        yield
    finally:
        os.dup2(old_fd, 2)
        os.close(old_fd)
        devnull.close()

# ── Step 1: Model ─────────────────────────────────────────────────────────────
_print("")
_print("┌─────────────────────────────────────────┐")
_print("│  🏔️   AndesCode  —  Starting up          │")
_print("└─────────────────────────────────────────┘")
_print("")

if not Path(MODEL_PATH).exists():
    _print(f"  ❌  Model not found: {MODEL_PATH}")
    _print(f"      See README.md → Quick Start → Step 2")
    sys.exit(1)

_print(f"  [1/4] Loading Gemma 4 26B ({os.path.basename(MODEL_PATH)})...")

with _suppress_stderr():
    llm = Llama(
        model_path   = MODEL_PATH,
        n_ctx        = MODEL_CONTEXT_WINDOW,   # enough for project map + code chunks + answer
        n_batch      = 1024,
        n_gpu_layers = -1,
        n_threads    = 6,
        use_mmap     = True,
        use_mlock    = False,
        verbose      = False,
    )

_cache_bytes = int(CACHE_SIZE_GB * 1024 ** 3)
llm.set_cache(LlamaCache(capacity_bytes=_cache_bytes))
_print(f"  [1/4] ✓ Model loaded  (KV cache: {CACHE_SIZE_GB:.0f}GB)")

# ── Step 2: Indexer ───────────────────────────────────────────────────────────
_print(f"  [2/4] Loading embedding model...")

INDEXER_READY   = False
search_codebase = None
_indexer_module = None

def _load_indexer() -> bool:
    global INDEXER_READY, search_codebase, _indexer_module
    if INDEXER_READY:
        return True
    try:
        with _suppress_stderr():
            import indexer as _idx
        _indexer_module = _idx
        search_codebase = _idx.search
        INDEXER_READY   = True
        return True
    except Exception as e:
        _print(f"  [2/4] ⚠  Indexer not available: {e}")
        return False

_load_indexer()
if INDEXER_READY and _indexer_module:
    try:
        _indexer_module.run_startup_integrity_probe()
    except Exception as e:
        _print(f"  [2/4] ⚠  Integrity startup probe unavailable: {e}")
_print(f"  [2/4] ✓ Embedding model ready" if INDEXER_READY
       else f"  [2/4] ⚠  Indexer unavailable")

_index_run_lock = threading.Lock()
_index_status_lock = threading.Lock()
_current_index_source = ""
_freshness_status_message = "Index freshness: checked on ask"
_auto_index_manager = None  # compatibility only; no background refresh is started.



def _set_index_source(source: str) -> None:
    global _current_index_source
    with _index_status_lock:
        _current_index_source = source


def _index_progress_state() -> dict:
    with _index_status_lock:
        source = _current_index_source
    indexing = _index_run_lock.locked()
    return {
        "indexing_in_progress": indexing,
        "manual_index_in_progress": bool(indexing and source == "manual"),
        "indexing_source": source if indexing else "",
    }



def _call_index_stream_factory(factory, path: str, force_refresh: bool):
    try:
        sig = inspect.signature(factory)
        supports_force = (
            "force_refresh" in sig.parameters
            or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        )
    except (TypeError, ValueError):
        supports_force = True
    if supports_force:
        return factory(path, force_refresh=force_refresh)
    return factory(path)


def _call_run_index_stream_compat(run_fn, path: str, source: str, emit_event, *, force_refresh: bool):
    try:
        sig = inspect.signature(run_fn)
        supports_force = (
            "force_refresh" in sig.parameters
            or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        )
    except (TypeError, ValueError):
        supports_force = True
    if supports_force:
        return run_fn(path, source=source, emit_event=emit_event, force_refresh=force_refresh)
    return run_fn(path, source=source, emit_event=emit_event)


def _read_integrity_probe_from_indexer() -> dict:
    """Optional startup visibility helper that tolerates partial indexer stubs."""
    if not (INDEXER_READY and _indexer_module):
        return {}
    getter = getattr(_indexer_module, "get_startup_integrity_probe", None)
    if not callable(getter):
        return {}
    try:
        return getter() or {}
    except Exception:
        return {}


def _indexable_current_hashes(root_path: Path) -> dict[str, str]:
    """Return current hashes for indexable project files using indexer inclusion rules."""
    if not _indexer_module:
        return {}
    collect_files = getattr(_indexer_module, "_collect_files", None)
    file_hash = getattr(_indexer_module, "_file_hash", None)
    if not callable(collect_files) or not callable(file_hash):
        return {}
    snapshot: dict[str, str] = {}
    for fp in collect_files(root_path):
        try:
            snapshot[str(Path(fp).relative_to(root_path))] = file_hash(Path(fp))
        except Exception:
            continue
    return snapshot


def _freshness_change_signature(
    *,
    indexed_root: str,
    changed_count: int,
    deleted_count: int,
    changed_paths: list[str] | None = None,
    deleted_paths: list[str] | None = None,
) -> str:
    """Stable opaque signature for one observed freshness-change state."""
    digest_input = {
        "indexed_root": indexed_root,
        "changed_count": int(changed_count),
        "deleted_count": int(deleted_count),
        "changed_paths": sorted(changed_paths or []),
        "deleted_paths": sorted(deleted_paths or []),
    }
    encoded = json.dumps(digest_input, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def _index_freshness_payload() -> dict:
    t0 = time.perf_counter()
    checked_at = datetime.now(timezone.utc).isoformat()
    if not _load_indexer() or not _indexer_module:
        return {
            "ok": False,
            "has_index": False,
            "indexed_root": "",
            "changed": False,
            "changed_count": 0,
            "deleted_count": 0,
            "change_signature": "",
            "checked_at": checked_at,
            "check_duration_ms": int((time.perf_counter() - t0) * 1000),
            "error": "Indexer not available",
        }
    try:
        hashes = _indexer_module._load_hashes() if hasattr(_indexer_module, "_load_hashes") else {}
    except Exception as exc:
        return {
            "ok": False,
            "has_index": False,
            "indexed_root": "",
            "changed": False,
            "changed_count": 0,
            "deleted_count": 0,
            "change_signature": "",
            "checked_at": checked_at,
            "check_duration_ms": int((time.perf_counter() - t0) * 1000),
            "error": f"Unable to read index snapshot: {exc}",
        }
    indexed_root = str(hashes.get("__root__", "") or "") if isinstance(hashes, dict) else ""
    indexed_hashes = {k: v for k, v in (hashes or {}).items() if not str(k).startswith("__")}
    has_index = bool(indexed_root and indexed_hashes)
    if not indexed_root or not indexed_hashes:
        return {
            "ok": True,
            "has_index": False,
            "indexed_root": indexed_root,
            "changed": False,
            "changed_count": 0,
            "deleted_count": 0,
            "change_signature": "",
            "checked_at": checked_at,
            "check_duration_ms": int((time.perf_counter() - t0) * 1000),
        }
    root_path = Path(indexed_root)
    if not root_path.exists():
        deleted_paths = sorted(indexed_hashes)
        deleted_count = len(deleted_paths)
        return {
            "ok": True,
            "has_index": True,
            "indexed_root": indexed_root,
            "changed": True,
            "changed_count": 0,
            "deleted_count": deleted_count,
            "change_signature": _freshness_change_signature(
                indexed_root=indexed_root,
                changed_count=0,
                deleted_count=deleted_count,
                deleted_paths=deleted_paths,
            ),
            "checked_at": checked_at,
            "check_duration_ms": int((time.perf_counter() - t0) * 1000),
        }
    current_hashes = _indexable_current_hashes(root_path)
    current_paths = set(current_hashes)
    indexed_paths = set(indexed_hashes)
    changed_paths = sorted(rel for rel, digest in current_hashes.items() if indexed_hashes.get(rel) != digest)
    deleted_paths = sorted(indexed_paths - current_paths)
    changed_count = len(changed_paths)
    deleted_count = len(deleted_paths)
    changed = bool(changed_count or deleted_count)
    return {
        "ok": True,
        "has_index": has_index,
        "indexed_root": indexed_root,
        "changed": changed,
        "changed_count": changed_count,
        "deleted_count": deleted_count,
        "change_signature": (
            _freshness_change_signature(
                indexed_root=indexed_root,
                changed_count=changed_count,
                deleted_count=deleted_count,
                changed_paths=changed_paths,
                deleted_paths=deleted_paths,
            )
            if changed else ""
        ),
        "checked_at": checked_at,
        "check_duration_ms": int((time.perf_counter() - t0) * 1000),
    }


def _refresh_index_before_answer_if_needed(emit_event=None) -> tuple[bool, str, dict]:
    freshness = _index_freshness_payload()
    if not freshness.get("ok", False):
        return False, str(freshness.get("error") or "Unable to check index freshness"), freshness
    if not freshness.get("has_index", False) or not freshness.get("changed", False):
        return True, "", freshness
    root = str(freshness.get("indexed_root") or "")
    if not root:
        return False, "No indexed project root is available for refresh.", freshness
    events = []
    def _emit(event):
        events.append(event)
        if callable(emit_event):
            emit_event(event)
    ok = _run_index_stream(root, source="query", emit_event=_emit, force_refresh=False)
    if not ok:
        message = "Index refresh failed before answering."
        for event in reversed(events):
            if event.get("type") == "error" and event.get("message"):
                message = str(event.get("message"))
                break
        return False, message, freshness
    return True, "", freshness


def _run_index_stream(path: str, source: str, emit_event, change_batch=None, force_refresh: bool = False) -> bool:
    _index_phase_log(source, "index_request_started", path=path)
    if not _load_indexer():
        _index_phase_log(source, "index_failed", failed_phase="load_indexer", error="Indexer not available")
        emit_event({"type": "error", "source": source, "message": "Indexer not available"})
        return False

    acquired = _index_run_lock.acquire(blocking=False)
    if not acquired:
        emit_event({"type": "status", "source": source, "message": "Index already in progress"})
        return False

    _set_index_source(source)
    try:
        from indexer import index_codebase_stream
        done_event = None
        error_event = None
        embedding_started_logged = False
        storage_started_logged = False
        for event in _call_index_stream_factory(index_codebase_stream, path, force_refresh):
            event = dict(event)
            event["source"] = source
            etype = event.get("type")
            if etype == "scan":
                _index_phase_log(source, "scan_done", files=event.get("files"), new=event.get("new"), unchanged=event.get("unchanged"))
            elif etype == "embed":
                if not embedding_started_logged:
                    _index_phase_log(source, "embedding_started", total=event.get("total"))
                    embedding_started_logged = True
                if event.get("done") == event.get("total"):
                    _index_phase_log(source, "embedding_completed", total=event.get("total"))
            elif etype == "store":
                if not storage_started_logged:
                    _index_phase_log(source, "storage_started", total=event.get("total"))
                    storage_started_logged = True
                if event.get("done") == event.get("total"):
                    _index_phase_log(source, "storage_completed", total=event.get("total"))
            elif etype == "mapping":
                _index_phase_log(source, "project_map_workspace_build_started", message=event.get("message", "mapping"))
            elif etype == "error":
                error_event = event
            elif etype == "done":
                _index_phase_log(
                    source,
                    "index_completed",
                    indexed=event.get("indexed"),
                    chunks=event.get("chunks"),
                    decision=event.get("decision"),
                )
                _index_phase_log(source, "project_map_workspace_build_completed", indexed=event.get("indexed"))
            emit_event(event)
            if event.get("type") == "done":
                done_event = event

        return bool(done_event and not error_event)
    except Exception as exc:
        _index_phase_log(source, "index_failed", failed_phase="run_index_stream", error=exc)
        emit_event({"type": "error", "source": source, "message": str(exc)})
        return False
    finally:
        _set_index_source("")
        _index_run_lock.release()



# ── Step 3: KV cache warm-up ──────────────────────────────────────────────────
_print(f"  [3/4] Warming KV cache...")

def _warm_cache() -> None:
    t0 = time.perf_counter()
    warm_prompt = (
        f"<s>{_BASE_SYSTEM}</s>\n"
        "<start_of_turn>user\nReady.<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    try:
        for _ in llm(warm_prompt, max_tokens=1, stream=True, echo=False):
            break
        _print(f"  [3/4] ✓ Cache warm  ({time.perf_counter()-t0:.1f}s prefill cached)")
    except Exception as e:
        _print(f"  [3/4] ⚠  Cache warm-up failed: {e}")

_warm_cache()
_print(f"  [4/4] Starting server on port {PORT}...")

# ── Thinking tag pattern ──────────────────────────────────────────────────────
_THINK_PATTERN = re.compile(r"<\|channel>.*?<channel\|>", re.DOTALL)
_active_index_session = False
_active_index_lock = threading.Lock()


def _set_active_index_session(active: bool) -> None:
    global _active_index_session
    with _active_index_lock:
        _active_index_session = bool(active)


def _get_active_index_session() -> bool:
    with _active_index_lock:
        return bool(_active_index_session)


def _index_runtime_state() -> dict:
    doc_count = 0
    project_map = {}
    indexed_root = ""
    has_persisted_index = False
    if INDEXER_READY and _indexer_module:
        try:
            doc_count = int(_indexer_module.col.count())
        except Exception:
            doc_count = 0
        try:
            project_map = _indexer_module._load_project_map() if hasattr(_indexer_module, "_load_project_map") else {}
        except Exception:
            project_map = {}
        try:
            hashes = _indexer_module._load_hashes() if hasattr(_indexer_module, "_load_hashes") else {}
            indexed_root = str(hashes.get("__root__", "") or "") if isinstance(hashes, dict) else ""
        except Exception:
            indexed_root = ""
        has_persisted_index = bool(doc_count > 0)
    active = _get_active_index_session()
    return {
        "doc_count": doc_count,
        "has_persisted_index": has_persisted_index,
        "active_index_session": active,
        "restored_requires_confirmation": bool(has_persisted_index and not active),
        "project_map": project_map if isinstance(project_map, dict) else {},
        "indexed_root": indexed_root,
        **_index_progress_state(),
    }

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="AndesCode", version="1.0.0")

_static_dir = BASE_DIR / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/ui")
@app.get("/ui/")
def serve_ui():
    ui_path = BASE_DIR / "static" / "index.html"
    if ui_path.exists():
        return HTMLResponse(ui_path.read_text())
    return HTMLResponse("<h1>UI not found</h1>", status_code=404)


@app.get("/")
@app.get("/v1")
def root():
    runtime_state = _index_runtime_state()
    auto_state = {}
    auto_message = _freshness_status_message
    integrity_probe = _read_integrity_probe_from_indexer()
    # Expose both startup snapshot and current env state so UI/debug tooling can
    # distinguish "started with debug" from "debug is currently enabled now".
    debug_mode_env_current = env_debug_mode()
    return {
        "status":    "running",
        "product":   "AndesCode",
        "version":   "1.0.0",
        "indexer":   INDEXER_READY,
        **runtime_state,
        "cache":     f"{CACHE_SIZE_GB:.0f}GB",
        "debug_mode_startup": DEBUG_MODE_STARTUP,
        "debug_mode_env_current": debug_mode_env_current,
        "debug_mode": debug_mode_env_current,
        "auto_index": auto_state,
        "auto_index_message": auto_message,
        "integrity_probe": integrity_probe,
    }


@app.get("/v1/index/state")
def index_state():
    integrity_probe = _read_integrity_probe_from_indexer()
    return {**_index_runtime_state(), "status_message": _freshness_status_message, "integrity_probe": integrity_probe}


@app.get("/v1/index/freshness")
def index_freshness():
    return _index_freshness_payload()



@app.post("/v1/conversation/export")
async def export_conversation(request: Request):
    body = await request.json()
    messages = body.get("messages") or []
    title = str(body.get("title") or "AndesCode Conversation")
    created_at = body.get("created_at")

    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="No conversation messages to export.")

    try:
        path = write_conversation_export(messages, title=title, created_at=created_at)
    except Exception as exc:
        audit.error(f"EXPORT | failed | error={exc}")
        raise HTTPException(status_code=500, detail=f"Failed to export conversation: {exc}")

    audit.info(f"EXPORT | conversation | path={_safe(str(path))} | messages={len(messages)}")
    return {"ok": True, "path": str(path)}


@app.get("/v1/models")
@app.get("/models")
def list_models():
    return {"object": "list", "data": [{
        "id": "andescode-gemma4-26b", "object": "model",
        "created": int(time.time()), "owned_by": "andescode",
    }]}


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat(request: Request):
    body       = await request.json()
    messages   = body.get("messages", [])
    stream     = body.get("stream", True)
    max_tokens = body.get("max_tokens", 1024)
    api_debug  = body.get("debug_mode")
    debug_mode = _resolve_request_debug_mode(api_debug)
    request_id = str(uuid.uuid4())[:8]
    t_start    = time.perf_counter()
    execution_mode = get_execution_mode()
    if not _get_active_index_session():
        raise HTTPException(
            status_code=409,
            detail="No active project is selected. Continue with the previous project or index a new one first.",
        )

    audit.info(f"REQUEST {request_id} | tokens={max_tokens} | messages={len(messages)}")
    audit.info(f"DEBUG_MODE {request_id} | request_flag={api_debug!r} | enabled={debug_mode}")
    audit.info(
        f"EXECUTION_MODE {request_id} | mode={execution_mode.value} | source={execution_mode_env_key()}"
    )

    if execution_mode == ExecutionMode.REMOTE_INFERENCE:
        _phase_log(request_id, "remote_mode_enabled", enabled=True)
        if stream:
            return StreamingResponse(
                _remote_proxy_stream_with_freshness(messages, max_tokens, request_id, debug_mode),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
            )
        ok, refresh_error, freshness = _refresh_index_before_answer_if_needed()
        if not ok:
            _phase_log(request_id, "index_refresh_failed", error=refresh_error)
            return _remote_error_payload(
                "index_refresh_failed",
                f"Index refresh failed before answering: {refresh_error}",
                request_id=request_id,
                details={"freshness": freshness},
            )
        if not freshness.get("has_index", False) and not hasattr(_indexer_module, "ROOT"):
            return _remote_error_payload(
                "index_not_ready",
                "Local index is required for REMOTE_INFERENCE mode retrieval.",
                request_id=request_id,
                details={"freshness": freshness},
            )
        payload, client_debug = _collect_local_remote_payload(
            messages=messages,
            request_id=request_id,
            max_tokens=max_tokens,
            debug_mode=debug_mode,
            stream=stream,
        )
        if payload is None:
            _phase_log(request_id, "remote_payload_build_failed")
            return client_debug or _remote_error_payload(
                "remote_payload_unavailable",
                "Failed to build remote inference payload.",
                request_id=request_id,
            )
        workspace_meta = payload.get("workspace", {}) if isinstance(payload, dict) else {}
        retrieval_meta = payload.get("retrieval", {}) if isinstance(payload, dict) else {}
        _phase_log(
            request_id,
            "remote_payload_ready",
            workspace_id=workspace_meta.get("workspace_id", "unknown"),
            branch=workspace_meta.get("branch", "unknown"),
            commit_hash=workspace_meta.get("commit_hash", "unknown"),
            retrieved_chunks=retrieval_meta.get("retrieved_chunk_count", 0),
        )
        endpoint = f"{REMOTE_INFERENCE_SERVER_URL}/v1/ask"
        serialized = json.dumps(payload).encode("utf-8")
        if stream:
            async def _proxy_stream():
                saw_done = False
                done_scan_buffer = ""
                try:
                    req = url_request.Request(
                        endpoint,
                        data=serialized,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    _phase_log(request_id, "remote_payload_send_started", endpoint=endpoint, stream=True)
                    with url_request.urlopen(req, timeout=180) as resp:
                        while True:
                            part = resp.read(4096)
                            if not part:
                                break
                            decoded = part.decode("utf-8", errors="ignore")
                            done_scan_buffer = f"{done_scan_buffer}{decoded}"[-4096:]
                            if _remote_sse_has_done_event(done_scan_buffer):
                                saw_done = True
                            yield decoded
                    _phase_log(request_id, "remote_payload_send_succeeded", stream=True)
                except url_error.URLError as exc:
                    reason = getattr(exc, "reason", exc)
                    _phase_log(request_id, "remote_payload_send_failed", stream=True, error=reason)
                    yield _remote_proxy_sse_error(
                        request_id,
                        "remote_unreachable",
                        f"Remote server unreachable: {reason}",
                    )
                    yield "data: [DONE]\n\n"
                except url_error.HTTPError as exc:
                    detail = exc.read().decode("utf-8", errors="ignore")
                    _phase_log(request_id, "remote_payload_send_failed", stream=True, status=exc.code)
                    code = "remote_http_error"
                    try:
                        parsed = json.loads(detail) if detail else {}
                        if isinstance(parsed, dict):
                            code = (
                                ((parsed.get("error") or {}).get("code"))
                                or parsed.get("code")
                                or code
                            )
                    except Exception:
                        pass
                    yield _remote_proxy_sse_error(request_id, code, detail or str(exc))
                    yield "data: [DONE]\n\n"
                except Exception as exc:
                    _phase_log(request_id, "remote_payload_send_failed", stream=True, error=exc)
                    yield _remote_proxy_sse_error(request_id, "remote_stream_interrupted", str(exc))
                    yield "data: [DONE]\n\n"
                else:
                    if not saw_done:
                        _phase_log(request_id, "remote_stream_interrupted", reason="missing_done_event")
                        yield _remote_proxy_sse_error(
                            request_id,
                            "remote_stream_interrupted",
                            "Remote stream ended unexpectedly before completion marker.",
                        )
                        yield "data: [DONE]\n\n"
            return StreamingResponse(
                _proxy_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
            )
        try:
            req = url_request.Request(endpoint, data=serialized, headers={"Content-Type": "application/json"}, method="POST")
            _phase_log(request_id, "remote_payload_send_started", endpoint=endpoint, stream=False)
            with url_request.urlopen(req, timeout=180) as resp:
                remote_response = json.loads(resp.read().decode("utf-8"))
                _phase_log(request_id, "remote_payload_send_succeeded", stream=False)
                if isinstance(remote_response, dict) and remote_response.get("ok") is False:
                    return remote_response
                if not isinstance(remote_response, dict):
                    return _remote_error_payload(
                        "remote_proxy_error",
                        "Remote response was not a JSON object",
                        request_id=request_id,
                        details=client_debug,
                    )
                answer_text = str(remote_response.get("answer") or "")
                remote_debug = remote_response.get("debug")
                answer_text = _enforce_edit_suggestion_answer(
                    answer_text,
                    remote_debug if isinstance(remote_debug, dict) else client_debug,
                    payload.get("query", {}).get("text", "") if isinstance(payload, dict) else "",
                )
                return {
                    "id": f"chatcmpl-{request_id}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "andescode-gemma4-26b",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": answer_text},
                        "finish_reason": "stop",
                    }],
                    "debug": (remote_debug if debug_mode else None),
                }
        except url_error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            _phase_log(request_id, "remote_payload_send_failed", stream=False, error=reason)
            return _remote_error_payload(
                "remote_unreachable",
                f"Remote server unreachable: {reason}",
                request_id=request_id,
                details=client_debug,
            )
        except Exception as exc:
            _phase_log(request_id, "remote_payload_send_failed", stream=False, error=exc)
            return _remote_error_payload("remote_proxy_error", str(exc), request_id=request_id, details=client_debug)

    if stream:
        # TODO(phase3): streaming path not yet wrapped by LocalAskOrchestrator — bypasses orchestration boundary
        return StreamingResponse(
            _stream_with_freshness(messages, max_tokens, request_id, t_start, debug_mode=debug_mode),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                     "X-Accel-Buffering": "no"},
        )

    ok, refresh_error, freshness = _refresh_index_before_answer_if_needed()
    if not ok:
        _phase_log(request_id, "index_refresh_failed", error=refresh_error)
        return {
            "ok": False,
            "error": {
                "code": "index_refresh_failed",
                "message": f"Index refresh failed before answering: {refresh_error}",
                "freshness": freshness,
            },
        }

    orchestrator = LocalAskOrchestrator(
        retrieval=FunctionRetrievalProvider(build_context_fn=_build_context),
        context_builder=FunctionAnswerContextBuilder(to_prompt_fn=_messages_to_prompt),
        answer_engine=LlamaAnswerEngine(llm=llm),
        strip_thinking=_strip_thinking,
        is_performance_query=_is_performance_query,
        validate_high_signal_output=_validate_high_signal_output,
    )
    try:
        text, debug_payload = orchestrator.run_non_stream(
            messages=messages,
            request_id=request_id,
            max_tokens=max_tokens,
            debug_mode=debug_mode,
        )
        audit.info(f"DEBUG_PAYLOAD {request_id} | generated={bool(debug_payload)}")
        user_query = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        if is_edit_suggestion_query(user_query):
            text = enforce_edit_suggestion_output(text, _edit_context_from_debug_payload(debug_payload, user_query))
        is_performance = _is_performance_query(user_query)
        _, filtered_out = _validate_high_signal_output(text, is_performance)
        query_type = EDIT_SUGGESTION if is_edit_suggestion_query(user_query) else ("performance" if is_performance else "general")
        audit.info(
            f"HIGH_SIGNAL {request_id} | query_type={query_type} | "
            f"applied={is_performance} | filtered_out_items={filtered_out}"
        )
    except Exception as e:
        audit.warning(f"INFERENCE_FAIL {request_id} | {_safe(e)}")
        return {"error": str(e)}, 500

    t_done = time.perf_counter()
    audit.info(f"RESPONSE {request_id} | chars={len(text)} | total={t_done-t_start:.1f}s")
    return {
        "id": f"chatcmpl-{request_id}", "object": "chat.completion",
        "created": int(time.time()), "model": "andescode-gemma4-26b",
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "debug": (debug_payload if debug_mode else None),
    }


async def _stream_with_freshness(messages: list, max_tokens: int, request_id: str, t_start: float, debug_mode: bool = False):
    freshness = _index_freshness_payload()
    if freshness.get("ok") and freshness.get("has_index") and freshness.get("changed"):
        yield _make_chunk("⚙️ _Refreshing index before answering..._", request_id)
        events = []
        def _emit(event):
            events.append(event)
        ok = _run_index_stream(str(freshness.get("indexed_root") or ""), source="query", emit_event=_emit, force_refresh=False)
        if not ok:
            message = "Index refresh failed before answering."
            for event in reversed(events):
                if event.get("type") == "error" and event.get("message"):
                    message = str(event.get("message"))
                    break
            yield _make_pipeline_error_event(request_id, "index_refresh", RuntimeError(message))
            yield _make_error_chunk(request_id, "index_refresh", RuntimeError(message))
            yield "data: [DONE]\n\n"
            return
    elif not freshness.get("ok", False):
        message = str(freshness.get("error") or "Unable to check index freshness")
        yield _make_pipeline_error_event(request_id, "index_freshness", RuntimeError(message))
        yield _make_error_chunk(request_id, "index_freshness", RuntimeError(message))
        yield "data: [DONE]\n\n"
        return

    async for chunk in _stream(messages, max_tokens, request_id, t_start, debug_mode=debug_mode):
        yield chunk


def _remote_payload_chunks_for_edit_context(payload: dict | None) -> list[dict]:
    chunks = (payload or {}).get("chunks", []) if isinstance(payload, dict) else []
    if not isinstance(chunks, list):
        return []
    return [
        {
            "file": c.get("path", c.get("file", "")),
            "path": c.get("path", c.get("file", "")),
            "content": c.get("content", ""),
            "language": c.get("language", ""),
            "source_type": c.get("source_type", "source_code"),
        }
        for c in chunks
        if isinstance(c, dict)
    ]


def _remote_sse_has_done_event(decoded: str) -> bool:
    for raw_line in decoded.splitlines():
        if not raw_line.startswith("data:"):
            continue
        payload = raw_line[len("data:"):].strip()
        if payload == "[DONE]":
            return True
    return False

def _remote_sse_text(decoded: str) -> str:
    text_parts: list[str] = []
    for raw_line in decoded.splitlines():
        if not raw_line.startswith("data: "):
            continue
        payload = raw_line[len("data: "):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except Exception:
            continue
        if isinstance(data, dict) and data.get("object") == "chat.completion.chunk":
            text_parts.append(str(data.get("choices", [{}])[0].get("delta", {}).get("content", "") or ""))
    return "".join(text_parts)


def _remote_sse_error_event(decoded: str) -> str | None:
    for raw_line in decoded.splitlines():
        if not raw_line.startswith("data: "):
            continue
        payload = raw_line[len("data: "):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except Exception:
            continue
        if isinstance(data, dict) and (data.get("object") == "andescode.error" or data.get("event") == "error"):
            return f"data: {json.dumps(data)}\n\n"
    return None

def _enforce_edit_suggestion_answer(text: str, debug_payload: dict | None, query: str) -> str:
    stripped = _strip_thinking(text or "", strip_edges=True)
    if not is_edit_suggestion_query(query):
        return stripped
    return enforce_edit_suggestion_output(stripped, _edit_context_from_debug_payload(debug_payload, query))


async def _remote_proxy_stream_with_freshness(messages: list, max_tokens: int, request_id: str, debug_mode: bool = False):
    freshness = _index_freshness_payload()
    if freshness.get("ok") and freshness.get("has_index") and freshness.get("changed"):
        yield _make_chunk("⚙️ _Refreshing index before answering..._", request_id)
        events = []
        def _emit(event):
            events.append(event)
        ok = _run_index_stream(str(freshness.get("indexed_root") or ""), source="query", emit_event=_emit, force_refresh=False)
        if not ok:
            message = "Index refresh failed before answering."
            for event in reversed(events):
                if event.get("type") == "error" and event.get("message"):
                    message = str(event.get("message"))
                    break
            yield _remote_proxy_sse_error(request_id, "index_refresh_failed", f"Index refresh failed before answering: {message}", phase="index_refresh")
            yield "data: [DONE]\n\n"
            return
    elif not freshness.get("ok", False):
        message = str(freshness.get("error") or "Unable to check index freshness")
        yield _remote_proxy_sse_error(request_id, "index_freshness_failed", message, phase="index_freshness")
        yield "data: [DONE]\n\n"
        return

    if not freshness.get("has_index", False) and not hasattr(_indexer_module, "ROOT"):
        yield _remote_proxy_sse_error(
            request_id,
            "index_not_ready",
            "Local index is required for REMOTE_INFERENCE mode retrieval.",
            phase="index_freshness",
        )
        yield "data: [DONE]\n\n"
        return

    payload, client_debug = _collect_local_remote_payload(
        messages=messages,
        request_id=request_id,
        max_tokens=max_tokens,
        debug_mode=debug_mode,
        stream=True,
    )
    if payload is None:
        yield _remote_proxy_sse_error(
            request_id,
            "remote_payload_unavailable",
            "Failed to build remote inference payload.",
            phase="remote_payload",
        )
        yield "data: [DONE]\n\n"
        return

    endpoint = f"{REMOTE_INFERENCE_SERVER_URL}/v1/ask"
    serialized = json.dumps(payload).encode("utf-8")
    saw_done = False
    done_scan_buffer = ""
    edit_query = is_edit_suggestion_query(payload.get("query", {}).get("text", "") if isinstance(payload, dict) else "")
    buffered_answer = ""
    sse_buffer = ""
    try:
        req = url_request.Request(
            endpoint,
            data=serialized,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        _phase_log(request_id, "remote_payload_send_started", endpoint=endpoint, stream=True)
        with url_request.urlopen(req, timeout=180) as resp:
            while True:
                part = resp.read(4096)
                if not part:
                    break
                decoded = part.decode("utf-8", errors="ignore")
                done_scan_buffer = f"{done_scan_buffer}{decoded}"[-4096:]
                if _remote_sse_has_done_event(done_scan_buffer):
                    saw_done = True
                if edit_query:
                    sse_buffer += decoded
                    while "\n\n" in sse_buffer:
                        event_text, sse_buffer = sse_buffer.split("\n\n", 1)
                        error_event = _remote_sse_error_event(event_text)
                        if error_event:
                            yield error_event
                            yield "data: [DONE]\n\n"
                            return
                        buffered_answer += _remote_sse_text(event_text)
                else:
                    yield decoded
        if edit_query and saw_done:
            if sse_buffer:
                error_event = _remote_sse_error_event(sse_buffer)
                if error_event:
                    yield error_event
                    yield "data: [DONE]\n\n"
                    return
                buffered_answer += _remote_sse_text(sse_buffer)
            enforced = _enforce_edit_suggestion_answer(buffered_answer, client_debug, payload.get("query", {}).get("text", ""))
            yield _make_chunk(enforced, request_id)
            yield "data: [DONE]\n\n"
        _phase_log(request_id, "remote_payload_send_succeeded", stream=True)
    except url_error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        _phase_log(request_id, "remote_payload_send_failed", stream=True, error=reason)
        yield _remote_proxy_sse_error(request_id, "remote_unreachable", f"Remote server unreachable: {reason}")
        yield "data: [DONE]\n\n"
    except url_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        _phase_log(request_id, "remote_payload_send_failed", stream=True, status=exc.code)
        yield _remote_proxy_sse_error(request_id, "remote_http_error", detail or str(exc))
        yield "data: [DONE]\n\n"
    except Exception as exc:
        _phase_log(request_id, "remote_payload_send_failed", stream=True, error=exc)
        yield _remote_proxy_sse_error(request_id, "remote_stream_interrupted", str(exc))
        yield "data: [DONE]\n\n"
    else:
        if not saw_done:
            _phase_log(request_id, "remote_stream_interrupted", reason="missing_done_event")
            yield _remote_proxy_sse_error(
                request_id,
                "remote_stream_interrupted",
                "Remote stream ended unexpectedly before completion marker.",
            )
            yield "data: [DONE]\n\n"


@app.post("/v1/ask")
async def remote_inference_ask(request: Request):
    body = await request.json()
    request_id = str(uuid.uuid4())[:8]
    try:
        remote_request = RemoteInferenceRequest.from_dict(body)
        request_id = remote_request.query.request_id
        _phase_log(
            request_id,
            "remote_ask_received",
            protocol_version=remote_request.client.protocol_version,
            chunk_count=len(remote_request.chunks),
        )
    except SchemaValidationError as exc:
        _phase_log(request_id, "remote_ask_validation_failed", error=exc)
        return _remote_error_payload("validation_error", str(exc), request_id=request_id)
    except Exception as exc:
        _phase_log(request_id, "remote_ask_invalid_payload", error=exc)
        return _remote_error_payload("invalid_payload", str(exc), request_id=request_id)

    if remote_request.client.protocol_version != RemoteProtocol.V1.value:
        _phase_log(
            request_id,
            "remote_ask_unsupported_protocol",
            received=remote_request.client.protocol_version,
        )
        return _remote_error_payload(
            "unsupported_protocol",
            "Only protocol andes.remote.v1 is supported.",
            request_id=request_id,
            details={"received": remote_request.client.protocol_version},
        )
    if not remote_request.chunks:
        _phase_log(request_id, "remote_ask_empty_retrieval")
        return _remote_error_payload("empty_retrieval", "chunks must be non-empty", request_id=request_id)

    max_tokens = remote_request.options.max_answer_tokens or 1024
    _phase_log(request_id, "remote_answer_generation_started", stream=remote_request.options.stream, max_tokens=max_tokens)
    prompt_messages, debug_payload = _build_remote_prompt_messages(remote_request)
    prompt = _messages_to_prompt(prompt_messages)

    if remote_request.options.stream:
        async def _remote_stream():
            final_text = ""
            edit_query = is_edit_suggestion_query(remote_request.query.text)
            try:
                for chunk in llm(prompt, max_tokens=max_tokens, stream=True, echo=False):
                    token = chunk["choices"][0]["text"]
                    if not token:
                        continue
                    final_text += token
                    if not edit_query:
                        yield _make_chunk(token, request_id)
                if edit_query:
                    final_text = _enforce_edit_suggestion_answer(final_text, debug_payload, remote_request.query.text)
                    yield _make_chunk(final_text, request_id)
                if remote_request.options.debug:
                    yield format_debug_sse_event(debug_payload)
                _phase_log(request_id, "remote_stream_completed", answer_chars=len(final_text))
            except Exception as exc:
                _phase_log(request_id, "remote_stream_failed", error=exc)
                yield _remote_proxy_sse_error(request_id, "generation_failed", str(exc), phase="remote_inference")
            finally:
                _phase_log(request_id, "remote_answer_generation_finished", stream=True)
                yield "data: [DONE]\n\n"
        return StreamingResponse(
            _remote_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    try:
        result = llm(prompt, max_tokens=max_tokens, echo=False, stream=False)
        text = result["choices"][0]["text"]
    except Exception as exc:
        _phase_log(request_id, "remote_answer_generation_failed", stream=False, error=exc)
        return _remote_error_payload("generation_failed", str(exc), request_id=request_id)

    text = _enforce_edit_suggestion_answer(text, debug_payload, remote_request.query.text)
    _phase_log(request_id, "remote_answer_generation_finished", stream=False, answer_chars=len(text or ""))
    response = {
        "ok": True,
        "event": "final_answer",
        "request_id": request_id,
        "answer": text,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    if remote_request.options.debug:
        response["debug"] = debug_payload
    return response


@app.post("/v1/debug/explain")
async def debug_explain(request: Request):
    body = await request.json()
    query = body.get("query", "")
    n_results = int(body.get("n_results", CONTEXT_CHUNKS))
    api_debug = body.get("debug_mode")
    debug_mode = _resolve_request_debug_mode(api_debug)
    audit.info(f"DEBUG_EXPLAIN | request_flag={api_debug!r} | enabled={debug_mode}")
    if not _indexer_module:
        return {"enabled": debug_mode, "error": "Indexer not available", "debug": None}
    if not query:
        return {"enabled": debug_mode, "error": "query is required", "debug": None}
    payload = _indexer_module.inspect_query_debug(query, n_results=n_results, debug_mode=debug_mode)
    return {"enabled": debug_mode, "debug": payload if debug_mode else None}


@app.post("/v1/index")
async def index_project(request: Request):
    body = await request.json()
    path = body.get("path", ".")
    force_refresh = bool(body.get("force_refresh") or body.get("reindex"))

    _set_active_index_session(False)

    if not _load_indexer():
        return {"error": "Indexer not available"}

    async def _generate():
        import asyncio, queue, threading

        q = queue.Queue()

        def _producer():
            try:
                _call_run_index_stream_compat(_run_index_stream, path, "manual", q.put, force_refresh=force_refresh)
            except Exception as e:
                q.put({"type": "error", "message": str(e)})
            finally:
                q.put(None)

        threading.Thread(target=_producer, daemon=True).start()
        loop = asyncio.get_event_loop()

        while True:
            try:
                event = await loop.run_in_executor(None, lambda: q.get(timeout=300))
            except Exception:
                break
            if event is None:
                break
            if event["type"] == "done":
                _set_active_index_session(True)
                audit.info(
                    f"INDEX | path={_safe(path)} | "
                    f"indexed={event.get('indexed')} | chunks={event.get('chunks')}"
                )
            yield f"data: {json.dumps(event)}\n\n"

        yield 'data: {"type": "end"}\n\n'

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no"},
    )


@app.post("/v1/index/clear")
async def clear_index_state():
    if INDEXER_READY and _indexer_module:
        try:
            _indexer_module.chroma.delete_collection(_indexer_module.COLLECTION)
        except Exception:
            pass
        try:
            _indexer_module.col = _indexer_module.chroma.get_or_create_collection(_indexer_module.COLLECTION)
        except Exception:
            pass
        for p in (
            _indexer_module.HASH_STORE,
            _indexer_module.PROJECT_MAP,
            _indexer_module.SYMBOL_INDEX,
            _indexer_module.WORKSPACE_INDEX,
            _indexer_module.INDEX_STATE,
            _indexer_module.INTEGRITY_STATE,
            _indexer_module.CHUNK_COUNT_STATE,
        ):
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass
        try:
            shutil.rmtree(_indexer_module.CACHE_DIR, ignore_errors=True)
        except Exception:
            pass
    _set_active_index_session(False)
    return {"ok": True, "doc_count": 0, "active_index_session": False}


@app.post("/v1/index/restore")
async def restore_index_state():
    if not INDEXER_READY or not _indexer_module:
        _set_active_index_session(False)
        return {"ok": False, "error": "Indexer not available"}
    try:
        project_map = _indexer_module._load_project_map() if hasattr(_indexer_module, "_load_project_map") else {}
    except Exception as exc:
        _set_active_index_session(False)
        return {"ok": False, "error": f"Unable to read persisted project metadata: {exc}"}
    try:
        doc_count = int(_indexer_module.col.count())
    except Exception as exc:
        _set_active_index_session(False)
        return {"ok": False, "error": f"Unable to validate persisted index chunks: {exc}"}
    if not isinstance(project_map, dict) or not project_map or doc_count <= 0:
        _set_active_index_session(False)
        return {"ok": False, "error": "No persisted index is available to restore."}
    _set_active_index_session(True)
    return {"ok": True, "active_index_session": True, "doc_count": doc_count, "project_map": project_map}


# ── Context builder ───────────────────────────────────────────────────────────

def _build_context(
    messages: list,
    request_id: str,
    debug_mode: bool = False,
    return_debug: bool = False,
) -> list | tuple[list, dict | None]:
    """
    Build the full system prompt with:
      1. Base instructions
      2. Project map header (always present after indexing)
      3. Retrieved code chunks with coverage metadata
    """
    query = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
    )
    reasoning_policy, query_type = _reasoning_policy_for_query(query)
    audit.info(
        f"HIGH_SIGNAL_MODE {request_id} | query_type={query_type} | "
        f"applied={bool(reasoning_policy)}"
    )
    if not INDEXER_READY:
        sections = build_prompt_sections(
            system_prefix=_BASE_SYSTEM,
            reasoning_policy=reasoning_policy,
            workspace_prefix="",
            retrieval_context="",
            user_turn="",
        )
        base = [{"role": "system", "content": serialize_prompt_sections(sections)}] + [
            m for m in messages if m.get("role") != "system"
        ]
        return (base, None) if return_debug else base

    if not query:
        return (messages, None) if return_debug else messages

    try:
        index_state = _indexer_module._load_index_state() if _indexer_module and hasattr(_indexer_module, "_load_index_state") else {}
        # ── Project map ───────────────────────────────────────────────────────
        pmap        = _indexer_module._load_project_map() if _indexer_module else {}
        ws          = _indexer_module._load_workspace_index() if _indexer_module else {}
        repo_fp     = _indexer_module.get_repo_fingerprint() if _indexer_module else ""
        cache       = getattr(_indexer_module, "CACHE", None) if _indexer_module else None
        map_section = ""
        if pmap:
            from indexer import format_project_map_for_prompt
            map_section = format_project_map_for_prompt(pmap)
        workspace_signature = cache.workspace_signature(ws) if cache and ws else ""

        # ── Code retrieval ────────────────────────────────────────────────────
        retrieval_debug = None
        if debug_mode or return_debug:
            chunks, retrieval_debug = search_codebase(
                query,
                n_results=CONTEXT_CHUNKS,
                debug_mode=(debug_mode or return_debug),
                return_debug=True,
            )
        else:
            chunks = search_codebase(query, n_results=CONTEXT_CHUNKS, debug_mode=debug_mode)

        edit_full_files: list[str] = []
        if is_edit_suggestion_query(query):
            chunks, edit_full_files = _expand_edit_suggestion_chunks(query, chunks, pmap, ws)
            if retrieval_debug is not None:
                retrieval_debug.setdefault("edit_suggestion", {})["full_files_loaded"] = list(edit_full_files)

        normalized = normalize_local_retrieval(
            query=query,
            chunks=chunks,
            strategy="direct_retrieval",
            top_k=CONTEXT_CHUNKS,
            retrieval_mode=get_execution_mode().value,
            index_state=index_state if isinstance(index_state, dict) else {},
        )
        normalized_chunks = normalized.to_prompt_chunks()
        if return_debug and retrieval_debug is None:
            retrieval_debug = {
                "query": query,
                "intent": EDIT_SUGGESTION if is_edit_suggestion_query(query) else "direct_retrieval",
                "retrieval_route": "direct_retrieval",
                "retrieval": {"route_taken": "direct_retrieval", "files_retrieved": []},
            }
        if retrieval_debug is not None:
            retrieval_debug["normalized_retrieval"] = normalized.to_debug_dict()

        if not normalized_chunks:
            empty_retrieval_context = ""
            if is_edit_suggestion_query(query):
                empty_retrieval_context = safe_context_fallback(["relevant files", "symbols or methods in relevant files"])
            sections = build_prompt_sections(
                system_prefix=_BASE_SYSTEM,
                reasoning_policy=reasoning_policy,
                workspace_prefix=map_section,
                retrieval_context=empty_retrieval_context,
                user_turn="",
            )
            if cache and repo_fp and workspace_signature:
                prefix = cache.prompt_prefix_get(repo_fp=repo_fp, workspace_signature=workspace_signature)
                if not prefix:
                    prefix = serialize_prompt_sections(sections)
                    cache.prompt_prefix_set(repo_fp=repo_fp, workspace_signature=workspace_signature, value=prefix)
                system = prefix
            else:
                system = serialize_prompt_sections(sections)
            audit.info(f"CONTEXT {request_id} | chunks=0 | no relevant code found")
            base = [{"role": "system", "content": system}] + [
                m for m in messages if m.get("role") != "system"
            ]
            return (base, retrieval_debug) if return_debug else base

        anchor_files = _extract_anchor_files(query)
        authoritative_files = []
        if ws:
            authoritative_files.extend(ws.get("manifests", []) or [])
            authoritative_files.extend((ws.get("config_graph", {}) or {}).get("config_files", []) or [])
        code_section, packing_info = _pack_context_section(
            query=query,
            map_section=map_section,
            chunks=normalized_chunks,
            anchor_files=anchor_files,
            conversation_messages=[m for m in messages if m.get("role") != "system"],
            authoritative_files=authoritative_files,
            request_id=request_id,
        )

        sections = build_prompt_sections(
            system_prefix=_BASE_SYSTEM,
            reasoning_policy=reasoning_policy,
            workspace_prefix=map_section,
            retrieval_context=code_section,
            user_turn="",
        )
        if cache and repo_fp and workspace_signature and not code_section.strip():
            prefix = cache.prompt_prefix_get(repo_fp=repo_fp, workspace_signature=workspace_signature)
            if not prefix:
                prefix = serialize_prompt_sections(sections)
                cache.prompt_prefix_set(repo_fp=repo_fp, workspace_signature=workspace_signature, value=prefix)
            system = prefix
        else:
            system = serialize_prompt_sections(sections)

        audit.info(
            f"CONTEXT {request_id} | chunks={len(normalized_chunks)} | "
            f"packed={packing_info['packed_chunks']} | files={packing_info['kept_files']}"
        )
        if retrieval_debug is not None:
            packed_chunks = packing_info["packed_chunks_raw"]
            retrieval_debug["final_context"] = {
                "files_used": [c.get("file") for c in packed_chunks if c.get("file")],
                "context_size": sum(len(c.get("content", "")) for c in packed_chunks),
                "packed_chunks": packed_chunks,
            }

        base = [{"role": "system", "content": system}] + [
            m for m in messages if m.get("role") != "system"
        ]
        return (base, retrieval_debug) if return_debug else base

    except Exception as e:
        audit.warning(f"CONTEXT_FAIL {request_id} | {_safe(e)}")
        return (messages, None) if return_debug else messages


_FILE_HEADER_RE = re.compile(r"^\s*(?:#|//|--|;)?\s*File:\s+", re.IGNORECASE)


def _chunk_raw_start_line(chunk: dict) -> int | None:
    raw = chunk.get("start_line", chunk.get("line"))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _chunk_start_line(chunk: dict) -> int | None:
    """Return a one-based start line for standalone chunk metadata checks."""
    value = _chunk_raw_start_line(chunk)
    if value is None:
        return None
    return value + 1 if value == 0 else value


def _chunk_start_line_for_base(chunk: dict, *, zero_based: bool) -> int | None:
    value = _chunk_raw_start_line(chunk)
    if value is None:
        return None
    return value + 1 if zero_based else value


def _strip_injected_file_header(lines: list[str]) -> tuple[list[str], int]:
    """Remove synthetic indexer file headers from chunk text before line math."""
    if not lines or not _FILE_HEADER_RE.match(lines[0]):
        return lines, 0
    start = 1
    if len(lines) > 1 and not lines[1].strip():
        start = 2
    return lines[start:], 1


def _chunk_end_line(chunk: dict, line_count: int, *, zero_based: bool = False) -> int | None:
    raw = chunk.get("end_line")
    if raw is not None:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return None
        if value < 0:
            return None
        return value + 1 if zero_based else value
    start = _chunk_start_line_for_base(chunk, zero_based=zero_based)
    if start is None:
        return None
    return start + max(line_count, 1) - 1


def _read_indexed_file_from_disk(filename: str) -> str | None:
    """Read the indexed file directly when the index root is available."""
    root = None
    try:
        if _indexer_module and hasattr(_indexer_module, "_load_hashes"):
            hashes = _indexer_module._load_hashes()
            if isinstance(hashes, dict) and hashes.get("__root__"):
                root = Path(str(hashes["__root__"])).resolve()
    except Exception as exc:
        audit.warning(f"EDIT_DISK_ROOT_LOOKUP_FAIL | {filename} | {exc}")
        root = None
    if root is None and _indexer_module and hasattr(_indexer_module, "ROOT"):
        try:
            root = Path(str(getattr(_indexer_module, "ROOT"))).resolve()
        except Exception:
            root = None
    if root is None:
        return None

    target = (root / filename).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return None
    if not target.exists() or not target.is_file():
        return None
    try:
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return target.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        audit.warning(f"EDIT_DISK_FILE_READ_FAIL | {filename} | {exc}")
        return None


def _deoverlap_indexed_chunks(ordered: list[dict]) -> tuple[str, bool, int]:
    merged_lines: list[str] = []
    current_end: int | None = None
    clean = True
    removed_headers = 0
    zero_based = any(_chunk_raw_start_line(chunk) == 0 for chunk in ordered)

    for chunk in ordered:
        content = str(chunk.get("content") or "")
        if not content:
            continue
        raw_lines = content.splitlines()
        source_lines, header_count = _strip_injected_file_header(raw_lines)
        removed_headers += header_count
        start = _chunk_start_line_for_base(chunk, zero_based=zero_based)
        end = _chunk_end_line(chunk, len(source_lines), zero_based=zero_based)
        if start is None or end is None or end < start:
            clean = False

        lines = list(source_lines)
        if current_end is not None and start is not None and start <= current_end:
            overlap = min(current_end - start + 1, len(lines))
            if overlap > 0:
                lines = lines[overlap:]
        elif current_end is not None and start is not None and start > current_end + 1:
            clean = False

        merged_lines.extend(lines)

        if end is not None:
            current_end = end if current_end is None else max(current_end, end)
        else:
            clean = False

    if ordered and _chunk_start_line_for_base(ordered[0], zero_based=zero_based) != 1:
        clean = False
    return "\n".join(merged_lines), clean, removed_headers


def _merge_indexed_file_chunks(filename: str) -> dict | None:
    """Load an edit target as direct file contents or de-overlapped indexed chunks."""
    if not INDEXER_READY or not _indexer_module or not filename:
        return None
    try:
        file_chunks = _indexer_module.get_chunks_for_file(filename)
    except Exception as exc:
        audit.warning(f"EDIT_FULL_FILE_FETCH_FAIL | {filename} | {exc}")
        return None
    if not file_chunks:
        return None
    ordered = sorted(file_chunks, key=lambda c: _chunk_start_line(c) or 1)
    first = ordered[0]
    disk_content = _read_indexed_file_from_disk(filename)
    if disk_content is not None:
        content = disk_content
        full_file = True
        partial = False
        source = "disk_file"
        removed_headers = 0
    else:
        content, clean, removed_headers = _deoverlap_indexed_chunks(ordered)
        full_file = bool(clean)
        partial = not full_file
        source = "indexed_chunks_deoverlapped" if clean else "indexed_chunks_merged_partial"

    if not content.strip():
        return None
    end_line = content.count("\n") + 1 if content else 1
    return {
        "file": filename,
        "content": content,
        "language": first.get("language", ""),
        "symbols": " ".join(str(c.get("symbols") or "") for c in ordered if c.get("symbols")),
        "score": max(float(c.get("score") or 0.0) for c in ordered),
        "line": 1,
        "end_line": end_line,
        "full_file": full_file,
        "coverage": {
            "returned": len(ordered),
            "total": len(ordered),
            "partial": partial,
            "source": source,
            "removed_repeated_file_headers": removed_headers,
        },
        "source_type": first.get("source_type", "source_code"),
        "authority": first.get("authority", "referenced"),
        "authority_reason": "edit_full_file_load" if full_file else "edit_indexed_chunk_merge_partial",
    }


def _discover_edit_related_files(query: str, chunks: list[dict], pmap: dict, workspace: dict) -> list[str]:
    """Combine filename, semantic, symbol, import-neighborhood, config, and test signals."""
    discovered: list[str] = []

    def add(path: str | None) -> None:
        if path and path not in discovered:
            discovered.append(path)

    for anchor in _extract_anchor_files(query):
        add(anchor)
    for chunk in chunks[:5]:
        add(chunk.get("file"))

    file_symbols = pmap.get("file_symbols", {}) if isinstance(pmap, dict) else {}
    query_lower = (query or "").lower()
    role_terms = ("viewmodel", "view_model", "service", "repository", "repo", "test", "config")
    for fname, symbols in file_symbols.items():
        stem = Path(fname).stem.lower()
        symbol_blob = " ".join(symbols or []).lower() if isinstance(symbols, list) else ""
        if any(term in query_lower and term in (stem + " " + symbol_blob) for term in role_terms):
            add(fname)
        elif any(sym.lower() in query_lower for sym in (symbols or []) if len(str(sym)) >= 4):
            add(fname)

    imports = ((workspace or {}).get("import_graph", {}) or {}).get("samples", {}) if isinstance(workspace, dict) else {}
    if not isinstance(imports, dict):
        imports = {}
    seeds = list(discovered)
    for seed in seeds:
        for dep in imports.get(seed, []) if isinstance(imports.get(seed, []), (list, tuple, set)) else []:
            add(dep)
        for source, deps in imports.items():
            if isinstance(deps, (list, tuple, set)) and seed in deps:
                add(source)

    file_to_module = (workspace or {}).get("file_to_module_map", {}) if isinstance(workspace, dict) else {}
    candidate_files = set(file_to_module) | set(file_symbols)
    for seed in list(discovered):
        stem = Path(seed).stem.lower()
        for fname in sorted(candidate_files):
            lower = fname.lower()
            if stem and stem in lower and "test" in lower:
                add(fname)
            if any(role in lower for role in ("viewmodel", "view_model", "service", "repository", "repo")) and Path(fname).parent == Path(seed).parent:
                add(fname)

    config_files = ((workspace or {}).get("config_graph", {}) or {}).get("config_files", []) if isinstance(workspace, dict) else []
    for cfg in config_files[:3]:
        add(cfg)
    for manifest in ((workspace or {}).get("manifests", []) or [])[:3] if isinstance(workspace, dict) else []:
        add(manifest)

    return discovered[:10]


def _expand_edit_suggestion_chunks(query: str, chunks: list[dict], pmap: dict, workspace: dict) -> tuple[list[dict], list[str]]:
    """Load full indexed files for likely edit targets and keep semantic chunks for support."""
    related_files = _discover_edit_related_files(query, chunks, pmap, workspace)
    expanded: list[dict] = []
    loaded: list[str] = []
    for fname in related_files:
        merged = _merge_indexed_file_chunks(fname)
        if merged:
            expanded.append(merged)
            loaded.append(fname)
    seen_identity = {(c.get("file"), c.get("line"), c.get("content")) for c in expanded}
    for chunk in chunks:
        identity = (chunk.get("file"), chunk.get("line"), chunk.get("content"))
        if identity not in seen_identity:
            expanded.append(chunk)
            seen_identity.add(identity)
    return expanded or chunks, loaded

def _remote_error_payload(code: str, message: str, *, request_id: str = "", details: dict | None = None) -> dict:
    return {
        "ok": False,
        "error": {
            "event": "error",
            "request_id": request_id or "unknown",
            "code": code,
            "message": message,
            "details": details or {},
        },
    }


def _remote_proxy_sse_error(request_id: str, code: str, message: str, *, phase: str = "remote_proxy") -> str:
    payload = {
        "object": "andescode.error",
        "error": {
            "phase": phase,
            "code": code,
            "request_id": request_id or "unknown",
            "message": message,
        },
    }
    return f"data: {json.dumps(payload)}\n\n"


def _authority_priority(authority: str) -> int:
    normalized = (authority or "").lower()
    if normalized == "declared":
        return 0
    if normalized == "referenced":
        return 1
    if normalized == "inferred":
        return 2
    return 3


def _build_remote_prompt_messages(remote_request: RemoteInferenceRequest) -> tuple[list, dict]:
    chunks_sorted = sorted(
        remote_request.chunks,
        key=lambda c: (_authority_priority(c.authority), -float(c.score), c.path, c.start_line),
    )
    query = remote_request.query.text
    reasoning_policy, query_type = _reasoning_policy_for_query(query)
    workspace_meta = remote_request.workspace
    retrieval_meta = remote_request.retrieval
    workspace_prefix = (
        "Remote workspace metadata (client-reported):\n"
        f"- Workspace ID: {workspace_meta.workspace_id}\n"
        f"- Repository: {workspace_meta.repo_name}\n"
        f"- Repo root: {workspace_meta.repo_root_name}\n"
        f"- Branch: {workspace_meta.branch}\n"
        f"- Commit: {workspace_meta.commit_hash}\n"
        f"- Dirty working tree: {workspace_meta.is_dirty}\n"
        f"- Retrieval strategy: {retrieval_meta.strategy}\n"
        f"- Retrieved chunks: {retrieval_meta.retrieved_chunk_count}\n\n"
        "Important: You only have the retrieved chunks below; never claim access to files that are not included."
    )
    retrieval_lines = []
    for idx, chunk in enumerate(chunks_sorted, start=1):
        retrieval_lines.append(
            f"[{idx}] {chunk.path}:{chunk.start_line}-{chunk.end_line} "
            f"(authority={chunk.authority}; source_type={chunk.source_type}; score={chunk.score:.3f})\n"
            f"{chunk.content}"
        )
    retrieval_context = "\n\n".join(retrieval_lines)
    sections = build_prompt_sections(
        system_prefix=_BASE_SYSTEM,
        reasoning_policy=reasoning_policy,
        workspace_prefix=workspace_prefix,
        retrieval_context=retrieval_context,
        user_turn="",
    )
    prompt_messages = [
        {"role": "system", "content": serialize_prompt_sections(sections)},
        {"role": "user", "content": query},
    ]
    debug_payload = {
        "mode": "remote_inference_server",
        "protocol_version": remote_request.client.protocol_version,
        "request_id": remote_request.query.request_id,
        "query_type": query_type,
        "workspace": {
            "repo_name": workspace_meta.repo_name,
            "branch": workspace_meta.branch,
            "commit_hash": workspace_meta.commit_hash,
            "is_dirty": workspace_meta.is_dirty,
        },
        "retrieval": {
            "strategy": retrieval_meta.strategy,
            "index_state": retrieval_meta.index_state,
            "retrieved_chunk_count": retrieval_meta.retrieved_chunk_count,
            "total_candidate_files": retrieval_meta.total_candidate_files,
            "files_retrieved": sorted({c.path for c in chunks_sorted}),
        },
        "final_context": {
            "files_used": sorted({c.path for c in chunks_sorted}),
            "packed_chunks": [
                {
                    "file": c.path,
                    "path": c.path,
                    "content": c.content,
                    "language": c.language or "",
                    "source_type": c.source_type,
                }
                for c in chunks_sorted
            ],
        },
    }
    return prompt_messages, debug_payload


def _collect_local_remote_payload(
    *,
    messages: list,
    request_id: str,
    max_tokens: int,
    debug_mode: bool,
    stream: bool,
) -> tuple[dict | None, dict | None]:
    # TODO(phase6): remote payload generation currently uses direct local retrieval only; align with planned-context/local orchestration retrieval for parity
    query = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "").strip()
    if not query:
        return None, _remote_error_payload("validation_error", "query is required", request_id=request_id)
    if not INDEXER_READY or not _indexer_module:
        return None, _remote_error_payload(
            "index_not_ready",
            "Local index is required for REMOTE_INFERENCE mode retrieval.",
            request_id=request_id,
        )
    index_state = _indexer_module._load_index_state() if hasattr(_indexer_module, "_load_index_state") else {}
    retrieval_debug = None
    if debug_mode:
        chunks, retrieval_debug = search_codebase(
            query,
            n_results=CONTEXT_CHUNKS,
            debug_mode=True,
            return_debug=True,
        )
    else:
        chunks = search_codebase(query, n_results=CONTEXT_CHUNKS, debug_mode=False)
    normalized = normalize_local_retrieval(
        query=query,
        chunks=chunks,
        strategy="remote_inference_client_payload",
        top_k=CONTEXT_CHUNKS,
        retrieval_mode=ExecutionMode.REMOTE_INFERENCE.value,
        index_state=index_state if isinstance(index_state, dict) else {},
    )
    if not normalized.chunks:
        return None, _remote_error_payload(
            "empty_retrieval",
            "No retrieved chunks were found for this query.",
            request_id=request_id,
        )
    repo_name = "unknown"
    branch = "unknown"
    commit_hash = "unknown"
    is_dirty = False
    repo_root_name = "workspace"
    if hasattr(_indexer_module, "ROOT"):
        try:
            root_path = Path(str(_indexer_module.ROOT)).resolve()
            repo_root_name = root_path.name or "workspace"
            repo_name = root_path.name or "unknown"
        except Exception:
            pass
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, cwd=str(_indexer_module.ROOT)).strip()
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=str(_indexer_module.ROOT)).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True, cwd=str(_indexer_module.ROOT)).strip()
        is_dirty = bool(status)
    except Exception:
        pass
    payload = {
        "client": {
            "client_version": "andescode-local-client-v1",
            "protocol_version": RemoteProtocol.V1.value,
            "platform": f"{platform.system().lower()}-{platform.machine().lower()}",
            "hostname": socket.gethostname(),
        },
        "workspace": {
            "workspace_id": repo_name,
            "repo_name": repo_name,
            "repo_root_name": repo_root_name,
            "branch": branch,
            "commit_hash": commit_hash,
            "is_dirty": is_dirty,
        },
        "query": {
            "request_id": request_id,
            "text": query,
            "requested_at": datetime.now(timezone.utc).isoformat(),
        },
        "retrieval": {
            "strategy": normalized.summary.strategy,
            "top_k": normalized.summary.top_k,
            "indexed_at": normalized.summary.indexed_at or datetime.now(timezone.utc).isoformat(),
            "index_state": normalized.summary.index_state,
            "total_candidate_files": normalized.summary.total_candidate_files,
            "retrieved_chunk_count": normalized.summary.retrieved_chunk_count,
            "retrieval_mode": normalized.summary.retrieval_mode,
        },
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "path": c.path,
                "language": c.language,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "score": c.score,
                "source_type": c.source_type,
                "authority": c.authority,
                "authority_reason": c.authority_reason,
                "content": c.content,
            }
            for c in normalized.chunks
        ],
        "options": {
            "stream": stream,
            "debug": debug_mode,
            "max_answer_tokens": max_tokens,
        },
    }
    payload_chunks = _remote_payload_chunks_for_edit_context(payload)
    client_debug = {
        "mode": "remote_inference_client",
        "request_id": request_id,
        "remote_server_url": REMOTE_INFERENCE_SERVER_URL,
        "retrieval": normalized.to_debug_dict(),
        "retrieval_debug": retrieval_debug,
        "final_context": {
            "files_used": [c.get("file") for c in payload_chunks if c.get("file")],
            "packed_chunks": payload_chunks,
        },
    }
    return payload, client_debug

# ── Two-step planning ─────────────────────────────────────────────────────────

_FILENAME_HINT_RE = re.compile(r"[\w./-]+\.(?:py|kt|java|js|ts|tsx|jsx|go|rs|swift|cpp|c|h|hpp|json|yaml|yml|toml|gradle|xml|md)")
_CALL_RE = re.compile(r"(?<!\w)([A-Za-z_]\w*)\s*\(")
_RX_CHAIN_RE = re.compile(r"\.(map|flatMap|zip|observeOn|subscribeOn)\s*(?:\(|\{)")
_UI_UPDATE_RE = re.compile(
    r"\b("
    r"submitList|notifyDataSetChanged|notifyItem(?:Inserted|Removed|Changed|RangeChanged)"
    r"|setText|setVisibility|setImage|setAdapter|invalidate|requestLayout|postInvalidate"
    r")\b"
)
_ENTRYPOINT_RE = re.compile(r"\b(onScroll|onScrolled|onTouch|onClick|onBindViewHolder|onChanged|doFrame)\b")
_MAIN_THREAD_RE = re.compile(r"AndroidSchedulers\.mainThread|Dispatchers\.Main|runOnUiThread")
_BACKGROUND_THREAD_RE = re.compile(r"Schedulers\.(io|computation|newThread)|Dispatchers\.(IO|Default)")
_EXCLUDED_CALL_NAMES = {
    "if", "for", "while", "switch", "catch", "return", "when", "else", "try", "synchronized", "super", "this"
}


def _extract_anchor_files(query: str) -> list[str]:
    if not query:
        return []
    seen = set()
    anchors = []
    for match in _FILENAME_HINT_RE.findall(query):
        if match not in seen:
            seen.add(match)
            anchors.append(match)
    return anchors


def _clean_step_name(raw: str) -> str:
    return raw.strip().split(".")[-1].strip()


def _derive_path_metrics(steps: list[str], snippet_blob: str) -> tuple[str, str, str, str]:
    normalized = " ".join(steps).lower() + " " + snippet_blob.lower()
    if any(k in normalized for k in ("onscroll", "onscrolled", "doframe", "onbindviewholder", "recyclerview", "bind")):
        frequency = "per frame"
    elif any(k in normalized for k in ("onclick", "ontouch", "gesture", "input", "onchanged")):
        frequency = "per event"
    elif any(k in normalized for k in ("oncreate", "init", "setup")):
        frequency = "once"
    else:
        frequency = "per event"

    if _MAIN_THREAD_RE.search(snippet_blob):
        thread = "main (proven)"
        thread_main = True
    elif _BACKGROUND_THREAD_RE.search(snippet_blob):
        thread = "background (proven)"
        thread_main = False
    elif any(_UI_UPDATE_RE.search(step) for step in steps):
        thread = "main (inferred)"
        thread_main = True
    else:
        thread = "background (inferred)"
        thread_main = False

    has_rx = any(("flatmap" in s.lower()) or ("zip" in s.lower()) for s in steps)
    has_ui_heavy = any(k in normalized for k in ("notifydatasetchanged", "onbindviewholder", "submitlist", "bind"))
    if frequency == "per frame" and (has_rx or has_ui_heavy or len(steps) >= 4):
        cost = "high"
    elif has_rx or has_ui_heavy or len(steps) >= 4:
        cost = "medium"
    else:
        cost = "low"

    risk = "yes" if thread_main and cost in {"medium", "high"} and frequency in {"per frame", "per event"} else "no"
    return frequency, thread, cost, risk


def _extract_execution_paths(chunks: list[dict], *, max_paths: int = 5) -> list[dict]:
    paths: list[dict] = []
    for chunk in chunks:
        content = chunk.get("content", "")
        if not content:
            continue
        file_name = chunk.get("file", "unknown")
        steps: list[str] = []
        snippets: list[str] = []
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            if _ENTRYPOINT_RE.search(line):
                entry = _ENTRYPOINT_RE.search(line).group(0)
                steps.append(entry)
                snippets.append(raw_line[:160])

            for rx in _RX_CHAIN_RE.findall(line):
                steps.append(rx)
                snippets.append(raw_line[:160])

            if _UI_UPDATE_RE.search(line):
                ui_step = _UI_UPDATE_RE.search(line).group(0)
                steps.append(ui_step)
                snippets.append(raw_line[:160])

            if "->" in line:
                parts = [p.strip() for p in line.split("->") if p.strip()]
                steps.extend(parts[:2])
                snippets.append(raw_line[:160])

            for call in _CALL_RE.findall(line):
                if call in _EXCLUDED_CALL_NAMES:
                    continue
                steps.append(call)
                snippets.append(raw_line[:160])

        dedup_steps: list[str] = []
        seen_steps = set()
        for step in steps:
            cleaned = _clean_step_name(step)
            key = cleaned.lower()
            if not cleaned or key in seen_steps:
                continue
            seen_steps.add(key)
            dedup_steps.append(cleaned)

        if len(dedup_steps) < 2:
            continue

        path_steps = dedup_steps[:6]
        snippet_text = content + "\n" + "\n".join(snippets[:4])
        frequency, thread, cost, risk = _derive_path_metrics(path_steps, snippet_text)
        paths.append(
            {
                "file": file_name,
                "steps": path_steps,
                "frequency": frequency,
                "thread": thread,
                "cost": cost,
                "risk": risk,
                "snippets": snippets[:3],
            }
        )

    def _rank(path: dict) -> tuple[int, int, int]:
        freq_rank = {"per frame": 3, "per event": 2, "once": 1}.get(path["frequency"], 1)
        cost_rank = {"high": 3, "medium": 2, "low": 1}.get(path["cost"], 1)
        risk_rank = 2 if path["risk"] == "yes" else 1
        return (freq_rank, cost_rank, risk_rank)

    paths.sort(key=_rank, reverse=True)
    return paths[:max(3, min(max_paths, 5))]


def _render_execution_path_context(paths: list[dict]) -> str:
    if not paths:
        return ""
    lines = [
        "## Structured Execution Paths",
        "",
        "_These are candidate execution-path hints; verify with retrieved code context below._",
        "",
    ]
    for idx, path in enumerate(paths, start=1):
        chain = " → ".join(path["steps"])
        lines.append(f"Path {idx}: {chain}")
        lines.append(f"- execution frequency: {path['frequency']}")
        lines.append(f"- thread: {path['thread']}")
        lines.append(f"- cost (relative): {path['cost']}")
        lines.append(f"- risk (main-thread blocking): {path['risk']}")
        lines.append("- supporting snippets:")
        for snippet in path["snippets"]:
            lines.append(f"  - `{snippet}`")
        lines.append("")
    return "\n".join(lines).strip() + "\n\n"


def _prioritize_chunk_candidates(
    chunks: list[dict],
    *,
    query: str = "",
    intent: str = "",
    anchor_files: list[str] | None = None,
    planned_files: list[str] | None = None,
    neighbor_files: list[str] | None = None,
) -> list[dict]:
    anchor_set = set(anchor_files or [])
    planned_set = set(planned_files or [])
    neighbor_set = set(neighbor_files or [])

    def _matches(target_file: str, candidates: set[str]) -> bool:
        if target_file in candidates:
            return True
        target_basename = Path(target_file).name
        return any(Path(candidate).name == target_basename for candidate in candidates)

    prioritized = []
    declaration_query = is_declaration_query(query, intent)
    authoritative_source_types = {"manifest", "build_file", "dependency_file", "config_file"}
    for idx, chunk in enumerate(chunks):
        fname = chunk.get("file", "")
        if _matches(fname, anchor_set):
            tier = 0
        elif _matches(fname, planned_set):
            tier = 1
        elif _matches(fname, neighbor_set):
            tier = 2
        else:
            tier = 3
        content = chunk.get("content", "")
        partial_note = ""
        coverage = chunk.get("coverage", {})
        if coverage.get("partial"):
            ret = coverage.get("returned")
            total = coverage.get("total")
            partial_note = (
                f"⚠️ Partial view — showing {ret}/{total} chunks from {fname}. "
                f"You may not have the complete file.\n"
            )
        full_file_note = f"_Full file retrieved: {fname}_\n" if chunk.get("full_file") else ""
        formatted = f"```{fname}\n{content}\n```\n\n"
        section_text = partial_note + full_file_note + formatted
        authority_rank = 1
        if declaration_query and chunk.get("source_type") in authoritative_source_types:
            authority_rank = 0

        prioritized.append(
            {
                "tier": tier,
                "authority_rank": authority_rank,
                "rank": idx,
                "file": fname,
                "chunk": chunk,
                "text": section_text,
                "est_tokens": estimate_tokens(section_text),
            }
        )
    prioritized.sort(key=lambda c: (c["tier"], c["authority_rank"], c["rank"], c["file"]))
    return prioritized


def _file_matches_authoritative(path: str, authoritative_files: set[str]) -> bool:
    if not path or not authoritative_files:
        return False
    if path in authoritative_files:
        return True
    base = Path(path).name
    return any(Path(p).name == base for p in authoritative_files)


def _authoritative_preference_key(path: str, source_type: str) -> tuple[int, int, int, int, str]:
    lower = (path or "").lower()
    is_shared = 1 if "buildsrc/" in lower else 0
    is_module_local = 0 if "/" in (path or "").replace("\\", "/") else 1
    is_build_like = 0 if source_type in {"build_file", "dependency_file"} else 1
    depth = lower.count("/")
    return (is_shared, is_module_local, is_build_like, depth, lower)


def _make_forced_authoritative_candidate(candidate: dict, max_chars: int = 1200) -> dict:
    """Force-packable authoritative candidate: partial chunk if original is too large."""
    out = dict(candidate)
    chunk = dict(candidate.get("chunk", {}))
    text = chunk.get("content", "") or ""
    if len(text) > max_chars:
        chunk["content"] = text[:max_chars]
        chunk["coverage"] = {
            "partial": True,
            "returned": 1,
            "total": max(chunk.get("coverage", {}).get("total", 1), 2),
        }
        out["chunk"] = chunk
        forced_text = (
            "⚠️ Partial authoritative excerpt forced into context to preserve declaration source-of-truth.\n"
            f"```{chunk.get('file', '')}\n{chunk['content']}\n```\n\n"
        )
        out["text"] = forced_text
        out["est_tokens"] = estimate_tokens(forced_text)
    out["tier"] = -1
    out["authority_rank"] = -1
    return out


def _enforce_authoritative_candidate(
    *,
    candidates: list[dict],
    authoritative_files: list[str] | None,
    budget_tokens: int,
) -> tuple[list[dict], dict | None]:
    authoritative_set = set(authoritative_files or [])
    authoritative_candidates = [
        c
        for c in candidates
        if c.get("chunk", {}).get("source_type") in {"manifest", "build_file", "dependency_file", "config_file"}
        and (
            not authoritative_set
            or _file_matches_authoritative(c.get("file", ""), authoritative_set)
        )
    ]
    if not authoritative_candidates:
        # If project-map authoritative set exists but retrieval didn't return any, this is a true retrieval miss.
        return candidates, None

    selected = sorted(
        authoritative_candidates,
        key=lambda c: _authoritative_preference_key(c.get("file", ""), c.get("chunk", {}).get("source_type", "")),
    )[0]
    forced = _make_forced_authoritative_candidate(selected, max_chars=1200)
    if budget_tokens > 0:
        for max_chars in (420, 180, 80, 24):
            if forced.get("est_tokens", 0) <= budget_tokens:
                break
            forced = _make_forced_authoritative_candidate(selected, max_chars=max_chars)
        if forced.get("est_tokens", 0) > budget_tokens:
            chunk = dict(selected.get("chunk", {}))
            chunk["content"] = (chunk.get("content", "") or "")[:24]
            forced_text = (
                "⚠️ Authoritative declaration file forced into context (minimal excerpt).\n"
                f"`{chunk.get('file', '')}`\n\n"
            )
            forced = {
                **selected,
                "chunk": chunk,
                "text": forced_text,
                "est_tokens": estimate_tokens(forced_text),
                "tier": -1,
                "authority_rank": -1,
            }
        forced["est_tokens"] = min(int(forced.get("est_tokens", 1)), max(1, budget_tokens))
    updated: list[dict] = []
    replaced = False
    for c in candidates:
        if (
            not replaced
            and c.get("file") == selected.get("file")
            and c.get("rank") == selected.get("rank")
        ):
            updated.append(forced)
            replaced = True
        else:
            updated.append(c)
    if not replaced:
        updated = [forced] + candidates
    return updated, forced.get("chunk", {})


def _pack_context_section(
    *,
    query: str,
    map_section: str,
    chunks: list[dict],
    anchor_files: list[str] | None = None,
    planned_files: list[str] | None = None,
    neighbor_files: list[str] | None = None,
    conversation_messages: list[dict] | None = None,
    authoritative_files: list[str] | None = None,
    request_id: str,
) -> tuple[str, dict]:
    conversation_text = _messages_to_prompt(
        [{"role": m.get("role"), "content": m.get("content", "")} for m in (conversation_messages or []) if m.get("role") != "system"]
    )
    budget = compute_context_budget(
        system_prompt=_BASE_SYSTEM,
        workspace_prefix=map_section,
        user_query=conversation_text or query,
        total_ctx=MODEL_CONTEXT_WINDOW,
        reserved_response=CONTEXT_RESERVED_RESPONSE_TOKENS,
        safety_margin=CONTEXT_SAFETY_MARGIN_TOKENS,
    )
    decl_query = is_declaration_query(query, intent="")
    candidates = _prioritize_chunk_candidates(
        chunks,
        query=query,
        intent="",
        anchor_files=anchor_files,
        planned_files=planned_files,
        neighbor_files=neighbor_files,
    )
    forced_authoritative_chunk = None
    if decl_query:
        candidates, forced_authoritative_chunk = _enforce_authoritative_candidate(
            candidates=candidates,
            authoritative_files=authoritative_files,
            budget_tokens=budget.context_budget_tokens,
        )
    packed = pack_chunks_to_budget(candidates, budget.context_budget_tokens)
    packed_raw_chunks = [c["chunk"] for c in packed.chunks]
    extracted_paths = _extract_execution_paths(packed_raw_chunks, max_paths=5)
    path_section = _render_execution_path_context(extracted_paths) if (_is_performance_query(query) or is_edit_suggestion_query(query)) else ""
    edit_section = ""
    if is_edit_suggestion_query(query):
        edit_ctx = build_edit_suggestion_context(packed_raw_chunks, query=query)
        edit_lines = [
            "## Edit Suggestion Retrieval Checklist",
            "",
            "- Likely edit/entry files read: " + (", ".join(edit_ctx.files) if edit_ctx.files else "none"),
            "- Symbols/methods/classes found: " + (", ".join(edit_ctx.symbols[:20]) if edit_ctx.symbols else "none"),
            "- Existing mechanisms detected: " + ("; ".join(edit_ctx.existing_mechanisms) if edit_ctx.existing_mechanisms else "none in retrieved context"),
            "- Inferred validation commands: " + ("; ".join(edit_ctx.validation_commands) if edit_ctx.validation_commands else "no test command could be inferred"),
            "",
        ]
        edit_section = "\n".join(edit_lines)
    code_section = path_section + edit_section + "## Retrieved Code (Validation Context)\n\n"
    for c in packed.chunks:
        code_section += c["text"]
    if packed.truncated:
        code_section += "\n_Context truncated to fit model window; highest-priority files were kept first._\n"
    has_authoritative = any(
        c["chunk"].get("source_type") in {"manifest", "build_file", "dependency_file", "config_file"}
        for c in packed.chunks
    )
    if decl_query and forced_authoritative_chunk and not has_authoritative:
        has_authoritative = True
    source_instruction = ""
    if decl_query:
        source_instruction = source_of_truth_guidance(query, intent="") or (
            "## Source-of-Truth Guidance\n"
            "- Prefer declared/config/build sources before code references.\n"
            "- Distinguish declared vs referenced vs inferred facts.\n"
            "- If source-of-truth files are missing, state that explicitly.\n\n"
        )
        if not has_authoritative:
            source_instruction += (
                "- No authoritative declaration/config/build chunks were retrieved in this context; "
                "state declaration files are missing before any inferred findings.\n\n"
            )
    elif has_authoritative:
        source_instruction = (
            "## Source-of-Truth Guidance\n"
            "- Prefer declared/config/build sources before code references.\n"
            "- Distinguish declared vs referenced vs inferred facts.\n"
            "- If source-of-truth files are missing, state that explicitly.\n\n"
        )
    audit.info(
        "CONTEXT_BUDGET %s | budget=%s | used=%s | considered=%s | packed=%s | truncated=%s | kept=%s | dropped=%s"
        % (
            request_id,
            budget.context_budget_tokens,
            packed.used_tokens,
            packed.considered_chunks,
            packed.packed_chunks,
            packed.truncated,
            packed.kept_files[:8],
            packed.dropped_files[:8],
        )
    )
    return source_instruction + code_section, {
        "budget_tokens": budget.context_budget_tokens,
        "used_tokens": packed.used_tokens,
        "considered_chunks": packed.considered_chunks,
        "packed_chunks": packed.packed_chunks,
        "truncated": packed.truncated,
        "kept_files": packed.kept_files,
        "dropped_files": packed.dropped_files,
        "packed_chunks_raw": [c["chunk"] for c in packed.chunks],
        "forced_authoritative_file": (forced_authoritative_chunk or {}).get("file"),
    }

def _plan_files(query: str, pmap: dict) -> list[str]:
    """
    Step 1: Fast planning call.
    Ask the model which files are most relevant to answer this question.
    Returns a list of filenames. Max 3 tokens × 5 filenames = very fast.
    """
    if not pmap:
        return []

    # Build a compact file list from the project map
    file_symbols = pmap.get("file_symbols", {})
    if not file_symbols:
        return []

    file_list = []
    for fname, syms in list(file_symbols.items())[:20]:
        sym_str = ", ".join(syms[:5]) if syms else "—"
        file_list.append(f"  {fname}: {sym_str}")

    planning_prompt = (
        f"<s>You are a code navigator. Given a developer question and a file list, "
        f"output ONLY the filenames most relevant to answer it. "
        f"One filename per line. Maximum 4 files. No explanation.</s>\n"
        f"<start_of_turn>user\n"
        f"Project: {pmap.get('project', 'unknown')} ({pmap.get('language', '')})\n"
        f"Files:\n" + "\n".join(file_list) + "\n\n"
        f"Question: {query}\n"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

    try:
        result = llm(planning_prompt, max_tokens=120, echo=False, stream=False)
        raw    = result["choices"][0]["text"].strip()

        # Extract filenames from the response
        found = []
        for line in raw.splitlines():
            line = line.strip().strip("-•").strip()
            if not line:
                continue
            # Match anything that looks like a file path
            m = re.search(r"[\w/\-_.]+\.\w+", line)
            if m:
                found.append(m.group(0))
            elif "." in line and "/" in line or line.endswith(
                (".py", ".kt", ".java", ".js", ".ts", ".go", ".rs", ".swift")
            ):
                found.append(line.split()[0])
        return found[:4]
    except Exception as e:
        audit.warning(f"PLAN_FAIL | {_safe(e)}")
        return []


def _diagnose_query(query: str, intent: str) -> dict:
    """Deterministic diagnosis stage before planning/generation."""
    mode = "architecture" if intent == "architecture_overview" else "bugfix"
    patch_intent = intent in {"code_fix_or_patch", EDIT_SUGGESTION}
    return {
        "mode": mode,
        "safe_semantic": semantic_cache_allowed(intent, retrieval_route_for_intent(intent)),
        "patch_intent": patch_intent,
    }


def _file_neighborhood(anchor_file: str, mode: str, workspace: dict, repo_fp: str) -> list[str]:
    cache = getattr(_indexer_module, "CACHE", None)
    if cache and repo_fp:
        cached = cache.neighborhood_get(repo_fp=repo_fp, mode=mode, anchor_file=anchor_file)
        if cached:
            return cached

    neighborhood = [anchor_file]
    # indexer stores import_graph as structured metadata:
    # {"edge_count": int, "samples": {source_file: [deps...]}}
    # (not as a raw adjacency map).
    import_graph = workspace.get("import_graph", {}) if isinstance(workspace, dict) else {}
    if not isinstance(import_graph, dict):
        import_graph = {}
    imports = import_graph.get("samples", {})
    if not isinstance(imports, dict):
        imports = {}

    file_to_module = workspace.get("file_to_module_map", {}) if isinstance(workspace, dict) else {}
    if not isinstance(file_to_module, dict):
        file_to_module = {}
    target_module = file_to_module.get(anchor_file)
    for source in sorted(imports):
        deps = imports.get(source)
        if not isinstance(deps, (list, tuple, set)):
            continue
        if anchor_file in deps and source not in neighborhood:
            neighborhood.append(source)
    anchor_deps = imports.get(anchor_file, [])
    if isinstance(anchor_deps, (list, tuple, set)):
        neighborhood.extend([d for d in anchor_deps if isinstance(d, str) and d not in neighborhood])
    if target_module:
        for f, module in sorted(file_to_module.items()):
            if module == target_module and f not in neighborhood:
                neighborhood.append(f)
    stem = Path(anchor_file).stem
    likely_tests = [f for f in sorted(file_to_module) if stem in f and "test" in f.lower()]
    neighborhood.extend([f for f in likely_tests if f not in neighborhood])
    final = neighborhood[:8]
    if cache and repo_fp:
        cache.neighborhood_set(repo_fp=repo_fp, mode=mode, anchor_file=anchor_file, value=final)
    return final


def _build_context_from_plan(
    messages: list,
    planned_files: list,
    request_id: str,
    diagnosis: dict | None = None,
    debug_mode: bool = False,
    return_debug: bool = False,
) -> tuple[list, list[str]] | tuple[list, list[str], dict | None]:
    """
    Step 2: Fetch all chunks from planned files + semantic search fallback.
    Returns (messages_with_context, files_loaded).
    """
    if not INDEXER_READY:
        return (messages, [], None) if return_debug else (messages, [])

    query = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
    )
    reasoning_policy, query_type = _reasoning_policy_for_query(query)
    audit.info(
        f"HIGH_SIGNAL_MODE {request_id} | query_type={query_type} | "
        f"applied={bool(reasoning_policy)}"
    )

    pmap        = _indexer_module._load_project_map() if _indexer_module else {}
    workspace   = _indexer_module._load_workspace_index() if _indexer_module else {}
    map_section = ""
    if pmap:
        from indexer import format_project_map_for_prompt
        map_section = format_project_map_for_prompt(pmap)
    repo_fp = _indexer_module.get_repo_fingerprint() if _indexer_module else ""
    mode = (diagnosis or {}).get("mode", "bugfix")
    index_state = _indexer_module._load_index_state() if _indexer_module and hasattr(_indexer_module, "_load_index_state") else {}

    all_chunks  = []
    files_loaded = []
    semantic_fallback_files = []

    debug_payload = None
    if debug_mode or return_debug:
        debug_payload = {
            "query": query,
            "intent": (diagnosis or {}).get("intent", "planned_context"),
            "retrieval_route": "planned_context",
            "orchestration_path": "planned_context",
            "retrieval": {
                "route_taken": "planned_context",
                "route_reason": "Planner + neighborhood expansion",
                "files_retrieved": [],
                "raw_candidates": list(planned_files),
                "selected_candidates": [],
                "authoritative_files_detected": [],
                "authoritative_files_required": [],
                "authoritative_files_retrieved": [],
                "authoritative_files_missing": [],
                "forced_authoritative_file": False,
                "authority_selection_reason": "",
                "authority_retrieval_mode": "",
                "declaration_answer_mode": "",
                "declaration_query_trigger_reason": "",
                "chunks_per_file": {},
                "coverage": {},
                "cache_hit": False,
                "orchestration_path": "planned_context",
            },
            "planning": {
                "planned_files": list(planned_files),
                "files_loaded": [],
                "semantic_fallback_files": [],
            },
            "final_context": {"files_used": [], "context_size": 0},
        }

    intent = (diagnosis or {}).get("intent", "")
    decl_query = is_declaration_query(query, intent=intent)
    declaration_query_trigger_reason = ""
    if intent in {"dependency_or_build_inventory", "config_lookup", "dependency_lookup"}:
        declaration_query_trigger_reason = "intent"
    elif has_declaration_keywords(query):
        declaration_query_trigger_reason = "keyword_fallback"
    authoritative_files = []
    if workspace:
        authoritative_files.extend(workspace.get("manifests", []) or [])
        authoritative_files.extend((workspace.get("config_graph", {}) or {}).get("config_files", []) or [])
    authoritative_files = sorted(set(authoritative_files))

    if not decl_query and authoritative_files and has_declaration_keywords(query):
        decl_query = True
        declaration_query_trigger_reason = "workspace_signal"

    def _matches_authoritative(path: str) -> bool:
        if path in authoritative_files:
            return True
        base = Path(path).name
        return any(Path(p).name == base for p in authoritative_files)

    def _set_planned_authority_modes(final_chunks: list[dict]) -> None:
        if debug_payload is None or not decl_query:
            return
        retrieved = debug_payload["retrieval"].get("authoritative_files_retrieved", [])
        missing = debug_payload["retrieval"].get("authoritative_files_missing", [])
        has_runtime_or_inferred = any(
            c.get("file")
            and not _matches_authoritative(c.get("file", ""))
            and c.get("source_type", "source_code") in {"source_code", "inferred"}
            for c in (final_chunks or [])
        )
        if retrieved and not missing and not has_runtime_or_inferred:
            debug_payload["retrieval"]["declaration_answer_mode"] = "declared_only"
            debug_payload["retrieval"]["authority_retrieval_mode"] = "direct_chunk_load"
        elif retrieved and has_runtime_or_inferred:
            debug_payload["retrieval"]["declaration_answer_mode"] = "declared_plus_runtime"
            debug_payload["retrieval"]["authority_retrieval_mode"] = "direct_chunk_load"
        elif retrieved and missing and not has_runtime_or_inferred:
            debug_payload["retrieval"]["declaration_answer_mode"] = "declared_partial_only"
            debug_payload["retrieval"]["authority_retrieval_mode"] = "direct_chunk_load"
        elif has_runtime_or_inferred:
            debug_payload["retrieval"]["declaration_answer_mode"] = "runtime_only_fallback"
            debug_payload["retrieval"]["authority_retrieval_mode"] = "runtime_fallback_used"
        else:
            debug_payload["retrieval"]["declaration_answer_mode"] = "missing_declarations"
            debug_payload["retrieval"]["authority_retrieval_mode"] = "workspace_only_detected_not_indexed"

    if debug_payload is not None:
        debug_payload["retrieval"]["authoritative_files_detected"] = list(authoritative_files)
        debug_payload["retrieval"]["authoritative_files_required"] = list(authoritative_files) if decl_query else []
        debug_payload["retrieval"]["authoritative_files_retrieved"] = []
        debug_payload["retrieval"]["authoritative_files_missing"] = list(authoritative_files)
        debug_payload["retrieval"]["declaration_query_trigger_reason"] = declaration_query_trigger_reason

    # Fetch full content from planned files + deterministic neighborhood expansion.
    expanded_files = []
    neighbor_files = []
    for fname in planned_files:
        neighborhood = _file_neighborhood(fname, mode, workspace, repo_fp)
        expanded_files.extend(neighborhood)
        neighbor_files.extend(neighborhood)
    for fname in expanded_files:
        try:
            if is_edit_suggestion_query(query):
                merged = _merge_indexed_file_chunks(fname)
                if merged:
                    all_chunks.append(merged)
                    files_loaded.append(fname)
                    continue
            file_chunks = _indexer_module.get_chunks_for_file(fname)
            if file_chunks:
                all_chunks.extend(file_chunks)
                files_loaded.append(fname)
        except Exception as e:
            audit.warning(f"FILE_FETCH_FAIL {request_id} | {fname} | {e}")

    if decl_query and authoritative_files:
        for fname in authoritative_files:
            if fname in files_loaded:
                continue
            try:
                file_chunks = _indexer_module.get_chunks_for_file(fname)
                if file_chunks:
                    all_chunks.extend(file_chunks)
                    files_loaded.append(fname)
            except Exception as e:
                audit.warning(f"AUTHORITATIVE_FETCH_FAIL {request_id} | {fname} | {e}")

    # Always add semantic search results to catch anything the planner missed
    try:
        semantic = search_codebase(query, n_results=3, debug_mode=debug_mode)
        for c in semantic:
            if c["file"] not in files_loaded:
                all_chunks.append(c)
                if c["file"] not in files_loaded:
                    files_loaded.append(c["file"])
                    semantic_fallback_files.append(c["file"])
    except Exception:
        pass

    if is_edit_suggestion_query(query):
        all_chunks, edit_loaded = _expand_edit_suggestion_chunks(query, all_chunks, pmap, workspace)
        for fname in edit_loaded:
            if fname not in files_loaded:
                files_loaded.append(fname)
        if debug_payload is not None:
            debug_payload.setdefault("edit_suggestion", {})["full_files_loaded"] = list(edit_loaded)

    if not all_chunks:
        authoritative_retrieved = sorted(
            [
                f for f in files_loaded
                if f in authoritative_files or any(Path(p).name == Path(f).name for p in authoritative_files)
            ]
        )
        if debug_payload is not None:
            debug_payload["planning"]["files_loaded"] = list(files_loaded)
            debug_payload["planning"]["semantic_fallback_files"] = list(semantic_fallback_files)
            debug_payload["retrieval"]["files_retrieved"] = list(files_loaded)
            debug_payload["retrieval"]["selected_candidates"] = list(files_loaded)
            debug_payload["retrieval"]["authoritative_files_retrieved"] = authoritative_retrieved
            debug_payload["retrieval"]["authoritative_files_missing"] = [
                p for p in authoritative_files if p not in authoritative_retrieved
            ]
            debug_payload["retrieval"]["forced_authoritative_file"] = bool(authoritative_retrieved)
            _set_planned_authority_modes([])
            debug_payload["final_context"]["files_used"] = list(files_loaded)
        if decl_query and authoritative_files and not authoritative_retrieved:
            audit.warning("authoritative retrieval failure (not packing failure)")
        return (messages, [], debug_payload) if return_debug else (messages, [])

    normalized = normalize_local_retrieval(
        query=query,
        chunks=all_chunks,
        strategy="planned_context",
        top_k=CONTEXT_CHUNKS,
        retrieval_mode=get_execution_mode().value,
        index_state=index_state if isinstance(index_state, dict) else {},
    )
    normalized_chunks = normalized.to_prompt_chunks()
    if debug_payload is not None:
        debug_payload["normalized_retrieval"] = normalized.to_debug_dict()

    anchor_files = _extract_anchor_files(query)
    code_section, packing_info = _pack_context_section(
        query=query,
        map_section=map_section,
        chunks=normalized_chunks,
        anchor_files=anchor_files,
        planned_files=planned_files,
        neighbor_files=neighbor_files,
        conversation_messages=[m for m in messages if m.get("role") != "system"],
        authoritative_files=authoritative_files,
        request_id=request_id,
    )

    sections = build_prompt_sections(
        system_prefix=_BASE_SYSTEM,
        reasoning_policy=reasoning_policy,
        workspace_prefix=map_section,
        retrieval_context=code_section,
        user_turn="",
    )
    system = serialize_prompt_sections(sections)

    audit.info(
        f"CONTEXT {request_id} | planned={planned_files} | "
        f"loaded={files_loaded} | chunks={len(normalized_chunks)} | "
        f"packed={packing_info['packed_chunks']} | kept={packing_info['kept_files']}"
    )

    final_messages = (
        [{"role": "system", "content": system}] + [
            m for m in messages if m.get("role") != "system"
        ]
    )
    if debug_payload is not None:
        debug_payload["planning"]["files_loaded"] = list(files_loaded)
        debug_payload["planning"]["semantic_fallback_files"] = list(semantic_fallback_files)
        debug_payload["retrieval"]["files_retrieved"] = list(files_loaded)
        debug_payload["retrieval"]["selected_candidates"] = list(files_loaded)
        authoritative_retrieved = sorted(
            [
                f for f in files_loaded
                if f in authoritative_files or any(Path(p).name == Path(f).name for p in authoritative_files)
            ]
        )
        debug_payload["retrieval"]["authoritative_files_retrieved"] = authoritative_retrieved
        debug_payload["retrieval"]["authoritative_files_missing"] = [
            p for p in authoritative_files if p not in authoritative_retrieved
        ]
        debug_payload["retrieval"]["forced_authoritative_file"] = bool(authoritative_retrieved)
        if decl_query and authoritative_files and not authoritative_retrieved:
            audit.warning("authoritative retrieval failure (not packing failure)")
        packed_chunks = packing_info["packed_chunks_raw"]
        _set_planned_authority_modes(packed_chunks)
        debug_payload["retrieval"]["chunks_per_file"] = {
            f: sum(1 for c in packed_chunks if c.get("file") == f) for f in sorted(set(files_loaded))
        }
        debug_payload["final_context"]["files_used"] = [c.get("file") for c in packed_chunks if c.get("file")]
        debug_payload["final_context"]["context_size"] = sum(len(c.get("content", "")) for c in packed_chunks)
        debug_payload["final_context"]["packed_chunks"] = packed_chunks
    return (final_messages, files_loaded, debug_payload) if return_debug else (final_messages, files_loaded)


# ── Streaming ─────────────────────────────────────────────────────────────────

async def _stream(messages: list, max_tokens: int, request_id: str, t_start: float, debug_mode: bool = False):
    think_open = "<|channel>"
    think_close = "<channel|>"
    phase = "request_received"
    final_text = ""
    debug_payload = None
    cache = None
    repo_fp = ""
    query = ""
    intent = "unknown"
    retrieval_route = "unknown"
    retrieval_signature = ""
    filtered_out = 0
    is_performance = False
    buffer_answer_for_contract = False
    stream_path = "direct_retrieval"
    cached_semantic = False
    debug_emitted = False
    fallback_reason = "context build returned no payload"
    try:
        _phase_log(request_id, "request_received", max_tokens=max_tokens, message_count=len(messages))
        yield _make_chunk("⚙️ _Analyzing request..._", request_id)

        query = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        is_performance = _is_performance_query(query)
        pmap = _indexer_module._load_project_map() if _indexer_module else {}
        phase = "intent_classified"
        decision = classify_query_intent_details(query)
        intent = decision["intent"]
        retrieval_route = decision["retrieval_route"]
        buffer_answer_for_contract = intent == EDIT_SUGGESTION
        _phase_log(request_id, "intent_classified", intent=intent, retrieval_route=retrieval_route)
        orchestration = orchestration_plan(intent)
        diagnosis = _diagnose_query(query, intent)
        repo_fp = _indexer_module.get_repo_fingerprint() if _indexer_module else ""
        cache = getattr(_indexer_module, "CACHE", None) if _indexer_module else None
        retrieval_signature = f"{retrieval_route}:{intent}:{query}:{CONTEXT_CHUNKS}"
        planned_files = []

        if cache and repo_fp and semantic_cache_allowed(intent, retrieval_route):
            phase = "semantic_cache_lookup"
            _phase_log(request_id, "semantic_cache_lookup_start")
            semantic_hit = cache.semantic_get(
                repo_fp=repo_fp,
                query=query,
                retrieval_signature=retrieval_signature,
                safe_class="descriptive",
            )
            _phase_log(request_id, "semantic_cache_lookup_result", hit=bool(semantic_hit))
            if semantic_hit:
                cached_semantic = True
                stream_path = "semantic_cache_hit"
                fallback_reason = "semantic cache hit"
                if debug_mode:
                    debug_payload = _build_cache_debug_payload(
                        query=query,
                        request_id=request_id,
                        repo_fp=repo_fp,
                        retrieval_signature=retrieval_signature,
                        intent=intent,
                        retrieval_route=retrieval_route,
                        semantic_hit=semantic_hit,
                    )
                audit.info(
                    f"STREAM_DEBUG_PAYLOAD {request_id} | generated={bool(debug_payload)} | path=semantic_cache_hit"
                )
                yield _make_chunk("\n🧩 _Semantic cache hit (safe descriptive answer)_\n\n", request_id)
                final_text, filtered_out = _validate_high_signal_output(semantic_hit, is_performance)
                yield _make_chunk(final_text, request_id)

        if not cached_semantic:
            if INDEXER_READY and pmap and query and not orchestration["skip_patch_plan"]:
                import asyncio

                phase = "planner"
                _phase_log(request_id, "planner_start")
                loop = asyncio.get_event_loop()
                if cache and repo_fp:
                    plan_cached = cache.patch_plan_get(repo_fp=repo_fp, query=query, target_signature="preplan")
                    if plan_cached:
                        planned_files = plan_cached.get("planned_files", [])
                        diagnosis = plan_cached.get("diagnosis", diagnosis)
                if not planned_files:
                    planned_files = await loop.run_in_executor(None, lambda: _plan_files(query, pmap))
                    if cache and repo_fp:
                        cache.patch_plan_set(
                            repo_fp=repo_fp,
                            query=query,
                            target_signature="preplan",
                            value={"diagnosis": diagnosis, "planned_files": planned_files},
                        )
                _phase_log(request_id, "planner_result", planned_files=planned_files)

            phase = "context_build"
            stream_path = "planned_context" if planned_files else "direct_retrieval"
            _phase_log(request_id, "context_build_start", path=stream_path)
            if planned_files and not orchestration["skip_neighborhood"]:
                short_names = [f.split("/")[-1] for f in planned_files]
                yield _make_chunk(f"\n📂 _Reading: {', '.join(short_names)}_", request_id)
                messages, _, debug_payload = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: _build_context_from_plan(
                        messages,
                        planned_files,
                        request_id,
                        diagnosis,
                        debug_mode=debug_mode,
                        return_debug=True,
                    ),
                )
                audit.info(
                    f"STREAM_DEBUG_PAYLOAD {request_id} | generated={bool(debug_payload)} | path={stream_path}"
                )
                if not debug_payload:
                    fallback_reason = "planned context build returned no payload"
            else:
                status = "Loading source-of-truth config..." if is_fast_path_intent(intent) else "Searching codebase..."
                yield _make_chunk(f"\n📂 _{status}_", request_id)
                messages, debug_payload = _build_context(
                    messages,
                    request_id,
                    debug_mode=debug_mode,
                    return_debug=True,
                )
                if debug_payload is not None:
                    retrieval = debug_payload.setdefault("retrieval", {})
                    retrieval.setdefault("orchestration_path", "direct_retrieval")
                    debug_payload["orchestration_path"] = "direct_retrieval"
                audit.info(
                    f"STREAM_DEBUG_PAYLOAD {request_id} | generated={bool(debug_payload)} | path={stream_path}"
                )
                if not debug_payload:
                    fallback_reason = "context build returned no payload"
            _phase_log(request_id, "context_build_result")

            t_context = time.perf_counter()
            ctx_s = t_context - t_start
            prompt = _messages_to_prompt(messages)
            yield _make_chunk(f"\n🧠 _Thinking... (ready in {ctx_s:.1f}s)_\n\n", request_id)
            _phase_log(request_id, "generation_start")

            buffer = ""
            in_think = False
            t_think_start = None
            t_think_total = 0.0
            t_first_token = None
            token_count = 0
            emitted_answer = False

            phase = "generation"
            for chunk in llm(prompt, max_tokens=max_tokens, stream=True, echo=False):
                token = chunk["choices"][0]["text"]
                buffer += token
                if not in_think and think_open in buffer:
                    before = buffer[:buffer.find(think_open)]
                    buffer = buffer[buffer.find(think_open):]
                    in_think = True
                    t_think_start = time.perf_counter()
                    if before.strip():
                        if not buffer_answer_for_contract and not emitted_answer:
                            yield _make_chunk("\n\n---\n\n", request_id)
                            emitted_answer = True
                        if t_first_token is None:
                            t_first_token = time.perf_counter()
                            _phase_log(request_id, "first_token_emitted", seconds=f"{t_first_token - t_start:.2f}")
                        token_count += 1
                        final_text += before
                        if not buffer_answer_for_contract:
                            yield _make_chunk(before, request_id)
                    continue
                if in_think:
                    if think_close in buffer:
                        after = buffer[buffer.find(think_close) + len(think_close):]
                        buffer = after
                        in_think = False
                        t_think_total += time.perf_counter() - t_think_start
                    continue
                if len(buffer) > len(think_open) + 4:
                    emit = _strip_thinking(buffer[:-len(think_open)])
                    buffer = buffer[-len(think_open):]
                    if emit:
                        if not buffer_answer_for_contract and not emitted_answer:
                            yield _make_chunk("\n\n---\n\n", request_id)
                            emitted_answer = True
                        if t_first_token is None:
                            t_first_token = time.perf_counter()
                            _phase_log(request_id, "first_token_emitted", seconds=f"{t_first_token - t_start:.2f}")
                        token_count += 1
                        final_text += emit
                        if not buffer_answer_for_contract:
                            yield _make_chunk(emit, request_id)

            if buffer and not in_think:
                remainder = _strip_thinking(buffer)
                if remainder:
                    if not buffer_answer_for_contract and not emitted_answer:
                        yield _make_chunk("\n\n---\n\n", request_id)
                        emitted_answer = True
                    final_text += remainder
                    if not buffer_answer_for_contract:
                        yield _make_chunk(remainder, request_id)

            filtered_text, filtered_out = _validate_high_signal_output(final_text, is_performance)
            if buffer_answer_for_contract:
                filtered_text = enforce_edit_suggestion_output(
                    filtered_text,
                    _edit_context_from_debug_payload(debug_payload, query),
                )
                yield _make_chunk("\n\n---\n\n", request_id)
                yield _make_chunk(filtered_text, request_id)
            elif is_performance and filtered_text.strip() and filtered_text.strip() != final_text.strip():
                yield _make_chunk(
                    "\n\n🔎 Refined high-signal summary:\n",
                    request_id,
                )
                yield _make_chunk(filtered_text, request_id)

            t_done = time.perf_counter()
            total_s = t_done - t_start
            think_s = t_think_total
            ttft_s = (t_first_token - t_start) if t_first_token else 0.0
            yield _make_chunk(
                f"\n\n---\n⏱ context `{ctx_s:.1f}s` · think `{think_s:.1f}s` · first token `{ttft_s:.1f}s` · total `{total_s:.1f}s`",
                request_id,
            )
            _phase_log(
                request_id,
                "generation_completed",
                context_s=f"{ctx_s:.1f}",
                think_s=f"{think_s:.1f}",
                ttft_s=f"{ttft_s:.1f}",
                total_s=f"{total_s:.1f}",
                chunks=token_count,
            )

        query_type = EDIT_SUGGESTION if intent == EDIT_SUGGESTION else ("performance" if is_performance else "general")
        audit.info(
            f"HIGH_SIGNAL {request_id} | query_type={query_type} | "
            f"applied={is_performance} | filtered_out_items={filtered_out}"
        )

        cache_value = filtered_text if 'filtered_text' in locals() else final_text
        if cache and repo_fp and semantic_cache_allowed(intent, retrieval_route) and cache_value.strip():
            cache.semantic_set(
                repo_fp=repo_fp,
                query=query,
                retrieval_signature=retrieval_signature,
                safe_class="descriptive",
                value=cache_value.strip(),
            )
            cache.flush_metrics()
        if debug_mode:
            payload_to_emit = debug_payload
            payload_kind = "full"
            if not payload_to_emit:
                payload_kind = "fallback"
                payload_to_emit = _minimal_debug_payload(
                    query=query,
                    request_id=request_id,
                    stream_path=stream_path if stream_path else "unknown",
                    reason=fallback_reason,
                    cache_hit=cached_semantic,
                    intent=intent,
                    retrieval_route=retrieval_route,
                    final_context={
                        "files_used": (debug_payload or {}).get("final_context", {}).get("files_used", []),
                        "context_size": (debug_payload or {}).get("final_context", {}).get("context_size", 0),
                    },
                )
            audit.info(
                f"STREAM_DEBUG_EMIT {request_id} | emitted=True | payload_kind={payload_kind} | path={stream_path}"
            )
            yield format_debug_sse_event(payload_to_emit)
            debug_emitted = True
    except Exception as e:
        stream_path = "error"
        fallback_reason = "pipeline exception"
        _phase_log(request_id, "pipeline_failed", failed_phase=phase, error=e)
        if debug_mode and not debug_emitted:
            error_payload = _minimal_debug_payload(
                query=query,
                request_id=request_id,
                stream_path="error",
                reason=f"{fallback_reason}: {phase}",
                cache_hit=cached_semantic,
                intent=intent,
                retrieval_route=retrieval_route,
            )
            audit.info(
                f"STREAM_DEBUG_EMIT {request_id} | emitted=True | payload_kind=fallback | path=error"
            )
            yield format_debug_sse_event(error_payload)
            debug_emitted = True
        yield _make_pipeline_error_event(request_id, phase, e)
        yield _make_error_chunk(request_id, phase, e)
    finally:
        yield "data: [DONE]\n\n"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _edit_context_from_debug_payload(debug_payload: dict | None, query: str):
    if not isinstance(debug_payload, dict):
        return build_edit_suggestion_context([], query=query)
    final_context = debug_payload.get("final_context") if isinstance(debug_payload.get("final_context"), dict) else {}
    chunks = final_context.get("packed_chunks") or final_context.get("packed_chunks_raw") or []
    if not isinstance(chunks, list) or not chunks:
        # Enforcement must be based on code actually sent to the model; if that
        # post-pack context is unavailable, fail closed instead of trusting
        # broader pre-pack retrieval candidates.
        return build_edit_suggestion_context([], query=query)
    if chunks and isinstance(chunks[0], dict) and "content" in chunks[0] and "file" not in chunks[0]:
        chunks = [{**c, "file": c.get("path", "")} for c in chunks]
    return build_edit_suggestion_context(chunks if isinstance(chunks, list) else [], query=query)

def _messages_to_prompt(messages: list) -> str:
    prompt = ""
    for m in messages:
        role    = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            prompt += f"<s>{content}</s>\n"
        elif role == "user":
            prompt += f"<start_of_turn>user\n{content}<end_of_turn>\n"
        elif role == "assistant":
            prompt += f"<start_of_turn>model\n{content}<end_of_turn>\n"
    prompt += "<start_of_turn>model\n"
    return prompt


def _strip_thinking(text: str, strip_edges: bool = False) -> str:
    text = _THINK_PATTERN.sub("", text)
    text = text.replace("$\\rightarrow$", "→").replace("$\\Rightarrow$", "⇒")
    return text.strip() if strip_edges else text


def _make_chunk(content: str, request_id: str) -> str:
    data = {
        "id":      f"chatcmpl-{request_id}",
        "object":  "chat.completion.chunk",
        "created": int(time.time()),
        "model":   "andescode-gemma4-26b",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    return f"data: {json.dumps(data)}\n\n"


if __name__ == "__main__":
    import webbrowser, threading

    _print(f"  [4/4] ✓ Server ready")
    _print("")
    _print("┌─────────────────────────────────────────┐")
    _print(f"│  ✅  AndesCode is running                │")
    _print(f"│                                         │")
    _print(f"│  🖥️   http://localhost:{PORT}/ui           │")
    _print(f"│  📋  Server log: {_safe(LOG_PATH)}      │")
    _print(f"│                                         │")
    _print(f"│  Your AI. Your code. Nobody else.       │")
    _print("└─────────────────────────────────────────┘")
    _print("")

    # Only auto-open browser when running standalone (not inside app.py wrapper)
    if not os.environ.get("ANDESCODE_APP_MODE"):
        def _open_browser():
            time.sleep(1.2)
            webbrowser.open(f"http://localhost:{PORT}/ui")
        threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
