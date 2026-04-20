# ── Offline enforcement — MUST be before any library import ──────────────────
# Zero egress enforced at OS level — no outbound connections during inference.
import os
os.environ["TRANSFORMERS_OFFLINE"]              = "1"
os.environ["HF_DATASETS_OFFLINE"]              = "1"
os.environ["HF_HUB_OFFLINE"]                   = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"]            = "false"

import contextlib
import json
import logging
import re
import sys
import time
import uuid
import threading
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")
from runtime_paths import get_runtime_log_path

import uvicorn
from fastapi import FastAPI, Request
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
from andes_cache.source_of_truth import source_of_truth_guidance, is_declaration_query
from andes_cache.routing import (
    classify_query_intent,
    classify_query_intent_details,
    retrieval_route_for_intent,
    is_fast_path_intent,
    semantic_cache_allowed,
    orchestration_plan,
)
from auto_index import AutoIndexManager, ChangeBatch

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
# Snapshot for startup visibility only.
# NOTE: If you change ANDESCODE_DEBUG_MODE in .env while the server is running,
# this startup value will not change until restart. Request-level debug still
# resolves per request via _resolve_request_debug_mode().
DEBUG_MODE_STARTUP = env_debug_mode()


def _resolve_request_debug_mode(api_debug: bool | None) -> bool:
    """Resolve debug mode for each request: API checkbox/body flag > current env."""
    return resolve_debug_mode(api_flag=api_debug, param_flag=env_debug_mode())

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
    query_type = "performance" if _is_performance_query(query) else "general"
    return (_HIGH_SIGNAL_PERFORMANCE_POLICY if query_type == "performance" else "", query_type)


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
_auto_status_lock = threading.Lock()
_auto_status_message = "Auto-index idle"


def _set_auto_status(message: str) -> None:
    global _auto_status_message
    with _auto_status_lock:
        _auto_status_message = message
    audit.info(f"AUTO_INDEX | {message}")


def _read_integrity_probe_from_indexer() -> dict:
    """
    Optional startup visibility helper.
    Keeps endpoint behavior testable even when the indexer is partially stubbed.
    """
    if not (INDEXER_READY and _indexer_module):
        return {}
    getter = getattr(_indexer_module, "get_startup_integrity_probe", None)
    if not callable(getter):
        return {}
    try:
        return getter() or {}
    except Exception:
        return {}


def _snapshot_relevant_files(root_path: Path) -> dict[str, str]:
    if not _indexer_module:
        return {}
    supported = set(getattr(_indexer_module, "SUPPORTED_EXTENSIONS", set()))
    manifests = set(getattr(_indexer_module, "MANIFEST_FILES", set()))
    skip_dirs = set(getattr(_indexer_module, "SKIP_DIRS", set()))
    file_hash = getattr(_indexer_module, "_file_hash")
    snapshot: dict[str, str] = {}
    for fp in root_path.rglob("*"):
        if not fp.is_file():
            continue
        rel = str(fp.relative_to(root_path))
        if not AutoIndexManager.is_relevant_project_path(
            rel,
            supported_suffixes=supported,
            authoritative_basenames=manifests,
            skip_dirs=skip_dirs,
        ):
            continue
        snapshot[rel] = file_hash(fp)
    return snapshot


def _run_index_stream(path: str, source: str, emit_event, change_batch: ChangeBatch | None = None) -> bool:
    _index_phase_log(source, "index_request_started", path=path)
    if not _load_indexer():
        _index_phase_log(source, "index_failed", failed_phase="load_indexer", error="Indexer not available")
        emit_event({"type": "error", "source": source, "message": "Indexer not available"})
        return False

    acquired = _index_run_lock.acquire(blocking=False)
    if not acquired:
        if source == "auto" and _auto_index_manager:
            _auto_index_manager.request_rerun_if_busy()
            _set_auto_status("Index already in progress; queued one follow-up auto-refresh")
        emit_event({"type": "status", "source": source, "message": "Index already in progress"})
        return False

    try:
        if source == "auto" and _auto_index_manager:
            _auto_index_manager.notify_auto_run_start()
            count = change_batch.count if change_batch else 0
            if change_batch and change_batch.deleted_paths:
                _set_auto_status(f"{count} file changes detected; file deletion detected, performing safe rebuild check")
            else:
                _set_auto_status(f"{count} file changes detected; running background refresh")
            emit_event({"type": "auto_status", "source": source, "message": "Detected file changes, refreshing index..."})

        from indexer import index_codebase_stream
        done_event = None
        embedding_started_logged = False
        storage_started_logged = False
        for event in index_codebase_stream(path):
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
            elif etype == "done":
                _index_phase_log(
                    source,
                    "index_completed",
                    indexed=event.get("indexed"),
                    chunks=event.get("chunks"),
                    decision=event.get("decision"),
                )
                _index_phase_log(source, "project_map_workspace_build_completed", indexed=event.get("indexed"))
            if source == "auto" and event.get("type") == "decision":
                _set_auto_status(event.get("message", "Auto-index decision emitted"))
            emit_event(event)
            if event.get("type") == "done":
                done_event = event

        if done_event and _auto_index_manager:
            _auto_index_manager.start_for_root(path)
            if source == "auto":
                decision = done_event.get("decision", "unknown")
                _set_auto_status(f"Auto-refresh complete ({decision})")
        return True
    except Exception as exc:
        _index_phase_log(source, "index_failed", failed_phase="run_index_stream", error=exc)
        emit_event({"type": "error", "source": source, "message": str(exc)})
        if source == "auto":
            _set_auto_status(f"Auto-refresh failed: {exc}")
        return False
    finally:
        _index_run_lock.release()
        if source == "auto" and _auto_index_manager:
            rerun = _auto_index_manager.notify_auto_run_end()
            if rerun:
                _set_auto_status("Additional file changes arrived during indexing; running one follow-up refresh")
                _start_auto_index(path, ChangeBatch(changed_paths=set(), deleted_paths=set()))


def _auto_run_index(path: str, batch: ChangeBatch) -> None:
    _run_index_stream(path, source="auto", emit_event=lambda _event: None, change_batch=batch)


def _start_auto_index(path: str, batch: ChangeBatch) -> bool:
    if _index_run_lock.locked():
        if _auto_index_manager:
            _auto_index_manager.request_rerun_if_busy()
        return False
    threading.Thread(target=_auto_run_index, args=(path, batch), daemon=True).start()
    return True


_auto_index_manager = AutoIndexManager(
    snapshot_fn=_snapshot_relevant_files,
    run_index_fn=_start_auto_index,
    status_logger=_set_auto_status,
    debounce_seconds=float(os.getenv("ANDESCODE_AUTO_INDEX_DEBOUNCE_SEC", "2.0")),
    poll_interval=float(os.getenv("ANDESCODE_AUTO_INDEX_POLL_SEC", "1.0")),
    enabled=AutoIndexManager.env_enabled(),
)

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
    doc_count = 0
    if INDEXER_READY and _indexer_module:
        try:
            doc_count = _indexer_module.col.count()
        except Exception:
            pass
    auto_state = _auto_index_manager.status() if _auto_index_manager else {}
    integrity_probe = {}
    with _auto_status_lock:
        auto_message = _auto_status_message
    integrity_probe = _read_integrity_probe_from_indexer()
    return {
        "status":    "running",
        "product":   "AndesCode",
        "version":   "1.0.0",
        "indexer":   INDEXER_READY,
        "doc_count": doc_count,
        "cache":     f"{CACHE_SIZE_GB:.0f}GB",
        "auto_index": auto_state,
        "auto_index_message": auto_message,
        "integrity_probe": integrity_probe,
    }


@app.get("/v1/index/state")
def index_state():
    state = _auto_index_manager.status() if _auto_index_manager else {}
    integrity_probe = {}
    with _auto_status_lock:
        message = _auto_status_message
    integrity_probe = _read_integrity_probe_from_indexer()
    return {**state, "status_message": message, "integrity_probe": integrity_probe}


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

    audit.info(f"REQUEST {request_id} | tokens={max_tokens} | messages={len(messages)}")

    if stream:
        return StreamingResponse(
            _stream(messages, max_tokens, request_id, t_start, debug_mode=debug_mode),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                     "X-Accel-Buffering": "no"},
        )

    messages, debug_payload = _build_context(
        messages, request_id, debug_mode=debug_mode, return_debug=True
    )
    user_query = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    is_performance = _is_performance_query(user_query)
    prompt   = _messages_to_prompt(messages)
    try:
        result = llm(prompt, max_tokens=max_tokens, echo=False)
        text   = _strip_thinking(result["choices"][0]["text"], strip_edges=True)
        text, filtered_out = _validate_high_signal_output(text, is_performance)
        query_type = "performance" if is_performance else "general"
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


@app.post("/v1/debug/explain")
async def debug_explain(request: Request):
    body = await request.json()
    query = body.get("query", "")
    n_results = int(body.get("n_results", CONTEXT_CHUNKS))
    api_debug = body.get("debug_mode")
    debug_mode = _resolve_request_debug_mode(api_debug)
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

    if not _load_indexer():
        return {"error": "Indexer not available"}

    async def _generate():
        import asyncio, queue, threading

        q = queue.Queue()

        def _producer():
            try:
                _run_index_stream(path, source="manual", emit_event=q.put)
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
    _auto_index_manager.stop()
    _set_auto_status("Index watcher stopped")
    return {"ok": True, "watcher_status": _auto_index_manager.status()}


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
        if debug_mode:
            chunks, retrieval_debug = search_codebase(
                query,
                n_results=CONTEXT_CHUNKS,
                debug_mode=debug_mode,
                return_debug=True,
            )
        else:
            chunks = search_codebase(query, n_results=CONTEXT_CHUNKS, debug_mode=debug_mode)

        if not chunks:
            sections = build_prompt_sections(
                system_prefix=_BASE_SYSTEM,
                reasoning_policy=reasoning_policy,
                workspace_prefix=map_section,
                retrieval_context="",
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
            chunks=chunks,
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
            f"CONTEXT {request_id} | chunks={len(chunks)} | "
            f"packed={packing_info['packed_chunks']} | files={packing_info['kept_files']}"
        )

        base = [{"role": "system", "content": system}] + [
            m for m in messages if m.get("role") != "system"
        ]
        return (base, retrieval_debug) if return_debug else base

    except Exception as e:
        audit.warning(f"CONTEXT_FAIL {request_id} | {_safe(e)}")
        return (messages, None) if return_debug else messages


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
    extracted_paths = _extract_execution_paths([c["chunk"] for c in packed.chunks], max_paths=5)
    path_section = _render_execution_path_context(extracted_paths) if _is_performance_query(query) else ""
    code_section = path_section + "## Retrieved Code (Validation Context)\n\n"
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
    patch_intent = intent == "code_fix_or_patch"
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

    all_chunks  = []
    files_loaded = []
    semantic_fallback_files = []

    debug_payload = None
    if debug_mode:
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
                "authoritative_files_retrieved": [],
                "authoritative_files_missing": [],
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

    decl_query = is_declaration_query(query, intent=(diagnosis or {}).get("intent", ""))
    authoritative_files = []
    if workspace:
        authoritative_files.extend(workspace.get("manifests", []) or [])
        authoritative_files.extend((workspace.get("config_graph", {}) or {}).get("config_files", []) or [])
    authoritative_files = sorted(set(authoritative_files))
    if debug_payload is not None:
        debug_payload["retrieval"]["authoritative_files_detected"] = list(authoritative_files)
        debug_payload["retrieval"]["authoritative_files_retrieved"] = []
        debug_payload["retrieval"]["authoritative_files_missing"] = list(authoritative_files)

    # Fetch full content from planned files + deterministic neighborhood expansion.
    expanded_files = []
    neighbor_files = []
    for fname in planned_files:
        neighborhood = _file_neighborhood(fname, mode, workspace, repo_fp)
        expanded_files.extend(neighborhood)
        neighbor_files.extend(neighborhood)
    for fname in expanded_files:
        try:
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

    if not all_chunks:
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
            debug_payload["final_context"]["files_used"] = list(files_loaded)
        if decl_query and authoritative_files:
            audit.warning("authoritative retrieval failure (not packing failure)")
        return (messages, [], debug_payload) if return_debug else (messages, [])

    anchor_files = _extract_anchor_files(query)
    code_section, packing_info = _pack_context_section(
        query=query,
        map_section=map_section,
        chunks=all_chunks,
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
        f"loaded={files_loaded} | chunks={len(all_chunks)} | "
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
        if decl_query and authoritative_files and not authoritative_retrieved:
            audit.warning("authoritative retrieval failure (not packing failure)")
        packed_chunks = packing_info["packed_chunks_raw"]
        debug_payload["retrieval"]["chunks_per_file"] = {
            f: sum(1 for c in packed_chunks if c.get("file") == f) for f in sorted(set(files_loaded))
        }
        debug_payload["final_context"]["files_used"] = [c.get("file") for c in packed_chunks if c.get("file")]
        debug_payload["final_context"]["context_size"] = sum(len(c.get("content", "")) for c in packed_chunks)
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
        _phase_log(request_id, "intent_classified", intent=intent, retrieval_route=retrieval_route)
        orchestration = orchestration_plan(intent)
        diagnosis = _diagnose_query(query, intent)
        repo_fp = _indexer_module.get_repo_fingerprint() if _indexer_module else ""
        cache = getattr(_indexer_module, "CACHE", None) if _indexer_module else None
        retrieval_signature = f"{retrieval_route}:{intent}:{query}:{CONTEXT_CHUNKS}"
        planned_files = []
        cached_semantic = False

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
            _phase_log(request_id, "context_build_start", path="planned_context" if planned_files else "direct_retrieval")
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
                        if not emitted_answer:
                            yield _make_chunk("\n\n---\n\n", request_id)
                            emitted_answer = True
                        if t_first_token is None:
                            t_first_token = time.perf_counter()
                            _phase_log(request_id, "first_token_emitted", seconds=f"{t_first_token - t_start:.2f}")
                        token_count += 1
                        final_text += before
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
                        if not emitted_answer:
                            yield _make_chunk("\n\n---\n\n", request_id)
                            emitted_answer = True
                        if t_first_token is None:
                            t_first_token = time.perf_counter()
                            _phase_log(request_id, "first_token_emitted", seconds=f"{t_first_token - t_start:.2f}")
                        token_count += 1
                        final_text += emit
                        yield _make_chunk(emit, request_id)

            if buffer and not in_think:
                remainder = _strip_thinking(buffer)
                if remainder:
                    if not emitted_answer:
                        yield _make_chunk("\n\n---\n\n", request_id)
                        emitted_answer = True
                    final_text += remainder
                    yield _make_chunk(remainder, request_id)

            filtered_text, filtered_out = _validate_high_signal_output(final_text, is_performance)
            if is_performance and filtered_text.strip() and filtered_text.strip() != final_text.strip():
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

        query_type = "performance" if is_performance else "general"
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
        if debug_mode and debug_payload:
            yield format_debug_sse_event(debug_payload)
    except Exception as e:
        _phase_log(request_id, "pipeline_failed", failed_phase=phase, error=e)
        yield _make_pipeline_error_event(request_id, phase, e)
        yield _make_error_chunk(request_id, phase, e)
    finally:
        yield "data: [DONE]\n\n"


# ── Helpers ───────────────────────────────────────────────────────────────────

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
