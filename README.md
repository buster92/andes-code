# 🏔️ AndesCode

**Private local AI for codebases. Local by default. Remote inference only when explicitly configured.**

[![License](https://img.shields.io/badge/license-Source--Available-lightgrey.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![Model](https://img.shields.io/badge/model-Gemma%204%2026B-orange)](https://huggingface.co/lmstudio-community/gemma-4-26B-A4B-it-GGUF)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Windows-lightgrey)](#hardware-guide)

AndesCode runs Gemma 4 26B on your hardware in `LOCAL` mode. It indexes your codebase, understands your project structure, and answers questions through its own native desktop interface. It can also run in `REMOTE_INFERENCE` mode when explicitly configured for private/self-hosted infrastructure.

## Execution Modes (at a glance)

- **`LOCAL` mode (default):** repository, index, retrieval, prompts, and inference stay on the same machine.
- **`REMOTE_INFERENCE` mode:** repository and index stay local, but selected retrieved chunks plus metadata are sent to your configured private remote server for inference.
- AndesCode does **not** provide a hosted SaaS.

---

## Why local AI for code?

Every cloud coding assistant has the same architecture: your code leaves your machine, hits someone else's server, and comes back as a suggestion. For most developers, that's a fine trade-off.

For some, it isn't.

| | AndesCode | GitHub Copilot | Cursor | Claude |
|---|---|---|---|---|
| Code stays on your machine | ✅ | ❌ | ❌ | ❌ |
| Works fully offline | ✅ | ❌ | ❌ | ❌ |
| No token bills | ✅ | ❌ | ❌ | ❌ |
| Local audit log | ✅ | ❌ | ❌ | ❌ |
| Frontier-class model | ✅ | ✅ | ✅ | ✅ |
| Deterministic / no outages | ✅ | ❌ | ❌ | ❌ |

AndesCode is built for developers who work with client code under NDA, operate in regulated industries (healthcare, legal, finance, defense), or simply believe their code is their own.

---

## Who this is for

- Teams working with sensitive or proprietary code (NDA, IP-heavy projects)
- Companies in regulated environments (finance, healthcare, legal)
- Developers who want full control over their AI tooling and data flow

---

## Features

- 🧠 **Gemma 4 26B** — high-capability open-weight model running entirely on your hardware
- 🔍 **Codebase-aware** — indexes your project, builds a project map, injects relevant context automatically
- 🗺️ **Project intelligence** — detects language, stack, entry points, domain, and key symbols on indexing
- 🔎 **Smart retrieval** — two-step planning (model selects relevant files first), query routing by filename/symbol/intent, and 4-axis re-ranking
- 🎯 **Token-aware context packing** — prompt assembly is budgeted against model context window, with deterministic priority-based truncation instead of overflow failures
- 🧱 **Multi-layer caching** — repo-fingerprint-scoped workspace/retrieval/neighborhood/prompt-prefix/patch-plan caches with strict invalidation
- 📌 **Deterministic routing for repo questions** — config/dependency/manifest questions use a source-of-truth config-first path before inferred code usage
- 🛠️ **Safe edit/apply primitive (v1)** — deterministic single-file exact-match edits with hash stale-context protection and unified diff preview
- ⚠️ **Coverage warnings** — the model is told when it has a partial view of a file, so it never pretends to have context it doesn't
- 🔒 **Execution-mode privacy controls** — `LOCAL` keeps inference on-device with offline flags; `REMOTE_INFERENCE` sends only selected retrieved chunks + metadata to your configured private server
- ⚡ **Fast** — KV cache warm-up on startup, 30–40 tokens/second on Apple Silicon, streaming responses
- 🖥️ **Native desktop app** — runs as a native window on macOS and Windows via the built-in web UI
- 📋 **Audit log** — every request logged locally with metadata only; proof of isolation for compliance

---

## Requirements

| Platform | Hardware | RAM / VRAM |
|---|---|---|
| Apple Silicon Mac | M1 / M2 / M3 / M4 | 32GB unified memory |
| Windows / Linux | NVIDIA RTX 3090, 4090, 5090 | 24–32GB VRAM |

- Python 3.10+
- ~18GB free disk space

---

## Quick Start

**1. Clone**

```bash
git clone https://github.com/buster92/andes-code
cd andes-code
```

**2. Run the launcher**

```bash
python3 launch.py
```

That's it. On first run the launcher:

- Detects your hardware (Apple Silicon → Metal, NVIDIA → CUDA)
- Installs all dependencies with the correct GPU flags
- Opens the AndesCode native window, which automatically:
  - Downloads Gemma 4 26B (~16GB) from Hugging Face — progress shown on screen, resumes if interrupted
  - Loads the model into memory
  - Starts the local server

From there, the app guides you through indexing your project and you can start asking questions immediately. On subsequent runs, `python3 launch.py` just starts the app — model already cached, ready in seconds.

---

## How It Works

```
Index your project
        ↓
Files are chunked with language-aware boundary detection
Embeddings stored in ChromaDB (local)
Project map built: language, stack, domain, entry points, symbol index
Workspace intelligence cached to disk (schema-versioned, artifact-level reuse)
        ↓
You ask a question in the AndesCode window
        ↓
Diagnosis + patch-plan stages run before final generation
Safe descriptive queries can reuse scoped semantic cache
Prompt built from deterministic sections for prefix/KV reuse
Config/dependency questions take a fast path (skip patch-planning flow)
        ↓
Step 1 — Planning: model scans your project map and identifies
         the most relevant files for your question
        ↓
Step 2 — Retrieval: those files are loaded in full, plus
         semantic search fills any gaps the planner missed
        ↓
Token-aware packing keeps anchor/planned/neighbor files first and
truncates lower-priority context when needed to stay under model limits
        ↓
Project map + code context injected into system prompt
Coverage warnings added if any file is only partially retrieved
        ↓
Gemma 4 generates a response grounded in your actual codebase
Streams to the UI with timing metadata
        ↓
Everything logged locally. Code never uploaded.
```


## Safe Edit/Apply (v1)

AndesCode includes a minimal, deterministic file edit primitive for controlled code updates.

### Edit model

```python
EditOperation(
  file_path="src/example.py",
  old_content="return old_value",
  new_content="return new_value",
)
```

### Safety guarantees

- Exact-match only: `old_content` must match exactly in the target file.
- No fuzzy matching or fallback behavior.
- Stale-context protection: apply is blocked when on-disk file hash differs from indexed hash.
- Writes are blocked when the file is missing or not indexed.
- Successful writes trigger single-file re-index only (no full rebuild).

### Diff preview

Use unified diff preview before apply:

```python
generate_diff_preview(old_text, new_text, file_path="src/example.py")
```

### v1 limitations

- Single-file operations only.
- One exact match required for deterministic replacement.
- No autonomous planning or multi-file orchestration.

---

## Privacy Model

### Always local

- Your source code (never read by any external server)
- ChromaDB vector embeddings of your code
- Every query and every response
- Runtime logs in `~/Documents/AndesCode/` (`server.log`, `app.log`)
- Project map, symbol index, and file hash cache

### Offline enforcement

Offline environment flags are set at process startup before model libraries initialize, preventing outbound network calls during inference.

```python
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"
```

### Downloaded once, then cached

| Item | Size | Source |
|---|---|---|
| Gemma 4 26B Q4 model | ~16 GB | Hugging Face |
| `all-MiniLM-L6-v2` embeddings | ~90 MB | Hugging Face |

Both are cached permanently after first run.

### Runtime log format and location

Runtime logs are written outside the repository to:

- `~/Documents/AndesCode/server.log` (chat + indexing pipeline phases)
- `~/Documents/AndesCode/app.log` (desktop wrapper lifecycle/setup)

Logs record metadata only — no code content, no query text, no responses. Absolute paths and usernames are stripped from log entries.

```
2026-04-08 09:15:33 | CHAT d24024dd | phase=request_received | max_tokens=1024 | message_count=1
2026-04-08 09:15:34 | CHAT d24024dd | phase=context_build_start | path=direct_retrieval
2026-04-08 09:15:42 | CHAT d24024dd | phase=generation_completed | context_s=1.1 | think_s=2.3 | ttft_s=2.1 | total_s=8.4 | chunks=47
```

**Logged:** request ID, token count, file names of retrieved chunks, timing.  
**Never logged:** query text, response text, code content, file paths, usernames.

### Troubleshooting stuck responses

If the UI shows status updates but no answer:

1. Open `~/Documents/AndesCode/server.log`.
2. Find the request by `request_id` and inspect the final `phase=...` line.
3. For failures, look for `phase=pipeline_failed` plus `failed_phase=...` and `error=...`.

The frontend now surfaces backend stream failures as visible assistant errors, and streams always terminate with `[DONE]` to prevent indefinite spinner/status hangs.

### Context budget tuning

AndesCode now computes a real prompt budget from the model context window before injecting retrieved code. If retrieval is too large, context is truncated by deterministic priority:

1. Anchor files explicitly mentioned in the question
2. Planner-selected files
3. Neighborhood-expanded files
4. Semantic fallback chunks

Relevant environment knobs:

- `MODEL_CONTEXT_WINDOW` (default `8192`)
- `CONTEXT_RESERVED_RESPONSE_TOKENS` (default `1400`)
- `CONTEXT_SAFETY_MARGIN_TOKENS` (default `256`)

### Network access summary

| Phase | Network | Notes |
|---|---|---|
| First-run model download | ✅ Once | ~16GB from Hugging Face |
| First-run embedding download | ✅ Once | ~90MB from Hugging Face |
| Indexing | ❌ Never | Fully local |
| Answering queries | ❌ Never | Fully local |

### Indexing privacy defaults

- Real dotenv files are skipped by default during indexing to reduce accidental secret ingestion (`.env`, `.env.local`, `.env.development`, `.env.production`, `.env.test`, `.env.staging`).
- Dotenv examples/templates are indexed (`.env.example`, `.env.sample`, `.env.template`, `example.env`, `sample.env`, `template.env`) so AndesCode can understand configuration shape without storing real secret values.
- Other env-like files (for example `config.env` or `secrets.env`) are skipped by default unless they match the explicit template/example allowlist.

---

## Hardware Guide

| Hardware | Model | Speed |
|---|---|---|
| Apple M1/M2 Pro 32GB | Gemma 4 26B Q4 | ~20–30 t/s |
| Apple M3/M4 Pro 32GB | Gemma 4 26B Q4 | ~30–40 t/s |
| Apple M2/M3 Max 64GB | Gemma 4 31B Q4 | ~25–35 t/s |
| NVIDIA RTX 3090/4090 24GB | Gemma 4 26B Q4 | ~35–50 t/s |
| NVIDIA RTX 5090 32GB | Gemma 4 31B Q4 | ~50–70 t/s |

---

## Configuration

All configuration lives in `.env`:

```bash
MODEL_PATH=models/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf
PORT=8080
CONTEXT_CHUNKS=5        # code chunks injected per query
MODEL_CONTEXT_WINDOW=8192
CONTEXT_RESERVED_RESPONSE_TOKENS=1400
CONTEXT_SAFETY_MARGIN_TOKENS=256
CACHE_SIZE_GB=2.0       # KV cache size allocated at startup
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
HF_HUB_OFFLINE=1
TOKENIZERS_PARALLELISM=false
ANDESCODE_EXECUTION_MODE=LOCAL  # LOCAL (default) or REMOTE_INFERENCE
ANDESCODE_REMOTE_SERVER_URL=http://127.0.0.1:8080  # used only in REMOTE_INFERENCE mode
```

For large projects or architectural questions, increase `CONTEXT_CHUNKS` to 7–10. The retrieval pipeline automatically widens its candidate pool for broad queries — this setting controls how many final chunks land in the prompt.

### Execution modes

- `LOCAL` (default): existing end-to-end behavior (local indexing, retrieval, and inference).
- `REMOTE_INFERENCE`: local indexing + local retrieval remain on the client host. The client builds a strict structured payload (`query`, workspace metadata, retrieval metadata, retrieved chunks, options) and sends it to `${ANDESCODE_REMOTE_SERVER_URL}/v1/ask` for inference-only generation. Server-side `/v1/ask` answers only from that payload.
- Remote payload contract reference: `docs/remote-inference-contract.md`.

#### Remote inference v1 boundaries (explicit)

- The **repository and index stay local on the client**.
- AndesCode sends only **selected retrieved chunks + retrieval/workspace metadata** to the configured remote server.
- AndesCode does **not** send the full repository to the server in `REMOTE_INFERENCE` mode.
- v1 is inference-only (Q&A). **Code editing/patch application is not included**.

#### Remote observability + failure handling

When `ANDESCODE_EXECUTION_MODE=REMOTE_INFERENCE`, AndesCode emits lightweight metadata logs (no full chunk dumps) in `andes_cache/audit.log`:

- Client/proxy path: `request_id`, execution mode enabled, `workspace_id`, branch/commit (if available), retrieved chunk count, payload send start/success/failure.
- Server inference path (`/v1/ask`): `request_id`, protocol version, received chunk count, validation failures, generation start/end, stream completion/failure.

Common remote error codes surfaced to clients:

- `remote_unreachable` — remote server cannot be reached.
- `validation_error` — payload failed schema validation.
- `unsupported_protocol` — protocol mismatch (currently only `andes.remote.v1`).
- `empty_retrieval` — no chunks available for remote inference.
- `remote_stream_interrupted` — streaming ended unexpectedly.

---

## Supported Languages

AndesCode indexes a broad set of text-based project files:

- **Code:** Python, JavaScript/TypeScript, Java, Kotlin, Swift, Go, Rust, C/C++, C#, Ruby, PHP
- **Data/statistics:** R, SQL, Jupyter notebooks
- **Docs:** Markdown, MDX, TXT
- **Config/build:** TOML, YAML/YML, XML, `.properties`, Gradle files, package manifests
- **Scripts/web:** SH, Bash, HTML, CSS

Binary, oversized, and generated files are skipped.

See `docs/indexing-policy.md` for the exact indexing and skip policy.

---

## Roadmap

- [ ] File watcher — automatic incremental re-index on save
- [ ] AST-aware chunking — deeper boundary detection beyond regex
- [ ] KVTC context compression — fit larger codebases in context
- [ ] Private tunnel (Tailscale/WireGuard) for mobile access
- [ ] iOS/Android chat client
- [ ] Cryptographic egress proof for SOC 2 compliance
- [ ] Pre-configured hardware bundle (Mac Mini)

---

## Security Model

AndesCode is designed to run fully locally and offline during inference.

However, users are responsible for validating their own environment and dependencies for compliance requirements. AndesCode does not claim formal certification (e.g., SOC 2, ISO) at this stage.

See `docs/security-threat-model.md` for detailed threat model and enterprise controls.

---

## FAQ

**Does any code leave my machine?**  
In `LOCAL` mode, no repository content leaves your machine during inference. In `REMOTE_INFERENCE` mode, AndesCode sends only the retrieved chunks and metadata required for answer generation to your configured remote server (`ANDESCODE_REMOTE_SERVER_URL`), not the full repository.

**Does it integrate with VS Code, Cursor, or other IDEs?**  
Not at this time. AndesCode is a standalone desktop app with its own interface. IDE plugin integration is on the roadmap but not currently supported.

**Can I use a different model?**  
Yes — any GGUF model compatible with llama.cpp. Update `MODEL_PATH` in `.env`.

**Does it work on Windows or Linux?**  
Yes, with an NVIDIA GPU. `launch.py` detects `nvidia-smi` and compiles llama-cpp-python with CUDA automatically. Metal acceleration is Apple Silicon only.

**Answers seem generic or miss important files. What's wrong?**  
Check that indexing completed — you should see `✅ Done — X files`. For large projects, increase `CONTEXT_CHUNKS` in `.env`. You can also reference a specific file by name in your question — AndesCode will load all indexed chunks from that file directly.

**How do I re-index after changing files?**  
Run `python3 indexer.py /path/to/your/project` again. MD5 hashing ensures only changed files are re-processed — unchanged files are reused from the existing index instantly.

**How do I inspect cache behavior?**  
See `docs/cache-debugging.md` for cache layout, metrics, and invalidation troubleshooting. You can run `python benchmark_cache.py` for cold vs warm cache instrumentation.

**How do I enable structured retrieval debug mode?**  
Debug mode is off by default. You can enable it via:
- Environment variable: `ANDESCODE_DEBUG_MODE=1`
- API flag: include `"debug_mode": true` in `/chat/completions` or `/v1/debug/explain`
- Function parameter: `search(query, debug_mode=True)` in `indexer.py`

When enabled, AndesCode emits a deterministic debug payload with intent, source-of-truth selection, retrieval/ranking decisions, and failure signals. The web UI shows it in a collapsible panel.

For declaration/config/dependency questions, debug payloads now include authoritative retrieval guarantees:
- `authoritative_files_detected`: authoritative paths found in workspace metadata.
- `authoritative_files_required`: authoritative paths that must be retrieved for this query.
- `authoritative_files_retrieved` / `authoritative_files_missing`: explicit split of indexed vs missing authoritative sources.
- `forced_authoritative_file`: whether authoritative context was force-included.
- `authority_selection_reason`: why authoritative paths were selected.
- `authority_retrieval_mode`: `direct_chunk_load`, `integrity_repair`, `workspace_only_detected_not_indexed`, `semantic_fallback_blocked`, or `runtime_fallback_used`.
- `declaration_answer_mode`: `declared_only`, `declared_plus_runtime`, `declared_partial_only`, `runtime_only_fallback`, or `missing_declarations`.

If AndesCode detects authoritative files in workspace metadata but cannot retrieve chunks from the index, it now emits an explicit limitation (`workspace_only_detected_not_indexed`) instead of silently degrading to inferred/runtime-only answers.

**Is there a hosted version?**  
No official AndesCode-hosted SaaS is available. Remote inference is supported only for self-hosted/private-network deployments that you configure and operate.

---

## License

AndesCode is source-available.

- Free for personal use and internal company use
- Commercial redistribution, resale, managed hosting, or offering AndesCode as a service requires a commercial license

See [LICENSE](LICENSE) for full terms.

This licensing model allows teams to use AndesCode freely inside their organization, while preventing third parties from reselling or hosting it as a competing service.

---

## Contributing

PRs welcome.

Highest-value contributions right now:

- Windows / Linux setup testing and documentation
- AST-aware chunking (deeper than current regex-based boundary detection)
- File watcher for automatic incremental re-indexing

## Test tiers and CI defaults

AndesCode tests are split by execution tier:

- `tests/unit/` — deterministic pure-logic tests (default CI suite).
- `tests/integration/` — indexer/embedding/server/model tests (opt-in).
- `tests/eval/` — quality evaluation suites (opt-in).

Default CI runs **only** `tests/unit/` using `requirements-ci.txt`, so it does **not** require:

- a running AndesCode server
- Hugging Face/network downloads
- cached embedding models
- a loaded LLM model
- local `audit.log`
- full runtime dependencies from `requirements.txt` (`llama-cpp-python`, `chromadb`, `sentence-transformers`, `huggingface_hub`)

Full runtime dependencies from `requirements.txt` are only required for local app usage plus opt-in integration/model/eval test tiers.

Run full validation locally when you have model + server dependencies available:

```bash
ANDESCODE_RUN_INTEGRATION_TESTS=1 ANDESCODE_RUN_MODEL_TESTS=1 python -m pytest tests/integration -v
ANDESCODE_RUN_EVAL_TESTS=1 python3 tests/eval/eval_runner.py --suite fast --fixture android
```

> Note 1: eval `fast` is model-free, but it is **not** embedding-free. It still depends on retrieval/index embedding availability.
>
> Note 2: answer eval (`--suite eval` or `--suite full` phase 2) is currently **Android-only** (`--fixture android`).
>
> Note 3: integration/model/server/eval tiers are all opt-in and intentionally excluded from default CI.

Optional future CI split:
- Keep default PR CI on `tests/unit/` for speed and determinism.
- Add a separate model-enabled workflow/job (manual, nightly, or protected-branch) for integration + answer eval.

---

## Built in the Andes. Runs everywhere.

AndesCode is built by an independent developer from Latin America. It exists because some teams require full control over their code, infrastructure, and data flow.

Source-available. Free for personal use and internal company use. Commercial redistribution, resale, managed hosting, or offering AndesCode as a service requires a commercial license.

---

*Private AI for your codebase — local by default, with optional private remote inference when explicitly configured.*
