# AndesCode Architecture

## High-level overview
AndesCode is a desktop application that starts a local backend process, indexes a selected repository into searchable code chunks, retrieves the most relevant chunks for each question, and then uses an LLM to generate an answer; generation can run locally with `llama.cpp` or be delegated to an optional remote inference server while retrieval remains local.

## System components

### 1) Desktop launcher (`app.py`)
- Creates the native desktop window.
- Runs first-start checks (Python, disk, hardware, dependencies, model availability).
- Starts `server.py` as a local subprocess and points the UI to it.
- Writes runtime logs under the app data directory.

### 2) Local server (`server.py`, FastAPI)
- Hosts the local API/UI endpoints.
- Exposes indexing and ask/chat endpoints.
- Coordinates retrieval, prompt building, and answer generation.
- Switches behavior based on execution mode (`LOCAL` or `REMOTE_INFERENCE`).

### 3) Indexer (`indexer.py`)
- Scans repository files recursively.
- Applies skip rules for ignored directories, real `.env` files, unsupported extensions, oversized/binary files.
- Splits files into chunks (language-aware boundaries for code, line-window chunking fallback, notebook cell extraction for `.ipynb`).
- Supports incremental indexing using file hashes and index-state decisions.

### 4) Vector store (ChromaDB)
- Uses a persistent ChromaDB collection to store chunk embeddings and metadata.
- Lives on local disk and is reused between sessions.

### 5) Embedding model
- Uses `sentence-transformers/all-MiniLM-L6-v2` to embed both repository chunks and user queries.
- Embeddings are generated locally.

### 6) LLM runtime (`llama.cpp`)
- `server.py` loads a local GGUF model via `llama_cpp.Llama`.
- Prompt context is built from project-map metadata plus retrieved chunks.
- Response is streamed or returned as a single completion.

### 7) Remote inference server (optional)
- Enabled by execution mode `REMOTE_INFERENCE`.
- Receives a structured payload (`/v1/ask`) containing query metadata + retrieved chunks.
- Generates an answer remotely and returns it to the local server.

## Data flow (step-by-step)

### A) Indexing flow
1. User selects a repository from the desktop app.
2. Local server calls the indexer.
3. Indexer collects candidate files (with skip/allow rules).
4. Files are chunked:
   - Source/code/text files: boundary-aware or line-window chunks.
   - Notebooks: cell source extraction, then chunking.
5. Chunk text is embedded with the embedding model.
6. Embeddings + metadata are stored in local ChromaDB.
7. Auxiliary local metadata is written (project/workspace/index state, hashes, cache).

### B) Query flow
1. User asks a question in the UI.
2. Local server extracts the user query.
3. Query embedding is generated locally.
4. Retriever searches ChromaDB and selects top chunks.
5. Prompt builder assembles:
   - base system instructions,
   - project/workspace context,
   - retrieved chunk context,
   - current user turn.
6. LLM generation runs in one of two modes:
   - `LOCAL`: local `llama.cpp` model generates answer.
   - `REMOTE_INFERENCE`: payload is sent to remote server, which generates answer.
7. Answer is returned to the UI (streaming or non-streaming).

## LOCAL vs REMOTE_INFERENCE

### `LOCAL`
- Indexing, embeddings, retrieval, prompt construction, and generation all run on the same machine.
- Repository content and prompts stay local by default.

### `REMOTE_INFERENCE`
- Indexing, embeddings, and retrieval still run locally.
- Local server sends query metadata + selected retrieved chunks to the remote `/v1/ask` endpoint.
- Remote server performs generation and returns answer events/text.
- Full repository synchronization is out of scope; only selected context is sent.

## Diagram

```text
[Desktop UI / app.py]
         |
         v
[Local FastAPI Server / server.py]
         |
         +--> [Indexer / indexer.py] --> [Local ChromaDB index]
         |
         +--> [Local Retrieval + Prompt Builder]
                         |
                         +--> LOCAL ----------> [LLM via llama.cpp]
                         |
                         +--> REMOTE_INFERENCE -> [Remote /v1/ask server]
```

## Storage locations
- **Index + vector DB (local):** `./index/` (Chroma data and index metadata files).
- **Logs:** `~/Documents/AndesCode/` (for example `app.log`, `server.log`).
- **Model files:** `./models/` if present, otherwise `~/Documents/AndesCode/models/`.
- **Cache:** `./index/cache/` (retrieval/prompt/index-related cache state).

## Key design principles
- **Local-first:** retrieval pipeline is local in both modes.
- **No default external APIs:** default execution path is fully local.
- **Deterministic indexing:** file hashing + index-state decisions guide rebuild/reuse behavior.
- **Retrieval/generation separation:** retrieval context is prepared before model inference and can be routed to local or remote generation.

## Related docs
- [Security Threat Model](./security-threat-model.md)
- [Indexing Policy](./indexing-policy.md)
- [Remote Inference Contract](./remote-inference-contract.md)
