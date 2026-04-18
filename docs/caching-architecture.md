# Caching Architecture

AndesCode now uses a **multi-layer cache** designed for repository-aware coding workflows.

## Layers

1. **Workspace cache**
   - Caches derived repository intelligence artifacts:
     - module graph
     - manifests
     - dependency inventory
     - import graph
     - config graph
     - entry points
     - file-to-module map
     - symbol index
   - Disk-backed and schema-versioned.
   - Supports artifact-level reuse; only changed artifact groups are recomputed.

2. **Retrieval cache**
   - Caches query routing output, ranked candidates, and workspace-structured retrieval responses.
   - Cache key includes:
     - repo fingerprint
     - normalized query
     - retrieval policy version
     - index version
   - Keys are route-aware (`intent` + `retrieval_route`) so config-first and semantic paths never collide.
   - Never reused across repo fingerprints.

3. **File-neighborhood cache**
   - Caches anchor-file neighborhood expansions for modes such as bugfix/architecture.
   - Neighborhood includes anchor file + importers/imported + module peers + likely tests.

4. **Prompt-prefix cache**
   - Prompt is split into deterministic sections:
     - system prefix
     - workspace prefix
     - retrieval context
     - user turn
   - Stable ordering/formatting improves backend prefix/KV reuse.

5. **Patch-plan cache**
   - Flow split into diagnosis → patch plan → generation.
   - Diagnosis + file plans are cached more aggressively than final output.
   - Plan keys are scoped by repo fingerprint and target signature.

6. **Scoped semantic cache**
   - Enabled only for safe descriptive classes (e.g., architecture explanations).
   - Disabled for code patch generation paths.
   - Guarded by repo fingerprint + retrieval signature + template version.

## Invalidation Rules

- Cache store uses explicit schema versioning (`.schema` marker).
- Repo fingerprint changes invalidate retrieval/prompt/neighborhood/patch/semantic layers.
- Project-root switches invalidate all layers, including workspace.
- File deletions trigger safe full index rebuild to prevent stale retrieval.

## Repo Fingerprinting

Fingerprint payload includes:
- resolved root path
- file hashes
- index version
- parser version
- prompt template version
- retrieval policy version

## Correctness Guardrails

- Keys are deterministic and explicit (no fuzzy key matching).
- Repo fingerprint boundary prevents cross-repo leakage.
- Semantic cache is never used for patch-producing intents.
- Config/dependency intents run a fast deterministic path (source-of-truth first, no patch-plan orchestration).
