# Cache Debugging

## Cache directory layout

Default cache path:

- `index/cache/`
  - `.schema`
  - `workspace/*.json`
  - `retrieval/*.json`
  - `file_neighborhood/*.json`
  - `prompt_prefix/*.json`
  - `patch_plan/*.json`
  - `semantic/*.json`
  - `metrics/cache_metrics.json`

## Useful checks

- Verify active repo fingerprint in `.file_hashes.json` (`__fingerprint__`).
- Confirm retrieval keys are separated by query normalization and index version.
- Inspect metrics for hit/miss/invalidation/stale prevention counters.

## Troubleshooting stale behavior

1. Re-index project so hashes and fingerprint are refreshed.
2. If files were deleted, verify a full rebuild event happened.
3. Bump cache schema version when changing cache value shape.
4. For safety-critical changes, clear `index/cache/` and re-run indexing.

## Semantic cache safety

Semantic cache only applies to descriptive query classes.
Patch/fix intents bypass semantic cache and continue through plan + generation.
