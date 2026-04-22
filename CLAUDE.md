# AndesCode – Architecture & Decision Log

## Query Intent Routing

The system classifies every incoming query into one of seven intents before
retrieval begins.  Classification is **deterministic and rule-based** — there
is no ML classifier.  All routing logic lives in two files:

| File | Responsibility |
|------|---------------|
| `andes_cache/routing.py` | Intent enum, scoring, regex rules, `classify_query_intent_details()` |
| `andes_cache/source_of_truth.py` | Declaration-domain keyword detection, path-hint expansion, source prioritisation |

### Intent enum

```
declaration_or_configuration   → retrieval_route: source_of_truth
dependency_or_build_inventory  → retrieval_route: source_of_truth
runtime_usage_or_reference     → retrieval_route: runtime_usage
symbol_lookup                  → retrieval_route: symbol_lookup
code_fix_or_patch              → retrieval_route: semantic
architecture_overview          → retrieval_route: semantic
generic_semantic               → retrieval_route: semantic
```

### How a query is classified (`routing.py`)

1. **Score phase** — each intent accumulates a score by counting how many of
   its vocabulary words appear in the tokenised query (`words` set).
2. **Explicit-phrase phase** — high-priority regex rules (with `\b` word
   boundaries) can override the score result.
3. **Tie-break** — `dep_score >= decl_score` favours dependency over
   declaration when both fire.

---

## Keyword Matching — Mandatory Rules

These rules exist to prevent false-positive intent routing.  They were
introduced after a series of bugs where general English words (e.g.
"declared", "good") accidentally triggered domain-specific retrieval paths.

### Rule 1 — Always use the word-tokenised set in `_score()`

`_score()` in `routing.py` MUST compare against the `words` set only:

```python
# CORRECT
return sum(1 for t in terms if t in words)

# WRONG — bare substring: "go" matches "good", "env" matches "environment"
return sum(1 for t in terms if t in words or t in q)
```

The `words` set is built with `re.findall(r"\w+", q)`, which already gives
exact word boundaries.

### Rule 2 — Use `\b` + prefix patterns, never bare substring `in q`

Whenever checking whether a query belongs to a domain (dependency, config,
manifest…), use a compiled regex with `\b` anchors.  For terms that have
inflected forms, use a prefix pattern:

```python
# CORRECT — prefix pattern catches all inflected forms, boundary prevents
# mid-word matches
re.search(r"\bdependenc\w+", query)   # → dependency, dependencies, …
re.search(r"\bconfig\w*",   query)   # → config, configured, configuration, …
re.search(r"\bcapabilit\w+", query)  # → capability, capabilities

# WRONG — bare substring
"dependenc" in query.lower()   # matches "independence"
"config"    in query.lower()   # matches "misconfigured"
```

The canonical definition lives in `_DECL_KW_RE` (source_of_truth.py).
**Do not duplicate keyword lists** — add new ecosystem terms there.

### Rule 3 — Never use a fragment as both start and end boundary

A pattern like `\bdependenc\b` is broken: word-boundary after the fragment
fails when inflectional suffixes follow ("ies", "y", "ent").

```python
# BROKEN — \b after fragment never fires in "dependencies"
re.search(r"\b(dependenc|capabilit)\b", q)

# CORRECT — no trailing boundary after a fragment
re.search(r"\b(dependenc\w+|capabilit\w+)", q)
```

### Rule 4 — "declared" is excluded from domain keyword lists

`"declared"` is a common English adjective.  It appears in queries that have
nothing to do with build files or manifests:

- "What variables are declared in this class?" → should be `symbol_lookup`
- "What dependencies are declared in this project?" → correctly classified by
  "dependenc\w+" alone, without needing "declared"

Adding "declared" to manifest/permission or declaration-domain keyword checks
causes those queries to be hijacked into the wrong retrieval path.  If a
future use-case genuinely needs "declared" as a signal, it must be combined
with at least one domain-specific term via a multi-token regex, not added to
a flat keyword list.

---

## Known Limitations (rule-based routing)

The following edge cases are accepted as known limitations.  They are
documented here so future contributors do not reintroduce "fixes" that break
the common cases.

| Query pattern | Risk | Reason accepted |
|---------------|------|-----------------|
| "go to the settings" | "go" scores dep_score | Extremely unlikely in a code assistant; "go" as a build tool is the dominant usage |
| "how does it manifest itself" | manifest triggers manifest path hints | "manifest" as a verb is rare in code queries |
| "librarian comments" | "librar\w*" catches it | A non-issue in practice; code codebases do not discuss librarians |
| "build on prior work" | "build\w*" scores dep_score | Typically co-occurs with runtime vocabulary which shifts intent back |

A proper fix for these edge cases requires an embedding-based or LLM-based
intent classifier.  The rule-based system should remain as a fast pre-filter
even if such a classifier is added.

---

## Files Changed (routing robustness fixes, Apr 2026)

| File | Change |
|------|--------|
| `andes_cache/routing.py` | `_score()`: dropped `or t in q` substring fallback |
| `andes_cache/source_of_truth.py` | Replaced `DECLARATION_QUERY_KEYWORDS` tuple + bare `any(k in q …)` with compiled `_DECL_KW_RE` regex; removed "declared" from keyword set; fixed `_intent_source_priority()` and `_candidate_match_factor()` to use word-tokenised sets instead of substring checks; removed "declared" from manifest path-hint expansion |
| `indexer.py` | Fixed broken `\bdependenc\b` / `\bcapabilit\b` patterns in workspace-summary routing (line ~1310) to use correct prefix patterns `dependenc\w+` / `capabilit\w+`; fixed `asks_permissions` check to remove "declared" |
