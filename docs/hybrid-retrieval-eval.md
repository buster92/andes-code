# Hybrid Retrieval Evaluation

AndesCode includes optional graph-aware hybrid retrieval behind `ANDESCODE_HYBRID_RETRIEVAL=1`. The feature augments semantic retrieval with code-graph evidence such as exact symbol matches, filename matches, direct and reverse import neighbors, and capped reference neighbors. The evaluation in this document measures whether that graph evidence helps hard, multi-file codebase understanding before any new product surface is introduced.

## Why hybrid retrieval exists

Plain semantic retrieval is strong for direct text matches, but hard codebase questions often require adjacent files that do not repeat the query terms. Examples include:

- A use case that delegates to a repository, which then calls a DAO and API service.
- An auth token read from storage and injected later by an HTTP interceptor.
- Hardware/framework code that reaches app logic through DI, observers, or callbacks.
- A symbol definition that is easy to retrieve but whose usage flow lives in imports or references.

Hybrid retrieval stays model-free and opt-in. It keeps normal AndesCode behavior unchanged unless `ANDESCODE_HYBRID_RETRIEVAL=1` is set or the A/B eval script explicitly toggles it for measurement.

## How to run the A/B eval

Run the Android pilot fixture:

```bash
python3 tests/eval/hybrid_retrieval_ab_eval.py --fixture android
```

Run the Python API or Rust CLI fixtures:

```bash
python3 tests/eval/hybrid_retrieval_ab_eval.py --fixture python_api
python3 tests/eval/hybrid_retrieval_ab_eval.py --fixture rust_cli
```

Run all registered fixtures:

```bash
python3 tests/eval/hybrid_retrieval_ab_eval.py --fixture all
```

You can also use the eval runner entry point:

```bash
python3 tests/eval/eval_runner.py --suite hybrid-ab --fixture android
```

The eval writes reports to `tests/eval/reports/` by default:

- `hybrid-retrieval-ab-<fixture>.json`
- `hybrid-retrieval-ab-<fixture>.md`

The eval does not require a loaded LLM. It writes a fresh golden fixture, indexes it, runs the same retrieval cases with `ANDESCODE_HYBRID_RETRIEVAL=0`, then runs the same cases with `ANDESCODE_HYBRID_RETRIEVAL=1`. Each case uses the same query, `n_results`, fixture, and expected files in both modes.

> **Index warning:** The eval indexes temporary golden fixtures using the normal AndesCode index location. Running it may replace your currently active indexed project. Re-index your real project after the eval before returning to normal use.

## How to read the report

The JSON and Markdown reports include one row per query/test id with:

- Fixture name and query/test id.
- Expected files.
- Baseline retrieved files.
- Hybrid retrieved files.
- Files added and removed by hybrid.
- Baseline and hybrid pass/fail.
- Classification: `improvement`, `regression`, or `unchanged`.
- Graph routes used when debug data is available.
- `graph_route_by_file`, `graph_seed_files`, and graph neighbor/boost evidence.
- Context sufficiency notes for pilot interpretation.

The scoring summary is designed to be copied into a pilot report:

- **Baseline pass count**: how many cases passed without graph-aware retrieval.
- **Hybrid pass count**: how many cases passed with graph-aware retrieval.
- **Net improvements**: failed baseline cases that pass with hybrid retrieval.
- **Net regressions**: passed baseline cases that fail with hybrid retrieval.
- **Unchanged cases**: both modes pass or both modes fail.
- **Noisy additions count**: hybrid-added files that do not match expected files. This is an approximate noise signal, not a human relevance judgment.

## Decision guidance

Use the report as pilot evidence rather than a single automated gate:

1. **Keep hybrid opt-in** when improvements are rare, noisy additions are high, or graph debug shows weak routes (for example mostly broad reference neighbors).
2. **Enable hybrid for selected intents** when gains cluster around hard multi-file questions such as dependency tracing, auth/token flow, upload/data flow, or symbol usage flow, while direct lookup remains unchanged.
3. **Consider making hybrid default** only after multiple fixtures show higher hybrid pass counts, near-zero regressions, acceptable noisy additions, and understandable graph route evidence for added files.

Normal runtime behavior remains unchanged by this eval. The only persistent outputs are the generated JSON and Markdown reports.
