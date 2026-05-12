# Hybrid Retrieval Evaluation

AndesCode includes optional **graph-aware hybrid retrieval** behind `ANDESCODE_HYBRID_RETRIEVAL=1`. Hybrid retrieval augments baseline retrieval with local code-graph evidence such as exact symbol matches, filename matches, direct and reverse import neighbors, and capped reference neighbors.

This model-free A/B eval exists to answer a practical pilot question: does hybrid retrieval recover better context for hard multi-file codebase questions without introducing unacceptable noise or regressions?

## What this eval measures

The eval measures **retrieval quality**, not answer quality. It runs deterministic retrieval precision cases against golden fixtures and compares:

- **Baseline retrieval:** `ANDESCODE_HYBRID_RETRIEVAL=0`
- **Hybrid retrieval:** `ANDESCODE_HYBRID_RETRIEVAL=1`

Each side uses the same fixture, query, `n_results`, and expected files. The eval intentionally avoids requiring an LLM so results stay fast, repeatable, and suitable for CI or pilot evidence gathering.

## What this eval does not measure

- It does **not** grade generated answers.
- It does **not** prove that every added file is useful to the final model response.
- It does **not** change normal AndesCode runtime behavior.
- It does **not** make hybrid retrieval default; hybrid retrieval remains opt-in unless `ANDESCODE_HYBRID_RETRIEVAL=1` is set.

## Why hybrid retrieval can help

Baseline retrieval is strong for direct text matches, but multi-file engineering questions often require files that do not repeat the query terms. Examples include:

- A use case delegating to a repository, which then calls a DAO and API service.
- An auth token read from storage and injected later by an HTTP interceptor.
- Hardware/framework code reaching app logic through DI, observers, or callbacks.
- A symbol definition whose usage flow lives in imports or references.

Graph-aware hybrid retrieval can make those adjacent files visible while preserving deterministic, inspectable evidence about why each graph-selected file appeared.

## How to run the model-free A/B eval

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

You can also use the central eval runner:

```bash
python3 tests/eval/eval_runner.py --suite hybrid-ab --fixture android
```

The eval writes generated reports under `tests/eval/reports/` by default:

- `hybrid-retrieval-ab-<fixture>.json`
- `hybrid-retrieval-ab-<fixture>.md`

> **Index warning:** The eval indexes temporary golden fixtures using the normal AndesCode index location. Running it may replace your currently active indexed project. Re-index your real project after the eval before returning to normal use.

## How to read the report

The JSON and Markdown reports include one row per query/test id with:

- Fixture name and query/test id.
- Expected files.
- Baseline retrieval files.
- Hybrid retrieval files.
- Files added and removed by hybrid retrieval.
- Baseline and hybrid pass/fail.
- Classification: `improvement`, `regression`, or `unchanged`.
- Graph routes used when debug data is available.
- `graph_route_by_file`, `graph_score_by_file`, `graph_seed_files`, and graph neighbor/boost evidence.
- Context sufficiency notes for pilot interpretation.

The scoring summary is designed to be copied into a pilot report:

- **Baseline pass count:** how many cases passed with baseline retrieval.
- **Hybrid pass count:** how many cases passed with hybrid retrieval.
- **Net improvements:** failed baseline cases that pass with hybrid retrieval.
- **Net regressions:** passed baseline cases that fail with hybrid retrieval.
- **Unchanged cases:** both modes pass or both modes fail.
- **Noisy additions count:** hybrid-added files that do not match expected files. This is an approximate noise signal, not a human relevance judgment.

## Interpretation guidance

Use the report as engineering evidence, not as a single automatic launch gate.

- **Regressions matter more than raw improvements.** A few wins are not enough if hybrid retrieval removes expected files from already-passing baseline retrieval cases.
- **Noisy additions matter.** Extra files can consume context budget and distract answer generation; inspect whether added files have credible graph routes and scores.
- **Graph evidence matters.** `graph_route_by_file`, `graph_score_by_file`, and `graph_seed_files` explain why hybrid retrieval selected a file and help reviewers distinguish useful dependency expansion from broad neighbor noise.
- **Do not over-trust improvements.** Treat improvements cautiously when they rely mostly on low-confidence reference neighbors, when noisy additions are high, or when fixture coverage is too small.

Decision guidance:

1. **Keep hybrid retrieval opt-in** when improvements are rare, noisy additions are high, graph evidence is weak, or any regression affects important direct-lookup behavior.
2. **Enable hybrid retrieval for selected intents** when gains cluster around hard multi-file questions such as dependency tracing, auth/token flow, upload/data flow, hardware/framework integration, or symbol usage flow.
3. **Consider making hybrid retrieval default** only after multiple fixtures show higher hybrid pass counts, near-zero regressions, acceptable noisy additions, and explainable graph evidence for added files.

Normal runtime behavior remains unchanged by this eval. The only persistent outputs are the generated JSON and Markdown reports.
