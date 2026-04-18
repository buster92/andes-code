# Testing Caches

## Unit + integration-style cache tests

```bash
python -m unittest tests/test_cache_layers.py -v
python -m unittest tests/test_retrieval_routing_behavior.py -v
```

Covers:
- cache key determinism
- store behavior + schema invalidation
- repo fingerprint changes
- no cross-repo reuse
- deterministic prompt serialization
- workspace + retrieval cache behavior
- partial invalidation
- patch-plan reuse
- semantic safety constraints
- benchmark/metrics output smoke checks

## Existing suite

Run the original suite (requires running server/model):

```bash
python test_andescode.py
```

## Interpreting benchmark checks

The benchmark assertions are lightweight and correctness-focused:
- cold vs warm calls execute and are measurable
- metrics file is emitted
- per-layer hits/misses are tracked
