import importlib
import sys
import types
import unittest
from types import SimpleNamespace


class _FakeEmbedding:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


class _FakeSentenceTransformer:
    def __init__(self, *_args, **_kwargs):
        pass

    def encode(self, _query):
        return _FakeEmbedding([0.1, 0.2, 0.3])


class _FakeCollection:
    def count(self):
        return 3

    def query(self, query_embeddings, n_results):  # noqa: ARG002
        return {
            "documents": [["runtime usage", "helper context"]],
            "metadatas": [[
                {"file": "src/runtime_usage.kt", "language": "kt", "symbols": "runtimeCall"},
                {"file": "src/another.kt", "language": "kt", "symbols": "helper"},
            ]],
            "distances": [[0.01, 0.05]],
        }

    def get(self, where=None, limit=None):  # noqa: ARG002
        return {"documents": []}


class _FakeChromaClient:
    def __init__(self, *_args, **_kwargs):
        pass

    def get_or_create_collection(self, *_args, **_kwargs):
        return _FakeCollection()


def _import_indexer_with_stubs():
    fake_sentence_transformers = types.ModuleType("sentence_transformers")
    fake_sentence_transformers.SentenceTransformer = _FakeSentenceTransformer
    sys.modules["sentence_transformers"] = fake_sentence_transformers

    fake_chromadb = types.ModuleType("chromadb")
    fake_chromadb.PersistentClient = _FakeChromaClient
    sys.modules["chromadb"] = fake_chromadb

    import indexer
    return importlib.reload(indexer)


class TestDirectRetrievalAuthoritative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.indexer = _import_indexer_with_stubs()

    def test_decl_query_keeps_authoritative_chunk_when_semantic_top_is_runtime(self):
        indexer = self.indexer
        indexer._load_workspace_index = lambda: {
            "manifests": [],
            "config_graph": {"config_files": ["app/buildSrc/dependencies.kt"]},
        }
        indexer.classify_query_intent_details = lambda _q: {
            "intent": "runtime_usage_or_reference",
            "retrieval_route": "semantic",
            "ambiguous": False,
            "allow_runtime_fallback": False,
            "strict_authority_mode": True,
        }
        indexer._structured_query_results = lambda _q: []
        indexer._load_json = lambda _p: {}
        indexer.get_repo_fingerprint = lambda: ""
        indexer._fetch_exact_file = lambda path, max_results=3: (
            [{"file": path, "content": "implementation(\"a:b:1.0\")", "language": "kt", "symbols": "dep"}]
            if path == "app/buildSrc/dependencies.kt"
            else []
        )
        indexer._add_coverage = lambda chunks: chunks
        indexer._rerank = lambda _q, candidates, track_reasons=False: sorted(  # noqa: ARG005
            candidates,
            key=lambda c: (c.get("file") != "src/runtime_usage.kt", c.get("score", 0.0)),
        )

        results = indexer.search("what dependencies are declared", n_results=1, debug_mode=False)
        self.assertTrue(results)
        self.assertTrue(
            any(
                c.get("file") == "app/buildSrc/dependencies.kt"
                or c.get("source_type") in {"dependency_file", "build_file", "manifest", "config_file"}
                for c in results
            )
        )

    def test_fetch_exact_file_recovers_suffix_path_mismatch_for_authoritative_file(self):
        indexer = _import_indexer_with_stubs()

        class _PathMismatchCollection:
            def count(self):
                return 2

            def get(self, where=None, limit=None):  # noqa: ARG002
                if where == {"file": "buildSrc/dependencies.kt"}:
                    return {"documents": [], "metadatas": []}
                return {
                    "documents": ["implementation(\"a:b:1.0\")"],
                    "metadatas": [{
                        "file": "app/buildSrc/dependencies.kt",
                        "language": "kt",
                        "line": 7,
                        "symbols": "dep",
                    }],
                }

        original_col = indexer.col
        indexer.col = _PathMismatchCollection()
        try:
            chunks = indexer._fetch_exact_file("buildSrc/dependencies.kt", max_results=5)
        finally:
            indexer.col = original_col

        self.assertTrue(chunks)
        self.assertEqual(chunks[0]["file"], "app/buildSrc/dependencies.kt")

    def test_authoritative_file_missing_from_index_returns_explicit_limitation(self):
        indexer = _import_indexer_with_stubs()
        indexer._load_workspace_index = lambda: {
            "manifests": [],
            "config_graph": {"config_files": ["project/deps.toml"]},
        }
        indexer.classify_query_intent_details = lambda _q: {
            "intent": "dependency_or_build_inventory",
            "retrieval_route": "semantic",
            "ambiguous": False,
            "allow_runtime_fallback": False,
            "strict_authority_mode": True,
        }
        indexer._structured_query_results = lambda _q: []
        indexer._load_json = lambda _p: {}
        indexer.get_repo_fingerprint = lambda: ""
        indexer._fetch_exact_file = lambda *_args, **_kwargs: []
        indexer._fetch_indexed_candidates_by_basename = lambda *_args, **_kwargs: {}

        results, payload = indexer.search(
            "what dependencies are declared",
            n_results=2,
            debug_mode=True,
            return_debug=True,
        )
        self.assertTrue(results)
        self.assertEqual(results[0].get("file"), "__source_of_truth_missing__")
        self.assertIn("Authoritative file detected but not retrievable from index", results[0].get("content", ""))
        self.assertEqual(payload["retrieval"]["authority_retrieval_mode"], "workspace_only_detected_not_indexed")
        self.assertEqual(payload["retrieval"]["declaration_answer_mode"], "missing_declarations")

    def test_authoritative_multichunk_file_keeps_all_chunks(self):
        indexer = _import_indexer_with_stubs()
        indexer._load_workspace_index = lambda: {
            "manifests": [],
            "config_graph": {"config_files": ["a/package.json"]},
        }
        indexer.classify_query_intent_details = lambda _q: {
            "intent": "dependency_or_build_inventory",
            "retrieval_route": "semantic",
            "ambiguous": False,
            "allow_runtime_fallback": False,
            "strict_authority_mode": True,
        }
        indexer._structured_query_results = lambda _q: []
        indexer._load_json = lambda _p: {}
        indexer.get_repo_fingerprint = lambda: ""
        indexer._fetch_exact_file = lambda path, max_results=6: [
            {"file": path, "content": "dep_a", "line": 1, "language": "", "symbols": ""},
            {"file": path, "content": "dep_b", "line": 20, "language": "", "symbols": ""},
            {"file": path, "content": "dep_c", "line": 40, "language": "", "symbols": ""},
            {"file": path, "content": "dep_d", "line": 90, "language": "", "symbols": ""},
        ]
        indexer._fetch_indexed_candidates_by_basename = lambda *_args, **_kwargs: {}
        indexer._add_coverage = lambda chunks: chunks
        indexer._rerank = lambda _q, candidates, track_reasons=False: sorted(candidates, key=lambda c: c.get("file", ""))

        results = indexer.search("what dependencies are declared", n_results=1, debug_mode=False)
        dep_chunks = [r for r in results if r.get("file") == "a/package.json"]
        self.assertEqual(len(dep_chunks), 4)
        self.assertEqual([c.get("content") for c in dep_chunks], ["dep_a", "dep_b", "dep_c", "dep_d"])

    def test_semantic_duplicates_do_not_readd_authoritative_chunks(self):
        indexer = _import_indexer_with_stubs()
        indexer._load_workspace_index = lambda: {
            "manifests": [],
            "config_graph": {"config_files": ["a/package.json"]},
        }
        indexer.classify_query_intent_details = lambda _q: {
            "intent": "dependency_or_build_inventory",
            "retrieval_route": "semantic",
            "ambiguous": False,
            "allow_runtime_fallback": False,
            "strict_authority_mode": True,
        }
        indexer._structured_query_results = lambda _q: []
        indexer._load_json = lambda _p: {}
        indexer.get_repo_fingerprint = lambda: ""
        indexer._fetch_exact_file = lambda path, max_results=6: [
            {"file": path, "content": "dep_a", "line": 1, "language": "", "symbols": ""},
            {"file": path, "content": "dep_b", "line": 2, "language": "", "symbols": ""},
        ]
        indexer._fetch_indexed_candidates_by_basename = lambda *_args, **_kwargs: {}
        indexer._add_coverage = lambda chunks: chunks
        indexer._rerank = lambda _q, candidates, track_reasons=False: [
            {"file": "a/package.json", "content": "dep_a", "line": 1, "score": 0.01},
            {"file": "src/runtime_usage.kt", "content": "runtime", "line": 5, "score": 0.02},
        ]

        results = indexer.search("what dependencies are declared", n_results=3, debug_mode=False)
        dep_a_count = sum(1 for r in results if r.get("file") == "a/package.json" and r.get("content") == "dep_a")
        self.assertEqual(dep_a_count, 1)
        self.assertTrue(any(r.get("file") == "a/package.json" and r.get("content") == "dep_b" for r in results))

    def test_partial_authority_debug_mode_is_declared_partial_only(self):
        indexer = _import_indexer_with_stubs()
        indexer._load_workspace_index = lambda: {
            "manifests": [],
            "config_graph": {"config_files": ["a/package.json", "b/pyproject.toml"]},
        }
        indexer.classify_query_intent_details = lambda _q: {
            "intent": "dependency_or_build_inventory",
            "retrieval_route": "semantic",
            "ambiguous": False,
            "allow_runtime_fallback": False,
            "strict_authority_mode": True,
        }
        indexer._structured_query_results = lambda _q: []
        indexer._load_json = lambda _p: {}
        indexer.get_repo_fingerprint = lambda: ""
        indexer._fetch_exact_file = lambda path, max_results=6: (
            [{"file": path, "content": "dep_only", "line": 1, "language": "", "symbols": ""}]
            if path == "a/package.json"
            else []
        )
        indexer._fetch_indexed_candidates_by_basename = lambda *_args, **_kwargs: {}
        indexer._add_coverage = lambda chunks: chunks
        indexer._rerank = lambda _q, candidates, track_reasons=False: candidates

        _results, payload = indexer.search("what dependencies are declared", n_results=1, debug_mode=True, return_debug=True)
        self.assertEqual(payload["retrieval"]["authoritative_files_retrieved"], ["a/package.json"])
        self.assertEqual(payload["retrieval"]["authoritative_files_missing"], ["b/pyproject.toml"])
        self.assertEqual(payload["retrieval"]["declaration_answer_mode"], "declared_partial_only")

    def test_cache_hit_preserves_multichunk_authoritative_shape(self):
        indexer = _import_indexer_with_stubs()
        indexer._load_workspace_index = lambda: {
            "manifests": [],
            "config_graph": {"config_files": ["a/package.json"]},
        }
        indexer.classify_query_intent_details = lambda _q: {
            "intent": "dependency_or_build_inventory",
            "retrieval_route": "semantic",
            "ambiguous": False,
            "allow_runtime_fallback": False,
            "strict_authority_mode": True,
        }
        indexer._structured_query_results = lambda _q: []
        indexer._load_json = lambda _p: {}
        indexer.get_repo_fingerprint = lambda: "repo-fp"
        indexer._fetch_exact_file = lambda path, max_results=6: [
            {"file": path, "content": "dep_a", "line": 1, "language": "", "symbols": ""},
            {"file": path, "content": "dep_b", "line": 2, "language": "", "symbols": ""},
            {"file": path, "content": "dep_c", "line": 3, "language": "", "symbols": ""},
        ]
        indexer._fetch_indexed_candidates_by_basename = lambda *_args, **_kwargs: {}
        indexer._add_coverage = lambda chunks: chunks
        indexer._rerank = lambda _q, candidates, track_reasons=False: candidates

        cache_store = {}

        def _cache_get(repo_fp, query, index_version, intent, retrieval_route):  # noqa: ARG001
            return cache_store.get((repo_fp, query, intent, retrieval_route))

        def _cache_set(repo_fp, query, index_version, value, intent, retrieval_route):  # noqa: ARG001
            cache_store[(repo_fp, query, intent, retrieval_route)] = list(value)

        indexer.CACHE = SimpleNamespace(retrieval_get=_cache_get, retrieval_set=_cache_set)

        first_results, first_payload = indexer.search(
            "what dependencies are declared",
            n_results=1,
            debug_mode=True,
            return_debug=True,
        )
        second_results, second_payload = indexer.search(
            "what dependencies are declared",
            n_results=1,
            debug_mode=True,
            return_debug=True,
        )

        self.assertEqual(first_results, second_results)
        self.assertEqual([c.get("content") for c in first_results], ["dep_a", "dep_b", "dep_c"])
        self.assertEqual(first_payload["retrieval"]["declaration_answer_mode"], second_payload["retrieval"]["declaration_answer_mode"])
        self.assertTrue(second_payload["retrieval"]["cache_hit"])
        self.assertEqual(len(second_payload["retrieval"]["selected_candidates"]), 3)
        self.assertEqual(second_payload["retrieval"]["authority_retrieval_mode"], "direct_chunk_load")

    def test_cache_hit_non_authoritative_respects_smaller_n_results(self):
        indexer = _import_indexer_with_stubs()
        indexer._load_workspace_index = lambda: {"manifests": [], "config_graph": {"config_files": []}}
        indexer.classify_query_intent_details = lambda _q: {
            "intent": "runtime_usage_or_reference",
            "retrieval_route": "semantic",
            "ambiguous": False,
            "allow_runtime_fallback": False,
            "strict_authority_mode": True,
        }
        indexer._structured_query_results = lambda _q: []
        indexer._load_json = lambda _p: {}
        indexer.get_repo_fingerprint = lambda: "repo-fp-runtime"
        indexer._add_coverage = lambda chunks: chunks
        indexer._rerank = lambda _q, candidates, track_reasons=False: candidates

        cache_store = {}

        def _cache_get(repo_fp, query, index_version, intent, retrieval_route):  # noqa: ARG001
            return cache_store.get((repo_fp, query, intent, retrieval_route))

        def _cache_set(repo_fp, query, index_version, value, intent, retrieval_route):  # noqa: ARG001
            cache_store[(repo_fp, query, intent, retrieval_route)] = list(value)

        indexer.CACHE = SimpleNamespace(retrieval_get=_cache_get, retrieval_set=_cache_set)

        first_results, first_payload = indexer.search(
            "what libraries are used",
            n_results=2,
            debug_mode=True,
            return_debug=True,
        )
        second_results, second_payload = indexer.search(
            "what libraries are used",
            n_results=1,
            debug_mode=True,
            return_debug=True,
        )

        self.assertEqual(len(first_results), 2)
        self.assertEqual(len(second_results), 1)
        self.assertTrue(second_payload["retrieval"]["cache_hit"])
        self.assertEqual(len(second_payload["retrieval"]["selected_candidates"]), 1)
        self.assertEqual(second_payload["retrieval"]["authority_retrieval_mode"], "runtime_fallback_used")
        self.assertIn(
            second_payload["retrieval"]["declaration_answer_mode"],
            {"", "declared_only", "declared_plus_runtime", "declared_partial_only", "runtime_only_fallback", "missing_declarations"},
        )

    def test_debug_payload_includes_authoritative_fields(self):
        indexer = _import_indexer_with_stubs()
        indexer._load_workspace_index = lambda: {
            "manifests": [],
            "config_graph": {"config_files": ["a/package.json"]},
        }
        indexer.classify_query_intent_details = lambda _q: {
            "intent": "dependency_or_build_inventory",
            "retrieval_route": "semantic",
            "ambiguous": False,
            "allow_runtime_fallback": False,
            "strict_authority_mode": True,
        }
        indexer._structured_query_results = lambda _q: []
        indexer._load_json = lambda _p: {}
        indexer.get_repo_fingerprint = lambda: ""
        indexer._fetch_exact_file = lambda path, max_results=6: [{"file": path, "content": "deps", "language": "", "symbols": ""}]
        indexer._fetch_indexed_candidates_by_basename = lambda *_args, **_kwargs: {}
        indexer._add_coverage = lambda chunks: chunks
        indexer._rerank = lambda _q, candidates, track_reasons=False: candidates

        _results, payload = indexer.search("what dependencies are declared", n_results=1, debug_mode=True, return_debug=True)
        retrieval = payload["retrieval"]
        for key in (
            "authoritative_files_detected",
            "authoritative_files_required",
            "authoritative_files_retrieved",
            "authoritative_files_missing",
            "forced_authoritative_file",
            "authority_selection_reason",
            "authority_retrieval_mode",
            "declaration_answer_mode",
            "declaration_query_trigger_reason",
        ):
            self.assertIn(key, retrieval)
        self.assertEqual(retrieval["declaration_query_trigger_reason"], "intent")

    def test_keyword_fallback_marks_declaration_trigger_reason(self):
        indexer = _import_indexer_with_stubs()
        indexer._load_workspace_index = lambda: {
            "manifests": [],
            "config_graph": {"config_files": ["a/package.json"]},
        }
        indexer.classify_query_intent_details = lambda _q: {
            "intent": "runtime_usage_or_reference",
            "retrieval_route": "runtime_usage",
            "ambiguous": False,
            "allow_runtime_fallback": True,
            "strict_authority_mode": False,
        }
        indexer._structured_query_results = lambda _q: []
        indexer._load_json = lambda _p: {}
        indexer.get_repo_fingerprint = lambda: ""
        indexer._fetch_exact_file = lambda path, max_results=6: [{"file": path, "content": "deps", "language": "", "symbols": ""}]
        indexer._fetch_indexed_candidates_by_basename = lambda *_args, **_kwargs: {}
        indexer._add_coverage = lambda chunks: chunks
        indexer._rerank = lambda _q, candidates, track_reasons=False: candidates

        _results, payload = indexer.search("what libraries does this project use", n_results=1, debug_mode=True, return_debug=True)
        retrieval = payload["retrieval"]
        self.assertEqual(retrieval["declaration_query_trigger_reason"], "keyword_fallback")
        self.assertTrue(retrieval["authoritative_files_required"])
