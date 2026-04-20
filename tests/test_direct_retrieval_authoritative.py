import importlib
import sys
import types
import unittest


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
