import unittest

from local_retrieval import normalize_local_retrieval


class TestLocalRetrievalNormalization(unittest.TestCase):
    def test_normalizes_line_range_and_authority_defaults(self):
        result = normalize_local_retrieval(
            query="where is auth configured",
            chunks=[
                {
                    "file": "app/config.yaml",
                    "line": 0,
                    "content": "enabled: true\nissuer: andes",
                    "source_type": "config_file",
                }
            ],
            strategy="direct_retrieval",
            top_k=5,
            retrieval_mode="LOCAL",
            index_state={"status": "ready", "last_indexed_at": "2026-04-20T12:00:00+00:00"},
        )

        self.assertEqual(result.summary.strategy, "direct_retrieval")
        self.assertEqual(result.summary.top_k, 5)
        self.assertEqual(result.summary.retrieved_chunk_count, 1)
        self.assertEqual(result.summary.index_state, "ready")
        self.assertEqual(result.summary.total_candidate_files, 1)
        self.assertEqual(result.chunks[0].start_line, 1)
        self.assertEqual(result.chunks[0].end_line, 2)
        self.assertEqual(result.chunks[0].authority, "declared")
        self.assertEqual(result.chunks[0].authority_reason, "source_type_authoritative")

    def test_preserves_explicit_authority_metadata(self):
        result = normalize_local_retrieval(
            query="where is foo used",
            chunks=[
                {
                    "file": "server.py",
                    "line": 20,
                    "end_line": 22,
                    "score": 0.88,
                    "source_type": "source_code",
                    "authority": "referenced",
                    "authority_reason": "runtime usage path",
                    "content": "foo()",
                }
            ],
            strategy="planned_context",
            top_k=3,
            retrieval_mode="LOCAL",
            index_state={"status": "ready"},
        )

        prompt_chunks = result.to_prompt_chunks()
        self.assertEqual(prompt_chunks[0]["line"], 20)
        self.assertEqual(prompt_chunks[0]["authority"], "referenced")
        self.assertEqual(prompt_chunks[0]["authority_reason"], "runtime usage path")
        self.assertEqual(result.summary.strategy, "planned_context")
        self.assertEqual(result.summary.top_k, 3)
        self.assertEqual(result.summary.retrieval_mode, "LOCAL")

    def test_total_candidate_files_uses_unique_file_count(self):
        result = normalize_local_retrieval(
            query="where is foo used",
            chunks=[
                {"file": "server.py", "line": 5, "content": "a()", "source_type": "source_code"},
                {"file": "server.py", "line": 10, "content": "b()", "source_type": "source_code"},
                {"file": "indexer.py", "line": 7, "content": "c()", "source_type": "source_code"},
            ],
            strategy="direct_retrieval",
            top_k=10,
            retrieval_mode="LOCAL",
            index_state={"status": "ready"},
        )
        self.assertEqual(result.summary.top_k, 10)
        self.assertEqual(result.summary.total_candidate_files, 2)


if __name__ == "__main__":
    unittest.main()
