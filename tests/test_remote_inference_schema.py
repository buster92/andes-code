import unittest
from datetime import datetime, timezone

from remote_inference_schema import (
    RemoteDebugEvent,
    RemoteErrorEvent,
    RemoteFinalAnswerEvent,
    RemoteInferenceRequest,
    SchemaValidationError,
)


def _valid_request_payload():
    return {
        "client": {
            "client_version": "1.0.0",
            "protocol_version": "andes.remote.v1",
            "platform": "darwin-arm64",
            "hostname": "devbox",
        },
        "workspace": {
            "workspace_id": "ws-123",
            "repo_name": "andes-code",
            "repo_root_name": "andes-code",
            "branch": "main",
            "commit_hash": "abc123",
            "is_dirty": False,
        },
        "query": {
            "request_id": "req-1",
            "text": "Where is indexing triggered?",
            "requested_at": datetime.now(timezone.utc).isoformat(),
        },
        "retrieval": {
            "strategy": "semantic_then_symbols",
            "top_k": 5,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "index_state": "ready",
            "total_candidate_files": 42,
            "retrieved_chunk_count": 2,
            "retrieval_mode": "hybrid",
        },
        "chunks": [
            {
                "chunk_id": "c1",
                "path": "server.py",
                "language": "python",
                "start_line": 10,
                "end_line": 20,
                "score": 0.91,
                "source_type": "code",
                "authority": "runtime",
                "authority_reason": "direct call chain",
                "content": "def chat(...): ...",
            }
        ],
        "options": {
            "stream": True,
            "debug": False,
            "max_answer_tokens": 512,
        },
    }


class TestRemoteInferenceSchema(unittest.TestCase):
    def test_valid_request_payload(self):
        model = RemoteInferenceRequest.from_dict(_valid_request_payload())
        dumped = model.to_dict()
        self.assertEqual(dumped["workspace"]["repo_name"], "andes-code")
        self.assertEqual(len(dumped["chunks"]), 1)

    def test_rejects_missing_required_field(self):
        payload = _valid_request_payload()
        del payload["workspace"]["repo_name"]
        with self.assertRaises(SchemaValidationError):
            RemoteInferenceRequest.from_dict(payload)

    def test_rejects_invalid_chunk_line_range(self):
        payload = _valid_request_payload()
        payload["chunks"][0]["start_line"] = 50
        payload["chunks"][0]["end_line"] = 20
        with self.assertRaises(SchemaValidationError):
            RemoteInferenceRequest.from_dict(payload)

    def test_response_event_schemas(self):
        now = datetime.now(timezone.utc)
        final_event = RemoteFinalAnswerEvent.from_dict(
            {
                "request_id": "req-1",
                "answer": "Done.",
                "finished_at": now.isoformat(),
            }
        )
        debug_event = RemoteDebugEvent.from_dict(
            {
                "request_id": "req-1",
                "payload": {"files": ["server.py"]},
            }
        )
        error_event = RemoteErrorEvent.from_dict(
            {
                "request_id": "req-1",
                "code": "bad_request",
                "message": "payload invalid",
            }
        )

        self.assertEqual(final_event.event, "final_answer")
        self.assertEqual(debug_event.event, "debug")
        self.assertEqual(error_event.event, "error")
        self.assertEqual(
            set(final_event.to_dict().keys()),
            {"event", "request_id", "answer", "finished_at"},
        )
        self.assertEqual(
            set(debug_event.to_dict().keys()),
            {"event", "request_id", "payload"},
        )
        self.assertEqual(
            set(error_event.to_dict().keys()),
            {"event", "request_id", "code", "message"},
        )
        self.assertEqual(final_event.to_dict()["finished_at"], now.isoformat())

    def test_response_event_schema_rejects_invalid_payloads(self):
        with self.assertRaises(SchemaValidationError):
            RemoteFinalAnswerEvent.from_dict(
                {"request_id": "", "answer": "ok", "finished_at": datetime.now(timezone.utc).isoformat()}
            )
        with self.assertRaises(SchemaValidationError):
            RemoteFinalAnswerEvent.from_dict(
                {"request_id": "req-1", "answer": "ok", "finished_at": "not-a-date"}
            )
        with self.assertRaises(SchemaValidationError):
            RemoteDebugEvent.from_dict({"request_id": "req-1", "payload": "not-a-dict"})
        with self.assertRaises(SchemaValidationError):
            RemoteErrorEvent.from_dict({"request_id": "req-1", "code": "", "message": "m"})
        with self.assertRaises(SchemaValidationError):
            RemoteErrorEvent.from_dict({"request_id": "req-1", "code": "bad_request", "message": ""})


if __name__ == "__main__":
    unittest.main()
