from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class RemoteProtocol(str, Enum):
    V1 = "andes.remote.v1"


class SchemaValidationError(ValueError):
    pass


def _require_str(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise SchemaValidationError(f"{field} is required and must be a non-empty string")
    return value


def _require_bool(payload: dict[str, Any], field: str) -> bool:
    value = payload.get(field)
    if not isinstance(value, bool):
        raise SchemaValidationError(f"{field} is required and must be a boolean")
    return value


def _require_int(payload: dict[str, Any], field: str, *, minimum: int | None = None) -> int:
    value = payload.get(field)
    if not isinstance(value, int):
        raise SchemaValidationError(f"{field} is required and must be an integer")
    if minimum is not None and value < minimum:
        raise SchemaValidationError(f"{field} must be >= {minimum}")
    return value


def _require_float(payload: dict[str, Any], field: str) -> float:
    value = payload.get(field)
    if not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{field} is required and must be numeric")
    return float(value)


def _parse_datetime(value: Any, field: str) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SchemaValidationError(f"{field} must be a valid ISO-8601 datetime") from exc
    raise SchemaValidationError(f"{field} is required and must be an ISO-8601 datetime string")


@dataclass
class ClientMetadata:
    client_version: str
    protocol_version: str
    platform: str
    hostname: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ClientMetadata":
        client_version = _require_str(payload, "client_version")
        protocol_version = _require_str(payload, "protocol_version")
        if protocol_version != RemoteProtocol.V1.value:
            raise SchemaValidationError("protocol_version must be andes.remote.v1")
        platform = _require_str(payload, "platform")
        hostname = payload.get("hostname")
        if hostname is not None and not isinstance(hostname, str):
            raise SchemaValidationError("hostname must be a string when provided")
        return cls(client_version=client_version, protocol_version=protocol_version, platform=platform, hostname=hostname)


@dataclass
class WorkspaceMetadata:
    workspace_id: str
    repo_name: str
    repo_root_name: str
    branch: str
    commit_hash: str
    is_dirty: bool

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorkspaceMetadata":
        return cls(
            workspace_id=_require_str(payload, "workspace_id"),
            repo_name=_require_str(payload, "repo_name"),
            repo_root_name=_require_str(payload, "repo_root_name"),
            branch=_require_str(payload, "branch"),
            commit_hash=_require_str(payload, "commit_hash"),
            is_dirty=_require_bool(payload, "is_dirty"),
        )


@dataclass
class QueryMetadata:
    request_id: str
    text: str
    requested_at: datetime
    question_type: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "QueryMetadata":
        question_type = payload.get("question_type")
        if question_type is not None and not isinstance(question_type, str):
            raise SchemaValidationError("question_type must be a string when provided")
        return cls(
            request_id=_require_str(payload, "request_id"),
            text=_require_str(payload, "text"),
            requested_at=_parse_datetime(payload.get("requested_at"), "requested_at"),
            question_type=question_type,
        )


@dataclass
class RetrievalSummary:
    strategy: str
    top_k: int
    indexed_at: datetime
    index_state: str
    total_candidate_files: int
    retrieved_chunk_count: int
    retrieval_mode: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RetrievalSummary":
        retrieval_mode = payload.get("retrieval_mode")
        if retrieval_mode is not None and not isinstance(retrieval_mode, str):
            raise SchemaValidationError("retrieval_mode must be a string when provided")
        return cls(
            strategy=_require_str(payload, "strategy"),
            top_k=_require_int(payload, "top_k", minimum=1),
            indexed_at=_parse_datetime(payload.get("indexed_at"), "indexed_at"),
            index_state=_require_str(payload, "index_state"),
            total_candidate_files=_require_int(payload, "total_candidate_files", minimum=0),
            retrieved_chunk_count=_require_int(payload, "retrieved_chunk_count", minimum=0),
            retrieval_mode=retrieval_mode,
        )


@dataclass
class RetrievedChunk:
    chunk_id: str
    path: str
    language: str | None
    start_line: int
    end_line: int
    score: float
    source_type: str
    authority: str
    authority_reason: str
    content: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RetrievedChunk":
        start_line = _require_int(payload, "start_line", minimum=1)
        end_line = _require_int(payload, "end_line", minimum=1)
        if end_line < start_line:
            raise SchemaValidationError("end_line must be greater than or equal to start_line")
        language = payload.get("language")
        if language is not None and not isinstance(language, str):
            raise SchemaValidationError("language must be a string when provided")
        return cls(
            chunk_id=_require_str(payload, "chunk_id"),
            path=_require_str(payload, "path"),
            language=language,
            start_line=start_line,
            end_line=end_line,
            score=_require_float(payload, "score"),
            source_type=_require_str(payload, "source_type"),
            authority=_require_str(payload, "authority"),
            authority_reason=_require_str(payload, "authority_reason"),
            content=_require_str(payload, "content"),
        )


@dataclass
class RemoteInferenceOptions:
    stream: bool = True
    debug: bool = False
    max_answer_tokens: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "RemoteInferenceOptions":
        if payload is None:
            return cls()
        stream = payload.get("stream", True)
        debug = payload.get("debug", False)
        if not isinstance(stream, bool):
            raise SchemaValidationError("options.stream must be a boolean")
        if not isinstance(debug, bool):
            raise SchemaValidationError("options.debug must be a boolean")
        max_answer_tokens = payload.get("max_answer_tokens")
        if max_answer_tokens is not None:
            if not isinstance(max_answer_tokens, int) or max_answer_tokens < 1:
                raise SchemaValidationError("options.max_answer_tokens must be an integer >= 1")
        return cls(stream=stream, debug=debug, max_answer_tokens=max_answer_tokens)


@dataclass
class RemoteInferenceRequest:
    client: ClientMetadata
    workspace: WorkspaceMetadata
    query: QueryMetadata
    retrieval: RetrievalSummary
    chunks: list[RetrievedChunk]
    options: RemoteInferenceOptions

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RemoteInferenceRequest":
        chunks_payload = payload.get("chunks")
        if not isinstance(chunks_payload, list) or not chunks_payload:
            raise SchemaValidationError("chunks is required and must be a non-empty list")
        return cls(
            client=ClientMetadata.from_dict(payload.get("client") or {}),
            workspace=WorkspaceMetadata.from_dict(payload.get("workspace") or {}),
            query=QueryMetadata.from_dict(payload.get("query") or {}),
            retrieval=RetrievalSummary.from_dict(payload.get("retrieval") or {}),
            chunks=[RetrievedChunk.from_dict(chunk) for chunk in chunks_payload],
            options=RemoteInferenceOptions.from_dict(payload.get("options")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RemoteFinalAnswerEvent:
    request_id: str
    answer: str
    finished_at: datetime
    event: str = "final_answer"

    def __post_init__(self):
        self.request_id = _require_str({"request_id": self.request_id}, "request_id")
        if not isinstance(self.answer, str):
            raise SchemaValidationError("answer is required and must be a string")
        if not isinstance(self.finished_at, datetime):
            raise SchemaValidationError("finished_at is required and must be a datetime")
        if self.event != "final_answer":
            raise SchemaValidationError("event must be final_answer")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RemoteFinalAnswerEvent":
        return cls(
            request_id=_require_str(payload, "request_id"),
            answer=payload.get("answer"),
            finished_at=_parse_datetime(payload.get("finished_at"), "finished_at"),
            event=payload.get("event", "final_answer"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "request_id": self.request_id,
            "answer": self.answer,
            "finished_at": self.finished_at.isoformat(),
        }


@dataclass
class RemoteDebugEvent:
    request_id: str
    payload: dict[str, Any]
    event: str = "debug"

    def __post_init__(self):
        self.request_id = _require_str({"request_id": self.request_id}, "request_id")
        if not isinstance(self.payload, dict):
            raise SchemaValidationError("payload is required and must be an object")
        if self.event != "debug":
            raise SchemaValidationError("event must be debug")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RemoteDebugEvent":
        return cls(
            request_id=_require_str(payload, "request_id"),
            payload=payload.get("payload"),
            event=payload.get("event", "debug"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "request_id": self.request_id,
            "payload": self.payload,
        }


@dataclass
class RemoteErrorEvent:
    request_id: str
    code: str
    message: str
    event: str = "error"

    def __post_init__(self):
        self.request_id = _require_str({"request_id": self.request_id}, "request_id")
        self.code = _require_str({"code": self.code}, "code")
        self.message = _require_str({"message": self.message}, "message")
        if self.event != "error":
            raise SchemaValidationError("event must be error")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RemoteErrorEvent":
        return cls(
            request_id=_require_str(payload, "request_id"),
            code=_require_str(payload, "code"),
            message=_require_str(payload, "message"),
            event=payload.get("event", "error"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "request_id": self.request_id,
            "code": self.code,
            "message": self.message,
        }
