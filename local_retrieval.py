from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


_AUTHORITATIVE_SOURCE_TYPES = {"manifest", "build_file", "dependency_file", "config_file"}


@dataclass
class LocalRetrievedChunk:
    chunk_id: str
    path: str
    start_line: int
    end_line: int
    language: str | None
    score: float
    source_type: str
    authority: str
    authority_reason: str
    content: str

    def to_prompt_chunk(self) -> dict[str, Any]:
        return {
            "file": self.path,
            "line": self.start_line,
            "language": self.language or "",
            "score": self.score,
            "source_type": self.source_type,
            "authority": self.authority,
            "authority_reason": self.authority_reason,
            "content": self.content,
        }


@dataclass
class LocalRetrievalSummary:
    strategy: str
    top_k: int
    retrieved_chunk_count: int
    total_candidate_files: int
    index_state: str
    retrieval_mode: str
    indexed_at: str | None = None


@dataclass
class LocalRetrievalResult:
    query: str
    chunks: list[LocalRetrievedChunk]
    summary: LocalRetrievalSummary
    metadata: dict[str, Any]

    def to_prompt_chunks(self) -> list[dict[str, Any]]:
        return [chunk.to_prompt_chunk() for chunk in self.chunks]

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "summary": asdict(self.summary),
            "chunks": [asdict(chunk) for chunk in self.chunks],
            "metadata": self.metadata,
        }


def _derive_authority(source_type: str) -> tuple[str, str]:
    if source_type in _AUTHORITATIVE_SOURCE_TYPES:
        return "declared", "source_type_authoritative"
    if source_type == "inferred":
        return "inferred", "source_type_inferred"
    return "referenced", "source_type_runtime"


def _normalize_line_range(chunk: dict[str, Any]) -> tuple[int, int]:
    raw_start = chunk.get("start_line", chunk.get("line", 1))
    try:
        start = int(raw_start)
    except (TypeError, ValueError):
        start = 1
    if start < 1:
        start = 1

    raw_end = chunk.get("end_line")
    if raw_end is not None:
        try:
            end = int(raw_end)
        except (TypeError, ValueError):
            end = start
        if end < start:
            end = start
        return start, end

    content = str(chunk.get("content", "") or "")
    line_span = max(content.count("\n") + 1, 1)
    return start, start + line_span - 1


def normalize_local_retrieval(
    *,
    query: str,
    chunks: list[dict[str, Any]] | None,
    strategy: str,
    top_k: int,
    retrieval_mode: str,
    index_state: dict[str, Any] | None = None,
) -> LocalRetrievalResult:
    normalized_chunks: list[LocalRetrievedChunk] = []
    for idx, chunk in enumerate(chunks or []):
        source_type = str(chunk.get("source_type") or "source_code")
        default_authority, default_reason = _derive_authority(source_type)
        authority = str(chunk.get("authority") or default_authority)
        authority_reason = str(chunk.get("authority_reason") or default_reason)
        start_line, end_line = _normalize_line_range(chunk)

        normalized_chunks.append(
            LocalRetrievedChunk(
                chunk_id=str(chunk.get("chunk_id") or f"local-{idx}"),
                path=str(chunk.get("file") or chunk.get("path") or ""),
                start_line=start_line,
                end_line=end_line,
                language=(str(chunk.get("language")) if chunk.get("language") is not None else None),
                score=float(chunk.get("score") or 0.0),
                source_type=source_type,
                authority=authority,
                authority_reason=authority_reason,
                content=str(chunk.get("content") or ""),
            )
        )

    unique_files = {c.path for c in normalized_chunks if c.path}
    indexed_at = None
    if index_state:
        indexed_at = index_state.get("last_indexed_at")
    if not indexed_at:
        indexed_at = datetime.now(timezone.utc).isoformat()

    # approximate: derived from unique files in retrieved chunks
    total_candidate_files = len(unique_files)

    summary = LocalRetrievalSummary(
        strategy=strategy,
        top_k=top_k,
        retrieved_chunk_count=len(normalized_chunks),
        total_candidate_files=total_candidate_files,
        index_state=(index_state or {}).get("status", "ready"),
        retrieval_mode=retrieval_mode,
        indexed_at=indexed_at,
    )
    metadata = {}
    return LocalRetrievalResult(query=query, chunks=normalized_chunks, summary=summary, metadata=metadata)
