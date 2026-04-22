# Remote Inference Contract (v1)

AndesCode now includes typed request/response schemas for distributed inference payloads in `remote_inference_schema.py`.
In LOCAL mode, retrieval is now normalized into this same shape internally before prompt packing so local and remote inference paths share the same retrieval result structure.
Some LOCAL summary fields (for example `total_candidate_files`) may be approximate when only local retrieved chunks are available.
The remote inference server endpoint is `POST /v1/ask` and must build answer context from the payload only.

## Request model (`RemoteInferenceRequest`)

### `client` (`ClientMetadata`)
| Field | Type | Required |
|---|---|---|
| `client_version` | `str` | Yes |
| `protocol_version` | `str` (must equal `andes.remote.v1`) | Yes |
| `platform` | `str` | Yes |
| `hostname` | `str` | No |

### `workspace` (`WorkspaceMetadata`)
| Field | Type | Required |
|---|---|---|
| `workspace_id` | `str` | Yes |
| `repo_name` | `str` | Yes |
| `repo_root_name` | `str` | Yes |
| `branch` | `str` | Yes |
| `commit_hash` | `str` | Yes |
| `is_dirty` | `bool` | Yes |

### `query` (`QueryMetadata`)
| Field | Type | Required |
|---|---|---|
| `request_id` | `str` | Yes |
| `text` | `str` | Yes |
| `requested_at` | ISO-8601 datetime | Yes |
| `question_type` | `str` | No |

### `retrieval` (`RetrievalSummary`)
| Field | Type | Required |
|---|---|---|
| `strategy` | `str` | Yes |
| `top_k` | `int` (>= 1) | Yes |
| `indexed_at` | ISO-8601 datetime | Yes |
| `index_state` | `str` | Yes |
| `total_candidate_files` | `int` (>= 0) | Yes |
| `retrieved_chunk_count` | `int` (>= 0) | Yes |
| `retrieval_mode` | `str` | No |

### `chunks` (`list[RetrievedChunk]`, non-empty)
| Field | Type | Required |
|---|---|---|
| `chunk_id` | `str` | Yes |
| `path` | `str` | Yes |
| `language` | `str` | No |
| `start_line` | `int` (>= 1) | Yes |
| `end_line` | `int` (>= `start_line`) | Yes |
| `score` | `float` | Yes |
| `source_type` | `str` | Yes |
| `authority` | `str` | Yes |
| `authority_reason` | `str` | Yes |
| `content` | `str` | Yes |

### `options` (`RemoteInferenceOptions`)
| Field | Type | Required |
|---|---|---|
| `stream` | `bool` | No (default `true`) |
| `debug` | `bool` | No (default `false`) |
| `max_answer_tokens` | `int` (>= 1) | No |

## Response/event models

### `RemoteFinalAnswerEvent`
| Field | Type | Required |
|---|---|---|
| `event` | literal `\"final_answer\"` | Yes |
| `request_id` | `str` | Yes |
| `answer` | `str` | Yes |
| `finished_at` | ISO-8601 datetime | Yes |

### `RemoteDebugEvent`
| Field | Type | Required |
|---|---|---|
| `event` | literal `\"debug\"` | Yes |
| `request_id` | `str` | Yes |
| `payload` | `dict` | Yes |

### `RemoteErrorEvent`
| Field | Type | Required |
|---|---|---|
| `event` | literal `\"error\"` | Yes |
| `request_id` | `str` | Yes |
| `code` | `str` | Yes |
| `message` | `str` | Yes |

## Notes

- Protocol version is currently `andes.remote.v1`.
- Embeddings, patch operations, and full repository uploads are intentionally out of scope for v1.
- Validation is strict and explicit via schema checks to ensure payload compatibility between client and server.
