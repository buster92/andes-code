# Remote Inference Contract (v1)

AndesCode now includes typed request/response schemas for distributed inference payloads in `remote_inference_schema.py`.

## Request model

`RemoteInferenceRequest` includes:
- `client` metadata
- `workspace` metadata
- `query` metadata
- `retrieval` summary
- `chunks` (retrieved context snippets)
- `options` (stream/debug/max tokens)

## Response/event models

- `RemoteFinalAnswerEvent`
- `RemoteDebugEvent`
- `RemoteErrorEvent`

## Notes

- Protocol version is currently `andes.remote.v1`.
- Embeddings, patch operations, and full repository uploads are intentionally out of scope for v1.
- Validation is strict and explicit via schema checks to ensure payload compatibility between client and server.
