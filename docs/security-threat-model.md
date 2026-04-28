# AndesCode Security Threat Model

## Purpose and scope

This document describes AndesCode's security posture, trust boundaries, and primary risks for enterprise evaluation.

It focuses on:

- Execution modes and data flow
- Data inventory and storage surfaces
- Network behavior and outbound dependencies
- Logging and sensitive data handling defaults
- Major risks and recommended controls

## 1) Execution modes

### LOCAL mode (default)

In `LOCAL` mode, AndesCode is designed to keep repository processing and inference on the same machine:

- Repository data remains local
- Indexing and retrieval remain local
- Prompts are built locally
- Inference runs locally

This is the highest-privacy operating mode for most enterprise use cases.

### REMOTE_INFERENCE mode (opt-in)

In `REMOTE_INFERENCE` mode, AndesCode keeps core data prep local and sends only the data needed for remote generation:

- Repository and index stay local
- AndesCode sends the user query text (prompt), selected retrieved chunks, and related metadata (and may include client identifiers such as hostname) to a user-configured private remote inference server
- Remote inference is only enabled when explicitly configured

### Hosted service statement

AndesCode does **not** provide an official AndesCode-hosted SaaS inference service.

## 2) Data inventory

AndesCode may process and/or persist the following data classes during normal operation:

- **Source files**: repository contents chosen by the user
- **Chunks**: segmented text derived from indexed files
- **Embeddings**: vector representations used for retrieval
- **Project map**: structural metadata about project files/relationships
- **Symbol index**: extracted code symbol metadata
- **Workspace index**: retrieval/search index artifacts for the workspace
- **Query text (user prompt)**: user-provided input used to generate responses; may be transmitted in `REMOTE_INFERENCE` mode
- **Audit logs**: operational metadata logs
- **Model files**: local model binaries/weights used for inference
- **Runtime cache**: cached retrieval/inference-related intermediates

## 3) Network behavior

AndesCode is designed for local-first operation, with limited network dependencies:

- First-run model download requires network access
- First-run embedding model download requires network access
- Indexing runs locally
- `LOCAL` mode answering runs locally
- `REMOTE_INFERENCE` mode outbound payload can include query text (prompt), retrieved context, and related metadata
- No third-party API is used for inference by default

## 4) Sensitive file policy

Default indexing safeguards are intended to reduce accidental sensitive ingestion:

- Real `.env` files are skipped by default
- Environment examples/templates (for example, `.env.example`) are indexable
- Binary, oversized, and generated files are skipped

Even with these defaults, organizations should review indexed-file scope and policy before regulated or high-assurance deployments.

## 5) Logs

AndesCode logging is metadata-oriented by design:

- Query text should not be logged
- Response text should not be logged
- Code content should not be logged
- Secret values should not be logged

Default log locations:

- `~/Documents/AndesCode/server.log`
- `~/Documents/AndesCode/app.log`

## 6) Main risks and mitigations

| Risk | Threat scenario | Primary mitigations |
|---|---|---|
| Accidental secret indexing | Sensitive files are unintentionally included in indexable scope | Keep default skip rules enabled, review indexing policy, add repository-specific deny rules, validate indexed corpus before production use |
| Remote inference misconfiguration | Prompt/query text and retrieved context are sent to an unintended or weakly protected remote endpoint | Disable `REMOTE_INFERENCE` unless needed, require explicit endpoint allowlisting, enforce private networking/VPN and TLS, verify remote server ownership |
| Model/download supply-chain risk | Downloaded model/embedding artifacts are tampered with or untrusted | Pin approved artifact sources, verify checksums/signatures where possible, mirror vetted artifacts internally |
| Dependency installation risk | Compromised package/dependency introduces malicious code | Use dependency pinning/lockfiles, private package proxies, software composition scanning, and controlled build pipelines |
| Local machine compromise | Endpoint malware or unauthorized local access exposes data, indexes, or runtime memory | Harden endpoint OS, use EDR/AV, full-disk encryption, least-privilege accounts, patch management, strict physical/device controls |
| Stale index/cache answers | Outdated index or cache yields incorrect answers that impact decisions | Enforce re-index cadence and freshness checks, provide operational runbooks, clear cache/index during critical updates |
| Audit log metadata exposure | Operational metadata in logs leaks workload patterns or project identifiers | Restrict log file permissions, centralize logs in secure SIEM, define retention limits, redact sensitive metadata fields |

## 7) Non-claims

For clarity in enterprise evaluations:

- AndesCode is not currently claiming SOC 2, ISO 27001, or other formal certification
- Customers are responsible for validating their deployment environment for applicable compliance requirements
- In `REMOTE_INFERENCE` mode, privacy and security properties depend on the user-operated remote inference server and its controls

## 8) Recommended enterprise controls

For enterprise pilots and production hardening, AndesCode recommends:

1. Run in `LOCAL` mode for highest privacy by default
2. Disable `REMOTE_INFERENCE` unless a clear business need exists
3. If using remote inference, place traffic on private network/VPN and enforce TLS
4. Review and customize indexing policy before onboarding sensitive repositories
5. Pin model artifacts and verify checksums/signatures where feasible
6. Restrict file permissions for runtime data and log directories
7. Perform full validation (security, legal, compliance, and operational) before pilot/rollout

---

If you are preparing an enterprise review package, pair this document with:

- `docs/indexing-policy.md`
- `docs/remote-inference-contract.md`
