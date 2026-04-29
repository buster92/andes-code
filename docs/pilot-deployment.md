# AndesCode Pilot Deployment Guide

## 1) Overview

A **pilot deployment** is a limited rollout to validate AndesCode with a small group before broader adoption.

In a pilot, AndesCode can run in two ways:

- **Per developer (local install):** each developer runs AndesCode on their own machine.
- **Shared machine (team-hosted):** one machine hosts AndesCode and team members connect to it.

Key behavior to validate during pilot:

- In **LOCAL mode**, AndesCode performs inference locally and **no code leaves your environment for inference**.
- **Remote inference is optional** and should be enabled only when your team explicitly chooses it.

---

## 2) Deployment options

### Option A: Per-developer (recommended for pilots)

Each developer:

- installs and runs AndesCode locally
- selects/indexes their own repository copy
- uses local resources for indexing and inference

**Pros**

- no shared infrastructure required
- fastest path to first value
- strong privacy boundary per device

**Cons**

- duplicated indexing work across developers
- quality/performance depends on each developer’s hardware

---

### Option B: Shared team machine (advanced)

A single machine:

- hosts AndesCode server/runtime
- maintains model artifacts and index centrally
- serves multiple team users over network

**Pros**

- centralized model and index management
- consistent performance profile across users (based on host machine)

**Cons**

- requires network and endpoint security setup
- requires access control and host hardening
- not the default path for initial pilots

---

## 3) Requirements

Plan for the following baseline requirements:

- **OS:** macOS (Apple Silicon recommended)
- **Disk:** ~20–30 GB free (models, cache, and indexes)
- **Memory (RAM):**
  - minimum for basic pilot: 16 GB
  - recommended for smoother experience: 32 GB+
- **Runtime:** Python 3.12
- **Network:** internet access for first-run downloads only (models/dependencies), then can operate locally in LOCAL mode

---

## 4) Setup steps (per-developer)

Use this flow for each pilot participant.

1. **Clone the repository**
   - Example:
     - `git clone <your-andescode-repo-url>`
     - `cd andes-code`

2. **Run the launcher**
   - Start AndesCode using your team’s standard launch command/script.

3. **Wait for first-run downloads**
   - Initial startup may download model/runtime dependencies.
   - This can take several minutes depending on bandwidth.

4. **Select repository**
   - In the AndesCode UI/CLI, point the tool to the target code repository.

5. **Run indexing**
   - Start indexing and wait for completion.
   - For large repositories, this may take longer on first pass.

6. **Start asking questions**
   - Run validation prompts about architecture, modules, and ownership boundaries.
   - Confirm answer quality against known code areas.

---

## 5) Security posture (important)

For security review, validate and communicate the following behaviors:

- **LOCAL mode:** no outbound inference calls.
- **Sensitive file handling:** `.env` files are not indexed.
- **Logging:** logs are metadata-only (not full source payloads).
- **If remote inference is enabled, the following may be sent:**
  - user query text
  - retrieved context chunks
  - associated metadata

See `security-threat-model.md` for the authoritative threat and control model.

---

## 6) Remote inference (optional)

Use remote inference only when it solves a clear pilot constraint, such as:

- developer laptops lacking sufficient local compute
- need for centralized GPU-backed inference

When enabled, remote inference can send:

- query text
- retrieved chunks
- metadata

Recommended enterprise requirements before enabling:

- run inference on a **private server** you control
- restrict access via **VPN and/or internal network controls**
- apply authentication, authorization, and audit logging on the server

---

## 7) Validation checklist

Use this checklist during pilot sign-off:

- [ ] no unexpected outbound traffic in LOCAL mode
- [ ] index contains expected files only
- [ ] sensitive files are excluded
- [ ] logs contain no code or secrets
- [ ] answers match repository structure

---

## 8) Known limitations

Set expectations early:

- large repositories may require significant indexing time
- response quality and latency depend on local hardware class
- no formal security/compliance certifications are claimed yet
- remote inference trust is only as strong as the server and network controls

---

## 9) Next steps after pilot

If pilot outcomes are positive, proceed in phases:

1. expand rollout to more developers/teams
2. centralize model hosting where it improves cost/performance
3. tune indexing inclusion/exclusion rules for your codebase
4. compare developer productivity and answer quality vs cloud-native alternatives

