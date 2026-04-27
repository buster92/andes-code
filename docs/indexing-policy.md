# Indexing Policy

This document describes what AndesCode indexes by default and what it intentionally skips for safety and performance.

## Supported file types

AndesCode indexes source/config/documentation text files across common extensions, including:

- Source code: `.py`, `.js`, `.ts`, `.jsx`, `.tsx`, `.go`, `.rs`, `.java`, `.cpp`, `.c`, `.h`, `.rb`, `.php`, `.cs`, `.swift`, `.kt`
- Build/config/data text: `.gradle`, `.xml`, `.toml`, `.properties`, `.yaml`, `.yml`, `.sql`, `.r`
- Documentation/scripts/web text: `.md`, `.mdx`, `.txt`, `.sh`, `.bash`, `.html`, `.css`
- Notebooks: `.ipynb` (cell sources are extracted and chunked as text)
- Manifest/build authority files by canonical basename (e.g., `requirements.txt`, `pyproject.toml`, `package.json`, `Cargo.toml`, `go.mod`, `pom.xml`, `Dockerfile`)

## Files skipped by default

AndesCode skips files/directories that are unlikely to be useful for semantic retrieval or are risky/noisy:

- Generated/vendor/cache/build directories (for example: `node_modules`, `dist`, `build`, `target`, `.venv`, `__pycache__`, `.pytest_cache`, etc.)
- Binary-looking files (null bytes or high non-text byte ratio)
- Oversized generic text files above the indexing size guard (`MAX_FILE_BYTES`)

## Dotenv safety defaults

### Real dotenv files are skipped

To reduce accidental indexing of secrets, AndesCode skips these by default:

- `.env`
- `.env.local`
- `.env.development`
- `.env.production`
- `.env.test`
- `.env.staging`

### Dotenv examples/templates are allowed

AndesCode only indexes explicit non-secret dotenv templates/examples so it can reason about config shape:

- `.env.example`
- `.env.sample`
- `.env.template`
- `example.env`
- `sample.env`
- `template.env`

All other dotenv/env files are skipped by default.
