# Changelog

## [unreleased] — Pipeline-driven redesign (v2.1 spec)

> Forecast only — proposed changes, not yet implemented.
>
> spec_id: `70ab2170-381a-4657-bcd1-28a40c6f369b`
> Source spec: `pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json`
> Detailed roadmap: `docs/redesign/ROADMAP.md`

### Proposed (Phase 0 — bug fixes, no architecture change)

- **Bug-1 (embeddings):** add `probe_dimension` method to `Embedder` protocol
  so `RemoteEmbedder` no longer hardcodes `1536` for unknown models (e.g.
  `qwen3-embedding:8b` which is 4096-dim) and `OllamaEmbedder.dimension` no
  longer returns `0` until the first embed call. ChromaDB collection
  validates `expected_dim` against stored dimension on init. Add HTTP-status
  classification (404 model-not-pulled, 401/403 auth, 429 rate-limited) on
  `EmbedderUnavailableError` beyond just timeout/network errors.
- **Bug-2 (config error surfacing):** wrap `_get_service` (server.py:49-54)
  in `try: ... except (pydantic.ValidationError, FileNotFoundError,
  NotADirectoryError)` and return structured `{"error":"config_missing",
  "missing":"AGENT_KB_SAVES_DIR", ...}` payload instead of opaque MCP
  `InternalError`. Add `saves_dir` `default_factory=~/.agent-kb/saves` so
  fresh installs work without env vars. Add `main()` startup precheck so
  missing config fails immediately with structured stderr. NOTE: the v2.0
  spec draft's "decorator NO-OP" diagnosis was inverted — current order
  (`@mcp.tool()` outer, `@_with_tool_timeout` inner) works empirically
  (tests/test_tool_timeout.py 6/6 passing including
  `test_every_registered_mcp_tool_is_wrapped`). DO NOT swap.

### Proposed (Phase 1 — Pattern A launcher)

- Replace `.mcp.json` `command:uv,args:[run,...]` with single
  cross-platform stdlib launcher: `command:python,
  args:[${CLAUDE_PLUGIN_ROOT}/bin/run_server.py]`. The launcher
  auto-bootstraps `.venv` via stdlib `venv` module, uses a SHA256
  sentinel + filelock to avoid redundant pip installs, and emits
  structured stderr for the three failure modes (`python_version`,
  `network_unreachable`, `read_only_filesystem`).
  `AGENT_KB_VENDORED_DEPS` provides an offline escape-hatch for
  air-gapped installs.

### Proposed (Phase 2 — RetrieverBackend abstraction)

- New module `src/agent_knowledgebase/backends/__init__.py` exposes
  `RetrieverBackend` `Protocol` and `get_backend(settings)` factory.
  `ChromadbBackend` wraps existing chromadb logic with zero behavior
  change. New `Settings.kb_backend: Literal['chromadb','markdown',
  'lightrag'] = 'chromadb'` (loaded from new `AGENT_KB_BACKEND` env or
  `kb_backend` JSON config key) — disambiguated from the existing
  `Settings.vectorstore` which stays as the chromadb-internal
  vector-store-provider choice. KnowledgebaseService routes all
  retrieval through `self._backend`.

### Proposed (Phase 3 — MarkdownWikiBackend opt-in)

- New `backends/markdown_backend.py` — Karpathy-style markdown wiki under
  `<saves_dir>/<kb-name>/wiki/{index.md,hot.md,log.md,pages/<slug>.md}`
  with sqlite FTS5 in-memory index. NO embedding. Per-source_type
  **positive-allow** list (`{file, website, api_endpoint with payload
  <2MB}`); other source_types raise `KB_INGESTOR_UNSUITABLE` unless
  `AGENT_KB_FORCE_WIKI_INGEST=1`. Sharded `index.md` above ~200 articles.

### Proposed (Phase 4 — Migration tooling, BEFORE default-flip)

- New `services/migration.py` with `export_to_markdown` /
  `import_from_markdown` / `export_chromadb_dump` /
  `import_chromadb_dump`. New additive `kb_migrate(kb_id, target_backend)`
  MCP tool (probe-4-safe). Read-fallback: if `Settings.kb_backend='markdown'`
  but `<kb-name>/wiki/` is missing while `<kb-name>/vector_store/` exists,
  `ChromadbBackend` serves the query (silent fallback, structured stderr
  emitted). Schema migration: `ALTER TABLE pages ADD COLUMN
  embedding_provider TEXT`, `embed_base_url TEXT`. Backfill existing pages
  from kb config.json. `create_embedder_for_model` now accepts
  `(model_name, provider, base_url)` from the per-page snapshot. The
  `migration_complete` sentinel file marks cutover.

### Proposed (Phase 5 — Embedding defaults flip)

- `embedding_provider` default flips `'ollama' → 'remote'`;
  `embedding_model` default → `'text-embedding-3-small'`. New `'fastembed'`
  provider (opt-in via `extras['embed-local-onnx']`, ~150MB).
  sentence-transformers moves to opt-in `extras['embed-local-st']`
  (+1.3GB). New `[project.optional-dependencies]` groups:
  `embed-local-onnx`, `embed-local-st`, `ingest-codebase` (tree-sitter,
  gitpython), `ingest-file` (pdfplumber), `ingest-sql` (sqlalchemy,
  sqlparse). `embedder_version` stamped per chunk; ingest rejects mixed
  versions on the same `kb_id` unless `AGENT_KB_AUTO_REEMBED=1`. MMR
  lambda re-tuned for fastembed int8 quantization delta. Realistic
  default install-size floor: **~700MB** (down from ~2GB; chromadb at
  ~400MB and tree-sitter-languages at ~100MB stay in default install).
  The earlier "~20MB" target was off by ~35x and has been corrected.
- **Gate:** Phase 5 PR cannot merge until Phase 4 read-fallback +
  per-page provider snapshot are in production.

### Proposed (Phase 6 — LightRAGBackend, deferred)

- Stub `backends/lightrag_backend.py` raising `NotImplementedError`.
  Design doc `docs/lightrag_integration_design.md`. Implement only if
  user adoption signals demand.

### Frozen across the redesign

- The 25 `kb_*` MCP tools (names, parameters, response shapes).
- Probe-4 contract (`source_type`, `uri`, `dedup_key`, `page_id`,
  `dominant_embedding_model`); backends return `null` for inapplicable
  fields, NOT omit them.
- `knowledgebase_stderr_log` 11-field schema.
- SQL-injection AST validator + read-only sqlite + 50-row hard-reject.
- `AGENT_KB_SAVES_DIR` semantics; cross-process-lock recipe;
  `pipeline_runs` telemetry columns.

## 0.7.0 — 2026-04-21

### Changed (breaking)
- Remote embedding provider is now a plain `httpx`-based HTTP client
  talking directly to `{base_url}/embeddings` — there is no longer any
  third-party SDK dependency. Any server speaking the common embeddings
  JSON contract (Ollama's `/v1`, vLLM, LocalAI, hosted endpoints) works.
- Renames (breaking config changes):
  - `embedding_provider` literal `"openai"` → `"remote"`.
  - `openai_api_key` → `embed_api_key`; env var
    `AGENT_KB_OPENAI_API_KEY` → `AGENT_KB_EMBED_API_KEY`.
  - `openai_base_url` → `embed_base_url`; env var
    `AGENT_KB_EMBED_BASE_URL` is now **required** for the `remote`
    provider (no implicit default endpoint).
  - Python class `OpenAIEmbedder` → `RemoteEmbedder`.
- The optional `openai` install extra has been removed from
  `pyproject.toml`. Existing callers should drop the `[openai]` extra
  from their install line; no replacement extra is required because
  `httpx` is already a core dependency.

### Preserved
- `EmbedderUnavailableError` and the FIELD-14 structured-error payload
  shape (`{error, model, phase, latency_ms, detail}`) are unchanged.
  `embed_timeout` / `embed_unreachable` tokens, `embed_timeout_seconds`,
  and `embed_max_retries` semantics are preserved across the migration.

## 0.6.0 — 2026-04-20

### Added
- `knowledgebase_stderr_log(...)` helper in `services/stderr_log.py` — a
  stdlib-only, single-line JSON stderr emitter used by the ingestion
  pipeline for structured lifecycle telemetry. Required fields: `kb_id`,
  `op`, `phase`, `elapsed_ms`, `rows_in`, `rows_ok`, `rows_skipped`,
  `rows_failed`, `dedup_policy`, `request_id`, `tool_caller_version`.
- 9 new columns on `pipeline_runs` (all additive, nullable): `ended_at`
  (TEXT), `ingested` (INTEGER), `skipped` (INTEGER), `replaced`
  (INTEGER), `failed` (INTEGER), `batch_size` (INTEGER), `dedup_policy`
  (TEXT), `request_id` (TEXT), `tool_caller_version` (TEXT). Migration
  is idempotent — running `_init_schema` twice is a no-op.
- `error_code` on structured stderr emissions follows
  SCREAMING_SNAKE_CASE on failure (e.g. `LOCK_ACQUIRE_FAILED`,
  `INGEST_SOURCE_FAILED`).
- `request_id` and `tool_caller_version` telemetry are captured on
  every PipelineRun row. `run_ingest_batch` synthesizes a uuid4 hex
  `request_id` when the caller omits one, so the row is always
  populated.
- `kb_ingest_service.run_ingest_batch(...)` — new batch entry point
  used by the MCP tool layer. Emits `ingest_batch_start` /
  `ingest_source` (per item) / `ingest_batch_end` structured stderr
  lines around the loop.

### Changed
- All six lock-diagnostic lines (previously bare
  `sys.stderr.write("[kb-lock] ...")` calls in
  `KnowledgebaseService.ingest_source` and `update_source`) now route
  through `knowledgebase_stderr_log` with `phase="lock"`,
  `dedup_policy="n/a"`, zero counters, and wall-clock
  milliseconds-held on `lock_release`.
- Per-source ingest outcomes route through structured stderr JSON
  instead of ad-hoc prints. One line per source with `rows_in=1` and
  exactly one of `rows_ok` / `rows_skipped` / `rows_failed` set to 1.

### Preserved
- `kb_pipeline_status` MCP tool signature stays exactly
  `def kb_pipeline_status(kb_id: str) -> str:`. The returned JSON
  simply picks up the 9 new fields from the updated `PipelineRun`
  pydantic model.
- `kb_ingest` and `kb_ingest_batch` MCP tool signatures are unchanged
  — `request_id` / `tool_caller_version` are threaded through at the
  service layer, not surfaced on the stable MCP contracts.

### Downgrade note (R11 mitigation)

Downgrading a v0.6.0 SQLite DB to 0.5.x / 0.2.x post-ingest requires
dropping the 9 additive columns. SQLite ≥ 3.35 added `DROP COLUMN IF
EXISTS`; earlier versions require a table rebuild, which is out of
scope for this note.

```sql
-- Requires SQLite >= 3.35.
ALTER TABLE pipeline_runs DROP COLUMN IF EXISTS ended_at;
ALTER TABLE pipeline_runs DROP COLUMN IF EXISTS ingested;
ALTER TABLE pipeline_runs DROP COLUMN IF EXISTS skipped;
ALTER TABLE pipeline_runs DROP COLUMN IF EXISTS replaced;
ALTER TABLE pipeline_runs DROP COLUMN IF EXISTS failed;
ALTER TABLE pipeline_runs DROP COLUMN IF EXISTS batch_size;
ALTER TABLE pipeline_runs DROP COLUMN IF EXISTS dedup_policy;
ALTER TABLE pipeline_runs DROP COLUMN IF EXISTS request_id;
ALTER TABLE pipeline_runs DROP COLUMN IF EXISTS tool_caller_version;
```

No data is copied or transformed — running v0.5.x after this
rollback is safe because the earlier code paths do not read the
dropped columns. The v0.6.0 code will re-add them on next open.

## 0.2.0 — 2026-04-12

### Added
- User-level (`~/.agent-kb/config.json`) and project-level (`./.agent-kb/config.json`) JSON config files, layered under environment variables.
- Six new config knobs: `chunk.token_encoding`, `query.default_top_k`, `query.hybrid.vector_weight`, `query.hybrid.fts_weight`, `query.hybrid.fetch_multiplier`, `ingest.excluded_dirs`.
- Five new MCP tools: `kb_config_path`, `kb_config_show`, `kb_config_get`, `kb_config_set`, `kb_config_validate`.
- Hybrid-weights-sum-to-1.0 validator on `Settings`.
- `ingest.excluded_dirs` list is consolidated into `Settings` as the single source of truth (previously duplicated in `DirectoryIngestor`).

### Changed
- `Settings` now layers defaults → user JSON → project JSON → env (highest priority).
- `token_chunk` and `DirectoryIngestor` now read their behavior from `Settings` instead of hardcoded module-level constants.
- `KnowledgebaseService.query/search/hybrid_query` and MCP `kb_query/kb_search` tools default `top_k` to `Settings.query_default_top_k` (via `None` sentinel) instead of the hardcoded `10`.

### Preserved
- `AGENT_KB_SAVES_DIR` remains required and env-only.
- Secrets (`AGENT_KB_EMBED_API_KEY`, `AGENT_KB_PINECONE_API_KEY`) remain env-only. JSON files reject them.

## 0.1.0 — 2026-04-11

Initial release.
