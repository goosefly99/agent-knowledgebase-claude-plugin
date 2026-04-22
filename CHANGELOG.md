# Changelog

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
