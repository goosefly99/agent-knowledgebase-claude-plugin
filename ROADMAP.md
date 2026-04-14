# agent-knowledgebase — Update Roadmap
Generated from run `data-etl-orchestrator-update-2026-04-13`

Source spec: `pipeline_mcp_data/specs/agent-knowledgebase-update-spec.json`
Codebase root: `C:\Users\olive\claude_projects\coding\agent_tools_dev\data_etl_orchestrator_dev\agent_knowledge_base_plugin_dev`
Language: Python 3.11+, uv, pydantic v2, chromadb, sqlalchemy, pytest.

## Release position
**Stage 1 of 4 — ships FIRST (security gate).**
- Gates: the SQL-injection AST validator for `kb_ingest source_type=sql_database` (where=) is the single highest-severity item across the entire release (validation-report kb-04). No orchestrator skill may suggest user-controlled where-clauses until this lands.
- Gates downstream: youtube-mcp, x-api-mcp, data-etl-orchestrator plugin.
- Gated by: nothing upstream. This is the security foundation.

## Blocking changes (must ship in this release)
- [x] **Replace where-clause validator with sqlparse AST allow-list.** Accept only: comparison operators (`=`, `!=`, `<`, `<=`, `>`, `>=`, `IN`, `LIKE`), logical (`AND`, `OR`, `NOT`), identifiers, literals, `IS NULL`/`IS NOT NULL`. Reject: comments (`--`, `/* */`), subqueries, multi-statements (`;`), `PRAGMA`, `ATTACH`, `DETACH`, function calls, `UNION`, backticks. Source: debate round-3 critic + validation kb-04 (high). Acceptance: `tests/test_sql_database_ingest.py::test_where_ast_validator_*` — one negative test per rejected category, passes on CI.
- [ ] **Pin sqlite read-only in the engine URL.** Append `?mode=ro` (URI-mode) or `immutable=1` when constructing the sqlalchemy engine; document Windows absolute-path quoting in `docs/sql-database-uri-grammar.md`. Source: debate round-3 critic. Acceptance: integration test asserts `engine.dialect.readonly` or that an `INSERT` on the source DB raises `OperationalError: attempt to write a readonly database`.
- [ ] **Hard-reject >50-row batches at release 1** (not soft-warn). Structured error with code `BATCH_SIZE_EXCEEDED` and remediation text. Source: debate round-3 synthesizer overturned the spec's soft-warn plan. Acceptance: `kb_ingest_batch` with 51 rows of `source_type=sql_database` returns a structured error and zero rows written.
- [ ] **Expose `dedup_key` and `dedup_policy` as first-class metadata fields** on `kb_ingest` / `kb_ingest_batch`; default `dedup_policy='skip'`. Source: spec objectives + debate round-3 advocate. Acceptance: three tests — skip, replace, force-add — each asserting the correct page count against a fixture DB.
- [ ] **Augment `kb_list_pages` and `kb_list_sources` response** with `source_type`, `uri`, `dedup_key`, `page_id` (pages) and `source_type`, `uri`, `source_id` (sources). Strictly additive — no renames. Source: spec + validation kb-03 (uncertain until audited). Acceptance: response JSON contains the new fields; existing fields unchanged; backward-compat unit test passes.
- [ ] **Per-`kb_id` serialization** via `asyncio.Lock` (or equivalent); explicitly document this is single-process-only. Source: debate round-3 critic. Acceptance: test fires two overlapping `kb_ingest_batch` calls against same `kb_id`; asserts serial execution; stderr log confirms lock acquisition/release.
- [ ] **Record embedding model per page** in metadata; surface dominant model in `kb_info` response. Source: debate round-3 synthesizer (embedding-mix risk). Acceptance: `kb_info` returns `dominant_embedding_model` + `embedding_model_counts`; `sentence_transformers_page` metadata contains `embedding_model` on every ingested row.

## Recommended changes (ship if feasible)
- [ ] Public-facing `docs/safe-where-clause-grammar.md` with the allow-list formally stated and 10 example filters users can copy.
- [ ] Cross-process lock recipe documented (e.g. Postgres advisory lock or a filesystem `flock`) for future multi-worker deployments — document only in release 1; don't implement.
- [ ] Structured `kb_pipeline_status` telemetry row per ingest run: `{kb_id, started_at, ended_at, ingested, skipped, replaced, failed, batch_size, dedup_policy}`.

## Accepted as-is
- `kb_create` / `kb_list` / `kb_info` / `kb_delete` / `kb_update_source` / `kb_remove_source` tool signatures unchanged.
- Authentication / permissions model unchanged (no new auth surface in this spec).
- No new external runtime dependencies (sqlparse is already transitively available via sqlalchemy; confirm in audit).
- 50-row cap value (validated by context7 chromadb docs — kb-02).

## Phased work breakdown

### Phase 1 — Audit
- files to touch: (none; read-only audit)
- files to read: `src/agent_knowledgebase/ingestors/sql_database.py`, `src/agent_knowledgebase/models.py`, `src/agent_knowledgebase/server.py` (handlers), `src/agent_knowledgebase/services/*.py`
- deliverable: audit note listing — (a) current uri-grammar parsing (if any), (b) current kb_list_pages/kb_list_sources response shape, (c) whether a dispatcher already exists, (d) whether sqlparse is already a dep.
- verification: `uv run python -c "import sqlparse"` succeeds; audit note committed in the PR description.

### Phase 2 — Security hardening (SQL-injection AST validator + read-only engine)
- files to touch: `src/agent_knowledgebase/ingestors/sql_database_ingestor.py` (create or update), `src/agent_knowledgebase/services/where_clause_validator.py` (new)
- tests to add: `tests/test_where_clause_validator.py` with one negative test per rejected AST category; `tests/test_sql_database_ingest.py::test_engine_is_readonly`
- verification: `uv run pytest tests/test_where_clause_validator.py tests/test_sql_database_ingest.py -v`; `uv run ruff check`; `uv run mypy src/agent_knowledgebase` (if configured)

### Phase 3 — Ingest service guardrails + dedup
- files to touch: `src/agent_knowledgebase/services/kb_ingest_service.py`, `src/agent_knowledgebase/services/dedup_service.py` (new), the MCP tool handlers for `kb_ingest` and `kb_ingest_batch`
- tests to add: `tests/test_kb_ingest_service.py` covering — (a) 51-row rejection, (b) per-kb_id serialization via overlapping async calls, (c) each of skip/replace/force-add dedup policies
- verification: `uv run pytest tests/test_kb_ingest_service.py -v`

### Phase 4 — Response augmentation + embedding-model tracking
- files to touch: `src/agent_knowledgebase/server.py` (or wherever kb_list_pages / kb_list_sources / kb_info are implemented), pydantic response schemas
- tests to add: `tests/test_kb_list_augmentation.py` asserting new fields present and old fields unchanged; `tests/test_kb_info_embedding_model.py`
- verification: `uv run pytest tests/test_kb_list_augmentation.py tests/test_kb_info_embedding_model.py -v`

### Phase 5 — Docs + cross-references
- files to touch: `docs/memory-pointer-protocol.md` (new), `docs/sql-database-uri-grammar.md` (new), `README.md` (cross-reference data-etl-orchestrator `skills/references/mcp-tool-contracts.md`)
- verification: docs exist; `README.md` cross-reference link present; no test gate.

### Phase 6 — End-to-end smoke
- verification:
  - `uv run pytest` (full suite passes)
  - `uv run ruff check`
  - Smoke: `kb_create` a scratch KB; point at a fixture sqlite DB; run `kb_ingest_batch(source_type='sql_database', dedup_policy='skip')` twice and confirm second run reports all `skipped`; run with `replace` and confirm rows replaced; run with 51 rows and confirm hard-reject; run `kb_info` and confirm `dominant_embedding_model` populated.

## Cross-plugin dependencies
- **This plugin's changes gate:**
  - `data-etl-orchestrator` — its `load-kb-from-sql` skill and every ingest skill's Stage-3 depend on the uri grammar + dedup_key + dedup_policy being live.
  - `youtube-mcp` and `x-api-mcp` (indirectly) — orchestrator calls their cache DBs via kb's sql_database ingestor.
- **This plugin is gated by:** nothing upstream. Ship first.

## Verification commands
```bash
# Type-check + lint
uv run ruff check
uv run mypy src/agent_knowledgebase

# Test suite
uv run pytest -v

# Targeted security tests
uv run pytest tests/test_where_clause_validator.py -v

# Smoke (manual)
uv run python -m agent_knowledgebase  # start MCP server
# then exercise kb_create / kb_ingest_batch / kb_info via MCP client
```

## Out of scope
- No new MCP tools. No renames or removals of existing tools or fields.
- No new vector-store backend (chromadb stays).
- No changes to `kb_create` / `kb_list` / `kb_info` signatures — only additive fields in `kb_info` response.
- No new authentication / permissions model.
- No cross-process lock implementation (documented only).
- No multi-dialect sql support in release 1 (sqlite only; document Postgres/MySQL as future work).
- Ingestor changes for non-sql source_types (file, website, codebase, etc.) stay untouched.
