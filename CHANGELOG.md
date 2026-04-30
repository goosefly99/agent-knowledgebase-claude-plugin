# Changelog

## 0.12.0 — 2026-04-30

> Vectorstore robustness + TextvecBackend opt-in. Three additive phases
> (A/B/C) shipped as one release. Default `Settings.kb_backend` stays
> `'chromadb'`; `'textvec'` is opt-in. No breaking changes.

### Added

- **Phase A — chromadb eager-warm.** `ChromadbBackend.warmup(kb_id)` +
  `KnowledgebaseService.warmup_all_chromadb_kbs()` + `server.py`
  daemon-thread trigger. Eliminates the 180s HNSW cold-load on first
  `kb_query`/`kb_search`/`kb_ingest` after fresh MCP server launch.
  Wallclock amortized into server-init (non-blocking; 30s wallclock
  cap; opt-out via `AGENT_KB_CHROMADB_EAGER_WARM=false`).
- **Phase A — chromadb filter passthrough.** `ChromadbBackend.query()`
  now honors non-None `filters` instead of raising `NotImplementedError`;
  filter dict is structurally validated then merged into the chromadb
  `where=` clause with `kb_id` placed last to preserve per-KB isolation.
  `ChromadbBackend.search()` rejects non-None filters with `ValueError`
  (FTS path has no metadata-filter support).
- **Phase B — `TextvecBackend`.** New 4th `RetrieverBackend` Protocol
  implementation over SQLite FTS5+BM25 + Porter tokenizer, on a
  contentless `chunks_fts` virtual table joined to `chunks` via
  integer rowid. Zero new external dependencies (stdlib `sqlite3`).
  Opt in via `AGENT_KB_BACKEND=textvec`. `info()` returns
  `dominant_embedding_model='lexical-fts5'` (chromadb / markdown /
  lightrag arms unchanged). Settings.kb_backend Literal widened to
  include `'textvec'`; default stays `'chromadb'`. New tunable fields:
  `fts5_tokenizer="porter unicode61"`, `bm25_k1=1.2`, `bm25_b=0.75`,
  `lexical_min_token_len=2`.
- **Phase B — FTS5 schema.** `chunks_fts` virtual table created
  idempotently at schema bootstrap; `chunks_ai`/`chunks_au`/`chunks_ad`
  triggers keep the index in sync. `Database.rebuild_fts5(kb_id) -> int`
  backfills the index for legacy v0.11.0 KBs whose chunks rows
  predate the triggers. Server runs an FTS5 availability probe at
  `main()` startup (`CREATE VIRTUAL TABLE _probe USING fts5(x)`);
  fails fast with structured stderr JSON on missing FTS5.
- **Phase B — `bin/benchmark_recall.py`.** Standalone harness for
  paired recall@k comparison between any two backends. CLI args
  `--kb-id`, `--queries-file`, `--baseline-backend`, `--target-backend`,
  `--top-k`. Human-readable + JSON output.
- **Phase C — `kb_migrate(kb_id, target_backend='textvec')`.**
  Backfills `chunks_fts` for legacy chromadb-stamped KBs without
  re-ingest. Acquires per-kb FileLock (timeout=0, non-blocking) +
  in-process `service._get_kb_lock(kb_id)`; idempotent via
  `<kb_root>/.migrated_to=textvec` sentinel + TOCTOU recheck inside
  lock. Three terminal states: `migrated`, `already_migrated`,
  `deferred` (lock contention; no raise). `MIGRATION_NO_ROWS`
  defensive warning when rebuild_fts5 returns 0 on a non-empty KB.
  `RuntimeError` with install hint if `filelock` unavailable
  (filelock remains intentionally non-default per
  `docs/cross-process-lock-recipe.md`).
- **Phase C — read-fallback hint.** `TextvecBackend.search` (and
  therefore `query` via delegation) emits a once-per-session
  `REBUILD_RECOMMENDED` stderr-log line when fetched chunks rows
  have non-NULL `embedding_provider` (legacy chromadb-stamped). Hint
  command: `kb_rebuild_index --backend=textvec`. Module-level set
  guards against re-emission per session per `kb_id`.

### Changed

- **`kb_migrate` MCP tool** — `target_backend` Literal widened from
  `{markdown, chromadb}` to `{markdown, chromadb, textvec}`. The
  `@mcp.tool()` outer + `@_with_tool_timeout` inner decorator order
  is preserved (frozen contract, pinned by
  `tests/contract/test_decorator_order.py`). The widening is purely
  additive — the existing `markdown` and `chromadb` arms behave
  bit-for-bit as before; only the new `textvec` arm is new.

### Frozen contracts (unchanged)

- 26 frozen `kb_*` MCP tool names + signatures + response shapes
- Probe-4 contract (5 keys: `source_type`, `uri`, `dedup_key`,
  `page_id`, `dominant_embedding_model`; inapplicable values are
  `None`, never omitted)
- 11-field `knowledgebase_stderr_log` schema
- `@mcp.tool()` outer + `@_with_tool_timeout` inner decorator order
- 180s `_with_tool_timeout` budget
- Cross-process FileLock per-`kb_id` (recipe in
  `docs/cross-process-lock-recipe.md`; `filelock` intentionally not
  a default runtime dep)
- SQL-injection AST validator + read-only sqlite (`?mode=ro`) +
  50-row hard-reject on `kb_ingest_batch(source_type=sql_database)`

### Tests

- 11 new test files (~124 new tests) across Phases A/B/C:
  `test_chromadb_eager_warm.py`,
  `test_chromadb_filter_passthrough.py`, `test_textvec_backend.py`,
  `test_textvec_fts5_recall.py`, `test_textvec_porter_tokenizer.py`,
  `test_textvec_metadata_filter_compose.py`,
  `test_fts5_startup_probe.py`, `test_database_rebuild_fts5.py`,
  `test_kb_migrate_to_textvec.py`,
  `test_textvec_legacy_kb_read_fallback.py`,
  `test_textvec_migration_idempotent.py`. Total suite:
  818 passed, 8 skipped.

### Deferred (gated on adoption signal)

- v2.2 Phase 3 — default flip + chromadb removal (now slated for
  0.13.0)
- v2.2 Phase 4 — cleanup (`services/embeddings.py`,
  `services/vectorstore.py`, `PipelinePhase.embed`,
  `_backfill_chunks_embedding_snapshot`)
- v2.2 Phase 5 — MultiQueryRetriever / LLM query rewrite (now
  slated for 0.14.0)

## 0.11.0+phase6 — 2026-04-24

> Phase 6 of the v2.1 redesign — `LightRAGBackend` stub. **Stub only**;
> activation deferred per the spec adoption-signal gate. No version
> bump from 0.11.0 — Phase 6 reserves the contract surface but adds
> no live retrieval path.
>
> spec_id: `70ab2170-381a-4657-bcd1-28a40c6f369b`
> Source spec: `pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json`

### Added

- **`backends/lightrag_backend.py`** — new `LightRAGBackend` class
  implementing the full `RetrieverBackend` Protocol surface (7
  methods). Construction succeeds (so callers can probe non-activated
  KBs); `info()` returns all 5 probe-4 keys with `None` values plus
  a `backend='lightrag'` discriminator + `status='unavailable'`;
  `health_check()` returns the same deferred-status payload. The
  remaining 5 methods (`index`, `query`, `search`, `delete`, `count`)
  raise `NotImplementedError` with a canonical activation message
  listing the three preconditions (`AGENT_KB_LIGHTRAG_URL`,
  `AGENT_KB_BACKEND=lightrag`, per-method implementation).
- **`docs/lightrag_backend.md`** — design doc covering: deferral
  rationale (no adoption signal), activation trigger conditions,
  sidecar architecture diagram, `docker-compose.yml` snippet for
  the LightRAG sidecar (image, ports, persistent volume, healthcheck,
  env vars), REST API contract (5 endpoints: `/insert`, `/query`,
  `/delete`, `/info`, `/health`), migration path from chromadb /
  markdown (requires extending `services/migration.py`), and an
  activation checklist that must go green before the stub can be
  promoted to live forwarding code.
- **`tests/test_lightrag_backend_stub.py`** — 11 tests pinning the
  stub's contract: importability, construction success, Protocol
  membership, probe-4 5-key shape on `info()`,
  `health_check()` deferred-status payload, NotImplementedError on
  the 5 raising methods (with the activation message verified
  verbatim), and the factory-branch promotion assertion.

### Changed

- **`backends/__init__.py::get_backend()`** — the `'lightrag'` branch
  now imports and constructs `LightRAGBackend(settings, service=...)`
  instead of raising `NotImplementedError` directly. Construction
  succeeds; the raises live on the per-method bodies and surface
  lazily on actual use.
- **`tests/test_retriever_backend_protocol.py`** — `_BACKEND_FACTORIES`
  extended with the `'lightrag'` variant so the parameterised
  Protocol-conformance tests cover all three concrete backends. The
  former `test_get_backend_factory_raises_for_lightrag_until_phase6`
  was inverted into a positive `test_get_backend_factory_returns_lightrag_stub_in_phase6`
  (matches the inversion Phase 3 made for the markdown branch). Two
  small targeted tests added to verify the lightrag stub's `index`
  and `query` raise with the canonical activation message.
- **`tests/test_redesign_contract.py`** — docstring updated to note
  that Phase 4 introduced the additive `kb_migrate` (set is now
  26 tools, not 25) and Phase 6 adds NO MCP tools (the 26-tool
  surface is preserved bit-for-bit).

### Frozen contracts (preserved)

- **The 26 MCP tools surface (25 frozen v0.6.0 + Phase 4's
  `kb_migrate`)** — unchanged. Phase 6 adds NO new MCP tools.
  `server.py` is untouched.
- **Decorator order** on every tool: `@mcp.tool()` outer,
  `@_with_tool_timeout` inner. Per `docs/redesign/MISTAKES.md` M-01.
- **`Settings.kb_backend` default** — still `'chromadb'`. Phase 6
  does NOT change any default.
- **`Settings.embedding_provider` / `embedding_model` defaults** —
  still `'remote'` / `'text-embedding-3-small'` (Phase 5).
- **Probe-4 contract** (`source_type`, `uri`, `dedup_key`,
  `page_id`, `dominant_embedding_model`) on `kb_info` /
  `kb_list_pages` / `kb_list_sources` — unchanged. The new
  `LightRAGBackend.info()` returns all 5 keys with `None` values per
  validation finding f-20.
- **`knowledgebase_stderr_log` 11-field schema** — unchanged. Phase
  6 emits no new stderr lines.
- **Database schema** — Phase 6 adds NO new columns / tables /
  migrations.

### Deferred (Phase 6 activation)

- **Live REST forwarding** to the LightRAG sidecar. Activation is
  conditional on adoption signal — see
  `docs/lightrag_backend.md` for the operator-facing trigger
  conditions and the activation checklist. If no operator opts in
  within 6 months of v0.11.0 GA, this stub may be deleted to reduce
  maintenance surface.
- **`kb_migrate(target_backend='lightrag')`** — currently rejected
  by the Phase 4 migration tool. Activation requires extending
  `services/migration.py` with `migrate_to_lightrag` (per the
  "Migration path" section of `docs/lightrag_backend.md`).

## 0.11.0 — 2026-04-24

> Phase 5 of the v2.1 redesign — embedding default-flip
> (`ollama/qwen3-embedding:8b` -> `remote/text-embedding-3-small`),
> opt-in `fastembed` provider, per-chunk `embedder_version` stamping,
> and a wholesale restructure of `pyproject.toml` to move
> `sentence-transformers`, `tree-sitter*`, `pdfplumber`, `sqlalchemy`,
> and `sqlparse` from default dependencies into opt-in extras.
>
> **BREAKING DEFAULT CHANGE.** Operators with v0.6.0/v0.10.x KBs
> ingested under the default Ollama embedder MUST stay on Phase 4
> (v0.10.0) before adopting Phase 5. The Phase 4 per-chunk provider
> snapshot is the safety net that keeps existing Ollama-ingested KBs
> queryable AFTER this default flip — without Phase 4, every existing
> KB would silently dimension-mismatch on the first `kb_query` after
> upgrade.
>
> spec_id: `70ab2170-381a-4657-bcd1-28a40c6f369b`
> Source spec: `pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json`

### Migration story for existing operators

| Scenario | Action |
|---|---|
| Brand-new install | No action. New KBs default to `provider='remote'`, `model='text-embedding-3-small'`. Set `AGENT_KB_EMBED_API_KEY` + `AGENT_KB_EMBED_BASE_URL` (e.g. `https://api.openai.com/v1`). |
| Existing v0.6.0/v0.10.x KB on Ollama (Phase 4 backfill ALREADY run) | No action — Phase 4's per-chunk provider snapshot keeps the original Ollama embedder wired in for queries against that KB. New KBs created post-upgrade will pick up the new default. |
| Existing v0.6.0 KB on Ollama, jumping v0.6.0 → v0.11.0 directly (Phase 4 backfill NEVER run) | **Auto-handled in v0.11.0.** First `kb_query` against the KB triggers an automatic backfill: chunks whose snapshot is `(qwen3-embedding..., NULL, NULL)` get `provider='ollama'` + `base_url='http://127.0.0.1:11434'` stamped retroactively, then the query proceeds normally. The auto-fix emits a structured `LEGACY_OLLAMA_KB_BACKFILLED` stderr line so operators see what happened. Defensive opt-out: explicitly set `AGENT_KB_EMBEDDING_PROVIDER=ollama` + `AGENT_KB_EMBEDDING_MODEL=qwen3-embedding:8b` env vars BEFORE first query, or manually run `backfill_provider_snapshot`, to avoid relying on the heuristic. |
| Existing KB on a non-Ollama legacy embedder (NULL provider, non-`qwen3-embedding` model) | Auto-backfill heuristic does NOT fire. Either run `backfill_provider_snapshot` manually OR set the matching `AGENT_KB_EMBEDDING_*` env vars before first query, otherwise snapshot resolves to remote-without-API-key and raises `ValueError`. |
| Want to switch an existing KB to a new embedder | Re-ingest under the new embedder. The mixed-version rejection (`EMBEDDER_VERSION_MISMATCH`) prevents accidental cross-embedder mixing; set `AGENT_KB_AUTO_REEMBED=1` to bypass while you re-ingest. |
| Want local embeddings (no network) | `pip install agent-knowledgebase[embed-local-onnx]` for fastembed (~150 MB), or `pip install agent-knowledgebase[embed-local-st]` for sentence-transformers (~1.3 GB). |

### Added

- **New `'fastembed'` embedding provider** in
  `services/embeddings.py`. Lazy-imports `qdrant-fastembed` only
  inside `FastembedEmbedder.__init__`; raises a friendly `RuntimeError`
  pointing at `pip install agent-knowledgebase[embed-local-onnx]` if
  the extra isn't installed. Same `Embedder` Protocol surface as the
  remote / ollama / sentence-transformers providers.
- **Per-chunk `embedder_version` stamping** — additive ALTER ADD
  migration adds one column to `chunks`:
  ```sql
  ALTER TABLE chunks ADD COLUMN embedder_version TEXT;
  ```
  Stamped on every newly-inserted chunk by `Database.insert_chunk`
  from `chunk.metadata['embedder_version']` (mirroring the Phase 4
  `embedding_provider` / `embed_base_url` columns). Strictly ALTER
  ADD — non-destructive — and idempotent across re-opens.
- **`Database.get_embedder_versions(kb_id)`** — returns the distinct
  set of stamped `embedder_version` values for a KB. Used by
  `KnowledgebaseService._ingest_source_locked` to detect mixed-
  version ingests; falls back to `json_extract(metadata,
  '$.embedder_version')` for chunks stamped via the metadata-only
  path.
- **`Embedder.embedder_version` Protocol property** — opaque
  string identifying the embedder's vector geometry. Format
  convention: `"<library>/<model_name>[@<quant_or_provider>]"`. The
  built-in providers report:
  - `RemoteEmbedder` → `"remote/<model>@<base_url>"`
  - `OllamaEmbedder` → `"ollama/<model>@<base_url>"`
  - `SentenceTransformerEmbedder` → `"sentence-transformers/<model>@hf-fp32"`
  - `FastembedEmbedder` → `"fastembed/<model>-int8"`
- **`EmbedderVersionMismatchError`** in
  `services/knowledgebase.py` (error_code
  `EMBEDDER_VERSION_MISMATCH`). Raised at the top of
  `_ingest_source_locked` when the incoming embedder's version
  doesn't match any version already stamped on the KB's chunks AND
  `AGENT_KB_AUTO_REEMBED=1` is not set. Carries `kb_id`,
  `existing_versions`, `incoming_version`, and a `to_payload()`
  method for JSON serialization. Non-destructive: the check runs
  BEFORE any source row is inserted.
- **`AGENT_KB_AUTO_REEMBED=1` env var** — documented bypass for the
  mixed-version rejection. Operators set this when intentionally
  re-embedding an existing KB under a new embedder. **IMPORTANT after
  bypass:** the bypass disables only the rejection — chunks ingested
  under the OLD embedder remain on disk under the OLD geometry. To
  avoid silently-unreachable mixed-version chunks at query time, the
  operator MUST follow up with a re-ingest of the prior chunks under
  the new embedder (or `kb_delete` + recreate). Phase 5 emits a
  structured `MIXED_EMBEDDER_VERSIONS_DETECTED` stderr line at most
  once per kb_id per process when the situation is detected at query
  time, so operators are not left guessing why a chunk that was
  visible last week is now missing from results.
- **Auto-backfill for legacy v0.6.0 Ollama KBs at query time (B-02).**
  The first `kb_query` against a KB whose chunks carry only the model
  name in metadata (NULL `embedding_provider` / NULL `embed_base_url`)
  AND whose model name starts with `qwen3-embedding` (the historical
  v0.6.0 Ollama default) triggers an automatic
  `Database.backfill_embedding_snapshot` UPDATE that stamps
  `provider='ollama'` + `base_url='http://127.0.0.1:11434'` onto the
  matching chunks. Idempotent (per-process suppression set + the
  underlying COALESCE-NULL filter). Emits a `LEGACY_OLLAMA_KB_BACKFILLED`
  structured stderr line. Without this heuristic, an operator who
  jumps v0.6.0 → v0.11.0 directly (skipping the explicit Phase 4
  backfill) would hit `ValueError("AGENT_KB_EMBED_API_KEY required for
  remote embeddings")` on first query.
- **Runtime `ValueError("AGENT_KB_*")` is now wrapped as
  `config_missing` (I-06).** The Phase 0 `_get_service` wrap covered
  config errors raised from `Settings.resolve_paths()`. After the
  Phase 5 default flip, fresh-install operators with no
  `AGENT_KB_EMBED_API_KEY` hit `ValueError("AGENT_KB_EMBED_API_KEY
  required for remote embeddings")` raised inside the embedder build —
  not via `resolve_paths`. The `_with_tool_timeout` wrapper now also
  catches the runtime `ValueError("AGENT_KB_*")` family and returns the
  same `{"error":"config_missing","missing":"AGENT_KB_<var>",...}`
  payload, so the FIRST kb_query call returns a clean structured
  diagnostic instead of a stack trace.
- **fastembed parity verified by Jaccard@10 test; MMR re-tuning
  deferred until empirical need surfaces.** The v0.11.0 cut initially
  shipped two `query_mmr_lambda_*` Settings fields + a
  `services.query.mmr_lambda_for(settings, embedder)` helper as
  forward-compat scaffolding for a fastembed-specific MMR diversity
  bias. Code review (B-03) caught that nothing in
  `QueryOrchestrator.query` / `search` / `hybrid_query` actually wires
  MMR into the chunk path — they all use raw cosine + FTS rank — so
  the fields + helper were dead surface and were removed in the
  follow-up patch. The `tests/test_fastembed_recall_parity.py`
  Jaccard@10 ≥ 0.95 result on a fixed corpus shows fastembed-int8 and
  sentence-transformers-fp32 already produce nearly-identical
  neighbour sets, so the spec's premise that fastembed needs
  MMR-compensation isn't supported empirically. If a future
  orchestrator change wires MMR into the query path, the helper +
  Settings fields can re-appear alongside the actual rerank logic.
- **`tests/test_embedder_version_mismatch.py`** — 9 tests covering
  the schema migration (additive + idempotent), the `insert_chunk`
  -> native column mapping, `get_embedder_versions` set semantics
  (distinct + dedup + NULL-skip + metadata fallback), the
  same-version pass, the different-version rejection (with
  `error_code` / `to_payload` shape), the
  `AGENT_KB_AUTO_REEMBED=1` bypass, and the legacy-NULL KB pass.
- **`tests/test_fastembed_dimension_compat.py`** — 4 tests
  pinning fastembed-MiniLM-class dimensionality (384), the
  dimension-cache property semantics, the int8-marker on
  `embedder_version`, and the Python-list return type.
  `pytest.importorskip("fastembed")` at the top — skipped when
  the `[embed-local-onnx]` extra is not installed.
- **`tests/test_fastembed_recall_parity.py`** — Jaccard@10
  parity test between fastembed-MiniLM-int8 and
  sentence-transformers-MiniLM-fp32 on a fixed 50-doc corpus,
  asserting >= 0.95 (validation finding f-09 sharpening). Skipped
  when either local-embedding extra is missing.
- **`docs/install_size.md`** — per-extra install-size table,
  cumulative table, regeneration command, and methodology. Per
  `docs/redesign/MISTAKES.md` M-03: do not quote install-size
  numbers without enumerating which deps stay vs move. Realistic
  default install ~700 MB (chromadb dominates); `[embed-local-st]`
  adds ~1.3 GB; `[embed-local-onnx]` adds ~150 MB.

### Changed

- **`Settings.embedding_provider`** default flipped from `'ollama'`
  to `'remote'`. The Literal type also gains `'fastembed'`. Existing
  v0.6.0/v0.10.x KBs continue to query under their original embedder
  via the Phase 4 per-chunk snapshot — proven by the unchanged
  `tests/test_per_page_provider_snapshot.py::test_query_embedder_for_uses_snapshot_after_default_flip`
  which still passes after this flip.
- **`Settings.embedding_model`** default flipped from
  `'qwen3-embedding:8b'` to `'text-embedding-3-small'`.
- **`SentenceTransformerEmbedder.__init__`** now lazy-imports
  `sentence_transformers` and raises a friendly `RuntimeError`
  pointing at `pip install agent-knowledgebase[embed-local-st]` if
  the extra is missing. The class itself stays — operators who
  install the extra still get the same surface.
- **`pyproject.toml`** restructured. Defaults now carry only the
  minimal-viable-retrieval set (`mcp`, `pydantic`,
  `pydantic-settings`, `chromadb`, `trafilatura`, `gitpython`,
  `httpx`, `tiktoken`). New `[project.optional-dependencies]`
  groups: `embed-local-onnx`, `embed-local-st`, `ingest-codebase`,
  `ingest-file`, `ingest-sql`, plus an `all` convenience extra.
  `pinecone` extra retained.
- **`KnowledgebaseService._ingest_source_locked`** — gains the
  Phase 5 mixed-version rejection at the top (BEFORE any source
  row is inserted) and stamps `embedder_version` on each chunk's
  metadata in the embed loop. Defensive `isinstance(version, str)`
  check keeps test mocks (whose `MagicMock.embedder_version`
  returns a MagicMock) from poisoning the stamping path.
- **`Database.insert_chunk`** stamps `embedder_version` from
  `chunk.metadata` into the new native column alongside the Phase
  4 `embedding_provider` / `embed_base_url` stamps.
- **`create_embedder_for_model`** signature gains an optional
  `version` kwarg (Phase 5 forward-compat — accepted and ignored
  today). The Phase 4 `(model_name, provider, base_url)` signature
  is preserved bit-for-bit.

### Frozen contracts (preserved)

- The 25 v0.6.0 `kb_*` MCP tools + Phase 4's `kb_migrate` =
  **26 tools surface**. Names, parameters, response shapes — all
  bit-for-bit identical. No new MCP tools in Phase 5.
- Decorator order on every tool: `@mcp.tool()` outer,
  `@_with_tool_timeout` inner. Per `docs/redesign/MISTAKES.md` M-01.
- `Settings.kb_backend` default — still `'chromadb'`. Phase 5
  changes embedding defaults, NOT backend default.
- Probe-4 contract (`source_type`, `uri`, `dedup_key`, `page_id`,
  `dominant_embedding_model`) on `kb_info` / `kb_list_pages` /
  `kb_list_sources` — unchanged.
- `knowledgebase_stderr_log` 11-field schema — unchanged.
- Database schema is strictly additive (`ALTER TABLE ADD COLUMN`
  with NULL defaults). Rollback SQL:
  ```sql
  -- Requires SQLite >= 3.35.
  ALTER TABLE chunks DROP COLUMN IF EXISTS embedder_version;
  ```

### Deferred

- **`requirements.lock` regeneration** — neither `uv` nor
  `pip-compile` was on the test environment's PATH at release-cut
  time. The pyproject.toml `[project] dependencies` and
  `[project.optional-dependencies]` blocks are the single source
  of truth for runtime resolution. To regenerate locally:
  ```bash
  uv pip compile pyproject.toml --generate-hashes -o requirements.lock
  # OR:
  pip install pip-tools && pip-compile --generate-hashes -o requirements.lock pyproject.toml
  ```
  Operators wanting a hash-pinned install should run the command
  from a clean checkout. Documented in `docs/install_size.md`.

### NOT in scope (Phase 6+)

- `LightRAGBackend` — Phase 6 (deferred, conditional on adoption
  signal). The Phase 6 entry above ships the stub class +
  `docs/lightrag_backend.md` design doc — no production LightRAG
  forwarding path in this release.

## 0.10.0 — 2026-04-24

> Phase 4 of the v2.1 redesign — migration tooling + dual-backend
> transition + per-page provider snapshot. **Phase 4 ships BEFORE
> Phase 5** (the embedding default-flip) because without the
> per-page provider snapshot + read-fallback, the global default
> change in Phase 5 would silently break every existing v0.6.0
> chromadb KB ingested with the Ollama default. Per validation
> findings f-10 / f-17.
>
> spec_id: `70ab2170-381a-4657-bcd1-28a40c6f369b`
> Source spec: `pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json`

### Added

- **New MCP tool: `kb_migrate(kb_id, target_backend)`** — additive
  (probe-4-safe). Cuts a knowledgebase over from one
  `RetrieverBackend` to another (`"chromadb"` <-> `"markdown"`;
  `"lightrag"` rejected — Phase 6 deferred). Materialises the new
  layout on disk, writes the routing sentinel
  `<saves_dir>/<kb-name>/.migrated_to=<backend>`, invalidates the
  per-KB backend cache. **Non-destructive**: the OLD backend's data
  remains on disk so rollback = delete the sentinel file. The
  decorator stack is identical to the existing 25 tools
  (`@mcp.tool()` outer, `@_with_tool_timeout` inner) per
  `docs/redesign/MISTAKES.md` M-01. The frozen MCP surface is now
  **26 tools** (was 25); the contract test
  `tests/test_redesign_contract.py::_FROZEN_MCP_TOOLS_V0_6_0` was
  extended in place — no existing tool removed.
- **New service module `services/migration.py`** with
  `export_to_markdown` / `import_from_markdown` /
  `export_chromadb_dump` / `import_chromadb_dump` plus the top-level
  `migrate(...)` orchestrator and the `backfill_provider_snapshot(...)`
  helper used by the Phase 5 gate. Lazy-imported by `server.py` so
  test environments without all migration deps still load the server.
- **Per-page (per-chunk) provider snapshot** — additive ALTER ADD
  migration adds two columns to `chunks`:
  ```sql
  ALTER TABLE chunks ADD COLUMN embedding_provider TEXT;
  ALTER TABLE chunks ADD COLUMN embed_base_url TEXT;
  ```
  Stamped on every newly-inserted chunk by `Database.insert_chunk`
  from the chunk's `metadata` blob (mirroring the existing
  `embedding_model` field). Read at query time by
  `Database.get_embedding_snapshot(kb_id)` which returns the
  dominant `(model, provider, base_url)` triple. Migration is
  strictly ALTER ADD — non-destructive — and idempotent across
  re-opens.
- **Backwards-compatible `create_embedder_for_model` signature.**
  Old call sites passing `(config, model_name)` continue to work
  unchanged; new Phase 4 call sites pass
  `(config, model_name, provider=..., base_url=...)` so the snapshot
  rebuilds the original embedder even after the Phase 5 default
  flip swaps the global `Settings.embedding_provider`. The Phase 0
  dimension-probe path is preserved (no signature break).
- **`Settings.kb_backend_per_kb`** field — `dict[str, str]` parsed
  from either the comma-separated env form
  `AGENT_KB_BACKEND_PER_KB="kb1=markdown,kb2=chromadb"` or a JSON
  object via the user/project config files. Validation rejects
  invalid backend names up-front. Resolution precedence
  (highest first): `<saves_dir>/<kb-name>/.migrated_to` sentinel >
  `kb_backend_per_kb[kb_id]` > global `Settings.kb_backend`.
- **Read-fallback** in `MarkdownWikiBackend.query()` /
  `MarkdownWikiBackend.search()`: when `kb_backend='markdown'` is
  active for a KB but `<kb-name>/wiki/` is missing while
  `<kb-name>/chroma/` (or `<kb-name>/vector_store/`) exists, the
  call is silently delegated to `ChromadbBackend` with a structured
  `phase='read_fallback'` /
  `error_code='MARKDOWN_READ_FALLBACK_TO_CHROMADB'` stderr_log
  emission carrying all 11 required fields.
- **`backends.resolve_backend_name(...)`** helper — exposes the
  routing precedence so callers (tests, docs, future tools) can
  introspect the effective backend choice for a given `kb_id`.
- **`backends.MIGRATED_SENTINEL_FILENAME`** constant
  (`".migrated_to"`) — single source of truth for the sentinel
  filename used by the migration flow and the routing resolver.
- **Per-KB backend cache** in `KnowledgebaseService` — backends
  resolved via `_backend_for(kb_id)` are cached behind a lock so
  the routing precedence is computed once per KB per process.
  Cache invalidated on `delete_kb` and after each successful
  `kb_migrate`.
- **`tests/test_migration_chromadb_to_markdown.py`** — 10 tests
  covering the full migration round-trip, sentinel write, post-
  migration query routing, non-destructive guarantee, dump round-
  trip, backfill (live + dry-run), and the kb_migrate decorator
  order spot-check.
- **`tests/test_per_page_provider_snapshot.py`** — 9 tests covering
  the schema migration (additive + idempotent), the
  `insert_chunk` -> native column mapping,
  `get_embedding_snapshot` resolution, the backwards-compat
  `create_embedder_for_model` signature, the explicit-override
  build path, the snapshot-driven query embedder rebuild after a
  simulated default flip, and the legacy-row backfill.
- **`tests/test_kb_backend_per_kb_routing.py`** — 11 tests pinning
  the per-KB routing parser, the precedence ordering (sentinel >
  per-kb > global), the malformed-env rejection, and the
  `_backend_for(kb_id)` cache picking the right backend.
- **`tests/test_read_fallback.py`** — 5 tests pinning the
  read-fallback semantics: fallback fires when wiki/ missing &
  chroma/ exists, the `vector_store/` alias is recognized, no
  fallback when wiki/ is present, no fallback when neither exists,
  and the structured stderr emission carries all 11 schema fields.
- **`docs/migration_guide.md`** — operator guide covering when to
  migrate, how to call `kb_migrate`, sentinel file semantics,
  per-KB routing precedence, read-fallback behavior, the
  embedder snapshot rationale (why Phase 5 is safe), backfill
  procedure for legacy KBs, and a step-by-step runbook.

### Changed

- `KnowledgebaseService._query_embedder_for(kb_id)` now reads
  `Database.get_embedding_snapshot(kb_id)` (the
  `(model, provider, base_url)` triple) instead of just the model
  name — passing the snapshot's provider + base_url into
  `create_embedder_for_model` so an existing v0.6.0 KB ingested
  under provider=ollama stays queryable AFTER the Phase 5 default
  flip to provider=remote/text-embedding-3-small.
- `KnowledgebaseService._ingest_source_locked` chunk-stamping path
  now includes `embedding_provider` and `embed_base_url` alongside
  the existing `embedding_model` and `kb_id` keys in
  `chunk.metadata`. Secrets (`embed_api_key`) are intentionally
  NOT stamped — the snapshot only carries non-secret routing
  metadata.
- `KnowledgebaseService.query` / `KnowledgebaseService.search` /
  the ingest write path / the source-deletion path now route
  through `self._backend_for(kb_id)` (Phase 4 per-KB routing)
  instead of `self._backend` directly. The global `self._backend`
  is preserved for legacy cross-KB scans (`get_page`,
  `get_source`) where no specific kb_id is in scope.
- `tests/test_redesign_contract.py::_FROZEN_MCP_TOOLS_V0_6_0` —
  extended to include the additive `kb_migrate` tool (the
  scaffold note at lines 152-173 explicitly anticipated this
  Phase 4 add). The contract test is still
  `@pytest.mark.skip`-ed scaffold; promotion is a Phase 5
  deliverable.
- `tests/test_knowledgebase.py::test_query_uses_per_kb_embedder_*`
  updated to patch `Database.get_embedding_snapshot` (the new
  Phase 4 helper) instead of `count_chunks_by_embedding_model`.
  The fake builder now accepts the `provider` / `base_url`
  kwargs and asserts they propagate through the snapshot.

### Frozen contracts (preserved)

- The 25 v0.6.0 `kb_*` MCP tools — names, parameters, response
  shapes — all bit-for-bit identical. `kb_migrate` is purely
  additive.
- Decorator order on every tool including `kb_migrate`:
  `@mcp.tool()` outer, `@_with_tool_timeout` inner. Per
  `docs/redesign/MISTAKES.md` M-01.
- `Settings.kb_backend` default — still `'chromadb'`. Phase 4
  does NOT touch the global default. (The Phase 5 flip is
  separate.)
- `Settings.embedding_provider` default — still `'ollama'`. The
  default flip is the Phase 5 deliverable (gated on this Phase 4
  release being in production).
- Probe-4 contract (`source_type`, `uri`, `dedup_key`,
  `page_id`, `dominant_embedding_model`) on `kb_info` /
  `kb_list_pages` / `kb_list_sources` — unchanged. Both backends'
  `info()` responses still return the 5 keys plus their
  backend-tagged diagnostic fields.
- `knowledgebase_stderr_log` 11-field schema — unchanged. The
  new `read_fallback` / `migrate_start` / `migrate_done`
  emissions all carry the full 11-field payload.
- Database schema is strictly additive (`ALTER TABLE ADD COLUMN`
  with NULL defaults). Rollback SQL:
  ```sql
  -- Requires SQLite >= 3.35.
  ALTER TABLE chunks DROP COLUMN IF EXISTS embedding_provider;
  ALTER TABLE chunks DROP COLUMN IF EXISTS embed_base_url;
  ```

### NOT in scope (Phase 5+)

- `embedding_provider` default flip from `'ollama'` to `'remote'`
  — Phase 5. Phase 4 establishes the safety net (snapshot +
  read-fallback) so the default flip is non-breaking. Phase 5
  has an explicit gate: confirm Phase 4 is in production and
  every existing KB has been backfilled via
  `backfill_provider_snapshot(...)`.
- `fastembed` provider, `embedder_version` stamping, install-size
  optimisations — all Phase 5.
- `LightRAGBackend` — Phase 6 (deferred).

## 0.9.0 — 2026-04-24

> Phase 3 of the v2.1 redesign — opt-in Karpathy-style markdown wiki
> backend. **chromadb stays the default**; markdown is purely opt-in
> via `AGENT_KB_BACKEND=markdown`. Zero behavior change for existing
> v0.6.0/0.8.x KBs unless the operator explicitly flips the env var.
>
> spec_id: `70ab2170-381a-4657-bcd1-28a40c6f369b`
> Source spec: `pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json`

### Added

- New `MarkdownWikiBackend` in
  `src/agent_knowledgebase/backends/markdown_backend.py` — second
  concrete implementation of the Phase 2 `RetrieverBackend` Protocol.
  Stores knowledge as Karpathy-style interlinked markdown files on
  disk (`docs/karpathy-llm-wiki.md` design pattern) instead of as
  chromadb vectors. Layout under `<saves_dir>/<kb-dir>/wiki/`:
  `raw/<source_id>.<ext>`, `pages/<slug>.md`, `index.md` (or sharded
  `index/<category>.md` above 200 pages), `log.md`, `hot.md`.
- New `KbIngestorUnsuitableError` (error_code `KB_INGESTOR_UNSUITABLE`)
  raised when `index()` is called with a `source_type` outside the
  positive-allow list `{file, website, api_endpoint with payload <2MB}`
  (validation finding f-15 — positive form, NOT a deny-list).
  Disallowed types include `sql_database`, `codebase`, `git_history`,
  `directory`, and `api_endpoint` payloads >= 2MB. Set
  `AGENT_KB_FORCE_WIKI_INGEST=1` to bypass the check; the bypass
  emits a structured `knowledgebase_stderr_log` warning carrying
  `error_code=KB_INGESTOR_UNSUITABLE_FORCED` so operators can grep
  for forced ingests.
- sqlite FTS5 in-memory index over `pages/*.md` content. Both
  `RetrieverBackend.query()` (vector path) and `search()` (keyword
  path) route through the FTS5 index — markdown has no embedding
  path, so `query()` re-routes to `search()` and surfaces scores in
  `[0, 1]` descending via `score = 1 / (1 + abs(rank))`.
- Sharded `index.md`: above 200 pages the index becomes a directory
  pointer with one shard per `source_type` under `wiki/index/`.
  Categorisation via `source_type` is the v0 placeholder
  (per spec.phases[3].tasks[4]).
- `tests/test_markdown_backend.py` — 13 tests covering single-article
  ingest, 250-article shard transition, allow-list rejection (each
  disallowed `source_type`, plus oversized api_endpoint), force-flag
  bypass + stderr_log emission, empty-input no-op, search-on-empty,
  delete by source_id / by ids, health_check, and factory wiring.
- `tests/contract/test_markdown_backend_probe4_shape.py` — pins the
  probe-4 contract on `MarkdownWikiBackend.info()` directly: every
  required key (`source_type`, `uri`, `dedup_key`, `page_id`,
  `dominant_embedding_model`) is present, with `dominant_embedding_model`
  ALWAYS `None` for the markdown backend (markdown does not embed)
  but never omitted (validation finding f-20).
- `docs/markdown_backend.md` — usage guide, allow-list rationale,
  per-source_type cost-model table (rough order-of-magnitude
  estimates for the deferred LLM page-extraction pass), sharded-index
  layout diagram, probe-4 contract table, and an explicit
  Limitations section calling out: synchronous-only ingest (no
  out-of-band LLM yet), no vector queries, no incremental update,
  filters pushdown limited to `source_type` / `slug`, no cross-process
  serialization, `source_type`-based categories.

### Changed

- `src/agent_knowledgebase/backends/__init__.py` — `get_backend()`
  factory's `'markdown'` branch now instantiates
  `MarkdownWikiBackend(settings, service=service)` instead of raising
  `NotImplementedError`. The `'lightrag'` branch stays as
  `NotImplementedError` (Phase 6 deferred).
- `tests/test_retriever_backend_protocol.py` — extended
  `_BACKEND_FACTORIES` with the markdown variant so all
  Protocol-conformance assertions (runtime-checkable Protocol
  membership, method signatures via `inspect.signature`) run
  parameterized across both backends. The "raises for markdown
  until Phase 3" guard test was inverted into a positive
  "returns markdown when opted in" check.

### NOT in scope (Phase 4+)

- `kb_migrate` MCP tool + `services/migration.py` + read-fallback
  semantics — Phase 4. Until that ships, switching `AGENT_KB_BACKEND`
  does NOT copy data between backends; a KB ingested under chromadb
  is not queryable via markdown without re-ingesting from raw sources.
- Out-of-band LLM page-extraction (the spec's
  `status='pending_extraction'` path). Phase 3 MVP is synchronous,
  no LLM — `index()` blocks until the page is written and the body
  is a deterministic transformer of the raw content (file body
  verbatim; website → trafilatura plain text; api_endpoint payload →
  fenced code block). Documented as a future enhancement in
  `docs/markdown_backend.md`.
- Embedding default-flip + `embedder_version` stamping — Phase 5
  (PRECEDED by Phase 4 per non-negotiable phase ordering).
- Decorator order swap in `server.py` — explicitly NOT changed (M-01).
- Server.py / decorator changes of any kind — Phase 3 touches NO
  MCP tool definitions and adds NO new MCP tools. The 25-tool frozen
  surface is preserved bit-for-bit.

## 0.8.1 — 2026-04-24

> Phase 2 of the v2.1 redesign — RetrieverBackend abstraction with the
> chromadb backend wired as the default. **Zero behavior change** for
> existing v0.6.0 KBs; the full v0.6.0 test suite passes unchanged.
>
> spec_id: `70ab2170-381a-4657-bcd1-28a40c6f369b`
> Source spec: `pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json`

### Added

- New `RetrieverBackend` Protocol and `get_backend(settings)` factory in
  `src/agent_knowledgebase/backends/__init__.py`. The Protocol defines
  the seven retrieval entry points (`index`, `query`, `search`, `delete`,
  `info`, `count`, `health_check`) every backend must satisfy.
- New `ChromadbBackend` in `src/agent_knowledgebase/backends/chromadb_backend.py`
  — a thin wrapper that delegates to the existing `QueryOrchestrator` /
  `VectorStore` / `WikiManager` / `Database` so the v0.6.0 fast-path is
  preserved bit-for-bit. The wrapper holds a back-reference to the owning
  `KnowledgebaseService` so per-KB lifecycle (`_KBContext`, lazy
  embedder, vectorstore cache, per-`kb_id` lock) stays in one place.
- New `Settings.kb_backend: Literal['chromadb','markdown','lightrag']`
  field (default `chromadb`), accepting both `AGENT_KB_BACKEND` (the
  spec name) and `AGENT_KB_KB_BACKEND` (the conventional pydantic-settings
  prefix form) as env aliases. Disambiguated from the existing
  `Settings.vectorstore` field which remains the chromadb-internal
  vector-store-provider choice in `{chromadb, pinecone}`.
- `tests/test_retriever_backend_protocol.py` — Protocol-conformance tests
  for every concrete backend (currently chromadb), parameterised so
  Phase 3's markdown backend slots in mechanically.
- `tests/contract/test_probe4_response_shape.py` — pins the probe-4
  contract (`source_type`, `uri`, `dedup_key`, `page_id`,
  `dominant_embedding_model`) across `kb_info` / `kb_list_pages` /
  `kb_list_sources` AND on the new `RetrieverBackend.info()` directly.
- `tests/contract/test_stderr_schema.py` — captures every emission
  during a representative `kb_ingest_batch` and asserts each line carries
  exactly the 11 required fields (no extras, no omissions).

### Changed

- `KnowledgebaseService.__init__` now calls `get_backend(self._config,
  service=self)` ONCE at construction time and stores the result as
  `self._backend`. All retrieval entry points (`query`, `search`, the
  ingest write path's chunk-add step, and the source-deletion path)
  now route through that instance instead of touching the chromadb
  helpers directly. With `kb_backend='chromadb'` (default) the runtime
  behavior is bit-for-bit identical to v0.8.0.
- `kb_backend` is exposed via `kb_config_get` / `kb_config_set` /
  `kb_config_show` (added to `DOT_TO_FLAT` in `config_files.py`).

### Coordination

- **data-etl-orchestrator >= v0.4.0 must accept
  `dominant_embedding_model = null`** before Phase 3 ships the markdown
  backend. The chromadb backend continues to populate this field once the
  KB has at least one ingested chunk; the markdown backend will always
  return `null` since it does not embed. The empty-KB branch of the
  Phase 2 chromadb behavior already exercises the `null` shape, so
  consumer-side support can land any time before Phase 3.

### NOT in scope (Phase 3+)

- `backends/markdown_backend.py` — Phase 3 deliverable.
- `backends/lightrag_backend.py` — Phase 6 stub.
- `services/migration.py` + `kb_migrate` MCP tool — Phase 4.
- Embedding default flip + `embedder_version` stamping — Phase 5.
- Decorator order swap — explicitly NOT changed (M-01).

## 0.8.0 — 2026-04-24

> Phase 1 of the v2.1 redesign — Pattern A single-launcher runtime
> bootstrap. `uv` is no longer a runtime dependency; any system
> Python>=3.11 on PATH is sufficient.
>
> spec_id: `70ab2170-381a-4657-bcd1-28a40c6f369b`
> Source spec: `pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json`

### Changed (breaking)

- **`.mcp.json` no longer invokes `uv`.** The new manifest invokes the
  stdlib-only Pattern A launcher directly:

  ```json
  {
    "mcpServers": {
      "agent-knowledgebase": {
        "command": "python",
        "args": ["${CLAUDE_PLUGIN_ROOT}/bin/run_server.py"]
      }
    }
  }
  ```

  Hosts that previously installed `uv` purely to run this plugin can
  uninstall it. Any system Python>=3.11 on PATH boots the server.

### Added

- **`bin/run_server.py`** — single-file cross-platform launcher. On
  first launch it self-bootstraps `${CLAUDE_PLUGIN_ROOT}/.venv` via
  the stdlib `venv` module, runs `pip install -r requirements.lock`
  inside the venv with a 30 s timeout, persists a SHA256 sentinel at
  `<venv>/.req-sha`, and `os.execv`s the server. Subsequent launches
  match the sentinel and skip pip install (fast path). Cross-process
  bootstrap serialisation uses stdlib `fcntl.flock` on Unix and
  `msvcrt.locking` on Windows — no dependency on the `filelock`
  package, which is not yet installed when the launcher first runs.
- **Three structured stderr error tokens** when bootstrap can't
  proceed: `python_version` (Python<3.11 detected),
  `network_unreachable` (pip install timed out at 30 s), and
  `read_only_filesystem` (`${CLAUDE_PLUGIN_ROOT}` not writable).
  Each emission is a single-line JSON document carrying the spec_id
  for traceability.
- **`AGENT_KB_VENDORED_DEPS=/path/to/wheels`** offline escape-hatch.
  When set, the launcher uses `pip install --no-index --find-links`
  so air-gapped corporate hosts can preload wheels via
  `pip download -d wheels/ -r requirements.lock` and bootstrap with
  no network access.
- **Cygwin / Git-Bash detection** in the launcher
  (`sys.platform=='win32'` AND `MSYSTEM` set) so a forward-slash
  `${CLAUDE_PLUGIN_ROOT}` from Git-Bash is normalised to a Windows
  path before subprocess invocation.
- **`requirements.lock`** — fully-pinned hashed lock file produced by
  `uv pip compile pyproject.toml --generate-hashes`. The launcher
  hashes this file and compares against the venv sentinel to decide
  whether to re-run pip install.
- **`[project.scripts] agent-knowledgebase-server =
  "agent_knowledgebase.server:main"`** so users who install the
  package via `pipx` or `pip` get a CLI entry point alongside the
  plugin.
- **`tests/test_run_server_bootstrap.py`** — fresh
  `CLAUDE_PLUGIN_ROOT` triggers venv create + sentinel write; a
  second invocation skips pip install; `AGENT_KB_VENDORED_DEPS`
  switches the install argv to `--no-index --find-links`.
- **`tests/test_run_server_failure_modes.py`** — patches the three
  failure conditions and asserts the structured stderr token plus
  `SystemExit(1)`.

### Removed

- **`requirements.lock.sample`** — replaced by the real generated
  `requirements.lock` referenced above.

### Documented as alternates (not implemented)

- **Pattern C (`pipx install agent-knowledgebase`)** — documented in
  README with the explicit caveat that the Claude Code plugin
  marketplace does not auto-run `pipx install`; the user must run it
  manually.
- **Pattern D (Docker GHCR image)** — mentioned in README as a
  zero-Python-on-host alternate. No Dockerfile in this phase
  (Phase 6 deliverable).

### Frozen contracts (validated unchanged in this phase)

- The 25 `kb_*` MCP tools (`tests/test_redesign_contract.py`,
  `tests/test_tool_timeout.py`, `tests/contract/test_decorator_order.py`).
- `knowledgebase_stderr_log` 11-field schema.
- Decorator order: `@mcp.tool()` outer, `@_with_tool_timeout` inner
  (per `docs/redesign/MISTAKES.md` M-01 — DO NOT swap).
- `AGENT_KB_SAVES_DIR` semantics + the Phase 0 default-factory
  fallback to `~/.agent-kb/saves`.

## 0.7.1 — 2026-04-24

> Note: pyproject.toml was bumped 0.6.0 → 0.7.1 (skipping 0.7.0). The
> 0.7.0 CHANGELOG entry was authored ahead of pyproject; this release
> reconciles them.

> Phase 0 of the v2.1 redesign — direct bug fixes shipped under the
> existing chromadb backend with zero architectural change.
>
> spec_id: `70ab2170-381a-4657-bcd1-28a40c6f369b`
> Source spec: `pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json`

### Fixed

- **Bug-1 (embedder dimension probing):** added `probe_dimension()` to the
  `Embedder` protocol and to all three concrete implementations.
  `RemoteEmbedder.dimension` no longer silently returns the hardcoded
  `1536` fallback for unknown models (e.g. `qwen3-embedding:8b` which is
  actually 4096-dim) — it consults `_REMOTE_EMBEDDING_DIMENSIONS` for the
  three known OpenAI models and otherwise issues a single dry-run embed
  of the literal string `"probe"` and caches the result.
  `OllamaEmbedder.dimension` no longer returns `0` until the first
  embed; accessing the property triggers a probe and caches the dimension.
  Both behaviours had previously guaranteed a dimension mismatch when a
  caller sized a ChromaDB collection from `.dimension` before any embed
  had run.
- **Bug-1c (HTTP-status classification on `EmbedderUnavailableError`):**
  both `RemoteEmbedder._call_embed` and `OllamaEmbedder._call_embed` now
  classify `httpx.HTTPStatusError` into structured payloads —
  `404 → {"error":"model_not_pulled"}`,
  `401/403 → {"error":"auth_failed"}`,
  `429 → {"error":"rate_limited", "retry_after":...}` — instead of
  letting the raw `HTTPStatusError` propagate as an opaque MCP error.
  Existing timeout / network classification is preserved.
- **Bug-2 (config error surfacing):** `_get_service` in `server.py` now
  catches `pydantic.ValidationError`, `FileNotFoundError`, and
  `NotADirectoryError` from `Settings().resolve_paths()` and returns a
  sentinel `_ConfigMissingService` whose every `kb_*` method returns the
  JSON string `{"error":"config_missing", "missing":"<env_var>",
  "detail":"..."}`. `main()` performs an identical precheck at startup
  and emits the same structured payload on stderr before
  `sys.exit(1)`, so misconfigured installs fail immediately with a
  clear message instead of opaquely on the first MCP call.

### Added

- `EmbedderDimensionMismatchError` in `services/embeddings.py` for
  surfacing collection-vs-embedder dimension drift; carries
  `expected` / `actual` / `model` / `collection` and a `to_payload()`
  helper for structured responses. Used by the new
  `tests/test_chromadb_dimension_mismatch.py` regression test that
  pins Bug-1a's resolution at the embedder layer.

### Changed

- `Settings.saves_dir` now has a `default_factory` returning
  `~/.agent-kb/saves`, so a fresh install boots with zero env vars set.
  `resolve_paths()` still raises `FileNotFoundError` /
  `NotADirectoryError` when the directory is missing or not a
  directory — that behaviour is preserved and is what the new
  `_get_service` catch turns into the structured payload.

### Decorator-order note (preserved, NOT changed)

The v2.0 spec draft asserted Bug-2 was a `@mcp.tool()` /
`@_with_tool_timeout` decorator NO-OP. Empirical verification —
`tests/test_tool_timeout.py` 6/6 passing including
`test_every_registered_mcp_tool_is_wrapped`, and a live timeout test
returning the structured `tool_timeout` payload — proves the current
order (`@mcp.tool()` outer, `@_with_tool_timeout` inner) is the
working order. **The decorators were NOT swapped.** A new
`tests/contract/test_decorator_order.py` is added purely as a
regression guard against accidental future inversion (asserts
`__wrapped__` exists on every registered tool). See
`docs/redesign/MISTAKES.md` M-01.

## [unreleased] — Pipeline-driven redesign (v2.1 spec)

> Forecast only — proposed changes for Phase 1+ (Pattern A launcher,
> RetrieverBackend abstraction, MarkdownWikiBackend, migration
> tooling, embedding default-flip, LightRAGBackend stub).
>
> spec_id: `70ab2170-381a-4657-bcd1-28a40c6f369b`
> Source spec: `pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json`
> Detailed roadmap: `docs/redesign/ROADMAP.md`

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
