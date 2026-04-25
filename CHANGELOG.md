# Changelog

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
