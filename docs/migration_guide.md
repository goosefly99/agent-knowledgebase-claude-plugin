# Migration Guide — chromadb <-> markdown wiki

> spec_id: `70ab2170-381a-4657-bcd1-28a40c6f369b`
> Source spec: `pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json`
> Phase 4 deliverable. **Phase 4 ships BEFORE Phase 5** so existing
> v0.6.0 KBs ingested under the Ollama default stay queryable after
> the embedding default-flip.

This guide covers the Phase 4 migration tooling: how to move a
knowledgebase between the v0.6.0 chromadb backend and the v0.9.0
opt-in markdown-wiki backend, what the routing semantics are, why the
per-page provider snapshot exists, and how to roll back.

---

## When to migrate

The Phase 3 markdown-wiki backend is **opt-in** and not appropriate
for every KB:

| Use case | Recommended backend |
|----------|---------------------|
| Vector similarity search over chunks | `chromadb` (default) |
| Karpathy-style human-readable wiki, FTS-only | `markdown` |
| Mixed workload (vector + browseable wiki) | Stay on `chromadb`; export to markdown for read-only browsing |
| Large KBs (>50k chunks) | `chromadb` (markdown FTS5 sharding kicks in above 200 pages but vector recall remains the strength) |
| Strict allow-list source_types: `{file, website, api_endpoint <2MB}` | `markdown` is fine |
| `sql_database`, `codebase`, `git_history`, `directory`, large `api_endpoint` | `chromadb` (markdown will reject these unless `AGENT_KB_FORCE_WIKI_INGEST=1`) |

`chromadb` remains the **default**. Phase 4 does NOT change the
default; it only adds the migration path so operators who want
markdown can move data in either direction without re-ingesting from
raw sources.

---

## How to call `kb_migrate`

`kb_migrate` is a new MCP tool added in v0.10.0 (Phase 4):

```python
# Via the MCP tool layer:
kb_migrate(kb_id="<kb-uuid>", target_backend="markdown")
# -> JSON: {"kb_id":..., "target_backend":"markdown",
#           "pages_migrated":N, "sentinel_path":"...",
#           "notes":[...], "spec_id":"70ab21..."}
```

Behavior:

1. The kb's existing data is read from the **current** backend
   (chromadb by default).
2. The new backend's on-disk layout is materialised under the
   existing `<saves_dir>/<kb-name>/` directory:
   - `target_backend="markdown"` writes `wiki/raw/`, `wiki/pages/`,
     `wiki/index.md` (or `wiki/index/<category>.md` shards above 200
     pages), `wiki/log.md`.
   - `target_backend="chromadb"` re-ingests `wiki/pages/*.md` back
     through the standard ingest pipeline (re-embeds via the configured
     embedder, re-inserts into chromadb).
3. The routing sentinel file is written:
   `<saves_dir>/<kb-name>/.migrated_to` containing the target backend
   name (e.g. `markdown`).
4. The service's per-KB backend cache is invalidated so the next
   `kb_query` / `kb_search` routes through the new backend.

`kb_migrate` raises `ValueError` for an unknown `kb_id` or for an
invalid `target_backend` (only `"chromadb"` and `"markdown"` are
accepted; `"lightrag"` is deferred to Phase 6).

---

## Sentinel file semantics

The file `<saves_dir>/<kb-name>/.migrated_to` is the **routing
signal**, not a delete operation. After `kb_migrate` succeeds:

- The OLD backend's data **remains on disk**.
- The sentinel file's content (single line, no newline-trim semantics)
  is one of `chromadb` or `markdown`.
- Future `kb_query` / `kb_search` calls for that `kb_id` resolve the
  backend by reading the sentinel **before** consulting any other
  config layer.
- To **roll back** a migration, delete the sentinel file:

  ```bash
  rm <saves_dir>/<kb-name>/.migrated_to
  ```

  Routing reverts to `kb_backend_per_kb[kb_id]` if set, or the global
  `Settings.kb_backend` otherwise.
- To **reclaim disk space** after a migration, manually delete the
  old backend's directory (`chroma/` for the chromadb data,
  `wiki/` for the markdown tree). This is intentionally manual so the
  migration is round-trippable until the operator commits.

---

## Per-KB routing precedence

Phase 4 introduces three routing layers. Resolution at backend-pick
time follows this strict precedence (highest first):

1. **`<saves_dir>/<kb-name>/.migrated_to` sentinel** — if the file
   exists with a recognized backend name, that wins.
2. **`Settings.kb_backend_per_kb[kb_id]`** — env-driven mapping
   accepting either the comma-separated form
   `kb_id1=markdown,kb_id2=chromadb` (via `AGENT_KB_BACKEND_PER_KB`)
   or a JSON object via the user/project config files.
3. **`Settings.kb_backend`** — the global default
   (`AGENT_KB_BACKEND`), which is `chromadb` unless explicitly
   overridden.

Example env setup running mixed backends in one process:

```bash
export AGENT_KB_BACKEND=chromadb           # global default
export AGENT_KB_BACKEND_PER_KB="kb-research=markdown,kb-code=chromadb"
```

The mapping is keyed on the **kb_id** (uuid), not the human-readable
KB name — so listing KBs in `Settings.kb_backend_per_kb` requires
knowing the ids returned by `kb_create` / `kb_list`.

---

## Read-fallback behavior

When `Settings.kb_backend='markdown'` is configured **globally** but
a particular kb has only chromadb data on disk
(`<kb-name>/wiki/` is missing while `<kb-name>/chroma/` exists), the
markdown backend's `query()` / `search()` methods silently delegate to
ChromadbBackend for that one call. The fallback is reported via a
structured `knowledgebase_stderr_log` line:

```json
{
  "kb_id": "<id>",
  "op": "kb_query",
  "phase": "read_fallback",
  "elapsed_ms": 0,
  "rows_in": 0, "rows_ok": 0, "rows_skipped": 0, "rows_failed": 0,
  "dedup_policy": "n/a",
  "request_id": null,
  "tool_caller_version": null,
  "error_code": "MARKDOWN_READ_FALLBACK_TO_CHROMADB",
  "error_message": "kb_backend=markdown but wiki/ missing for kb_id=...; serving query via chromadb fallback. Run kb_migrate(kb_id=..., target_backend='markdown') to materialise the wiki/ layout."
}
```

This means an operator can flip
`AGENT_KB_BACKEND=markdown` globally and still keep existing KBs
queryable while gradually running `kb_migrate` against each one. No
silent breakage at flip-time — the worst case is a logged read-
fallback, not an empty result set or a stack trace.

The fallback also recognizes the `vector_store/` alias in addition
to `chroma/` for forward compatibility with potential future
directory renaming.

---

## Embedder snapshot rationale (why Phase 5 is safe)

Phase 4's most important deliverable is the **per-page (per-chunk)
provider snapshot** stored in two new columns on the `chunks` table:

```sql
ALTER TABLE chunks ADD COLUMN embedding_provider TEXT;
ALTER TABLE chunks ADD COLUMN embed_base_url TEXT;
```

Every chunk inserted on or after v0.10.0 carries the provider name
(`ollama` / `remote` / `sentence-transformers`) and the base URL it
was embedded against, alongside the existing `metadata.embedding_model`
field. The snapshot is read at query time by
`KnowledgebaseService._query_embedder_for(kb_id)` which calls

```python
create_embedder_for_model(
    config,
    dominant_model,
    provider=dominant_provider,
    base_url=dominant_base_url,
)
```

passing the snapshot's `(model, provider, base_url)` triple.

This means: even after the Phase 5 default-flip swaps
`Settings.embedding_provider='ollama'` to
`Settings.embedding_provider='remote'`, an existing v0.6.0 KB ingested
under Ollama+`qwen3-embedding:8b` (4096-dim) is rebuilt with the
**original** Ollama embedder for retrieval — its 4096-dim vectors are
not silently mismatched against a freshly-defaulted 1536-dim
`text-embedding-3-small`. The Phase 4 -> Phase 5 ordering exists
precisely because of this dependency.

### Backfilling legacy KBs

For KBs ingested **before** v0.10.0 (where chunks have NULL columns
in `embedding_provider` / `embed_base_url`), use the backfill helper:

```python
from agent_knowledgebase.services.migration import backfill_provider_snapshot
from agent_knowledgebase.services.knowledgebase import KnowledgebaseService
from agent_knowledgebase.config import Settings

svc = KnowledgebaseService(Settings().resolve_paths())
# Dry-run across every KB:
backfill_provider_snapshot(svc, dry_run=True)
# {kb_id1: 1234, kb_id2: 567, ...}  (rows that would be touched)

# Commit:
backfill_provider_snapshot(svc)
```

Rules:

- The backfill reads `Settings.embedding_provider` /
  `Settings.embed_base_url` (the process-wide values) and stamps them
  onto every chunk row that currently has NULL columns.
- Existing non-null values are **never overwritten** — the SQL uses
  `COALESCE(embedding_provider, ?)` so a previously-stamped chunk is
  left alone.
- The backfill is idempotent: re-running on a fully-stamped KB is a
  no-op (returns 0 rows updated).
- **Phase 5 will not merge** until the backfill has been run against
  every existing KB. Run the dry-run first to confirm coverage.

---

## Operator runbook: chromadb -> markdown migration

```bash
# 1. Stop ingest for the target KB (no enforcement; convention).
# 2. Confirm the per-page snapshot is populated:
python -c "
from agent_knowledgebase.services.knowledgebase import KnowledgebaseService
from agent_knowledgebase.services.migration import backfill_provider_snapshot
from agent_knowledgebase.config import Settings
svc = KnowledgebaseService(Settings().resolve_paths())
print(backfill_provider_snapshot(svc, dry_run=True))
"

# 3. Backfill if any KB shows non-zero rows:
python -c "
from agent_knowledgebase.services.knowledgebase import KnowledgebaseService
from agent_knowledgebase.services.migration import backfill_provider_snapshot
from agent_knowledgebase.config import Settings
svc = KnowledgebaseService(Settings().resolve_paths())
print(backfill_provider_snapshot(svc))
"

# 4. Run the migration via the MCP tool (any MCP-aware client):
#    kb_migrate(kb_id="<kb-uuid>", target_backend="markdown")
#
#    Or programmatically:
python -c "
from agent_knowledgebase.services.knowledgebase import KnowledgebaseService
from agent_knowledgebase.config import Settings
svc = KnowledgebaseService(Settings().resolve_paths())
print(svc.migrate(kb_id='<kb-uuid>', target_backend='markdown'))
"

# 5. Verify the wiki layout exists:
ls "$AGENT_KB_SAVES_DIR/<kb-name>/wiki/pages/"

# 6. Issue a query and confirm the result tags backend=markdown:
#    kb_query(kb_id="<kb-uuid>", text="...", top_k=5)
```

---

## Limitations

- **Migration is non-incremental.** A second run rebuilds the wiki
  tree from scratch (page writes are idempotent overwrites; the same
  source_id produces the same slug). For large KBs this is O(N) on
  source count.
- **markdown -> chromadb re-embeds.** The reverse direction
  re-ingests pages through the standard ingest pipeline, which means
  re-embedding via the currently-configured embedder. Plan for the
  cost.
- **No cross-process locking.** Running `kb_migrate` against the
  same `kb_id` from two processes simultaneously is unsupported (the
  built-in per-KB `threading.Lock` only serializes within one
  process). See `docs/cross-process-lock-recipe.md` for opt-in
  recipes.
- **Allow-list relaxation during export.** `export_to_markdown`
  temporarily sets `AGENT_KB_FORCE_WIKI_INGEST=1` so source_types
  outside the markdown allow-list (`sql_database`, `codebase`, etc.)
  can be migrated losslessly. The corresponding
  `KB_INGESTOR_UNSUITABLE_FORCED` stderr_log line will fire for each
  such source — operators should treat those as informational, not
  errors, during a migration window.
- **`lightrag` target unsupported.** `kb_migrate` rejects
  `target_backend='lightrag'` (Phase 6 deferred deliverable).
