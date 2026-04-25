# Markdown Wiki Backend (Phase 3, OPT-IN)

> spec_id: `70ab2170-381a-4657-bcd1-28a40c6f369b`
> Source spec: `pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json`
> Phase: 3 (Phase 3 — MarkdownWikiBackend)
> Status: opt-in. **Default backend remains `chromadb`.**

The markdown wiki backend stores knowledge as Karpathy-style
interlinked markdown files on disk
(`docs/karpathy-llm-wiki.md`) instead of as chromadb vectors.
It implements the same `RetrieverBackend` Protocol the rest of
agent-knowledgebase routes through, so opting in is a single
environment-variable flip.

---

## How to opt in

Set the env var before launching the server:

```bash
export AGENT_KB_BACKEND=markdown
python bin/run_server.py
```

Or add it to your project / user `config.json`:

```json
{
  "kb_backend": "markdown"
}
```

`AGENT_KB_BACKEND=chromadb` (the default) restores the v0.6.0
behavior. The flip is **per-process** — per-KB overrides arrive in
Phase 4 via `AGENT_KB_BACKEND_PER_KB`.

---

## On-disk layout

```
<saves_dir>/<kb-dir>/wiki/
  raw/
    <source_id>.<ext>           # immutable copy of the raw source
  pages/
    <slug>.md                   # one markdown file per ingested article
  index.md                      # content-oriented catalog (flat form ≤200 pages)
  index/                        # appears once page count > 200 (sharded form)
    <category>.md               # one shard per source_type
  log.md                        # chronological op log (append-only)
  hot.md                        # ~500-char hot cache (placeholder, future)
```

Each `pages/<slug>.md` carries a JSON-frontmatter header so the
backend can round-trip metadata back into `info()` / `search()`
results without an extra index file:

```markdown
---
{
  "page_id": "hello-world-a1b2c3",
  "source_id": "src-hello-world",
  "source_type": "file",
  "uri": "/tmp/hello.txt",
  "dedup_key": "hello-key",
  "title": "Hello World",
  "created_at": "2026-04-24T19:32:18.123456+00:00",
  "spec_id": "70ab2170-381a-4657-bcd1-28a40c6f369b"
}
---
# /tmp/hello.txt

The quick brown fox...
```

The slug is `<title>-<sha256[:6]>` — deterministic, lower-case,
alphanumeric-with-dashes, capped at 80 characters with a 6-character
content hash suffix to avoid collisions on similar titles.

---

## Allow-list rationale

The wiki abstraction is **not** a universal replacement for the
chromadb backend. Wiki pages are designed to be human-readable
markdown summaries of one logical source — an article, a web page, a
small JSON response. Source types whose payloads either don't fit
that mould (a 50-row SQL slice is not a "page") or don't fit on disk
as a single readable file (a multi-GB codebase) raise
`KB_INGESTOR_UNSUITABLE` so callers don't accidentally produce
unreadable garbage.

The list is **positive-allow** (validation finding f-15) — adding a
new source_type requires explicit opt-in here, not a deny-list
update:

| `source_type`         | Allow? | Why |
|-----------------------|--------|-----|
| `file`                | YES   | One file → one wiki page is the canonical case. |
| `website`             | YES   | Trafilatura plain text → one wiki page. |
| `api_endpoint` < 2 MB | YES   | Small JSON / text response renders cleanly inside a fenced code block. |
| `api_endpoint` ≥ 2 MB | NO    | Per-page LLM extraction cost grows linearly with payload; 2 MB is the empirical break-even. |
| `sql_database`        | NO    | A 50-row slice is not a "page" — chromadb embeds it; markdown can't summarise it. |
| `codebase`            | NO    | Multi-file / multi-language; one giant wiki page would be unreadable. |
| `git_history`         | NO    | Time-series, not document-shaped; no obvious page identity. |
| `directory`           | NO    | Same problem as `codebase` at smaller scale. |

### Force-flag override

Set `AGENT_KB_FORCE_WIKI_INGEST=1` to bypass the check. Ingest then
proceeds and a structured `knowledgebase_stderr_log` warning is
emitted with `error_code=KB_INGESTOR_UNSUITABLE_FORCED`, so operators
can grep for forced bypasses:

```json
{
  "kb_id": "...",
  "op": "markdown_backend_force_ingest",
  "phase": "markdown_ingest",
  "elapsed_ms": 0,
  "rows_in": 1, "rows_ok": 0, "rows_skipped": 0, "rows_failed": 0,
  "dedup_policy": "n/a",
  "request_id": null, "tool_caller_version": null,
  "error_code": "KB_INGESTOR_UNSUITABLE_FORCED",
  "error_message": "source_type='codebase': ... — proceeding because AGENT_KB_FORCE_WIKI_INGEST=1"
}
```

---

## Cost model (per source_type, MVP)

> Methodology: rough order-of-magnitude estimates assuming a future
> LLM-page-extraction pass at ~$0.50/MTok input, ~$1.50/MTok output
> (current Anthropic Claude Haiku tier as of 2026-04). Numbers below
> are FUTURE — Phase 3 MVP is **synchronous, no LLM** (raw content →
> deterministic page body), so today's per-ingest LLM cost is **$0.00**.
> The table below is the planning surface for the deferred out-of-band
> LLM pass and the rationale for the allow-list.

| source_type     | Typical payload | Est. input tokens | Est. output tokens | Est. LLM cost / page | Notes |
|-----------------|-----------------|-------------------|--------------------|----------------------|-------|
| `file`          | 1 KB – 100 KB   | 1–25 k            | 200–800            | $0.001 – $0.014      | Sweet-spot — most ingests fall here. |
| `website`       | 5 KB – 50 KB    | 1–13 k            | 300–1 k            | $0.001 – $0.008      | Trafilatura strips boilerplate first. |
| `api_endpoint`<2MB | 100 B – 2 MB | 0.025 k – 500 k   | 200–2 k            | $0.000 – $0.253      | Long-tail dominated by the 2 MB ceiling. |
| `api_endpoint`≥2MB | 2 MB+        | 500 k+            | 2 k+               | $0.253+              | **BLOCKED** — past the break-even. |
| `sql_database`  | n/a (rows)      | n/a               | n/a                | n/a                  | **BLOCKED** — wrong abstraction; use chromadb. |
| `codebase`      | 10 MB – 1 GB    | 2 M – 250 M       | 5 k – 50 k         | $1.27 – $159         | **BLOCKED** — far past break-even. |
| `git_history`   | 1 MB – 100 MB   | 250 k – 25 M      | 1 k – 10 k         | $0.13 – $15.9        | **BLOCKED** — time-series, no page identity. |
| `directory`     | 100 KB – 100 MB | 25 k – 25 M       | 1 k – 10 k         | $0.013 – $15.9       | **BLOCKED** — multi-file, no single page. |

Re-measure when:

- The per-token LLM price changes by more than 2x.
- The deferred LLM page-extraction lands (Phase 3+, see Limitations).
- A new source_type joins the allow-list.

---

## Sharded index

For small KBs (≤ 200 pages) `index.md` is a single flat catalog:

```markdown
# Wiki Index

_Pages: 12 | spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b_

- [Article 001](pages/article-001-a1b2c3.md) `file` — /tmp/article-001.txt
- [Article 002](pages/article-002-d4e5f6.md) `website` — https://example.com/002
- ...
```

Once the page count crosses 200 the next ingest migrates the index
into a directory pointer:

```markdown
# Wiki Index (sharded)

_Pages: 250 | shards: 2 | spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b_

## Categories

- [file](index/file.md) (125 pages)
- [website](index/website.md) (125 pages)
```

Each shard `index/<source_type>.md` lists only its own bucket's pages.
Categories derive from `source_type` for v0 (per
spec.phases[3].tasks[4]) — refining the categorisation (e.g. via LLM
topic clustering) is future work.

The transition is **idempotent**: ingest above 200 always re-shards;
the flat form returns automatically if a delete drops the count back
under the threshold.

---

## Querying

Both `RetrieverBackend.query()` (the vector path) and
`RetrieverBackend.search()` (the keyword path) route through a
**sqlite FTS5 in-memory index** built lazily over `pages/*.md` on
first read after ingest / delete.

- `query()` re-routes to `search()` (markdown has no embedding path)
  and surfaces scores in `[0, 1]` descending via
  `score = 1 / (1 + abs(rank))`.
- `search()` accepts an optional `filters` dict; only the keys
  `source_type` and `slug` are honored. Other keys are silently
  ignored (markdown has no general filter pushdown).
- Empty / malformed queries return `[]` instead of raising — the
  MCP-level `kb_query` / `kb_search` tools already render the empty
  result correctly.

The FTS5 connection is per-`kb_id` and **invalidated on every
`index()` / `delete()` call**, so the next read rebuilds. There is
no on-disk persistence of the FTS5 index — the markdown files
themselves are the source of truth.

Connections are built with `check_same_thread=False` and all access
to the per-`kb_id` cache and to each cached connection's sqlite
operations is serialized through a `threading.Lock` (`_fts_lock`),
so concurrent `search()` / `index()` / `delete()` calls on the same
`kb_id` from different threads behave correctly — see the module
docstring `Thread-safe:` line.

---

## Probe-4 contract

`info(kb_id=...)` returns AT LEAST the v0.6.0 frozen keys:

| Key                          | Markdown-backend value |
|------------------------------|------------------------|
| `source_type`                | First page's `source_type` (or `None` on empty KB) |
| `uri`                        | First page's `uri` (or `None`) |
| `dedup_key`                  | First page's `dedup_key` (or `None`) |
| `page_id`                    | First page's `page_id` (or `None`) |
| `dominant_embedding_model`   | **Always `None`** (markdown does not embed) |

`None` here means **present with value `None`** — never omitted
(validation finding f-20).

> **"First page" rule.** "First page" here means the
> **slug-alphabetical-first** page in `pages/` (the first entry
> returned by walking `pages/*.md` sorted ascending). This is
> deterministic but arbitrary — not the first by ingest time, not
> the most queried. Operators wanting KB-level summary statistics
> (counts by source_type, recent timestamps, etc.) should wait for
> the future Phase 4 `kb_info` aggregates; today's `info()` shape is
> pinned to the probe-4 contract for backward compatibility.

Backend-diagnostic fields like `vector_count` / `embedding_provider`
likewise surface as `None`.

---

## Limitations (MVP)

The Phase 3 MVP is **deliberately small**. Future enhancements:

1. **Synchronous ingest only.** The spec line about "out-of-band LLM
   page-extraction / `status=pending_extraction`" is **deferred**.
   For now `index()` blocks until the page is written, no LLM is
   invoked, and the page body is a deterministic transformer of the
   raw content (file body verbatim; website → trafilatura plain text;
   api_endpoint payload → fenced code block). This keeps Phase 3
   scope contained and avoids introducing an LLM-call dependency
   into the write path before the migration tooling (Phase 4) lands.
2. **No vector queries.** `query()` re-routes to `search()`. If your
   workload relies on semantic similarity, stay on the chromadb
   backend until LightRAG (Phase 6) lands.
3. **No incremental update.** Re-ingesting a `source_id` that
   already exists creates a new page rather than updating the
   existing one. Use `delete(source_id=...)` first to replace.
   `KnowledgebaseService.update_source` already routes through
   `delete + index` so the round-trip works at the service level.
4. **No `filters` pushdown beyond `source_type` / `slug`.** The
   chromadb backend's `where=` machinery is not yet mirrored.
5. **In-process threading is safe; cross-process is not.** The
   per-`kb_id` `threading.Lock` from `KnowledgebaseService` and the
   backend's own `_fts_lock` together serialize concurrent
   `index()` / `search()` / `delete()` calls on the same `kb_id`
   inside one Python process — including the MCP tool-timeout worker
   thread that may invoke a tool from a different thread than the
   one that built the FTS5 in-memory cache. Multi-process
   deployments still need the `filelock` / `flock` recipe documented
   in `docs/cross-process-lock-recipe.md`; the in-memory FTS5 cache
   is per-process and offers no cross-process coherence.
6. **Categories = `source_type` for v0.** Sharded index categorisation
   is a placeholder; LLM topic clustering or tag-based bucketing is
   future work.
7. **Ingest is not transactional across files.** `_write_one` writes
   raw → page → log as three separate calls and a multi-document
   `index()` call writes those triples per-document, with no
   cross-doc transaction. Each individual file write is per-file
   atomic via `tempfile + os.replace`, so a crash mid-write never
   produces a half-written file; but a crash mid-batch can leave
   the wiki with an orphan raw file (crash between raw and page),
   an unlogged page (crash between page and log), or a partially
   ingested batch (crash between docs N and N+1). Recovery: rerun
   `kb_ingest` / `kb_ingest_batch` for the same source_id(s) — page
   writes are idempotent overwrites and the duplicated raw is
   harmless.

---

## Migration

Until Phase 4 ships `kb_migrate`, switching `AGENT_KB_BACKEND` does
**not** copy data between the two backends — a KB ingested under
chromadb cannot be queried via the markdown backend (and vice versa)
without re-ingesting from raw sources. Phase 4 introduces an explicit
`kb_migrate(kb_id, target_backend)` MCP tool plus read-fallback
semantics (markdown-flag set but `wiki/` missing → silently fall
back to chromadb).

---

## See also

- `docs/karpathy-llm-wiki.md` — the design pattern this backend implements.
- `docs/redesign/AGENTS.md` §1.2 — probe-4 contract preservation.
- `docs/redesign/MISTAKES.md` M-01 — decorator-order pitfall (do NOT swap).
- `docs/redesign/ROADMAP.md` Phase 3 — files-to-touch and verification.
- `pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json` lines 270-290 — Phase 3 spec block.
