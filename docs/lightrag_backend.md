# LightRAG Backend (Phase 6, DEFERRED — stub only)

> spec_id: `70ab2170-381a-4657-bcd1-28a40c6f369b`
> Source spec: `pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json`
> Phase: 6 (Phase 6 — LightRAGBackend, deferred, conditional)
> Status: **stub only**. Default backend remains `chromadb`. The
> `'lightrag'` branch of `get_backend()` constructs a
> `LightRAGBackend` instance, but every non-trivial method raises
> `NotImplementedError`. See "Activation checklist" below.

LightRAG (`HKUDS/LightRAG` — <https://github.com/HKUDS/LightRAG>) is a
graph-augmented retrieval system that combines an LLM-derived
knowledge graph with a vector index. The Phase 6 design intent is to
expose LightRAG as a drop-in `RetrieverBackend` so operators who
already run a LightRAG sidecar can route `kb_query` / `kb_search`
calls through it without rewriting any consumer code.

This document is the **design** for that integration. No code beyond
the stub class exists today.

---

## Why LightRAG was deferred

Phase 6 is conditional per the v2.1 spec
(`implementation.phases[6].description`):

> "OPTIONAL — implement only if user adoption signals demand. Forwards
> queries to localhost:9621 LightRAG REST API. Out of scope for v0.8.x
> — documented as future path."

Three reasons drove the deferral:

1. **No empirical demand.** No operator on the v0.8.x / v0.9.x /
   v0.10.x / v0.11.0 release cycle has asked for graph-augmented
   retrieval. The chromadb default + the markdown opt-in cover the
   currently-shipping use cases.
2. **Sidecar operational cost.** LightRAG runs as a separate process
   that holds an LLM client (for entity / relation extraction during
   ingest) and a graph store. The sidecar is non-trivial to operate
   compared to the in-process chromadb / sqlite-FTS5 paths.
3. **Surface preservation risk.** Implementing the forward path
   without a real adoption signal means maintaining test coverage,
   docs, and breakage detection for code nobody runs. The stub
   reserves the contract surface at near-zero ongoing cost.

**Empirical caveat:** LightRAG support is conditional on adoption
signal. If no operator opts in within 6 months of v0.11.0 GA, this
stub may be deleted to reduce maintenance surface. The deletion would
be coordinated via a deprecation note in the next release CHANGELOG
followed by removal in the release after.

---

## When to activate

The trigger conditions for promoting this stub from
`raise NotImplementedError` to a live backend:

- An operator (internal or external) commits to running a LightRAG
  sidecar in production AND has empirical evidence the
  graph-augmented retrieval surface measurably outperforms the
  chromadb hybrid path on their corpus.
- That operator can absorb the LightRAG sidecar's operational cost
  (LLM API quota for entity/relation extraction during ingest, plus
  the graph-store disk + memory footprint).
- The activation work is scoped to a single phase (not bundled with
  Phase 7+ deliverables) and gated on Phase 5's per-chunk
  `embedder_version` snapshot being in production — LightRAG's vector
  index uses its own embedder, so the snapshot ensures existing
  v0.6.0 / v0.10.x / v0.11.0 KBs stay queryable under their original
  embedder even if a per-KB cutover to `kb_backend='lightrag'` later
  flips the global default.

If those conditions aren't met within 6 months of v0.11.0 GA, the
stub is a candidate for removal (see "Empirical caveat" above).

---

## Architecture

LightRAG runs as a sidecar service. The agent-knowledgebase MCP
server holds a thin HTTP client that forwards each
`RetrieverBackend` Protocol method to the corresponding LightRAG
REST endpoint:

```
+-------------------------------+               +---------------------------+
| agent-knowledgebase MCP server|               |  LightRAG sidecar         |
|                               |   HTTP/JSON   |  (HKUDS/LightRAG image)   |
|  KnowledgebaseService         | ------------> |                           |
|    self._backend =            |   localhost   |  /insert  /query /delete  |
|      LightRAGBackend(...)     |   :9621       |  /info    /health         |
|                               | <------------ |                           |
+-------------------------------+               +---------------------------+
                                                              |
                                                              v
                                              +----------------------------+
                                              |  Persistent state          |
                                              |    graph store (e.g.       |
                                              |    NetworkX pickle / Neo4j)|
                                              |    vector store (faiss)    |
                                              |    LLM cache               |
                                              +----------------------------+
```

The sidecar holds **all** retrieval state. The agent-knowledgebase
process becomes a stateless HTTP forwarder for the
`kb_backend='lightrag'` KBs. Mixed-backend deployments (some KBs on
chromadb, some on lightrag) are supported via `Settings.kb_backend_per_kb`
or the `<saves_dir>/<kb-name>/.migrated_to=lightrag` sentinel
(Phase 4's per-KB routing primitive).

---

## Activation steps

The flip from "stub" to "live" is gated on the activation checklist
at the bottom of this document. Once the checklist is green, the
operator-facing activation is:

1. **Run the LightRAG sidecar** (see the docker-compose snippet
   below).
2. **Set `AGENT_KB_LIGHTRAG_URL`** to the sidecar's HTTP address
   (defaults to `http://localhost:9621` if unset).
3. **Set `AGENT_KB_BACKEND=lightrag`** to flip the global default,
   OR set `AGENT_KB_BACKEND_PER_KB='kb1=lightrag'` to opt in
   per-KB, OR run `kb_migrate(kb_id='kb1', target_backend='lightrag')`
   once the migration tooling supports the lightrag target (see
   "Migration path" below).
4. **Restart the MCP server** — backend selection happens at
   construction time so the env var change requires a restart.

`AGENT_KB_BACKEND=chromadb` (the default) restores the v0.6.0
behavior with no LightRAG involvement.

---

## docker-compose.yml integration for the LightRAG sidecar

A minimal production-shaped sidecar configuration:

```yaml
# docker-compose.yml
#
# Run alongside the agent-knowledgebase MCP server. The
# agent-knowledgebase process can stay outside docker (it talks to
# the sidecar over localhost) or be wrapped in its own service —
# both shapes work, the sidecar deployment is independent.

services:
  lightrag:
    # Pin to a specific tag in production. The :latest tag is shown
    # for brevity; HKUDS/LightRAG ships tagged release images on
    # GitHub Container Registry once activation is in production.
    image: ghcr.io/hkuds/lightrag:latest
    container_name: agent-kb-lightrag-sidecar
    restart: unless-stopped
    ports:
      # Default LightRAG REST port. agent-knowledgebase reads
      # AGENT_KB_LIGHTRAG_URL (default http://localhost:9621) so
      # remap freely if 9621 collides locally.
      - "9621:9621"
    volumes:
      # Persist the graph store, vector index, and LLM cache across
      # container restarts. Without this volume LightRAG re-indexes
      # from scratch on every restart, which is expensive.
      - ./lightrag_data:/app/data
      # Optional: mount a host-side LightRAG config file so operators
      # can pin chunk size, retrieval mode default, embedder choice,
      # etc. without rebuilding the image.
      - ./lightrag_config.yaml:/app/config.yaml:ro
    environment:
      # LightRAG needs an LLM API key for entity / relation
      # extraction during ingest. The agent-knowledgebase process
      # does NOT see this key — it stays inside the sidecar.
      # Use a .env file or docker secrets instead of inlining the
      # value here.
      OPENAI_API_KEY: ${OPENAI_API_KEY:?required for LightRAG ingest}
      # Embedding provider for LightRAG's internal vector index.
      # Independent of agent-knowledgebase's
      # AGENT_KB_EMBEDDING_PROVIDER setting — LightRAG's vector
      # index is self-contained.
      EMBEDDING_PROVIDER: openai
      EMBEDDING_MODEL: text-embedding-3-small
    healthcheck:
      # Lets docker-compose surface "unhealthy" before the
      # agent-knowledgebase server hits the sidecar with real
      # traffic. The /health endpoint is part of the REST contract
      # listed below.
      test: ["CMD", "wget", "-qO-", "http://localhost:9621/health"]
      interval: 30s
      timeout: 5s
      retries: 5
      start_period: 60s

  # Optional sibling service — the agent-knowledgebase MCP server
  # itself. Skip this section if you run the MCP server outside
  # docker (the typical Claude Code plugin invocation).
  agent-knowledgebase:
    build: .
    depends_on:
      lightrag:
        condition: service_healthy
    environment:
      AGENT_KB_BACKEND: lightrag
      AGENT_KB_LIGHTRAG_URL: http://lightrag:9621
      AGENT_KB_SAVES_DIR: /data/saves
    volumes:
      - ./kb_saves:/data/saves
```

Notes:

- The sidecar image is pinned to a specific tag in any real
  deployment. `:latest` is shown for brevity.
- The `./lightrag_data` host volume is the single piece of mutable
  state — back it up via the same path your other persistent
  volumes use.
- `OPENAI_API_KEY` is consumed by the sidecar, NOT by
  agent-knowledgebase. agent-knowledgebase's own
  `AGENT_KB_EMBED_API_KEY` is independent and only read when
  `kb_backend != 'lightrag'`.
- Health-check polls the sidecar's `/health` endpoint (part of the
  REST contract below) so docker-compose surfaces a startup failure
  before agent-knowledgebase sends real traffic.

---

## REST API contract

The stub will eventually call these endpoints. Treat the schemas
below as the **target** — the LightRAG project's authoritative API
docs live at <https://github.com/HKUDS/LightRAG> (see the README's
"API" section and the `lightrag/api/` Python module for the FastAPI
endpoint definitions).

### `POST /insert`

Body:
```json
{
  "kb_id": "<string>",
  "documents": [
    {
      "id": "<string>",
      "content": "<string>",
      "metadata": {"source_type": "...", "uri": "...", "...": "..."}
    }
  ]
}
```

Response: `{"inserted": <int>, "request_id": "<string>"}`.

Implements `RetrieverBackend.index(...)`.

### `POST /query`

Body:
```json
{
  "kb_id": "<string>",
  "query": "<string>",
  "mode": "naive | local | global | hybrid",
  "top_k": <int>,
  "filters": {"...": "..."}
}
```

Response: list of result rows shaped like
`{"content": ..., "source_id": ..., "source_type": ..., "score": ..., "metadata": {...}}`.

Implements both `RetrieverBackend.query()` (mode=`hybrid` —
LightRAG's graph-augmented default) and `RetrieverBackend.search()`
(mode=`naive` — keyword-only).

### `POST /delete`

Body:
```json
{
  "kb_id": "<string>",
  "ids": ["<string>", "..."],
  "source_id": "<string-or-null>"
}
```

Response: `{"deleted": <int>}`.

Implements `RetrieverBackend.delete(...)`.

### `GET /info?kb_id=...`

Response:
```json
{
  "kb_id": "<string>",
  "entity_count": <int>,
  "relation_count": <int>,
  "chunk_count": <int>,
  "first_source": {
    "source_type": "...",
    "uri": "...",
    "dedup_key": "...",
    "page_id": "..."
  },
  "dominant_embedding_model": "<string>"
}
```

Implements `RetrieverBackend.info(...)` after reshaping into the
probe-4 contract (`source_type`, `uri`, `dedup_key`, `page_id`,
`dominant_embedding_model` — all five keys present, `None` when
inapplicable).

### `GET /health`

Response: `{"status": "ok | degraded | unavailable", "version": "..."}`.

Implements `RetrieverBackend.health_check()`.

---

## Migration path from chromadb / markdown to lightrag

Phase 4 shipped the `kb_migrate(kb_id, target_backend)` MCP tool
plus `services/migration.py` with `export_to_markdown` /
`import_from_markdown` / `export_chromadb_dump` /
`import_chromadb_dump`. **Activating LightRAG support requires a
parallel addition** to the migration module:

1. Add `export_to_lightrag` and `import_from_lightrag` (or a single
   `migrate_to_lightrag` orchestrator) that walk the source
   backend's pages / chunks and POST them to the sidecar's
   `/insert` endpoint.
2. Extend `kb_migrate(target_backend=...)` to accept `'lightrag'`
   (currently rejected — see `services/migration.py`).
3. Write `<saves_dir>/<kb-name>/.migrated_to=lightrag` after a
   successful cutover so the per-KB routing in
   `backends/__init__.py::resolve_backend_name` picks up the new
   backend on the next request without restarting the MCP server.

Until that lands, the only way to populate a `kb_backend='lightrag'`
KB is to re-ingest from the original raw sources via
`kb_ingest_batch`.

The reverse migration (LightRAG -> chromadb / markdown) requires
the inverse `export_from_lightrag` path. Because LightRAG's graph
store is not 1:1 with the chunk-shaped data the other backends
expect, the export is **lossy** — entity / relation triples are
serialised into the chunks' `metadata` blob and the graph
structure cannot be perfectly round-tripped back.

---

## Activation checklist

Before flipping any `NotImplementedError` to live forwarding code,
the following must be green:

- [ ] At least one operator commits to running the LightRAG
      sidecar in production.
- [ ] LightRAG REST API contract verified against the `HKUDS/LightRAG`
      release version the operator runs (the endpoint list above is
      a design target — confirm against the actual release).
- [ ] `tests/test_lightrag_backend_stub.py` extended with live-mode
      tests that use a docker-compose test sidecar (skipped when
      the sidecar is not running). The skipped tests cover at
      minimum: insert + query round-trip, delete, info() probe-4
      shape with a populated sidecar, health_check status flip
      from "unavailable" to "ok".
- [ ] `services/migration.py` extended with `migrate_to_lightrag`
      (per "Migration path" above) AND `kb_migrate(target_backend=
      'lightrag')` no longer rejected.
- [ ] `docs/migration_guide.md` updated with the lightrag target.
- [ ] `Settings.lightrag_url` field added (with `AGENT_KB_LIGHTRAG_URL`
      alias, default `http://localhost:9621`).
- [ ] Phase 4 + Phase 5 still in production — the per-chunk
      `(embedder_version, embedding_provider, embed_base_url)`
      snapshot is the safety net that lets existing chromadb /
      markdown KBs stay queryable under their original embedder
      after a per-KB flip to lightrag.
- [ ] CHANGELOG entry promotes Phase 6 from "stub-only" to "live"
      and notes any operator-facing migration steps.
- [ ] `tests/test_redesign_contract.py::_FROZEN_MCP_TOOLS_V0_6_0` is
      reviewed — Phase 6 activation should NOT add any new MCP tool
      (the existing 26 surface stays). The 26-tool frozen set is
      preserved bit-for-bit.
- [ ] `ruff check src/ tests/` clean.
- [ ] Full pytest pass on Linux + macOS + Windows CI matrix.

When every box is checked the stub can be promoted in a dedicated
PR scoped to "Phase 6 activation" with the spec_id
`70ab2170-381a-4657-bcd1-28a40c6f369b` cited in the commit message.
