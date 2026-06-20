# agent-knowledgebase Plugin Guide (v0.13.0)

A personal knowledgebase MCP server, packaged as a Claude Code plugin. Ingests sources, builds wiki artifacts, and answers vector-search queries — all from a single bundled Docker runtime.

This guide is the comprehensive reference for v0.13.0. The repo's `README.md` is the install quickstart; this document is the deeper operational and architectural picture. For per-version change notes see `CHANGELOG.md`; for migration from v0.12.x see [Section 11](#11-migration-from-v012x).

---

## Table of contents

1. [What this plugin is](#1-what-this-plugin-is)
2. [Architecture](#2-architecture)
3. [Installation](#3-installation)
4. [Per-host customisation](#4-per-host-customisation)
5. [Configuration](#5-configuration)
6. [Embedding providers](#6-embedding-providers)
7. [MCP tool surface](#7-mcp-tool-surface)
8. [Storage layout](#8-storage-layout)
9. [Operational behaviour](#9-operational-behaviour)
10. [Error tokens reference](#10-error-tokens-reference)
11. [Migration from v0.12.x](#11-migration-from-v012x)
12. [Testing & verification](#12-testing--verification)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. What this plugin is

`agent-knowledgebase` is an MCP (Model Context Protocol) server that gives an agent a long-lived, queryable knowledgebase. It exposes ~25 `kb_*` tools covering knowledgebase lifecycle, source ingestion, wiki maintenance, semantic and keyword search, and configuration management.

It is shipped as a Claude Code plugin (`agent-knowledgebase-auto-dev`) that registers a single MCP server (`agent-knowledgebase`). The plugin's launcher (`bin/run_server.py`) is a thin Docker shim — the actual MCP server runs **inside** a long-lived `agent-knowledgebase` container so the host needs only the Docker CLI and a running daemon. There is no host-Python prerequisite.

v0.13.0 represents the Docker-first runtime + collapsed embedder/vectorstore surface. The earlier "stdlib-venv bootstrap" model is gone, as are several backends that complicated maintenance without carrying their weight.

---

## 2. Architecture

```
+-------------+                                  +---------------------------+
|   Host      |     stdio JSON-RPC               |  Docker container         |
|             |  ----------------------------->  |  agent-knowledgebase      |
|  bin/       |                                  |                           |
|  run_       |   docker exec -i ... python      |  +---------------------+  |
|  server.py  |   -m agent_knowledgebase.server  |  |  FastMCP server     |  |
|  (shim)     |                                  |  |  (mcp/server)       |  |
+-------------+                                  |  +---------+-----------+  |
                                                 |            |              |
                                                 |            v              |
                                                 |  KnowledgebaseService     |
                                                 |  (per-KB threading.Lock)  |
                                                 |    |              |       |
                                                 |    v              v       |
                                                 |  SQLite       ChromaDB    |
                                                 |  (metadata)   (vectors)   |
                                                 |                           |
                                                 +---------------------------+
                                                              |
                                                              | HTTP /api/embed
                                                              v
                                                 +---------------------------+
                                                 |  agent-kb-ollama          |
                                                 |  (sidecar; default        |
                                                 |   embedding provider)     |
                                                 +---------------------------+
```

**Roles:**

- **Plugin shim** (`bin/run_server.py`) — runs on the host. Verifies Docker is on `PATH` and that the `agent-knowledgebase` container is in state `running`. On success, runs `docker exec -i agent-knowledgebase python -m agent_knowledgebase.server` as a child via `subprocess.run`, inheriting stdio so the MCP transport pipes flow straight through. The Python launcher stays alive as the long-lived parent for the full session — necessary on Windows, where `os.execvp` would kill the original PID (`_P_OVERLAY` semantics) and Claude Code's harness would close the connection within milliseconds of `initialize`. On failure, emits a single-line JSON error on stderr (`docker_not_installed` / `container_not_running` / `inspect_failed`) and exits 1.
- **Container** (`agent-knowledgebase`) — long-running, idle (`tail -f /dev/null`). The plugin's shim does a fresh `docker exec` per MCP session. State persists in either a named volume (`agent-kb-data`, the default) or a host bind (set via `docker-compose.override.yml`).
- **Sidecar** (`agent-kb-ollama`) — bundled Ollama instance reachable at `http://ollama:11434` over the docker bridge network. Holds embedding model weights in `/root/.ollama` (named volume `ollama-models` by default). Required only when `embedding_provider="ollama"` AND no override pulls in a different sidecar; the `sentence-transformers` and `openai` providers do not use it.
- **Embedding** — three providers (`ollama` default, `sentence-transformers` local, `openai` via API key). See [Section 6](#6-embedding-providers) for the full picture including the v0.13.0 auto-fallback contract.
- **Vectorstore** — ChromaDB only. One collection per knowledgebase, persisted under `<saves_dir>/<kb-dir>/chroma/`.

**Concurrency model:** the MCP server is synchronous (no `async def` anywhere in `src/agent_knowledgebase/`). FastMCP runs tool handlers in a thread pool. To prevent races when multiple tool calls target the *same* `kb_id` concurrently, `KnowledgebaseService.ingest_source` holds a per-`kb_id` `threading.Lock` for the entire pipeline duration. Calls against *different* `kb_id`s run concurrently. This is single-process serialisation only — see `docs/cross-process-lock-recipe.md` for opt-in `filelock` / `portalocker` / Postgres-advisory recipes when running multiple Python processes.

---

## 3. Installation

### Prerequisites

- **Docker** — Docker Desktop (macOS, Windows) or Docker Engine (Linux). The `docker` CLI must be on the host's `PATH`.
- ~5 GB free disk for the default Ollama embedding model, OR an `OPENAI_API_KEY`, OR acceptance of the `sentence-transformers` fallback (~80 MB downloads at first use).

The plugin shim refuses to start (with a structured error) if the `agent-knowledgebase` container isn't in `running` state, so a missing Docker install fails loudly rather than silently.

### Quickstart (default Ollama provider)

```bash
# From the plugin's checkout directory
cd ${CLAUDE_PLUGIN_ROOT}/docker
docker compose up -d
docker exec agent-kb-ollama ollama pull qwen3-embedding:8b   # one-time, ~5 GB
```

The Ollama pull is deliberately a manual step so first-boot doesn't block on a 5 GB download.

### Quickstart (OpenAI provider)

```bash
# Copy the override template and edit
cp docker/docker-compose.override.yml.example docker/docker-compose.override.yml
# Set OPENAI_API_KEY + provider in the override (or in your shell env)
cd ${CLAUDE_PLUGIN_ROOT}/docker
docker compose up -d
```

The Ollama sidecar can be disabled in the override via `profiles: [disabled]` when only the OpenAI provider is in use.

### Quickstart (sentence-transformers provider)

The model loads from HuggingFace on first use. Bind a host-cache directory in your override so the weights don't end up trapped in the container:

```yaml
# docker/docker-compose.override.yml
services:
  agent-knowledgebase:
    volumes:
      - /your/host/saves:/data/saves
      - /your/host/hf-cache:/data/hf-cache
    environment:
      AGENT_KB_EMBEDDING_PROVIDER: sentence-transformers
      AGENT_KB_EMBEDDING_MODEL: all-MiniLM-L6-v2
      HF_HOME: /data/hf-cache
```

Pre-create the HF cache directory on the host before `up -d`.

### Stopping

```bash
cd ${CLAUDE_PLUGIN_ROOT}/docker && docker compose down
```

State persists across `down`/`up` cycles in the named volumes. Drop them only with explicit `docker volume rm agent-kb-data` / `docker volume rm ollama-models` if you want a clean slate.

---

## 4. Per-host customisation

### Why `cd docker/` (not `-f docker/...`)

Always run `docker compose` from the `docker/` directory, not from the plugin root with `-f docker/docker-compose.yml`. The `-f` form **disables auto-discovery** of `docker-compose.override.yml`, silently falling back to named volumes on the boot drive even when a host-specific override exists. Run from `docker/`:

```bash
cd ${CLAUDE_PLUGIN_ROOT}/docker
docker compose up -d
```

### `docker-compose.override.yml`

This file is **gitignored** (it contains host-specific paths and possibly secrets). The tracked template is `docker/docker-compose.override.yml.example`. Common overrides:

```yaml
# 1) Bind-mount KB data off the boot drive (Windows example)
services:
  agent-knowledgebase:
    volumes:
      - W:/agent_knowledgebase_mcp/saves:/data/saves
  ollama:
    volumes:
      - W:/agent_knowledgebase_mcp/ollama-models:/root/.ollama
```

```yaml
# 2) Use OpenAI, disable the Ollama sidecar
services:
  agent-knowledgebase:
    volumes:
      - /path/on/your/host:/data/saves
    environment:
      AGENT_KB_EMBEDDING_PROVIDER: openai
      AGENT_KB_EMBEDDING_MODEL: text-embedding-3-small
      OPENAI_API_KEY: sk-...
  ollama:
    profiles: [disabled]
```

```yaml
# 3) Use sentence-transformers, bind HF cache to host
services:
  agent-knowledgebase:
    volumes:
      - /path/on/your/host/saves:/data/saves
      - /path/on/your/host/hf-cache:/data/hf-cache
    environment:
      AGENT_KB_EMBEDDING_PROVIDER: sentence-transformers
      AGENT_KB_EMBEDDING_MODEL: all-MiniLM-L6-v2
      HF_HOME: /data/hf-cache
```

The base compose file uses `${VAR:-default}` syntax for the embedder env vars, so simple swaps like `AGENT_KB_EMBEDDING_PROVIDER=openai docker compose up -d` work without any override edit. Volume binds always require the override.

---

## 5. Configuration

### Resolution layers (highest priority last)

1. Built-in defaults (declared on `Settings` fields)
2. User-level JSON — `~/.agent-kb/config.json` (override path with `AGENT_KB_USER_CONFIG`)
3. Project-level JSON — `./.agent-kb/config.json` (override path with `AGENT_KB_PROJECT_CONFIG`, or control the project root via `AGENT_KB_PROJECT_DIR` / `CLAUDE_PROJECT_DIR` / `PWD`)
4. Environment variables prefixed with `AGENT_KB_` (always win)

The merged result is what the running server uses. `kb_config_show` returns the merged values plus a per-key provenance map naming which layer won.

### Required environment variables

| Variable | Purpose |
|---|---|
| `AGENT_KB_SAVES_DIR` | Base directory for KB storage. Required; the server fails loudly with a structured `config_missing` payload if missing or pointing at a non-directory. Default: `~/.agent-kb/saves` (the directory must exist on disk). Each KB lives at `<saves_dir>/<sanitized-name>/`. |

### Embedding env vars

| Variable | Default | Notes |
|---|---|---|
| `AGENT_KB_EMBEDDING_PROVIDER` | `ollama` | `Literal["ollama", "sentence-transformers", "openai"]`. Pydantic rejects other values at config load with a structured runtime error. |
| `AGENT_KB_EMBEDDING_MODEL` | `qwen3-embedding:8b` | Provider-specific. Suggested defaults: `all-MiniLM-L6-v2` (sentence-transformers), `text-embedding-3-small` (openai). |
| `AGENT_KB_EMBED_BASE_URL` | `http://ollama:11434` (ollama) / `https://api.openai.com/v1` (openai) / unused (sentence-transformers) | The provider-specific URL suffix (`/api/embed` for ollama, `/embeddings` for openai) is appended automatically. |
| `AGENT_KB_EMBED_TIMEOUT_SECONDS` | `30.0` | Per-call wall-clock bound for remote embed requests. |
| `AGENT_KB_EMBED_MAX_RETRIES` | `0` | Retries after transport failure. |

### Secrets policy

Secrets MUST come from environment variables, never from JSON config:

- `OPENAI_API_KEY` — required when `embedding_provider="openai"`. Use the standard OpenAI variable name; do **not** prefix with `AGENT_KB_`. Rejected loudly via `config_missing` payload when unset.

Setting any secret in a JSON config file is rejected at load time.

### Other env vars

| Variable | Default | Purpose |
|---|---|---|
| `AGENT_KB_CHROMADB_EAGER_WARM` | `true` | Eager-warm all chromadb collections at MCP server startup (Phase A). Amortises the ~180 s HNSW cold-load into startup time rather than the first tool call. Set to `false` to disable. |
| `AGENT_KB_CHUNK_SIZE` | `512` | Default chunk size in tokens. |
| `AGENT_KB_CHUNK_OVERLAP` | `64` | Chunk overlap in tokens; must be `< chunk_size`. |
| `AGENT_KB_CHUNK_TOKEN_ENCODING` | `cl100k_base` | tiktoken encoding for token-based chunking. |
| `AGENT_KB_QUERY_DEFAULT_TOP_K` | `10` | Default top_k for `kb_query` / `kb_search`. |
| `AGENT_KB_QUERY_HYBRID_VECTOR_WEIGHT` | `0.7` | Weight of vector scores in hybrid search. Must sum to 1.0 with FTS weight. |
| `AGENT_KB_QUERY_HYBRID_FTS_WEIGHT` | `0.3` | Weight of FTS scores in hybrid search. |
| `AGENT_KB_QUERY_HYBRID_FETCH_MULTIPLIER` | `2` | Over-fetch multiplier before merging hybrid results. |
| `AGENT_KB_INGEST_EXCLUDED_DIRS` | (see below) | Comma-separated list of directory names to skip during recursive ingestion. Full replacement of the default list. |
| `AGENT_KB_EXPORT_PATH` | unset | Markdown export directory used by `kb_export`. |
| `AGENT_KB_AUTO_REEMBED` | `0` | Bypass for the `EMBEDDER_VERSION_MISMATCH` rejection. See [Section 6](#6-embedding-providers). |

**Default `ingest_excluded_dirs`:** `__pycache__`, `node_modules`, `.git`, `.venv`, `.mypy_cache`, `.pytest_cache`, `dist`, `build`, `venv`, `.tox`, `.ruff_cache`, `.eggs`, `.idea`, `.vscode`, `.hg`, `.svn`.

---

## 6. Embedding providers

### The three providers

#### `ollama` (default)

- POSTs to `<base_url>/api/embed` with the Ollama JSON shape.
- Parses `{"embeddings": [[...]]}` responses.
- No API key required (Ollama doesn't authenticate locally).
- Default base URL `http://ollama:11434` matches the bundled sidecar.
- Default model `qwen3-embedding:8b` (4096-dim).
- Embedder version: `ollama/<model>@<base_url>`.

The qwen3 model must be pulled into the Ollama instance once — `docker exec agent-kb-ollama ollama pull qwen3-embedding:8b` or, if the bind-mount points elsewhere, into whichever Ollama is in front of `embed_base_url`.

#### `sentence-transformers`

- Local embedder backed by the `sentence-transformers` library, bundled in the Docker image (`embed-local-st` extra is installed by `requirements.lock`).
- Lazy-imports `sentence_transformers` inside `__init__` so the module loads cleanly even if the extra is missing in non-Docker contexts.
- Models pulled from HuggingFace on first use; bind `HF_HOME` to a host directory to keep weights off the container layer.
- Suggested model: `all-MiniLM-L6-v2` (384-dim, ~80 MB).
- Embedder version: `sentence-transformers/<model>@hf-fp32`.

#### `openai`

- POSTs to `<base_url>/embeddings` with the OpenAI JSON contract: `{"input": [...], "model": ...}` → `{"data": [{"index": N, "embedding": [...]}]}`.
- Authenticates via `Authorization: Bearer ${OPENAI_API_KEY}`. The API key MUST come from the `OPENAI_API_KEY` env var.
- Compatible with any /v1/embeddings-style endpoint (real OpenAI, vLLM, llama.cpp server, etc.) by adjusting `embed_base_url`.
- Known dimensions hardcoded for `text-embedding-3-small` (1536), `text-embedding-3-large` (3072), `text-embedding-ada-002` (1536). Unknown models trigger a one-call `probe` against the live endpoint.
- Embedder version: `openai/<model>@<base_url>` (the base URL is included so two endpoints serving the same nominal model are treated as distinct).

### Auto-fallback contract (v0.13.0)

When `embedding_provider="ollama"` AND the configured `embed_base_url` is unreachable at first **global-default-embedder** build, the global default silently swaps to `SentenceTransformerEmbedder("all-MiniLM-L6-v2")` and emits the `OLLAMA_UNREACHABLE_FALLBACK_TO_SENTENCE_TRANSFORMERS` stderr token (single-line JSON).

**Crucial scope:** the auto-fallback is for the **global default only**. Per-snapshot embedder rebuilds (the path used at query and ingest time on existing KBs) do NOT fall back. A KB that was previously ingested with ollama embeddings cannot be queried under a different embedder without producing geometrically meaningless similarity scores, so the per-snapshot rebuild raises `EmbedderUnavailableError` loudly when the original embedder is unreachable.

Observable consequences:

- A fresh `kb_create` followed by `kb_ingest` while ollama is down: succeeds under sentence-transformers; the new chunks are stamped with `provider="sentence-transformers"`, `model="all-MiniLM-L6-v2"`. All future queries against this KB use sentence-transformers.
- `kb_query` against a KB whose chunks were ingested under ollama, while ollama is down: fails with `embed_unreachable`. The plugin does not silently swap embedding spaces.
- `kb_ingest` *appending to* a KB whose existing chunks were ingested under ollama, while ollama is down: same — fails loudly. (Mixing two embedders in one KB would degrade query quality silently.)

The fallback model is hardcoded to `all-MiniLM-L6-v2`. To use a different sentence-transformers model as the default when ollama is down, either set `AGENT_KB_EMBEDDING_PROVIDER=sentence-transformers` explicitly, or open an issue for a configurable fallback knob.

### Per-snapshot rebuild (the query path)

Every chunk inserted into the KB carries a per-chunk **embedding snapshot** — a `(model, provider, base_url)` triple stamped at ingest time. At query time, `KnowledgebaseService._query_embedder_for(kb_id)`:

1. Reads the dominant snapshot for the KB via `Database.get_embedding_snapshot(kb_id)` (single sqlite-side aggregation).
2. If the snapshot matches the currently-configured embedder, returns the cached global embedder (the fast path).
3. Otherwise, rebuilds the matching embedder via `create_embedder_for_model(config, model, provider=..., base_url=...)` and caches it per `(kb_id, model, provider, base_url)`.

This is what makes `kb_query` correct even when an old KB was ingested under one provider and the configured default has since changed. It is also the mechanism the auto-fallback specifically does NOT participate in — the rebuild calls `_build_embedder` directly, not `create_embedder`.

### Embedder version stamping (Phase 5)

Every embedder exposes `embedder_version` — an opaque string uniquely identifying its vector geometry beyond the nominal model name (`<library>/<model>[@<provider_or_quant>]`). Two embedders with the same model name but different libraries (e.g. fastembed-MiniLM-int8 vs sentence-transformers-MiniLM-fp32) report different versions because their vectors are not interchangeable under cosine similarity even at matching dimensions.

The version stamp is stored per chunk. At ingest time, attempting to add chunks under embedder `V_new` to a KB carrying chunks under `V_old` raises `EmbedderVersionMismatchError` (`error_code="EMBEDDER_VERSION_MISMATCH"`). Set `AGENT_KB_AUTO_REEMBED=1` to bypass; **after bypass**, re-ingest the existing chunks under the new embedder or call `kb_delete` and recreate, otherwise the KB will hold mixed-version chunks and the snapshot resolver picks the dominant tuple at query time, leaving minority-version chunks silently unreachable.

The mixed-version situation, if it does occur, is announced via a `MIXED_EMBEDDER_VERSIONS_DETECTED` stderr line at most once per `kb_id` per process.

### Legacy v0.6.0 KB auto-backfill

KBs ingested with v0.6.0 stamped only the model name in chunk metadata; provider and base_url were `NULL`. After the Phase 5 default flip (and again after the v0.13.0 changes), naively reading those snapshots would route through the global default embedder — which since v0.13.0 may be `openai` (and fail without `OPENAI_API_KEY`), or sentence-transformers (and produce wrong-dimension vectors).

When `_query_embedder_for` detects a snapshot of `(model, None, None)` AND the model name matches a known historical Ollama default (`qwen3-embedding...`), it auto-stamps `provider='ollama'` + `base_url='http://127.0.0.1:11434'` for that KB's chunks and re-reads the snapshot. The auto-fix is announced via a `LEGACY_OLLAMA_KB_BACKFILLED` stderr line and is idempotent (in-process set guards against repeated probes).

---

## 7. MCP tool surface

All tools return single-string JSON payloads. Each is wrapped with `_with_tool_timeout`.

### Lifecycle

| Tool | Signature | Purpose |
|---|---|---|
| `kb_create` | `(name: str, description: str = "")` | Create a new knowledgebase. Returns the KB record. |
| `kb_delete` | `(kb_id: str)` | Delete a KB and all data (sources, pages, chunks, vectors, on-disk dir). |
| `kb_list` | `()` | List all KBs (enriched with source/page counts). |
| `kb_info` | `(kb_id: str)` | Return one KB record (with counts). |

### Ingestion

| Tool | Signature | Purpose |
|---|---|---|
| `kb_ingest` | `(kb_id, source_type, uri, metadata="{}", dedup_key=None, dedup_policy="skip")` | Run the full ingest pipeline for a single source. `source_type` is one of `file` / `directory` / `codebase` / `website` / `sql_database` / `git_history` / `api_endpoint`. `dedup_policy` is `skip` / `replace` / `force-add`. |
| `kb_ingest_batch` | `(kb_id, sources_json_array, dedup_policy="skip")` | Batch ingest. Each row in `sources` is `{source_type, uri, metadata?, dedup_key?, dedup_policy?}`. Hard-rejects more than 50 `source_type="sql_database"` rows in a single call (`BatchSizeExceededError`). |
| `kb_update_source` | `(source_id: str)` | Re-ingest a source — delete old chunks and re-run the pipeline. |
| `kb_remove_source` | `(source_id: str)` | Remove a source and all its chunks. |
| `kb_list_sources` | `(kb_id: str)` | List all sources in a KB. |
| `kb_get_source` | `(source_id: str)` | Get one source's record. |
| `kb_pipeline_status` | `(kb_id: str)` | Recent pipeline run records (for ingest telemetry). |

### Query / search

| Tool | Signature | Purpose |
|---|---|---|
| `kb_query` | `(kb_id, text, top_k=None)` | Vector similarity search. On embedder unreachable, returns the structured `embed_*` payload instead of raising. |
| `kb_search` | `(kb_id, text, top_k=None)` | Keyword search (FTS5). |
| `kb_get_page` | `(page_id: str)` | Fetch one wiki page. |
| `kb_list_pages` | `(kb_id, page_type="")` | List pages, optionally filtered by type (`entity` / `concept` / `summary` / `index` / `comparison` / `synthesis`). Each result is enriched with `page_id`, `source_type`, `uri`, `dedup_key` from the first source. |
| `kb_get_links` | `(page_id, direction="outbound")` | Get the link graph for a page; `direction` is `outbound` (default) or `inbound`. |

### Wiki maintenance

| Tool | Signature | Purpose |
|---|---|---|
| `kb_lint` | `(kb_id: str)` | Run wiki health checks: orphan pages, missing wikilink targets, stale source refs, coverage gaps, index drift. Returns a structured report. |
| `kb_lint_fix` | `(kb_id: str)` | Auto-fix the fixable lint issues. |
| `kb_rebuild_index` | `(kb_id: str)` | Rebuild the wiki index page. |
| `kb_export` | `(kb_id, output_dir)` | Write all wiki pages out as markdown files. |

### Configuration

| Tool | Signature | Purpose |
|---|---|---|
| `kb_config_path` | `()` | Return the user/project config file paths and whether they exist; `resolved_via` is `env_override` when an `AGENT_KB_*_CONFIG` env var supplied the path. |
| `kb_config_show` | `(scope="merged")` | Return configured values. `scope` is `merged` / `user` / `project` / `env` / `defaults`. `merged` includes a `provenance` map naming which layer won. |
| `kb_config_get` | `(key: str)` | Return one setting (dotted path, e.g. `embedding.model`). Includes `value`, `provenance`, and `effective_type`. |
| `kb_config_set` | `(scope, key, value)` | Persist a setting in the user or project JSON. Rejects writes for `FORBIDDEN_KEYS` (secrets and `embedding.api_key`). |
| `kb_config_validate` | `()` | Validate the merged config against the `Settings` schema. |

---

## 8. Storage layout

Inside the container, `AGENT_KB_SAVES_DIR=/data/saves` (default). Bound to either the named volume `agent-kb-data` or a host path via the override.

```
/data/saves/                                # AGENT_KB_SAVES_DIR
├── .index.json                             # {kb_id: dir_name} registry
├── <kb-dir>/                               # one subdir per KB
│   ├── knowledgebase.db                    # SQLite metadata (kb, sources, pages, chunks, embedding snapshots)
│   └── chroma/                             # ChromaDB persistence
│       ├── chroma.sqlite3
│       └── <collection-uuid>/...
├── <other-kb-dir>/
│   └── ...
```

Other paths inside the container:

```
/root/.ollama/                              # Ollama model weights (only present when sidecar is enabled)
└── models/blobs/sha256-<hash>              # the actual GGUFs / safetensors

/data/hf-cache/                             # only present when override binds it for sentence-transformers
└── hub/models--sentence-transformers--all-MiniLM-L6-v2/...
```

The host-side mapping is whatever you put in the override; for example:

```
W:/agent_knowledgebase_mcp/saves           ↔  /data/saves
W:/agent_knowledgebase_mcp/ollama-models   ↔  /root/.ollama
W:/agent_knowledgebase_mcp/hf-cache        ↔  /data/hf-cache
```

The KB directory name is `sanitize_kb_dir_name(name)` — lowercase, alphanumerics + dashes, no spaces. Empty/all-special names become `unnamed`. Two KBs with names that sanitize to the same dir name are rejected at create time.

---

## 9. Operational behaviour

### Cold-start mitigation (Phase A eager-warm)

ChromaDB lazy-loads HNSW indexes on first collection access, which can be a ~180 s wait on a populated KB. v0.12.0's Phase A added eager-warm: at MCP server init, `KnowledgebaseService.warmup_all_chromadb_kbs()` runs in a daemon thread and pre-loads every chromadb-backed KB's collection. The wait is amortised into server startup rather than the first tool call.

The eager-warm has a 30 s wallclock cap and is non-blocking (server init returns regardless). Set `AGENT_KB_CHROMADB_EAGER_WARM=false` to opt out.

### Per-KB serialisation

`ingest_source` holds a per-`kb_id` `threading.Lock` for the entire pipeline duration. Two `kb_ingest` calls against the same KB serialise; calls against different KBs run concurrently. There is no cross-process locking — see `docs/cross-process-lock-recipe.md` for opt-in `filelock` / `portalocker` / Postgres-advisory recipes when running multiple Python processes.

### Structured stderr logging

Every lifecycle event in the ingestion pipeline emits a single-line JSON record on stderr via `knowledgebase_stderr_log`. The contract (every record carries every key):

| Key | Type | Meaning |
|---|---|---|
| `kb_id` | string | Target knowledgebase ID. |
| `op` | string | Short label (`ingest_source`, `lock_acquire`, `lock_acquired`, `lock_release`, etc.). |
| `phase` | string | A `PipelinePhase` value or one of `pre_run` / `post_run` / `lock` / `auto` / `warn`. |
| `elapsed_ms` | int | Milliseconds; zero for instantaneous events. |
| `rows_in` / `rows_ok` / `rows_skipped` / `rows_failed` | int | Per-batch row counts. |
| `dedup_policy` | string | A `DedupPolicy` value or `n/a`. |
| `request_id` / `tool_caller_version` | string \| null | Caller-supplied correlators. |
| `error_code` / `error_message` | string | Optional, only on failure or warning emissions. |

Consumers MUST tee stderr to a per-request log file rather than fire-and-forget; dropping stderr reopens the silent-DB-failure blind spot the v0.6.0 work was designed to close.

### Embedder unreachable handling

`kb_query` is the one tool that catches `EmbedderUnavailableError` and returns the structured payload instead of raising. The payload shape:

```json
{
  "error": "embed_timeout|embed_unreachable|model_not_pulled|auth_failed|rate_limited|probe_failed",
  "model": "qwen3-embedding:8b",
  "phase": "embed_query|embed_batch|probe_dimension",
  "latency_ms": 30000,
  "detail": "TimeoutException",
  "retry_after": "60"        // only on rate_limited (429 Retry-After header)
}
```

Other tools propagate the exception. Callers should treat `embed_*` errors as transient and `auth_failed` / `model_not_pulled` as configuration errors.

---

## 10. Error tokens reference

### Plugin shim (host-side, before the container is even reached)

| Token | When | Action |
|---|---|---|
| `docker_not_installed` | `docker` is not on `PATH`. | Install Docker. |
| `container_not_running` | `agent-knowledgebase` container is missing or not in `running` state. | `cd ${CLAUDE_PLUGIN_ROOT}/docker && docker compose up -d` |
| `inspect_failed` | `docker inspect` raised an unexpected error. | Check Docker daemon health. |

### Configuration

| Token | When | Action |
|---|---|---|
| `config_missing` | A required setting is absent OR `AGENT_KB_SAVES_DIR` resolves to something that doesn't exist / isn't a directory. | Set the missing env var; payload `missing` field names which one. |

### Embedding (returned by `kb_query`; raised by other tools)

| Token | When |
|---|---|
| `embed_timeout` | Wall-clock bound exceeded waiting for the remote embedder. |
| `embed_unreachable` | Network error reaching the embedder. |
| `model_not_pulled` | HTTP 404 from the embedder — the configured model isn't available on the server. |
| `auth_failed` | HTTP 401/403 — bad/missing API key. |
| `rate_limited` | HTTP 429 — payload includes `retry_after` from the response header. |
| `probe_failed` | An embedder's `probe_dimension` returned an empty vector list. |

### Lifecycle (stderr-emitted, structured records via `knowledgebase_stderr_log`)

| Token | When |
|---|---|
| `OLLAMA_UNREACHABLE_FALLBACK_TO_SENTENCE_TRANSFORMERS` | Auto-fallback fired during global-default embedder build. Inline JSON line on stderr (not via `knowledgebase_stderr_log` — no `kb_id` in this scope). |
| `LEGACY_OLLAMA_KB_BACKFILLED` | A v0.6.0 KB with `(model, None, None)` snapshot was auto-stamped with `provider='ollama'` + the historical Ollama base URL. |
| `MIXED_EMBEDDER_VERSIONS_DETECTED` | A KB carries chunks under more than one `embedder_version`; the snapshot resolver picks the dominant tuple, minority chunks are silently unreachable. Re-ingest minority chunks or `kb_delete` + recreate. |
| `EMBEDDER_VERSION_MISMATCH` | Ingest attempt under a different embedder version than the KB's existing chunks. Bypass with `AGENT_KB_AUTO_REEMBED=1` (and re-ingest existing chunks afterward). |

---

## 11. Migration from v0.12.x

### Breaking changes

- **Plugin requires Docker.** Hosts no longer need Python 3.11+ on `PATH`. The shim refuses to start with `container_not_running` if the container isn't up.
- **Backends collapsed to chromadb-only.** `markdown`, `lightrag`, `textvec` backends and the `kb_backend_per_kb` routing surface are removed. The `kb_migrate` MCP tool is also gone (no migration targets remain).
- **Vectorstore collapsed to chromadb-only.** Pinecone removed. The `vectorstore` config field, `AGENT_KB_PINECONE_*` env vars, and the pinecone optional-extra are gone.
- **Embedding providers collapsed to three.** `Settings.embedding_provider` is now `Literal["ollama", "sentence-transformers", "openai"]`. `remote` and `fastembed` providers are removed. Default flips from `remote` + `text-embedding-3-small` to `ollama` + `qwen3-embedding:8b` (matched to the bundled sidecar).
- **`AGENT_KB_EMBED_API_KEY` → `OPENAI_API_KEY`.** The openai provider reads the industry-standard env var directly. The `AGENT_KB_EMBED_API_KEY` env var, the `embedding.api_key` JSON field, and the `embed_api_key` Settings field are removed.
- **textvec-only schema removed.** The `chunks_fts` virtual table and its sync triggers are dropped from `database.py`. `wiki_pages_fts` (used by chromadb's wiki search) is retained.

### What still works without changes

- KBs ingested under v0.12.0 with the default `chromadb` backend continue to work — the on-disk chromadb layout is unchanged. Mount your existing saves dir into the container and queries succeed.
- Legacy v0.6.0 KBs ingested under Ollama benefit from the `LEGACY_OLLAMA_KB_BACKFILLED` auto-stamp described in [Section 6](#6-embedding-providers).

### Steps

1. Install Docker if not present.
2. (Optional) Copy `docker/docker-compose.override.yml.example` to `docker/docker-compose.override.yml` and bind your existing `~/.agent-kb/saves` (or wherever your KBs live) into `/data/saves`.
3. `cd ${CLAUDE_PLUGIN_ROOT}/docker && docker compose up -d`
4. If staying on Ollama: `docker exec agent-kb-ollama ollama pull qwen3-embedding:8b`. If switching to OpenAI: set `OPENAI_API_KEY` in the override and `AGENT_KB_EMBEDDING_PROVIDER=openai`. If switching to sentence-transformers: see [Section 4](#4-per-host-customisation) for the HF cache bind.
5. Drop any `AGENT_KB_EMBED_API_KEY` / `AGENT_KB_PINECONE_*` env vars from your environment / configs.

KBs on the `markdown`, `lightrag`, `textvec`, or `pinecone` backends must be migrated to chromadb manually before upgrading; v0.13.0 cannot read those layouts.

---

## 12. Testing & verification

### Test layout

- `tests/test_docker_shim.py` — host-side shim error-token contract.
- `tests/test_docker_container_required.py` — container-required guard.
- `tests/test_chromadb_mutability.py` — pins ChromaDBStore's insert/update/delete contract.
- `tests/test_embedder_fallback.py` — auto-fallback contract (probe-fail-and-swap, probe-success-keep, no-fallback-for-other-providers, per-snapshot-rebuild-bypasses-fallback, preserves-original-error-on-st-init-failure).
- Plus the broader unit + integration suite — `tests/` total runs at 575 passed, 2 skipped on the v0.13.0 baseline.

### Run

```bash
python -m ruff check src/ tests/ bin/
python -m pytest tests/ -q
```

### End-to-end smoke (Ollama default)

A minimal smoke test that exercises the full pipeline from inside the running container:

```bash
docker exec -e AGENT_KB_SAVES_DIR=/data/saves \
            -e AGENT_KB_EMBEDDING_PROVIDER=ollama \
            -e AGENT_KB_EMBEDDING_MODEL=qwen3-embedding:8b \
            -e AGENT_KB_EMBED_BASE_URL=http://ollama:11434 \
            agent-knowledgebase python -c "
import tempfile, time, json
from pathlib import Path
from agent_knowledgebase.config import Settings
from agent_knowledgebase.models import SourceType
from agent_knowledgebase.services.knowledgebase import KnowledgebaseService

svc = KnowledgebaseService(Settings())
kb = svc.create_kb(name=f'smoke_{time.time_ns()}', description='smoke')
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / 's.md'
    p.write_text('Ollama provides embeddings.', encoding='utf-8')
    src = svc.ingest_source(kb_id=kb.id, source_type=SourceType.file, uri=str(p))
results = svc.query(kb_id=kb.id, text='Who provides embeddings?', top_k=3)
svc.delete_kb(kb.id)
print(json.dumps({'ok': True, 'kb_id': kb.id, 'n_results': len(results)}))
"
```

Expected end-to-end latency: ~3–5 s warm, ~15 s cold (model load).

### End-to-end smoke (auto-fallback)

Stop the Ollama sidecar, then run the same script with `AGENT_KB_EMBEDDING_PROVIDER=ollama` configured. Expected: the script succeeds, the global embedder is `SentenceTransformerEmbedder`, and `OLLAMA_UNREACHABLE_FALLBACK_TO_SENTENCE_TRANSFORMERS` appears on stderr.

```bash
docker stop agent-kb-ollama
# ... run the smoke script (with HF_HOME bound; see Section 4) ...
docker start agent-kb-ollama
```

---

## 13. Troubleshooting

### `container_not_running`

The plugin shim refuses to start because the `agent-knowledgebase` container is not in state `running`. Run:

```bash
cd ${CLAUDE_PLUGIN_ROOT}/docker && docker compose up -d
```

If `docker compose up -d` fails with "no such file", you ran from the wrong directory — must `cd` into `docker/` first (see [Section 4](#4-per-host-customisation) for the reasoning).

### `embed_unreachable` on a KB that previously worked

The KB was ingested with one embedder; that embedder is now unreachable. Two situations:

1. **Ollama sidecar is stopped or crashed.** `docker ps` to confirm; `docker compose up -d ollama` to restart.
2. **The configured `embed_base_url` was changed.** The KB's chunks are stamped with the original base URL — the per-snapshot rebuild tries to reach the original endpoint, not the current configured one. Fix: revert the base URL change, or `kb_delete` + recreate under the new endpoint.

The auto-fallback does **not** apply here. Cross-embedder queries would produce wrong scores; the loud failure is intentional.

### `model_not_pulled`

The embedder server is reachable but doesn't have the requested model. For Ollama: `docker exec agent-kb-ollama ollama pull <model>`. For OpenAI: check the model name spelling; the OpenAI API returns 404 for unrecognised models.

### `EMBEDDER_VERSION_MISMATCH`

A KB carries chunks under one embedder version and an ingest is attempting to add chunks under a different one. Three options:

1. Switch the configured embedder back to the original. (Run `kb_config_get embedding.model` / `kb_config_get embedding.provider` to see what's currently set; check the KB's first-page `embedding_model` metadata to see what it was ingested under.)
2. Re-ingest the existing sources under the new embedder. The path: `kb_delete` → `kb_create` → re-`kb_ingest` for each source.
3. `AGENT_KB_AUTO_REEMBED=1` to bypass — but then re-ingest the existing sources, otherwise you'll have a permanently mixed-version KB whose minority chunks silently fail to surface.

### Cold-start ~180 s on first tool call

`AGENT_KB_CHROMADB_EAGER_WARM=true` (default) should mostly eliminate this; if you set it to `false`, the first `kb_query`/`kb_search`/`kb_ingest` after server start will block while ChromaDB loads the HNSW index. Re-enable eager-warm or accept the per-cold-start latency.

### KB data ends up on the boot drive instead of the bind-mounted host path

The most common cause: running `docker compose -f docker/docker-compose.yml ...` from the plugin root instead of `cd docker/ && docker compose ...`. The `-f` form disables auto-discovery of `docker-compose.override.yml`, so your bind-mount override is silently ignored and Docker falls back to the named volume. Always run from `docker/`.

### sentence-transformers downloads land in the container layer (lost on `down`)

The container has no host bind for the HF cache. Add the override block in [Section 4](#4-per-host-customisation) (Quickstart sentence-transformers).

### KB index out of sync (KB dir on disk but no entry in `.index.json`, or vice versa)

Symptom: `kb_list` shows nothing but a directory exists in `saves/`, or `kb_create` rejects a name with `directory already exists`. Cause: a previous run crashed mid-create or someone manually `rm`'d a KB dir.

Fix: align the two by hand. Either:

- Delete the orphaned directory: `docker exec agent-knowledgebase rm -rf /data/saves/<orphan-dir>` — `.index.json` will catch up on next `_save_index` call.
- Or remove the orphan entry from `.index.json` manually.

This is a rare hand-fix and should not happen during normal operation.

---

## Appendix A — Reference files in this repo

| Purpose | Path |
|---|---|
| Plugin shim (Docker launcher) | `bin/run_server.py` |
| MCP server entry point | `src/agent_knowledgebase/server.py` |
| Lifecycle / ingestion / query orchestration | `src/agent_knowledgebase/services/knowledgebase.py` |
| Embedder providers + auto-fallback | `src/agent_knowledgebase/services/embeddings.py` |
| Settings + `Literal` validation | `src/agent_knowledgebase/config.py` |
| ChromaDB vectorstore | `src/agent_knowledgebase/services/vectorstore.py` |
| Database (SQLite, snapshots, FTS) | `src/agent_knowledgebase/database.py` |
| Docker base compose | `docker/docker-compose.yml` |
| Docker host override (gitignored) | `docker/docker-compose.override.yml` |
| Override template (tracked) | `docker/docker-compose.override.yml.example` |
| Dockerfile | `docker/Dockerfile` |
| Mutability invariants | `tests/test_chromadb_mutability.py` |
| Shim error tokens | `tests/test_docker_shim.py` |
| Container-required smoke | `tests/test_docker_container_required.py` |
| Auto-fallback contract | `tests/test_embedder_fallback.py` |
| Cross-process locking | `docs/cross-process-lock-recipe.md` |
| SQL ingestion grammar | `docs/sql-database-uri-grammar.md` |
| Safe `WHERE` clause grammar | `docs/safe-where-clause-grammar.md` |
| User-facing install docs | `README.md` |
| Per-version change notes | `CHANGELOG.md` |
