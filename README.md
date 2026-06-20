# agent-knowledgebase

Personal knowledgebase plugin — ingest sources, build wiki artifacts, query with vector search.

Exposes an MCP server (`agent-knowledgebase`) packaged as a Claude Code plugin (`agent-knowledgebase-auto-dev`).

## Installation

The plugin runs the MCP server inside a Docker container. The plugin shim refuses to start if the `agent-knowledgebase` container is not in `running` state.

### Prerequisites

- **Docker** (Docker Desktop on macOS/Windows; docker engine on Linux). `docker` must be on PATH.
- One of:
  - The bundled Ollama sidecar (default — installs nothing extra), then a one-time model pull (~5 GB):
    ```bash
    cd ${CLAUDE_PLUGIN_ROOT}/docker && docker compose up -d
    docker exec agent-kb-ollama ollama pull qwen3-embedding:8b
    ```
  - **OR** an `OPENAI_API_KEY` (set in the environment or in `docker/docker-compose.override.yml`) to use OpenAI embeddings instead.

### Starting the container

```bash
cd ${CLAUDE_PLUGIN_ROOT}/docker
docker compose up -d
```

Run `docker compose` from `docker/` (not from the plugin root with `-f`) so any `docker-compose.override.yml` you place alongside the base file is auto-loaded.

This starts two services:
- `agent-knowledgebase` — the MCP server host (idle until `docker exec` connects).
- `agent-kb-ollama` — the Ollama sidecar that serves embedding requests.

Persistent state lives in two named volumes (`agent-kb-data` for sqlite + chromadb, `ollama-models` for downloaded Ollama models). Both survive `docker compose down`; remove them with `docker volume rm` if you want a clean slate.

### Stopping

```bash
cd ${CLAUDE_PLUGIN_ROOT}/docker && docker compose down
```

### Per-host customization

Copy `docker/docker-compose.override.yml.example` to `docker/docker-compose.override.yml` and edit. Common overrides: bind-mount your host saves directory, switch to OpenAI embeddings, disable the Ollama sidecar.

### Plugin error tokens

The plugin shim emits structured single-line JSON to stderr on failure:

- `docker_not_installed` — install Docker and ensure `docker` is on PATH.
- `container_not_running` — run `cd ${CLAUDE_PLUGIN_ROOT}/docker && docker compose up -d`.

## Ecosystem version floor

This plugin ships as **v0.6.0** as the kb-first Stage 1 drop of the v0.3.0 sibling
ecosystem (kb 0.6.0 + yt 0.5.0 + x-api 0.4.0). The `data-etl-orchestrator`
contract probe (probe 4, ensemble-contract) refuses to dispatch unless
`agent-knowledgebase >= 0.6.0`. If you pin an older version, the orchestrator
will emit a structured upgrade message telling the user to install v0.6.0.
See `skills/references/contract-probe-protocol.md` in the orchestrator repo
for the full probe.

## Consumer-side every-call stderr capture (R1 mitigation)

Callers MUST tee the kb server's stderr to a log file per request — never
fire-and-forget. The v0.6.0 `knowledgebase_stderr_log` helper emits a single
structured JSON line per lifecycle phase (`kb_id`, `op`, `phase`,
`elapsed_ms`, `rows_in`, `rows_ok`, `rows_skipped`, `rows_failed`,
`dedup_policy`, `request_id`, `tool_caller_version`). Dropping these lines
reopens the silent-DB-failure blind spot that probe 4 assertion (d) was
designed to close — a composite-PK conflict on `tweets` (or equivalent kb
write) will surface on stderr within 100 ms, but only if the caller is
reading stderr. The orchestrator's probe 4 enforces this invariant at
preflight; direct callers bypassing the orchestrator must replicate the
stderr-tee discipline.

## Configuration

The plugin resolves settings in four layers, highest priority last:

1. Built-in defaults
2. User-level JSON — `~/.agent-kb/config.json` (override path with `$AGENT_KB_USER_CONFIG`)
3. Project-level JSON — `./.agent-kb/config.json` (override path with `$AGENT_KB_PROJECT_CONFIG`, or control the project root via `$AGENT_KB_PROJECT_DIR` / `$CLAUDE_PROJECT_DIR` / `$PWD`)
4. Environment variables (always win)

### Required environment variables

- `AGENT_KB_SAVES_DIR` — base directory for knowledgebase data. Required; the plugin fails loudly if missing or invalid. Each KB lives at `<saves_dir>/<sanitized-name>/`.

### Secrets (environment-only, never in JSON)

These must always come from the environment. Setting them in a JSON config file is rejected loudly.

- `OPENAI_API_KEY` — required when `embedding.provider = "openai"`. Use the standard OpenAI environment variable name; do NOT prefix with `AGENT_KB_`.

### Example `config.json`

```json
{
  "embedding": {
    "provider": "ollama",
    "model": "qwen3-embedding:8b"
  },
  "chunk": {
    "size": 512,
    "overlap": 64,
    "token_encoding": "cl100k_base"
  },
  "query": {
    "default_top_k": 10,
    "hybrid": {
      "vector_weight": 0.7,
      "fts_weight": 0.3,
      "fetch_multiplier": 2
    }
  },
  "ingest": {
    "excluded_dirs": [
      "__pycache__", "node_modules", ".git", ".venv",
      ".mypy_cache", ".pytest_cache", "dist", "build",
      "venv", ".tox", ".ruff_cache", ".eggs",
      ".idea", ".vscode", ".hg", ".svn"
    ]
  }
}
```

### Configurable keys

| JSON path | Env var | Default |
|---|---|---|
| `embedding.provider` | `AGENT_KB_EMBEDDING_PROVIDER` | `"ollama"` |
| `embedding.model` | `AGENT_KB_EMBEDDING_MODEL` | `"qwen3-embedding:8b"` |
| `embedding.base_url` | `AGENT_KB_EMBED_BASE_URL` | `null` (defaults: `http://ollama:11434` for ollama, `https://api.openai.com/v1` for openai) |
| `embedding.timeout_seconds` | `AGENT_KB_EMBED_TIMEOUT_SECONDS` | `30.0` |
| `embedding.max_retries` | `AGENT_KB_EMBED_MAX_RETRIES` | `0` |
| `chunk.size` | `AGENT_KB_CHUNK_SIZE` | `512` |
| `chunk.overlap` | `AGENT_KB_CHUNK_OVERLAP` | `64` |
| `chunk.token_encoding` | `AGENT_KB_CHUNK_TOKEN_ENCODING` | `"cl100k_base"` |
| `query.default_top_k` | `AGENT_KB_QUERY_DEFAULT_TOP_K` | `10` |
| `query.hybrid.vector_weight` | `AGENT_KB_QUERY_HYBRID_VECTOR_WEIGHT` | `0.7` |
| `query.hybrid.fts_weight` | `AGENT_KB_QUERY_HYBRID_FTS_WEIGHT` | `0.3` |
| `query.hybrid.fetch_multiplier` | `AGENT_KB_QUERY_HYBRID_FETCH_MULTIPLIER` | `2` |
| `ingest.excluded_dirs` | `AGENT_KB_INGEST_EXCLUDED_DIRS` (comma-separated) | see below |

Default `ingest.excluded_dirs`: `__pycache__`, `node_modules`, `.git`, `.venv`, `venv`, `.tox`, `.mypy_cache`, `.pytest_cache`, `.ruff_cache`, `.eggs`, `dist`, `build`, `.idea`, `.vscode`, `.hg`, `.svn`. Setting this key is a full replacement — not a merge.

### Validation rules

- `chunk.overlap` must be `< chunk.size`.
- `query.hybrid.vector_weight + query.hybrid.fts_weight` must sum to `1.0` (±1e-6).
- Unknown keys, forbidden keys, or malformed JSON fail loudly with the file path and reason.

### Config MCP tools

The plugin exposes five tools for inspecting and mutating config at runtime:

- `kb_config_path` — show where each file resolves and whether it exists.
- `kb_config_show(scope)` — dump `"merged"`, `"user"`, `"project"`, `"env"`, or `"defaults"`.
- `kb_config_get(key)` — one key with its resolved value and provenance layer.
- `kb_config_set(scope, key, value)` — atomic write to user or project file with validation.
- `kb_config_validate` — dry-run resolver; flags broken files before they break startup.

## Multi-process deployments

The kb server uses a per-`kb_id` `threading.Lock` that serializes ingest within a single Python process. For multi-process deployments, see [`docs/cross-process-lock-recipe.md`](docs/cross-process-lock-recipe.md) for opt-in locking recipes (`filelock`, `portalocker`, Postgres advisory lock, or raw `fcntl` / `msvcrt`). The built-in lock does NOT protect against concurrent ingest from a second server process. Enforcement of one of these recipes as a runtime dependency is tracked as a v0.4.0 planning item; until then, all four recipes are opt-in.
