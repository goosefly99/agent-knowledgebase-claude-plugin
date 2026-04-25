# agent-knowledgebase

Personal knowledgebase plugin — ingest sources, build wiki artifacts, query with vector search.

Exposes an MCP server (`agent-knowledgebase`) packaged as a Claude Code plugin (`agent-knowledgebase-auto-dev`).

## Installation

### Pattern A — Claude Code marketplace (default, since v0.8.0)

Install the plugin from the marketplace entry `agent-knowledgebase-auto-dev`. The only host requirement is **Python >=3.11 on PATH**. `uv` is no longer required as a runtime dependency.

On first launch the bundled `bin/run_server.py` self-bootstraps a venv at `${CLAUDE_PLUGIN_ROOT}/.venv` (using the stdlib `venv` module), runs `pip install -r requirements.lock` inside it, and persists a SHA256 sentinel so subsequent launches skip pip install entirely (fast path). The full `.mcp.json` is just:

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

The launcher emits structured single-line JSON stderr if it cannot bootstrap. The three error tokens are:

- `python_version` — the host Python is older than 3.11. Install Python 3.11+.
- `network_unreachable` — pip install timed out at 30 s. Either fix network access or set `AGENT_KB_VENDORED_DEPS=/path/to/wheels` (see below).
- `read_only_filesystem` — `${CLAUDE_PLUGIN_ROOT}` is not writable. Mount it writable or relocate the plugin root to a writable directory.

#### Air-gapped / offline (`AGENT_KB_VENDORED_DEPS`)

For corporate hosts behind captive portals or air-gapped environments, vendor wheels once on a connected host:

```bash
pip download -d wheels/ -r requirements.lock
```

Then ship `wheels/` to the target host and set:

```bash
export AGENT_KB_VENDORED_DEPS=/path/to/wheels
```

The launcher will use `pip install --no-index --find-links=$AGENT_KB_VENDORED_DEPS` and skip PyPI entirely.

### Pattern C — `pipx install agent-knowledgebase` (alternate)

If you prefer not to let the plugin manage its own venv, you can install the package directly:

```bash
pipx install agent-knowledgebase
```

This exposes the `agent-knowledgebase-server` CLI entry point (registered via `[project.scripts]`). Wire it into your own MCP host config as `command: agent-knowledgebase-server`.

> **Marketplace caveat:** the Claude Code plugin marketplace does **not** auto-run `pipx install` for you. If you want this pattern, run the `pipx install` command manually before configuring the plugin. Pattern A above is what the marketplace ships out of the box and is the default for that reason.

### Pattern D — Docker (alternate, zero Python on host)

A future GHCR Docker image (`ghcr.io/...`) will let you run the server without any Python on the host. The Dockerfile is a Phase 6 deliverable of the v2.1 redesign and is not yet shipped — track `docs/redesign/ROADMAP.md`.

### PEP-723 single-file scripts (future)

PEP-723 inline-metadata single-file launchers are tracked as a future-only design and not in scope for v0.8.0.

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

- `AGENT_KB_EMBED_API_KEY` — Bearer token for the remote embeddings endpoint (only needed for `embedding.provider = "remote"`; use any placeholder for servers that ignore auth such as local Ollama).
- `AGENT_KB_PINECONE_API_KEY` — Pinecone key (only needed for `vectorstore = "pinecone"`).

### Example `config.json`

```json
{
  "vectorstore": "chromadb",
  "embedding": {
    "provider": "sentence-transformers",
    "model": "all-MiniLM-L6-v2"
  },
  "pinecone": {
    "index": null,
    "environment": null
  },
  "export_path": null,
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
| `vectorstore` | `AGENT_KB_VECTORSTORE` | `"chromadb"` |
| `embedding.provider` | `AGENT_KB_EMBEDDING_PROVIDER` | `"sentence-transformers"` |
| `embedding.model` | `AGENT_KB_EMBEDDING_MODEL` | `"all-MiniLM-L6-v2"` |
| `embedding.base_url` | `AGENT_KB_EMBED_BASE_URL` | `null` (required when `embedding.provider = "remote"`) |
| `embedding.timeout_seconds` | `AGENT_KB_EMBED_TIMEOUT_SECONDS` | `30.0` |
| `embedding.max_retries` | `AGENT_KB_EMBED_MAX_RETRIES` | `0` |
| `pinecone.index` | `AGENT_KB_PINECONE_INDEX` | `null` |
| `pinecone.environment` | `AGENT_KB_PINECONE_ENVIRONMENT` | `null` |
| `export_path` | `AGENT_KB_EXPORT_PATH` | `null` |
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
