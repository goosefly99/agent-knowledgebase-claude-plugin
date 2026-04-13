# agent-knowledgebase

Personal knowledgebase plugin — ingest sources, build wiki artifacts, query with vector search.

Exposes an MCP server (`agent-knowledgebase`) packaged as a Claude Code plugin (`agent-knowledgebase-auto-dev`).

## Installation

Install the plugin from the marketplace entry `agent-knowledgebase-auto-dev`. See `pyproject.toml` for Python dependencies (Python >=3.11).

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

- `AGENT_KB_OPENAI_API_KEY` — OpenAI key (only needed for `embedding.provider = "openai"`).
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
