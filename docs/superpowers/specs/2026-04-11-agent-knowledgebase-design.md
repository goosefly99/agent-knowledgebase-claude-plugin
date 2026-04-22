# Agent Knowledgebase Plugin — Design Spec

## Overview

A Python-based Claude Code plugin implementing the "LLM Wiki" pattern. Exposes MCP tools for creating, managing, and querying personal knowledgebases with vector search. Write-path operations (ingest/load/update) require explicit pipeline enforcement; read-path operations (query/search) are always available.

## 1. Plugin Packaging

```
agent-knowledgebase/
  .claude-plugin/
    plugin.json          # name, description, version, author, repository
  .mcp.json              # MCP server startup via uv
  pyproject.toml         # Python deps, project metadata
  src/
    agent_knowledgebase/
      __init__.py
      server.py          # MCP server entry point
      config.py          # Configuration from env vars
      models.py          # Pydantic data models
      services/
        __init__.py
        pipeline.py      # Pipeline state machine
        ingestion.py     # Source ingestors
        vectorstore.py   # ChromaDB / Pinecone abstraction
        wiki.py          # Wiki artifact management (SQLite)
        query.py         # Query/search orchestration
        embeddings.py    # Embedding provider abstraction
        lint.py          # Wiki health checks
        export.py        # Markdown export for Obsidian
        knowledgebase.py # Top-level KB lifecycle
      ingestors/
        __init__.py
        file.py          # Single file ingestion
        directory.py     # Directory tree ingestion
        codebase.py      # Language-aware code chunking (tree-sitter)
        website.py       # Web content extraction (trafilatura)
        sql_database.py  # SQL database schema/data ingestion
        git_history.py   # Git log/diff ingestion
        api_endpoint.py  # REST API endpoint ingestion
  tests/
    __init__.py
    conftest.py
    test_config.py
    test_models.py
    test_pipeline.py
    test_ingestion.py
    test_vectorstore.py
    test_wiki.py
    test_query.py
    test_embeddings.py
    test_lint.py
    test_export.py
    test_knowledgebase.py
    test_server.py
```

**plugin.json:**
```json
{
  "name": "agent-knowledgebase",
  "description": "Personal knowledgebase plugin — ingest sources, build wiki artifacts, query with vector search",
  "version": "0.1.0",
  "author": { "name": "olive" },
  "repository": "https://github.com/goosefly99/agent-knowledgebase-claude-plugin.git"
}
```

**.mcp.json:**
```json
{
  "mcpServers": {
    "agent-knowledgebase": {
      "command": "uv",
      "args": ["run", "--directory", "${CLAUDE_PLUGIN_ROOT}", "python", "-m", "agent_knowledgebase.server"]
    }
  }
}
```

## 2. Configuration

All configuration via environment variables with sensible defaults:

| Variable | Default | Description |
|---|---|---|
| `AGENT_KB_DB_PATH` | `~/.agent-kb/knowledgebase.db` | SQLite database path |
| `AGENT_KB_VECTORSTORE` | `chromadb` | `chromadb` or `pinecone` |
| `AGENT_KB_CHROMA_PATH` | `~/.agent-kb/chroma` | ChromaDB persistence directory |
| `AGENT_KB_PINECONE_API_KEY` | (none) | Pinecone API key |
| `AGENT_KB_PINECONE_INDEX` | (none) | Pinecone index name |
| `AGENT_KB_PINECONE_ENVIRONMENT` | (none) | Pinecone environment |
| `AGENT_KB_EMBEDDING_PROVIDER` | `sentence-transformers` | `sentence-transformers` or `remote` |
| `AGENT_KB_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Model name for embeddings |
| `AGENT_KB_EMBED_API_KEY` | (none) | Bearer token for the remote embeddings endpoint (if using `remote` provider) |
| `AGENT_KB_EMBED_BASE_URL` | (none) | Base URL for the remote embeddings endpoint, e.g. `http://localhost:11434/v1` (required for `remote` provider) |
| `AGENT_KB_CHUNK_SIZE` | `512` | Default chunk size in tokens |
| `AGENT_KB_CHUNK_OVERLAP` | `64` | Chunk overlap in tokens |
| `AGENT_KB_EXPORT_PATH` | (none) | Markdown export directory (optional) |

Config loaded via a Pydantic `Settings` class using `pydantic-settings` for env var parsing.

## 3. Data Models (Pydantic)

**Knowledgebase:**
- `id: str` (UUID)
- `name: str`
- `description: str`
- `created_at: datetime`
- `updated_at: datetime`
- `source_count: int`
- `page_count: int`
- `config: dict` (per-KB overrides)

**Source:**
- `id: str` (UUID)
- `kb_id: str`
- `source_type: SourceType` (enum: file, directory, codebase, website, sql_database, git_history, api_endpoint)
- `uri: str` (path, URL, connection string)
- `metadata: dict`
- `ingested_at: datetime`
- `chunk_count: int`
- `status: SourceStatus` (enum: pending, ingesting, ingested, failed)

**WikiPage:**
- `id: str` (UUID)
- `kb_id: str`
- `title: str`
- `content: str` (markdown)
- `page_type: PageType` (enum: entity, concept, summary, index, comparison, synthesis)
- `source_ids: list[str]` (provenance)
- `tags: list[str]`
- `created_at: datetime`
- `updated_at: datetime`
- `inbound_links: list[str]` (page IDs)
- `outbound_links: list[str]` (page IDs)

**Chunk:**
- `id: str` (UUID)
- `source_id: str`
- `kb_id: str`
- `content: str`
- `metadata: dict` (position, file path, language, etc.)
- `embedding_id: str` (vectorstore reference)

**PipelineRun:**
- `id: str` (UUID)
- `kb_id: str`
- `source_id: str`
- `phase: PipelinePhase` (enum)
- `status: RunStatus` (enum: pending, running, completed, failed)
- `started_at: datetime`
- `completed_at: datetime | None`
- `error: str | None`
- `metadata: dict`

## 4. Service Modules

### pipeline.py — Pipeline State Machine
Enforces phase ordering for write operations:
```
initialize → read_source → chunk → embed → integrate_wiki → finalize
```
- Tracks `PipelineRun` records in SQLite
- Validates phase transitions (no skipping)
- Supports retry on failed phases
- Read-path operations bypass the pipeline entirely

### ingestion.py — Ingestion Orchestrator
Dispatches to the appropriate ingestor based on `source_type`. Manages the read_source and chunk phases of the pipeline.

### vectorstore.py — Vector Store Abstraction
Common interface for ChromaDB and Pinecone:
- `add(chunks, embeddings)` → store vectors
- `query(embedding, top_k, filters)` → ranked results
- `delete(ids)` → remove vectors
- `get(ids)` → retrieve by ID
- Factory pattern: `create_vectorstore(config) → VectorStore`

### wiki.py — Wiki Artifact Manager
SQLite + FTS5 for structured wiki pages:
- CRUD operations for WikiPage records
- Full-text search over page content
- Link graph management (inbound/outbound)
- Handles the `integrate_wiki` phase: creates/updates pages based on ingested chunks

### query.py — Query Orchestrator
Combines vectorstore similarity search with wiki FTS:
- Vector search for semantic relevance
- FTS for keyword matching
- Hybrid ranking with configurable weights
- Returns results with source citations

### embeddings.py — Embedding Provider
Abstraction over embedding models:
- `embed(texts: list[str]) → list[list[float]]`
- `embed_query(text: str) → list[float]`
- Providers: `SentenceTransformerEmbedder`, `RemoteEmbedder` (httpx-based)
- Factory: `create_embedder(config) → Embedder`

### lint.py — Wiki Health Checks
Lint operations:
- Orphan pages (no inbound links)
- Missing pages (referenced but don't exist)
- Stale references (links to deleted sources)
- Contradictions (flag pages with conflicting claims)
- Coverage gaps (topics mentioned but lacking pages)
- Index drift (index.md out of sync)

### export.py — Markdown Export
Export wiki to markdown files for Obsidian:
- One `.md` file per WikiPage
- YAML frontmatter with tags, dates, source counts
- Wikilinks (`[[Page Title]]`) for cross-references
- Asset handling for any embedded content
- Incremental export (only changed pages)

### knowledgebase.py — KB Lifecycle
Top-level operations:
- Create/list/get/delete knowledgebases
- Coordinate ingest (pipeline + ingestion + vectorstore + wiki)
- Statistics and health metrics

## 5. Ingestors

All ingestors implement a common interface:
```python
class Ingestor(Protocol):
    def read(self, uri: str, metadata: dict) -> list[RawContent]: ...
    def chunk(self, content: list[RawContent], config: ChunkConfig) -> list[Chunk]: ...
```

| Ingestor | Description | Key Dependencies |
|---|---|---|
| FileIngestor | Single file (text, PDF, markdown, JSON, etc.) | Built-in, `pdfplumber` for PDF |
| DirectoryIngestor | Recursive directory tree | Uses FileIngestor per file, respects .gitignore |
| CodebaseIngestor | Language-aware code chunking | `tree-sitter` for AST parsing |
| WebsiteIngestor | Web page content extraction | `trafilatura` |
| SqlDatabaseIngestor | Database schema + sample data | `sqlalchemy` for DB connectivity |
| GitHistoryIngestor | Git commits, diffs, authors | `gitpython` |
| ApiEndpointIngestor | REST API response ingestion | `httpx` |

## 6. MCP Tools

### Write-Path (Pipeline-Gated)

| Tool | Description |
|---|---|
| `kb_create` | Create a new knowledgebase |
| `kb_delete` | Delete a knowledgebase and all data |
| `kb_ingest` | Ingest a source into a KB (starts pipeline) |
| `kb_ingest_batch` | Batch ingest multiple sources |
| `kb_update_source` | Re-ingest a source (update) |
| `kb_remove_source` | Remove a source and its chunks |
| `kb_lint` | Run wiki health checks |
| `kb_lint_fix` | Auto-fix lint issues |
| `kb_rebuild_index` | Rebuild the wiki index page |
| `kb_export` | Export wiki to markdown |

### Read-Path (Always Available)

| Tool | Description |
|---|---|
| `kb_list` | List all knowledgebases |
| `kb_info` | Get KB details and statistics |
| `kb_query` | Semantic query across KB |
| `kb_search` | Keyword search across KB |
| `kb_get_page` | Get a specific wiki page |
| `kb_list_pages` | List wiki pages with filters |
| `kb_get_source` | Get source details |
| `kb_list_sources` | List sources in a KB |
| `kb_get_links` | Get link graph for a page |
| `kb_pipeline_status` | Check pipeline run status |

## 7. SQLite Schema

```sql
CREATE TABLE knowledgebases (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    config TEXT, -- JSON
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE sources (
    id TEXT PRIMARY KEY,
    kb_id TEXT NOT NULL REFERENCES knowledgebases(id),
    source_type TEXT NOT NULL,
    uri TEXT NOT NULL,
    metadata TEXT, -- JSON
    ingested_at TEXT,
    chunk_count INTEGER DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending'
);

CREATE TABLE wiki_pages (
    id TEXT PRIMARY KEY,
    kb_id TEXT NOT NULL REFERENCES knowledgebases(id),
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    page_type TEXT NOT NULL,
    tags TEXT, -- JSON array
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE wiki_links (
    from_page_id TEXT NOT NULL REFERENCES wiki_pages(id),
    to_page_id TEXT NOT NULL REFERENCES wiki_pages(id),
    PRIMARY KEY (from_page_id, to_page_id)
);

CREATE TABLE page_sources (
    page_id TEXT NOT NULL REFERENCES wiki_pages(id),
    source_id TEXT NOT NULL REFERENCES sources(id),
    PRIMARY KEY (page_id, source_id)
);

CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(id),
    kb_id TEXT NOT NULL REFERENCES knowledgebases(id),
    content TEXT NOT NULL,
    metadata TEXT, -- JSON
    embedding_id TEXT
);

CREATE TABLE pipeline_runs (
    id TEXT PRIMARY KEY,
    kb_id TEXT NOT NULL REFERENCES knowledgebases(id),
    source_id TEXT REFERENCES sources(id),
    phase TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    started_at TEXT NOT NULL,
    completed_at TEXT,
    error TEXT,
    metadata TEXT -- JSON
);

CREATE VIRTUAL TABLE wiki_pages_fts USING fts5(
    title, content, tags,
    content=wiki_pages,
    content_rowid=rowid
);
```

## 8. Dependencies

```toml
[project]
name = "agent-knowledgebase"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "mcp>=1.0.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "chromadb>=0.5",
    "sentence-transformers>=3.0",
    "trafilatura>=1.0",
    "tree-sitter>=0.23",
    "tree-sitter-languages>=1.10",
    "gitpython>=3.1",
    "sqlalchemy>=2.0",
    "httpx>=0.27",
    "pdfplumber>=0.11",
    "tiktoken>=0.7",
]

[project.optional-dependencies]
pinecone = ["pinecone-client>=3.0"]

[dependency-groups]
dev = ["pytest>=8.0", "pytest-asyncio>=0.24", "ruff>=0.5"]
```

## 9. Testing Strategy

- **pytest** with `pytest-asyncio` for async MCP handlers
- Unit tests per service module
- Integration tests for pipeline flows
- Mock vectorstore/embeddings for fast tests
- Fixtures in `conftest.py` for common KB/source/page setup
- Test SQLite databases via tmp_path fixture

## 10. Repository

- Repo: `goosefly99/agent-knowledgebase-claude-plugin`
- Will be added to `goosefly99-plugins` marketplace after launch
- Standard GitHub repo with MIT license
