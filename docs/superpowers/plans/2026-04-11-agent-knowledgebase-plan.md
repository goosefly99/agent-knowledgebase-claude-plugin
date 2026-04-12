# Agent Knowledgebase Plugin — Implementation Plan

**Spec:** `docs/superpowers/specs/2026-04-11-agent-knowledgebase-design.md`
**Repo:** `goosefly99/agent-knowledgebase-claude-plugin`

## Task 1: Project Scaffold and Plugin Packaging

**Goal:** Create the project skeleton — directory structure, plugin metadata, Python packaging, and MCP server entry point that starts but exposes no tools yet.

**Files to create:**
- `.claude-plugin/plugin.json` — plugin metadata
- `.mcp.json` — MCP server startup config (uv)
- `pyproject.toml` — Python project with all dependencies
- `src/agent_knowledgebase/__init__.py` — package init with version
- `src/agent_knowledgebase/server.py` — MCP server skeleton (FastMCP, no tools yet)
- `src/agent_knowledgebase/config.py` — Settings class with all env vars from spec Section 2
- `tests/__init__.py`
- `tests/conftest.py` — shared fixtures (tmp_path DB, test config)
- `tests/test_config.py` — test config loading and defaults

**Verification:** `uv run python -m agent_knowledgebase.server` starts without error. `uv run pytest tests/test_config.py` passes.

**Context:** This is the foundation. Every subsequent task depends on this scaffold existing.

---

## Task 2: Data Models and SQLite Schema

**Goal:** Implement Pydantic models and SQLite database initialization with all tables from spec Section 7.

**Files to create:**
- `src/agent_knowledgebase/models.py` — All Pydantic models (Knowledgebase, Source, WikiPage, Chunk, PipelineRun) with enums (SourceType, SourceStatus, PageType, PipelinePhase, RunStatus)
- `src/agent_knowledgebase/database.py` — SQLite connection management, schema creation (all tables + FTS5), basic CRUD helpers
- `tests/test_models.py` — model validation tests
- `tests/test_database.py` — schema creation, CRUD tests

**Verification:** `uv run pytest tests/test_models.py tests/test_database.py` passes.

**Context:** Models are used by every service module. Database is the persistence backbone.

---

## Task 3: Embedding Provider Abstraction

**Goal:** Implement the embeddings service with SentenceTransformer (default) and OpenAI providers behind a common interface.

**Files to create:**
- `src/agent_knowledgebase/services/__init__.py`
- `src/agent_knowledgebase/services/embeddings.py` — `Embedder` protocol, `SentenceTransformerEmbedder`, `OpenAIEmbedder`, `create_embedder()` factory
- `tests/test_embeddings.py` — test both providers (mock OpenAI, real sentence-transformers with small model)

**Verification:** `uv run pytest tests/test_embeddings.py` passes.

**Context:** Vectorstore and query services depend on embeddings.

---

## Task 4: Vector Store Abstraction

**Goal:** Implement the vectorstore service with ChromaDB (default) and Pinecone providers behind a common interface.

**Files to create:**
- `src/agent_knowledgebase/services/vectorstore.py` — `VectorStore` protocol, `ChromaDBStore`, `PineconeStore`, `create_vectorstore()` factory. Methods: `add()`, `query()`, `delete()`, `get()`
- `tests/test_vectorstore.py` — test ChromaDB provider (real local instance), mock Pinecone

**Verification:** `uv run pytest tests/test_vectorstore.py` passes.

**Context:** Depends on embeddings for vector dimensions. Used by ingestion and query.

---

## Task 5: Pipeline State Machine

**Goal:** Implement the pipeline service that enforces phase ordering for write operations.

**Files to create:**
- `src/agent_knowledgebase/services/pipeline.py` — `PipelineManager` class. Phase enum: initialize → read_source → chunk → embed → integrate_wiki → finalize. Methods: `start_run()`, `advance_phase()`, `fail_phase()`, `retry_phase()`, `get_status()`. Validates transitions, persists PipelineRun records to SQLite.
- `tests/test_pipeline.py` — test valid transitions, invalid transitions rejected, retry logic, persistence

**Verification:** `uv run pytest tests/test_pipeline.py` passes.

**Context:** Depends on database module. Used by the knowledgebase service to gate write operations.

---

## Task 6: Source Ingestors

**Goal:** Implement all 7 ingestors from spec Section 5.

**Files to create:**
- `src/agent_knowledgebase/ingestors/__init__.py` — `Ingestor` protocol, `RawContent` dataclass, `ChunkConfig`, ingestor registry
- `src/agent_knowledgebase/ingestors/file.py` — `FileIngestor` (text, markdown, JSON, PDF via pdfplumber)
- `src/agent_knowledgebase/ingestors/directory.py` — `DirectoryIngestor` (recursive, .gitignore-aware)
- `src/agent_knowledgebase/ingestors/codebase.py` — `CodebaseIngestor` (tree-sitter AST chunking)
- `src/agent_knowledgebase/ingestors/website.py` — `WebsiteIngestor` (trafilatura extraction)
- `src/agent_knowledgebase/ingestors/sql_database.py` — `SqlDatabaseIngestor` (SQLAlchemy introspection)
- `src/agent_knowledgebase/ingestors/git_history.py` — `GitHistoryIngestor` (gitpython log/diff)
- `src/agent_knowledgebase/ingestors/api_endpoint.py` — `ApiEndpointIngestor` (httpx fetch + parse)
- `src/agent_knowledgebase/services/ingestion.py` — `IngestionOrchestrator` dispatches to appropriate ingestor
- `tests/test_ingestion.py` — test each ingestor with fixtures (file, directory, mock git repo, mock HTTP)

**Verification:** `uv run pytest tests/test_ingestion.py` passes.

**Context:** Depends on models. Used during the read_source and chunk pipeline phases.

---

## Task 7: Wiki Artifact Manager

**Goal:** Implement the wiki service for managing structured wiki pages in SQLite with FTS5.

**Files to create:**
- `src/agent_knowledgebase/services/wiki.py` — `WikiManager` class. CRUD for WikiPage. FTS search. Link graph management (add/remove/query links). Index page generation. Handles the integrate_wiki pipeline phase.
- `tests/test_wiki.py` — test CRUD, FTS search, link management, index generation

**Verification:** `uv run pytest tests/test_wiki.py` passes.

**Context:** Depends on database and models. Central to the LLM Wiki pattern.

---

## Task 8: Query Orchestrator

**Goal:** Implement hybrid query combining vectorstore similarity search with wiki FTS.

**Files to create:**
- `src/agent_knowledgebase/services/query.py` — `QueryOrchestrator` class. Methods: `query()` (semantic), `search()` (keyword), `hybrid_query()` (combined). Returns ranked results with source citations.
- `tests/test_query.py` — test semantic search, keyword search, hybrid ranking, citation formatting

**Verification:** `uv run pytest tests/test_query.py` passes.

**Context:** Depends on vectorstore, wiki, and embeddings services.

---

## Task 9: Lint and Export Services

**Goal:** Implement wiki health checks and markdown export.

**Files to create:**
- `src/agent_knowledgebase/services/lint.py` — `WikiLinter` class. Checks: orphan pages, missing pages, stale refs, coverage gaps, index drift. Returns structured lint report. Optional auto-fix.
- `src/agent_knowledgebase/services/export.py` — `MarkdownExporter` class. Export wiki pages to .md files with YAML frontmatter, wikilinks, incremental export.
- `tests/test_lint.py` — test each lint check with crafted wiki states
- `tests/test_export.py` — test export output, frontmatter, wikilinks, incremental behavior

**Verification:** `uv run pytest tests/test_lint.py tests/test_export.py` passes.

**Context:** Depends on wiki service.

---

## Task 10: Knowledgebase Lifecycle Service

**Goal:** Implement the top-level KB service that coordinates all sub-services.

**Files to create:**
- `src/agent_knowledgebase/services/knowledgebase.py` — `KnowledgebaseService` class. Methods: `create_kb()`, `delete_kb()`, `list_kbs()`, `get_kb()`, `ingest_source()` (coordinates pipeline → ingestion → vectorstore → wiki), `update_source()`, `remove_source()`, `get_stats()`.
- `tests/test_knowledgebase.py` — integration tests for full lifecycle (create → ingest → query → delete)

**Verification:** `uv run pytest tests/test_knowledgebase.py` passes.

**Context:** Depends on all services. This is the internal API that MCP tools call.

---

## Task 11: MCP Server and Tools

**Goal:** Wire up all 20 MCP tools in the server, connecting to the knowledgebase service.

**Files to modify:**
- `src/agent_knowledgebase/server.py` — Register all 20 tools from spec Section 6. Write-path tools call pipeline-gated methods. Read-path tools call direct query methods. Each tool: validate input, call service, format response.

**Files to create:**
- `tests/test_server.py` — test tool registration, input validation, response format

**Verification:** `uv run pytest tests/test_server.py` passes. `uv run python -m agent_knowledgebase.server` starts and registers all tools.

**Context:** Depends on knowledgebase service. This is the public API surface.

---

## Task 12: End-to-End Integration and Repository Setup

**Goal:** Run full integration test, set up git repo, push to GitHub.

**Steps:**
1. Run full test suite: `uv run pytest`
2. Run type checker: `uv run ruff check src/ tests/`
3. Fix any issues
4. Initialize git repo, commit all files
5. Create GitHub repo `goosefly99/agent-knowledgebase-claude-plugin`
6. Push to remote

**Verification:** All tests pass. Lint clean. Repo pushed.

## Dependency Graph

```
Task 1 (scaffold)
  ├── Task 2 (models + DB)
  │     ├── Task 3 (embeddings)
  │     │     └── Task 4 (vectorstore)
  │     ├── Task 5 (pipeline)
  │     ├── Task 6 (ingestors)
  │     └── Task 7 (wiki)
  │           ├── Task 8 (query) [also depends on Task 4]
  │           └── Task 9 (lint + export)
  └── Task 10 (KB service) [depends on Tasks 3-9]
        └── Task 11 (MCP server)
              └── Task 12 (integration + repo)
```

Tasks 3, 5, 6, 7 can run in parallel after Task 2 completes.
Tasks 8 and 9 can run in parallel after their dependencies.
