# Agents Guide — agent-knowledgebase Redesign (spec_id `70ab2170-381a-4657-bcd1-28a40c6f369b`)

> Source spec: `pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json`
> Validation report: `pipeline_mcp_data/runs/agent-kb-redesign-validation-report.json`
> Status: scaffolding generated 2026-04-24 — implementation pending.

This document is the standing brief for any subagent (Claude or otherwise) that
picks up redesign work. Read it once at the start of every session before
touching code. The redesign supersedes v0.6.0/0.7.0 contracts only where the
spec explicitly says so — everything else is **frozen**.

---

## 1. Frozen contracts (do NOT regress)

The v0.6.0 surface is the floor. Any redesign work must keep these intact.

### 1.1 The 25 `kb_*` MCP tools

`@mcp.tool()`-decorated functions in `src/agent_knowledgebase/server.py` form a
**frozen surface**. The full list (verbatim from the spec, empirically confirmed
at decorator hits on lines 164,180,195,225,265,280,295,313,328,343,364,376,393,
416,433,450,482,499,514,532,602,628,695,724,767):

```
kb_create, kb_delete, kb_ingest, kb_ingest_batch, kb_update_source,
kb_remove_source, kb_lint, kb_lint_fix, kb_rebuild_index, kb_export,
kb_list, kb_info, kb_query, kb_search, kb_get_page, kb_list_pages,
kb_get_source, kb_list_sources, kb_get_links, kb_pipeline_status,
kb_config_path, kb_config_show, kb_config_get, kb_config_set,
kb_config_validate
```

You may **add** new tools (Phase 4 adds `kb_migrate`). You may NOT rename,
remove, or change the parameter names / response shapes of any of the 25.

### 1.2 Probe-4 contract preservation

`kb_info`, `kb_list_pages`, and `kb_list_sources` MUST return JSON whose shape
matches v0.6.0. Specifically the fields probe-4 (data-etl-orchestrator
>= v0.4.0) inspects: `source_type`, `uri`, `dedup_key`, `page_id`,
`dominant_embedding_model`. New backends MUST return `null` for inapplicable
fields, NOT omit them. Pinned by `tests/contract/test_probe4_response_shape.py`
(Phase 2 deliverable).

### 1.3 `knowledgebase_stderr_log` 11-field schema

Every emission of `services/stderr_log.py::knowledgebase_stderr_log(...)`
contains exactly these fields: `kb_id`, `op`, `phase`, `elapsed_ms`, `rows_in`,
`rows_ok`, `rows_skipped`, `rows_failed`, `dedup_policy`, `request_id`,
`tool_caller_version`. Pinned by `tests/contract/test_stderr_schema.py`
(Phase 0 / Phase 2 deliverable).

### 1.4 SQL-injection AST validator

`src/agent_knowledgebase/services/where_clause_validator.py` plus the
read-only sqlite engine (`?mode=ro`) plus the 50-row hard-reject on
`kb_ingest_batch(source_type=sql_database)` are frozen. The backend refactor
in Phase 2 explicitly preserves the boundary that `ingestors/sql_database*`
lives OUTSIDE `backends/`. Any change here requires both
`tests/test_where_clause_validator.py` and `tests/test_sql_database_ingest.py`
to stay green.

### 1.5 Decorator order is **already correct**

`@mcp.tool()` is the OUTER decorator; `@_with_tool_timeout` is INNER.
`tests/test_tool_timeout.py` passes 6/6 today including
`test_every_registered_mcp_tool_is_wrapped`. **DO NOT SWAP.** The v2.0 spec
draft inverted this — see §3 below for the corrected Bug-2 diagnosis. Phase 0
adds `tests/contract/test_decorator_order.py` purely as a regression guard
against accidental future inversion.

---

## 2. Phase ordering (v2.1 corrected)

The **only** valid ordering. Do not parallelize across phases without
explicit owner sign-off — Phase 4 must finish before Phase 5 starts, period.

| # | Phase | Description | Verification |
|---|-------|-------------|--------------|
| 0 | Bug fixes (chromadb backend, no architecture change) | Wrap `_get_service` in try/except; add `saves_dir` default factory; probe embedding dimension on first use; classify HTTP-status errors. | `pytest tests/test_get_service_config_missing.py tests/test_embedder_dimension_probe.py tests/test_chromadb_dimension_mismatch.py tests/test_tool_timeout.py -v` |
| 1 | Pattern A launcher | `bin/run_server.py` stdlib-venv auto-bootstrap; replace `.mcp.json` `command:uv` with `command:python args:[bin/run_server.py]`. | `pytest tests/test_run_server_bootstrap.py tests/test_run_server_failure_modes.py -v`; CI matrix Win/macOS/Linux |
| 2 | RetrieverBackend abstraction | `backends/__init__.py` factory + Protocol; `ChromadbBackend` wraps existing logic; route `KnowledgebaseService` through `self._backend`. | `pytest tests/test_retriever_backend_protocol.py tests/contract/test_probe4_response_shape.py -v`; full suite passes unchanged |
| 3 | MarkdownWikiBackend (opt-in) | `backends/markdown_backend.py`; sqlite FTS5; positive-allow source_type list; sharded index above ~200 articles. | `pytest tests/test_markdown_backend.py tests/contract/test_markdown_backend_probe4_shape.py -v` |
| 4 | **Migration tooling + per-page provider snapshot** (PRECEDES default-flip) | `services/migration.py`; `kb_migrate` MCP tool; read-fallback when wiki/ missing; ALTER TABLE pages ADD embedding_provider + embed_base_url; backfill script. | `pytest tests/test_migration_chromadb_to_markdown.py tests/test_per_page_provider_snapshot.py -v` |
| 5 | Embedding default-flip | `embedding_provider` default → `remote`; `embedding_model` default → `text-embedding-3-small`; fastembed opt-in; `extras['embed-local-st']` for sentence-transformers; embedder_version stamping. | `pytest tests/test_fastembed_recall_parity.py tests/test_fastembed_dimension_compat.py tests/test_embedder_version_mismatch.py -v` |
| 6 | LightRAGBackend (deferred) | Stub only. Design doc `docs/lightrag_integration_design.md`. No code. | n/a — deferred until adoption signal |

**The Phase 4 → Phase 5 ordering is non-negotiable.** Reversing it silently
breaks every existing v0.6.0 KB ingested with the Ollama default
(qwen3-embedding:8b, 4096-dim) the moment the global default flips to
text-embedding-3-small (1536-dim). Per-page `(embedding_provider,
embed_base_url)` snapshots ship in Phase 4 so `create_embedder_for_model` can
faithfully rebuild the original embedder after the default change.

---

## 3. Bug-2 corrected diagnosis (CRITICAL — DO NOT SWAP DECORATORS)

The v2.0 spec draft claimed Bug-2 was a decorator NO-OP at the MCP boundary.
This is **wrong**. Empirical verification (`tests/test_tool_timeout.py`
6/6 passing, including `test_every_registered_mcp_tool_is_wrapped`) confirms
the current decorator order works. The actual user-visible Bug-2 is:

> `src/agent_knowledgebase/server.py:49-54` — `_get_service()` calls
> `Settings().resolve_paths()` without exception handling.
> `config.py:42-46` declares `saves_dir: Path = Field(...)` with no default,
> so `pydantic.ValidationError` raises when `AGENT_KB_SAVES_DIR` is unset.
> `config.py:217-220` raises `FileNotFoundError` when the path doesn't exist.
> Either propagates as opaque `MCP InternalError` on every `kb_*` call.

### Fix (Phase 0)

1. Wrap `_get_service` body in `try: ... except (pydantic.ValidationError,
   FileNotFoundError, NotADirectoryError) as exc:` and return a sentinel
   structured-error service whose tool methods all return JSON
   `{"error":"config_missing", "missing":"AGENT_KB_SAVES_DIR", "detail":"..."}`.
2. Add `saves_dir` `default_factory=lambda: Path.home() / ".agent-kb" / "saves"`
   in `config.py` so fresh installs work with zero env vars.
3. Add a `main()` startup precheck so missing config fails immediately with
   structured stderr instead of on the first MCP call.
4. Add `tests/test_get_service_config_missing.py`.
5. Add `tests/contract/test_decorator_order.py` as a **regression guard only**
   — it asserts `__wrapped__` exists, NOT a specific decorator nesting.

---

## 4. Verification commands per phase

Every phase must run the project gate before declaring complete:

```bash
# Project gate (run after every phase)
ruff check
pytest -v

# Phase 0 specific
pytest tests/test_get_service_config_missing.py \
       tests/test_embedder_dimension_probe.py \
       tests/test_chromadb_dimension_mismatch.py \
       tests/test_tool_timeout.py \
       tests/contract/test_decorator_order.py -v

# Phase 1 specific
pytest tests/test_run_server_bootstrap.py \
       tests/test_run_server_failure_modes.py -v

# Phase 2 specific
pytest tests/test_retriever_backend_protocol.py \
       tests/contract/test_probe4_response_shape.py \
       tests/contract/test_stderr_schema.py -v

# Phase 3 specific
pytest tests/test_markdown_backend.py \
       tests/contract/test_markdown_backend_probe4_shape.py -v

# Phase 4 specific (BEFORE Phase 5)
pytest tests/test_migration_chromadb_to_markdown.py \
       tests/test_per_page_provider_snapshot.py -v

# Phase 5 specific
pytest tests/test_fastembed_recall_parity.py \
       tests/test_fastembed_dimension_compat.py \
       tests/test_embedder_version_mismatch.py -v
```

**Pre-existing lint/type errors are not an excuse.** Per global CLAUDE.md: fix
project-wide errors or explicitly enumerate each one and ask before continuing.

---

## 5. File layout introduced by the redesign

```
bin/
  run_server.py                          # Pattern A launcher (Phase 1)
docs/
  redesign/
    AGENTS.md                            # this file
    ROADMAP.md                           # phased work breakdown
    MISTAKES.md                          # standing pitfalls
  install_size.md                        # Phase 5 deliverable
  markdown_backend.md                    # Phase 3 deliverable
  migration_guide.md                     # Phase 4 deliverable
  embedder_compat.md                     # Phase 5 deliverable
  lightrag_integration_design.md         # Phase 6 deliverable
src/
  agent_knowledgebase/
    backends/
      __init__.py                        # factory + Protocol (Phase 2)
      chromadb_backend.py                # wraps existing chromadb logic (Phase 2)
      markdown_backend.py                # Karpathy-style wiki (Phase 3)
      lightrag_backend.py                # stub (Phase 6)
    services/
      migration.py                       # export/import (Phase 4)
tests/
  contract/
    test_decorator_order.py              # regression guard (Phase 0)
    test_probe4_response_shape.py        # all backends (Phase 2)
    test_stderr_schema.py                # 11-field schema (Phase 2)
    test_markdown_backend_probe4_shape.py # markdown probe-4 (Phase 3)
  test_get_service_config_missing.py     # Phase 0
  test_embedder_dimension_probe.py       # Phase 0
  test_chromadb_dimension_mismatch.py    # Phase 0
  test_run_server_bootstrap.py           # Phase 1
  test_run_server_failure_modes.py       # Phase 1
  test_retriever_backend_protocol.py     # Phase 2
  test_markdown_backend.py               # Phase 3
  test_migration_chromadb_to_markdown.py # Phase 4
  test_per_page_provider_snapshot.py     # Phase 4
  test_fastembed_recall_parity.py        # Phase 5
  test_fastembed_dimension_compat.py     # Phase 5
  test_embedder_version_mismatch.py      # Phase 5
  test_redesign_contract.py              # invariants stub (this scaffold)
requirements.lock                        # generated by pip-compile (Phase 1)
requirements.lock.sample                 # this scaffold
```

---

## 6. Working rules for subagents

- **Read the spec first.** Do not infer requirements from this AGENTS.md alone.
  This is a summary; the spec at `pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json`
  is canonical.
- **One phase per PR.** Do not bundle Phase 0 fixes with Phase 1 launcher work.
- **Branch from `auto_dev`**, not `master`. Open PRs against `master`.
- **Verify before claiming complete.** Run `ruff check` and `pytest -v`.
  Fix project-wide errors. Per-phase verification commands listed in §4.
- **Never run agents directly against `master`.** All agent work happens on
  feature branches. Per global CLAUDE.md.
- **Cite the spec_id in every commit message** so future archeology connects
  the change back to this design.
- **Read MISTAKES.md before editing** — it captures the pitfalls discovered
  during validation that are easy to repeat.
