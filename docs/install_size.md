# Install size — agent-knowledgebase v0.11.0

> Phase 5 deliverable (spec_id `70ab2170-381a-4657-bcd1-28a40c6f369b`).
> Source: `pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json`
> implementation.phases[5].tasks[8] + `docs/redesign/MISTAKES.md` M-03.

## TL;DR

**Default install ≈ 700 MB realistic floor.** Not the previously-claimed
~20 MB (off by ~35×) — `chromadb` alone pulls ~400 MB and the rest of
the runtime essentials (`httpx`, `pydantic`, `mcp`, `tiktoken`,
`trafilatura`, `gitpython`) round out the rest. Per `MISTAKES.md` M-03,
do not quote install-size numbers without enumerating which deps stay
in defaults vs which moved to extras.

The Phase 5 restructure moved `sentence-transformers`, `tree-sitter*`,
`pdfplumber`, `sqlalchemy`, and `sqlparse` from the default `[project]
dependencies` block into `[project.optional-dependencies]` extras.
Operators who don't need local embeddings, codebase ingest, PDF ingest,
or SQL ingest skip ~1.6 GB of optional weight.

## Default install (~700 MB)

| Package | ~Size | Why it stays in defaults |
|---|---|---|
| `chromadb` | ~400 MB | Default vectorstore. Bundled DuckDB + onnxruntime. |
| `mcp` (FastMCP) | ~50 MB | The MCP server contract. |
| `pydantic` + `pydantic-settings` | ~25 MB | Settings + model validation. |
| `httpx` | ~15 MB | Remote embedder + ingest. |
| `tiktoken` | ~10 MB | Token-based chunking. |
| `trafilatura` | ~150 MB | Default web-ingest extractor (lxml + dependencies). |
| `gitpython` | ~5 MB | git_history source ingestor. |
| transitive | ~50 MB | requests, certifi, idna, etc. |
| **Default total** | **~700 MB** | — |

## Optional extras

Install with `pip install agent-knowledgebase[<extra>]`. The `all`
convenience extra pulls every group at once.

### `embed-local-st` — sentence-transformers (~1.3 GB)

| Package | ~Size | Why opt-in |
|---|---|---|
| `sentence-transformers` | ~150 MB | Library itself. |
| `torch` (transitive) | ~1.0 GB | Required runtime. |
| `transformers` (transitive) | ~150 MB | Tokenizers + model loaders. |
| **Extra adds** | **~1.3 GB** | — |

When to install: you want a local fp32 embedder via the
`SentenceTransformerEmbedder` provider (`embedding_provider='sentence-transformers'`).

### `embed-local-onnx` — qdrant-fastembed (~150 MB)

| Package | ~Size | Why opt-in |
|---|---|---|
| `fastembed` | ~30 MB | Library + bundled model loaders. |
| `onnxruntime` (transitive) | ~120 MB | int8 ONNX inference. |
| **Extra adds** | **~150 MB** | — |

When to install: you want a small, fast local embedder via the
`FastembedEmbedder` provider (`embedding_provider='fastembed'`).
Roughly an order of magnitude smaller than the
sentence-transformers extra. Note that fastembed-MiniLM-int8 vectors
are NOT interchangeable with sentence-transformers-MiniLM-fp32
vectors — Phase 5 enforces this via per-chunk `embedder_version`
stamping.

### `ingest-codebase` — placeholder (0 MB)

| Package | ~Size | Why opt-in |
|---|---|---|
| _(none)_ | 0 MB | Codebase ingestion uses regex-based symbol detection — no tree-sitter dependency in v0.11.0. |
| **Extra adds** | **0 MB** | — |

The `[ingest-codebase]` extra is retained as an empty placeholder for a
future tree-sitter-backed chunker. Today, `source_type=codebase`
ingestion needs no extra dependency — symbols are detected via regex.
Installing this extra is a no-op on v0.11.0.

### `ingest-file` — pdfplumber (~30 MB)

| Package | ~Size | Why opt-in |
|---|---|---|
| `pdfplumber` | ~10 MB | PDF text extraction. |
| `pdfminer.six` (transitive) | ~20 MB | Underlying PDF parser. |
| **Extra adds** | **~30 MB** | — |

When to install: you ingest PDFs via `source_type=file`. Plain-text and
markdown files don't need this extra.

### `ingest-sql` — sqlalchemy + sqlparse (~50 MB)

| Package | ~Size | Why opt-in |
|---|---|---|
| `sqlalchemy` | ~30 MB | DB engine abstraction (used in read-only mode). |
| `sqlparse` | ~5 MB | Required by the SQL-injection AST validator. |
| transitive | ~15 MB | greenlet, etc. |
| **Extra adds** | **~50 MB** | — |

When to install: you ingest `source_type=sql_database`. The Phase 2
where-clause AST validator imports `sqlparse`; without this extra the
SQL ingestor module raises `ImportError` on first use.

### `pinecone` — managed vectorstore (~10 MB)

| Package | ~Size | Why opt-in |
|---|---|---|
| `pinecone-client` | ~10 MB | Pinecone REST/gRPC client. |
| **Extra adds** | **~10 MB** | — |

When to install: you set `vectorstore='pinecone'`. Default chromadb
needs neither this nor any other extra.

## Cumulative table

| Install set | Total | vs default |
|---|---|---|
| Default | ~700 MB | baseline |
| Default + `embed-local-onnx` | ~850 MB | +150 MB |
| Default + `embed-local-st` | ~2.0 GB | +1.3 GB |
| Default + `ingest-codebase` | ~700 MB | +0 MB (placeholder) |
| Default + `ingest-file` | ~730 MB | +30 MB |
| Default + `ingest-sql` | ~750 MB | +50 MB |
| `all` | ~2.0 GB | +1.3 GB |

## Regenerating `requirements.lock`

The committed `requirements.lock.sample` carries the v0.6.0 wide
default. The Phase 5 narrower default needs a fresh lock generated
against the new `[project] dependencies` block. Generation command:

```bash
# preferred — uv (fast, deterministic)
uv pip compile pyproject.toml --generate-hashes -o requirements.lock

# fallback — pip-tools
pip install pip-tools
pip-compile --generate-hashes -o requirements.lock pyproject.toml
```

If neither tool is on the PATH at release-cut time, the lock file
regeneration is acceptable to defer as a follow-up — the
pyproject.toml dependency list is the single source of truth for
runtime resolution. CHANGELOG calls out the deferred regen explicitly
so operators can re-run the command from a clean checkout when needed.

## Measurement methodology

Sizes above are estimates from a fresh `pip install` against an empty
venv on Python 3.11 / Linux x86_64, rounded to the nearest 5 MB.
Windows + macOS will differ by ~10% due to pre-built wheel layouts.

```bash
# To re-measure on your platform:
python -m venv /tmp/agent-kb-measure
/tmp/agent-kb-measure/bin/pip install --no-cache-dir .
du -sh /tmp/agent-kb-measure/lib/python*/site-packages
```

Re-measure after every dependency change. This file is regenerated as
part of the v0.11.0 release; subsequent releases that touch
dependencies should refresh the tables here.
