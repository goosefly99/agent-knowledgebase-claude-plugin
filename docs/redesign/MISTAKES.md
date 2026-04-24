# MISTAKES.md — agent-knowledgebase Redesign (spec_id `70ab2170-381a-4657-bcd1-28a40c6f369b`)

> Per global CLAUDE.md `Self-Improvement Loop` rule:
>
> > When a mistake is made during a task, append a one-sentence note to
> > `MISTAKES.md` in the project root describing what happened and what the
> > correct approach is. At the start of each new session, read `MISTAKES.md`
> > before doing any work, if it exists.
>
> This file is the redesign-scoped log. The project root may also have its own
> `MISTAKES.md`; this one specifically captures pitfalls discovered during the
> v2.0 → v2.1 spec validation pass against
> `pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json`.
>
> **Read this file before starting any redesign-related task.**

---

## Standing rules (validated 2026-04-24)

### M-01 — Decorator-order claims require empirical verification

> **Don't claim a decorator is a NO-OP without verifying decorator-evaluation
> order in `tests/test_tool_timeout.py` first.**

The v2.0 spec draft asserted that `@mcp.tool()` outside `@_with_tool_timeout`
made the timeout dead code at the MCP boundary. This was wrong.
`tests/test_tool_timeout.py` has been passing 6/6 the entire time — including
`test_every_registered_mcp_tool_is_wrapped` which empirically asserts
`hasattr(fn, '__wrapped__')` for every registered tool. A live timeout test
on a slow tool returns the structured `{"error":"tool_timeout", ...}` payload,
proving the wrap engages.

**Correct approach:** before claiming any decorator pattern is broken, run the
relevant test file. If the test passes, the pattern works. If you must investigate
order, use `python -c "import inspect; from agent_knowledgebase.server import mcp;
fn = mcp._tool_manager._tools['kb_create'].fn; print(hasattr(fn, '__wrapped__'),
fn.__wrapped__.__name__)"` to inspect the registered function chain.

The corollary: **DO NOT swap `@mcp.tool()` and `@_with_tool_timeout`.** Their
current order (`@mcp.tool()` outer, `@_with_tool_timeout` inner) is the working
order. Inverting them would cause FastMCP to register the raw function and
make the timeout dead code at the MCP boundary — flipping a working setup into a
broken one. Validation finding f-21 specifically caught a v2.0 MISTAKES.md entry
that, if followed, would have produced this regression.

Source: validation findings f-05, f-13, f-18, f-21.

---

### M-02 — Don't ship a default-flip phase before its read-fallback phase

> **Don't ship a default-flip phase before its read-fallback phase.**

The v2.0 spec ordered Phase 4 (embedding default-flip to remote/fastembed) BEFORE
Phase 5 (data migration with read-fallback semantics). Existing v0.6.0 KBs were
ingested with the Ollama default `qwen3-embedding:8b` (4096-dim). Flipping the
global default to `text-embedding-3-small` (1536-dim) without read-fallback in
production would cause every existing KB to dimension-mismatch on the first
`kb_query` after upgrade.

**Correct approach (v2.1):** read-fallback semantics + per-page
`(embedding_provider, embed_base_url)` snapshot ship FIRST (now Phase 4). Only
then can the global default be flipped (now Phase 5). Phase 5 has an explicit
gate task: confirm Phase 4 backfill is in production before merging the
default-flip PR.

Generalize: any change that flips a global default which is stamped per-record
must be preceded by:

1. A backfill that ensures every existing record carries enough metadata to
   reconstruct the OLD behavior.
2. A read-fallback that uses that per-record metadata at read time.
3. A test that demonstrates an existing record continues to work after the
   default change.

Source: validation findings f-10, f-17.

---

### M-03 — Don't quote install-size numbers without enumerating which deps stay/move

> **Don't quote install-size numbers without enumerating which deps stay/move.**

The v2.0 spec claimed default install size would drop from `~2GB` to `~20MB`
(remote-only) or `~100MB` (fastembed-local opt-in). Realistic floor is ~700MB
because `chromadb` (~400MB) + `tree-sitter-languages` (~100MB) +
`onnxruntime-via-fastembed` (~150MB if opted in) + everything else
(`httpx`, `pydantic`, `mcp`, `tiktoken`, `trafilatura`) stay in the default
install. The `~20MB` claim was off by ~35x.

**Correct approach:** when quoting install-size deltas, enumerate exactly which
packages stay in the default install and which move to extras. Per v2.1:

| Default install (~700MB) | Optional extras |
|--------------------------|------------------|
| chromadb (~400MB) | extras['embed-local-onnx'] = qdrant-fastembed (+150MB) |
| tree-sitter-languages (~100MB) | extras['embed-local-st'] = sentence-transformers (+1.3GB) |
| httpx, pydantic, mcp, tiktoken, trafilatura (~rest) | extras['ingest-codebase'] = tree-sitter, gitpython |
| | extras['ingest-file'] = pdfplumber |
| | extras['ingest-sql'] = sqlalchemy, sqlparse |

Re-measure after every dependency change. `docs/install_size.md` is a Phase 5
deliverable that gets regenerated from a fresh `pip install` measurement, not
hand-edited.

Source: validation finding f-11.

---

## How to add new entries

1. Encounter a mistake → fix it → write a one-sentence note here describing
   what happened and what the correct approach is.
2. If the same mistake recurs despite an entry here, promote the rule into
   `docs/redesign/AGENTS.md` or the project-root CLAUDE.md as a standing
   rule. `MISTAKES.md` is a log; AGENTS.md / CLAUDE.md are the durable
   control surface.

Each entry header format: `### M-NN — One-line summary`. Subsequent paragraphs
explain what went wrong, what the correct approach is, and a source reference
(validation finding ID, debate round, or PR link).
