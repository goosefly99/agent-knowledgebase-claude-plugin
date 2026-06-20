# tests/integration/

Developer-only end-to-end smoke + diagnostic harness for `agent-knowledgebase`.

These scripts are **not** collected by pytest (filenames start with `smoke_` /
`probe_`, not `test_`). They drive the real MCP server and/or the in-process
`KnowledgebaseService` against a real embedder and a real persistent KB —
useful for reproducing concurrency / timeout / cold-load bugs that unit
tests with mocked vector stores cannot exercise.

## Required environment

| Variable                | Purpose                                                  | Required by         |
| ----------------------- | -------------------------------------------------------- | ------------------- |
| `AGENT_KB_SAVES_DIR`    | Persistent saves directory (chromadb + sqlite live here) | All smoke scripts   |
| `AGENT_KB_TEST_KB_ID`   | UUID of the KB to drive. If unset, resolved by name.     | Optional override   |
| `AGENT_KB_TEST_KB_NAME` | KB name to look up when `AGENT_KB_TEST_KB_ID` is unset.  | Defaults `claude-rag` |
| `AGENT_KB_TEST_YT_DB`   | Path to a YouTube cache sqlite (for `smoke_02` only)     | `smoke_02_*` only   |

`CLAUDE_PLUGIN_ROOT` is auto-resolved by each script from `__file__`; you do
not need to set it manually.

## Workflow

The numeric prefix indicates the order the scripts were originally run during
bug-investigation. Most are independent once a KB exists; you typically only
need `smoke_01` first to provision the KB.

```
smoke_01_create_kb.py            create the KB (idempotent)
smoke_02_export_transcripts.py   regenerate the fixture corpus from a YT cache DB (optional)
smoke_03_ingest_single.py        ingest the smallest transcript via MCP
smoke_03b_ingest_retry.py        retry an ingest after a prior failure
smoke_03c_check_state.py         read-only state inspection
smoke_04_direct_ingest.py        in-process ingest with timing (bypass MCP)
smoke_05_mcp_ingest_one.py       single MCP ingest with timing
smoke_06_bulk_ingest.py          ingest all 15 fixture transcripts in-process
smoke_07_query.py                in-process query verification
smoke_08_mcp_min_repro.py        MCP _with_tool_timeout deadlock min repro
smoke_09_mcp_two_calls.py        cold/warm chromadb HNSW load hypothesis
smoke_10_query_claude_obsidian.py vector vs keyword retrieval comparison via MCP
smoke_11_query_diagnostic.py     two-call kb_query + in-process kb_search traceback capture
smoke_12_kb_search_isolated.py   isolate the kb_search 'RustBindingsAPI' error
smoke_13_top_hits_full.py        print full content of top hits
probe_mcp_tools_list.py          MCP `tools/list` capability probe
```

## Running

From the plugin root:

```bash
export AGENT_KB_SAVES_DIR=/path/to/saves
.venv/bin/python tests/integration/smoke_01_create_kb.py
```

Each script prints a progress trace to stdout. Errors and the captured server
stderr (when applicable) are written to stdout for easy copy-paste.

## Fixtures

`fixtures/transcripts/` holds 15 YouTube transcripts (~530 KB total) used as
the canonical ingest corpus. They were the input that surfaced the original
180s cold-load + RustBindings bugs; keeping them in-tree makes the regression
harness deterministic. `fixtures/transcripts_manifest.json` lists each file's
`video_id`, `path` (relative to its own directory), and `size`.

## Bug references

| Diagnostic script           | Bug it targets                                                  |
| --------------------------- | --------------------------------------------------------------- |
| `smoke_04_direct_ingest`    | Confirms in-process ingest finishes in ~3s (control)            |
| `smoke_08_mcp_min_repro`    | MCP `_with_tool_timeout` 180s deadlock (single-call repro)      |
| `smoke_09_mcp_two_calls`    | Cold/warm chromadb HNSW load hypothesis                         |
| `smoke_11_query_diagnostic` | First-call timeout on `kb_query` + `kb_search` traceback capture |
| `smoke_12_kb_search_isolated` | Isolate the `RustBindingsAPI` chromadb side-effect            |
