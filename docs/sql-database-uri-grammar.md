# sql_database URI grammar

Pinned URI grammar for `kb_ingest` / `kb_ingest_batch` when
`source_type='sql_database'`. Release 1 is **sqlite-only**.

## Shape

```
sqlite:///<absolute-path>?table=<table-or-view>&where=<where-clause>
```

- `table` and `where` are **ingestor-layer** query parameters. They are
  stripped by the ingestor before the SQLite URI is handed to the driver.
- The `where` clause is independently validated by the sqlparse AST
  allow-list in `services/where_clause_validator.py` (see `CODE_*`
  rejection codes).

## Read-only guarantee

The ingestor rewrites every incoming `sqlite:///...` URI to SQLite URI-mode
with `mode=ro` before calling `create_engine`:

```
sqlite:///<path>               ->   sqlite:///file:<path>?mode=ro&uri=true
```

Writes against the resulting engine raise
`OperationalError: attempt to write a readonly database`. This is a
defense-in-depth pin on top of the where-clause validator -- even if a
clause sneaks past the validator, the engine cannot mutate the source DB.

## Windows absolute-path quoting

Windows absolute paths must use **forward slashes** and **no drive-colon
escaping**. The leading `///` in `sqlite:///` represents the required
empty authority followed by the single absolute-path slash; the drive
letter follows directly.

**Correct**

```
sqlite:///C:/Users/alice/mcp-cache/youtube.db?table=videos&where=published_at>'2026-01-01'
```

**Anti-pattern (do not use)**

```
sqlite:///C:\Users\alice\mcp-cache\youtube.db     # backslashes -- broken
sqlite:////C:/Users/alice/mcp-cache/youtube.db    # four slashes -- broken
sqlite://C:/Users/alice/mcp-cache/youtube.db      # two slashes, drive as host -- broken
```

URL-encode any special characters in the `where` clause (e.g. `>` as
`%3E`, spaces as `%20`). The `table` value should be a bare identifier
with no encoding needed.
