"""SqlDatabaseIngestor -- introspect a database schema via SQLAlchemy.

Release 1 supports only the ``sqlite`` dialect.  Every sqlite engine is
constructed in SQLite's URI-mode with ``mode=ro`` so the source database
is pinned read-only at the driver layer; any ``INSERT`` / ``UPDATE`` /
``DELETE`` attempted against the engine raises
``sqlite3.OperationalError: attempt to write a readonly database``.

See ``docs/sql-database-uri-grammar.md`` for the pinned URI grammar and
the Windows absolute-path quoting rules this module enforces.
"""

from __future__ import annotations

from urllib.parse import urlsplit

from agent_knowledgebase.ingestors import ChunkConfig, Chunk, RawContent, token_chunk


# ---------------------------------------------------------------------------
# Read-only engine URL helper
# ---------------------------------------------------------------------------


def _build_readonly_sqlite_url(uri: str) -> str:
    """Convert a caller-supplied ``sqlite:///<path>`` URI to a read-only form.

    SQLAlchemy exposes SQLite's URI-mode via a special query-string shape:
    ``sqlite:///file:<path>?mode=ro&uri=true`` (see the SQLAlchemy sqlite
    dialect docs).  We use that exact shape because ``uri=true`` must
    appear in the URL's query string -- it is ignored if passed via
    ``create_engine.connect_args``.

    The helper:

    * Strips ingestor-layer query parameters (``table``, ``where``) from
      the caller's URI -- those are consumed by the ingestor itself and
      are not valid SQLite URI parameters.
    * Preserves Windows absolute paths.  On Windows, a URI of the form
      ``sqlite:///C:/Users/alice/cache.db`` has ``netloc=""`` and
      ``path="/C:/Users/alice/cache.db"``; the leading slash is kept so
      the produced ``file:/C:/...`` URI is unambiguous to SQLite.
    * Leaves non-sqlite URIs untouched -- Release 1 is sqlite-only but
      the ingestor stays forward-compatible.  Non-sqlite URIs reach
      ``create_engine`` unchanged and the caller is responsible for
      their read-only posture (e.g. Postgres with a read-only role).

    Examples:
        >>> _build_readonly_sqlite_url("sqlite:///tmp/cache.db")
        'sqlite:///file:/tmp/cache.db?mode=ro&uri=true'
        >>> _build_readonly_sqlite_url("sqlite:///C:/data/cache.db")
        'sqlite:///file:/C:/data/cache.db?mode=ro&uri=true'
        >>> _build_readonly_sqlite_url(
        ...     "sqlite:///tmp/cache.db?table=videos&where=id>1"
        ... )
        'sqlite:///file:/tmp/cache.db?mode=ro&uri=true'
    """
    split = urlsplit(uri)
    if split.scheme not in {"sqlite", "sqlite+pysqlite"}:
        return uri

    # In a sqlite:/// URL the absolute db path lives entirely in the path
    # component.  e.g. "sqlite:///C:/x.db" -> path="/C:/x.db".  We keep
    # the leading slash so the SQLite URI disambiguates absolute paths.
    db_path = split.path
    if not db_path:
        # "sqlite://" (in-memory with shared cache) or malformed; leave alone.
        return uri

    return f"{split.scheme}:///file:{db_path}?mode=ro&uri=true"


class SqlDatabaseIngestor:
    """Ingest schema and optional sample rows from a SQL database.

    Uses *SQLAlchemy* ``inspect()`` to enumerate tables, columns, and
    foreign-key relationships.  The URI must be a valid SQLAlchemy
    connection string (e.g. ``sqlite:///path/to/db`` or
    ``postgresql://user:pass@host/db``).

    Release 1 scope: only ``sqlite`` URIs are actively exercised, and
    those are transformed into SQLite URI-mode with ``mode=ro`` before
    the engine is constructed.  Non-sqlite URIs pass through unchanged.
    """

    # ------------------------------------------------------------------
    # Ingestor interface
    # ------------------------------------------------------------------

    def read(self, uri: str, metadata: dict | None = None) -> list[RawContent]:
        """Inspect the database at *uri* and return per-table content."""
        from sqlalchemy import create_engine, inspect, text

        meta = metadata or {}
        sample_rows: int = int(meta.get("sample_rows", 0))

        engine_url = _build_readonly_sqlite_url(uri)
        engine = create_engine(engine_url)
        inspector = inspect(engine)
        table_names = inspector.get_table_names()

        results: list[RawContent] = []
        for table in table_names:
            lines: list[str] = [f"Table: {table}", ""]

            # Columns
            columns = inspector.get_columns(table)
            lines.append("Columns:")
            for col in columns:
                nullable = "NULL" if col.get("nullable", True) else "NOT NULL"
                default = col.get("default", "")
                default_str = f" DEFAULT {default}" if default else ""
                lines.append(f"  - {col['name']}: {col['type']} {nullable}{default_str}")

            # Primary key
            pk = inspector.get_pk_constraint(table)
            if pk and pk.get("constrained_columns"):
                lines.append(f"\nPrimary Key: {', '.join(pk['constrained_columns'])}")

            # Foreign keys
            fks = inspector.get_foreign_keys(table)
            if fks:
                lines.append("\nForeign Keys:")
                for fk in fks:
                    cols = ", ".join(fk["constrained_columns"])
                    ref_table = fk["referred_table"]
                    ref_cols = ", ".join(fk["referred_columns"])
                    lines.append(f"  - ({cols}) -> {ref_table}({ref_cols})")

            # Optional sample rows
            if sample_rows > 0:
                with engine.connect() as conn:
                    rows = conn.execute(
                        text(f"SELECT * FROM {table} LIMIT :n"),  # noqa: S608
                        {"n": sample_rows},
                    ).fetchall()
                    if rows:
                        lines.append(f"\nSample rows ({len(rows)}):")
                        for row in rows:
                            lines.append(f"  {row}")

            content = "\n".join(lines)
            results.append(
                RawContent(
                    text=content,
                    metadata={
                        "connection_string": uri,
                        "table_name": table,
                        **(metadata or {}),
                    },
                )
            )

        engine.dispose()
        return results

    def chunk(self, contents: list[RawContent], config: ChunkConfig) -> list[Chunk]:
        """Chunk database content using token-window splitting."""
        return token_chunk(contents, config)
