"""SqlDatabaseIngestor -- introspect a database schema via SQLAlchemy."""

from __future__ import annotations

from agent_knowledgebase.ingestors import ChunkConfig, Chunk, RawContent, token_chunk


class SqlDatabaseIngestor:
    """Ingest schema and optional sample rows from a SQL database.

    Uses *SQLAlchemy* ``inspect()`` to enumerate tables, columns, and
    foreign-key relationships.  The URI must be a valid SQLAlchemy
    connection string (e.g. ``sqlite:///path/to/db`` or
    ``postgresql://user:pass@host/db``).
    """

    # ------------------------------------------------------------------
    # Ingestor interface
    # ------------------------------------------------------------------

    def read(self, uri: str, metadata: dict | None = None) -> list[RawContent]:
        """Inspect the database at *uri* and return per-table content."""
        from sqlalchemy import create_engine, inspect, text

        meta = metadata or {}
        sample_rows: int = int(meta.get("sample_rows", 0))

        engine = create_engine(uri)
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
