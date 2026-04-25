"""SQLite database layer for agent-knowledgebase."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from agent_knowledgebase.models import (
    Chunk,
    Knowledgebase,
    PipelineRun,
    Source,
    WikiPage,
)

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS knowledgebases (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    config TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    kb_id TEXT NOT NULL REFERENCES knowledgebases(id),
    source_type TEXT NOT NULL,
    uri TEXT NOT NULL,
    metadata TEXT,
    ingested_at TEXT,
    chunk_count INTEGER DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    dedup_key TEXT
);

CREATE TABLE IF NOT EXISTS wiki_pages (
    id TEXT PRIMARY KEY,
    kb_id TEXT NOT NULL REFERENCES knowledgebases(id),
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    page_type TEXT NOT NULL,
    tags TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS wiki_links (
    from_page_id TEXT NOT NULL REFERENCES wiki_pages(id),
    to_page_id TEXT NOT NULL REFERENCES wiki_pages(id),
    PRIMARY KEY (from_page_id, to_page_id)
);

CREATE TABLE IF NOT EXISTS page_sources (
    page_id TEXT NOT NULL REFERENCES wiki_pages(id),
    source_id TEXT NOT NULL REFERENCES sources(id),
    PRIMARY KEY (page_id, source_id)
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(id),
    kb_id TEXT NOT NULL REFERENCES knowledgebases(id),
    content TEXT NOT NULL,
    metadata TEXT,
    embedding_id TEXT
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id TEXT PRIMARY KEY,
    kb_id TEXT NOT NULL REFERENCES knowledgebases(id),
    source_id TEXT REFERENCES sources(id),
    phase TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    started_at TEXT NOT NULL,
    completed_at TEXT,
    error TEXT,
    metadata TEXT
);
"""

_FTS_SQL = """\
CREATE VIRTUAL TABLE IF NOT EXISTS wiki_pages_fts USING fts5(
    title, content, tags,
    content=wiki_pages,
    content_rowid=rowid
);
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso(dt: Optional[datetime]) -> Optional[str]:
    """Convert a datetime to ISO-8601 string, or return *None*."""
    if dt is None:
        return None
    return dt.isoformat()


def _json_dumps(obj: object) -> Optional[str]:
    if obj is None:
        return None
    return json.dumps(obj)


def _json_loads(text: Optional[str]) -> Any:
    if text is None:
        return None
    return json.loads(text)


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------


class Database:
    """Thin wrapper around :mod:`sqlite3` for agent-knowledgebase storage.

    Parameters
    ----------
    db_path:
        Filesystem path for the SQLite database.  Parent directories are
        created automatically.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False lets a connection built on thread A be
        # reused from thread B (e.g., the MCP tool-timeout worker thread
        # created lazily on the first @_with_tool_timeout call).  MCP's
        # stdio transport serialises tool calls, and the per-KB
        # ingestion lock in KnowledgebaseService serialises writes, so
        # only one caller uses the connection at a time.
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(_SCHEMA_SQL)
        # FTS virtual table must be created outside executescript in some
        # SQLite builds, so we run it separately.
        cur.executescript(_FTS_SQL)
        self._conn.commit()
        # Additive migration: add dedup_key column to pre-existing databases
        # that were created before this column was added to the schema DDL.
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(sources)").fetchall()}
        if "dedup_key" not in cols:
            self._conn.execute("ALTER TABLE sources ADD COLUMN dedup_key TEXT")
            self._conn.commit()
        # v0.6.0 additive migration: the kb-pipeline-status telemetry row
        # adds 9 new columns to `pipeline_runs`. Additive-only; running
        # _init_schema twice is a no-op (guarded by PRAGMA column probe).
        # Rollback SQL is documented in CHANGELOG.md.
        run_cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(pipeline_runs)").fetchall()
        }
        _PIPELINE_RUN_MIGRATION_COLUMNS: list[tuple[str, str]] = [
            ("ended_at", "TEXT"),
            ("ingested", "INTEGER"),
            ("skipped", "INTEGER"),
            ("replaced", "INTEGER"),
            ("failed", "INTEGER"),
            ("batch_size", "INTEGER"),
            ("dedup_policy", "TEXT"),
            ("request_id", "TEXT"),
            ("tool_caller_version", "TEXT"),
        ]
        added_any = False
        for col_name, col_type in _PIPELINE_RUN_MIGRATION_COLUMNS:
            if col_name not in run_cols:
                self._conn.execute(
                    f"ALTER TABLE pipeline_runs ADD COLUMN {col_name} {col_type}"
                )
                added_any = True
        if added_any:
            self._conn.commit()
        # v0.10.0 / Phase 4 additive migration: per-page provider snapshot.
        # `chunks` table gains `embedding_provider` and `embed_base_url`
        # columns (both nullable TEXT) so an existing v0.6.0 KB ingested
        # under provider=ollama/base_url=http://127.0.0.1:11434 stays
        # queryable after the Phase 5 default flip to remote/text-
        # embedding-3-small. The snapshot is read at query time by
        # ``create_embedder_for_model(model_name, provider=..., base_url=...)``
        # so the original embedder is faithfully rebuilt even after the
        # global Settings defaults move on. Strictly ALTER ADD —
        # non-destructive; rollback SQL documented in CHANGELOG.md.
        #
        # v0.11.0 / Phase 5 additive migration: per-chunk
        # ``embedder_version`` column. Stamped at ingest time by
        # ``KnowledgebaseService._ingest_source_locked`` so a KB whose
        # chunks were ingested under e.g. ``fastembed/MiniLM-L6-v2-int8``
        # cannot be silently mixed with chunks ingested under
        # ``sentence-transformers/all-MiniLM-L6-v2@hf-fp32`` — same
        # nominal model_name, different vector geometry. The mixed-
        # version rejection at ingest time uses
        # :meth:`get_embedder_versions` (below); the
        # ``AGENT_KB_AUTO_REEMBED=1`` env var is the documented
        # bypass. Strictly ALTER ADD — non-destructive — and idempotent
        # across re-opens. spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b
        chunk_cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(chunks)").fetchall()
        }
        _CHUNK_PROVIDER_SNAPSHOT_COLUMNS: list[tuple[str, str]] = [
            ("embedding_provider", "TEXT"),
            ("embed_base_url", "TEXT"),
            # Phase 5 (v0.11.0):
            ("embedder_version", "TEXT"),
        ]
        added_provider_snapshot = False
        for col_name, col_type in _CHUNK_PROVIDER_SNAPSHOT_COLUMNS:
            if col_name not in chunk_cols:
                self._conn.execute(
                    f"ALTER TABLE chunks ADD COLUMN {col_name} {col_type}"
                )
                added_provider_snapshot = True
        if added_provider_snapshot:
            self._conn.commit()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    # ------------------------------------------------------------------
    # Knowledgebase CRUD
    # ------------------------------------------------------------------

    def insert_knowledgebase(self, kb: Knowledgebase) -> None:
        self._conn.execute(
            "INSERT INTO knowledgebases (id, name, description, config, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                kb.id,
                kb.name,
                kb.description,
                _json_dumps(kb.config),
                _iso(kb.created_at),
                _iso(kb.updated_at),
            ),
        )
        self._conn.commit()

    def get_knowledgebase(self, id: str) -> Optional[Knowledgebase]:
        row = self._conn.execute(
            "SELECT * FROM knowledgebases WHERE id = ?", (id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_knowledgebase(row)

    def list_knowledgebases(self) -> list[Knowledgebase]:
        rows = self._conn.execute(
            "SELECT * FROM knowledgebases ORDER BY created_at"
        ).fetchall()
        return [self._row_to_knowledgebase(r) for r in rows]

    def delete_knowledgebase(self, id: str) -> None:
        self._conn.execute("DELETE FROM knowledgebases WHERE id = ?", (id,))
        self._conn.commit()

    @staticmethod
    def _row_to_knowledgebase(row: sqlite3.Row) -> Knowledgebase:
        return Knowledgebase(
            id=row["id"],
            name=row["name"],
            description=row["description"] or "",
            config=_json_loads(row["config"]) or {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # ------------------------------------------------------------------
    # Source CRUD
    # ------------------------------------------------------------------

    def insert_source(self, source: Source) -> None:
        self._conn.execute(
            "INSERT INTO sources "
            "(id, kb_id, source_type, uri, metadata, ingested_at, chunk_count, status, dedup_key) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                source.id,
                source.kb_id,
                source.source_type.value,
                source.uri,
                _json_dumps(source.metadata),
                _iso(source.ingested_at),
                source.chunk_count,
                source.status.value,
                source.dedup_key,
            ),
        )
        self._conn.commit()

    def get_source(self, id: str) -> Optional[Source]:
        row = self._conn.execute("SELECT * FROM sources WHERE id = ?", (id,)).fetchone()
        if row is None:
            return None
        return self._row_to_source(row)

    def list_sources(self, kb_id: str) -> list[Source]:
        rows = self._conn.execute(
            "SELECT * FROM sources WHERE kb_id = ?", (kb_id,)
        ).fetchall()
        return [self._row_to_source(r) for r in rows]

    def delete_source(self, id: str) -> None:
        self._conn.execute("DELETE FROM sources WHERE id = ?", (id,))
        self._conn.commit()

    @staticmethod
    def _row_to_source(row: sqlite3.Row) -> Source:
        return Source(
            id=row["id"],
            kb_id=row["kb_id"],
            source_type=row["source_type"],
            uri=row["uri"],
            metadata=_json_loads(row["metadata"]) or {},
            ingested_at=row["ingested_at"],
            chunk_count=row["chunk_count"],
            status=row["status"],
            dedup_key=row["dedup_key"],
        )

    def find_source_by_dedup_key(self, kb_id: str, dedup_key: str) -> Optional[Source]:
        """Return the first Source in *kb_id* whose dedup_key matches, or None."""
        row = self._conn.execute(
            "SELECT * FROM sources WHERE kb_id = ? AND dedup_key = ? LIMIT 1",
            (kb_id, dedup_key),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_source(row)

    # ------------------------------------------------------------------
    # WikiPage CRUD
    # ------------------------------------------------------------------

    def insert_wiki_page(self, page: WikiPage) -> None:
        self._conn.execute(
            "INSERT INTO wiki_pages "
            "(id, kb_id, title, content, page_type, tags, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                page.id,
                page.kb_id,
                page.title,
                page.content,
                page.page_type.value,
                _json_dumps(page.tags),
                _iso(page.created_at),
                _iso(page.updated_at),
            ),
        )
        # Sync FTS index.
        self._conn.execute(
            "INSERT INTO wiki_pages_fts (rowid, title, content, tags) "
            "SELECT rowid, title, content, ? FROM wiki_pages WHERE id = ?",
            (_json_dumps(page.tags), page.id),
        )
        # Store source links.
        for source_id in page.source_ids:
            self._conn.execute(
                "INSERT OR IGNORE INTO page_sources (page_id, source_id) VALUES (?, ?)",
                (page.id, source_id),
            )
        # Store outbound wiki links.
        for target_id in page.outbound_links:
            self._conn.execute(
                "INSERT OR IGNORE INTO wiki_links (from_page_id, to_page_id) VALUES (?, ?)",
                (page.id, target_id),
            )
        self._conn.commit()

    def get_wiki_page(self, id: str) -> Optional[WikiPage]:
        row = self._conn.execute(
            "SELECT * FROM wiki_pages WHERE id = ?", (id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_wiki_page(row)

    def list_wiki_pages(self, kb_id: str) -> list[WikiPage]:
        rows = self._conn.execute(
            "SELECT * FROM wiki_pages WHERE kb_id = ?", (kb_id,)
        ).fetchall()
        return [self._row_to_wiki_page(r) for r in rows]

    def update_wiki_page(self, page: WikiPage) -> None:
        """Update an existing wiki page and sync FTS."""
        # Delete old FTS entry.
        self._conn.execute(
            "DELETE FROM wiki_pages_fts WHERE rowid = "
            "(SELECT rowid FROM wiki_pages WHERE id = ?)",
            (page.id,),
        )
        # Update the page.
        self._conn.execute(
            "UPDATE wiki_pages SET title=?, content=?, page_type=?, tags=?, updated_at=? WHERE id=?",
            (
                page.title,
                page.content,
                page.page_type.value,
                _json_dumps(page.tags),
                _iso(page.updated_at),
                page.id,
            ),
        )
        # Re-insert FTS entry.
        self._conn.execute(
            "INSERT INTO wiki_pages_fts(rowid, title, content, tags) "
            "SELECT rowid, title, content, tags FROM wiki_pages WHERE id = ?",
            (page.id,),
        )
        self._conn.commit()

    def get_wiki_page_by_title(self, kb_id: str, title: str) -> Optional[WikiPage]:
        """Find a wiki page by its title within a KB."""
        row = self._conn.execute(
            "SELECT * FROM wiki_pages WHERE kb_id = ? AND title = ?",
            (kb_id, title),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_wiki_page(row)

    def list_wiki_pages_by_type(self, kb_id: str, page_type: str) -> list[WikiPage]:
        """List wiki pages in a KB filtered by page type."""
        rows = self._conn.execute(
            "SELECT * FROM wiki_pages WHERE kb_id = ? AND page_type = ?",
            (kb_id, page_type),
        ).fetchall()
        return [self._row_to_wiki_page(r) for r in rows]

    def delete_wiki_page(self, id: str) -> None:
        # Remove FTS entry.
        self._conn.execute(
            "DELETE FROM wiki_pages_fts WHERE rowid = "
            "(SELECT rowid FROM wiki_pages WHERE id = ?)",
            (id,),
        )
        # Remove link/source associations.
        self._conn.execute(
            "DELETE FROM wiki_links WHERE from_page_id = ? OR to_page_id = ?", (id, id)
        )
        self._conn.execute("DELETE FROM page_sources WHERE page_id = ?", (id,))
        self._conn.execute("DELETE FROM wiki_pages WHERE id = ?", (id,))
        self._conn.commit()

    def _row_to_wiki_page(self, row: sqlite3.Row) -> WikiPage:
        page_id: str = row["id"]
        source_ids = [
            r["source_id"]
            for r in self._conn.execute(
                "SELECT source_id FROM page_sources WHERE page_id = ?", (page_id,)
            ).fetchall()
        ]
        inbound = [
            r["from_page_id"]
            for r in self._conn.execute(
                "SELECT from_page_id FROM wiki_links WHERE to_page_id = ?", (page_id,)
            ).fetchall()
        ]
        outbound = [
            r["to_page_id"]
            for r in self._conn.execute(
                "SELECT to_page_id FROM wiki_links WHERE from_page_id = ?", (page_id,)
            ).fetchall()
        ]
        return WikiPage(
            id=page_id,
            kb_id=row["kb_id"],
            title=row["title"],
            content=row["content"],
            page_type=row["page_type"],
            tags=_json_loads(row["tags"]) or [],
            source_ids=source_ids,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            inbound_links=inbound,
            outbound_links=outbound,
        )

    # ------------------------------------------------------------------
    # Wiki links
    # ------------------------------------------------------------------

    def add_wiki_link(self, from_id: str, to_id: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO wiki_links (from_page_id, to_page_id) VALUES (?, ?)",
            (from_id, to_id),
        )
        self._conn.commit()

    def remove_wiki_link(self, from_id: str, to_id: str) -> None:
        self._conn.execute(
            "DELETE FROM wiki_links WHERE from_page_id = ? AND to_page_id = ?",
            (from_id, to_id),
        )
        self._conn.commit()

    def get_wiki_links(self, page_id: str, direction: Literal["inbound", "outbound"] = "outbound") -> list[str]:
        """Return linked page IDs.

        Parameters
        ----------
        page_id:
            The page whose links are queried.
        direction:
            ``"outbound"`` (default) or ``"inbound"``.
        """
        if direction == "inbound":
            rows = self._conn.execute(
                "SELECT from_page_id FROM wiki_links WHERE to_page_id = ?", (page_id,)
            ).fetchall()
            return [r["from_page_id"] for r in rows]
        # outbound
        rows = self._conn.execute(
            "SELECT to_page_id FROM wiki_links WHERE from_page_id = ?", (page_id,)
        ).fetchall()
        return [r["to_page_id"] for r in rows]

    # ------------------------------------------------------------------
    # Page-source associations
    # ------------------------------------------------------------------

    def add_page_source(self, page_id: str, source_id: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO page_sources (page_id, source_id) VALUES (?, ?)",
            (page_id, source_id),
        )
        self._conn.commit()

    def get_page_sources(self, page_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT source_id FROM page_sources WHERE page_id = ?", (page_id,)
        ).fetchall()
        return [r["source_id"] for r in rows]

    def remove_page_source(self, page_id: str, source_id: str) -> None:
        """Remove a page-source association."""
        self._conn.execute(
            "DELETE FROM page_sources WHERE page_id = ? AND source_id = ?",
            (page_id, source_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Chunk CRUD
    # ------------------------------------------------------------------

    def insert_chunk(self, chunk: Chunk) -> None:
        # Phase 4: stamp the per-page provider snapshot into native
        # columns when present in chunk.metadata so the read-path's
        # snapshot resolution (get_embedding_snapshot) sees non-NULL
        # column values instead of having to fall back to the metadata
        # blob. The metadata bag is left intact so older readers keep
        # working unchanged. spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b
        #
        # Phase 5 (v0.11.0): also stamp ``embedder_version`` so the
        # mixed-version rejection in
        # ``KnowledgebaseService._ingest_source_locked`` can compare
        # the incoming embedder against the kb's existing embedders
        # cheaply via :meth:`get_embedder_versions`.
        meta = chunk.metadata if isinstance(chunk.metadata, dict) else {}
        embedding_provider = meta.get("embedding_provider")
        embed_base_url = meta.get("embed_base_url")
        embedder_version = meta.get("embedder_version")
        self._conn.execute(
            "INSERT INTO chunks (id, source_id, kb_id, content, metadata, "
            "embedding_id, embedding_provider, embed_base_url, embedder_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                chunk.id,
                chunk.source_id,
                chunk.kb_id,
                chunk.content,
                _json_dumps(chunk.metadata),
                chunk.embedding_id,
                embedding_provider,
                embed_base_url,
                embedder_version,
            ),
        )
        self._conn.commit()

    def get_chunk(self, id: str) -> Optional[Chunk]:
        row = self._conn.execute("SELECT * FROM chunks WHERE id = ?", (id,)).fetchone()
        if row is None:
            return None
        return self._row_to_chunk(row)

    def list_chunks(self, source_id: str) -> list[Chunk]:
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE source_id = ?", (source_id,)
        ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def delete_chunk(self, id: str) -> None:
        self._conn.execute("DELETE FROM chunks WHERE id = ?", (id,))
        self._conn.commit()

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            id=row["id"],
            source_id=row["source_id"],
            kb_id=row["kb_id"],
            content=row["content"],
            metadata=_json_loads(row["metadata"]) or {},
            embedding_id=row["embedding_id"],
        )

    # ------------------------------------------------------------------
    # PipelineRun CRUD
    # ------------------------------------------------------------------

    def insert_pipeline_run(self, run: PipelineRun) -> None:
        self._conn.execute(
            "INSERT INTO pipeline_runs "
            "(id, kb_id, source_id, phase, status, started_at, completed_at, error, metadata, "
            "ended_at, ingested, skipped, replaced, failed, batch_size, dedup_policy, "
            "request_id, tool_caller_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run.id,
                run.kb_id,
                run.source_id,
                run.phase.value,
                run.status.value,
                _iso(run.started_at),
                _iso(run.completed_at),
                run.error,
                _json_dumps(run.metadata),
                _iso(run.ended_at),
                run.ingested,
                run.skipped,
                run.replaced,
                run.failed,
                run.batch_size,
                run.dedup_policy,
                run.request_id,
                run.tool_caller_version,
            ),
        )
        self._conn.commit()

    def get_pipeline_run(self, id: str) -> Optional[PipelineRun]:
        row = self._conn.execute(
            "SELECT * FROM pipeline_runs WHERE id = ?", (id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_pipeline_run(row)

    def list_pipeline_runs(self, kb_id: str) -> list[PipelineRun]:
        rows = self._conn.execute(
            "SELECT * FROM pipeline_runs WHERE kb_id = ? ORDER BY started_at", (kb_id,)
        ).fetchall()
        return [self._row_to_pipeline_run(r) for r in rows]

    def update_pipeline_run(self, run: PipelineRun) -> None:
        """Update an existing pipeline run record."""
        self._conn.execute(
            "UPDATE pipeline_runs SET phase=?, status=?, started_at=?, completed_at=?, "
            "error=?, metadata=?, ended_at=?, ingested=?, skipped=?, replaced=?, "
            "failed=?, batch_size=?, dedup_policy=?, request_id=?, tool_caller_version=? "
            "WHERE id=?",
            (
                run.phase.value,
                run.status.value,
                _iso(run.started_at),
                _iso(run.completed_at),
                run.error,
                _json_dumps(run.metadata),
                _iso(run.ended_at),
                run.ingested,
                run.skipped,
                run.replaced,
                run.failed,
                run.batch_size,
                run.dedup_policy,
                run.request_id,
                run.tool_caller_version,
                run.id,
            ),
        )
        self._conn.commit()

    def delete_pipeline_run(self, id: str) -> None:
        self._conn.execute("DELETE FROM pipeline_runs WHERE id = ?", (id,))
        self._conn.commit()

    @staticmethod
    def _row_to_pipeline_run(row: sqlite3.Row) -> PipelineRun:
        # The new v0.6.0 columns may not be present on rows inserted before
        # the migration ran (sqlite3.Row access raises IndexError for absent
        # keys). Probe the row's keys so this helper keeps working on legacy
        # databases that haven't been re-opened through `_init_schema` yet.
        try:
            keys = set(row.keys())
        except AttributeError:  # pragma: no cover — defensive for non-Row rows
            keys = set()
        return PipelineRun(
            id=row["id"],
            kb_id=row["kb_id"],
            source_id=row["source_id"],
            phase=row["phase"],
            status=row["status"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            error=row["error"],
            metadata=_json_loads(row["metadata"]) or {},
            ended_at=row["ended_at"] if "ended_at" in keys else None,
            ingested=row["ingested"] if "ingested" in keys else None,
            skipped=row["skipped"] if "skipped" in keys else None,
            replaced=row["replaced"] if "replaced" in keys else None,
            failed=row["failed"] if "failed" in keys else None,
            batch_size=row["batch_size"] if "batch_size" in keys else None,
            dedup_policy=row["dedup_policy"] if "dedup_policy" in keys else None,
            request_id=row["request_id"] if "request_id" in keys else None,
            tool_caller_version=(
                row["tool_caller_version"] if "tool_caller_version" in keys else None
            ),
        )

    # ------------------------------------------------------------------
    # Full-text search
    # ------------------------------------------------------------------

    def search_wiki_fts(self, query: str, kb_id: str) -> list[WikiPage]:
        """Search wiki pages using FTS5.

        Parameters
        ----------
        query:
            FTS5 match expression.
        kb_id:
            Restrict results to this knowledgebase.
        """
        rows = self._conn.execute(
            "SELECT wp.* FROM wiki_pages wp "
            "JOIN wiki_pages_fts fts ON wp.rowid = fts.rowid "
            "WHERE wiki_pages_fts MATCH ? AND wp.kb_id = ? "
            "ORDER BY rank",
            (query, kb_id),
        ).fetchall()
        return [self._row_to_wiki_page(r) for r in rows]

    # ------------------------------------------------------------------
    # Bulk operations (used by KnowledgebaseService)
    # ------------------------------------------------------------------

    def update_source(self, source: Source) -> None:
        """Update an existing source record."""
        self._conn.execute(
            "UPDATE sources SET kb_id=?, source_type=?, uri=?, metadata=?, "
            "ingested_at=?, chunk_count=?, status=?, dedup_key=? WHERE id=?",
            (
                source.kb_id,
                source.source_type.value,
                source.uri,
                _json_dumps(source.metadata),
                _iso(source.ingested_at),
                source.chunk_count,
                source.status.value,
                source.dedup_key,
                source.id,
            ),
        )
        self._conn.commit()

    def delete_chunks_by_source(self, source_id: str) -> None:
        """Delete all chunks belonging to a source."""
        self._conn.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))
        self._conn.commit()

    def delete_page_sources_by_source(self, source_id: str) -> None:
        """Delete all page-source associations for a given source."""
        self._conn.execute("DELETE FROM page_sources WHERE source_id = ?", (source_id,))
        self._conn.commit()

    def delete_sources_by_kb(self, kb_id: str) -> None:
        """Delete all sources belonging to a knowledgebase.

        Also removes page-source associations referencing those sources.
        """
        source_ids = [
            r["id"]
            for r in self._conn.execute(
                "SELECT id FROM sources WHERE kb_id = ?", (kb_id,)
            ).fetchall()
        ]
        for sid in source_ids:
            self._conn.execute(
                "DELETE FROM page_sources WHERE source_id = ?", (sid,)
            )
        self._conn.execute("DELETE FROM sources WHERE kb_id = ?", (kb_id,))
        self._conn.commit()

    def delete_wiki_pages_by_kb(self, kb_id: str) -> None:
        """Delete all wiki pages belonging to a knowledgebase.

        Also cleans up FTS entries, links, and page-source associations.
        """
        page_ids = [
            r["id"]
            for r in self._conn.execute(
                "SELECT id FROM wiki_pages WHERE kb_id = ?", (kb_id,)
            ).fetchall()
        ]
        for page_id in page_ids:
            self.delete_wiki_page(page_id)

    def delete_chunks_by_kb(self, kb_id: str) -> None:
        """Delete all chunks belonging to a knowledgebase."""
        self._conn.execute("DELETE FROM chunks WHERE kb_id = ?", (kb_id,))
        self._conn.commit()

    def delete_pipeline_runs_by_kb(self, kb_id: str) -> None:
        """Delete all pipeline runs belonging to a knowledgebase."""
        self._conn.execute("DELETE FROM pipeline_runs WHERE kb_id = ?", (kb_id,))
        self._conn.commit()

    def delete_pipeline_runs_by_source(self, source_id: str) -> None:
        """Delete all pipeline runs associated with a source."""
        self._conn.execute(
            "DELETE FROM pipeline_runs WHERE source_id = ?", (source_id,)
        )
        self._conn.commit()

    def count_sources(self, kb_id: str) -> int:
        """Count the number of sources in a knowledgebase."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM sources WHERE kb_id = ?", (kb_id,)
        ).fetchone()
        return row["cnt"]

    def count_wiki_pages(self, kb_id: str) -> int:
        """Count the number of wiki pages in a knowledgebase."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM wiki_pages WHERE kb_id = ?", (kb_id,)
        ).fetchone()
        return row["cnt"]

    def count_chunks_by_embedding_model(self, kb_id: str) -> dict[str, int]:
        """Return a mapping of embedding model name -> chunk count for *kb_id*.

        Reads only the ``metadata`` JSON column from the chunks table and
        tallies the ``"embedding_model"`` key.  Chunks whose metadata lacks
        the key (e.g. ingested before this feature was added) are skipped.
        """
        rows = self._conn.execute(
            "SELECT metadata FROM chunks WHERE kb_id = ?", (kb_id,)
        ).fetchall()
        counts: dict[str, int] = {}
        for row in rows:
            meta = _json_loads(row["metadata"])
            if not isinstance(meta, dict):
                continue
            model = meta.get("embedding_model")
            if model is None:
                continue
            counts[model] = counts.get(model, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Phase 4 — per-page (per-chunk) embedding-provider snapshot
    # ------------------------------------------------------------------

    def get_embedding_snapshot(
        self, kb_id: str
    ) -> tuple[Optional[str], Optional[str], Optional[str]] | None:
        """Return the *(model, provider, base_url)* snapshot for *kb_id*.

        Reads the per-chunk snapshot stamped at ingest time (see Phase 4
        spec task: "Stamp ``(embedding_provider, embed_base_url)``
        alongside ``dominant_embedding_model`` on every page").

        Resolution order, picking the dominant tuple by chunk count:

        1. Native ``chunks.embedding_provider`` / ``chunks.embed_base_url``
           columns added by the Phase 4 ALTER TABLE migration.
        2. Fallback to ``chunks.metadata["embedding_provider"]`` /
           ``chunks.metadata["embed_base_url"]`` so KBs ingested via the
           metadata-bag path are still recognized.
        3. ``chunks.metadata["embedding_model"]`` for the model name (the
           v0.6.0 stamping path that pre-dates Phase 4).

        Returns ``None`` when no chunks exist for *kb_id*. Otherwise
        returns a 3-tuple ``(model, provider, base_url)`` where
        ``provider`` and ``base_url`` may individually be ``None`` for
        legacy chunks that were not backfilled.

        Performance (I-01): aggregation runs SQLite-side via
        ``json_extract`` + ``GROUP BY`` so a 100k-chunk KB pays a
        single index pass and zero Python-side ``json.loads`` calls,
        rather than the previous full table scan + per-row Python
        decoding. ``COALESCE`` collapses the native-column / metadata
        fallback into the same group key in one pass.
        """
        # Single-pass aggregation: group by the COALESCE-resolved
        # (model, provider, base_url) tuple and count. ORDER BY count
        # DESC + LIMIT 1 picks the dominant tuple without pulling all
        # groups into Python.
        row = self._conn.execute(
            "SELECT "
            "  json_extract(metadata, '$.embedding_model') AS model, "
            "  COALESCE("
            "    embedding_provider, "
            "    json_extract(metadata, '$.embedding_provider')"
            "  ) AS provider, "
            "  COALESCE("
            "    embed_base_url, "
            "    json_extract(metadata, '$.embed_base_url')"
            "  ) AS base_url, "
            "  COUNT(*) AS cnt "
            "FROM chunks WHERE kb_id = ? "
            "GROUP BY model, provider, base_url "
            "ORDER BY cnt DESC LIMIT 1",
            (kb_id,),
        ).fetchone()
        if row is None:
            return None
        return (row["model"], row["provider"], row["base_url"])

    def backfill_embedding_snapshot(
        self,
        *,
        kb_id: str,
        provider: Optional[str],
        base_url: Optional[str],
        embedding_model: Optional[str] = None,
    ) -> int:
        """Backfill ``embedding_provider`` / ``embed_base_url`` columns
        for chunks in *kb_id* that lack them.

        Used by the Phase 4 backfill script (``services/migration.py``
        ``backfill_provider_snapshot``) to populate legacy v0.6.0 KBs
        from each KB's ``config.json`` so the snapshot read path
        (:meth:`get_embedding_snapshot`) returns non-None values for
        existing rows. Strictly UPDATE — non-destructive (only sets
        columns that are currently NULL; never overwrites a stamped
        value).

        ``embedding_model`` (I-02): when supplied, the UPDATE is
        restricted to rows whose ``metadata.embedding_model`` matches.
        This avoids misstamping a heterogeneous KB whose chunks were
        ingested under MULTIPLE different embedders — without the
        filter, the current process-wide ``Settings.embedding_provider``
        would be stamped onto chunks that came from a different
        embedder (e.g. stamping ``provider='remote'`` onto chunks
        actually produced by Ollama). Callers should pass the model
        whose ``(provider, base_url)`` they're stamping.

        Returns the number of rows updated.
        """
        if provider is None and base_url is None:
            return 0
        # Only touch rows missing BOTH columns so a partially-backfilled
        # KB stays consistent across reruns. When embedding_model is
        # supplied, additionally restrict to rows whose stamped
        # ``metadata.embedding_model`` matches — sqlite's json_extract
        # handles the metadata blob server-side so we don't pay the
        # Python-side json.loads cost per row.
        if embedding_model is None:
            cur = self._conn.execute(
                "UPDATE chunks SET embedding_provider = COALESCE(embedding_provider, ?), "
                "embed_base_url = COALESCE(embed_base_url, ?) "
                "WHERE kb_id = ? AND (embedding_provider IS NULL OR embed_base_url IS NULL)",
                (provider, base_url, kb_id),
            )
        else:
            cur = self._conn.execute(
                "UPDATE chunks SET embedding_provider = COALESCE(embedding_provider, ?), "
                "embed_base_url = COALESCE(embed_base_url, ?) "
                "WHERE kb_id = ? "
                "AND (embedding_provider IS NULL OR embed_base_url IS NULL) "
                "AND json_extract(metadata, '$.embedding_model') = ?",
                (provider, base_url, kb_id, embedding_model),
            )
        self._conn.commit()
        return cur.rowcount or 0

    # ------------------------------------------------------------------
    # Phase 5 — embedder_version stamping
    # ------------------------------------------------------------------

    def get_embedder_versions(self, kb_id: str) -> set[str]:
        """Return the set of distinct ``embedder_version`` values for *kb_id*.

        Used by ``KnowledgebaseService._ingest_source_locked`` to detect
        mixed-version ingests: when a KB already carries chunks under
        embedder version ``A`` and a new ingest would write chunks under
        embedder version ``B`` (``A != B``), the ingest is rejected with
        ``EMBEDDER_VERSION_MISMATCH`` unless ``AGENT_KB_AUTO_REEMBED=1``
        is set.

        The version string is opaque to this layer — the embedder
        provides it via :attr:`Embedder.embedder_version` (Phase 5)
        with a format like ``"fastembed/MiniLM-L6-v2-int8"`` or
        ``"sentence-transformers/all-MiniLM-L6-v2@hf-fp32"``.

        NULL values (legacy v0.6.0/v0.10.x chunks ingested before the
        Phase 5 column existed) are silently ignored so a legacy KB
        can still receive new ingests under whatever embedder is
        configured today.

        Resolution order, mirroring :meth:`get_embedding_snapshot`:

        1. Native ``chunks.embedder_version`` column.
        2. Fallback to ``json_extract(metadata, '$.embedder_version')``
           so chunks stamped via the metadata-only path are still
           recognized.

        spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b
        """
        rows = self._conn.execute(
            "SELECT DISTINCT COALESCE("
            "  embedder_version, "
            "  json_extract(metadata, '$.embedder_version')"
            ") AS version "
            "FROM chunks WHERE kb_id = ?",
            (kb_id,),
        ).fetchall()
        return {row["version"] for row in rows if row["version"] is not None}
