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
        self._conn = sqlite3.connect(str(db_path))
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
        self._conn.execute(
            "INSERT INTO chunks (id, source_id, kb_id, content, metadata, embedding_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                chunk.id,
                chunk.source_id,
                chunk.kb_id,
                chunk.content,
                _json_dumps(chunk.metadata),
                chunk.embedding_id,
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
            "(id, kb_id, source_id, phase, status, started_at, completed_at, error, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
            "UPDATE pipeline_runs SET phase=?, status=?, started_at=?, completed_at=?, error=?, metadata=? WHERE id=?",
            (run.phase.value, run.status.value, _iso(run.started_at), _iso(run.completed_at), run.error, _json_dumps(run.metadata), run.id),
        )
        self._conn.commit()

    def delete_pipeline_run(self, id: str) -> None:
        self._conn.execute("DELETE FROM pipeline_runs WHERE id = ?", (id,))
        self._conn.commit()

    @staticmethod
    def _row_to_pipeline_run(row: sqlite3.Row) -> PipelineRun:
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
