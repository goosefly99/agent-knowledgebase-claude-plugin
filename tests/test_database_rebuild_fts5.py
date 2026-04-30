"""Regression tests for Database.rebuild_fts5().

Pins the Phase B bug fix: the original SQL queried `chunks_fts` directly in the
NOT EXISTS subquery, which raises OperationalError("no such column: T.text") on
contentless FTS5 tables. The fix queries `chunks_fts_docsize` instead (the FTS5
shadow table that tracks indexed doc-ids without reading content).

Test strategy: insert chunks normally (trigger fires, FTS5 is populated), then
clear chunks_fts via the FTS5 'delete-all' command to simulate desync (e.g. a
legacy pre-trigger KB), then call rebuild_fts5() and assert it backfills the
correct number of rows, and that a second call returns 0 (idempotency).

Note: contentless FTS5 tables do not support plain ``DELETE FROM chunks_fts``
— that raises the same OperationalError("no such column: T.text") as the
original bug. The correct way to clear a contentless FTS5 index is via the
FTS5 special command: ``INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all')``.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

import pytest

from agent_knowledgebase.config import Settings, sanitize_kb_dir_name
from agent_knowledgebase.database import Database


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def kb_id() -> str:
    return f"rebuild-fts5-test-{uuid.uuid4().hex[:8]}"


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    saves = tmp_path / "saves"
    saves.mkdir()
    return Settings(saves_dir=saves)


@pytest.fixture()
def db(settings: Settings, kb_id: str) -> Database:
    """Create a per-KB SQLite DB with knowledgebases and sources rows seeded."""
    safe_dir = sanitize_kb_dir_name(kb_id)
    db_path = settings.kb_db_path(safe_dir)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    database = Database(db_path=db_path)

    now = datetime.utcnow().isoformat()
    database._conn.execute(
        "INSERT OR IGNORE INTO knowledgebases (id, name, description, config, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (kb_id, kb_id, "", "{}", now, now),
    )
    database._conn.execute(
        "INSERT OR IGNORE INTO sources (id, kb_id, source_type, uri, dedup_key, status) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (f"src-{kb_id}", kb_id, "file", "/tmp/test.txt", "rebuild-fts5-dedup", "ingested"),
    )
    database._conn.commit()
    return database


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _insert_chunks(database: Database, kb_id: str, n: int = 5) -> list[str]:
    """Insert n chunks rows via INSERT and return their ids."""
    source_id = f"src-{kb_id}"
    ids: list[str] = []
    for i in range(n):
        chunk_id = f"chunk-{i:04d}-{uuid.uuid4().hex[:6]}"
        database._conn.execute(
            "INSERT INTO chunks (id, source_id, kb_id, content, metadata) VALUES (?, ?, ?, ?, ?)",
            (
                chunk_id,
                source_id,
                kb_id,
                f"REBUILDTOKEN sentence {i} quick brown fox jumps over the lazy dog.",
                "{}",
            ),
        )
        ids.append(chunk_id)
    database._conn.commit()
    return ids


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRebuildFts5:
    def test_rebuild_fts5_backfills_desynced_index(self, db: Database, kb_id: str) -> None:
        """rebuild_fts5() inserts N rows when chunks_fts is empty (desynced state).

        This exercises the exact SQL path that was broken before the fix:
        the NOT EXISTS subquery over chunks_fts_docsize must not raise
        OperationalError("no such column: T.text").
        """
        n = 5
        _insert_chunks(db, kb_id, n=n)

        # Simulate desync: wipe chunks_fts so rebuild_fts5 has work to do.
        db._conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all')")
        db._conn.commit()

        backfilled = db.rebuild_fts5(kb_id)
        assert backfilled == n, f"expected {n} rows backfilled, got {backfilled}"

    def test_rebuild_fts5_idempotent_returns_zero(self, db: Database, kb_id: str) -> None:
        """A second call to rebuild_fts5() returns 0 — no duplicate indexing."""
        _insert_chunks(db, kb_id, n=4)

        # First pass: wipe FTS and rebuild.
        db._conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all')")
        db._conn.commit()
        first = db.rebuild_fts5(kb_id)
        assert first == 4

        # Second pass: all rowids already in chunks_fts_docsize → 0 inserts.
        second = db.rebuild_fts5(kb_id)
        assert second == 0, f"expected 0 on second call, got {second}"

    def test_rebuild_fts5_fts_match_works_after_rebuild(self, db: Database, kb_id: str) -> None:
        """After rebuild_fts5(), chunks_fts MATCH returns the expected rows."""
        _insert_chunks(db, kb_id, n=3)

        db._conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all')")
        db._conn.commit()
        db.rebuild_fts5(kb_id)

        rows = db._conn.execute(
            "SELECT count(*) AS cnt FROM chunks_fts WHERE chunks_fts MATCH 'REBUILDTOKEN'",
        ).fetchone()
        assert rows["cnt"] == 3, f"expected 3 FTS matches after rebuild, got {rows['cnt']}"

    def test_rebuild_fts5_no_cross_kb_bleed(self, db: Database, kb_id: str, settings: Settings) -> None:
        """rebuild_fts5() only backfills rows for the specified kb_id."""
        other_kb_id = f"other-{uuid.uuid4().hex[:8]}"
        now = datetime.utcnow().isoformat()

        # Register the other KB in the same DB.
        db._conn.execute(
            "INSERT OR IGNORE INTO knowledgebases (id, name, description, config, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (other_kb_id, other_kb_id, "", "{}", now, now),
        )
        db._conn.execute(
            "INSERT OR IGNORE INTO sources (id, kb_id, source_type, uri, dedup_key, status) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (f"src-{other_kb_id}", other_kb_id, "file", "/tmp/other.txt", "other-dedup", "ingested"),
        )
        db._conn.commit()

        _insert_chunks(db, kb_id, n=3)
        _insert_chunks(db, other_kb_id, n=7)

        # Wipe FTS for both KBs.
        db._conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all')")
        db._conn.commit()

        # Only rebuild for kb_id.
        backfilled = db.rebuild_fts5(kb_id)
        assert backfilled == 3, f"expected 3 rows for kb_id, got {backfilled}"

        # other_kb_id rows must NOT appear in chunks_fts_docsize yet.
        other_count = db._conn.execute(
            "SELECT count(*) AS cnt FROM chunks WHERE kb_id = ? "
            "AND EXISTS (SELECT 1 FROM chunks_fts_docsize WHERE id = chunks.rowid)",
            (other_kb_id,),
        ).fetchone()["cnt"]
        assert other_count == 0, f"expected 0 other-kb rows indexed, got {other_count}"

    def test_rebuild_fts5_empty_kb_returns_zero(self, db: Database, kb_id: str) -> None:
        """rebuild_fts5() on a KB with no chunks returns 0."""
        result = db.rebuild_fts5(kb_id)
        assert result == 0
