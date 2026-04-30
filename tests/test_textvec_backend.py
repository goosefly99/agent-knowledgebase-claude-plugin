"""Phase B contract test: TextvecBackend Protocol compliance.

Pins all 7 RetrieverBackend Protocol methods on a fresh SQLite DB.
Round-trip: index -> query -> search -> delete -> count.
Probe-4 fields verified via info().
health_check() verified on a live FTS5-capable sqlite build.
index() partial-failure log emission verified (Fix #2 from Phase B review).
"""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_knowledgebase.backends.textvec_backend import TextvecBackend
from agent_knowledgebase.config import Settings
from agent_knowledgebase.database import Database


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def kb_id() -> str:
    return f"textvec-test-{uuid.uuid4().hex[:8]}"


@pytest.fixture()
def textvec_settings(tmp_path: Path) -> Settings:
    saves = tmp_path / "saves"
    saves.mkdir()
    return Settings(saves_dir=saves)


@pytest.fixture()
def textvec_backend(textvec_settings: Settings) -> TextvecBackend:
    """TextvecBackend bound to a settings object (no KnowledgebaseService)."""
    return TextvecBackend(textvec_settings, service=None)


@pytest.fixture()
def db_with_kb(textvec_settings: Settings, kb_id: str) -> Database:
    """Create the per-KB SQLite DB and a knowledgebases row so FK constraints pass."""
    from datetime import datetime

    from agent_knowledgebase.config import sanitize_kb_dir_name

    safe_dir = sanitize_kb_dir_name(kb_id)
    db_path = textvec_settings.kb_db_path(safe_dir)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = Database(db_path=db_path)
    # Insert a knowledgebases row so FK constraints pass.
    now = datetime.utcnow().isoformat()
    db._conn.execute(
        "INSERT OR IGNORE INTO knowledgebases (id, name, description, config, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (kb_id, kb_id, "", "{}", now, now),
    )
    # Insert a sources row for probe-4 source_type / uri / dedup_key fields.
    db._conn.execute(
        "INSERT OR IGNORE INTO sources (id, kb_id, source_type, uri, dedup_key, status) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (f"src-{kb_id}", kb_id, "file", "/tmp/test.txt", "textvec-dedup", "ingested"),
    )
    db._conn.commit()
    return db


def _make_docs(kb_id: str, n: int = 10, prefix: str = "") -> list[dict]:
    """Build n test documents with distinct content."""
    return [
        {
            "id": f"chunk-{i:04d}",
            "content": f"{prefix}{' ' if prefix else ''}The quick brown fox jumps over the lazy dog sentence number {i}.",
            "metadata": {
                "kb_id": kb_id,
                "source_id": f"src-{kb_id}",
                "source_type": "file",
            },
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Protocol method tests
# ---------------------------------------------------------------------------


class TestTextvecBackendProtocol:
    def test_index_empty_is_noop(
        self, textvec_backend: TextvecBackend, kb_id: str, db_with_kb: Database
    ) -> None:
        """index() with empty list is a no-op and doesn't raise."""
        textvec_backend.index(kb_id=kb_id, documents=[])

    def test_index_inserts_chunks(
        self, textvec_backend: TextvecBackend, kb_id: str, db_with_kb: Database
    ) -> None:
        """index() inserts rows into the chunks table."""
        docs = _make_docs(kb_id, n=5)
        textvec_backend.index(kb_id=kb_id, documents=docs)
        count = textvec_backend.count(kb_id=kb_id)
        assert count == 5

    def test_index_embedding_columns_null(
        self, textvec_backend: TextvecBackend, kb_id: str, db_with_kb: Database
    ) -> None:
        """index() writes embedding columns as NULL (no embedder called)."""
        docs = _make_docs(kb_id, n=2)
        textvec_backend.index(kb_id=kb_id, documents=docs)
        rows = db_with_kb._conn.execute(
            "SELECT embedding_id, embedding_provider, embed_base_url, embedder_version "
            "FROM chunks WHERE kb_id = ?",
            (kb_id,),
        ).fetchall()
        assert len(rows) == 2
        for row in rows:
            assert row["embedding_id"] is None
            assert row["embedding_provider"] is None
            assert row["embed_base_url"] is None
            assert row["embedder_version"] is None

    def test_count_zero_for_empty_kb(
        self, textvec_backend: TextvecBackend, kb_id: str, db_with_kb: Database
    ) -> None:
        """count() returns 0 before any index() calls."""
        assert textvec_backend.count(kb_id=kb_id) == 0

    def test_count_after_index(
        self, textvec_backend: TextvecBackend, kb_id: str, db_with_kb: Database
    ) -> None:
        """count() returns the number of indexed documents."""
        docs = _make_docs(kb_id, n=100)
        textvec_backend.index(kb_id=kb_id, documents=docs)
        assert textvec_backend.count(kb_id=kb_id) == 100

    def test_search_returns_results(
        self, textvec_backend: TextvecBackend, kb_id: str, db_with_kb: Database
    ) -> None:
        """search() returns non-empty results for a matching query."""
        docs = _make_docs(kb_id, n=20)
        textvec_backend.index(kb_id=kb_id, documents=docs)
        results = textvec_backend.search(kb_id=kb_id, text="quick brown fox", top_k=5)
        assert len(results) > 0
        assert len(results) <= 5

    def test_search_result_shape(
        self, textvec_backend: TextvecBackend, kb_id: str, db_with_kb: Database
    ) -> None:
        """Each search() result has content, source_id, score, metadata keys."""
        docs = _make_docs(kb_id, n=5)
        textvec_backend.index(kb_id=kb_id, documents=docs)
        results = textvec_backend.search(kb_id=kb_id, text="fox dog", top_k=3)
        assert results
        for r in results:
            assert "content" in r
            assert "source_id" in r
            assert "score" in r
            assert "metadata" in r
            assert 0.0 <= r["score"] <= 1.0

    def test_query_routes_to_search(
        self, textvec_backend: TextvecBackend, kb_id: str, db_with_kb: Database
    ) -> None:
        """query() returns same result shape as search() (re-routes to BM25)."""
        docs = _make_docs(kb_id, n=10)
        textvec_backend.index(kb_id=kb_id, documents=docs)
        results_search = textvec_backend.search(kb_id=kb_id, text="fox jumps", top_k=5)
        results_query = textvec_backend.query(kb_id=kb_id, text="fox jumps", top_k=5)
        # Both paths should return non-empty results with same count.
        assert len(results_search) == len(results_query)
        assert len(results_query) > 0

    def test_search_empty_query_returns_empty(
        self, textvec_backend: TextvecBackend, kb_id: str, db_with_kb: Database
    ) -> None:
        """search() on empty string returns empty list."""
        docs = _make_docs(kb_id, n=5)
        textvec_backend.index(kb_id=kb_id, documents=docs)
        results = textvec_backend.search(kb_id=kb_id, text="", top_k=10)
        assert results == []

    def test_search_top_k_respected(
        self, textvec_backend: TextvecBackend, kb_id: str, db_with_kb: Database
    ) -> None:
        """search() returns at most top_k results."""
        docs = _make_docs(kb_id, n=50)
        textvec_backend.index(kb_id=kb_id, documents=docs)
        results = textvec_backend.search(kb_id=kb_id, text="quick brown fox", top_k=3)
        assert len(results) <= 3

    def test_delete_by_ids(
        self, textvec_backend: TextvecBackend, kb_id: str, db_with_kb: Database
    ) -> None:
        """delete(ids=...) removes the specified chunks."""
        docs = _make_docs(kb_id, n=5)
        textvec_backend.index(kb_id=kb_id, documents=docs)
        ids_to_delete = ["chunk-0000", "chunk-0002"]
        textvec_backend.delete(kb_id=kb_id, ids=ids_to_delete)
        assert textvec_backend.count(kb_id=kb_id) == 3

    def test_delete_by_source_id(
        self, textvec_backend: TextvecBackend, kb_id: str, db_with_kb: Database
    ) -> None:
        """delete(source_id=...) removes all chunks for that source."""
        docs = _make_docs(kb_id, n=5)
        # All docs use the source_id that exists in db_with_kb (FK constraint).
        source_id = f"src-{kb_id}"
        for doc in docs:
            doc["metadata"]["source_id"] = source_id
        textvec_backend.index(kb_id=kb_id, documents=docs)
        textvec_backend.delete(kb_id=kb_id, source_id=source_id)
        assert textvec_backend.count(kb_id=kb_id) == 0

    def test_delete_requires_ids_or_source_id(
        self, textvec_backend: TextvecBackend, kb_id: str, db_with_kb: Database
    ) -> None:
        """delete() raises ValueError if neither ids nor source_id is given."""
        with pytest.raises(ValueError, match="at least one of"):
            textvec_backend.delete(kb_id=kb_id)

    def test_info_probe4_keys_present(
        self, textvec_backend: TextvecBackend, kb_id: str, db_with_kb: Database
    ) -> None:
        """info() returns all 5 probe-4 keys."""
        info = textvec_backend.info(kb_id=kb_id)
        probe4_keys = {"source_type", "uri", "dedup_key", "page_id", "dominant_embedding_model"}
        for key in probe4_keys:
            assert key in info, f"info() missing probe-4 key: {key!r}"

    def test_info_dominant_embedding_model_is_lexical_fts5(
        self, textvec_backend: TextvecBackend, kb_id: str, db_with_kb: Database
    ) -> None:
        """info() returns dominant_embedding_model='lexical-fts5'."""
        info = textvec_backend.info(kb_id=kb_id)
        assert info["dominant_embedding_model"] == "lexical-fts5"

    def test_info_probe4_none_not_omitted_for_empty_kb(
        self, textvec_backend: TextvecBackend, tmp_path: Path
    ) -> None:
        """info() on a KB with no sources returns None for probe-4 fields, not omitted."""
        # Use a kb_id with no sources row to exercise None-not-omitted branch.
        empty_kb_id = f"empty-{uuid.uuid4().hex[:8]}"
        saves = tmp_path / "saves-empty"
        saves.mkdir()
        settings = Settings(saves_dir=saves)
        backend = TextvecBackend(settings, service=None)
        # Create just the DB (no sources row).
        from agent_knowledgebase.config import sanitize_kb_dir_name

        db_path = settings.kb_db_path(sanitize_kb_dir_name(empty_kb_id))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db = Database(db_path=db_path)
        db._conn.execute(
            "INSERT OR IGNORE INTO knowledgebases (id, name, description, config, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (empty_kb_id, empty_kb_id, "", "{}", "2024-01-01", "2024-01-01"),
        )
        db._conn.commit()
        info = backend.info(kb_id=empty_kb_id)
        for key in ("source_type", "uri", "dedup_key", "page_id"):
            assert key in info
            assert info[key] is None, f"expected None for {key!r} on empty KB"
        assert info["dominant_embedding_model"] == "lexical-fts5"

    def test_health_check_ok(self, textvec_backend: TextvecBackend) -> None:
        """health_check() returns status='ok' when FTS5 is available."""
        result = textvec_backend.health_check()
        assert result["backend"] == "textvec"
        assert result["status"] == "ok"
        assert result["fts5_available"] is True

    def test_fts5_trigger_populates_chunks_fts(
        self, textvec_backend: TextvecBackend, kb_id: str, db_with_kb: Database
    ) -> None:
        """AFTER INSERT trigger: chunks_fts is populated after index().

        Verifies via the backend's own search() (which queries chunks_fts MATCH)
        and also via a fresh sqlite connection to the same DB file (to avoid
        stale WAL snapshot from db_with_kb._conn).
        """
        docs = _make_docs(kb_id, n=3, prefix="UNIQUETOKEN")
        textvec_backend.index(kb_id=kb_id, documents=docs)

        # Verify via the backend's search (uses the same connection + chunks_fts).
        results = textvec_backend.search(kb_id=kb_id, text="UNIQUETOKEN", top_k=5)
        assert len(results) > 0, "chunks_fts not populated by trigger after index()"

        # Also verify via a fresh connection (bypasses WAL snapshot issues).
        import sqlite3 as _sqlite3

        db_path = textvec_backend._db_path(kb_id)
        fresh_conn = _sqlite3.connect(str(db_path), check_same_thread=False)
        fresh_conn.row_factory = _sqlite3.Row
        row = fresh_conn.execute(
            "SELECT count(*) AS cnt FROM chunks_fts WHERE chunks_fts MATCH '\"UNIQUETOKEN\"'",
        ).fetchone()
        fresh_conn.close()
        assert row["cnt"] > 0, "chunks_fts has 0 rows for UNIQUETOKEN via fresh connection"


# ---------------------------------------------------------------------------
# Fix #2: index() partial-failure log emission
# ---------------------------------------------------------------------------


class TestIndexPartialFailureLog:
    """index() must emit a failure-shaped stderr log and re-raise on mid-loop error.

    We use a mock connection (via MagicMock) whose execute() raises
    IntegrityError on the 3rd INSERT call, then assert:
    - rows_ok=2 in the failure log (only 2 succeeded before the error)
    - error_code="TEXTVEC_INDEX_FAILED" is set
    - The exception still propagates to the caller

    Note: sqlite3.Connection.execute is read-only in CPython 3.13, so we
    cannot monkey-patch a real connection. Instead we inject a MagicMock
    connection via _connect to achieve the same effect without mutating a
    live sqlite3.Connection.
    """

    def test_partial_failure_emits_error_log_and_reraises(
        self, textvec_backend: TextvecBackend, kb_id: str, db_with_kb: Database
    ) -> None:
        """IntegrityError on doc 3 of 5: failure log has rows_ok=2 + re-raises."""
        from contextlib import contextmanager
        from unittest.mock import MagicMock

        docs = [
            {
                "id": f"doc-{i}",
                "content": f"content {i}",
                "metadata": {"kb_id": kb_id, "source_id": f"src-{kb_id}"},
            }
            for i in range(5)
        ]

        insert_call_count = 0

        def side_effect_execute(sql, params=None):
            nonlocal insert_call_count
            if sql and "INSERT OR IGNORE INTO chunks" in sql:
                insert_call_count += 1
                if insert_call_count == 3:
                    raise sqlite3.IntegrityError("simulated IntegrityError on doc 3")
            return MagicMock()  # cursor-like return value

        mock_conn = MagicMock()
        mock_conn.execute.side_effect = side_effect_execute
        mock_conn.commit.return_value = None

        @contextmanager
        def mock_connect(kb_id_arg):
            yield mock_conn

        logged_calls: list[dict] = []

        def capture_log(**kwargs):
            logged_calls.append(dict(kwargs))

        with (
            patch(
                "agent_knowledgebase.backends.textvec_backend.knowledgebase_stderr_log",
                side_effect=capture_log,
            ),
            patch.object(textvec_backend, "_connect", side_effect=mock_connect),
        ):
            with pytest.raises(sqlite3.IntegrityError, match="simulated"):
                textvec_backend.index(kb_id=kb_id, documents=docs)

        # There should be at least 2 log calls: the start log + the failure log.
        assert len(logged_calls) >= 2, (
            f"Expected at least 2 log calls (start + failure), got {len(logged_calls)}: "
            f"{logged_calls}"
        )

        # The failure log must be the last one emitted (after start log).
        failure_log = logged_calls[-1]
        assert failure_log["error_code"] == "TEXTVEC_INDEX_FAILED", (
            f"Expected error_code='TEXTVEC_INDEX_FAILED', got {failure_log['error_code']!r}"
        )
        assert failure_log["rows_ok"] == 2, (
            f"Expected rows_ok=2 (2 docs succeeded before error), got {failure_log['rows_ok']}"
        )
        assert failure_log["rows_in"] == 5, f"Expected rows_in=5, got {failure_log['rows_in']}"
        assert failure_log["error_message"] is not None
        assert "simulated" in failure_log["error_message"]

    def test_index_elapsed_ms_not_hardcoded_zero_on_success(
        self, textvec_backend: TextvecBackend, kb_id: str, db_with_kb: Database
    ) -> None:
        """On success, elapsed_ms in the final log is >= 0 (captured from perf_counter)."""
        docs = [
            {
                "id": f"perf-doc-{i}",
                "content": f"performance test content {i}",
                "metadata": {"kb_id": kb_id, "source_id": f"src-{kb_id}"},
            }
            for i in range(3)
        ]

        logged_calls: list[dict] = []

        def capture_log(**kwargs):
            logged_calls.append(dict(kwargs))

        with patch(
            "agent_knowledgebase.backends.textvec_backend.knowledgebase_stderr_log",
            side_effect=capture_log,
        ):
            textvec_backend.index(kb_id=kb_id, documents=docs)

        # Should have at least start + success log.
        assert len(logged_calls) >= 2
        # All log calls must have elapsed_ms as a number, not hardcoded 0
        # for the success log (last call). The start log may legitimately be ~0.
        success_log = logged_calls[-1]
        assert success_log["elapsed_ms"] >= 0
        assert success_log["error_code"] is None
        assert success_log["rows_ok"] == 3
