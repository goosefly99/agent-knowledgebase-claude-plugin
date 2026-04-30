"""Phase C contract test: migrate_to_textvec() end-to-end + filelock-deferred path.

Pins:
- End-to-end: ingest a fresh chromadb-stamped KB (small synthetic corpus),
  run migrate_to_textvec, assert sentinel exists, chunks_fts populated,
  return shape {"status": "migrated", "rows": N, "elapsed_ms": M}.
- Filelock-deferred path: simulate filelock contention (mock FileLock.acquire
  to raise Timeout), assert return is {"status": "deferred", "reason": ...}.
- Stderr-log emission: assert log line emitted with correct phase / error_code.

spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b (Phase C)
"""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_knowledgebase.backends import MIGRATED_SENTINEL_FILENAME
from agent_knowledgebase.config import Settings, sanitize_kb_dir_name
from agent_knowledgebase.services.knowledgebase import KnowledgebaseService
from agent_knowledgebase.services.migration import migrate_to_textvec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def saves_dir(tmp_path: Path) -> Path:
    d = tmp_path / "saves"
    d.mkdir()
    return d


@pytest.fixture()
def settings(saves_dir: Path) -> Settings:
    return Settings(saves_dir=saves_dir)


@pytest.fixture()
def service(settings: Settings) -> KnowledgebaseService:
    """Build a KnowledgebaseService with mocked embedder/vectorstore."""
    svc = KnowledgebaseService(settings)
    svc._embedder_instance = MagicMock()
    svc._embedder_instance.embed.side_effect = lambda texts: [[0.1, 0.2, 0.3]] * len(texts)
    svc._embedder_instance.model_name = "test-embedder"
    svc._get_vectorstore = lambda kb_id: MagicMock(  # type: ignore[assignment]
        add=MagicMock(return_value=None),
        delete=MagicMock(return_value=None),
        count=MagicMock(return_value=0),
    )
    mock_ingestion = MagicMock()
    mock_ingestion.ingest.side_effect = lambda *_, **__: []
    svc._ingestion = mock_ingestion
    return svc


def _plant_chunks(
    svc: KnowledgebaseService,
    kb_id: str,
    n: int = 5,
    *,
    stamp_embedding_provider: bool = True,
) -> list[str]:
    """Insert n chunk rows directly into the KB SQLite, bypassing ingest pipeline.

    When stamp_embedding_provider=True, sets embedding_provider='qwen3-embedding'
    to simulate legacy chromadb-stamped KB. The chunks_fts trigger fires on INSERT
    into chunks; we then wipe chunks_fts to simulate the pre-textvec state.
    """
    ctx = svc._ctx(kb_id)  # noqa: SLF001
    db = ctx.db
    source_id = f"src-{kb_id}"

    # Ensure source row exists.
    db._conn.execute(
        "INSERT OR IGNORE INTO sources (id, kb_id, source_type, uri, dedup_key, status) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (source_id, kb_id, "file", "/tmp/test.txt", "migrate-dedup", "ingested"),
    )
    db._conn.commit()

    chunk_ids = []
    for i in range(n):
        chunk_id = f"chunk-{i:04d}-{uuid.uuid4().hex[:6]}"
        provider = "qwen3-embedding" if stamp_embedding_provider else None
        db._conn.execute(
            "INSERT INTO chunks (id, source_id, kb_id, content, metadata, embedding_provider) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                chunk_id,
                source_id,
                kb_id,
                f"MIGRATETOKEN sentence {i} quick brown fox jumps over the lazy dog.",
                "{}",
                provider,
            ),
        )
        chunk_ids.append(chunk_id)
    db._conn.commit()

    # Wipe chunks_fts to simulate legacy state (pre-textvec trigger).
    db._conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all')")
    db._conn.commit()

    return chunk_ids


# ---------------------------------------------------------------------------
# End-to-end migration test
# ---------------------------------------------------------------------------


class TestMigrateToTextvecEndToEnd:
    def test_migration_returns_migrated_status(
        self,
        service: KnowledgebaseService,
    ) -> None:
        """migrate_to_textvec returns status='migrated' with expected shape."""
        kb = service.create_kb("migrate-e2e-test")
        n = 5
        _plant_chunks(service, kb.id, n=n)

        result = migrate_to_textvec(kb.id, service)

        assert result["status"] == "migrated"
        assert result["rows"] == n
        assert isinstance(result["elapsed_ms"], int)
        assert result["elapsed_ms"] >= 0

    def test_sentinel_file_written(
        self,
        service: KnowledgebaseService,
        settings: Settings,
    ) -> None:
        """Sentinel file .migrated_to is written with content 'textvec'."""
        kb = service.create_kb("migrate-sentinel-test")
        _plant_chunks(service, kb.id, n=3)

        migrate_to_textvec(kb.id, service)

        # service._index maps kb_id → dir_name (based on the KB *name*, not id).
        dir_name = service._index.get(kb.id) or sanitize_kb_dir_name(kb.id)  # noqa: SLF001
        kb_root = settings.saves_dir / dir_name
        sentinel = kb_root / MIGRATED_SENTINEL_FILENAME
        assert sentinel.is_file(), f"Sentinel not found at {sentinel}"
        assert sentinel.read_text(encoding="utf-8").strip() == "textvec"

    def test_chunks_fts_populated_after_migration(
        self,
        service: KnowledgebaseService,
    ) -> None:
        """After migrate_to_textvec, FTS5 MATCH returns rows."""
        kb = service.create_kb("migrate-fts-test")
        n = 4
        _plant_chunks(service, kb.id, n=n)

        result = migrate_to_textvec(kb.id, service)
        assert result["status"] == "migrated"

        ctx = service._ctx(kb.id)  # noqa: SLF001
        rows = ctx.db._conn.execute(  # noqa: SLF001
            "SELECT count(*) AS cnt FROM chunks_fts WHERE chunks_fts MATCH 'MIGRATETOKEN'",
        ).fetchone()
        assert rows["cnt"] == n, f"Expected {n} FTS matches, got {rows['cnt']}"

    def test_rows_count_matches_chunk_count(
        self,
        service: KnowledgebaseService,
    ) -> None:
        """rows in result == number of chunks backfilled."""
        kb = service.create_kb("migrate-rows-count-test")
        n = 7
        _plant_chunks(service, kb.id, n=n)

        result = migrate_to_textvec(kb.id, service)
        assert result["rows"] == n


# ---------------------------------------------------------------------------
# Filelock-deferred path
# ---------------------------------------------------------------------------


class TestMigrateToTextvecFilelockDeferred:
    def test_filelock_timeout_returns_deferred(
        self,
        service: KnowledgebaseService,
    ) -> None:
        """When filelock cannot be acquired, returns status='deferred' without raising."""
        kb = service.create_kb("migrate-deferred-test")
        _plant_chunks(service, kb.id, n=3)

        from filelock import Timeout as FilelockTimeout

        with patch("filelock.FileLock.acquire", side_effect=FilelockTimeout("")):
            result = migrate_to_textvec(kb.id, service)

        assert result["status"] == "deferred"
        assert result["reason"] == "ingest_in_progress"
        assert result["rows"] == 0
        assert isinstance(result["elapsed_ms"], int)

    def test_filelock_deferred_emits_stderr_log(
        self,
        service: KnowledgebaseService,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """On filelock timeout, a MIGRATION_DEFERRED stderr log line is emitted."""
        import json

        kb = service.create_kb("migrate-deferred-log-test")
        _plant_chunks(service, kb.id, n=3)

        from filelock import Timeout as FilelockTimeout

        with patch("filelock.FileLock.acquire", side_effect=FilelockTimeout("")):
            migrate_to_textvec(kb.id, service)

        captured = capsys.readouterr()
        log_lines = [json.loads(line) for line in captured.err.splitlines() if line.strip()]
        deferred_lines = [ln for ln in log_lines if ln.get("error_code") == "MIGRATION_DEFERRED"]
        assert deferred_lines, "Expected a MIGRATION_DEFERRED log line; got:\n" + "\n".join(
            repr(ln) for ln in log_lines
        )


# ---------------------------------------------------------------------------
# Stderr log emission on success
# ---------------------------------------------------------------------------


class TestMigrateToTextvecStderrLog:
    def test_success_emits_stderr_log(
        self,
        service: KnowledgebaseService,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """A successful migration emits a log line with phase='migration'."""
        import json

        kb = service.create_kb("migrate-log-test")
        _plant_chunks(service, kb.id, n=3)

        migrate_to_textvec(kb.id, service)

        captured = capsys.readouterr()
        log_lines = [json.loads(line) for line in captured.err.splitlines() if line.strip()]
        migration_lines = [ln for ln in log_lines if ln.get("phase") == "migration"]
        assert migration_lines, (
            "Expected at least one log line with phase='migration'; got:\n"
            + "\n".join(repr(ln) for ln in log_lines)
        )
        # Assert 11-field schema on migration lines.
        required = {
            "kb_id",
            "op",
            "phase",
            "elapsed_ms",
            "rows_in",
            "rows_ok",
            "rows_skipped",
            "rows_failed",
            "dedup_policy",
            "request_id",
            "tool_caller_version",
        }
        for ln in migration_lines:
            missing = required - set(ln)
            assert not missing, f"Migration log line missing fields {sorted(missing)}: {ln!r}"

    def test_success_log_contains_error_message(
        self,
        service: KnowledgebaseService,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """The success log line carries error_message='kb migrated to textvec backend'."""
        import json

        kb = service.create_kb("migrate-msg-test")
        _plant_chunks(service, kb.id, n=2)

        migrate_to_textvec(kb.id, service)

        captured = capsys.readouterr()
        log_lines = [json.loads(line) for line in captured.err.splitlines() if line.strip()]
        migration_lines = [ln for ln in log_lines if ln.get("phase") == "migration"]
        # The success line's error_message should mention the migration.
        migrated_lines = [
            ln
            for ln in migration_lines
            if "migrated" in (ln.get("error_message") or "") and ln.get("error_code") is None
        ]
        assert migrated_lines, (
            "Expected a migration success log line with error_message containing 'migrated' "
            "and error_code=None; got:\n" + "\n".join(repr(ln) for ln in migration_lines)
        )
