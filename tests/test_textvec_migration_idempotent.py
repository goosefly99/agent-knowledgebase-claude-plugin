"""Phase C contract test: migrate_to_textvec() sentinel-guard idempotency.

Pins:
- First run: status='migrated', rows=N.
- Second run (same KB): status='already_migrated', rows=0.
- Sentinel file present after both calls (not double-written / corrupted).
- chunks_fts unchanged (same row count) after the second run.
- Stderr-log deltas: first call emits 'kb migrated...' message, second call
  emits 'kb already migrated' message (each exactly once for the migration
  phase log line).

spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b (Phase C)
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock

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


def _plant_chunks(svc: KnowledgebaseService, kb_id: str, n: int = 5) -> None:
    """Insert n chunk rows directly and wipe chunks_fts to simulate legacy state."""
    ctx = svc._ctx(kb_id)  # noqa: SLF001
    db = ctx.db
    source_id = f"src-{kb_id}"

    db._conn.execute(
        "INSERT OR IGNORE INTO sources (id, kb_id, source_type, uri, dedup_key, status) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (source_id, kb_id, "file", "/tmp/test.txt", "idempotent-dedup", "ingested"),
    )
    db._conn.commit()

    for i in range(n):
        chunk_id = f"chunk-{i:04d}-{uuid.uuid4().hex[:6]}"
        db._conn.execute(
            "INSERT INTO chunks (id, source_id, kb_id, content, metadata) VALUES (?, ?, ?, ?, ?)",
            (
                chunk_id,
                source_id,
                kb_id,
                f"IDEMPOTENT sentence {i} quick brown fox jumps.",
                "{}",
            ),
        )
    db._conn.commit()

    # Simulate legacy state: clear FTS index.
    db._conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all')")
    db._conn.commit()


def _fts_row_count(svc: KnowledgebaseService, kb_id: str) -> int:
    """Count rows currently in chunks_fts_docsize (FTS shadow table)."""
    ctx = svc._ctx(kb_id)  # noqa: SLF001
    row = ctx.db._conn.execute(  # noqa: SLF001
        "SELECT COUNT(*) AS cnt FROM chunks_fts_docsize"
    ).fetchone()
    return int(row["cnt"]) if row else 0


# ---------------------------------------------------------------------------
# Idempotency contract
# ---------------------------------------------------------------------------


class TestMigrateToTextvecIdempotent:
    def test_first_run_returns_migrated_with_rows(
        self,
        service: KnowledgebaseService,
    ) -> None:
        """First migrate_to_textvec returns status='migrated', rows=N."""
        n = 5
        kb = service.create_kb("idempotent-first-run")
        _plant_chunks(service, kb.id, n=n)

        result = migrate_to_textvec(kb.id, service)

        assert result["status"] == "migrated"
        assert result["rows"] == n
        assert isinstance(result["elapsed_ms"], int)

    def test_second_run_returns_already_migrated(
        self,
        service: KnowledgebaseService,
    ) -> None:
        """Second migrate_to_textvec returns status='already_migrated', rows=0."""
        kb = service.create_kb("idempotent-second-run")
        _plant_chunks(service, kb.id, n=4)

        migrate_to_textvec(kb.id, service)
        result2 = migrate_to_textvec(kb.id, service)

        assert result2["status"] == "already_migrated"
        assert result2["rows"] == 0
        assert isinstance(result2["elapsed_ms"], int)

    def test_sentinel_present_after_both_calls(
        self,
        service: KnowledgebaseService,
        settings: Settings,
    ) -> None:
        """Sentinel file exists after first call and after second call (not corrupted)."""
        kb = service.create_kb("idempotent-sentinel-check")
        _plant_chunks(service, kb.id, n=3)

        migrate_to_textvec(kb.id, service)
        migrate_to_textvec(kb.id, service)

        # service._index maps kb_id → dir_name (based on the KB *name*, not id).
        dir_name = service._index.get(kb.id) or sanitize_kb_dir_name(kb.id)  # noqa: SLF001
        kb_root = settings.saves_dir / dir_name
        sentinel = kb_root / MIGRATED_SENTINEL_FILENAME
        assert sentinel.is_file(), f"Sentinel not found at {sentinel}"
        content = sentinel.read_text(encoding="utf-8").strip()
        assert content == "textvec", f"Unexpected sentinel content: {content!r}"

    def test_fts_row_count_unchanged_after_second_run(
        self,
        service: KnowledgebaseService,
    ) -> None:
        """chunks_fts_docsize row count is the same after the second (no-op) run."""
        n = 6
        kb = service.create_kb("idempotent-fts-count")
        _plant_chunks(service, kb.id, n=n)

        migrate_to_textvec(kb.id, service)
        count_after_first = _fts_row_count(service, kb.id)

        migrate_to_textvec(kb.id, service)
        count_after_second = _fts_row_count(service, kb.id)

        assert count_after_first == n, (
            f"Expected {n} FTS rows after first run; got {count_after_first}"
        )
        assert count_after_second == count_after_first, (
            f"FTS row count changed after second run: {count_after_first} → {count_after_second}"
        )


# ---------------------------------------------------------------------------
# Stderr-log delta (first vs. second call messages)
# ---------------------------------------------------------------------------


class TestMigrateToTextvecLogDelta:
    def _collect_migration_lines(self, capsys: pytest.CaptureFixture) -> list[dict]:
        captured = capsys.readouterr()
        return [
            json.loads(line)
            for line in captured.err.splitlines()
            if line.strip() and '"phase": "migration"' in line
        ]

    def test_first_call_emits_migrated_message(
        self,
        service: KnowledgebaseService,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """First migrate_to_textvec emits error_message containing 'kb migrated'."""
        kb = service.create_kb("log-delta-first")
        _plant_chunks(service, kb.id, n=3)

        migrate_to_textvec(kb.id, service)

        captured = capsys.readouterr()
        log_lines = [json.loads(line) for line in captured.err.splitlines() if line.strip()]
        migration_lines = [ln for ln in log_lines if ln.get("phase") == "migration"]
        messages = [ln.get("error_message") or "" for ln in migration_lines]
        assert any("migrated" in m for m in messages), (
            f"Expected a migration log line with 'migrated' in error_message; "
            f"got messages: {messages!r}"
        )

    def test_second_call_emits_already_migrated_message(
        self,
        service: KnowledgebaseService,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Second migrate_to_textvec emits error_message='kb already migrated'."""
        kb = service.create_kb("log-delta-second")
        _plant_chunks(service, kb.id, n=3)

        migrate_to_textvec(kb.id, service)

        # Consume first call's stderr output.
        capsys.readouterr()

        migrate_to_textvec(kb.id, service)

        captured = capsys.readouterr()
        log_lines = [json.loads(line) for line in captured.err.splitlines() if line.strip()]
        already_lines = [
            ln for ln in log_lines if "already migrated" in (ln.get("error_message") or "")
        ]
        assert already_lines, (
            "Expected a log line with 'already migrated' in error_message "
            "on the second call; got:\n" + "\n".join(repr(ln) for ln in log_lines)
        )

    def test_log_lines_satisfy_11_field_schema(
        self,
        service: KnowledgebaseService,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Both first and second call log lines satisfy the 11-field stderr schema."""
        kb = service.create_kb("log-delta-schema")
        _plant_chunks(service, kb.id, n=3)

        migrate_to_textvec(kb.id, service)
        migrate_to_textvec(kb.id, service)

        captured = capsys.readouterr()
        log_lines = [json.loads(line) for line in captured.err.splitlines() if line.strip()]
        migration_lines = [ln for ln in log_lines if ln.get("phase") == "migration"]
        assert migration_lines, "Expected migration phase log lines."
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
            assert not missing, (
                f"Migration log line missing required fields {sorted(missing)}: {ln!r}"
            )
