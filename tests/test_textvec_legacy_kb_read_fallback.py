"""Phase C contract test: TextvecBackend read-fallback hint emission.

Pins:
- Hint emission: a chunk with embedding_provider='qwen3-embedding' triggers
  a REBUILD_RECOMMENDED log line on the first query.
- Once-per-session: three queries on the same kb_id emit the hint exactly once.
- Per-kb_id isolation: querying KB-A and KB-B (both stamped legacy) emits the
  hint once per KB (2 total log lines).
- Clean KB: querying a KB with all embedding_provider IS NULL emits NO hint.

spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b (Phase C)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import pytest

from agent_knowledgebase.backends.textvec_backend import TextvecBackend
from agent_knowledgebase.config import Settings, sanitize_kb_dir_name
from agent_knowledgebase.database import Database


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_hint_set() -> None:
    """Reset the module-level hint-emitted set between tests.

    The set is intentionally session-scoped (process lifetime). Resetting it
    here ensures tests are isolated from each other and can assert the
    'exactly-once per kb_id' contract fresh every test.
    """
    import agent_knowledgebase.backends.textvec_backend as tv_mod

    tv_mod._rebuild_hint_emitted_kbs.clear()
    yield
    tv_mod._rebuild_hint_emitted_kbs.clear()


@pytest.fixture()
def saves(tmp_path: Path) -> Path:
    d = tmp_path / "saves"
    d.mkdir()
    return d


@pytest.fixture()
def settings(saves: Path) -> Settings:
    return Settings(saves_dir=saves)


def _make_db(settings: Settings, kb_id: str) -> Database:
    """Create a per-KB SQLite DB with a knowledgebases row seeded."""
    safe_dir = sanitize_kb_dir_name(kb_id)
    db_path = settings.kb_db_path(safe_dir)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = Database(db_path=db_path)
    now = datetime.utcnow().isoformat()
    db._conn.execute(
        "INSERT OR IGNORE INTO knowledgebases "
        "(id, name, description, config, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (kb_id, kb_id, "", "{}", now, now),
    )
    db._conn.execute(
        "INSERT OR IGNORE INTO sources (id, kb_id, source_type, uri, dedup_key, status) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (f"src-{kb_id}", kb_id, "file", "/tmp/test.txt", "legacy-dedup", "ingested"),
    )
    db._conn.commit()
    return db


def _insert_chunk(
    db: Database,
    kb_id: str,
    chunk_id: str,
    content: str,
    *,
    embedding_provider: str | None = None,
) -> None:
    """Insert a single chunk row with optional embedding_provider stamping."""
    source_id = f"src-{kb_id}"
    db._conn.execute(
        "INSERT INTO chunks "
        "(id, source_id, kb_id, content, metadata, embedding_provider) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (chunk_id, source_id, kb_id, content, "{}", embedding_provider),
    )
    db._conn.commit()


# ---------------------------------------------------------------------------
# Hint emission: legacy-stamped KB triggers REBUILD_RECOMMENDED
# ---------------------------------------------------------------------------


class TestRebuildHintEmission:
    def test_hint_emitted_for_legacy_stamped_kb(
        self,
        settings: Settings,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Querying a KB with embedding_provider non-NULL emits REBUILD_RECOMMENDED."""
        kb_id = f"legacy-{uuid.uuid4().hex[:8]}"
        db = _make_db(settings, kb_id)
        _insert_chunk(
            db,
            kb_id,
            "cA",
            "HINTTOKEN fox jumped over the lazy dog",
            embedding_provider="qwen3-embedding",
        )

        backend = TextvecBackend(settings, service=None)
        _ = backend.search(kb_id=kb_id, text="HINTTOKEN fox", top_k=5)

        captured = capsys.readouterr()
        log_lines = [json.loads(line) for line in captured.err.splitlines() if line.strip()]
        hint_lines = [ln for ln in log_lines if ln.get("error_code") == "REBUILD_RECOMMENDED"]
        assert hint_lines, "Expected a REBUILD_RECOMMENDED log line; got:\n" + "\n".join(
            repr(ln) for ln in log_lines
        )
        # Verify the hint message mentions the cleanup command.
        msg = hint_lines[0].get("error_message", "")
        assert "kb_rebuild_index" in msg, (
            f"Expected 'kb_rebuild_index' in error_message, got: {msg!r}"
        )

    def test_hint_not_emitted_for_clean_kb(
        self,
        settings: Settings,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Querying a KB with all embedding_provider IS NULL emits NO hint."""
        kb_id = f"clean-{uuid.uuid4().hex[:8]}"
        db = _make_db(settings, kb_id)
        _insert_chunk(db, kb_id, "cB", "HINTTOKEN clean fox jumped", embedding_provider=None)

        backend = TextvecBackend(settings, service=None)
        _ = backend.search(kb_id=kb_id, text="HINTTOKEN clean", top_k=5)

        captured = capsys.readouterr()
        log_lines = [json.loads(line) for line in captured.err.splitlines() if line.strip()]
        hint_lines = [ln for ln in log_lines if ln.get("error_code") == "REBUILD_RECOMMENDED"]
        assert not hint_lines, (
            f"Did not expect a REBUILD_RECOMMENDED log line for clean KB; got: {hint_lines!r}"
        )


# ---------------------------------------------------------------------------
# Once-per-session: exactly one hint per kb_id
# ---------------------------------------------------------------------------


class TestRebuildHintOncePerSession:
    def test_hint_emitted_exactly_once_across_multiple_queries(
        self,
        settings: Settings,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Three queries on the same legacy-stamped kb_id emit the hint exactly once."""
        kb_id = f"once-{uuid.uuid4().hex[:8]}"
        db = _make_db(settings, kb_id)
        _insert_chunk(db, kb_id, "cC", "HINTTOKEN once fox", embedding_provider="qwen3-embedding")

        backend = TextvecBackend(settings, service=None)
        # Query 3 times.
        backend.search(kb_id=kb_id, text="HINTTOKEN once", top_k=5)
        backend.search(kb_id=kb_id, text="HINTTOKEN fox", top_k=5)
        backend.search(kb_id=kb_id, text="HINTTOKEN", top_k=5)

        captured = capsys.readouterr()
        log_lines = [json.loads(line) for line in captured.err.splitlines() if line.strip()]
        hint_lines = [ln for ln in log_lines if ln.get("error_code") == "REBUILD_RECOMMENDED"]
        assert len(hint_lines) == 1, (
            f"Expected exactly 1 REBUILD_RECOMMENDED log line across 3 queries; "
            f"got {len(hint_lines)}: {hint_lines!r}"
        )

    def test_query_also_emits_hint_once(
        self,
        settings: Settings,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """query() routes through search() and also gets the once-per-session hint."""
        kb_id = f"query-once-{uuid.uuid4().hex[:8]}"
        db = _make_db(settings, kb_id)
        _insert_chunk(
            db, kb_id, "cD", "HINTTOKEN query once fox", embedding_provider="qwen3-embedding"
        )

        backend = TextvecBackend(settings, service=None)
        backend.query(kb_id=kb_id, text="HINTTOKEN query once", top_k=5)
        backend.query(kb_id=kb_id, text="HINTTOKEN fox", top_k=5)

        captured = capsys.readouterr()
        log_lines = [json.loads(line) for line in captured.err.splitlines() if line.strip()]
        hint_lines = [ln for ln in log_lines if ln.get("error_code") == "REBUILD_RECOMMENDED"]
        assert len(hint_lines) == 1, (
            f"Expected exactly 1 REBUILD_RECOMMENDED hint across 2 query() calls; "
            f"got {len(hint_lines)}"
        )


# ---------------------------------------------------------------------------
# Per-kb_id isolation: different KBs each get their own hint
# ---------------------------------------------------------------------------


class TestRebuildHintPerKbIsolation:
    def test_hint_emitted_once_per_kb(
        self,
        settings: Settings,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Querying KB-A and KB-B (both legacy-stamped) emits the hint once each (2 total)."""
        kb_a = f"hint-a-{uuid.uuid4().hex[:8]}"
        kb_b = f"hint-b-{uuid.uuid4().hex[:8]}"
        db_a = _make_db(settings, kb_a)
        db_b = _make_db(settings, kb_b)
        _insert_chunk(db_a, kb_a, "cE", "HINTTOKEN kba fox", embedding_provider="qwen3-embedding")
        _insert_chunk(db_b, kb_b, "cF", "HINTTOKEN kbb fox", embedding_provider="qwen3-embedding")

        backend = TextvecBackend(settings, service=None)
        backend.search(kb_id=kb_a, text="HINTTOKEN kba", top_k=5)
        backend.search(kb_id=kb_b, text="HINTTOKEN kbb", top_k=5)
        # Query again — should not re-emit for either.
        backend.search(kb_id=kb_a, text="HINTTOKEN fox", top_k=5)
        backend.search(kb_id=kb_b, text="HINTTOKEN fox", top_k=5)

        captured = capsys.readouterr()
        log_lines = [json.loads(line) for line in captured.err.splitlines() if line.strip()]
        hint_lines = [ln for ln in log_lines if ln.get("error_code") == "REBUILD_RECOMMENDED"]
        assert len(hint_lines) == 2, (
            f"Expected 2 REBUILD_RECOMMENDED lines (one per KB); got {len(hint_lines)}: "
            f"{hint_lines!r}"
        )
        # Each hint should carry the correct kb_id.
        hint_kbs = {ln.get("kb_id") for ln in hint_lines}
        assert kb_a in hint_kbs, f"Expected kb_a={kb_a!r} in hint kb_ids: {hint_kbs!r}"
        assert kb_b in hint_kbs, f"Expected kb_b={kb_b!r} in hint kb_ids: {hint_kbs!r}"


# ---------------------------------------------------------------------------
# 11-field schema compliance on hint log line
# ---------------------------------------------------------------------------


class TestRebuildHintSchemaCompliance:
    def test_hint_log_line_has_11_required_fields(
        self,
        settings: Settings,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """The REBUILD_RECOMMENDED log line satisfies the 11-field stderr schema."""
        kb_id = f"schema-{uuid.uuid4().hex[:8]}"
        db = _make_db(settings, kb_id)
        _insert_chunk(db, kb_id, "cG", "HINTTOKEN schema fox", embedding_provider="qwen3-embedding")

        backend = TextvecBackend(settings, service=None)
        backend.search(kb_id=kb_id, text="HINTTOKEN schema", top_k=5)

        captured = capsys.readouterr()
        log_lines = [json.loads(line) for line in captured.err.splitlines() if line.strip()]
        hint_lines = [ln for ln in log_lines if ln.get("error_code") == "REBUILD_RECOMMENDED"]
        assert hint_lines, "No REBUILD_RECOMMENDED line found."
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
        for ln in hint_lines:
            missing = required - set(ln)
            assert not missing, (
                f"REBUILD_RECOMMENDED log line missing fields {sorted(missing)}: {ln!r}"
            )
