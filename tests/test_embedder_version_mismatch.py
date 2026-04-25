"""Phase 5 — mixed embedder_version rejection at ingest time.

spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b (v2.1)
Source: pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json
        implementation.phases[5].tasks[5,6] — stamp embedder_version
        per chunk; reject mixed embedder_version on same kb_id at
        ingest time unless AGENT_KB_AUTO_REEMBED=1.

What's being tested
-------------------

* ``Database`` schema migration adds ``embedder_version`` column to
  ``chunks`` (additive, ALTER TABLE only, defaults to NULL).
* ``Database.insert_chunk`` propagates ``metadata['embedder_version']``
  into the native column.
* ``Database.get_embedder_versions(kb_id)`` returns the distinct set
  of stamped versions (skipping NULLs from legacy/unstamped chunks).
* ``KnowledgebaseService._ingest_source_locked`` raises
  ``EmbedderVersionMismatchError`` when the incoming embedder's
  version differs from any version already stamped on the KB's
  chunks AND ``AGENT_KB_AUTO_REEMBED=1`` is NOT set.
* ``AGENT_KB_AUTO_REEMBED=1`` allows the ingest through; the new
  chunks carry the new version.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent_knowledgebase.config import Settings
from agent_knowledgebase.database import Database
from agent_knowledgebase.models import Chunk, SourceType
from agent_knowledgebase.services.knowledgebase import (
    EmbedderVersionMismatchError,
    KnowledgebaseService,
)


# ---------------------------------------------------------------------------
# Schema migration adds the column
# ---------------------------------------------------------------------------


def test_chunks_table_has_embedder_version_column(tmp_path: Path) -> None:
    """Phase 5 ALTER ADD adds a single TEXT column named embedder_version."""
    db = Database(db_path=tmp_path / "x.db")
    cols = {row[1] for row in db._conn.execute("PRAGMA table_info(chunks)").fetchall()}
    assert "embedder_version" in cols, (
        "Phase 5 migration must add chunks.embedder_version column"
    )
    db.close()


def test_chunks_embedder_version_migration_is_idempotent(tmp_path: Path) -> None:
    """Re-opening the same DB does not raise DUPLICATE COLUMN."""
    db_path = tmp_path / "x.db"
    db = Database(db_path=db_path)
    db.close()
    # Re-open: schema bootstrap should be a no-op.
    db2 = Database(db_path=db_path)
    cols = {row[1] for row in db2._conn.execute("PRAGMA table_info(chunks)").fetchall()}
    assert "embedder_version" in cols
    db2.close()


# ---------------------------------------------------------------------------
# insert_chunk + get_embedder_versions round trip
# ---------------------------------------------------------------------------


def _seed_kb_and_source(db: Database, kb_id: str = "kb-1") -> None:
    db._conn.execute(
        "INSERT INTO knowledgebases (id, name, created_at, updated_at) "
        "VALUES (?, ?, ?, ?)",
        (kb_id, kb_id, "2026-04-24T00:00:00", "2026-04-24T00:00:00"),
    )
    db._conn.execute(
        "INSERT INTO sources (id, kb_id, source_type, uri, status) "
        "VALUES (?, ?, ?, ?, ?)",
        (f"src-{kb_id}", kb_id, "file", f"x://{kb_id}", "ingested"),
    )
    db._conn.commit()


def test_insert_chunk_stamps_embedder_version_native_column(
    tmp_path: Path,
) -> None:
    """``insert_chunk`` propagates ``metadata['embedder_version']``
    into the native ``embedder_version`` column."""
    db = Database(db_path=tmp_path / "x.db")
    _seed_kb_and_source(db)
    db.insert_chunk(
        Chunk(
            kb_id="kb-1",
            source_id="src-kb-1",
            content="hello",
            metadata={
                "embedding_model": "text-embedding-3-small",
                "embedder_version": "remote/text-embedding-3-small@https://api.openai.com/v1",
            },
        )
    )
    row = db._conn.execute(
        "SELECT embedder_version FROM chunks WHERE kb_id = ?", ("kb-1",)
    ).fetchone()
    assert (
        row["embedder_version"]
        == "remote/text-embedding-3-small@https://api.openai.com/v1"
    )
    db.close()


def test_get_embedder_versions_returns_distinct_set(tmp_path: Path) -> None:
    """``get_embedder_versions`` returns the distinct set of stamped
    versions (deduplicating dupes, ignoring NULLs)."""
    db = Database(db_path=tmp_path / "x.db")
    _seed_kb_and_source(db)
    db.insert_chunk(
        Chunk(
            kb_id="kb-1",
            source_id="src-kb-1",
            content="a",
            metadata={"embedder_version": "remote/t-3-small@x"},
        )
    )
    db.insert_chunk(
        Chunk(
            kb_id="kb-1",
            source_id="src-kb-1",
            content="b",
            metadata={"embedder_version": "remote/t-3-small@x"},
        )
    )
    db.insert_chunk(
        Chunk(
            kb_id="kb-1",
            source_id="src-kb-1",
            content="c",
            metadata={"embedder_version": "fastembed/MiniLM-L6-v2-int8"},
        )
    )
    versions = db.get_embedder_versions("kb-1")
    assert versions == {
        "remote/t-3-small@x",
        "fastembed/MiniLM-L6-v2-int8",
    }
    db.close()


def test_get_embedder_versions_ignores_null_legacy_chunks(
    tmp_path: Path,
) -> None:
    """Legacy chunks with NULL embedder_version don't appear in the set."""
    db = Database(db_path=tmp_path / "x.db")
    _seed_kb_and_source(db)
    # Legacy-shaped insert: no embedder_version stamping.
    db._conn.execute(
        "INSERT INTO chunks (id, source_id, kb_id, content, metadata, embedding_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("legacy-1", "src-kb-1", "kb-1", "old", "{}", None),
    )
    db._conn.commit()
    assert db.get_embedder_versions("kb-1") == set()
    db.close()


def test_get_embedder_versions_falls_back_to_metadata_blob(
    tmp_path: Path,
) -> None:
    """When the native column is NULL but ``metadata.embedder_version``
    is set, the JSON fallback recovers it (mirrors the snapshot
    fallback in get_embedding_snapshot)."""
    db = Database(db_path=tmp_path / "x.db")
    _seed_kb_and_source(db)
    # Insert with embedder_version ONLY in metadata (simulate a chunk
    # ingested via a code path that only stamped the metadata blob).
    db._conn.execute(
        "INSERT INTO chunks (id, source_id, kb_id, content, metadata, embedding_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            "meta-1",
            "src-kb-1",
            "kb-1",
            "x",
            '{"embedder_version": "remote/v@x"}',
            None,
        ),
    )
    db._conn.commit()
    assert db.get_embedder_versions("kb-1") == {"remote/v@x"}
    db.close()


# ---------------------------------------------------------------------------
# Mixed-version rejection at ingest time
# ---------------------------------------------------------------------------


class _StubEmbedder:
    """Minimal Embedder that lets us pin the version to anything we want."""

    def __init__(
        self,
        *,
        model_name: str,
        version: str,
        dimension: int = 4,
    ) -> None:
        self._model_name = model_name
        self._version = version
        self._dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4][: self._dimension] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4][: self._dimension]

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedder_version(self) -> str:
        return self._version

    def probe_dimension(self) -> int:
        return self._dimension


def _build_service_with_stub(
    saves: Path,
    *,
    embedder_version: str,
    model_name: str = "stub-model",
) -> KnowledgebaseService:
    """Build a service wired to a `_StubEmbedder`.

    Bypasses ``create_embedder`` entirely — we set ``_embedder_instance``
    on the service so the mixed-version check sees our pinned
    version.
    """
    settings = Settings(
        saves_dir=saves,
        embedding_provider="ollama",
        embedding_model=model_name,
    ).resolve_paths()
    svc = KnowledgebaseService(settings)
    svc._embedder_instance = _StubEmbedder(
        model_name=model_name, version=embedder_version
    )
    return svc


def _ingest_text(svc: KnowledgebaseService, kb_id: str, body: str) -> Any:
    """Ingest a single in-memory text snippet via the file source_type."""
    # Use a tmpfile under the saves_dir to avoid path quirks.
    saves = svc._config.saves_dir
    tmpfile = saves / f"_test_{kb_id}_{abs(hash(body))}.txt"
    tmpfile.write_text(body, encoding="utf-8")
    return svc.ingest_source(
        kb_id=kb_id, source_type=SourceType.file, uri=str(tmpfile)
    )


def test_ingest_under_same_version_succeeds(tmp_path: Path) -> None:
    """Two ingests under the SAME embedder_version: no rejection."""
    saves = tmp_path / "saves"
    saves.mkdir()
    svc = _build_service_with_stub(saves, embedder_version="stub/v1")
    kb = svc.create_kb(name="same-ver")
    _ingest_text(svc, kb.id, "first body")
    _ingest_text(svc, kb.id, "second body")
    versions = svc._ctx(kb.id).db.get_embedder_versions(kb.id)
    assert versions == {"stub/v1"}


def test_ingest_under_different_version_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second ingest under a DIFFERENT embedder_version raises
    EmbedderVersionMismatchError when AGENT_KB_AUTO_REEMBED is not set."""
    monkeypatch.delenv("AGENT_KB_AUTO_REEMBED", raising=False)
    saves = tmp_path / "saves"
    saves.mkdir()
    svc = _build_service_with_stub(saves, embedder_version="stub/v1")
    kb = svc.create_kb(name="diff-ver")
    _ingest_text(svc, kb.id, "first body")

    # Swap embedder to a different version.
    svc._embedder_instance = _StubEmbedder(
        model_name="stub-model", version="stub/v2"
    )
    with pytest.raises(EmbedderVersionMismatchError) as excinfo:
        _ingest_text(svc, kb.id, "second body")
    err = excinfo.value
    assert err.error_code == "EMBEDDER_VERSION_MISMATCH"
    assert err.kb_id == kb.id
    assert err.existing_versions == {"stub/v1"}
    assert err.incoming_version == "stub/v2"
    payload = err.to_payload()
    assert payload["error_code"] == "EMBEDDER_VERSION_MISMATCH"
    assert payload["bypass_env"] == "AGENT_KB_AUTO_REEMBED=1"


def test_ingest_under_different_version_passes_with_bypass_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``AGENT_KB_AUTO_REEMBED=1`` lets a different embedder_version through."""
    monkeypatch.setenv("AGENT_KB_AUTO_REEMBED", "1")
    saves = tmp_path / "saves"
    saves.mkdir()
    svc = _build_service_with_stub(saves, embedder_version="stub/v1")
    kb = svc.create_kb(name="bypass")
    _ingest_text(svc, kb.id, "first body")

    svc._embedder_instance = _StubEmbedder(
        model_name="stub-model", version="stub/v2"
    )
    # No raise — bypass env is honoured.
    _ingest_text(svc, kb.id, "second body")

    versions = svc._ctx(kb.id).db.get_embedder_versions(kb.id)
    assert versions == {"stub/v1", "stub/v2"}, (
        "AUTO_REEMBED bypass must let the new version land in the KB"
    )


def test_ingest_into_legacy_kb_with_no_version_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A KB that holds only NULL-version legacy chunks accepts a new
    versioned ingest without rejection (existing_versions is empty)."""
    monkeypatch.delenv("AGENT_KB_AUTO_REEMBED", raising=False)
    saves = tmp_path / "saves"
    saves.mkdir()
    svc = _build_service_with_stub(saves, embedder_version="stub/v1")
    kb = svc.create_kb(name="legacy-mix")
    # Plant a legacy-shaped chunk directly with NULL embedder_version.
    db = svc._ctx(kb.id).db
    db._conn.execute(
        "INSERT INTO sources (id, kb_id, source_type, uri, status) "
        "VALUES (?, ?, ?, ?, ?)",
        ("src-legacy", kb.id, "file", "x://legacy", "ingested"),
    )
    db._conn.execute(
        "INSERT INTO chunks (id, source_id, kb_id, content, metadata, embedding_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("legacy-c1", "src-legacy", kb.id, "old", "{}", None),
    )
    db._conn.commit()
    # Existing versions is empty — no rejection.
    _ingest_text(svc, kb.id, "new body")
    versions = db.get_embedder_versions(kb.id)
    assert versions == {"stub/v1"}
