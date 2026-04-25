"""Phase 4 — per-page (per-chunk) provider snapshot regression tests.

spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b (v2.1)
Source: pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json
        implementation.phases[4].tasks[6]+[8] — stamp
        ``(embedding_provider, embed_base_url)`` per chunk so an
        existing v0.6.0 KB ingested under provider=ollama stays
        queryable AFTER the Phase 5 default flip swaps the global
        ``Settings.embedding_provider`` to ``'remote'``.

What's being tested
-------------------

* ``Database`` schema migration adds ``embedding_provider`` /
  ``embed_base_url`` columns to ``chunks`` (additive, ALTER TABLE
  only, defaults to NULL).
* ``insert_chunk`` stamps both columns from ``chunk.metadata`` so the
  read path can see them via the native columns AND the metadata blob.
* ``Database.get_embedding_snapshot`` returns the dominant
  ``(model, provider, base_url)`` tuple for the KB.
* ``KnowledgebaseService._query_embedder_for`` reads the snapshot and
  passes ``provider=`` / ``base_url=`` overrides into
  ``create_embedder_for_model`` — proving the original embedder is
  faithfully rebuilt even after the global default flip.
* ``create_embedder_for_model(config, model, provider=..., base_url=...)``
  honours the explicit overrides instead of falling back to config.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from agent_knowledgebase.config import Settings
from agent_knowledgebase.database import Database
from agent_knowledgebase.models import Chunk
from agent_knowledgebase.services.embeddings import (
    OllamaEmbedder,
    RemoteEmbedder,
    create_embedder_for_model,
)


# ---------------------------------------------------------------------------
# Schema migration is additive (ALTER ADD only)
# ---------------------------------------------------------------------------


def test_chunks_table_has_provider_snapshot_columns(tmp_path: Path) -> None:
    """The Phase 4 migration adds two TEXT columns on ``chunks``."""
    db = Database(db_path=tmp_path / "x.db")
    cols = {row[1] for row in db._conn.execute("PRAGMA table_info(chunks)").fetchall()}
    assert "embedding_provider" in cols, (
        "Phase 4 migration must add chunks.embedding_provider column"
    )
    assert "embed_base_url" in cols, (
        "Phase 4 migration must add chunks.embed_base_url column"
    )
    db.close()


def test_schema_migration_is_idempotent(tmp_path: Path) -> None:
    """Running ``_init_schema`` twice is a no-op (no DUPLICATE COLUMN error)."""
    db_path = tmp_path / "x.db"
    db = Database(db_path=db_path)
    db.close()
    # Re-open should NOT raise even though both columns already exist.
    db2 = Database(db_path=db_path)
    cols = {row[1] for row in db2._conn.execute("PRAGMA table_info(chunks)").fetchall()}
    assert "embedding_provider" in cols
    assert "embed_base_url" in cols
    db2.close()


# ---------------------------------------------------------------------------
# insert_chunk + get_embedding_snapshot round-trip
# ---------------------------------------------------------------------------


def _make_chunk(
    *,
    kb_id: str,
    source_id: str = "src-1",
    model: str = "qwen3-embedding:8b",
    provider: str = "ollama",
    base_url: str = "http://127.0.0.1:11434",
) -> Chunk:
    return Chunk(
        kb_id=kb_id,
        source_id=source_id,
        content="hello world",
        metadata={
            "embedding_model": model,
            "embedding_provider": provider,
            "embed_base_url": base_url,
        },
    )


def test_insert_chunk_stamps_native_columns(tmp_path: Path) -> None:
    """``insert_chunk`` propagates metadata snapshot to native columns."""
    db = Database(db_path=tmp_path / "x.db")
    # We bypass the FK constraint by populating bare rows; this test
    # exercises the schema/insertion mapping not the FK chain.
    db._conn.execute(
        "INSERT INTO knowledgebases (id, name, created_at, updated_at) "
        "VALUES (?, ?, ?, ?)",
        ("kb-1", "kb-1", "2026-04-24T00:00:00", "2026-04-24T00:00:00"),
    )
    db._conn.execute(
        "INSERT INTO sources (id, kb_id, source_type, uri, status) "
        "VALUES (?, ?, ?, ?, ?)",
        ("src-1", "kb-1", "file", "x://a", "ingested"),
    )
    db._conn.commit()
    db.insert_chunk(_make_chunk(kb_id="kb-1"))

    row = db._conn.execute(
        "SELECT embedding_provider, embed_base_url FROM chunks WHERE kb_id = ?",
        ("kb-1",),
    ).fetchone()
    assert row["embedding_provider"] == "ollama"
    assert row["embed_base_url"] == "http://127.0.0.1:11434"
    db.close()


def test_get_embedding_snapshot_returns_dominant_tuple(tmp_path: Path) -> None:
    """``get_embedding_snapshot`` returns the (model, provider, base_url)
    triple of the most-stamped chunk."""
    db = Database(db_path=tmp_path / "x.db")
    db._conn.execute(
        "INSERT INTO knowledgebases (id, name, created_at, updated_at) "
        "VALUES (?, ?, ?, ?)",
        ("kb-1", "kb-1", "2026-04-24T00:00:00", "2026-04-24T00:00:00"),
    )
    db._conn.execute(
        "INSERT INTO sources (id, kb_id, source_type, uri, status) "
        "VALUES (?, ?, ?, ?, ?)",
        ("src-1", "kb-1", "file", "x://a", "ingested"),
    )
    db._conn.commit()
    # Two chunks with ollama/qwen, one with remote/text-embedding-3-small.
    for _ in range(2):
        db.insert_chunk(_make_chunk(kb_id="kb-1"))
    db.insert_chunk(
        _make_chunk(
            kb_id="kb-1",
            model="text-embedding-3-small",
            provider="remote",
            base_url="https://api.openai.com/v1",
        )
    )

    snap = db.get_embedding_snapshot("kb-1")
    assert snap == (
        "qwen3-embedding:8b",
        "ollama",
        "http://127.0.0.1:11434",
    )
    db.close()


def test_get_embedding_snapshot_empty_returns_none(tmp_path: Path) -> None:
    db = Database(db_path=tmp_path / "x.db")
    assert db.get_embedding_snapshot("nonexistent-kb") is None
    db.close()


# ---------------------------------------------------------------------------
# create_embedder_for_model honours provider/base_url overrides
# ---------------------------------------------------------------------------


def test_create_embedder_for_model_legacy_one_arg_signature_still_works(
    tmp_path: Path,
) -> None:
    """Backwards-compat: ``(config, model_name)`` keeps Phase 0 behavior."""
    saves = tmp_path / "saves"
    saves.mkdir()
    # Use ollama provider so no API key is required.
    settings = Settings(
        saves_dir=saves,
        embedding_provider="ollama",
        embedding_model="qwen3-embedding:8b",
    ).resolve_paths()
    # Same model as configured -> returns the configured embedder.
    embedder = create_embedder_for_model(settings, "qwen3-embedding:8b")
    assert isinstance(embedder, OllamaEmbedder)
    assert embedder.model_name == "qwen3-embedding:8b"


def test_create_embedder_for_model_with_provider_override_builds_remote(
    tmp_path: Path,
) -> None:
    """When provider='remote' is supplied, a RemoteEmbedder is built
    even though the global config says ``provider='ollama'`` — proving
    the snapshot can rebuild a non-default embedder."""
    saves = tmp_path / "saves"
    saves.mkdir()
    settings = Settings(
        saves_dir=saves,
        embedding_provider="ollama",
        embedding_model="qwen3-embedding:8b",
        # Provide an api key so the remote-build path doesn't reject.
        embed_api_key="sk-test",
    ).resolve_paths()
    embedder = create_embedder_for_model(
        settings,
        "text-embedding-3-small",
        provider="remote",
        base_url="https://api.openai.com/v1",
    )
    assert isinstance(embedder, RemoteEmbedder)
    assert embedder.model_name == "text-embedding-3-small"


def test_query_embedder_for_uses_snapshot_after_default_flip(
    tmp_path: Path,
) -> None:
    """End-to-end: ingest under provider=ollama, flip the live config
    to provider=remote, query the same KB — the snapshot resolution
    must rebuild the original Ollama embedder for that KB.

    This is the spec's Phase 5 safety property in test form.
    """
    from agent_knowledgebase.services.knowledgebase import KnowledgebaseService

    saves = tmp_path / "saves"
    saves.mkdir()
    # Initial settings -> ingest path stamps provider=ollama on chunks.
    settings = Settings(
        saves_dir=saves,
        embedding_provider="ollama",
        embedding_model="qwen3-embedding:8b",
        embed_base_url="http://127.0.0.1:11434",
    ).resolve_paths()

    svc = KnowledgebaseService(settings)
    kb = svc.create_kb(name="snap")

    # Hand-insert a chunk with the ollama snapshot so we don't have to
    # spin up a real embedder. This mirrors what _ingest_source_locked
    # would have stamped.
    ctx = svc._ctx(kb.id)
    ctx.db._conn.execute(
        "INSERT INTO sources (id, kb_id, source_type, uri, status) "
        "VALUES (?, ?, ?, ?, ?)",
        ("src-snap", kb.id, "file", "x://snap", "ingested"),
    )
    ctx.db._conn.commit()
    ctx.db.insert_chunk(
        Chunk(
            id="chunk-snap",
            kb_id=kb.id,
            source_id="src-snap",
            content="snapshot test",
            metadata={
                "embedding_model": "qwen3-embedding:8b",
                "embedding_provider": "ollama",
                "embed_base_url": "http://127.0.0.1:11434",
            },
        )
    )

    # Rebuild the service with a flipped config: provider=remote,
    # model=text-embedding-3-small. This simulates the Phase 5
    # default-flip operating against an existing Phase 4-stamped KB.
    flipped = Settings(
        saves_dir=saves,
        embedding_provider="remote",
        embedding_model="text-embedding-3-small",
        embed_base_url="https://api.openai.com/v1",
        embed_api_key="sk-test",
    ).resolve_paths()
    svc2 = KnowledgebaseService(flipped)

    # Patch create_embedder_for_model so we can inspect what got
    # passed in without actually instantiating an embedder. Patch
    # in the knowledgebase module's namespace because the symbol is
    # imported there (resolved at module load).
    captured: dict = {}

    def fake_factory(config, model_name, *, provider=None, base_url=None):
        captured["model_name"] = model_name
        captured["provider"] = provider
        captured["base_url"] = base_url
        # Return a sentinel; the test only inspects the call args.
        return ("rebuilt", model_name, provider, base_url)

    with patch(
        "agent_knowledgebase.services.knowledgebase.create_embedder_for_model",
        fake_factory,
    ):
        result = svc2._query_embedder_for(kb.id)
    assert captured["model_name"] == "qwen3-embedding:8b", (
        "snapshot must rebuild with the ORIGINAL ollama model, not "
        "the post-flip default text-embedding-3-small"
    )
    assert captured["provider"] == "ollama", (
        "snapshot must override provider so post-flip defaults don't "
        "silently break the existing v0.6.0 KB"
    )
    assert captured["base_url"] == "http://127.0.0.1:11434"
    # Sentinel from our fake factory came back unchanged:
    assert result == ("rebuilt", "qwen3-embedding:8b", "ollama", "http://127.0.0.1:11434")


# ---------------------------------------------------------------------------
# Backfill helper updates legacy NULL columns from kb config
# ---------------------------------------------------------------------------


def test_backfill_stamps_columns_for_legacy_chunks(tmp_path: Path) -> None:
    """``backfill_embedding_snapshot`` populates NULL columns from
    the supplied (provider, base_url) tuple, never overwriting existing
    non-null values."""
    db = Database(db_path=tmp_path / "x.db")
    db._conn.execute(
        "INSERT INTO knowledgebases (id, name, created_at, updated_at) "
        "VALUES (?, ?, ?, ?)",
        ("kb-leg", "kb-leg", "2026-04-24T00:00:00", "2026-04-24T00:00:00"),
    )
    db._conn.execute(
        "INSERT INTO sources (id, kb_id, source_type, uri, status) "
        "VALUES (?, ?, ?, ?, ?)",
        ("src-leg", "kb-leg", "file", "x://a", "ingested"),
    )
    db._conn.commit()
    # Insert a "legacy" chunk row directly (without provider snapshot).
    db._conn.execute(
        "INSERT INTO chunks (id, source_id, kb_id, content, metadata, embedding_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("chunk-leg", "src-leg", "kb-leg", "x", "{}", None),
    )
    db._conn.commit()
    rows_updated = db.backfill_embedding_snapshot(
        kb_id="kb-leg",
        provider="ollama",
        base_url="http://127.0.0.1:11434",
    )
    assert rows_updated == 1
    row = db._conn.execute(
        "SELECT embedding_provider, embed_base_url FROM chunks WHERE id = ?",
        ("chunk-leg",),
    ).fetchone()
    assert row["embedding_provider"] == "ollama"
    assert row["embed_base_url"] == "http://127.0.0.1:11434"
    # Re-running backfill is a no-op on already-populated rows.
    rerun = db.backfill_embedding_snapshot(
        kb_id="kb-leg",
        provider="other",
        base_url="other",
    )
    assert rerun == 0
    db.close()


# ---------------------------------------------------------------------------
# I-02 — backfill restricts by metadata.embedding_model
# ---------------------------------------------------------------------------


def test_backfill_skips_chunks_with_mismatched_embedding_model(
    tmp_path: Path,
) -> None:
    """I-02: ``backfill_provider_snapshot`` must only stamp chunks
    whose ``metadata.embedding_model`` matches the current
    ``Settings.embedding_model``. A mixed-history KB whose chunks were
    ingested under MULTIPLE different embedders must NOT have the
    process-wide provider misstamped onto the chunks that came from a
    different embedder.

    Bug history: the old implementation read
    ``service._config.embedding_provider`` and stamped every NULL
    chunk regardless of model, so a switch from Ollama to Remote
    would silently misstamp the older Ollama chunks with
    ``provider='remote'``.
    """
    from agent_knowledgebase.config import Settings
    from agent_knowledgebase.services.knowledgebase import KnowledgebaseService
    from agent_knowledgebase.services.migration import backfill_provider_snapshot

    saves = tmp_path / "saves"
    saves.mkdir()
    # Live config says model=text-embedding-3-small, provider=remote.
    settings = Settings(
        saves_dir=saves,
        embedding_provider="remote",
        embedding_model="text-embedding-3-small",
        embed_base_url="https://api.openai.com/v1",
        embed_api_key="sk-test",
    ).resolve_paths()
    svc = KnowledgebaseService(settings)
    kb = svc.create_kb(name="bf-mixed")
    ctx = svc._ctx(kb.id)

    # Insert a source row so the FK constraints are satisfied.
    ctx.db._conn.execute(
        "INSERT INTO sources (id, kb_id, source_type, uri, status) "
        "VALUES (?, ?, ?, ?, ?)",
        (f"src-{kb.id}", kb.id, "file", "x://mixed", "ingested"),
    )
    ctx.db._conn.commit()

    # 5 chunks with embedding_model='text-embedding-3-small' (matches
    # current settings); should be stamped.
    for i in range(5):
        ctx.db._conn.execute(
            "INSERT INTO chunks (id, source_id, kb_id, content, metadata, embedding_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                f"c-match-{i}",
                f"src-{kb.id}",
                kb.id,
                f"matching {i}",
                json.dumps({"embedding_model": "text-embedding-3-small"}),
                None,
            ),
        )
    # 5 chunks with embedding_model='qwen3-embedding:8b' (DIFFERENT
    # model — must NOT be misstamped with provider='remote').
    for i in range(5):
        ctx.db._conn.execute(
            "INSERT INTO chunks (id, source_id, kb_id, content, metadata, embedding_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                f"c-skip-{i}",
                f"src-{kb.id}",
                kb.id,
                f"skip {i}",
                json.dumps({"embedding_model": "qwen3-embedding:8b"}),
                None,
            ),
        )
    ctx.db._conn.commit()

    summary = backfill_provider_snapshot(svc, kb_id=kb.id)
    assert summary[kb.id] == 5, (
        f"expected 5 matching chunks updated, got {summary[kb.id]} — "
        f"the mismatched-model chunks must be left untouched"
    )

    # Matching-model chunks: stamped with remote.
    matching = ctx.db._conn.execute(
        "SELECT embedding_provider, embed_base_url FROM chunks "
        "WHERE id LIKE 'c-match-%'"
    ).fetchall()
    for row in matching:
        assert row["embedding_provider"] == "remote", (
            f"matching-model chunk should be stamped 'remote', got "
            f"{row['embedding_provider']!r}"
        )
        assert row["embed_base_url"] == "https://api.openai.com/v1"

    # Mismatched-model chunks: STILL NULL (the bug catcher).
    skipped = ctx.db._conn.execute(
        "SELECT embedding_provider, embed_base_url FROM chunks "
        "WHERE id LIKE 'c-skip-%'"
    ).fetchall()
    for row in skipped:
        assert row["embedding_provider"] is None, (
            f"mismatched-model chunk MUST stay NULL (no misstamp); "
            f"got provider={row['embedding_provider']!r}. "
            f"This is the I-02 bug: the old backfill stamped every NULL "
            f"row with the process-wide provider, silently corrupting "
            f"mixed-embedder KBs."
        )
        assert row["embed_base_url"] is None


def test_backfill_embedding_snapshot_db_method_honors_model_filter(
    tmp_path: Path,
) -> None:
    """I-02 unit test for the Database method directly: the
    ``embedding_model=`` kwarg must restrict the UPDATE to matching
    rows. Without the kwarg, all NULL rows are stamped (legacy
    behavior, kept for backwards compat).
    """
    db = Database(db_path=tmp_path / "x.db")
    db._conn.execute(
        "INSERT INTO knowledgebases (id, name, created_at, updated_at) "
        "VALUES (?, ?, ?, ?)",
        ("kb-mf", "kb-mf", "2026-04-24T00:00:00", "2026-04-24T00:00:00"),
    )
    db._conn.execute(
        "INSERT INTO sources (id, kb_id, source_type, uri, status) "
        "VALUES (?, ?, ?, ?, ?)",
        ("src-mf", "kb-mf", "file", "x://a", "ingested"),
    )
    db._conn.commit()
    # 3 model-A chunks, 2 model-B chunks, all with NULL provider.
    import json as _json
    for i in range(3):
        db._conn.execute(
            "INSERT INTO chunks (id, source_id, kb_id, content, metadata, embedding_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (f"a{i}", "src-mf", "kb-mf", "x",
             _json.dumps({"embedding_model": "model-A"}), None),
        )
    for i in range(2):
        db._conn.execute(
            "INSERT INTO chunks (id, source_id, kb_id, content, metadata, embedding_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (f"b{i}", "src-mf", "kb-mf", "x",
             _json.dumps({"embedding_model": "model-B"}), None),
        )
    db._conn.commit()

    # Stamp only model-A chunks.
    n = db.backfill_embedding_snapshot(
        kb_id="kb-mf",
        provider="provider-A",
        base_url="url-A",
        embedding_model="model-A",
    )
    assert n == 3

    # model-A: stamped.
    rows_a = db._conn.execute(
        "SELECT embedding_provider FROM chunks WHERE id LIKE 'a%'"
    ).fetchall()
    assert all(r["embedding_provider"] == "provider-A" for r in rows_a)
    # model-B: untouched.
    rows_b = db._conn.execute(
        "SELECT embedding_provider FROM chunks WHERE id LIKE 'b%'"
    ).fetchall()
    assert all(r["embedding_provider"] is None for r in rows_b)
    db.close()
