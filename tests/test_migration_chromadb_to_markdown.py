"""Phase 4 — chromadb -> markdown migration round-trip.

spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b (v2.1)
Source: pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json
        implementation.phases[4].tasks[7] — ingest under chromadb,
        run migrate, assert wiki/ created, assert kb_query routes to
        markdown backend, assert old chromadb collection still
        readable until cutover sentinel written.

Test surface
------------

* ``services.migration.export_to_markdown`` writes a wiki tree
  reflecting the source data.
* ``KnowledgebaseService.migrate(kb_id=..., target_backend='markdown')``
  writes the ``.migrated_to`` sentinel file and invalidates the
  per-KB backend cache.
* After the migration, ``svc.query`` routes through the markdown
  backend (verified by the markdown backend's signature backend tag).
* The old chromadb chunks remain on disk — non-destructive.
* ``backfill_provider_snapshot`` populates legacy NULL chunk columns.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_knowledgebase.backends import (
    MIGRATED_SENTINEL_FILENAME,
    resolve_backend_name,
)
from agent_knowledgebase.config import Settings
from agent_knowledgebase.models import Chunk
from agent_knowledgebase.services.knowledgebase import KnowledgebaseService
from agent_knowledgebase.services.migration import (
    backfill_provider_snapshot,
    export_chromadb_dump,
    export_to_markdown,
    import_chromadb_dump,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def chromadb_service(tmp_path: Path) -> KnowledgebaseService:
    """Spin up a KnowledgebaseService rooted at ``tmp_path/saves``.

    The default ``kb_backend='chromadb'`` is preserved — this fixture
    is the v0.6.0 baseline against which migration is exercised.
    """
    saves = tmp_path / "saves"
    saves.mkdir()
    settings = Settings(
        saves_dir=saves,
        embedding_provider="ollama",
        embedding_model="qwen3-embedding:8b",
        embed_base_url="http://127.0.0.1:11434",
    ).resolve_paths()
    return KnowledgebaseService(settings)


def _seed_kb_with_chunks(svc: KnowledgebaseService, *, kb_name: str) -> str:
    """Create a KB and hand-insert two chunks (no real embedder)."""
    kb = svc.create_kb(name=kb_name)
    ctx = svc._ctx(kb.id)
    ctx.db._conn.execute(
        "INSERT INTO sources (id, kb_id, source_type, uri, status, dedup_key) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (f"src-{kb.id}", kb.id, "file", "/tmp/seed.txt", "ingested", "seed-key"),
    )
    ctx.db._conn.commit()
    for i in range(2):
        ctx.db.insert_chunk(
            Chunk(
                id=f"chunk-{kb.id}-{i}",
                kb_id=kb.id,
                source_id=f"src-{kb.id}",
                content=f"chunk {i} body alpha beta gamma",
                metadata={
                    "embedding_model": "qwen3-embedding:8b",
                    "embedding_provider": "ollama",
                    "embed_base_url": "http://127.0.0.1:11434",
                },
            )
        )
    return kb.id


# ---------------------------------------------------------------------------
# export_to_markdown writes a wiki tree
# ---------------------------------------------------------------------------


def test_export_to_markdown_writes_wiki_tree(chromadb_service):
    """Calling export_to_markdown materialises pages/, index.md, log.md."""
    kb_id = _seed_kb_with_chunks(chromadb_service, kb_name="exp-md")
    dest = chromadb_service._config.saves_dir / chromadb_service._index[kb_id]
    pages_written = export_to_markdown(chromadb_service, kb_id, dest)
    assert pages_written == 1, "one source -> one wiki page"
    wiki_root = dest / "wiki"
    assert (wiki_root / "pages").is_dir()
    assert any((wiki_root / "pages").glob("*.md"))
    assert (wiki_root / "index.md").is_file()


# ---------------------------------------------------------------------------
# migrate(kb_id, 'markdown') writes the sentinel and re-routes queries
# ---------------------------------------------------------------------------


def test_migrate_writes_sentinel_and_reroutes_query(chromadb_service):
    kb_id = _seed_kb_with_chunks(chromadb_service, kb_name="mig-md")

    # Pre-condition: backend resolution returns chromadb (the global default).
    assert resolve_backend_name(
        chromadb_service._config, kb_id=kb_id, service=chromadb_service
    ) == "chromadb"

    result = chromadb_service.migrate(kb_id=kb_id, target_backend="markdown")
    assert result["target_backend"] == "markdown"
    assert result["pages_migrated"] == 1
    sentinel = Path(result["sentinel_path"])
    assert sentinel.is_file()
    assert sentinel.name == MIGRATED_SENTINEL_FILENAME
    assert sentinel.read_text(encoding="utf-8").strip() == "markdown"

    # Post-condition: backend resolution honours the sentinel.
    assert resolve_backend_name(
        chromadb_service._config, kb_id=kb_id, service=chromadb_service
    ) == "markdown"
    # And the per-KB backend cache was invalidated so the next query
    # builds a markdown backend.
    backend = chromadb_service._backend_for(kb_id)
    from agent_knowledgebase.backends.markdown_backend import MarkdownWikiBackend
    assert isinstance(backend, MarkdownWikiBackend)


def test_migrate_is_non_destructive_old_chromadb_data_remains(chromadb_service):
    """The OLD chromadb data is left intact — sentinel is just routing."""
    kb_id = _seed_kb_with_chunks(chromadb_service, kb_name="non-dest")
    ctx = chromadb_service._ctx(kb_id)
    chunks_before = len(list(ctx.db._conn.execute(
        "SELECT id FROM chunks WHERE kb_id = ?", (kb_id,)
    ).fetchall()))
    assert chunks_before == 2

    chromadb_service.migrate(kb_id=kb_id, target_backend="markdown")

    # Chunks still in SQLite (not deleted).
    chunks_after = len(list(ctx.db._conn.execute(
        "SELECT id FROM chunks WHERE kb_id = ?", (kb_id,)
    ).fetchall()))
    assert chunks_after == chunks_before
    assert chunks_after == 2


def test_migrate_rejects_invalid_target_backend(chromadb_service):
    kb_id = _seed_kb_with_chunks(chromadb_service, kb_name="bad-target")
    with pytest.raises(ValueError, match="not supported"):
        chromadb_service.migrate(kb_id=kb_id, target_backend="lightrag")
    with pytest.raises(ValueError, match="not supported"):
        chromadb_service.migrate(kb_id=kb_id, target_backend="bogus")


def test_migrate_unknown_kb_id_raises(chromadb_service):
    with pytest.raises(ValueError, match="unknown kb_id"):
        chromadb_service.migrate(kb_id="does-not-exist", target_backend="markdown")


# ---------------------------------------------------------------------------
# After migration, kb_query routes through markdown (FTS5)
# ---------------------------------------------------------------------------


def test_kb_query_after_migration_routes_to_markdown(chromadb_service):
    kb_id = _seed_kb_with_chunks(chromadb_service, kb_name="kbq-md")
    chromadb_service.migrate(kb_id=kb_id, target_backend="markdown")

    # Query through the high-level service — should hit the markdown
    # FTS5 path (no embedder involvement). With our seed data, FTS5
    # MATCH on "alpha" should return at least one result.
    results = chromadb_service.search(kb_id, "alpha", top_k=5)
    assert results, "expected at least one FTS5 hit after migration"
    # The markdown backend tags metadata.backend='markdown' for hits.
    assert any(
        r.metadata.get("backend") == "markdown" for r in results
    ), "results must come from the markdown backend after migration"


# ---------------------------------------------------------------------------
# chromadb dump round-trip preserves data
# ---------------------------------------------------------------------------


def test_chromadb_dump_round_trip(chromadb_service, tmp_path: Path):
    """``export_chromadb_dump`` -> ``import_chromadb_dump`` preserves rows."""
    kb_id = _seed_kb_with_chunks(chromadb_service, kb_name="dump-rt")
    dump_path = tmp_path / "dump.json"
    rows_dumped = export_chromadb_dump(chromadb_service, kb_id, dump_path)
    assert rows_dumped == 2

    # Wipe the chunks and re-import.
    ctx = chromadb_service._ctx(kb_id)
    ctx.db._conn.execute("DELETE FROM chunks WHERE kb_id = ?", (kb_id,))
    ctx.db._conn.commit()
    rows_restored = import_chromadb_dump(chromadb_service, kb_id, dump_path)
    assert rows_restored == 2

    # Round-trip preserved.
    after = list(ctx.db._conn.execute(
        "SELECT id FROM chunks WHERE kb_id = ?", (kb_id,)
    ).fetchall())
    assert len(after) == 2


# ---------------------------------------------------------------------------
# backfill_provider_snapshot populates legacy KBs from kb config
# ---------------------------------------------------------------------------


def test_backfill_provider_snapshot_legacy_kb(chromadb_service):
    kb_id = _seed_kb_with_chunks(chromadb_service, kb_name="bf-leg")
    # Wipe the snapshot columns to simulate a v0.6.0 KB ingested before
    # the Phase 4 migration ran (the backfill should re-populate them).
    ctx = chromadb_service._ctx(kb_id)
    ctx.db._conn.execute(
        "UPDATE chunks SET embedding_provider = NULL, embed_base_url = NULL "
        "WHERE kb_id = ?",
        (kb_id,),
    )
    ctx.db._conn.commit()

    summary = backfill_provider_snapshot(chromadb_service, kb_id=kb_id)
    assert summary == {kb_id: 2}, "expected 2 chunks updated"

    # Verify the columns are populated.
    rows = ctx.db._conn.execute(
        "SELECT embedding_provider, embed_base_url FROM chunks WHERE kb_id = ?",
        (kb_id,),
    ).fetchall()
    for row in rows:
        assert row["embedding_provider"] == "ollama"
        assert row["embed_base_url"] == "http://127.0.0.1:11434"


def test_backfill_provider_snapshot_dry_run(chromadb_service):
    kb_id = _seed_kb_with_chunks(chromadb_service, kb_name="bf-dry")
    ctx = chromadb_service._ctx(kb_id)
    ctx.db._conn.execute(
        "UPDATE chunks SET embedding_provider = NULL, embed_base_url = NULL "
        "WHERE kb_id = ?",
        (kb_id,),
    )
    ctx.db._conn.commit()

    # Dry-run reports the count but doesn't write.
    summary = backfill_provider_snapshot(
        chromadb_service, kb_id=kb_id, dry_run=True
    )
    assert summary == {kb_id: 2}
    rows = ctx.db._conn.execute(
        "SELECT embedding_provider FROM chunks WHERE kb_id = ?",
        (kb_id,),
    ).fetchall()
    assert all(row["embedding_provider"] is None for row in rows)


# ---------------------------------------------------------------------------
# kb_migrate MCP tool surface check
# ---------------------------------------------------------------------------


def test_kb_migrate_is_registered_with_correct_decorator_order():
    """The new tool ships with @mcp.tool() outer + @_with_tool_timeout inner.

    The contract is enforced by tests/contract/test_decorator_order.py
    via ``__wrapped__``; this test is a smoke check that the same
    invariant holds for the new tool name (so a regression is local).
    """
    from agent_knowledgebase.server import mcp

    assert "kb_migrate" in mcp._tool_manager._tools, (
        "kb_migrate MCP tool must be registered"
    )
    fn = mcp._tool_manager._tools["kb_migrate"].fn
    assert hasattr(fn, "__wrapped__"), (
        "kb_migrate must be wrapped by @_with_tool_timeout (decorator order: "
        "@mcp.tool() outer, @_with_tool_timeout inner)"
    )
