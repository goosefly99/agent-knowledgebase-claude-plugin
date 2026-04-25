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


# ---------------------------------------------------------------------------
# B-01 — markdown -> chromadb reverse migration regression
# ---------------------------------------------------------------------------


def _make_mocked_chromadb_service(tmp_path: Path) -> KnowledgebaseService:
    """Return a service with mocked embedder + vectorstore so the full
    ingest pipeline runs to completion without an Ollama / OpenAI
    server. Vectorstore stores chunks in a per-kb_id Python dict so
    ``count()`` reflects the live state.
    """
    from unittest.mock import MagicMock

    saves = tmp_path / "saves"
    saves.mkdir()
    settings = Settings(
        saves_dir=saves,
        embedding_provider="ollama",
        embedding_model="qwen3-embedding:8b",
        embed_base_url="http://127.0.0.1:11434",
    ).resolve_paths()
    svc = KnowledgebaseService(settings)
    # Stub the embedder so embed() returns a fixed vector per text and
    # model_name matches the seed.
    embedder = MagicMock()
    embedder.embed.side_effect = lambda texts: [[0.1, 0.2, 0.3]] * len(texts)
    embedder.model_name = "qwen3-embedding:8b"
    svc._embedder_instance = embedder

    # Per-kb_id in-memory vectorstore so count() can be asserted on.
    stores: dict[str, dict[str, dict]] = {}

    def _vs_for(kb_id: str):
        rec = stores.setdefault(kb_id, {})
        vs = MagicMock()

        def _add(*, ids, embeddings, documents, metadatas):
            for i, e, d, m in zip(ids, embeddings, documents, metadatas):
                rec[i] = {
                    "embedding": e, "document": d, "metadata": m
                }

        def _delete(target_ids):
            for i in target_ids:
                rec.pop(i, None)

        vs.add.side_effect = _add
        vs.delete.side_effect = _delete
        vs.count.side_effect = lambda: len(rec)
        return vs

    svc._get_vectorstore = _vs_for  # type: ignore[assignment]
    # Expose the per-kb stores so tests can assert on them.
    svc._test_vector_stores = stores  # type: ignore[attr-defined]
    return svc


def test_full_chromadb_to_markdown_to_chromadb_roundtrip(tmp_path: Path) -> None:
    """B-01: reverse migration must actually land vectors in chromadb.

    Bug history: ``import_from_markdown`` previously called
    ``service.ingest_source(...)`` which routed the index write through
    ``_backend_for(kb_id)``. The OLD ``.migrated_to`` sentinel still
    said ``markdown`` mid-migration, so the cache returned the markdown
    backend and the re-ingested pages went BACK into the wiki —
    leaving zero vectors in chromadb. The fix instantiates
    ``ChromadbBackend`` directly and threads it as
    ``explicit_backend`` so the write bypasses the per-KB cache.
    """
    from agent_knowledgebase.models import SourceType

    svc = _make_mocked_chromadb_service(tmp_path)
    kb = svc.create_kb(name="rt")

    # Step 1 — ingest a fixture source under chromadb. We use a
    # SourceType.file ingest with a fake content payload via a mocked
    # IngestionOrchestrator.
    from unittest.mock import MagicMock

    from agent_knowledgebase.models import Chunk

    fake_ingestion = MagicMock()
    fake_ingestion.ingest.side_effect = lambda *a, **kw: [
        Chunk(source_id="", kb_id="", content="alpha beta gamma", metadata={})
    ]
    svc._ingestion = fake_ingestion

    svc.ingest_source(
        kb_id=kb.id,
        source_type=SourceType.file,
        uri="/tmp/seed.txt",
        dedup_key="rt-seed",
    )

    # Sanity: vectors landed in chromadb.
    initial_count = svc._test_vector_stores[kb.id]
    assert len(initial_count) >= 1, (
        f"expected initial chromadb to have >=1 vector, got {len(initial_count)}"
    )

    # Step 2 — migrate to markdown.
    result_md = svc.migrate(kb_id=kb.id, target_backend="markdown")
    assert result_md["target_backend"] == "markdown"
    wiki_root = svc._config.kb_data_dir(svc._index[kb.id]) / "wiki"
    assert wiki_root.is_dir(), "wiki/ tree must be created by markdown migration"

    # Verify routing flipped.
    from agent_knowledgebase.backends.markdown_backend import MarkdownWikiBackend
    assert isinstance(svc._backend_for(kb.id), MarkdownWikiBackend), (
        "post-migration backend cache must return markdown backend"
    )

    # Step 3 — reverse migrate back to chromadb. THIS IS THE BUG TRIGGER.
    result_cdb = svc.migrate(kb_id=kb.id, target_backend="chromadb")
    assert result_cdb["target_backend"] == "chromadb"
    assert result_cdb["pages_migrated"] >= 1, (
        "reverse migration must report at least one page migrated"
    )

    # Step 4 — THE BUG CATCHER. After the reverse migration, the
    # in-memory chromadb vectorstore for this KB must contain
    # vectors. Pre-fix, the writes were silently routed back into the
    # wiki and the chromadb store stayed empty.
    final_count = len(svc._test_vector_stores[kb.id])
    assert final_count > 0, (
        f"REGRESSION: reverse migration left chromadb empty (count={final_count}). "
        f"The B-01 fix routes writes through an explicit ChromadbBackend instance "
        f"so the still-valid 'markdown' sentinel cannot misroute them."
    )

    # Step 5 — routing flipped back to chromadb (sentinel rewritten).
    from agent_knowledgebase.backends.chromadb_backend import ChromadbBackend
    assert isinstance(svc._backend_for(kb.id), ChromadbBackend), (
        "post-reverse-migration backend cache must return chromadb backend"
    )


# ---------------------------------------------------------------------------
# B-02 — migrate() acquires the per-kb_id lock
# ---------------------------------------------------------------------------


def test_migrate_acquires_per_kb_lock(chromadb_service):
    """B-02: ``migrate()`` must enter ``_get_kb_lock(kb_id)`` for the
    cutover so concurrent kb_query/kb_search don't see torn state.
    """
    from unittest.mock import MagicMock, patch

    kb_id = _seed_kb_with_chunks(chromadb_service, kb_name="lock-test")

    # Wrap the real lock with a MagicMock so we can assert __enter__
    # was called by migrate(). We use a real lock object underneath so
    # lock semantics are preserved if any code path actually contends.
    real_lock = chromadb_service._get_kb_lock(kb_id)
    enter_calls: list = []
    exit_calls: list = []

    class _SpyLock:
        def __init__(self, inner):
            self._inner = inner

        def __enter__(self):
            enter_calls.append(True)
            return self._inner.__enter__()

        def __exit__(self, *args):
            exit_calls.append(True)
            return self._inner.__exit__(*args)

    spy = _SpyLock(real_lock)
    with patch.object(
        chromadb_service,
        "_get_kb_lock",
        MagicMock(return_value=spy),
    ):
        chromadb_service.migrate(kb_id=kb_id, target_backend="markdown")

    assert enter_calls, "migrate() must acquire the per-kb_id lock"
    assert exit_calls, "migrate() must release the per-kb_id lock"


def test_migrate_emits_lock_telemetry(
    chromadb_service, capsys: pytest.CaptureFixture[str]
):
    """B-02: ``migrate()`` must emit lock_acquire / lock_acquired /
    lock_release stderr lines mirroring the ingest_source pattern.
    """
    import json as _json

    kb_id = _seed_kb_with_chunks(chromadb_service, kb_name="lock-telemetry")
    chromadb_service.migrate(kb_id=kb_id, target_backend="markdown")

    err = capsys.readouterr().err
    phases_seen: set[str] = set()
    for line in err.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = _json.loads(line)
        except _json.JSONDecodeError:
            continue
        if payload.get("op") == "kb_migrate" and payload.get("kb_id") == kb_id:
            phase = payload.get("phase")
            if phase in {"lock_acquire", "lock_acquired", "lock_release"}:
                phases_seen.add(phase)
    assert phases_seen == {"lock_acquire", "lock_acquired", "lock_release"}, (
        f"expected all 3 lock phases on op=kb_migrate stream, got {phases_seen}"
    )


# ---------------------------------------------------------------------------
# B-03 — sentinel atomic-write uses tempfile prefix (NOT Path.with_suffix)
# ---------------------------------------------------------------------------


def test_sentinel_tmp_file_uses_correct_prefix(
    chromadb_service, monkeypatch: pytest.MonkeyPatch
):
    """B-03: the sentinel's tmp file must be created with a name
    starting with ``.migrated_to.`` and landing inside ``kb_root``.

    Bug history: ``Path('/x/.migrated_to').with_suffix('.tmp')``
    returns ``Path('/x/.tmp')`` because pathlib treats a leading dot
    as a suffix. The tmp file landed in the wrong place AND collided
    across concurrent migrations on the same kb_root. The fix uses
    :func:`tempfile.mkstemp` keyed on the ``.migrated_to.`` prefix.
    """
    import os

    from agent_knowledgebase.backends import MIGRATED_SENTINEL_FILENAME

    kb_id = _seed_kb_with_chunks(chromadb_service, kb_name="tmp-prefix")

    # Capture the tmp_path passed to os.replace and assert its name
    # has the .migrated_to. prefix and lives inside kb_root (NOT
    # kb_root.parent — the with_suffix bug landed it in saves_dir).
    captured: dict = {}
    real_replace = os.replace

    def _spy_replace(src, dst):
        # Only capture the sentinel-rename, not unrelated atomic writes
        # done by markdown_backend during export.
        if str(dst).endswith(MIGRATED_SENTINEL_FILENAME):
            captured["src"] = str(src)
            captured["dst"] = str(dst)
        return real_replace(src, dst)

    monkeypatch.setattr("agent_knowledgebase.services.migration.os.replace", _spy_replace)
    chromadb_service.migrate(kb_id=kb_id, target_backend="markdown")

    assert "src" in captured, (
        "migration must call os.replace to atomically install the sentinel"
    )
    src_path = Path(captured["src"])
    dst_path = Path(captured["dst"])

    # Tmp filename starts with the sentinel name, ensuring
    # _read_migrated_sentinel-style scans don't pick the tmp by mistake
    # AND that a future bug regressing to with_suffix immediately fails.
    assert src_path.name.startswith(f"{MIGRATED_SENTINEL_FILENAME}."), (
        f"tmp file name {src_path.name!r} must start with "
        f"'{MIGRATED_SENTINEL_FILENAME}.' (the with_suffix bug produced "
        f"'.tmp' in the parent dir)"
    )
    # Tmp file is in the SAME directory as the final sentinel — required
    # for os.replace atomicity (cross-filesystem rename is not atomic).
    assert src_path.parent == dst_path.parent, (
        f"tmp file must live in kb_root next to sentinel; got "
        f"src={src_path}, dst={dst_path}"
    )


def test_sentinel_atomic_write_does_not_use_with_suffix() -> None:
    """B-03 source-level guard: the migration module must not use the
    buggy ``Path.with_suffix('.tmp')`` pattern for the sentinel write.
    """
    import inspect

    from agent_knowledgebase.services import migration

    src = inspect.getsource(migration._write_migrated_sentinel)
    assert ".with_suffix(\".tmp\")" not in src and ".with_suffix('.tmp')" not in src, (
        "_write_migrated_sentinel must not use Path.with_suffix('.tmp') — "
        "the dot-prefixed sentinel name makes that pattern strip the name "
        "rather than append a suffix"
    )


# ---------------------------------------------------------------------------
# I-05 — env var no longer mutated by export_to_markdown
# ---------------------------------------------------------------------------


def test_export_to_markdown_does_not_mutate_force_env(chromadb_service) -> None:
    """I-05: ``export_to_markdown`` must not touch
    ``AGENT_KB_FORCE_WIKI_INGEST`` — it now uses a private
    ``_index_with_force`` kwarg-equivalent instead of mutating
    process-wide environment state.
    """
    import os

    kb_id = _seed_kb_with_chunks(chromadb_service, kb_name="env-mutation")
    dest = chromadb_service._config.saves_dir / chromadb_service._index[kb_id]

    # Pre-state: env var unset.
    pre = os.environ.get("AGENT_KB_FORCE_WIKI_INGEST")
    assert pre is None, "test pre-condition: env var should be unset"

    export_to_markdown(chromadb_service, kb_id, dest)

    # Post-state: env var STILL unset (no mutation, no leak).
    post = os.environ.get("AGENT_KB_FORCE_WIKI_INGEST")
    assert post is None, (
        f"export_to_markdown must not mutate AGENT_KB_FORCE_WIKI_INGEST; "
        f"got post-state {post!r}"
    )


def test_export_to_markdown_preserves_pre_existing_env_value(
    chromadb_service, monkeypatch: pytest.MonkeyPatch
) -> None:
    """I-05: when the env var is already set by the operator,
    ``export_to_markdown`` must leave it exactly as it found it (no
    accidental ``del`` on the set-then-restore path).
    """
    import os

    kb_id = _seed_kb_with_chunks(chromadb_service, kb_name="env-preserve")
    dest = chromadb_service._config.saves_dir / chromadb_service._index[kb_id]

    monkeypatch.setenv("AGENT_KB_FORCE_WIKI_INGEST", "operator-value")
    export_to_markdown(chromadb_service, kb_id, dest)
    assert os.environ.get("AGENT_KB_FORCE_WIKI_INGEST") == "operator-value"


def test_export_to_markdown_restores_env_on_inner_exception(
    chromadb_service, monkeypatch: pytest.MonkeyPatch
) -> None:
    """I-05 (defense in depth): even if the index call raises, the
    env var must remain unmutated. Easiest way to guarantee this:
    don't touch the env var at all (the new code uses the private
    ``_index_with_force`` entry instead).
    """
    import os

    from agent_knowledgebase.backends.markdown_backend import MarkdownWikiBackend

    kb_id = _seed_kb_with_chunks(chromadb_service, kb_name="env-on-exc")
    dest = chromadb_service._config.saves_dir / chromadb_service._index[kb_id]

    pre = os.environ.get("AGENT_KB_FORCE_WIKI_INGEST")

    def _boom(self, **kwargs):
        raise RuntimeError("simulated index failure")

    monkeypatch.setattr(MarkdownWikiBackend, "_index_with_force", _boom)
    with pytest.raises(RuntimeError, match="simulated index failure"):
        export_to_markdown(chromadb_service, kb_id, dest)

    post = os.environ.get("AGENT_KB_FORCE_WIKI_INGEST")
    assert post == pre, (
        f"env var must round-trip across an exception in the inner index "
        f"call; pre={pre!r}, post={post!r}"
    )


# ---------------------------------------------------------------------------
# I-01 — sqlite-side aggregation in get_embedding_snapshot
# ---------------------------------------------------------------------------


def test_get_embedding_snapshot_uses_sqlite_aggregation(tmp_path: Path) -> None:
    """I-01: ``get_embedding_snapshot`` must aggregate sqlite-side
    (single GROUP BY pass) rather than scanning every row + decoding
    json in Python. We assert two properties:

    1. The implementation does NOT iterate metadata via Python
       json.loads in a hot loop (source-level guard).
    2. A 1000-chunk read completes in <100ms (perf assertion).
    """
    import inspect
    import time as _time

    from agent_knowledgebase.database import Database
    from agent_knowledgebase.models import Chunk

    # 1. Source-level guard: the implementation should not reference
    #    ``_json_loads`` or run a Python-side group loop. This catches
    #    a regression to the old per-row scan.
    src = inspect.getsource(Database.get_embedding_snapshot)
    assert "_json_loads" not in src, (
        "get_embedding_snapshot must not call _json_loads — sqlite-side "
        "json_extract should handle the metadata blob server-side"
    )
    assert "GROUP BY" in src.upper(), (
        "get_embedding_snapshot must use a GROUP BY query for sqlite-side "
        "aggregation"
    )

    # 2. Perf: 1000 chunks complete in <100ms.
    db = Database(db_path=tmp_path / "perf.db")
    db._conn.execute(
        "INSERT INTO knowledgebases (id, name, created_at, updated_at) "
        "VALUES (?, ?, ?, ?)",
        ("kb-perf", "kb-perf", "2026-04-24T00:00:00", "2026-04-24T00:00:00"),
    )
    db._conn.execute(
        "INSERT INTO sources (id, kb_id, source_type, uri, status) "
        "VALUES (?, ?, ?, ?, ?)",
        ("src-perf", "kb-perf", "file", "x://a", "ingested"),
    )
    db._conn.commit()
    for i in range(1000):
        db.insert_chunk(
            Chunk(
                id=f"chunk-{i}",
                kb_id="kb-perf",
                source_id="src-perf",
                content=f"content-{i}",
                metadata={
                    "embedding_model": "qwen3-embedding:8b",
                    "embedding_provider": "ollama",
                    "embed_base_url": "http://127.0.0.1:11434",
                },
            )
        )

    # Warm the page cache so the first call's IO doesn't dominate.
    db.get_embedding_snapshot("kb-perf")

    t0 = _time.monotonic()
    for _ in range(10):
        snap = db.get_embedding_snapshot("kb-perf")
    elapsed_ms_avg = ((_time.monotonic() - t0) / 10) * 1000.0

    assert snap == (
        "qwen3-embedding:8b",
        "ollama",
        "http://127.0.0.1:11434",
    )
    assert elapsed_ms_avg < 100, (
        f"snapshot read avg {elapsed_ms_avg:.1f}ms over 1000 chunks exceeds "
        f"100ms budget — sqlite-side aggregation regression"
    )
    db.close()


def test_get_embedding_snapshot_output_unchanged_for_mixed_corpus(
    tmp_path: Path,
) -> None:
    """I-01: the new sqlite-side aggregation must return the same
    dominant tuple as the legacy Python-side aggregation for a mixed
    corpus (multiple models, multiple providers).
    """
    from agent_knowledgebase.database import Database
    from agent_knowledgebase.models import Chunk

    db = Database(db_path=tmp_path / "mixed.db")
    db._conn.execute(
        "INSERT INTO knowledgebases (id, name, created_at, updated_at) "
        "VALUES (?, ?, ?, ?)",
        ("kb-mix", "kb-mix", "2026-04-24T00:00:00", "2026-04-24T00:00:00"),
    )
    db._conn.execute(
        "INSERT INTO sources (id, kb_id, source_type, uri, status) "
        "VALUES (?, ?, ?, ?, ?)",
        ("src-mix", "kb-mix", "file", "x://a", "ingested"),
    )
    db._conn.commit()
    # 5 ollama chunks
    for i in range(5):
        db.insert_chunk(
            Chunk(
                id=f"c-ol-{i}",
                kb_id="kb-mix",
                source_id="src-mix",
                content="x",
                metadata={
                    "embedding_model": "qwen3-embedding:8b",
                    "embedding_provider": "ollama",
                    "embed_base_url": "http://127.0.0.1:11434",
                },
            )
        )
    # 3 remote chunks
    for i in range(3):
        db.insert_chunk(
            Chunk(
                id=f"c-rm-{i}",
                kb_id="kb-mix",
                source_id="src-mix",
                content="x",
                metadata={
                    "embedding_model": "text-embedding-3-small",
                    "embedding_provider": "remote",
                    "embed_base_url": "https://api.openai.com/v1",
                },
            )
        )
    snap = db.get_embedding_snapshot("kb-mix")
    # Dominant tuple is the ollama one (5 > 3).
    assert snap == (
        "qwen3-embedding:8b",
        "ollama",
        "http://127.0.0.1:11434",
    )
    db.close()
