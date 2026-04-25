"""Phase 5 (B-02) — auto-backfill regression for legacy v0.6.0 Ollama KBs.

A v0.6.0 KB stamps only ``embedding_model`` in chunk metadata; the
Phase 4 ``embedding_provider`` / ``embed_base_url`` columns stay NULL
until ``backfill_provider_snapshot`` runs. After the Phase 5 default
flip to ``provider='remote'``, a kb_query against such a KB would
otherwise resolve the snapshot to ``(model, None, None)`` and try to
build a remote embedder using the post-flip global default — which
raises ``ValueError("AGENT_KB_EMBED_API_KEY required for remote
embeddings")``.

This test pins the auto-backfill heuristic that catches the historical
``qwen3-embedding`` model family and stamps Ollama defaults
retroactively, so the snapshot rebuild picks up the original embedder
without operator intervention. spec_id:
70ab2170-381a-4657-bcd1-28a40c6f369b
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_knowledgebase.config import Settings
from agent_knowledgebase.services.embeddings import (
    DEFAULT_OLLAMA_BASE_URL,
    OllamaEmbedder,
)
from agent_knowledgebase.services.knowledgebase import KnowledgebaseService


def test_legacy_v0_6_0_kb_with_null_provider_auto_backfills_to_ollama(
    tmp_path: Path,
) -> None:
    """B-02: a v0.6.0 KB with NULL provider snapshot must auto-backfill
    to ``provider='ollama'`` on first ``_query_embedder_for`` call.

    Setup:
      - Settings flipped to remote/text-embedding-3-small with NO API key.
      - Insert a chunk with ``metadata={'embedding_model':
        'qwen3-embedding:8b'}`` directly (provider/base_url columns NULL).

    Expected:
      - ``_query_embedder_for(kb_id)`` returns an OllamaEmbedder (NOT a
        ValueError stack trace).
      - The chunk's snapshot now reads
        ``('qwen3-embedding:8b', 'ollama', '<DEFAULT_OLLAMA_BASE_URL>')``.
    """
    saves = tmp_path / "saves"
    saves.mkdir()

    # Phase-5-default-flipped settings: remote provider, no API key.
    settings = Settings(
        saves_dir=saves,
        embedding_provider="remote",
        embedding_model="text-embedding-3-small",
        embed_base_url="https://api.openai.com/v1",
        embed_api_key=None,
    ).resolve_paths()
    svc = KnowledgebaseService(settings)
    kb = svc.create_kb(name="legacy-ollama-kb")

    # Insert a chunk that mirrors a v0.6.0 row: model in metadata,
    # NULL provider columns. Bypass insert_chunk's column-stamping
    # path by writing the row via raw SQL so the columns truly stay
    # NULL (insert_chunk would also stamp them from metadata, which
    # the test would then succeed for the wrong reason).
    ctx = svc._ctx(kb.id)
    ctx.db._conn.execute(
        "INSERT INTO sources (id, kb_id, source_type, uri, status) "
        "VALUES (?, ?, ?, ?, ?)",
        (f"src-{kb.id}", kb.id, "file", "x://legacy", "ingested"),
    )
    ctx.db._conn.execute(
        "INSERT INTO chunks (id, source_id, kb_id, content, metadata, "
        "embedding_id, embedding_provider, embed_base_url, embedder_version) "
        "VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL)",
        (
            "chunk-legacy",
            f"src-{kb.id}",
            kb.id,
            "legacy chunk content",
            json.dumps({"embedding_model": "qwen3-embedding:8b"}),
            None,
        ),
    )
    ctx.db._conn.commit()

    # Sanity: the snapshot before the auto-backfill is the legacy NULL shape.
    pre_snap = ctx.db.get_embedding_snapshot(kb.id)
    assert pre_snap == ("qwen3-embedding:8b", None, None), (
        f"Pre-auto-backfill snapshot must be the legacy NULL shape; "
        f"got {pre_snap!r}"
    )

    # Act — first query call. Must NOT raise ValueError.
    embedder = svc._query_embedder_for(kb.id)
    assert isinstance(embedder, OllamaEmbedder), (
        f"Auto-backfill must rebuild as OllamaEmbedder; got "
        f"{type(embedder).__name__}"
    )
    assert embedder.model_name == "qwen3-embedding:8b"

    # The snapshot now reads the populated triple.
    post_snap = ctx.db.get_embedding_snapshot(kb.id)
    assert post_snap == (
        "qwen3-embedding:8b",
        "ollama",
        DEFAULT_OLLAMA_BASE_URL,
    ), f"Post-auto-backfill snapshot must reflect the stamped tuple; got {post_snap!r}"


def test_auto_backfill_is_idempotent_per_process(tmp_path: Path) -> None:
    """B-02: a second ``_query_embedder_for`` call MUST NOT re-run
    the SQL UPDATE — the in-process suppression set short-circuits."""
    saves = tmp_path / "saves"
    saves.mkdir()
    settings = Settings(
        saves_dir=saves,
        embedding_provider="remote",
        embedding_model="text-embedding-3-small",
        embed_base_url="https://api.openai.com/v1",
        embed_api_key=None,
    ).resolve_paths()
    svc = KnowledgebaseService(settings)
    kb = svc.create_kb(name="legacy-idempotent-kb")

    ctx = svc._ctx(kb.id)
    ctx.db._conn.execute(
        "INSERT INTO sources (id, kb_id, source_type, uri, status) "
        "VALUES (?, ?, ?, ?, ?)",
        (f"src-{kb.id}", kb.id, "file", "x://legacy", "ingested"),
    )
    ctx.db._conn.execute(
        "INSERT INTO chunks (id, source_id, kb_id, content, metadata, "
        "embedding_id, embedding_provider, embed_base_url, embedder_version) "
        "VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL)",
        (
            "chunk-legacy-idem",
            f"src-{kb.id}",
            kb.id,
            "x",
            json.dumps({"embedding_model": "qwen3-embedding:8b"}),
            None,
        ),
    )
    ctx.db._conn.commit()

    # First call performs the backfill.
    svc._query_embedder_for(kb.id)
    assert kb.id in svc._auto_backfilled_kbs

    # Patch backfill_embedding_snapshot to prove it isn't called again.
    calls: list[str] = []
    original = ctx.db.backfill_embedding_snapshot

    def _spy(**kwargs):  # noqa: ANN001 - test spy
        calls.append("called")
        return original(**kwargs)

    ctx.db.backfill_embedding_snapshot = _spy  # type: ignore[method-assign]
    try:
        svc._query_embedder_for(kb.id)
    finally:
        ctx.db.backfill_embedding_snapshot = original  # type: ignore[method-assign]

    assert calls == [], (
        "Second _query_embedder_for must NOT re-invoke the backfill SQL; "
        "the per-process suppression set should short-circuit."
    )


def test_auto_backfill_skips_non_legacy_models(tmp_path: Path) -> None:
    """B-02: the heuristic only fires for ``qwen3-embedding*`` model names.

    A KB whose NULL-snapshot model is something else (e.g. a custom
    Ollama model name an operator chose explicitly) must NOT be
    auto-backfilled — that would silently misstamp it as Ollama when
    it might in fact have been ingested via the now-deprecated
    sentence-transformers default.
    """
    saves = tmp_path / "saves"
    saves.mkdir()
    settings = Settings(
        saves_dir=saves,
        embedding_provider="remote",
        embedding_model="text-embedding-3-small",
        embed_base_url="https://api.openai.com/v1",
        embed_api_key="sk-test",  # set so the build doesn't reject
    ).resolve_paths()
    svc = KnowledgebaseService(settings)
    kb = svc.create_kb(name="legacy-other-model-kb")

    ctx = svc._ctx(kb.id)
    ctx.db._conn.execute(
        "INSERT INTO sources (id, kb_id, source_type, uri, status) "
        "VALUES (?, ?, ?, ?, ?)",
        (f"src-{kb.id}", kb.id, "file", "x://legacy", "ingested"),
    )
    ctx.db._conn.execute(
        "INSERT INTO chunks (id, source_id, kb_id, content, metadata, "
        "embedding_id, embedding_provider, embed_base_url, embedder_version) "
        "VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL)",
        (
            "chunk-other",
            f"src-{kb.id}",
            kb.id,
            "x",
            json.dumps({"embedding_model": "all-MiniLM-L6-v2"}),
            None,
        ),
    )
    ctx.db._conn.commit()

    # Call the resolver. The heuristic should NOT fire (model name
    # doesn't start with qwen3-embedding), so no auto-backfill row
    # should be added to the suppression set.
    try:
        svc._query_embedder_for(kb.id)
    except ValueError:
        # ValueError is acceptable here for the right reasons (the KB
        # snapshot stays NULL-NULL and the build path can't make
        # decisions). What we're pinning is the negation: the
        # heuristic must not have triggered.
        pass
    assert kb.id not in svc._auto_backfilled_kbs
    # Snapshot stays in the legacy shape — heuristic did not stamp it.
    snap = ctx.db.get_embedding_snapshot(kb.id)
    assert snap == ("all-MiniLM-L6-v2", None, None), (
        f"non-legacy-model KB snapshot must stay NULL after the "
        f"heuristic skips it; got {snap!r}"
    )
