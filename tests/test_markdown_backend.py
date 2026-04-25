"""Phase 3 tests for :class:`MarkdownWikiBackend`.

spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b (v2.1)
Source: pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json
        implementation.phases[3] (Phase 3 — MarkdownWikiBackend)

Test surface (per spec.phases[3].tasks[7]):

* Single-article ingest: ``index.md`` updated; ``search()`` returns the
  article.
* 250-article ingest: ``index.md`` becomes the directory pointer and
  per-shard files appear under ``wiki/index/<source_type>.md``.
* ``source_type=sql_database`` ingest raises
  :class:`KbIngestorUnsuitableError` with code ``KB_INGESTOR_UNSUITABLE``.
* ``AGENT_KB_FORCE_WIKI_INGEST=1`` bypasses the check AND emits a
  structured stderr_log warning carrying ``error_code``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_knowledgebase.backends.markdown_backend import (
    KbIngestorUnsuitableError,
    MarkdownWikiBackend,
)
from agent_knowledgebase.config import Settings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def markdown_backend(tmp_path: Path) -> MarkdownWikiBackend:
    """Service-less markdown backend rooted at ``tmp_path/saves``.

    Service-less means we exercise the ``saves_dir / kb_id`` fallback
    path on :meth:`_kb_root` — the unit tests below don't need the
    full :class:`KnowledgebaseService` lifecycle.
    """
    saves = tmp_path / "saves"
    saves.mkdir()
    settings = Settings(saves_dir=saves, kb_backend="markdown").resolve_paths()
    return MarkdownWikiBackend(settings)


def _make_doc(
    *,
    source_type: str,
    title: str,
    content: str,
    uri: str | None = None,
    doc_id: str | None = None,
    dedup_key: str | None = None,
) -> dict[str, object]:
    """Tiny helper for building documents in the shape ``index()`` expects."""
    return {
        "id": doc_id or f"src-{title.replace(' ', '-')}",
        "content": content,
        "metadata": {
            "source_type": source_type,
            "title": title,
            "uri": uri or f"https://example.com/{title}",
            "dedup_key": dedup_key,
        },
    }


# ---------------------------------------------------------------------------
# Single-article ingest -> index.md updated, search() returns it
# ---------------------------------------------------------------------------


def test_ingest_one_article_writes_layout_and_search_returns_it(
    markdown_backend: MarkdownWikiBackend,
) -> None:
    """End-to-end: ingest 1 article, assert the layout exists and the
    page is recoverable via :meth:`search`.
    """
    kb_id = "kb-single"
    doc = _make_doc(
        source_type="file",
        title="Hello World",
        content="The quick brown fox jumps over the lazy dog.",
        uri="/tmp/hello.txt",
        dedup_key="hello-key",
    )

    markdown_backend.index(kb_id=kb_id, documents=[doc])

    wiki_root = markdown_backend._wiki_root(kb_id)
    assert (wiki_root / "raw").is_dir()
    assert (wiki_root / "pages").is_dir()
    assert (wiki_root / "index.md").is_file()
    assert (wiki_root / "log.md").is_file()

    pages = list((wiki_root / "pages").glob("*.md"))
    assert len(pages) == 1, f"expected 1 page, got {len(pages)}: {pages}"

    # index.md catalogs the page (flat form for small KB).
    index_text = (wiki_root / "index.md").read_text(encoding="utf-8")
    assert "# Wiki Index" in index_text
    assert "Hello World" in index_text
    assert "`file`" in index_text  # source_type tag

    # log.md captures the ingest event.
    log_text = (wiki_root / "log.md").read_text(encoding="utf-8")
    assert "ingest" in log_text
    assert "Hello World" in log_text

    # search() returns it.
    results = markdown_backend.search(kb_id=kb_id, text="quick brown fox", top_k=5)
    assert results, "expected search() to return at least one result"
    top = results[0]
    assert "quick brown fox" in top["content"]
    assert top["source_type"] == "file"
    assert 0.0 < top["score"] <= 1.0
    assert top["metadata"]["backend"] == "markdown"

    # query() routes through search() (no embedding path) and yields
    # the same shape.
    qresults = markdown_backend.query(kb_id=kb_id, text="quick brown fox", top_k=5)
    assert len(qresults) == len(results)


# ---------------------------------------------------------------------------
# 250-article ingest -> index.md becomes a directory pointer; shards exist
# ---------------------------------------------------------------------------


def test_ingest_250_articles_shards_index(
    markdown_backend: MarkdownWikiBackend,
) -> None:
    """Above ``_SHARD_THRESHOLD`` (200) the index migrates to per-category
    shards under ``wiki/index/<source_type>.md`` and the top-level
    ``index.md`` becomes a pointer.
    """
    kb_id = "kb-shard"
    docs = [
        _make_doc(
            source_type="file" if i % 2 == 0 else "website",
            title=f"Article {i:03d}",
            content=f"Body of article {i}, mentioning topic alpha.",
            uri=f"/tmp/article-{i:03d}.txt",
            doc_id=f"src-{i:03d}",
        )
        for i in range(250)
    ]
    markdown_backend.index(kb_id=kb_id, documents=docs)

    wiki_root = markdown_backend._wiki_root(kb_id)
    pages = list((wiki_root / "pages").glob("*.md"))
    assert len(pages) == 250

    index_dir = wiki_root / "index"
    assert index_dir.is_dir(), "expected wiki/index/ directory after sharding"

    # We seeded 2 source_types alternately, so we expect 2 shard files.
    file_shard = index_dir / "file.md"
    website_shard = index_dir / "website.md"
    assert file_shard.is_file()
    assert website_shard.is_file()

    # Top-level index.md is the pointer form.
    top = (wiki_root / "index.md").read_text(encoding="utf-8")
    assert "Wiki Index (sharded)" in top
    assert "Categories" in top
    assert "(125 pages)" in top  # 250 / 2

    # Each shard lists its bucket count and points back to pages/.
    file_text = file_shard.read_text(encoding="utf-8")
    assert "Pages: 125" in file_text
    assert "../pages/" in file_text


# ---------------------------------------------------------------------------
# Allow-list rejection: source_type=sql_database -> KB_INGESTOR_UNSUITABLE
# ---------------------------------------------------------------------------


def test_ingest_sql_database_raises_unsuitable(
    markdown_backend: MarkdownWikiBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the force flag, ingesting a disallowed source_type
    raises :class:`KbIngestorUnsuitableError` with the SCREAMING_SNAKE
    code ``KB_INGESTOR_UNSUITABLE``.
    """
    monkeypatch.delenv("AGENT_KB_FORCE_WIKI_INGEST", raising=False)
    doc = _make_doc(
        source_type="sql_database",
        title="rows from cache",
        content="SELECT * FROM cache",
        uri="sqlite:///cache.db",
    )
    with pytest.raises(KbIngestorUnsuitableError) as excinfo:
        markdown_backend.index(kb_id="kb-bad", documents=[doc])
    assert excinfo.value.error_code == "KB_INGESTOR_UNSUITABLE"
    payload = excinfo.value.to_payload()
    assert payload["error_code"] == "KB_INGESTOR_UNSUITABLE"
    assert payload["source_type"] == "sql_database"
    assert "remediation" in payload
    assert payload["spec_id"] == "70ab2170-381a-4657-bcd1-28a40c6f369b"

    # Confirm no partial KB was written to disk.
    wiki_root = markdown_backend._wiki_root("kb-bad")
    assert not wiki_root.exists() or not list(
        (wiki_root / "pages").glob("*.md") if (wiki_root / "pages").is_dir() else []
    )


def test_each_disallowed_source_type_raises(
    markdown_backend: MarkdownWikiBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All of the spec's "anything else" source_types raise UNSUITABLE.

    Mirrors spec.phases[3].tasks[5]'s "anything else (sql_database,
    codebase, git_history, directory, large api_endpoint)" enumeration.
    """
    monkeypatch.delenv("AGENT_KB_FORCE_WIKI_INGEST", raising=False)
    for st in ("sql_database", "codebase", "git_history", "directory"):
        doc = _make_doc(
            source_type=st,
            title=f"x-{st}",
            content="content",
            uri=f"x://{st}",
        )
        with pytest.raises(KbIngestorUnsuitableError) as excinfo:
            markdown_backend.index(kb_id=f"kb-{st}", documents=[doc])
        assert excinfo.value.source_type == st


def test_api_endpoint_above_2mb_raises_unsuitable(
    markdown_backend: MarkdownWikiBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per spec, ``api_endpoint`` payloads >= 2MB are also blocked."""
    monkeypatch.delenv("AGENT_KB_FORCE_WIKI_INGEST", raising=False)
    big = "a" * (2 * 1024 * 1024 + 1)
    doc = _make_doc(
        source_type="api_endpoint",
        title="huge response",
        content=big,
        uri="https://api.example.com/big",
    )
    with pytest.raises(KbIngestorUnsuitableError) as excinfo:
        markdown_backend.index(kb_id="kb-big-api", documents=[doc])
    assert "2MB" in excinfo.value.reason or "byte wiki limit" in excinfo.value.reason


def test_api_endpoint_under_2mb_is_allowed(
    markdown_backend: MarkdownWikiBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The positive-allow form: small api_endpoint payloads ingest fine."""
    monkeypatch.delenv("AGENT_KB_FORCE_WIKI_INGEST", raising=False)
    doc = _make_doc(
        source_type="api_endpoint",
        title="small response",
        content='{"ok": true}',
        uri="https://api.example.com/small",
    )
    markdown_backend.index(kb_id="kb-api", documents=[doc])
    assert markdown_backend.count(kb_id="kb-api") == 1


# ---------------------------------------------------------------------------
# Force-flag bypass: stderr_log warning emitted, ingest proceeds
# ---------------------------------------------------------------------------


def test_force_flag_bypasses_check_and_emits_stderr_warning(
    markdown_backend: MarkdownWikiBackend,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``AGENT_KB_FORCE_WIKI_INGEST=1`` proceeds with a stderr_log
    warning (per spec.phases[3].tasks[5] and validation finding f-15).

    The warning carries ``error_code=KB_INGESTOR_UNSUITABLE_FORCED`` so
    operators can grep for forced bypasses.
    """
    monkeypatch.setenv("AGENT_KB_FORCE_WIKI_INGEST", "1")
    doc = _make_doc(
        source_type="codebase",
        title="forced codebase",
        content="def hello(): pass",
        uri="repo://local/file.py",
    )
    markdown_backend.index(kb_id="kb-force", documents=[doc])

    # Page WAS written (force flag worked).
    assert markdown_backend.count(kb_id="kb-force") == 1

    # And a structured warning landed on stderr.
    captured = capsys.readouterr()
    assert captured.err, "expected a stderr_log line for the forced ingest"
    # Find the JSON payload — there can be multiple lines (one per emit).
    found_force_warning = False
    for line in captured.err.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("error_code") == "KB_INGESTOR_UNSUITABLE_FORCED":
            found_force_warning = True
            # 11-field schema must remain intact.
            for key in (
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
            ):
                assert key in payload, f"force-warning missing field {key!r}"
            assert payload["kb_id"] == "kb-force"
            assert payload["op"] == "markdown_backend_force_ingest"
            break
    assert found_force_warning, (
        "expected a stderr_log line with error_code="
        "KB_INGESTOR_UNSUITABLE_FORCED but none found"
    )


# ---------------------------------------------------------------------------
# Empty-input / no-op behaviour
# ---------------------------------------------------------------------------


def test_index_empty_documents_is_noop(
    markdown_backend: MarkdownWikiBackend,
) -> None:
    """Mirrors :class:`ChromadbBackend.index` — empty input is a no-op."""
    markdown_backend.index(kb_id="kb-empty", documents=[])
    # No filesystem layout created.
    assert not markdown_backend._wiki_root("kb-empty").exists()


def test_search_on_empty_kb_returns_empty(
    markdown_backend: MarkdownWikiBackend,
) -> None:
    """Search before any ingest returns ``[]`` instead of raising."""
    out = markdown_backend.search(kb_id="kb-empty", text="anything", top_k=5)
    assert out == []


# ---------------------------------------------------------------------------
# Delete by source_id and by ids
# ---------------------------------------------------------------------------


def test_delete_by_source_id_removes_page(
    markdown_backend: MarkdownWikiBackend,
) -> None:
    """``delete(source_id=...)`` removes the page + raw dump."""
    kb_id = "kb-del-src"
    doc = _make_doc(
        source_type="file", title="Doomed", content="will be deleted",
        uri="/tmp/doomed.txt", doc_id="src-doomed",
    )
    markdown_backend.index(kb_id=kb_id, documents=[doc])
    assert markdown_backend.count(kb_id=kb_id) == 1

    markdown_backend.delete(kb_id=kb_id, source_id="src-doomed")
    assert markdown_backend.count(kb_id=kb_id) == 0


def test_delete_requires_at_least_one_target(
    markdown_backend: MarkdownWikiBackend,
) -> None:
    """``delete()`` without ids/source_id raises ``ValueError`` (mirrors
    ChromadbBackend.delete contract)."""
    with pytest.raises(ValueError, match="at least one"):
        markdown_backend.delete(kb_id="kb-noop")


# ---------------------------------------------------------------------------
# health_check + spec_version
# ---------------------------------------------------------------------------


def test_health_check_reports_markdown(
    markdown_backend: MarkdownWikiBackend,
) -> None:
    """``health_check()`` returns the canonical backend tag."""
    health = markdown_backend.health_check()
    assert health["backend"] == "markdown"
    assert health["status"] in {"ok", "degraded"}
    assert health["spec_version"] == "2.1"
    assert "fts5_available" in health


# ---------------------------------------------------------------------------
# Service-bound construction also routes through saves_dir / kb_id
# ---------------------------------------------------------------------------


def test_get_backend_factory_returns_markdown_when_settings_say_markdown(
    tmp_path: Path,
) -> None:
    """The factory wires the markdown backend when opted in.

    Phase 3 promoted the ``'markdown'`` branch of ``get_backend()`` from
    ``NotImplementedError`` to a real instance — confirm here.
    """
    from agent_knowledgebase.backends import get_backend

    saves = tmp_path / "saves"
    saves.mkdir()
    settings = Settings(saves_dir=saves, kb_backend="markdown").resolve_paths()
    backend = get_backend(settings)
    assert isinstance(backend, MarkdownWikiBackend)


# ---------------------------------------------------------------------------
# Phase 3 review I-2: stale shard dir cleanup when delete drops below threshold
# ---------------------------------------------------------------------------


def test_delete_below_shard_threshold_removes_stale_shard_dir(
    markdown_backend: MarkdownWikiBackend,
) -> None:
    """Going sharded then back below ``_SHARD_THRESHOLD`` (200) must
    drop the ``wiki/index/`` shard directory and restore the flat
    top-level ``index.md`` form.

    Regression for code-quality review I-2: previously
    ``_write_flat_index`` had a comment about cleanup but no code,
    so stale ``wiki/index/<category>.md`` files persisted after a
    delete that crossed back below the threshold.
    """
    kb_id = "kb-shrink"
    # 1. Ingest 250 docs -> sharded form (>200 threshold).
    docs = [
        _make_doc(
            source_type="file" if i % 2 == 0 else "website",
            title=f"Article {i:03d}",
            content=f"Body of article {i}, mentioning topic alpha.",
            uri=f"/tmp/article-{i:03d}.txt",
            doc_id=f"src-{i:03d}",
        )
        for i in range(250)
    ]
    markdown_backend.index(kb_id=kb_id, documents=docs)

    wiki_root = markdown_backend._wiki_root(kb_id)
    index_dir = wiki_root / "index"
    # Pre-condition: sharded form is on disk.
    assert index_dir.is_dir(), "expected sharded form before delete"
    assert (index_dir / "file.md").is_file()
    assert (index_dir / "website.md").is_file()
    top_before = (wiki_root / "index.md").read_text(encoding="utf-8")
    assert "Wiki Index (sharded)" in top_before

    # 2. Delete enough pages to drop back below 200 (delete 100 -> 150
    # remaining, well under threshold).
    for i in range(100):
        markdown_backend.delete(kb_id=kb_id, source_id=f"src-{i:03d}")

    # 3. Post-condition: flat form, no shard dir.
    assert markdown_backend.count(kb_id=kb_id) == 150
    assert not index_dir.exists(), (
        "expected wiki/index/ shard directory to be removed when "
        "page count crossed back below the shard threshold"
    )
    top_after = (wiki_root / "index.md").read_text(encoding="utf-8")
    assert "# Wiki Index" in top_after
    assert "Wiki Index (sharded)" not in top_after, (
        "top-level index.md should be the flat form, not the pointer form"
    )
    assert "_Pages: 150" in top_after

    # info()['sharded_index'] should also reflect the cleanup.
    info = markdown_backend.info(kb_id=kb_id)
    assert info["sharded_index"] is False


# ---------------------------------------------------------------------------
# Phase 3 review I-3: kb_id / source_id path-traversal sanitization
# ---------------------------------------------------------------------------


def test_kb_id_with_traversal_components_stays_inside_saves_dir(
    tmp_path: Path,
) -> None:
    """A hostile ``kb_id`` (``"../escape"``) must NOT escape ``saves_dir``.

    Regression for code-quality review I-3: the service-less
    ``_kb_root`` fallback previously composed ``saves_dir / kb_id``
    verbatim, which let ``kb_id="../escape"`` resolve to
    ``<saves_dir>/../escape`` — outside the saves boundary.
    """
    saves = tmp_path / "saves"
    saves.mkdir()
    settings = Settings(saves_dir=saves, kb_backend="markdown").resolve_paths()
    backend = MarkdownWikiBackend(settings, service=None)

    doc = _make_doc(
        source_type="file",
        title="hello",
        content="hello world",
        uri="/tmp/hello.txt",
        doc_id="src-hello",
    )
    # The hostile kb_id must NOT raise, but the resulting path MUST
    # be inside saves_dir (the sanitizer maps "../escape" -> "escape").
    backend.index(kb_id="../escape", documents=[doc])
    saves_resolved = saves.resolve()
    kb_root = backend._kb_root("../escape").resolve()
    assert str(kb_root).startswith(str(saves_resolved)), (
        f"kb_root {kb_root!r} escaped saves_dir {saves_resolved!r}"
    )

    # And the would-be escape path on disk must NOT exist.
    escape_target = (saves.parent / "escape").resolve()
    assert not escape_target.exists(), (
        f"path-traversal target {escape_target!r} was created on disk; "
        "kb_id sanitization failed"
    )


def test_source_id_with_traversal_components_stays_inside_raw_dir(
    markdown_backend: MarkdownWikiBackend,
) -> None:
    """A hostile ``doc.id`` (``"../escape"``) must NOT escape ``raw/``.

    Regression for code-quality review I-3: ``_write_one`` previously
    composed ``raw_dir / f"{source_id}.{ext}"`` verbatim, so a doc
    with ``id="../escape"`` would write to
    ``<raw_dir>/../escape.txt`` — i.e. one level up, into ``wiki/``.
    """
    kb_id = "kb-traversal-sid"
    doc = _make_doc(
        source_type="file",
        title="hostile",
        content="should land safely under raw/",
        uri="/tmp/hostile.txt",
        doc_id="../escape",
    )
    markdown_backend.index(kb_id=kb_id, documents=[doc])

    raw_dir = markdown_backend._raw_dir(kb_id).resolve()
    raw_files = list(raw_dir.glob("*"))
    assert raw_files, "expected at least one raw file under raw/"
    for raw_file in raw_files:
        assert str(raw_file.resolve()).startswith(str(raw_dir)), (
            f"raw file {raw_file!r} escaped raw_dir {raw_dir!r}"
        )

    # The would-be escape target ("../escape.txt" relative to raw/)
    # would land in the wiki/ root — assert that path was NOT created.
    wiki_root = markdown_backend._wiki_root(kb_id)
    assert not (wiki_root / "escape.txt").exists(), (
        "source_id sanitization let a raw file escape into wiki/"
    )


# ---------------------------------------------------------------------------
# Phase 3 review I-1: FTS5 thread safety smoke test
# ---------------------------------------------------------------------------


def test_concurrent_search_and_index_does_not_raise(
    markdown_backend: MarkdownWikiBackend,
) -> None:
    """Concurrent ``index()`` + ``search()`` from multiple threads on
    the same ``kb_id`` must not raise ``sqlite3.ProgrammingError``
    or corrupt the cache.

    Regression for code-quality review I-1: pre-fix, the FTS5
    connection was built with the default ``check_same_thread=True``
    so any cross-thread reuse raised ``ProgrammingError``; the lock
    addition makes concurrent access serialize cleanly.
    """
    import threading

    kb_id = "kb-thread"
    # Seed the KB so the FTS5 cache is buildable.
    seed_doc = _make_doc(
        source_type="file", title="seed",
        content="seed content for FTS5 thread test",
        uri="/tmp/seed.txt", doc_id="src-seed",
    )
    markdown_backend.index(kb_id=kb_id, documents=[seed_doc])

    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def search_worker(idx: int) -> None:
        try:
            barrier.wait(timeout=5.0)
            for _ in range(10):
                markdown_backend.search(
                    kb_id=kb_id, text="seed content", top_k=5
                )
        except BaseException as exc:  # noqa: BLE001 — propagate to assertion
            errors.append(exc)

    def index_worker(idx: int) -> None:
        try:
            barrier.wait(timeout=5.0)
            for j in range(5):
                doc = _make_doc(
                    source_type="file",
                    title=f"t{idx}-{j}",
                    content=f"thread {idx} doc {j} payload alpha",
                    uri=f"/tmp/t{idx}-{j}.txt",
                    doc_id=f"src-t{idx}-{j}",
                )
                markdown_backend.index(kb_id=kb_id, documents=[doc])
        except BaseException as exc:  # noqa: BLE001 — propagate to assertion
            errors.append(exc)

    threads = [
        threading.Thread(target=search_worker, args=(i,)) for i in range(4)
    ] + [
        threading.Thread(target=index_worker, args=(i,)) for i in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert not errors, f"concurrent ops raised: {errors!r}"
    # Final state should reflect all index() inserts (4 threads * 5 docs
    # + 1 seed = 21 pages).
    assert markdown_backend.count(kb_id=kb_id) == 21
