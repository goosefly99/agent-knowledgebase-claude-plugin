"""KB migration tooling — chromadb <-> markdown wiki round-trip.

spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b (v2.1)
Source: pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json
        implementation.phases[4]  (Phase 4 — Migration tooling +
        dual-backend transition + per-page provider snapshot)

Phase 4 deliverable. Migration tooling SHIPS BEFORE the Phase 5
embedding default-flip; without it (and the per-page provider snapshot
+ read-fallback) every existing v0.6.0 chromadb KB indexed with the
Ollama default would silently break the moment Phase 5 swaps the
global ``embedding_provider`` default to ``'remote'``.

Public surface
--------------

* :func:`export_to_markdown(service, kb_id, dest)` — render every
  page in *kb_id* as a wiki markdown file under *dest*.
* :func:`import_from_markdown(service, kb_id, src)` — re-ingest a
  ``wiki/`` tree into the chromadb backend by walking
  ``src/wiki/pages/*.md``.
* :func:`export_chromadb_dump(service, kb_id, dest)` — dump the per-KB
  chromadb collection (id + content + metadata + embedding) to a
  portable JSON file.
* :func:`import_chromadb_dump(service, kb_id, src)` — restore a
  per-KB collection from such a JSON file.
* :func:`migrate(service, kb_id, target_backend)` — orchestrates a
  full chromadb -> markdown (or markdown -> chromadb) cutover and
  writes the ``.migrated_to`` sentinel file on success.
* :func:`backfill_provider_snapshot(service, kb_id=None, dry_run=False)`
  — populate the Phase 4 ``chunks.embedding_provider`` /
  ``chunks.embed_base_url`` columns for legacy KBs whose chunks were
  ingested before the snapshot existed. Reads the kb's
  ``Settings.embedding_provider`` / ``Settings.embed_base_url`` to
  decide what to stamp.

Non-destructive contract
------------------------

``migrate()`` writes the new backend's data on disk AND emits the
``.migrated_to`` sentinel; the OLD backend's data is left intact. The
sentinel is the routing signal — operators wishing to reclaim the old
storage must delete the old directory manually. This deliberately
keeps the migration round-trippable: rolling back is just removing the
sentinel file.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_knowledgebase.backends import MIGRATED_SENTINEL_FILENAME
from agent_knowledgebase.config import sanitize_kb_dir_name
from agent_knowledgebase.services.stderr_log import knowledgebase_stderr_log

if TYPE_CHECKING:
    from agent_knowledgebase.services.knowledgebase import KnowledgebaseService


_VALID_BACKENDS = frozenset({"chromadb", "markdown"})
_SPEC_ID = "70ab2170-381a-4657-bcd1-28a40c6f369b"

# Sentinel value written inside the .migrated_to file for textvec migration.
_TEXTVEC_SENTINEL_CONTENT = "textvec"

# Lock filename for cross-process coordination (see docs/cross-process-lock-recipe.md).
_INGEST_LOCK_FILENAME = ".ingest.lock"

# Filelock timeout in seconds: 0 = try-once (non-blocking), return deferred on contention.
_MIGRATION_FILELOCK_TIMEOUT = 0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class MigrationResult:
    """Summary of a single :func:`migrate` invocation.

    Returned to the MCP layer so ``kb_migrate`` can serialize it as the
    JSON tool response. ``sentinel_path`` is the absolute path written
    on disk; operators rolling back a migration delete that file to
    revert routing.
    """

    kb_id: str
    target_backend: str
    pages_migrated: int
    sentinel_path: str
    notes: list[str]

    def to_payload(self) -> dict[str, Any]:
        return {
            "kb_id": self.kb_id,
            "target_backend": self.target_backend,
            "pages_migrated": self.pages_migrated,
            "sentinel_path": self.sentinel_path,
            "notes": list(self.notes),
            "spec_id": _SPEC_ID,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_kb_dir(service: "KnowledgebaseService", kb_id: str) -> Path:
    """Return the per-KB on-disk root directory for *kb_id*.

    Mirrors :class:`MarkdownWikiBackend._kb_root` — uses
    ``service._index`` when available, otherwise falls back to
    ``saves_dir / sanitize_kb_dir_name(kb_id)``.
    """
    dir_name = service._index.get(kb_id) if hasattr(service, "_index") else None  # noqa: SLF001
    if dir_name is None:
        dir_name = sanitize_kb_dir_name(kb_id)
    return service._config.kb_data_dir(dir_name)  # noqa: SLF001 — by design


def _write_migrated_sentinel(kb_root: Path, target_backend: str) -> Path:
    """Atomically write ``<kb_root>/.migrated_to`` containing the backend name.

    Uses :func:`tempfile.mkstemp` keyed on the ``.migrated_to.``
    prefix so the tmp file lands beside the sentinel (same filesystem,
    enabling atomic ``os.replace``) and concurrent migrations targeting
    the same kb_root never collide on the same tmp filename.

    Note on the prior bug: an earlier implementation used the pathlib
    suffix-replacement helper to derive the tmp path from the sentinel
    name, which silently stripped the dot-prefixed filename (treating
    it as a suffix) and produced a tmp file in the wrong directory
    AND a process-wide collision target. The mkstemp route avoids
    both bugs. See test_sentinel_tmp_file_uses_correct_prefix.
    """
    sentinel = kb_root / MIGRATED_SENTINEL_FILENAME
    kb_root.mkdir(parents=True, exist_ok=True)
    # delete=False so we control the lifecycle: we rename the file into
    # place via os.replace; on any failure we unlink the leftover tmp.
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{MIGRATED_SENTINEL_FILENAME}.",
        suffix=".tmp",
        dir=str(kb_root),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(target_backend)
        os.replace(tmp_path, sentinel)
    except Exception:
        # Best-effort cleanup; propagate the original exception.
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise
    return sentinel


# ---------------------------------------------------------------------------
# chromadb -> markdown export
# ---------------------------------------------------------------------------


def export_to_markdown(
    service: "KnowledgebaseService",
    kb_id: str,
    dest: Path,
) -> int:
    """Render every page in *kb_id* as a markdown wiki tree under *dest*.

    Pulls ``Source``/``Chunk`` data from the v0.6.0 chromadb-backed
    SQLite database for ``kb_id`` and re-creates the wiki layout under
    ``dest`` using the same ``MarkdownWikiBackend`` that Phase 3 ships
    (so the resulting tree is interchangeable with one that was
    natively ingested under ``kb_backend='markdown'``).

    Returns the number of source documents written. The destination is
    a clean ``MarkdownWikiBackend`` invocation — existing files at
    *dest* are left in place; only newly-rendered files are written.
    """
    from agent_knowledgebase.backends.markdown_backend import MarkdownWikiBackend

    ctx = service._ctx(kb_id)  # noqa: SLF001 — by design
    sources = ctx.db.list_sources(kb_id)

    # Build a service-less MarkdownWikiBackend rooted at *dest*. We
    # construct a new Settings via model_copy so the saves_dir is
    # *dest.parent* and the kb_id is the leaf of *dest* — this keeps
    # the markdown backend's path-resolution rules intact while
    # letting callers pick an arbitrary destination directory.
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    # When dest is a sibling of saves_dir, point the backend at a
    # synthetic saves_dir = dest.parent and kb_id = dest.name so
    # ``_kb_root`` resolves to ``dest`` exactly.
    synthetic_saves = dest.parent
    synthetic_kb_id = dest.name
    settings_for_export = service._config.model_copy(  # noqa: SLF001
        update={"saves_dir": synthetic_saves, "kb_backend": "markdown"}
    )
    md_backend = MarkdownWikiBackend(settings_for_export, service=None)

    pages_written = 0
    for source in sources:
        chunks = ctx.db.list_chunks(source.id)
        if not chunks:
            # Source with no chunks (e.g. failed ingest) — represent as a
            # placeholder page so the wiki round-trip is lossless.
            content = f"_Source {source.uri!r} has no ingested chunks._"
        else:
            # Re-assemble a single body from all chunks for that source
            # — markdown wiki is per-source, not per-chunk.
            content = "\n\n".join(c.content for c in chunks)
        doc = {
            "id": source.id,
            "content": content,
            "metadata": {
                "source_type": source.source_type.value,
                "uri": source.uri,
                "dedup_key": source.dedup_key,
                "title": source.uri,
            },
        }
        # The markdown allow-list rejects sql_database/codebase/etc.
        # During migration we want to be lossless, so call the private
        # ``_index_with_force`` entry point instead of mutating the
        # process-wide ``AGENT_KB_FORCE_WIKI_INGEST`` env var (which
        # would race with concurrent unrelated wiki ingests). The
        # private entry is documented as migration-only and is NOT
        # part of the public RetrieverBackend Protocol surface (see
        # I-05).
        md_backend._index_with_force(  # noqa: SLF001 — by design
            kb_id=synthetic_kb_id, documents=[doc]
        )
        pages_written += 1
    return pages_written


# ---------------------------------------------------------------------------
# markdown -> chromadb import
# ---------------------------------------------------------------------------


def import_from_markdown(
    service: "KnowledgebaseService",
    kb_id: str,
    src: Path,
    *,
    _lock_already_held: bool = False,
) -> int:
    """Re-ingest a ``src/wiki/`` tree into the *kb_id* chromadb backend.

    Walks ``src/wiki/pages/*.md`` and runs each page through the
    standard ingest pipeline as a ``source_type=file`` source. Each
    page's frontmatter ``source_id`` becomes the dedup_key so re-runs
    are idempotent. Returns the count of pages re-ingested.

    Phase 4 routing fix (B-01): the index write does NOT route through
    ``service._backend_for(kb_id)``. The OLD ``.migrated_to`` sentinel
    still says ``markdown`` while this import is in flight, so the
    cached per-KB backend is the markdown backend — and routing the
    re-ingest through it would write the pages back into the wiki and
    leave ZERO vectors in chromadb. Instead, instantiate a
    :class:`ChromadbBackend` directly and pass it as
    ``explicit_backend`` to the ingest pipeline so writes land in the
    real chromadb collection regardless of what the sentinel says.
    The sentinel is rewritten by the orchestrator AFTER this import
    completes. spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b

    ``_lock_already_held`` (Phase-4-migration-only): when ``True``,
    skip ``ingest_source`` (which would re-acquire the per-KB lock
    and self-deadlock since ``threading.Lock`` is non-reentrant) and
    call ``_ingest_source_locked`` directly. ``migrate()`` holds the
    lock around this whole import path (B-02), so this kwarg is set
    to ``True`` from there.
    """
    from agent_knowledgebase.backends.chromadb_backend import ChromadbBackend
    from agent_knowledgebase.backends.markdown_backend import _parse_page
    from agent_knowledgebase.models import SourceType

    src = Path(src)
    pages_dir = src / "wiki" / "pages"
    if not pages_dir.is_dir():
        # Permit callers to point directly at <kb-name> OR at <kb-name>/wiki
        alt_dir = src / "pages"
        if alt_dir.is_dir():
            pages_dir = alt_dir
    if not pages_dir.is_dir():
        return 0

    # Build a chromadb backend bound to the same service so the index
    # write lands in the right collection even though the sentinel
    # still says 'markdown'. Mirror the Settings.kb_backend='chromadb'
    # branch so the backend's vectorstore wiring uses the existing
    # service plumbing without any further config edits.
    chromadb_backend = ChromadbBackend(service._config, service=service)  # noqa: SLF001
    imported = 0
    for path in sorted(pages_dir.glob("*.md")):
        rec = _parse_page(path)
        if rec is None:
            continue
        # Use the original uri so probe-4 fields stay populated.
        uri = rec.uri or str(path)
        if _lock_already_held:
            service._ingest_source_locked(  # noqa: SLF001 — by design
                kb_id,
                SourceType.file,
                uri,
                {
                    "source_type": rec.source_type or "file",
                    "title": rec.title,
                    "imported_from_markdown": True,
                },
                rec.source_id or rec.slug,
                "replace",
                explicit_backend=chromadb_backend,
            )
        else:
            service.ingest_source(
                kb_id=kb_id,
                source_type=SourceType.file,
                uri=uri,
                metadata={
                    "source_type": rec.source_type or "file",
                    "title": rec.title,
                    "imported_from_markdown": True,
                },
                dedup_key=rec.source_id or rec.slug,
                dedup_policy="replace",
                explicit_backend=chromadb_backend,
            )
        imported += 1
    return imported


# ---------------------------------------------------------------------------
# chromadb dump (portable JSON snapshot)
# ---------------------------------------------------------------------------


def export_chromadb_dump(
    service: "KnowledgebaseService",
    kb_id: str,
    dest: Path,
) -> int:
    """Dump per-KB chromadb collection contents to a portable JSON file.

    Each row is ``{id, content, metadata, embedding}`` — embedding is
    optional (may be ``None`` for vectorstores that don't expose
    vectors). Returns the row count written. The dump format is a
    single JSON object ``{"kb_id": ..., "spec_id": ..., "rows":
    [{...}, ...]}`` so future schema evolutions can be detected.
    """
    ctx = service._ctx(kb_id)  # noqa: SLF001
    chunks = []
    for source in ctx.db.list_sources(kb_id):
        chunks.extend(ctx.db.list_chunks(source.id))
    rows: list[dict[str, Any]] = []
    for chunk in chunks:
        rows.append(
            {
                "id": chunk.id,
                "source_id": chunk.source_id,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "embedding_id": chunk.embedding_id,
            }
        )
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kb_id": kb_id,
        "spec_id": _SPEC_ID,
        "format_version": "1",
        "rows": rows,
    }
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(dest)
    return len(rows)


def import_chromadb_dump(
    service: "KnowledgebaseService",
    kb_id: str,
    src: Path,
) -> int:
    """Restore a per-KB chromadb dump produced by :func:`export_chromadb_dump`.

    Inserts each row into the per-KB SQLite ``chunks`` table; vectors
    that carry an ``embedding`` field are NOT re-inserted into
    chromadb here — re-embedding via :meth:`KnowledgebaseService.update_source`
    is the supported re-vectorize path. Returns the number of rows
    restored. Idempotent: existing chunk-id collisions are skipped.
    """
    from agent_knowledgebase.models import Chunk

    src = Path(src)
    if not src.is_file():
        return 0
    payload = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"chromadb dump at {src} is not a JSON object — got {type(payload).__name__}"
        )
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"chromadb dump at {src} has rows={rows!r}; expected list")
    ctx = service._ctx(kb_id)  # noqa: SLF001
    inserted = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        chunk_id = row.get("id")
        if chunk_id is None or ctx.db.get_chunk(chunk_id) is not None:
            continue
        chunk = Chunk(
            id=chunk_id,
            source_id=row.get("source_id", ""),
            kb_id=kb_id,
            content=row.get("content", ""),
            metadata=row.get("metadata") or {},
            embedding_id=row.get("embedding_id"),
        )
        ctx.db.insert_chunk(chunk)
        inserted += 1
    return inserted


# ---------------------------------------------------------------------------
# Backfill: legacy chunks get embedding_provider / embed_base_url stamped
# ---------------------------------------------------------------------------


def backfill_provider_snapshot(
    service: "KnowledgebaseService",
    *,
    kb_id: str | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Backfill ``chunks.embedding_provider`` / ``chunks.embed_base_url``.

    Per-Phase-5 gate (per spec): every existing v0.6.0 KB must have
    its embedding-provider snapshot populated BEFORE Phase 5 may flip
    the global default. This routine reads each KB's
    ``Settings.embedding_provider`` / ``Settings.embed_base_url`` (the
    process-wide values, since the v0.6.0 schema stored no per-KB
    config) and stamps them onto every chunk row that currently has
    NULL columns.

    When ``kb_id`` is ``None`` the backfill runs across every KB known
    to the service. When ``dry_run`` is ``True`` no writes occur — the
    return dict still reports the number of rows that *would* have
    been touched.

    Heterogeneous-KB safety (I-02)
    ------------------------------

    The UPDATE is restricted to chunks whose ``metadata.embedding_model``
    matches ``Settings.embedding_model``. Chunks with a different stamped
    model (e.g. a KB that was partially ingested under one embedder, then
    further ingested under another, then has its config flipped) are
    LEFT UNTOUCHED so we never misstamp them with the wrong
    ``(provider, base_url)`` tuple. Operators with mixed-embedder history
    must call :meth:`Database.backfill_embedding_snapshot` directly per
    model. See ``docs/migration_guide.md`` for the multi-model recipe.

    Returns ``{kb_id: rows_updated}``.
    """
    results: dict[str, int] = {}
    targets = [kb_id] if kb_id is not None else list(service._index.keys())  # noqa: SLF001
    provider = service._config.embedding_provider  # noqa: SLF001
    base_url = service._config.embed_base_url  # noqa: SLF001
    embedding_model = service._config.embedding_model  # noqa: SLF001
    for kid in targets:
        ctx = service._ctx(kid)  # noqa: SLF001
        if dry_run:
            # Count rows with NULL columns AND matching model — mirrors
            # the WHERE clause backfill_embedding_snapshot will use so
            # the dry-run count is the actual number of rows the live
            # run would touch (not an over-count that includes
            # mismatched-model chunks the live run would correctly
            # skip).
            row = ctx.db._conn.execute(  # noqa: SLF001
                "SELECT COUNT(*) AS cnt FROM chunks WHERE kb_id = ? "
                "AND (embedding_provider IS NULL OR embed_base_url IS NULL) "
                "AND json_extract(metadata, '$.embedding_model') = ?",
                (kid, embedding_model),
            ).fetchone()
            results[kid] = int(row["cnt"]) if row is not None else 0
        else:
            results[kid] = ctx.db.backfill_embedding_snapshot(
                kb_id=kid,
                provider=provider,
                base_url=base_url,
                embedding_model=embedding_model,
            )
    return results


# ---------------------------------------------------------------------------
# Top-level orchestrator (kb_migrate MCP tool entry point)
# ---------------------------------------------------------------------------


def migrate(
    service: "KnowledgebaseService",
    *,
    kb_id: str,
    target_backend: str,
) -> MigrationResult:
    """Orchestrate a full backend cutover for *kb_id*.

    Steps:

    1. Validate ``target_backend`` is in ``{'chromadb', 'markdown'}``.
    2. Detect the current backend from ``Settings`` + per-KB overrides.
    3. Acquire the per-kb_id ``threading.Lock`` (mirrors
       ``ingest_source`` / ``update_source`` — see B-02). Concurrent
       ``kb_query`` / ``kb_search`` against the same KB during a
       migration would otherwise observe torn state mid-cutover.
    4. Run the appropriate export/import pair so the new backend has a
       complete copy of the data on disk.
    5. Write ``<kb_root>/.migrated_to`` containing the target backend
       name. The OLD backend's data is intentionally left intact —
       rollback = delete the sentinel file.
    6. Invalidate the service's per-KB backend cache so the next read
       picks up the new routing.

    Emits structured stderr lines at start and finish AND on lock
    acquire / acquired / release boundaries (mirroring the existing
    ingest_source pattern) so operators can observe the cutover.
    Raises :class:`ValueError` for an invalid target or an unknown
    ``kb_id``.

    spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b
    """
    if target_backend not in _VALID_BACKENDS:
        raise ValueError(
            f"target_backend={target_backend!r} not supported by "
            f"kb_migrate; expected one of {sorted(_VALID_BACKENDS)}"
        )
    if kb_id not in service._index:  # noqa: SLF001
        raise ValueError(f"unknown kb_id={kb_id!r}")
    kb_root = _resolve_kb_dir(service, kb_id)
    notes: list[str] = []

    knowledgebase_stderr_log(
        kb_id=kb_id,
        op="kb_migrate",
        phase="migrate_start",
        elapsed_ms=0,
        rows_in=0,
        rows_ok=0,
        rows_skipped=0,
        rows_failed=0,
        dedup_policy="n/a",
        request_id=None,
        tool_caller_version=None,
    )

    # B-02: acquire the per-kb_id lock for the entire export/import +
    # sentinel-write + cache-invalidate sequence. Without it, a
    # concurrent kb_query against the same KB would observe a torn
    # state mid-cutover (e.g. after the wiki/ tree is materialized
    # but before the sentinel is written, the per-KB cache still
    # routes through chromadb but the new data is half-on-disk under
    # wiki/). Emit lock_acquire / lock_acquired / lock_release
    # telemetry mirroring the ingest_source pattern so operators can
    # see the lock boundaries on the same op="kb_migrate" stream.
    knowledgebase_stderr_log(
        kb_id=kb_id,
        op="kb_migrate",
        phase="lock_acquire",
        elapsed_ms=0,
        rows_in=0,
        rows_ok=0,
        rows_skipped=0,
        rows_failed=0,
        dedup_policy="n/a",
        request_id=None,
        tool_caller_version=None,
    )
    lock_acquired_at = time.monotonic()
    pages_migrated = 0
    with service._get_kb_lock(kb_id):  # noqa: SLF001 — by design
        knowledgebase_stderr_log(
            kb_id=kb_id,
            op="kb_migrate",
            phase="lock_acquired",
            elapsed_ms=0,
            rows_in=0,
            rows_ok=0,
            rows_skipped=0,
            rows_failed=0,
            dedup_policy="n/a",
            request_id=None,
            tool_caller_version=None,
        )
        try:
            if target_backend == "markdown":
                # Render the wiki tree into the kb's existing root so the
                # markdown backend can see it without any further plumbing.
                # The path is <saves_dir>/<dir_name>/wiki/, which matches
                # MarkdownWikiBackend's natural layout.
                pages_migrated = export_to_markdown(service, kb_id, kb_root)
                notes.append(f"wrote {pages_migrated} markdown pages under {kb_root / 'wiki'}")
            elif target_backend == "chromadb":
                # Reverse direction: re-ingest pages/*.md back into chromadb.
                # ``_lock_already_held=True`` so the import bypasses
                # ``ingest_source``'s lock acquisition (which would
                # self-deadlock against the lock we already hold —
                # threading.Lock is non-reentrant).
                pages_migrated = import_from_markdown(
                    service, kb_id, kb_root, _lock_already_held=True
                )
                notes.append(f"re-ingested {pages_migrated} markdown pages back into chromadb")

            sentinel = _write_migrated_sentinel(kb_root, target_backend)
            notes.append(f"wrote sentinel {sentinel.name}={target_backend}")

            # Drop the cached per-KB backend so the next read consults the
            # new sentinel + target backend.
            service._invalidate_backend_cache(kb_id)  # noqa: SLF001
        finally:
            elapsed_ms = int((time.monotonic() - lock_acquired_at) * 1000)
            knowledgebase_stderr_log(
                kb_id=kb_id,
                op="kb_migrate",
                phase="lock_release",
                elapsed_ms=elapsed_ms,
                rows_in=0,
                rows_ok=0,
                rows_skipped=0,
                rows_failed=0,
                dedup_policy="n/a",
                request_id=None,
                tool_caller_version=None,
            )

    knowledgebase_stderr_log(
        kb_id=kb_id,
        op="kb_migrate",
        phase="migrate_done",
        elapsed_ms=0,
        rows_in=pages_migrated,
        rows_ok=pages_migrated,
        rows_skipped=0,
        rows_failed=0,
        dedup_policy="n/a",
        request_id=None,
        tool_caller_version=None,
    )

    return MigrationResult(
        kb_id=kb_id,
        target_backend=target_backend,
        pages_migrated=pages_migrated,
        sentinel_path=str(sentinel),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Phase C — textvec migration (FTS5 backfill for legacy chromadb-stamped KBs)
# ---------------------------------------------------------------------------


def migrate_to_textvec(kb_id: str, service: "KnowledgebaseService") -> dict:
    """Backfill ``chunks_fts`` for a legacy chromadb-stamped KB.

    Idempotent via sentinel guard: if
    ``<saves_dir>/<kb-dir>/.migrated_to`` already contains ``"textvec"``,
    return early with ``status='already_migrated'`` without touching
    ``chunks_fts``. Otherwise: acquire per-kb cross-process filelock,
    call :meth:`Database.rebuild_fts5`, write the sentinel, emit a
    structured stderr-log line, and return a summary dict.

    The sentinel is the same ``.migrated_to`` file used by the existing
    chromadb→markdown migration path.  Once written, future
    ``kb_query`` / ``kb_search`` calls that route through
    :func:`~agent_knowledgebase.backends.resolve_backend_name` will
    automatically select :class:`TextvecBackend` for this KB because the
    sentinel takes precedence over both ``kb_backend_per_kb`` and the
    global ``kb_backend`` default.

    Cross-process lock
    ------------------

    Uses :class:`filelock.FileLock` keyed on
    ``<kb-root>/.ingest.lock`` (the same path recipe documented in
    ``docs/cross-process-lock-recipe.md``).  The FileLock coordinates
    with other ``migrate_to_textvec`` callers (in any process) AND with
    external wrappers that adopt the cross-process-lock-recipe.  It does
    **not** block in-process ``ingest_source`` calls; those are protected
    separately by the in-process ``threading.Lock`` acquired below, while
    SQLite WAL serialises writers at the engine level for concurrent SQL.

    ``filelock`` is an intentional non-runtime dependency of this plugin
    (see ``docs/cross-process-lock-recipe.md``), but a destructive-ish
    FTS5 migration **must not** proceed without cross-process locking.
    If ``filelock`` is not installed this function raises
    :class:`RuntimeError` with an install hint rather than silently
    falling through.

    The lock is acquired with a zero-second timeout so that another
    ``migrate_to_textvec`` (or an external-wrapper holder) already
    holding the lock causes an immediate return with
    ``status='deferred'`` rather than a raise — the caller can retry
    at a safe moment.

    Returns
    -------
    dict
        ``{"status": "migrated"|"already_migrated"|"deferred", "rows": N, "elapsed_ms": M}``
        with an optional ``"warning"`` key when ``rows == 0`` on a non-empty KB.
        On ``status='deferred'``, an additional ``"reason": "ingest_in_progress"`` key
        is included.

    spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b
    """
    # Upfront validation: reject unknown kb_ids before creating any
    # directory or acquiring any lock, to prevent stray-directory leaks.
    if kb_id not in service._index:  # noqa: SLF001 — by design
        raise ValueError(f"unknown kb_id={kb_id!r}")

    t0 = time.monotonic()

    def _elapsed_ms() -> int:
        return int((time.monotonic() - t0) * 1000)

    kb_root = _resolve_kb_dir(service, kb_id)
    kb_root.mkdir(parents=True, exist_ok=True)

    # Sentinel guard (idempotency): check before acquiring the lock.
    sentinel_path = kb_root / MIGRATED_SENTINEL_FILENAME
    if sentinel_path.is_file():
        try:
            content = sentinel_path.read_text(encoding="utf-8").strip()
        except OSError:
            content = ""
        if content == _TEXTVEC_SENTINEL_CONTENT:
            elapsed = _elapsed_ms()
            knowledgebase_stderr_log(
                kb_id=kb_id,
                op="kb_migrate",
                phase="migration",
                elapsed_ms=elapsed,
                rows_in=0,
                rows_ok=0,
                rows_skipped=0,
                rows_failed=0,
                dedup_policy="n/a",
                request_id=None,
                tool_caller_version=None,
                error_code=None,
                error_message="kb already migrated",
            )
            return {"status": "already_migrated", "rows": 0, "elapsed_ms": elapsed}

    # Cross-process filelock: timeout=0 → non-blocking try-once.
    # filelock is intentionally NOT a runtime dep of this plugin
    # (docs/cross-process-lock-recipe.md), but a destructive-ish FTS5
    # migration must not proceed without cross-process locking — fail
    # loudly with an actionable install hint rather than silently
    # skipping the lock.
    lock_path = kb_root / _INGEST_LOCK_FILENAME
    try:
        from filelock import FileLock
    except ImportError as exc:
        knowledgebase_stderr_log(
            kb_id=kb_id,
            op="kb_migrate",
            phase="migration",
            elapsed_ms=_elapsed_ms(),
            rows_in=0,
            rows_ok=0,
            rows_skipped=0,
            rows_failed=0,
            dedup_policy="n/a",
            request_id=None,
            tool_caller_version=None,
            error_code="MIGRATION_FILELOCK_MISSING",
            error_message=(
                "filelock not installed; cross-process safe migration unavailable. "
                "Install filelock in the wrapper environment per "
                "docs/cross-process-lock-recipe.md (e.g. `pip install filelock>=3.0`)."
            ),
        )
        raise RuntimeError(
            "filelock not installed; cannot run migrate_to_textvec safely. "
            "Install with `pip install filelock>=3.0` per "
            "docs/cross-process-lock-recipe.md."
        ) from exc

    try:
        lock = FileLock(str(lock_path), timeout=_MIGRATION_FILELOCK_TIMEOUT)
        lock.acquire()
    except Exception as exc:  # noqa: BLE001
        # FilelockTimeout or any acquire failure → deferred.
        elapsed = _elapsed_ms()
        knowledgebase_stderr_log(
            kb_id=kb_id,
            op="kb_migrate",
            phase="migration",
            elapsed_ms=elapsed,
            rows_in=0,
            rows_ok=0,
            rows_skipped=0,
            rows_failed=0,
            dedup_policy="n/a",
            request_id=None,
            tool_caller_version=None,
            error_code="MIGRATION_DEFERRED",
            error_message=str(exc),
        )
        return {
            "status": "deferred",
            "reason": "ingest_in_progress",
            "rows": 0,
            "elapsed_ms": elapsed,
        }

    try:
        # Re-check sentinel inside the lock to handle TOCTOU race.
        if sentinel_path.is_file():
            try:
                content = sentinel_path.read_text(encoding="utf-8").strip()
            except OSError:
                content = ""
            if content == _TEXTVEC_SENTINEL_CONTENT:
                elapsed = _elapsed_ms()
                knowledgebase_stderr_log(
                    kb_id=kb_id,
                    op="kb_migrate",
                    phase="migration",
                    elapsed_ms=elapsed,
                    rows_in=0,
                    rows_ok=0,
                    rows_skipped=0,
                    rows_failed=0,
                    dedup_policy="n/a",
                    request_id=None,
                    tool_caller_version=None,
                    error_code=None,
                    error_message="kb already migrated",
                )
                return {"status": "already_migrated", "rows": 0, "elapsed_ms": elapsed}

        # In-process serialisation: acquire the per-kb_id threading.Lock
        # inside the cross-process FileLock, mirroring the layering used
        # by migrate().  This closes the in-process concurrency gap when
        # two threads in the same MCP server process race on the same KB.
        with service._get_kb_lock(kb_id):  # noqa: SLF001 — by design
            # Run FTS5 backfill via Database.rebuild_fts5().
            ctx = service._ctx(kb_id)  # noqa: SLF001 — by design
            try:
                rows = ctx.db.rebuild_fts5(kb_id)
            except Exception as exc:  # noqa: BLE001
                elapsed = _elapsed_ms()
                knowledgebase_stderr_log(
                    kb_id=kb_id,
                    op="kb_migrate",
                    phase="migration",
                    elapsed_ms=elapsed,
                    rows_in=0,
                    rows_ok=0,
                    rows_skipped=0,
                    rows_failed=1,
                    dedup_policy="n/a",
                    request_id=None,
                    tool_caller_version=None,
                    error_code="MIGRATION_FAILED",
                    error_message=str(exc),
                )
                raise

            # Write .migrated_to sentinel (atomic via mkstemp + os.replace).
            _write_migrated_sentinel(kb_root, _TEXTVEC_SENTINEL_CONTENT)

            # Invalidate the per-KB backend cache so the next query routes
            # through TextvecBackend immediately (sentinel controls routing).
            service._invalidate_backend_cache(kb_id)  # noqa: SLF001

            elapsed = _elapsed_ms()

            # Warn when no chunks were backfilled on a KB that has chunk rows
            # (e.g. chunks.content is NULL / empty on every row — edge case).
            result: dict[str, object] = {
                "status": "migrated",
                "rows": rows,
                "elapsed_ms": elapsed,
            }
            error_code: str | None = None
            error_message: str | None = "kb migrated to textvec backend"
            if rows == 0:
                # Check whether the KB is genuinely empty or anomalous.
                try:
                    chunk_count_row = ctx.db._conn.execute(  # noqa: SLF001
                        "SELECT COUNT(*) AS cnt FROM chunks WHERE kb_id = ?", (kb_id,)
                    ).fetchone()
                    total_chunks = int(chunk_count_row["cnt"]) if chunk_count_row else 0
                except Exception:  # noqa: BLE001
                    total_chunks = 0
                if total_chunks > 0:
                    error_code = "MIGRATION_NO_ROWS"
                    error_message = "no_chunks_text"
                    result["warning"] = "no_chunks_text"

            knowledgebase_stderr_log(
                kb_id=kb_id,
                op="kb_migrate",
                phase="migration",
                elapsed_ms=elapsed,
                rows_in=0,
                rows_ok=rows,
                rows_skipped=0,
                rows_failed=0,
                dedup_policy="n/a",
                request_id=None,
                tool_caller_version=None,
                error_code=error_code,
                error_message=error_message,
            )
            return result

    finally:
        try:
            lock.release()
        except Exception:  # noqa: BLE001
            pass


__all__ = [
    "MigrationResult",
    "backfill_provider_snapshot",
    "export_chromadb_dump",
    "export_to_markdown",
    "import_chromadb_dump",
    "import_from_markdown",
    "migrate",
    "migrate_to_textvec",
]
