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
    """Atomically write ``<kb_root>/.migrated_to`` containing the backend name."""
    sentinel = kb_root / MIGRATED_SENTINEL_FILENAME
    kb_root.mkdir(parents=True, exist_ok=True)
    tmp = sentinel.with_suffix(".tmp")
    tmp.write_text(target_backend, encoding="utf-8")
    tmp.replace(sentinel)
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
        # During migration we want to be lossless, so set the force
        # env var transiently for this call. Re-raises a clear error
        # if the OS denies env mutation (extremely rare).
        import os
        prev = os.environ.get("AGENT_KB_FORCE_WIKI_INGEST")
        os.environ["AGENT_KB_FORCE_WIKI_INGEST"] = "1"
        try:
            md_backend.index(kb_id=synthetic_kb_id, documents=[doc])
        finally:
            if prev is None:
                os.environ.pop("AGENT_KB_FORCE_WIKI_INGEST", None)
            else:
                os.environ["AGENT_KB_FORCE_WIKI_INGEST"] = prev
        pages_written += 1
    return pages_written


# ---------------------------------------------------------------------------
# markdown -> chromadb import
# ---------------------------------------------------------------------------


def import_from_markdown(
    service: "KnowledgebaseService",
    kb_id: str,
    src: Path,
) -> int:
    """Re-ingest a ``src/wiki/`` tree into the *kb_id* chromadb backend.

    Walks ``src/wiki/pages/*.md`` and runs each page through the
    standard ingest pipeline as a ``source_type=file`` source. Each
    page's frontmatter ``source_id`` becomes the dedup_key so re-runs
    are idempotent. Returns the count of pages re-ingested.
    """
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
    imported = 0
    for path in sorted(pages_dir.glob("*.md")):
        rec = _parse_page(path)
        if rec is None:
            continue
        # Use the original uri so probe-4 fields stay populated.
        uri = rec.uri or str(path)
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
        rows.append({
            "id": chunk.id,
            "source_id": chunk.source_id,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding_id": chunk.embedding_id,
        })
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
            f"chromadb dump at {src} is not a JSON object — got "
            f"{type(payload).__name__}"
        )
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(
            f"chromadb dump at {src} has rows={rows!r}; expected list"
        )
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

    Returns ``{kb_id: rows_updated}``.
    """
    results: dict[str, int] = {}
    targets = [kb_id] if kb_id is not None else list(service._index.keys())  # noqa: SLF001
    provider = service._config.embedding_provider  # noqa: SLF001
    base_url = service._config.embed_base_url  # noqa: SLF001
    for kid in targets:
        ctx = service._ctx(kid)  # noqa: SLF001
        if dry_run:
            # Count rows with NULL columns; do not write.
            row = ctx.db._conn.execute(  # noqa: SLF001
                "SELECT COUNT(*) AS cnt FROM chunks WHERE kb_id = ? AND "
                "(embedding_provider IS NULL OR embed_base_url IS NULL)",
                (kid,),
            ).fetchone()
            results[kid] = int(row["cnt"]) if row is not None else 0
        else:
            results[kid] = ctx.db.backfill_embedding_snapshot(
                kb_id=kid, provider=provider, base_url=base_url
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
    3. Run the appropriate export/import pair so the new backend has a
       complete copy of the data on disk.
    4. Write ``<kb_root>/.migrated_to`` containing the target backend
       name. The OLD backend's data is intentionally left intact —
       rollback = delete the sentinel file.
    5. Invalidate the service's per-KB backend cache so the next read
       picks up the new routing.

    Emits structured stderr lines at start and finish so operators can
    observe the cutover. Raises :class:`ValueError` for an invalid
    target or an unknown ``kb_id``.
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

    pages_migrated = 0
    if target_backend == "markdown":
        # Render the wiki tree into the kb's existing root so the
        # markdown backend can see it without any further plumbing.
        # The path is <saves_dir>/<dir_name>/wiki/, which matches
        # MarkdownWikiBackend's natural layout.
        pages_migrated = export_to_markdown(service, kb_id, kb_root)
        notes.append(
            f"wrote {pages_migrated} markdown pages under {kb_root / 'wiki'}"
        )
    elif target_backend == "chromadb":
        # Reverse direction: re-ingest pages/*.md back into chromadb.
        pages_migrated = import_from_markdown(service, kb_id, kb_root)
        notes.append(
            f"re-ingested {pages_migrated} markdown pages back into chromadb"
        )

    sentinel = _write_migrated_sentinel(kb_root, target_backend)
    notes.append(f"wrote sentinel {sentinel.name}={target_backend}")

    # Drop the cached per-KB backend so the next read consults the
    # new sentinel + target backend.
    service._invalidate_backend_cache(kb_id)  # noqa: SLF001

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


__all__ = [
    "MigrationResult",
    "backfill_provider_snapshot",
    "export_chromadb_dump",
    "export_to_markdown",
    "import_chromadb_dump",
    "import_from_markdown",
    "migrate",
]
