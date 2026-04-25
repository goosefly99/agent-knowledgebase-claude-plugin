"""Karpathy-style markdown wiki backend (Phase 3 deliverable, OPT-IN).

spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b (v2.1)
Source: pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json
        implementation.phases[3] (Phase 3 — MarkdownWikiBackend)

This module is the Phase 3 deliverable: a *second* implementation of the
:class:`agent_knowledgebase.backends.RetrieverBackend` Protocol that
stores knowledge as Karpathy-style markdown wiki files on disk
(``docs/karpathy-llm-wiki.md``) instead of as chromadb vectors. It is
**opt-in**: ``Settings.kb_backend`` defaults to ``'chromadb'`` and only
``AGENT_KB_BACKEND=markdown`` routes through this code.

Layout under ``<saves_dir>/<kb-dir>/wiki/``::

    raw/<source_id>.<ext>      # immutable raw source dump
    wiki/index.md              # content-oriented catalog (sharded above 200)
    wiki/index/<category>.md   # per-category shard (created on transition)
    wiki/hot.md                # ~500-char hot cache (placeholder for now)
    wiki/log.md                # chronological op log
    wiki/pages/<slug>.md       # one file per ingested article

Retrieval is a sqlite FTS5 in-memory index over ``pages/*.md`` (NO
embedding). The Protocol's ``query()`` (vector path) re-routes to
``search()`` for this backend and rescales scores into [0, 1] descending.

Per-source_type **POSITIVE ALLOW-list** (NOT a deny-list — see f-15)::

    {file, website, api_endpoint with payload < 2MB}

Anything else (``sql_database``, ``codebase``, ``git_history``,
``directory``, ``api_endpoint`` >= 2MB) raises
:class:`KbIngestorUnsuitableError` UNLESS the ``AGENT_KB_FORCE_WIKI_INGEST=1``
env override is set, in which case ingest proceeds and a structured
:func:`knowledgebase_stderr_log` warning is emitted. See
``docs/markdown_backend.md`` for the per-source_type cost-model rationale.

MVP scope (v2.1)
----------------

Synchronous ingest: raw content -> deterministic page markdown via a
simple transformer (file body verbatim; website -> trafilatura plain text
when available, else raw bytes; api_endpoint payload -> fenced code
block). The spec line about "out-of-band LLM page-extraction /
status=pending_extraction" is deferred to a future release — see
``docs/markdown_backend.md`` "Limitations" section.

Probe-4 contract
----------------

``info(kb_id=...)`` returns AT LEAST the v0.6.0 keys::

    source_type, uri, dedup_key, page_id, dominant_embedding_model

For the markdown backend ``dominant_embedding_model`` is ALWAYS
``None`` (markdown does not embed) — but the key is **present**, NOT
omitted (validation finding f-20). Backend-diagnostic fields like
``vector_count`` / ``embedding_provider`` likewise return ``None``.

Thread-safe: per-kb_id FTS5 cache guarded by ``_fts_lock``; concurrent
``index()`` and ``search()`` calls on the same kb_id serialize.
Connections are built with ``check_same_thread=False`` and every
sqlite operation against a cached connection is held under the lock.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_knowledgebase.config import sanitize_kb_dir_name

if TYPE_CHECKING:
    from agent_knowledgebase.config import Settings
    from agent_knowledgebase.services.knowledgebase import KnowledgebaseService


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


_BACKEND_NAME = "markdown"
_SPEC_VERSION = "2.1"
_SPEC_ID = "70ab2170-381a-4657-bcd1-28a40c6f369b"

# Positive-allow list of source_type values the markdown backend accepts.
# Per spec implementation.phases[3].tasks[5] and validation finding f-15
# (positive form, NOT a deny-list). Anything outside this set raises
# KbIngestorUnsuitableError unless AGENT_KB_FORCE_WIKI_INGEST=1.
_WIKI_ALLOW_TYPES: frozenset[str] = frozenset({"file", "website", "api_endpoint"})

# api_endpoint payloads >= this many bytes also raise UNSUITABLE — cost
# of LLM page-extraction (a future enhancement) plus on-disk markdown
# size scales with payload, and a 2MB+ JSON dump is almost always a
# misuse of the wiki abstraction.
_API_ENDPOINT_MAX_BYTES: int = 2 * 1024 * 1024  # 2 MiB

# Sharding threshold: when pages count crosses this, index.md becomes a
# directory pointer to per-category index/<category>.md files. Per
# spec.phases[3].tasks[4] domain-specialist break-even.
_SHARD_THRESHOLD: int = 200

# Override env var — when set to "1", ingest of disallowed source_types
# proceeds with a stderr_log warning instead of raising.
_FORCE_INGEST_ENV: str = "AGENT_KB_FORCE_WIKI_INGEST"

# Slugifier configuration.
_SLUG_MAX_LEN: int = 80
_SLUG_HASH_LEN: int = 6


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class KbIngestorUnsuitableError(RuntimeError):
    """Raised by :class:`MarkdownWikiBackend.index` when a document's
    ``source_type`` is outside the positive-allow list and the
    ``AGENT_KB_FORCE_WIKI_INGEST=1`` override is NOT set.

    The exception's ``error_code`` attribute is the SCREAMING_SNAKE_CASE
    code (``"KB_INGESTOR_UNSUITABLE"``) that the MCP tool layer can
    surface to callers; ``to_payload()`` serializes it into the
    structured-error dict the existing error-rendering helpers expect.
    """

    error_code: str = "KB_INGESTOR_UNSUITABLE"

    def __init__(
        self,
        *,
        source_type: str | None,
        reason: str,
    ) -> None:
        self.source_type = source_type
        self.reason = reason
        super().__init__(
            f"source_type={source_type!r} not allowed by markdown backend: "
            f"{reason}"
        )

    def to_payload(self) -> dict[str, Any]:
        """Return the structured-error JSON-serialisable payload."""
        return {
            "error": "kb_ingestor_unsuitable",
            "error_code": self.error_code,
            "source_type": self.source_type,
            "reason": self.reason,
            "remediation": (
                "either pick a source_type in {file, website, "
                "api_endpoint<2MB}, switch AGENT_KB_BACKEND back to "
                "'chromadb' (the default), or set "
                f"{_FORCE_INGEST_ENV}=1 to force ingest at your own risk"
            ),
            "spec_id": _SPEC_ID,
        }


# ---------------------------------------------------------------------------
# Slug derivation
# ---------------------------------------------------------------------------


_SLUG_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_SLUG_DUP_DASH_RE = re.compile(r"-{2,}")


def _slugify(title: str, *, content_hash_seed: str) -> str:
    """Deterministic slug: lowercase, non-alnum -> ``-``, max 80 chars,
    plus a 6-char sha256 suffix derived from ``content_hash_seed`` so
    similar titles never collide.
    """
    base = title.lower()
    base = _SLUG_NON_ALNUM_RE.sub("-", base)
    base = _SLUG_DUP_DASH_RE.sub("-", base).strip("-")
    if not base:
        base = "page"
    if len(base) > _SLUG_MAX_LEN:
        base = base[:_SLUG_MAX_LEN].rstrip("-")
    digest = hashlib.sha256(content_hash_seed.encode("utf-8")).hexdigest()[:_SLUG_HASH_LEN]
    return f"{base}-{digest}"


# ---------------------------------------------------------------------------
# Per-page metadata (header frontmatter parsing)
# ---------------------------------------------------------------------------


_FRONTMATTER_RE = re.compile(
    r"^---\n(?P<yaml>.*?)\n---\n(?P<body>.*)$",
    re.DOTALL,
)


@dataclass(frozen=True)
class _PageRecord:
    """In-memory view of a single ``pages/<slug>.md`` file."""

    slug: str
    page_id: str
    source_id: str
    source_type: str
    uri: str | None
    dedup_key: str | None
    title: str
    body: str
    created_at: str


def _format_frontmatter(meta: dict[str, Any]) -> str:
    """Render a tiny JSON-style YAML-ish header for a page.

    We use a JSON-in-YAML-fence so reads round-trip via stdlib
    ``json.loads`` without needing PyYAML as a runtime dependency.
    """
    payload = json.dumps(meta, sort_keys=True, indent=2)
    return f"---\n{payload}\n---\n"


def _parse_page(path: Path) -> _PageRecord | None:
    """Read ``pages/<slug>.md`` back into a :class:`_PageRecord`.

    Returns ``None`` if the file isn't a wiki page (e.g. partial write).
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return None
    try:
        meta = json.loads(match.group("yaml"))
    except json.JSONDecodeError:
        return None
    if not isinstance(meta, dict):
        return None
    body = match.group("body")
    return _PageRecord(
        slug=path.stem,
        page_id=str(meta.get("page_id", path.stem)),
        source_id=str(meta.get("source_id", "")),
        source_type=str(meta.get("source_type", "")),
        uri=meta.get("uri"),
        dedup_key=meta.get("dedup_key"),
        title=str(meta.get("title", path.stem)),
        body=body,
        created_at=str(meta.get("created_at", "")),
    )


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class MarkdownWikiBackend:
    """Markdown-wiki :class:`RetrieverBackend` (Phase 3, opt-in).

    Construction parameters
    -----------------------
    settings:
        The resolved :class:`Settings` instance the owning service was
        built from. Read for ``saves_dir`` resolution and the optional
        force-ingest env override (which is *also* read at call time so
        tests can flip it via ``monkeypatch.setenv`` without rebuilding
        the backend).
    service:
        Optional back-reference to the owning
        :class:`KnowledgebaseService`. Used to resolve the per-KB
        directory name via ``service._index`` so per-KB filesystem
        layout matches the rest of the codebase (sanitized name from
        ``Settings.kb_data_dir``). When ``service is None`` the backend
        falls back to ``saves_dir / sanitize_kb_dir_name(kb_id)`` so
        unit tests can drive the backend in isolation while still
        enjoying the same path-traversal defense the service-bound
        path inherits from :meth:`KnowledgebaseService.create`.
    """

    def __init__(
        self,
        settings: "Settings",
        *,
        service: "KnowledgebaseService | None" = None,
    ) -> None:
        self._settings = settings
        self._service = service
        # Per-kb_id sqlite FTS5 connections. Built lazily on first
        # search() or query() call so an empty KB doesn't pay the
        # disk-walk cost. Invalidated on index() / delete().
        #
        # Concurrency: the MCP server may dispatch tool calls from
        # multiple threads (the tool-timeout worker thread, future async
        # transports, etc). sqlite3 connections built with the default
        # ``check_same_thread=True`` raise ProgrammingError when reused
        # across threads, AND a single connection's execute() is racy
        # under concurrent use even with that flag turned off. We
        # therefore build connections with ``check_same_thread=False``
        # and serialize ALL access to the cache and to each cached
        # connection's sqlite operations through ``_fts_lock``. The
        # lock spans both the ``self._fts_indexes`` dict and per-conn
        # cursor calls, so the read/write critical section is held
        # contiguously per ``kb_id``.
        self._fts_lock: threading.Lock = threading.Lock()
        self._fts_indexes: dict[str, sqlite3.Connection] = {}

    # ------------------------------------------------------------------
    # Filesystem layout helpers
    # ------------------------------------------------------------------

    def _kb_root(self, kb_id: str) -> Path:
        """Return ``<saves_dir>/<kb-dir>/`` for ``kb_id``.

        When a service back-reference is bound, prefer the same
        sanitized directory the service uses (``service._index``); fall
        back to ``saves_dir / sanitize_kb_dir_name(kb_id)`` for
        service-less unit tests.

        The service-less branch ALWAYS routes ``kb_id`` through
        :func:`sanitize_kb_dir_name` so a hostile or malformed ``kb_id``
        (``"../escape"``, ``"a/b"``, ``"./foo"``) cannot escape
        ``saves_dir`` via path-traversal — the sanitizer strips
        anything not in ``[\\w\\s-]``, collapses whitespace, and falls
        back to ``"unnamed"`` for empty results, matching the same
        defense the service-bound path inherits from
        :meth:`KnowledgebaseService.create`.
        """
        if self._service is not None:
            dir_name = self._service._index.get(kb_id)  # noqa: SLF001 — by design
            if dir_name is None:
                # Mirror chromadb_backend's "lookup fails -> kb_id as
                # dirname" so error messages locate the missing KB at
                # the same path the create flow would have used. We
                # still sanitize here to keep the service-bound branch
                # equivalent to the service-less branch on unknown ids.
                dir_name = sanitize_kb_dir_name(kb_id)
            return self._settings.kb_data_dir(dir_name)
        return self._settings.saves_dir / sanitize_kb_dir_name(kb_id)

    def _wiki_root(self, kb_id: str) -> Path:
        return self._kb_root(kb_id) / "wiki"

    def _raw_dir(self, kb_id: str) -> Path:
        return self._wiki_root(kb_id) / "raw"

    def _pages_dir(self, kb_id: str) -> Path:
        return self._wiki_root(kb_id) / "pages"

    def _index_dir(self, kb_id: str) -> Path:
        return self._wiki_root(kb_id) / "index"

    def _index_md(self, kb_id: str) -> Path:
        return self._wiki_root(kb_id) / "index.md"

    def _log_md(self, kb_id: str) -> Path:
        return self._wiki_root(kb_id) / "log.md"

    def _hot_md(self, kb_id: str) -> Path:
        return self._wiki_root(kb_id) / "hot.md"

    def _ensure_layout(self, kb_id: str) -> None:
        """Create the wiki directory tree if missing (idempotent)."""
        self._raw_dir(kb_id).mkdir(parents=True, exist_ok=True)
        self._pages_dir(kb_id).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Allow-list enforcement
    # ------------------------------------------------------------------

    @staticmethod
    def _force_flag() -> bool:
        """Re-read the env var at call time so tests can flip it."""
        return os.environ.get(_FORCE_INGEST_ENV, "") == "1"

    @staticmethod
    def _doc_source_type(doc: dict[str, Any]) -> str | None:
        """Resolve ``source_type`` from either top-level or metadata."""
        st = doc.get("source_type")
        if not st:
            md = doc.get("metadata") or {}
            st = md.get("source_type")
        if st is None:
            return None
        # Accept SourceType enum members or plain strings.
        return getattr(st, "value", st)

    def _check_source_allowed(self, doc: dict[str, Any], *, kb_id: str) -> None:
        """Raise :class:`KbIngestorUnsuitableError` if ``doc`` is
        outside the positive-allow list, unless the force-flag is set
        (in which case emit a structured stderr_log warning and
        proceed).
        """
        st = self._doc_source_type(doc)
        if st in _WIKI_ALLOW_TYPES:
            if st == "api_endpoint":
                payload = doc.get("content", "")
                if isinstance(payload, str):
                    payload_size = len(payload.encode("utf-8"))
                elif isinstance(payload, (bytes, bytearray)):
                    payload_size = len(payload)
                else:
                    payload_size = len(str(payload).encode("utf-8"))
                if payload_size >= _API_ENDPOINT_MAX_BYTES:
                    if self._force_flag():
                        self._emit_force_warning(
                            kb_id=kb_id,
                            source_type=st,
                            reason=(
                                f"api_endpoint payload {payload_size} bytes "
                                f">= {_API_ENDPOINT_MAX_BYTES} byte wiki limit"
                            ),
                        )
                        return
                    raise KbIngestorUnsuitableError(
                        source_type=st,
                        reason=(
                            f"api_endpoint payload {payload_size} bytes "
                            f">= {_API_ENDPOINT_MAX_BYTES} byte wiki limit"
                        ),
                    )
            return
        if self._force_flag():
            self._emit_force_warning(
                kb_id=kb_id,
                source_type=st,
                reason=(
                    f"source_type={st!r} outside positive-allow list "
                    f"{sorted(_WIKI_ALLOW_TYPES)}"
                ),
            )
            return
        raise KbIngestorUnsuitableError(
            source_type=st,
            reason=(
                f"source_type={st!r} outside markdown-backend allow-list "
                f"{sorted(_WIKI_ALLOW_TYPES)}"
            ),
        )

    @staticmethod
    def _emit_force_warning(
        *, kb_id: str, source_type: str | None, reason: str
    ) -> None:
        """Emit a structured stderr_log line for force-ingest bypasses.

        Routed through :func:`knowledgebase_stderr_log` so the 11-field
        schema (pinned by ``tests/contract/test_stderr_schema.py``) is
        honored.
        """
        # Imported lazily to avoid a circular import path during module
        # load (services.stderr_log -> services.knowledgebase ->
        # backends -> backends.markdown_backend).
        from agent_knowledgebase.services.stderr_log import knowledgebase_stderr_log

        knowledgebase_stderr_log(
            kb_id=kb_id,
            op="markdown_backend_force_ingest",
            phase="markdown_ingest",
            elapsed_ms=0,
            rows_in=1,
            rows_ok=0,
            rows_skipped=0,
            rows_failed=0,
            dedup_policy="n/a",
            request_id=None,
            tool_caller_version=None,
            error_code="KB_INGESTOR_UNSUITABLE_FORCED",
            error_message=(
                f"source_type={source_type!r}: {reason} — proceeding "
                f"because {_FORCE_INGEST_ENV}=1"
            ),
        )

    # ------------------------------------------------------------------
    # Ingest write path
    # ------------------------------------------------------------------

    def index(
        self,
        *,
        kb_id: str,
        documents: list[dict[str, Any]],
    ) -> None:
        """Write each ``documents`` entry to the wiki on disk.

        Each entry MUST carry at least ``id`` (used as ``source_id``,
        falls back to a content-hash if missing) and ``content``;
        ``metadata.source_type`` (or top-level ``source_type``) drives
        the allow-list check; ``metadata.uri`` /
        ``metadata.dedup_key`` propagate to the page frontmatter.

        Empty ``documents`` is a no-op (mirrors
        :class:`ChromadbBackend.index`).

        For Phase 3 MVP this method is **synchronous and blocking**;
        out-of-band LLM page-extraction (the spec's
        ``status=pending_extraction`` path) is deferred — see
        ``docs/markdown_backend.md``.

        Atomicity caveat (multi-document batch)
        ---------------------------------------
        The allow-list check fires up-front for **every** document so
        a single bad doc fails the whole batch BEFORE any write. After
        validation passes, however, ``_write_one`` is called per
        document in sequence with **no cross-document transaction**:
        if the process crashes after writing document N out of M, the
        first N pages remain on disk and the index/log reflect only
        the successful subset. Recovery is to re-run ``kb_ingest_batch``
        for the same batch — page writes are idempotent overwrites
        (same ``source_id`` produces the same ``slug`` and the same
        page file). See :meth:`_write_one` for the per-document
        atomicity caveat (raw → page → log is also non-transactional
        across the three files).
        """
        if not documents:
            return
        # Validate ALL documents up-front so a partially-written batch
        # doesn't leave a half-ingested KB on disk because one
        # malformed doc was rejected mid-loop. (This does NOT prevent
        # crash-mid-batch — see Atomicity caveat in the docstring.)
        for doc in documents:
            self._check_source_allowed(doc, kb_id=kb_id)

        self._ensure_layout(kb_id)

        # Hold the per-backend lock for the entire write+rebuild path.
        # The spec promises concurrent index()/search() calls on the
        # same kb_id serialize, and we need that for THREE separate
        # reasons:
        #   1. log.md is opened for append from each _write_one — on
        #      POSIX that's atomic per-write but on Windows interleaved
        #      partial lines are possible.
        #   2. index.md is rewritten via tempfile + os.replace per
        #      batch; on Windows two parallel os.replace() calls onto
        #      the same target raise PermissionError.
        #   3. The FTS5 cache is invalidated at the end and rebuilt
        #      lazily on the next search(); a concurrent search()
        #      between (rebuild_index_md) and (cache.pop) would see
        #      a stale-but-still-mounted index.
        # The lock is fine-grained per-MarkdownWikiBackend instance,
        # not per-kb_id; in the typical single-process MCP server with
        # one backend instance this serializes ALL ingest writes
        # globally, which matches the behaviour of the chromadb
        # backend's per-collection client locks.
        with self._fts_lock:
            for doc in documents:
                self._write_one(kb_id=kb_id, doc=doc)
            self._rebuild_index_md(kb_id)
            # Invalidate the FTS5 cache so subsequent search()/query()
            # re-walks the pages directory.
            self._fts_indexes.pop(kb_id, None)

    def _write_one(self, *, kb_id: str, doc: dict[str, Any]) -> None:
        """Write a single document: raw + page + log entry.

        Atomicity caveat
        ----------------
        This method writes THREE files (raw dump, page markdown, log
        entry) in sequence. There is **no** cross-file transaction. If
        the process crashes mid-call:

        * crash between (1) and (2) leaves an orphan ``raw/<id>.<ext>``
          with no page or log entry.
        * crash between (2) and (3) leaves a written page that is
          missing from ``log.md`` (search/index queries still return it
          because they walk ``pages/`` directly).

        Recovery is **manual**: rerun ``kb_ingest`` for the same
        ``source_id`` — page writes are idempotent overwrites, so the
        same ``(raw, page, log)`` triple is produced. The orphan raw
        file from a previous crash is harmless (it gets overwritten by
        the rerun). True multi-file atomicity is deferred — see
        ``docs/markdown_backend.md`` Limitations §7.

        Each individual file write is per-file atomic (tempfile +
        ``os.replace``) via :func:`_atomic_write_text` /
        :func:`_atomic_write_bytes` so a crash MID-WRITE never produces
        a half-written file; the ordering across files is what is not
        transactional.
        """
        metadata = dict(doc.get("metadata") or {})
        source_type = self._doc_source_type(doc) or "unknown"
        uri = metadata.get("uri") or doc.get("uri") or ""
        dedup_key = metadata.get("dedup_key") or doc.get("dedup_key")
        content = doc.get("content", "")
        raw_source_id = str(
            doc.get("id") or doc.get("source_id") or _stable_id(content)
        )
        # Defense-in-depth: a hostile or malformed ``source_id``
        # (``"../escape"``, ``"a/b/c"``, embedded null bytes) would
        # otherwise compose a raw filename pointing outside ``raw/``.
        # Sanitize through the same helper used for kb_id directory
        # names; if sanitization yields the empty fallback "unnamed",
        # fall back to a content hash so two distinct hostile inputs
        # don't collide on the same file.
        source_id = _sanitize_source_id(raw_source_id, content=content)
        title = (
            metadata.get("title")
            or doc.get("title")
            or _derive_title(uri, content)
        )

        # 1. Raw dump (immutable copy of source content). Per-file atomic.
        raw_path = self._raw_dir(kb_id) / f"{source_id}.{_raw_ext(source_type)}"
        if isinstance(content, (bytes, bytearray)):
            _atomic_write_bytes(raw_path, bytes(content))
        else:
            _atomic_write_text(raw_path, str(content))

        # 2. Page markdown (deterministic transformer). Per-file atomic.
        body = _render_page_body(source_type=source_type, content=content, uri=uri)
        slug = _slugify(title, content_hash_seed=f"{source_id}:{title}")
        page_meta = {
            "page_id": slug,
            "source_id": source_id,
            "source_type": source_type,
            "uri": uri or None,
            "dedup_key": dedup_key,
            "title": title,
            "created_at": datetime.now(UTC).isoformat(),
            "spec_id": _SPEC_ID,
        }
        page_text = _format_frontmatter(page_meta) + body
        _atomic_write_text(self._pages_dir(kb_id) / f"{slug}.md", page_text)

        # 3. Log entry (Karpathy-style chronological op log). Append-only,
        # so a partial line is the only failure mode here — and that
        # remains a possibility (we cannot atomically append). The
        # docstring above documents the recovery path.
        ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%SZ")
        log_line = f"## [{ts}] ingest | {title}\n- slug: `{slug}`\n- source_type: `{source_type}`\n\n"
        with self._log_md(kb_id).open("a", encoding="utf-8") as fh:
            fh.write(log_line)

    # ------------------------------------------------------------------
    # Index.md construction (sharded above _SHARD_THRESHOLD pages)
    # ------------------------------------------------------------------

    def _rebuild_index_md(self, kb_id: str) -> None:
        """Regenerate ``index.md`` (or its sharded equivalent)."""
        pages = self._read_all_pages(kb_id)
        if len(pages) <= _SHARD_THRESHOLD:
            self._write_flat_index(kb_id, pages)
            return
        self._write_sharded_index(kb_id, pages)

    def _write_flat_index(self, kb_id: str, pages: list[_PageRecord]) -> None:
        """Single-file ``index.md`` listing every page (small KB form).

        Drops any pre-existing ``wiki/index/`` shard directory so the
        flat form is the ONLY remaining index representation. This
        matters when a delete drops the page count back under
        :data:`_SHARD_THRESHOLD` after a previous shard pass — without
        this cleanup ``info()['sharded_index']`` and operator-facing
        ``ls wiki/`` would still report stale shards on disk.
        """
        lines = [
            "# Wiki Index",
            "",
            f"_Pages: {len(pages)} | spec_id: {_SPEC_ID}_",
            "",
        ]
        for page in sorted(pages, key=lambda p: p.title.lower()):
            uri_suffix = f" — {page.uri}" if page.uri else ""
            lines.append(
                f"- [{page.title}](pages/{page.slug}.md) "
                f"`{page.source_type}`{uri_suffix}"
            )
        _atomic_write_text(self._index_md(kb_id), "\n".join(lines) + "\n")
        # Drop any stale shard directory if we crossed back below
        # threshold (no-op for the typical growing-KB case). We use
        # ignore_errors=True so a missing shard dir or a transient OS
        # error during cleanup doesn't fail the whole index rebuild.
        shutil.rmtree(self._index_dir(kb_id), ignore_errors=True)

    def _write_sharded_index(self, kb_id: str, pages: list[_PageRecord]) -> None:
        """Sharded ``index.md`` -> ``index/<source_type>.md`` files.

        Categories derive from ``source_type`` for v0 (per spec.phases[3].tasks[4]
        — "Categories may be derived from source_type for v0 — refine later.").
        """
        self._index_dir(kb_id).mkdir(parents=True, exist_ok=True)
        # Bucket pages by source_type, then write one shard per bucket.
        buckets: dict[str, list[_PageRecord]] = {}
        for page in pages:
            buckets.setdefault(page.source_type or "other", []).append(page)

        # Write per-shard files.
        for category, bucket in sorted(buckets.items()):
            shard_lines = [
                f"# Wiki Index — {category}",
                "",
                f"_Pages: {len(bucket)} | spec_id: {_SPEC_ID}_",
                "",
            ]
            for page in sorted(bucket, key=lambda p: p.title.lower()):
                uri_suffix = f" — {page.uri}" if page.uri else ""
                shard_lines.append(
                    f"- [{page.title}](../pages/{page.slug}.md){uri_suffix}"
                )
            _atomic_write_text(
                self._index_dir(kb_id) / f"{category}.md",
                "\n".join(shard_lines) + "\n",
            )

        # Top-level index.md becomes a directory pointer.
        top_lines = [
            "# Wiki Index (sharded)",
            "",
            f"_Pages: {len(pages)} | shards: {len(buckets)} | "
            f"spec_id: {_SPEC_ID}_",
            "",
            "## Categories",
            "",
        ]
        for category, bucket in sorted(buckets.items()):
            top_lines.append(
                f"- [{category}](index/{category}.md) ({len(bucket)} pages)"
            )
        _atomic_write_text(self._index_md(kb_id), "\n".join(top_lines) + "\n")

    # ------------------------------------------------------------------
    # Read path: page enumeration + FTS5
    # ------------------------------------------------------------------

    def _read_all_pages(self, kb_id: str) -> list[_PageRecord]:
        """Walk ``pages/*.md`` and return parsed records."""
        pages_dir = self._pages_dir(kb_id)
        if not pages_dir.is_dir():
            return []
        records: list[_PageRecord] = []
        for path in sorted(pages_dir.glob("*.md")):
            rec = _parse_page(path)
            if rec is not None:
                records.append(rec)
        return records

    def _build_fts_index(self, kb_id: str) -> sqlite3.Connection:
        """(Re)build an in-memory sqlite FTS5 index for ``kb_id``.

        Connections are built with ``check_same_thread=False`` so the
        same connection can be reused from a different thread (the MCP
        tool-timeout worker thread, for example). Concurrent execute()
        calls on a single sqlite connection are still racy, so callers
        MUST hold ``self._fts_lock`` for the entire read/write critical
        section. :meth:`_get_fts_conn` and the search path below already
        do this; tests that drop the lock are responsible for
        re-acquiring it before issuing further sqlite calls.
        """
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE VIRTUAL TABLE pages USING fts5("
            "slug, page_id, source_id, source_type, uri, title, content"
            ")"
        )
        for rec in self._read_all_pages(kb_id):
            conn.execute(
                "INSERT INTO pages(slug, page_id, source_id, source_type, "
                "uri, title, content) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    rec.slug,
                    rec.page_id,
                    rec.source_id,
                    rec.source_type,
                    rec.uri or "",
                    rec.title,
                    rec.body,
                ),
            )
        conn.commit()
        return conn

    def _get_fts_conn(self, kb_id: str) -> sqlite3.Connection:
        """Return cached FTS5 conn, building lazily on first use.

        MUST be called with ``self._fts_lock`` held — the lock spans
        both the cache lookup and the build path so a second thread
        racing on the same ``kb_id`` doesn't construct two parallel
        in-memory indexes. The lock also keeps the returned connection
        from being concurrently mutated by another thread for the
        duration of the caller's sqlite operations.
        """
        conn = self._fts_indexes.get(kb_id)
        if conn is None:
            conn = self._build_fts_index(kb_id)
            self._fts_indexes[kb_id] = conn
        return conn

    # ------------------------------------------------------------------
    # RetrieverBackend Protocol — read paths
    # ------------------------------------------------------------------

    def query(
        self,
        *,
        kb_id: str,
        text: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Vector path re-routes to :meth:`search` (no embedding here).

        Per Protocol docstring: "Backends without a vector path
        (markdown) re-route through :meth:`search` and rescale scores."
        Scores returned by :meth:`search` are already in ``[0, 1]``
        descending so no second rescale is needed.

        Phase 4 read-fallback: when the markdown layout is missing for
        ``kb_id`` but a chromadb collection exists on disk
        (``<saves_dir>/<kb-name>/chroma/``), this call is silently
        served by ChromadbBackend so an existing v0.6.0 KB ingested
        under chromadb stays queryable even after
        ``Settings.kb_backend`` flips to markdown. A structured
        ``phase='read_fallback'`` line is emitted to stderr so
        operators see the divergence.
        """
        fallback = self._chromadb_read_fallback(kb_id)
        if fallback is not None:
            return fallback.query(
                kb_id=kb_id, text=text, top_k=top_k, filters=filters
            )
        return self.search(
            kb_id=kb_id, text=text, top_k=top_k, filters=filters
        )

    def _has_wiki_layout(self, kb_id: str) -> bool:
        """Return True when ``<kb-name>/wiki/pages/`` exists with files."""
        pages_dir = self._pages_dir(kb_id)
        if not pages_dir.is_dir():
            return False
        for _ in pages_dir.glob("*.md"):
            return True
        return False

    def _has_chromadb_collection(self, kb_id: str) -> bool:
        """Return True when a chromadb collection lives on disk for *kb_id*.

        Looks under ``<saves_dir>/<kb-name>/chroma/`` (the conventional
        path used by :meth:`Settings.kb_chroma_path`); the spec also
        accepts the ``vector_store/`` alias for forward compatibility
        with potential future renaming.
        """
        kb_root = self._kb_root(kb_id)
        return (kb_root / "chroma").is_dir() or (kb_root / "vector_store").is_dir()

    def _chromadb_read_fallback(self, kb_id: str):
        """Return a :class:`ChromadbBackend` to serve a fallback read.

        Returns ``None`` when no fallback is needed (either the wiki
        layout is present so markdown can serve, OR there is no
        chromadb collection on disk so falling back would also fail).
        Otherwise constructs a ChromadbBackend bound to the same
        service, emits a structured ``read_fallback`` stderr line, and
        returns it. spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b
        """
        if self._has_wiki_layout(kb_id):
            return None
        if not self._has_chromadb_collection(kb_id):
            return None
        # Lazy import to avoid the circular path at module load.
        from agent_knowledgebase.backends.chromadb_backend import ChromadbBackend
        from agent_knowledgebase.services.stderr_log import knowledgebase_stderr_log

        knowledgebase_stderr_log(
            kb_id=kb_id,
            op="kb_query",
            phase="read_fallback",
            elapsed_ms=0,
            rows_in=0,
            rows_ok=0,
            rows_skipped=0,
            rows_failed=0,
            dedup_policy="n/a",
            request_id=None,
            tool_caller_version=None,
            error_code="MARKDOWN_READ_FALLBACK_TO_CHROMADB",
            error_message=(
                f"kb_backend=markdown but wiki/ missing for kb_id={kb_id!r}; "
                "serving query via chromadb fallback. Run kb_migrate(kb_id="
                f"{kb_id!r}, target_backend='markdown') to materialise the "
                "wiki/ layout."
            ),
        )
        return ChromadbBackend(self._settings, service=self._service)

    def search(
        self,
        *,
        kb_id: str,
        text: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """FTS5 MATCH against page content.

        Returns rows shaped like
        :class:`agent_knowledgebase.services.query.SearchResult`'s
        ``__dict__`` (``content``, ``source_id``, ``source_type``,
        ``score``, ``metadata``) so the existing service-level
        ``SearchResult(**row)`` round-trip in
        :meth:`KnowledgebaseService.query` keeps working.

        ``filters`` is honored only for the keys ``source_type`` and
        ``slug``; other keys are silently ignored (markdown has no
        general filter pushdown). FTS5 ``rank`` is converted to a
        ``[0, 1]`` descending score via ``score = 1 / (1 + abs(rank))``.

        Concurrent ``search()`` calls on the same ``kb_id`` serialize
        on ``self._fts_lock`` — see the module docstring for the
        threading guarantee.

        Phase 4 read-fallback: same semantics as :meth:`query` — if
        ``<kb-name>/wiki/`` is missing while ``<kb-name>/chroma/``
        exists, the ChromadbBackend serves the search and a structured
        stderr line is emitted.
        """
        fallback = self._chromadb_read_fallback(kb_id)
        if fallback is not None:
            return fallback.search(
                kb_id=kb_id, text=text, top_k=top_k, filters=filters
            )
        if top_k <= 0:
            return []
        cleaned = _sanitize_fts_query(text)
        if not cleaned:
            return []
        # Hold the lock for both _get_fts_conn (cache mutation) and the
        # subsequent sqlite execute() — a single sqlite3.Connection is
        # NOT safe under concurrent execute() even with
        # check_same_thread=False, and the cache may be invalidated by
        # a parallel index()/delete() between the two calls.
        with self._fts_lock:
            conn = self._get_fts_conn(kb_id)
            try:
                rows = conn.execute(
                    "SELECT slug, page_id, source_id, source_type, uri, "
                    "title, content, rank FROM pages WHERE pages MATCH ? "
                    "ORDER BY rank LIMIT ?",
                    (cleaned, top_k),
                ).fetchall()
            except sqlite3.OperationalError:
                # Empty in-memory index or malformed query syntax —
                # return an empty result rather than blowing up the
                # MCP call.
                return []

        results: list[dict[str, Any]] = []
        for row in rows:
            if filters:
                st_filter = filters.get("source_type")
                slug_filter = filters.get("slug")
                if st_filter is not None and row["source_type"] != st_filter:
                    continue
                if slug_filter is not None and row["slug"] != slug_filter:
                    continue
            score = 1.0 / (1.0 + abs(float(row["rank"])))
            results.append(
                {
                    "content": row["content"],
                    "source_id": row["source_id"],
                    "source_type": row["source_type"],
                    "score": score,
                    "metadata": {
                        "slug": row["slug"],
                        "page_id": row["page_id"],
                        "source_type": row["source_type"],
                        "uri": row["uri"] or None,
                        "title": row["title"],
                        "backend": _BACKEND_NAME,
                    },
                }
            )
        return results

    def delete(
        self,
        *,
        kb_id: str,
        ids: list[str] | None = None,
        source_id: str | None = None,
    ) -> None:
        """Delete pages by ``page_id`` (slug-derived) or by source_id.

        At least one of ``ids`` or ``source_id`` must be supplied
        (mirrors :class:`ChromadbBackend.delete`).

        Concurrent ``delete()`` calls on the same ``kb_id`` serialize
        on ``self._fts_lock`` — same threading guarantee as
        :meth:`index` and :meth:`search`.
        """
        if ids is None and source_id is None:
            raise ValueError(
                "MarkdownWikiBackend.delete requires at least one of "
                "ids=... or source_id=..."
            )
        pages_dir = self._pages_dir(kb_id)
        if not pages_dir.is_dir():
            return
        target_ids: set[str] = set(ids or [])
        # Hold the lock across the page scan + unlink + index rebuild
        # + cache invalidation. See index() for the rationale on why
        # the rebuild step in particular MUST be serialized (Windows
        # PermissionError on parallel os.replace onto index.md).
        with self._fts_lock:
            deleted_any = False
            for path in pages_dir.glob("*.md"):
                rec = _parse_page(path)
                if rec is None:
                    continue
                if rec.page_id in target_ids or (
                    source_id is not None and rec.source_id == source_id
                ):
                    path.unlink()
                    # Also clean the raw dump (best-effort).
                    for raw in self._raw_dir(kb_id).glob(f"{rec.source_id}.*"):
                        raw.unlink()
                    deleted_any = True
            if deleted_any:
                self._rebuild_index_md(kb_id)
                self._fts_indexes.pop(kb_id, None)

    # ------------------------------------------------------------------
    # RetrieverBackend Protocol — metadata / health
    # ------------------------------------------------------------------

    def info(self, *, kb_id: str) -> dict[str, Any]:
        """Backend health + per-KB metadata.

        Returns the v0.6.0 probe-4 contract fields PLUS markdown-backend
        diagnostics. Inapplicable embedding fields are ``None`` (NOT
        omitted) per validation finding f-20.

        Probe-4 fields (``source_type``, ``uri``, ``dedup_key``,
        ``page_id``) reflect the **slug-alphabetical-first** page in
        ``pages/`` (the first entry returned by
        :meth:`_read_all_pages`, which sorts the ``pages/*.md`` glob
        ascending). This is **deterministic but arbitrary** — the
        first page by slug ordering, not the first page by ingest
        time, not the most-queried page. Operators wanting KB-level
        summary statistics (page count by source_type, recent ingest
        timestamps, etc) should wait for the future Phase 4
        ``kb_info`` aggregates; the current ``info()`` shape is pinned
        to the probe-4 contract for backward compatibility.
        """
        pages = self._read_all_pages(kb_id)
        first = pages[0] if pages else None
        return {
            # Probe-4 frozen keys (None when inapplicable).
            "source_type": first.source_type if first else None,
            "uri": first.uri if first else None,
            "dedup_key": first.dedup_key if first else None,
            "page_id": first.page_id if first else None,
            "dominant_embedding_model": None,  # markdown does not embed
            # Backend diagnostics (additive — not used by probe-4).
            "backend": _BACKEND_NAME,
            "spec_version": _SPEC_VERSION,
            "page_count": len(pages),
            "vector_count": None,
            "embedding_provider": None,
            "embedding_model_counts": {},
            "sharded_index": self._index_dir(kb_id).is_dir(),
        }

    def count(self, *, kb_id: str) -> int:
        """Return the page count (markdown's analogue of chunk count)."""
        return len(self._read_all_pages(kb_id))

    def health_check(self) -> dict[str, Any]:
        """Backend-wide health probe (no kb_id).

        Returns ``{"backend": "markdown", "status": "ok",
        "spec_version": "2.1", "fts5_available": True|False}``. The
        ``fts5_available`` flag round-trips a tiny in-memory CREATE
        VIRTUAL TABLE so misconfigured Python builds without the FTS5
        sqlite extension surface as ``status="degraded"``.
        """
        fts5_available = _probe_fts5_available()
        return {
            "backend": _BACKEND_NAME,
            "status": "ok" if fts5_available else "degraded",
            "spec_version": _SPEC_VERSION,
            "fts5_available": fts5_available,
        }


# ---------------------------------------------------------------------------
# Module-level helpers (free functions; no backend state)
# ---------------------------------------------------------------------------


def _stable_id(content: Any) -> str:
    """Deterministic id derived from content hash — used when caller
    omits ``id``.
    """
    if isinstance(content, (bytes, bytearray)):
        digest = hashlib.sha256(bytes(content)).hexdigest()
    else:
        digest = hashlib.sha256(str(content).encode("utf-8")).hexdigest()
    return f"src-{digest[:16]}"


def _derive_title(uri: str, content: Any) -> str:
    """Fallback title derivation when caller didn't supply one."""
    if uri:
        # Trim querystring + fragment; take last non-empty path component.
        clean = uri.split("?", 1)[0].split("#", 1)[0].rstrip("/")
        tail = clean.rsplit("/", 1)[-1]
        if tail:
            return tail
    if isinstance(content, (bytes, bytearray)):
        try:
            preview = content[:80].decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 — best-effort fallback
            preview = ""
    else:
        preview = str(content)[:80]
    preview = preview.strip().splitlines()[0] if preview.strip() else ""
    return preview or "untitled"


def _raw_ext(source_type: str) -> str:
    """File extension for the raw dump."""
    return {
        "file": "txt",
        "website": "html",
        "api_endpoint": "json",
    }.get(source_type, "txt")


def _render_page_body(*, source_type: str, content: Any, uri: str) -> str:
    """Deterministic source -> page-body transformer (MVP).

    No LLM — keeps Phase 3 scope contained. The spec's "out-of-band LLM
    page-extraction" path is deferred (see ``docs/markdown_backend.md``).
    """
    if isinstance(content, (bytes, bytearray)):
        try:
            text_content = content.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            text_content = ""
    else:
        text_content = str(content)

    if source_type == "website":
        # Try trafilatura for HTML -> plain text extraction; fall back
        # to the raw bytes when trafilatura isn't installed or fails.
        try:
            import trafilatura  # type: ignore[import-untyped]

            extracted = trafilatura.extract(text_content) or text_content
        except Exception:  # noqa: BLE001 — best-effort extraction
            extracted = text_content
        return f"# {uri or 'website'}\n\n{extracted}\n"

    if source_type == "api_endpoint":
        return (
            f"# {uri or 'api_endpoint'}\n\n"
            f"```json\n{text_content}\n```\n"
        )

    # Default (file or unknown): treat content as plain text.
    return f"# {uri or 'page'}\n\n{text_content}\n"


_FTS_FORBID_RE = re.compile(r'[\"\']')


def _sanitize_fts_query(text: str) -> str:
    """Strip characters that confuse FTS5 MATCH parsing.

    FTS5 treats unescaped quotes specially; for the MVP we strip them
    and pass the cleaned string as a single quoted phrase so callers
    don't have to think about FTS5 syntax. Empty result returns ``""``.
    """
    cleaned = _FTS_FORBID_RE.sub(" ", text or "").strip()
    if not cleaned:
        return ""
    # Wrap in double quotes so FTS5 treats the entire string as a phrase
    # and operators like ``-`` or ``OR`` inside user text don't change
    # behaviour unexpectedly.
    return f'"{cleaned}"'


def _probe_fts5_available() -> bool:
    """Return ``True`` if the local sqlite build supports FTS5."""
    try:
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE VIRTUAL TABLE _t USING fts5(x)")
            return True
        finally:
            conn.close()
    except sqlite3.OperationalError:
        return False
    except Exception:  # noqa: BLE001 — degraded path
        return False


# ---------------------------------------------------------------------------
# Path-safety + atomic-write helpers
# ---------------------------------------------------------------------------


def _sanitize_source_id(raw: str, *, content: Any) -> str:
    """Sanitize a caller-supplied ``source_id`` for safe use as a filename.

    Routes through :func:`sanitize_kb_dir_name` so a hostile or
    malformed ``source_id`` (``"../escape"``, ``"a/b/c"``, embedded
    control chars) cannot escape the ``raw/`` directory or collide
    with a different doc by walking upward via ``..``. If the
    sanitizer falls back to its empty-input default ``"unnamed"``
    AND the original ``raw`` was non-empty (e.g. ``raw="../"`` →
    sanitizer strips everything → ``"unnamed"``), we substitute a
    content-derived stable id so two distinct hostile inputs don't
    collide on the same file. An originally-empty ``raw`` already
    routes through :func:`_stable_id` upstream so the ``"unnamed"``
    branch only applies to non-empty hostile input.
    """
    safe = sanitize_kb_dir_name(raw)
    if safe == "unnamed" and raw and raw != "unnamed":
        return _stable_id(content)
    return safe


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomically write ``content`` (utf-8) to ``path``.

    Writes to a same-directory tempfile then ``os.replace()`` it onto
    the target so a crash mid-write never produces a half-written
    file. Caller is responsible for ensuring ``path.parent`` exists
    (the wiki layout calls ``_ensure_layout`` up-front so this is
    already guaranteed for ingest writes).
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp_path, path)
    except Exception:
        # Best-effort cleanup of the tempfile on failure — propagate
        # the original exception either way.
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    """Atomically write raw bytes to ``path`` (binary analogue of
    :func:`_atomic_write_text`).
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(content)
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


__all__ = [
    "MarkdownWikiBackend",
    "KbIngestorUnsuitableError",
]
