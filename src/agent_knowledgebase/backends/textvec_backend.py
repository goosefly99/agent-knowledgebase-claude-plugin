"""SQLite FTS5+BM25 :class:`RetrieverBackend` (Phase B deliverable, OPT-IN).

spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b (v2.2)
Source: pipeline_mcp_data/specs/agent-kb-textvec-spec-v2.2.json
        Phase B — TextvecBackend opt-in (FTS5+BM25 over chunks)

This module is the Phase B deliverable: a SQLite FTS5+BM25 implementation of
the :class:`agent_knowledgebase.backends.RetrieverBackend` Protocol. Opt-in:
``Settings.kb_backend`` defaults to ``'chromadb'``; only
``AGENT_KB_BACKEND=textvec`` routes through this code.

Architecture
------------

``chunks_fts`` is a **contentless** FTS5 virtual table declared in
``database.py`` as part of the per-KB SQLite schema bootstrap
(``CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(...)``).
It is synced to the ``chunks`` table via AFTER-INSERT / AFTER-UPDATE /
AFTER-DELETE triggers also declared in ``database.py``. TextvecBackend
writes through the existing ``chunks`` table; the triggers populate
``chunks_fts`` automatically.

Retrieval
---------

* **Tokenizer:** ``porter unicode61`` — English stemming + Unicode
  normalisation. Configurable via ``Settings.fts5_tokenizer``.
* **Ranker:** BM25 (FTS5 default; ``ORDER BY rank ASC`` — lower is
  better in FTS5 rank semantics).
* **Filter composition:** symbolic metadata filters (``kb_id``,
  ``source_type``, ``dedup_key``) compose with full-text MATCH via SQL
  ``AND`` using parameter binding — NEVER string interpolation.
  SQL injection prevention is a frozen contract.

Probe-4 contract
----------------

``info(kb_id=...)`` returns AT LEAST the v0.6.0 keys::

    source_type, uri, dedup_key, page_id, dominant_embedding_model

For TextvecBackend ``dominant_embedding_model`` is ALWAYS
``'lexical-fts5'`` — the backend is lexical, not vector-based. All
embedding-related chunk columns (``embedding_provider``,
``embed_base_url``, ``embedder_version``, ``embedding_id``) are
written as NULL at ingest time.

Filter validation
-----------------

``_validate_filter_dict`` is copied verbatim from
``chromadb_backend.py`` rather than imported. The decision: both
backends enforce structurally identical filter contracts (non-empty
string keys, primitive scalar values), but the error messages reference
the specific backend by name to aid debugging. If the contract
diverges in the future (e.g. Chromadb adds operator filters like
``{$gt: 5}`` while textvec stays simple-equality), having separate
copies avoids a shared-code coupling point. The trade-off acknowledged
here is a ~10-line duplication — acceptable for the isolation benefit.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

from agent_knowledgebase.config import sanitize_kb_dir_name
from agent_knowledgebase.services.stderr_log import knowledgebase_stderr_log

if TYPE_CHECKING:
    from agent_knowledgebase.config import Settings
    from agent_knowledgebase.services.knowledgebase import KnowledgebaseService


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_BACKEND_NAME = "textvec"
_SPEC_VERSION = "2.2"
_SPEC_ID = "70ab2170-381a-4657-bcd1-28a40c6f369b"

# Scalar primitive types accepted as filter values.
# See _validate_filter_dict docstring and module-level "Filter validation"
# note above for the rationale on copying vs. importing from chromadb_backend.
_FILTER_ALLOWED_VALUE_TYPES = (str, int, float, bool)

# FTS5-special characters that may cause parse errors in MATCH expressions.
# Strip quotes, FTS5 operators, and punctuation that could be misinterpreted.
# We use AND-of-tokens semantics (FTS5 default when no operator given),
# so we strip anything that would be parsed as an operator.
_FTS_FORBID_RE = re.compile(r"[\"\'()\-\*\^]")

# Metadata columns in the chunks table that are supported as SQL equality
# filters (composable with MATCH). This is a frozen allow-list; free-form
# column injection is never permitted.
_ALLOWED_FILTER_COLUMNS: frozenset[str] = frozenset(
    {"source_id", "source_type", "dedup_key", "kb_id", "embedding_provider"}
)


# ---------------------------------------------------------------------------
# Filter validation (copied from chromadb_backend.py — see module docstring)
# ---------------------------------------------------------------------------


def _validate_filter_dict(filters: dict[str, Any]) -> None:
    """Validate a textvec filter dict for structural safety.

    Accepts only non-empty string keys and primitive scalar values
    (str / int / float / bool). Nested dicts, lists, and callables are
    rejected to prevent type-mismatch errors at SQL bind time and to
    surface clear error messages at the backend boundary.

    Raises
    ------
    ValueError
        If any key is not a non-empty string.
    TypeError
        If any value is not a primitive scalar type.
    """
    for key, value in filters.items():
        if not isinstance(key, str) or not key:
            raise ValueError(
                f"TextvecBackend filter keys must be non-empty strings; "
                f"got key {key!r} of type {type(key).__name__!r}"
            )
        if not isinstance(value, _FILTER_ALLOWED_VALUE_TYPES):
            raise TypeError(
                f"TextvecBackend filter values must be str / int / float / bool; "
                f"key {key!r} has value of type {type(value).__name__!r}: {value!r}. "
                f"Nested dicts, lists, and callables are not allowed."
            )


# ---------------------------------------------------------------------------
# SQL query helpers
# ---------------------------------------------------------------------------


def _sanitize_fts_query(text: str) -> str:
    """Strip FTS5-special characters from a query string.

    FTS5 special characters (double/single quotes) are stripped because
    they change tokenization semantics. The result is passed as a raw
    FTS5 MATCH expression — NOT wrapped in phrase quotes — so multi-word
    queries match documents where the tokens appear in ANY order (FTS5
    default AND logic). This gives better recall than phrase matching for
    natural language queries.

    Returns ``""`` on empty input (caller must check before executing SQL).
    """
    cleaned = _FTS_FORBID_RE.sub(" ", text or "").strip()
    return cleaned


def _build_filter_sql(
    filters: dict[str, Any] | None,
    *,
    kb_id: str,
    extra_params: list[Any],
) -> str:
    """Build the WHERE clause suffix for a chunks-table filter.

    Always includes ``chunks.kb_id = ?`` (pre-injected as the first
    ``extra_params`` entry) for per-KB isolation. Caller-supplied
    ``filters`` compose as additional ``AND chunks.<key> = ?`` terms.

    Only columns in :data:`_ALLOWED_FILTER_COLUMNS` are permitted;
    unknown keys raise :exc:`ValueError` to prevent free-form SQL
    injection (double-guarded: the caller also ran ``_validate_filter_dict``
    which rejects non-string keys).

    Parameters
    ----------
    filters:
        Optional dict already validated by ``_validate_filter_dict``.
    kb_id:
        The KB's id, always injected first.
    extra_params:
        Mutable list. This function appends bind values in the order
        they appear in the returned WHERE suffix.

    Returns
    -------
    str
        SQL fragment like ``" AND chunks.kb_id = ? AND chunks.source_type = ?"``.
    """
    # kb_id is always the first positional param in the query, so it's
    # already bound by the caller; we only build the suffix for additional filters.
    parts: list[str] = []
    if filters:
        for key, value in filters.items():
            if key == "kb_id":
                # kb_id is already baked into every query; skip to avoid
                # double-binding or a contradictory filter.
                continue
            if key not in _ALLOWED_FILTER_COLUMNS:
                raise ValueError(
                    f"TextvecBackend: unknown filter column {key!r}; "
                    f"allowed: {sorted(_ALLOWED_FILTER_COLUMNS)}"
                )
            parts.append(f"chunks.{key} = ?")
            extra_params.append(value)
    return (" AND " + " AND ".join(parts)) if parts else ""


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class TextvecBackend:
    """SQLite FTS5+BM25 :class:`RetrieverBackend` (Phase B, opt-in).

    Uses the per-KB ``knowledgebase.db`` SQLite file, which already
    stores the ``chunks`` table populated by the ingest pipeline.
    ``chunks_fts`` (declared in ``database.py``) is a contentless FTS5
    virtual table synced to ``chunks`` via triggers; this backend
    writes to ``chunks`` and reads through ``chunks_fts``.

    Construction parameters
    -----------------------
    settings:
        The resolved :class:`Settings` instance. Used to locate the
        per-KB SQLite file via ``Settings.kb_db_path(dir_name)``.
    service:
        Optional back-reference to the owning
        :class:`KnowledgebaseService`. When supplied, the per-KB
        directory name is resolved via ``service._index[kb_id]``;
        otherwise :func:`sanitize_kb_dir_name` is used (safe fallback
        for service-less unit tests).
    """

    def __init__(
        self,
        settings: "Settings",
        *,
        service: "KnowledgebaseService | None" = None,
    ) -> None:
        self._settings = settings
        self._service = service
        # Per-kb_id connection cache.  The same thread-safety rationale
        # as MarkdownWikiBackend applies: build connections with
        # check_same_thread=False and serialize all sqlite calls through
        # _conn_lock.
        self._conn_lock: threading.Lock = threading.Lock()
        self._conns: dict[str, sqlite3.Connection] = {}

    # ------------------------------------------------------------------
    # Filesystem layout helpers
    # ------------------------------------------------------------------

    def _db_path(self, kb_id: str) -> Path:
        """Return the SQLite path for ``kb_id``.

        Mirrors ``Settings.kb_db_path(dir_name)`` but resolves the
        directory name the same way the service does (``service._index``
        first; falls back to ``sanitize_kb_dir_name``).
        """
        if self._service is not None:
            dir_name = self._service._index.get(kb_id)  # noqa: SLF001 — by design
            if dir_name is None:
                dir_name = sanitize_kb_dir_name(kb_id)
        else:
            dir_name = sanitize_kb_dir_name(kb_id)
        return self._settings.kb_db_path(dir_name)

    @contextmanager
    def _connect(self, kb_id: str) -> Generator[sqlite3.Connection, None, None]:
        """Yield a cached sqlite3.Connection for *kb_id*, thread-safe.

        Connections are built with ``check_same_thread=False`` and
        WAL mode so concurrent readers don't block the writer thread.
        All callers must hold ``_conn_lock`` — this context manager
        acquires it for the duration of the ``with`` block.
        """
        with self._conn_lock:
            conn = self._conns.get(kb_id)
            if conn is None:
                db_path = self._db_path(kb_id)
                conn = sqlite3.connect(str(db_path), check_same_thread=False)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA foreign_keys=ON")
                self._conns[kb_id] = conn
            yield conn

    # ------------------------------------------------------------------
    # RetrieverBackend Protocol — write path
    # ------------------------------------------------------------------

    def index(
        self,
        *,
        kb_id: str,
        documents: list[dict[str, Any]],
    ) -> None:
        """Write chunk rows to the ``chunks`` table.

        The contentless ``chunks_fts`` table is populated automatically
        by the AFTER-INSERT trigger declared in ``database.py``. This
        method does NOT invoke any embedder — embedding columns
        (``embedding_provider``, ``embed_base_url``, ``embedder_version``,
        ``embedding_id``) are ALL written as NULL.

        Each ``documents`` entry must carry at least ``id`` and
        ``content``. ``metadata`` (a dict), ``source_id`` (str), and
        ``kb_id`` (str) are extracted from the dict when present.

        Empty ``documents`` is a no-op (mirrors ChromadbBackend.index).
        """
        if not documents:
            return

        knowledgebase_stderr_log(
            kb_id=kb_id,
            op="textvec_index",
            phase="index",
            elapsed_ms=0,
            rows_in=len(documents),
            rows_ok=0,
            rows_skipped=0,
            rows_failed=0,
            dedup_policy="n/a",
            request_id=None,
            tool_caller_version=None,
            error_code=None,
            error_message=None,
        )

        import json as _json

        with self._connect(kb_id) as conn:
            for doc in documents:
                meta = doc.get("metadata") or {}
                source_id = doc.get("source_id") or meta.get("source_id") or ""
                doc_kb_id = doc.get("kb_id") or meta.get("kb_id") or kb_id
                conn.execute(
                    "INSERT OR IGNORE INTO chunks "
                    "(id, source_id, kb_id, content, metadata, "
                    "embedding_id, embedding_provider, embed_base_url, embedder_version) "
                    "VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, NULL)",
                    (
                        doc["id"],
                        source_id,
                        doc_kb_id,
                        doc.get("content", ""),
                        _json.dumps(meta) if meta else None,
                    ),
                )
            conn.commit()

        knowledgebase_stderr_log(
            kb_id=kb_id,
            op="textvec_index",
            phase="index",
            elapsed_ms=0,
            rows_in=len(documents),
            rows_ok=len(documents),
            rows_skipped=0,
            rows_failed=0,
            dedup_policy="n/a",
            request_id=None,
            tool_caller_version=None,
            error_code=None,
            error_message=None,
        )

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
        """BM25/FTS5 query — routes through the same ``chunks_fts`` MATCH
        path as :meth:`search`.

        TextvecBackend has no vector path; ``query()`` is semantically
        identical to ``search()`` (BM25 over text tokens). Per the
        Protocol docstring: "Backends without a vector path re-route
        through search() and rescale scores."

        Raises
        ------
        ValueError / TypeError
            From ``_validate_filter_dict`` if ``filters`` is malformed.
        """
        return self.search(kb_id=kb_id, text=text, top_k=top_k, filters=filters)

    def search(
        self,
        *,
        kb_id: str,
        text: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """FTS5 MATCH + BM25 rank against ``chunks_fts``.

        SQL shape::

            SELECT chunks.* FROM chunks
            JOIN chunks_fts ON chunks.rowid = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
              AND chunks.kb_id = ?
              [AND chunks.<filter_key> = ? ...]
            ORDER BY rank ASC
            LIMIT ?

        ``rank`` is FTS5's built-in BM25 score (lower = better match;
        ``ASC`` returns best results first). Results are converted to
        ``[0, 1]`` descending scores via ``score = 1 / (1 + abs(rank))``.

        Raises
        ------
        ValueError / TypeError
            From ``_validate_filter_dict`` if ``filters`` is malformed.
        """
        if filters is not None:
            _validate_filter_dict(filters)

        if top_k <= 0:
            return []

        cleaned = _sanitize_fts_query(text)
        if not cleaned:
            return []

        # Build params list: [match_expr, kb_id, ...filter_values..., top_k]
        extra_params: list[Any] = []
        filter_suffix = _build_filter_sql(filters, kb_id=kb_id, extra_params=extra_params)

        sql = (
            "SELECT chunks.id, chunks.source_id, chunks.kb_id, "
            "chunks.content, chunks.metadata, rank "
            "FROM chunks "
            "JOIN chunks_fts ON chunks.rowid = chunks_fts.rowid "
            f"WHERE chunks_fts MATCH ? AND chunks.kb_id = ?{filter_suffix} "
            "ORDER BY rank ASC LIMIT ?"
        )
        params: list[Any] = [cleaned, kb_id, *extra_params, top_k]

        import json as _json

        results: list[dict[str, Any]] = []
        try:
            with self._connect(kb_id) as conn:
                rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            # FTS5 table missing or malformed query — return empty.
            return []

        for row in rows:
            rank = float(row["rank"])
            score = 1.0 / (1.0 + abs(rank))
            meta = {}
            if row["metadata"]:
                try:
                    meta = _json.loads(row["metadata"])
                except (ValueError, TypeError):
                    meta = {}
            results.append(
                {
                    "content": row["content"],
                    "source_id": row["source_id"],
                    "source_type": meta.get("source_type"),
                    "score": score,
                    "metadata": {
                        **meta,
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
        """Delete chunks by explicit id list or by source_id.

        The AFTER-DELETE trigger on ``chunks`` fires the FTS5 cleanup
        automatically — no manual ``chunks_fts`` delete is needed.

        At least one of ``ids`` or ``source_id`` must be supplied.
        """
        if ids is None and source_id is None:
            raise ValueError(
                "TextvecBackend.delete requires at least one of ids=... or source_id=..."
            )

        target_ids: list[str] = list(ids or [])

        with self._connect(kb_id) as conn:
            if source_id is not None:
                rows = conn.execute(
                    "SELECT id FROM chunks WHERE source_id = ? AND kb_id = ?",
                    (source_id, kb_id),
                ).fetchall()
                target_ids.extend(r["id"] for r in rows)

            if not target_ids:
                return

            # Use parameterized placeholders — never string interpolation.
            placeholders = ",".join("?" * len(target_ids))
            conn.execute(
                f"DELETE FROM chunks WHERE id IN ({placeholders}) AND kb_id = ?",
                [*target_ids, kb_id],
            )
            conn.commit()

    # ------------------------------------------------------------------
    # RetrieverBackend Protocol — metadata / health
    # ------------------------------------------------------------------

    def info(self, *, kb_id: str) -> dict[str, Any]:
        """Backend health + per-KB metadata.

        Probe-4 contract (validation finding f-20):
        Returns AT LEAST ``source_type``, ``uri``, ``dedup_key``,
        ``page_id``, ``dominant_embedding_model``.

        ``dominant_embedding_model`` is always ``'lexical-fts5'`` for
        this backend — it is lexical-only with no vector representation.
        Inapplicable fields (e.g. ``page_id`` — not directly relevant
        for chunk-level FTS) return ``None``, NOT omitted.
        """
        try:
            with self._connect(kb_id) as conn:
                # Probe-4: first source for source_type / uri / dedup_key.
                src_row = conn.execute(
                    "SELECT source_type, uri, dedup_key FROM sources WHERE kb_id = ? LIMIT 1",
                    (kb_id,),
                ).fetchone()
                # page_id: first wiki page (may not exist for textvec KBs).
                page_row = conn.execute(
                    "SELECT id FROM wiki_pages WHERE kb_id = ? LIMIT 1",
                    (kb_id,),
                ).fetchone()
                chunk_count_row = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM chunks WHERE kb_id = ?",
                    (kb_id,),
                ).fetchone()
        except sqlite3.OperationalError:
            src_row = None
            page_row = None
            chunk_count_row = None

        chunk_count = chunk_count_row["cnt"] if chunk_count_row else 0

        return {
            # Probe-4 frozen keys (None when inapplicable).
            "source_type": src_row["source_type"] if src_row else None,
            "uri": src_row["uri"] if src_row else None,
            "dedup_key": src_row["dedup_key"] if src_row else None,
            "page_id": page_row["id"] if page_row else None,
            "dominant_embedding_model": "lexical-fts5",
            # Backend diagnostics (additive).
            "backend": _BACKEND_NAME,
            "spec_version": _SPEC_VERSION,
            "chunk_count": chunk_count,
            "vector_count": None,
            "embedding_provider": None,
            "embedding_model_counts": {},
        }

    def count(self, *, kb_id: str) -> int:
        """Return the total chunk count for ``kb_id``."""
        try:
            with self._connect(kb_id) as conn:
                row = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM chunks WHERE kb_id = ?",
                    (kb_id,),
                ).fetchone()
                return row["cnt"] if row else 0
        except sqlite3.OperationalError:
            return 0

    def health_check(self) -> dict[str, Any]:
        """Backend-wide health probe (no kb_id).

        Verifies FTS5 is available in the running Python build by
        attempting to create an in-memory FTS5 table. Returns
        ``status='degraded'`` if the sqlite build lacks FTS5.
        """
        fts5_ok = _probe_fts5_available()
        return {
            "backend": _BACKEND_NAME,
            "status": "ok" if fts5_ok else "degraded",
            "spec_version": _SPEC_VERSION,
            "fts5_available": fts5_ok,
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


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


__all__ = ["TextvecBackend"]
