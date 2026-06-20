"""ChromaDB-backed :class:`RetrieverBackend` (Phase 2 wrapper, zero behavior change).

spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b (v2.1)
Source: pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json
        architecture.components[1] (RetrieverBackend factory)
        + implementation.phases[2] (this Phase 2 deliverable)

ChromadbBackend is a thin wrapper around the existing
``services/query.py``, ``services/vectorstore.py``, ``services/wiki.py``
and ``database.py`` integration.  It does NOT reimplement chromadb
logic — it delegates every call into the existing
:class:`agent_knowledgebase.services.knowledgebase.KnowledgebaseService`
helpers (``_ctx``, ``_get_vectorstore``, ``_query_embedder_for``,
``_embedder``) so the v0.6.0 fast-path stays bit-for-bit identical.

The backend takes a back-reference to the owning
:class:`KnowledgebaseService` at construction time (passed by the
factory in ``backends/__init__.py``).  This intentionally trades a
small circular-reference risk for a much simpler surface than
duplicating the per-KB lifecycle here — the service already manages
SQLite handles, vectorstore caches, per-kb_id locks, and lazy embedder
construction.

Probe-4 contract (validation finding f-20):
    info(kb_id=...) MUST return at least the keys ``source_type``,
    ``uri``, ``dedup_key``, ``page_id``, ``dominant_embedding_model``.
    Inapplicable fields return ``None`` (NOT omitted).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from agent_knowledgebase.services.stderr_log import knowledgebase_stderr_log

if TYPE_CHECKING:
    from agent_knowledgebase.config import Settings
    from agent_knowledgebase.services.knowledgebase import KnowledgebaseService


_BACKEND_NAME = "chromadb"
_SPEC_VERSION = "2.1"

# Scalar primitive types accepted as filter values in a chromadb where= dict.
# Chromadb's Mongo-style filter uses string keys and primitive scalar values;
# nested dicts / lists are not valid scalar filter values in this context.
_FILTER_ALLOWED_VALUE_TYPES = (str, int, float, bool)


def _validate_filter_dict(filters: dict[str, Any]) -> None:
    """Validate a chromadb filter dict for structural safety.

    Chromadb's ``where=`` parameter accepts a Mongo-style dict whose keys
    are metadata field names (strings) and whose values are primitive
    scalars (str / int / float / bool). This validator enforces:

    * All keys must be non-empty strings.
    * All values must be primitive scalars — nested dicts, lists, or
      callables are rejected to prevent schema-mismatch errors downstream
      and to provide a clear error message at the backend boundary.

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
                f"ChromadbBackend filter keys must be non-empty strings; "
                f"got key {key!r} of type {type(key).__name__!r}"
            )
        if not isinstance(value, _FILTER_ALLOWED_VALUE_TYPES):
            raise TypeError(
                f"ChromadbBackend filter values must be str / int / float / bool; "
                f"key {key!r} has value of type {type(value).__name__!r}: {value!r}. "
                f"Nested dicts, lists, and callables are not allowed."
            )


class ChromadbBackend:
    """Wrap the v0.6.0 chromadb retrieval pipeline behind the
    :class:`agent_knowledgebase.backends.RetrieverBackend` Protocol.

    Construction parameters
    -----------------------
    settings:
        The resolved :class:`Settings` instance the owning service was
        built from.  Stored for future per-backend tuning (e.g. the
        chromadb-internal vector-store provider — ``settings.vectorstore``
        — is read by ``services/vectorstore.py::create_vectorstore`` when
        the wrapped service spins up the per-KB store).
    service:
        Back-reference to the owning :class:`KnowledgebaseService`.
        Optional so unit tests can instantiate the backend in isolation
        and assert Protocol conformance without booting the full
        service; methods that touch per-KB resources require it and
        will raise a clear :class:`RuntimeError` if called when
        ``service is None``.
    """

    def __init__(
        self,
        settings: "Settings",
        *,
        service: "KnowledgebaseService | None" = None,
    ) -> None:
        self._settings = settings
        self._service = service

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _require_service(self) -> "KnowledgebaseService":
        """Return the bound service or raise a clear error.

        Tests are allowed to instantiate ``ChromadbBackend(settings)``
        without a service to exercise the Protocol contract; calling
        through a per-KB method in that mode is unsupported and raises
        here so the failure is local to the call instead of a generic
        ``AttributeError`` deep in the service plumbing.
        """
        if self._service is None:
            raise RuntimeError(
                "ChromadbBackend was constructed without a "
                "KnowledgebaseService back-reference; per-kb operations "
                "are unsupported in this mode. Pass service=... when "
                "constructing the backend (the get_backend factory does "
                "this automatically when called from "
                "KnowledgebaseService.__init__)."
            )
        return self._service

    # ------------------------------------------------------------------
    # RetrieverBackend Protocol
    # ------------------------------------------------------------------

    def index(
        self,
        *,
        kb_id: str,
        documents: list[dict[str, Any]],
    ) -> None:
        """Add documents to the per-KB chromadb collection.

        Each ``documents`` entry must carry ``id``, ``content``,
        ``metadata``, and ``embedding`` (a ``list[float]`` produced by
        the same embedder the read path will use).  ``metadata``
        SHOULD include ``kb_id`` and ``embedding_model`` so the v0.6.0
        ``where`` filter and ``count_chunks_by_embedding_model`` path
        keep working — the caller (``KnowledgebaseService.ingest_source``)
        is responsible for stamping these.

        Empty ``documents`` is a no-op (mirrors
        :class:`agent_knowledgebase.services.vectorstore.ChromaDBStore.add`).
        """
        if not documents:
            return
        service = self._require_service()
        vs = service._get_vectorstore(kb_id)  # noqa: SLF001 — by design
        vs.add(
            ids=[doc["id"] for doc in documents],
            embeddings=[doc["embedding"] for doc in documents],
            documents=[doc["content"] for doc in documents],
            metadatas=[doc.get("metadata", {}) for doc in documents],
        )

    def query(
        self,
        *,
        kb_id: str,
        text: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Vector-similarity query — delegates to the existing
        :class:`agent_knowledgebase.services.query.QueryOrchestrator`.

        When ``filters`` is non-None the dict is structurally validated
        (non-empty string keys, primitive scalar values) and then merged
        with the existing ``{"kb_id": kb_id}`` chromadb where-filter so
        callers can narrow results to a specific metadata field without
        bypassing the per-KB isolation filter.

        Raises
        ------
        ValueError
            If any filter key is not a non-empty string.
        TypeError
            If any filter value is not a primitive scalar (str/int/float/bool).
        """
        if filters is not None:
            _validate_filter_dict(filters)

        from agent_knowledgebase.services.query import QueryOrchestrator

        service = self._require_service()
        ctx = service._ctx(kb_id)  # noqa: SLF001 — by design
        vs = service._get_vectorstore(kb_id)  # noqa: SLF001 — by design
        embedder = service._query_embedder_for(kb_id)  # noqa: SLF001

        if filters is not None:
            # start with the caller-supplied filter keys and overlay the kb_id
            # isolation filter last so per-KB isolation cannot be subverted by collision.
            where = {**filters, "kb_id": kb_id}
            embedding = embedder.embed_query(text)
            raw_results = vs.query(embedding, top_k=top_k, where=where)
            from agent_knowledgebase.services.query import SearchResult

            search_results = []
            for qr in raw_results:
                similarity = max(0.0, min(1.0, 1.0 - qr.distance))
                search_results.append(
                    SearchResult(
                        content=qr.document,
                        source_id=qr.id,
                        source_type="chunk",
                        score=similarity,
                        metadata=qr.metadata,
                    )
                )
            return [asdict(r) for r in search_results]

        orchestrator = QueryOrchestrator(vs, embedder, ctx.wiki)
        # QueryOrchestrator.query already injects where={"kb_id": kb_id}.
        results = orchestrator.query(text, kb_id, top_k=top_k)
        return [asdict(r) for r in results]

    def search(
        self,
        *,
        kb_id: str,
        text: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """FTS search over wiki pages — delegates to
        :meth:`QueryOrchestrator.search`.

        ``filters`` MUST be ``None``. The FTS search path runs through the
        wiki index and has no metadata-filter support; passing a non-None
        ``filters`` raises :exc:`ValueError` rather than silently dropping
        the caller's intent.

        Use :meth:`query` with ``filters=`` for vector-search with metadata
        filtering.

        Raises
        ------
        ValueError
            If ``filters`` is not ``None``.
        """
        if filters is not None:
            raise ValueError(
                "ChromadbBackend.search does not honor filters; the wiki/FTS path "
                "has no metadata-filter support. Pass filters=None or use query()."
            )

        from agent_knowledgebase.services.query import QueryOrchestrator

        service = self._require_service()
        ctx = service._ctx(kb_id)  # noqa: SLF001 — by design
        vs = service._get_vectorstore(kb_id)  # noqa: SLF001 — by design
        embedder = service._query_embedder_for(kb_id)  # noqa: SLF001
        orchestrator = QueryOrchestrator(vs, embedder, ctx.wiki)
        results = orchestrator.search(text, kb_id, top_k=top_k)
        return [asdict(r) for r in results]

    def delete(
        self,
        *,
        kb_id: str,
        ids: list[str] | None = None,
        source_id: str | None = None,
    ) -> None:
        """Delete documents by explicit id list or by ``source_id``.

        ``source_id`` resolves to the chunk-id list via the existing
        ``Database.list_chunks`` lookup, mirroring
        :meth:`KnowledgebaseService.remove_source`.  At least one of
        ``ids`` or ``source_id`` must be supplied; supplying both is
        permitted (the union is deleted).
        """
        if ids is None and source_id is None:
            raise ValueError(
                "ChromadbBackend.delete requires at least one of ids=... or source_id=..."
            )
        service = self._require_service()
        ctx = service._ctx(kb_id)  # noqa: SLF001 — by design
        target_ids: list[str] = list(ids or [])
        if source_id is not None:
            chunks = ctx.db.list_chunks(source_id)
            target_ids.extend(c.id for c in chunks)
        if not target_ids:
            return
        vs = service._get_vectorstore(kb_id)  # noqa: SLF001 — by design
        vs.delete(target_ids)

    def info(self, *, kb_id: str) -> dict[str, Any]:
        """Backend health + per-KB metadata.

        Returns the v0.6.0 probe-4 contract fields PLUS backend-tagged
        diagnostics so a future ``kb_health`` tool can introspect.

        Probe-4 keys (validation finding f-20 mitigation):
        ``source_type``, ``uri``, ``dedup_key``, ``page_id``,
        ``dominant_embedding_model``.  Each is sourced from the FIRST
        :class:`Source` returned by ``Database.list_sources(kb_id)``
        (matches the convention used by
        :meth:`KnowledgebaseService.list_pages` for its denormalized
        first-source fields), and from
        ``Database.count_chunks_by_embedding_model(kb_id)`` for
        ``dominant_embedding_model``.

        Field selection rule: when multiple sources exist,
        ``source_type`` / ``uri`` / ``dedup_key`` are populated from
        ``sources[0]`` (the first row returned by
        ``Database.list_sources(kb_id)``, whose ordering follows the
        underlying SQL ``ORDER BY`` clause). Phase 3's
        ``MarkdownBackend.info()`` should match this choice for shape
        consistency.

        Inapplicable fields (e.g. an empty KB has no first source so
        ``source_type`` / ``uri`` / ``dedup_key`` are ``None``) are
        present with value ``None``, NOT omitted.
        """
        service = self._require_service()
        ctx = service._ctx(kb_id)  # noqa: SLF001 — by design

        sources = ctx.db.list_sources(kb_id)
        first = sources[0] if sources else None

        pages = ctx.db.list_wiki_pages(kb_id)
        first_page = pages[0] if pages else None

        model_counts = ctx.db.count_chunks_by_embedding_model(kb_id)
        dominant: str | None = (
            max(model_counts, key=lambda m: model_counts[m]) if model_counts else None
        )

        return {
            # Probe-4 frozen keys (None when inapplicable).
            "source_type": first.source_type.value if first else None,
            "uri": first.uri if first else None,
            "dedup_key": first.dedup_key if first else None,
            "page_id": first_page.id if first_page else None,
            "dominant_embedding_model": dominant,
            # Backend diagnostics (additive — not used by probe-4).
            "backend": _BACKEND_NAME,
            "spec_version": _SPEC_VERSION,
            "source_count": len(sources),
            "page_count": len(pages),
            "embedding_model_counts": dict(model_counts),
        }

    def count(self, *, kb_id: str) -> int:
        """Return the total stored chunk count for ``kb_id``.

        Reads from the per-KB chromadb collection's ``count()`` so the
        result reflects actual vectors on disk (matches what the
        v0.6.0 ``VectorStore.count`` Protocol returns).
        """
        service = self._require_service()
        vs = service._get_vectorstore(kb_id)  # noqa: SLF001 — by design
        return vs.count()

    def health_check(self) -> dict[str, Any]:
        """Backend-wide health probe (no kb_id).

        Returns ``{"backend": "chromadb", "status": "ok",
        "spec_version": "2.1", "vectorstore_provider": <chromadb|pinecone>}``
        so callers can surface whether the chromadb backend is wired
        to a local PersistentClient or a Pinecone namespace under the
        hood.  Phase 4+ may extend with an actual round-trip ping.
        """
        return {
            "backend": _BACKEND_NAME,
            "status": "ok",
            "spec_version": _SPEC_VERSION,
            "vectorstore_provider": getattr(self._settings, "vectorstore", "chromadb"),
        }

    def warmup(self, kb_id: str) -> None:
        """Force-load the HNSW index for *kb_id* into memory.

        Calls ``_get_vectorstore(kb_id)`` to open the chromadb
        collection, then calls ``vs.count()`` which forces the HNSW
        graph to load off disk into RAM. This amortises the ~180 s
        cold-load into server-init time rather than the first tool call.

        Failures are caught and logged via :func:`knowledgebase_stderr_log`
        with ``error_code="EAGER_WARM_FAILED"`` — they are never re-raised
        so a single failing KB does not abort the warmup of the others.

        Phase A — vectorstore robustness.
        spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b
        """
        try:
            service = self._require_service()
            vs = service._get_vectorstore(kb_id)  # noqa: SLF001 — by design
            vs.count()  # forces HNSW graph load
        except Exception as exc:  # noqa: BLE001 — warmup failures must not propagate
            knowledgebase_stderr_log(
                kb_id=kb_id,
                op="eager_warm",
                phase="warmup",
                elapsed_ms=0,
                rows_in=0,
                rows_ok=0,
                rows_skipped=0,
                rows_failed=1,
                dedup_policy="n/a",
                request_id=None,
                tool_caller_version=None,
                error_code="EAGER_WARM_FAILED",
                error_message=f"warmup failed for kb_id={kb_id!r}: {exc}",
            )


__all__ = ["ChromadbBackend"]
