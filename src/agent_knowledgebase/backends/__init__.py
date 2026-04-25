"""
agent_knowledgebase.backends — Retrieval backend abstraction.

spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b (v2.1)
Source: pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json
        architecture.components[1] "AGENT_KB_BACKEND feature flag +
        RetrieverBackend protocol + backend factory"

This module is the single swap-point introduced by spec v2.1 Phase 2 to
preserve the v0.6.0 ``kb_*`` MCP tool contract while permitting alternate
retrieval strategies (chromadb / markdown-wiki / lightrag-graph). The 25
frozen MCP tools on ``server.py`` route through ``KnowledgebaseService``,
which selects ONE :class:`RetrieverBackend` once at construction time via
:func:`get_backend` and stores it as ``self._backend``. All retrieval
code paths route through that instance.

The factory dispatches on ``settings.kb_backend`` (``Literal[
'chromadb','markdown','lightrag']``, default ``'chromadb'``, loaded from
``AGENT_KB_BACKEND`` env or the ``kb_backend`` JSON config key —
disambiguated from the existing ``Settings.vectorstore`` field which
remains the chromadb-internal vector-store-provider choice).

PHASE STATUS

* Phase 2 (this module + ``chromadb_backend.py``): ChromadbBackend wraps
  existing ``services/query.py`` + ``services/vectorstore.py`` +
  ``services/wiki.py`` — zero behavior change asserted by the existing
  v0.6.0 test suite passing unchanged.
* Phase 3: MarkdownWikiBackend reads filesystem directly under
  ``<saves_dir>/<kb-name>/wiki/``, sqlite FTS5 in-memory index over
  ``pages/*.md``, NO embedding. Per-source_type positive-allow list:
  ``{file, website, api_endpoint with payload <2MB}``.
* Phase 6: LightRAGBackend forwards to ``localhost:9621`` REST API.
  Stub only until adoption signal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agent_knowledgebase.config import Settings
    from agent_knowledgebase.services.knowledgebase import KnowledgebaseService


_SPEC_ID = "70ab2170-381a-4657-bcd1-28a40c6f369b"
_SPEC_VERSION = "2.1"


@runtime_checkable
class RetrieverBackend(Protocol):
    """Protocol every retrieval backend must satisfy.

    Method set deliberately mirrors the existing
    ``services/vectorstore.py::VectorStore`` Protocol (lines 44-77) plus
    ``health_check()`` and a clear separation between vector-``query()``
    and keyword-``search()`` paths so hybrid retrieval routes correctly
    through each backend.

    All entry points take ``kb_id`` so a single backend instance owned by
    ``KnowledgebaseService`` can fan out across many KBs without holding
    per-KB state on the backend itself; the backend resolves per-KB
    resources lazily (the Phase 2 chromadb implementation delegates to
    the service's existing ``_KBContext`` cache).

    Probe-4 contract: every backend's :meth:`info` MUST return a dict
    containing AT LEAST the v0.6.0 keys (``source_type``, ``uri``,
    ``dedup_key``, ``page_id``, ``dominant_embedding_model``).
    Inapplicable fields return ``None`` — they MUST NOT be omitted
    (validation finding f-20 mitigation).
    """

    def index(  # noqa: D401  (concise verb form per Protocol convention)
        self,
        *,
        kb_id: str,
        documents: list[dict[str, Any]],
    ) -> None:
        """Add documents to the backend's index for ``kb_id``.

        Each ``documents`` entry is a dict carrying at minimum ``id``,
        ``content``, and ``metadata``; the chromadb backend additionally
        requires an ``embedding`` field (a ``list[float]``). Backends
        that do not embed (markdown, lightrag) ignore the field.
        """
        ...

    def query(
        self,
        *,
        kb_id: str,
        text: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Vector-similarity query.

        Returns a list of result rows shaped like
        :class:`agent_knowledgebase.services.query.SearchResult`'s
        ``__dict__`` (``content``, ``source_id``, ``source_type``,
        ``score``, ``metadata``). Backends without a vector path
        (markdown) re-route through :meth:`search` and rescale scores.
        """
        ...

    def search(
        self,
        *,
        kb_id: str,
        text: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Keyword/FTS search over wiki pages."""
        ...

    def delete(
        self,
        *,
        kb_id: str,
        ids: list[str] | None = None,
        source_id: str | None = None,
    ) -> None:
        """Delete documents by id list or by source_id."""
        ...

    def info(self, *, kb_id: str) -> dict[str, Any]:
        """Backend health + metadata for ``kb_id``.

        Probe-4 contract: must return at least the v0.6.0 keys
        (``source_type``, ``uri``, ``dedup_key``, ``page_id``,
        ``dominant_embedding_model``). Inapplicable fields return
        ``None`` (NOT omitted).
        """
        ...

    def count(self, *, kb_id: str) -> int:
        """Return the total stored document count for ``kb_id``."""
        ...

    def health_check(self) -> dict[str, Any]:
        """Backend-wide health probe (no kb_id).

        Returns at minimum ``{"backend": "<name>", "status":
        "ok"|"degraded"|"unavailable", "spec_version":
        "<version>"}``. Used by future ``kb_health`` MCP tools (Phase 4+).
        """
        ...


def get_backend(
    settings: "Settings",
    *,
    service: "KnowledgebaseService | None" = None,
) -> RetrieverBackend:
    """Backend factory. Reads ``settings.kb_backend`` and dispatches.

    Valid values: ``'chromadb'`` (default), ``'markdown'``,
    ``'lightrag'``.

    Disambiguated from the existing ``settings.vectorstore`` field
    (which stays as the chromadb-internal vector-store-provider choice
    in ``{chromadb, pinecone}``). See spec architecture.components[1]
    for full discussion.

    The optional ``service`` keyword is the
    :class:`KnowledgebaseService` instance the backend will wrap. The
    ChromadbBackend uses it to reach existing per-KB plumbing
    (``_ctx``, ``_get_vectorstore``, ``_query_embedder_for``,
    ``_embedder``, ``_ingestion``) without duplicating that logic.
    Tests can pass ``service=None`` to instantiate the backend in
    isolation; methods that need service plumbing will fail with a
    clear error in that mode.
    """
    backend_name = getattr(settings, "kb_backend", "chromadb")

    if backend_name == "chromadb":
        from .chromadb_backend import ChromadbBackend
        return ChromadbBackend(settings, service=service)

    if backend_name == "markdown":
        from .markdown_backend import MarkdownWikiBackend
        return MarkdownWikiBackend(settings, service=service)

    if backend_name == "lightrag":
        raise NotImplementedError(
            "LightRAGBackend is a deferred Phase 6 deliverable. Set "
            "AGENT_KB_BACKEND=chromadb (the default) — LightRAG support "
            "is conditional on adoption signal."
        )

    raise ValueError(
        f"Unknown AGENT_KB_BACKEND={backend_name!r}; "
        f"valid values: chromadb (default), markdown, lightrag"
    )


__all__ = ["RetrieverBackend", "get_backend"]
