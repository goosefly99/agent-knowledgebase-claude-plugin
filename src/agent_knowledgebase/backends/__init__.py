"""
agent_knowledgebase.backends — Retrieval backend abstraction (SCAFFOLD STUB).

spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b (v2.1)
Source: pipeline_mcp_data/specs/agent-kb-redesign-spec-v2.1.json
        architecture.components[1] "AGENT_KB_BACKEND feature flag + RetrieverBackend protocol + backend factory"

This module is the **single swap-point** introduced by spec v2.1 Phase 2 to
preserve the v0.6.0 `kb_*` MCP tool contract while permitting alternate
retrieval strategies (chromadb / markdown-wiki / lightrag-graph). The 25
frozen MCP tools on `server.py` route through `KnowledgebaseService`, which
selects ONE `RetrieverBackend` once at construction time via
`get_backend(settings)` and stores it as `self._backend`. All retrieval code
paths route through that instance.

The factory dispatches on `settings.kb_backend` (Literal[
'chromadb','markdown','lightrag'], default 'chromadb', loaded from
`AGENT_KB_BACKEND` env or the `kb_backend` JSON config key — disambiguated
from the existing `Settings.vectorstore` field which remains the
chromadb-backend internal vector-store-provider choice).

PHASE 2 SCOPE (this scaffold ships only the Protocol + factory skeleton; the
ChromadbBackend / MarkdownWikiBackend / LightRAGBackend implementations are
Phase 2/3/6 work):

    Phase 2: ChromadbBackend wraps existing services/query.py +
             services/vectorstore.py + services/wiki.py — zero behavior
             change asserted by existing test suite passing unchanged.

    Phase 3: MarkdownWikiBackend reads filesystem directly under
             <saves_dir>/<kb-name>/wiki/, sqlite FTS5 in-memory index over
             pages/*.md, NO embedding. Per-source_type positive-allow list:
             {file, website, api_endpoint with payload <2MB}.

    Phase 6: LightRAGBackend forwards to localhost:9621 REST API. Stub only
             until adoption signal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    # Avoid a hard import at module-load time so the scaffold does not require
    # the full Settings class to be importable in environments that haven't
    # finished Phase 0 bug fixes yet.
    from agent_knowledgebase.config import Settings


_SPEC_ID = "70ab2170-381a-4657-bcd1-28a40c6f369b"
_SPEC_VERSION = "2.1"


@runtime_checkable
class RetrieverBackend(Protocol):
    """
    Protocol every retrieval backend must satisfy.

    Method set deliberately mirrors the existing
    `services/vectorstore.py::VectorStore` Protocol (lines 44-77) plus
    `health_check()` and a clear separation between vector-`query()` and
    keyword-`search()` paths so hybrid retrieval routes correctly through
    each backend.

    Probe-4 contract: every backend's `info()` MUST return a dict containing
    AT LEAST the v0.6.0 keys (source_type, uri, dedup_key, page_id,
    dominant_embedding_model). Inapplicable fields return None — they MUST
    NOT be omitted (validation finding f-20 mitigation).
    """

    def index(  # noqa: D401  (concise verb form per Protocol convention)
        self,
        *,
        kb_id: str,
        documents: list[dict[str, Any]],
    ) -> None:
        """
        Add documents to the backend's index for `kb_id`.

        TODO[Phase 2]: Implement ChromadbBackend.index by delegating to the
                       existing chromadb collection.add(...) call.
        TODO[Phase 3]: Implement MarkdownWikiBackend.index by writing
                       wiki/pages/<slug>.md and updating wiki/index.md
                       (sharded above ~200 articles).
        TODO[Phase 3]: MarkdownWikiBackend.index enforces the per-source_type
                       positive-allow list — raises KB_INGESTOR_UNSUITABLE
                       error unless source_type is in {file, website,
                       api_endpoint with payload <2MB} or
                       AGENT_KB_FORCE_WIKI_INGEST=1 is set.
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
        """
        Vector-similarity query.

        Returns a list of result rows, each containing AT LEAST: page_id,
        score, snippet. Markdown backend returns null for vector-only fields.

        TODO[Phase 2]: ChromadbBackend.query routes through embedder.embed_query
                       + collection.query.
        TODO[Phase 3]: MarkdownWikiBackend.query has no real vector path —
                       falls back to FTS5 search() and rescales scores, OR
                       raises NotImplementedError if vector-only callers
                       cannot accept FTS scores.
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
        """
        Keyword/FTS search.

        TODO[Phase 2]: ChromadbBackend.search uses existing sqlite FTS5 path.
        TODO[Phase 3]: MarkdownWikiBackend.search uses sqlite FTS5 in-memory
                       over pages/*.md content.
        """
        ...

    def delete(
        self,
        *,
        kb_id: str,
        ids: list[str] | None = None,
        source_id: str | None = None,
    ) -> None:
        """
        Delete documents by id list or by source_id.

        TODO[Phase 2]: ChromadbBackend.delete delegates to collection.delete.
        TODO[Phase 3]: MarkdownWikiBackend.delete removes the slug page and
                       updates the index.
        """
        ...

    def info(self, *, kb_id: str) -> dict[str, Any]:
        """
        Backend health + metadata for `kb_id`.

        Probe-4 contract: must return at least the v0.6.0 keys
        (source_type, uri, dedup_key, page_id, dominant_embedding_model).
        Inapplicable fields return None.

        TODO[Phase 2]: ChromadbBackend.info reads from chromadb collection
                       metadata + sqlite pages table.
        TODO[Phase 3]: MarkdownWikiBackend.info returns
                       dominant_embedding_model=None (markdown does not embed)
                       — probe-4 must accept null. Coordination issue filed
                       against data-etl-orchestrator >=v0.4.0 in Phase 2.
        """
        ...


def get_backend(settings: "Settings") -> RetrieverBackend:
    """
    Backend factory. Reads `settings.kb_backend` and dispatches.

    Valid values: 'chromadb' (default), 'markdown', 'lightrag'.

    Disambiguated from the existing `settings.vectorstore` field (which stays
    as the chromadb-backend internal vector-store-provider choice in
    {chromadb, pinecone}). See spec architecture.components[1] for full
    discussion.

    TODO[Phase 2]: Replace the NotImplementedError raises with real backend
                   instantiations. ChromadbBackend(settings) wraps existing
                   logic with zero behavior change.
    TODO[Phase 3]: MarkdownWikiBackend(settings) — Karpathy-style backend.
    TODO[Phase 6]: LightRAGBackend(settings) — HTTP forwarder to
                   localhost:9621.

    The dispatch is deliberately left as a stub so the v0.6.0 chromadb
    fast-path remains working (KnowledgebaseService still constructs the
    chromadb store directly today). Phase 2 promotes the chromadb path
    through this factory; this scaffold only declares the contract.
    """
    backend_name = getattr(settings, "kb_backend", "chromadb")

    if backend_name == "chromadb":
        # TODO[Phase 2]: from .chromadb_backend import ChromadbBackend
        #                return ChromadbBackend(settings)
        raise NotImplementedError(
            "ChromadbBackend is a Phase 2 deliverable; until Phase 2 lands, "
            "KnowledgebaseService constructs the chromadb store directly."
        )

    if backend_name == "markdown":
        # TODO[Phase 3]: from .markdown_backend import MarkdownWikiBackend
        #                return MarkdownWikiBackend(settings)
        raise NotImplementedError(
            "MarkdownWikiBackend is a Phase 3 deliverable. Set "
            "AGENT_KB_BACKEND=chromadb (the default) until Phase 3 lands."
        )

    if backend_name == "lightrag":
        # TODO[Phase 6]: from .lightrag_backend import LightRAGBackend
        #                return LightRAGBackend(settings)
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
