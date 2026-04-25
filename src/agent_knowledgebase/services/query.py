"""Query orchestrator combining vector similarity search with wiki FTS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agent_knowledgebase.models import WikiPage
from agent_knowledgebase.services.embeddings import Embedder
from agent_knowledgebase.services.vectorstore import QueryResult, VectorStore
from agent_knowledgebase.services.wiki import WikiManager

if TYPE_CHECKING:
    from agent_knowledgebase.config import Settings


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """A single search result with source citation."""

    content: str
    source_id: str  # chunk ID or page ID
    source_type: str  # "chunk" or "wiki_page"
    score: float  # relevance score (0-1, higher is better)
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_scores(results: list[SearchResult]) -> list[SearchResult]:
    """Min-max normalize scores to the 0-1 range within a result set.

    If all scores are identical the results are returned with score 1.0.
    An empty list is returned unchanged.
    """
    if not results:
        return results

    scores = [r.score for r in results]
    min_score = min(scores)
    max_score = max(scores)
    span = max_score - min_score

    for result in results:
        result.score = (result.score - min_score) / span if span > 0 else 1.0

    return results


# ---------------------------------------------------------------------------
# Settings-derived helpers
# ---------------------------------------------------------------------------


def hybrid_weights_from(settings: "Settings") -> tuple[float, float, int]:
    """Return ``(vector_weight, fts_weight, fetch_multiplier)`` from settings."""
    return (
        settings.query_hybrid_vector_weight,
        settings.query_hybrid_fts_weight,
        settings.query_hybrid_fetch_multiplier,
    )


def default_top_k(settings: "Settings") -> int:
    """Return the default top-k value from settings."""
    return settings.query_default_top_k


# B-03: ``mmr_lambda_for`` and the matching ``Settings.query_mmr_lambda_*``
# fields were deleted in v0.11.0 follow-up. Nothing in
# :class:`QueryOrchestrator` actually invokes MMR re-ranking — the chunk
# path uses raw cosine + FTS rank — so shipping the helper + config knobs
# without the rerank pass was unimplemented surface. The
# ``test_fastembed_recall_parity`` Jaccard@10 ≥ 0.95 result demonstrates
# fastembed-int8 and ST-fp32 already produce nearly-identical neighbour
# sets, removing the empirical motivation for fastembed-specific MMR
# tuning. If a future version of the orchestrator wires MMR into the
# query path, the helper can be re-introduced alongside the actual
# rerank logic.
# spec_id: 70ab2170-381a-4657-bcd1-28a40c6f369b


# ---------------------------------------------------------------------------
# QueryOrchestrator
# ---------------------------------------------------------------------------


class QueryOrchestrator:
    """Hybrid query engine combining vectorstore similarity search with wiki FTS."""

    def __init__(
        self,
        vectorstore: VectorStore,
        embedder: Embedder,
        wiki: WikiManager,
    ) -> None:
        self._vectorstore = vectorstore
        self._embedder = embedder
        self._wiki = wiki

    # ------------------------------------------------------------------
    # Semantic search
    # ------------------------------------------------------------------

    def query(self, text: str, kb_id: str, top_k: int = 10) -> list[SearchResult]:
        """Semantic query using the vectorstore.

        1. Embed the query text.
        2. Search the vectorstore for similar chunks.
        3. Return a ranked :class:`SearchResult` list with citations.
        """
        embedding = self._embedder.embed_query(text)
        query_results: list[QueryResult] = self._vectorstore.query(
            embedding, top_k=top_k, where={"kb_id": kb_id}
        )

        search_results: list[SearchResult] = []
        for qr in query_results:
            # ChromaDB uses cosine distance so similarity = 1 - distance.
            # Clamp to [0, 1] to guard against floating-point edge cases.
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
        return search_results

    # ------------------------------------------------------------------
    # Keyword / FTS search
    # ------------------------------------------------------------------

    def search(self, text: str, kb_id: str, top_k: int = 10) -> list[SearchResult]:
        """Keyword search using wiki full-text search.

        1. Search wiki pages via FTS.
        2. Return a ranked :class:`SearchResult` list.

        Scores are based on result position (FTS returns ranked results).
        """
        pages: list[WikiPage] = self._wiki.search(text, kb_id)

        if not pages:
            return []

        search_results: list[SearchResult] = []
        total = len(pages)
        for index, page in enumerate(pages):
            # Rank-based scoring: first result gets highest score.
            score = 1.0 - (index / total)
            search_results.append(
                SearchResult(
                    content=page.content,
                    source_id=page.id,
                    source_type="wiki_page",
                    score=score,
                    metadata={
                        "title": page.title,
                        "page_type": page.page_type.value,
                        "tags": page.tags,
                    },
                )
            )

        return search_results[:top_k]

    # ------------------------------------------------------------------
    # Hybrid search
    # ------------------------------------------------------------------

    def hybrid_query(
        self,
        text: str,
        kb_id: str,
        top_k: int = 10,
        vector_weight: float = 0.7,
        fts_weight: float = 0.3,
        fetch_multiplier: int = 2,
    ) -> list[SearchResult]:
        """Combined semantic + keyword search.

        1. Run both ``query()`` and ``search()`` with over-fetching.
        2. Normalize scores to 0-1 range within each result set.
        3. Combine: ``vector_weight * vector_score + fts_weight * fts_score``.
        4. Deduplicate by matching content across result sets.
        5. Return *top_k* results sorted by combined score (descending).
        """
        # Over-fetch to get better merge candidates.
        fetch_k = top_k * fetch_multiplier

        vector_results = self.query(text, kb_id, top_k=fetch_k)
        fts_results = self.search(text, kb_id, top_k=fetch_k)

        # Normalize scores within each set independently.
        _normalize_scores(vector_results)
        _normalize_scores(fts_results)

        # Index FTS results by content for deduplication / merging.
        fts_by_content: dict[str, SearchResult] = {}
        for fr in fts_results:
            fts_by_content[fr.content] = fr

        merged: list[SearchResult] = []
        seen_contents: set[str] = set()

        # Process vector results, merging with FTS where content matches.
        for vr in vector_results:
            combined_score = vector_weight * vr.score
            fts_match = fts_by_content.get(vr.content)
            if fts_match is not None:
                combined_score += fts_weight * fts_match.score
                seen_contents.add(vr.content)

            merged.append(
                SearchResult(
                    content=vr.content,
                    source_id=vr.source_id,
                    source_type=vr.source_type,
                    score=combined_score,
                    metadata=vr.metadata,
                )
            )

        # Add FTS-only results (not already merged via vector results).
        for fr in fts_results:
            if fr.content not in seen_contents:
                merged.append(
                    SearchResult(
                        content=fr.content,
                        source_id=fr.source_id,
                        source_type=fr.source_type,
                        score=fts_weight * fr.score,
                        metadata=fr.metadata,
                    )
                )

        # Sort by combined score descending, then truncate.
        merged.sort(key=lambda r: r.score, reverse=True)
        return merged[:top_k]
